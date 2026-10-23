"""metrics_store: lightweight on-device SQLite store for live perf/spend metrics.

Separate from route_log.jsonl (learning.py), which stays the audit/learning
log unchanged. This store exists purely to answer windowed dashboard/tray
queries fast (p90 TTFT, p90 tokens/min, request volume, error rate, spend)
without re-scanning the whole JSONL log on every tray refresh tick.

SQLite + WAL: safe concurrent writes from the proxy's request threads and
reads from the tray app / dashboard without either side blocking the other.
Percentiles are computed in Python over the (small, personal-scale) window
rows fetched for a query -- simplest-correct, no SQL percentile extension
needed.
"""

import math
import os
import sqlite3
import threading
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    use_case TEXT,
    model_key TEXT,
    provider TEXT,
    status_code INTEGER,
    ttft_s REAL,
    duration_s REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    cost_is_estimate INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rm_ts ON request_metrics(ts);
CREATE INDEX IF NOT EXISTS idx_rm_model_ts ON request_metrics(model_key, ts);
"""

# One lock per process is enough: sqlite3 connections are cheap to open per
# call at this volume (personal-scale, not high-frequency trading), and this
# avoids holding a long-lived connection across threads.
_lock = threading.Lock()

# Autoprune throttle, mirrors learning.py's autolearn throttle pattern.
_PRUNE_MIN_INTERVAL_S = 300.0
_last_prune = {}


def db_path(config):
    return os.path.join(BASE, (config.get("metrics", {}) or {}).get("db_file", "metrics.db"))


def _connect(config):
    path = db_path(config)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _enabled(config):
    return (config.get("metrics", {}) or {}).get("enabled", True)


def record(config, entry):
    """Append one request's metrics row. No-op if metrics are disabled."""
    if not _enabled(config):
        return
    with _lock:
        conn = _connect(config)
        try:
            conn.execute(
                "INSERT INTO request_metrics "
                "(ts, use_case, model_key, provider, status_code, ttft_s, duration_s, "
                " tokens_in, tokens_out, cost_usd, cost_is_estimate) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.get("ts", time.time()),
                    entry.get("use_case"),
                    entry.get("model_key"),
                    entry.get("provider"),
                    entry.get("status_code"),
                    entry.get("ttft_s"),
                    entry.get("duration_s"),
                    entry.get("tokens_in"),
                    entry.get("tokens_out"),
                    entry.get("cost_usd"),
                    1 if entry.get("cost_is_estimate") else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    _maybe_prune(config)


def query_window(config, since_ts):
    """Raw rows with ts >= since_ts, as dicts, oldest first."""
    with _lock:
        conn = _connect(config)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM request_metrics WHERE ts >= ? ORDER BY ts ASC", (since_ts,)
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def prune(config, now=None):
    """Delete rows older than metrics.retention_days (default 7)."""
    retention_days = float((config.get("metrics", {}) or {}).get("retention_days", 7))
    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    with _lock:
        conn = _connect(config)
        try:
            conn.execute("DELETE FROM request_metrics WHERE ts < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()


def _maybe_prune(config, force=False):
    if not _enabled(config):
        return
    key = db_path(config)
    now = time.time()
    if not force and now - _last_prune.get(key, 0.0) < _PRUNE_MIN_INTERVAL_S:
        return
    _last_prune[key] = now
    try:
        prune(config, now=now)
    except Exception:
        pass  # best-effort housekeeping; never block request logging on it


def percentile(values, pct):
    """Nearest-rank percentile over a list of numbers. None for empty input."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (pct / 100.0) * (len(vals) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    lo, hi = vals[int(f)], vals[int(c)]
    return lo + (hi - lo) * (k - f)


def window_summary(config, window_seconds, now=None, buckets=12, live_threshold_s=60):
    """Aggregate metrics over the trailing `window_seconds`.

    Returns a dict with: requests, errors, error_rate, ttft_p50_s, ttft_p90_s,
    tokens_per_min_p90, spend_actual_usd, spend_estimated_usd, spend_total_usd,
    models: {model_key: {requests, errors, use_cases:[{use_case,count}, ...],
    status: 'live'|'recent', ttft_p90_s, tokens_per_sec_p90, reliability}},
    series: [{bucket_start, requests, errors}, ...] (oldest first).
    """
    now = now if now is not None else time.time()
    since_ts = now - window_seconds
    rows = query_window(config, since_ts)

    requests = len(rows)
    errors = sum(1 for r in rows if (r.get("status_code") or 0) >= 400)
    error_rate = (errors / requests) if requests else 0.0

    ttfts = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
    ttft_p50 = percentile(ttfts, 50)
    ttft_p90 = percentile(ttfts, 90)

    tok_per_min = []
    for r in rows:
        dur = r.get("duration_s") or 0
        ttft = r.get("ttft_s") or 0
        toks = r.get("tokens_out") or 0
        gen_s = max(1e-9, dur - ttft)
        if toks:
            tok_per_min.append((toks / gen_s) * 60.0)
    tpm_p90 = percentile(tok_per_min, 90)

    spend_actual = sum(
        float(r.get("cost_usd") or 0) for r in rows if not r.get("cost_is_estimate")
    )
    spend_estimated = sum(
        float(r.get("cost_usd") or 0) for r in rows if r.get("cost_is_estimate")
    )

    models = {}
    for r in rows:
        key = r.get("model_key") or "?"
        m = models.setdefault(key, {
            "requests": 0, "errors": 0, "_use_cases": {}, "_ttfts": [],
            "_tpm": [], "last_ts": 0.0,
        })
        m["requests"] += 1
        if (r.get("status_code") or 0) >= 400:
            m["errors"] += 1
        uc = r.get("use_case")
        if uc:
            m["_use_cases"][uc] = m["_use_cases"].get(uc, 0) + 1
        if r.get("ttft_s") is not None:
            m["_ttfts"].append(r["ttft_s"])
        dur = r.get("duration_s") or 0
        ttft = r.get("ttft_s") or 0
        toks = r.get("tokens_out") or 0
        gen_s = max(1e-9, dur - ttft)
        if toks:
            m["_tpm"].append((toks / gen_s) * 60.0)
        m["last_ts"] = max(m["last_ts"], r.get("ts") or 0)

    for key, m in models.items():
        ucs_sorted = sorted(m["_use_cases"].items(), key=lambda kv: -kv[1])
        m["use_cases"] = [{"use_case": uc, "count": c} for uc, c in ucs_sorted]
        m["ttft_p90_s"] = percentile(m["_ttfts"], 90)
        _tpm_p90 = percentile(m["_tpm"], 90) if m["_tpm"] else None
        m["tokens_per_sec_p90"] = (_tpm_p90 / 60.0) if _tpm_p90 is not None else None
        m["status"] = "live" if (now - m["last_ts"]) <= live_threshold_s else "recent"
        del m["_use_cases"], m["_ttfts"], m["_tpm"]

    # Bucketed volume series for the sparkline.
    bucket_s = window_seconds / buckets if buckets else window_seconds
    series = [{"bucket_start": since_ts + i * bucket_s, "requests": 0, "errors": 0}
              for i in range(buckets)]
    for r in rows:
        idx = int((r["ts"] - since_ts) / bucket_s) if bucket_s else 0
        idx = max(0, min(buckets - 1, idx))
        series[idx]["requests"] += 1
        if (r.get("status_code") or 0) >= 400:
            series[idx]["errors"] += 1

    return {
        "window_seconds": window_seconds,
        "requests": requests,
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "ttft_p50_s": round(ttft_p50, 3) if ttft_p50 is not None else None,
        "ttft_p90_s": round(ttft_p90, 3) if ttft_p90 is not None else None,
        "tokens_per_min_p90": round(tpm_p90, 1) if tpm_p90 is not None else None,
        "spend_actual_usd": round(spend_actual, 6),
        "spend_estimated_usd": round(spend_estimated, 6),
        "spend_total_usd": round(spend_actual + spend_estimated, 6),
        "models": models,
        "series": series,
    }


WINDOW_SECONDS = {"1h": 3600, "24h": 86400}


def resolve_window_seconds(config, window=None):
    """Resolve a window name ('1h'/'24h') or raw seconds to seconds, falling
    back to config.metrics.default_window, then 1h."""
    if window is None:
        window = (config.get("metrics", {}) or {}).get("default_window", "1h")
    if isinstance(window, (int, float)):
        return float(window)
    if window in WINDOW_SECONDS:
        return WINDOW_SECONDS[window]
    try:
        return float(window)
    except (TypeError, ValueError):
        return WINDOW_SECONDS["1h"]
