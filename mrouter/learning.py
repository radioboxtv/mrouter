"""Route log + learned adjustments: the router gets smarter from observed use.

Every routed request appends one JSON line to route_log.jsonl. Learned
adjustments in learned.json are now recomputed automatically — `append_log`
triggers a throttled recompute and `append_feedback` triggers one
immediately, so routing improves from real traffic with **zero manual
steps**. `mrouter learn` still exists for a forced, synchronous recompute
(useful right after editing the log by hand, or in scripts) but a human
should never need to run it as part of normal operation.
Adjustments are bounded so a few bad samples cannot wreck routing.
"""

import json
import os
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Throttle for the automatic recompute triggered by ordinary routed-request
# logging (high volume); feedback always recomputes immediately since it's a
# low-volume, high-signal human action. Keyed by learned.json path so tests
# with isolated tmp paths don't bleed state into each other.
AUTOLEARN_MIN_INTERVAL_S = 5.0
_last_autolearn = {}


def _path(config, key, default):
    return os.path.join(BASE, (config.get("learning", {}) or {}).get(key, default))


def log_path(config):
    return _path(config, "log_file", "route_log.jsonl")


def learned_path(config):
    return _path(config, "learned_file", "learned.json")


def append_log(config, entry):
    if not (config.get("learning", {}) or {}).get("enabled", True):
        return
    entry = {"ts": time.time(), **entry}
    path = log_path(config)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    _maybe_autolearn(config)


def _maybe_autolearn(config, force=False):
    """Recompute learned.json in-process so routing adapts with no CLI step.

    Throttled to AUTOLEARN_MIN_INTERVAL_S per learned-file path: cheap enough
    (plain JSON aggregation over a log that's usually small) that recomputing
    every few seconds under load is a non-issue, and it means the very next
    request can already use the latest adjustment.
    """
    if not (config.get("learning", {}) or {}).get("enabled", True):
        return
    key = learned_path(config)
    now = time.time()
    if not force and now - _last_autolearn.get(key, 0.0) < AUTOLEARN_MIN_INTERVAL_S:
        return
    _last_autolearn[key] = now
    try:
        save_learned(config)
    except Exception as e:
        print(f"[mrouter] autolearn failed ({type(e).__name__}: {e}); will retry", flush=True)


def load_log(config):
    path = log_path(config)
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def append_feedback(config, use_case, model_key, rating, note=""):
    """rating: 1 (good) or -1 (bad). Returns the written entry.

    Feedback is low-volume and high-signal (a human just acted), so unlike
    ordinary request logging it forces an immediate, unthrottled relearn —
    the very next routed request already reflects it.
    """
    entry = {
        "type": "feedback", "use_case": use_case, "routed_model_key": model_key,
        "rating": int(rating), "note": note or "", "source": "manual",
    }
    append_log(config, entry)
    _maybe_autolearn(config, force=True)
    return entry


def _norm_key(entry):
    return f"{entry.get('use_case')}|{entry.get('routed_model_key')}"


def compute_learned(config, entries=None):
    """Derive adjustments from the log. Returns { 'uc|model': {adj, samples, ...} }."""
    cfg = config.get("learning", {}) or {}
    max_adj = float(cfg.get("max_adjustment", 8.0))
    min_samples = int(cfg.get("min_samples_for_auto_adj", 5))
    entries = entries if entries is not None else load_log(config)

    stats = {}
    for e in entries:
        key = _norm_key(e)
        if not key or key.endswith("|None"):
            continue
        s = stats.setdefault(key, {
            "samples": 0, "ok": 0, "errors": 0, "lat_sum": 0.0,
            "pos": 0, "neg": 0, "cost_sum": 0.0,
        })
        if e.get("type") == "feedback":
            if e.get("rating", 0) > 0:
                s["pos"] += 1
            elif e.get("rating", 0) < 0:
                s["neg"] += 1
            continue
        s["samples"] += 1
        status = e.get("status_code") or 200
        if status and status < 400:
            s["ok"] += 1
        else:
            s["errors"] += 1
        if e.get("ttft_s") is not None:
            s["lat_sum"] += float(e["ttft_s"])
        if e.get("actual_cost_usd") is not None:
            s["cost_sum"] += float(e["actual_cost_usd"])

    learned = {}
    for key, s in stats.items():
        adj = 0.0
        if s["samples"] >= min_samples and s["samples"] > 0:
            success_rate = s["ok"] / max(1, s["samples"])
            adj += (success_rate - 0.5) * 8.0  # +/- up to 4 for reliability
        net_fb = s["pos"] - s["neg"]
        if net_fb:
            adj += max(-4.0, min(4.0, net_fb * 1.5))  # explicit feedback bites harder
        adj = max(-max_adj, min(max_adj, adj))
        avg_lat = round(s["lat_sum"] / s["ok"], 3) if s["ok"] else None
        learned[key] = {
            "adj": round(adj, 2), "samples": s["samples"],
            "success_rate": round(s["ok"] / max(1, s["samples"]), 3) if s["samples"] else None,
            "pos": s["pos"], "neg": s["neg"], "avg_latency_s": avg_lat,
            "total_cost_usd": round(s["cost_sum"], 6),
        }
    return learned


def save_learned(config, learned=None):
    learned = learned if learned is not None else compute_learned(config)
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "adjustments": learned}
    with open(learned_path(config), "w") as f:
        json.dump(out, f, indent=2)
    return out


def load_learned(config):
    path = learned_path(config)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return (json.load(f).get("adjustments") or {})
    except Exception:
        return {}


def adjustment_for(learned, use_case, model_key):
    if not learned:
        return 0.0
    entry = learned.get(f"{use_case}|{model_key}")
    return float(entry.get("adj", 0.0)) if entry else 0.0