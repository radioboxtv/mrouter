"""Eval results: schema, storage, and query for custom evaluation outcomes.

This module provides a first-class capability source for evaluations:
runners (benchmarks, rubric scorers, LLM-judge, human review) submit
results here, and the routing engine can query them as a scoring signal.

Storage: appended to route_log.jsonl under type='eval_result' so it
shares the same audit trail as routing requests and feedback. The
metrics_store SQLite table (eval_results) is the fast dashboard path,
mirroring the existing request_metrics pattern.

Design goals:
- Zero new dependencies (stdlib json + sqlite only).
- Append-only log (route_log.jsonl) for auditability.
- Windowed query interface matching metrics_store semantics.
- Extensible result schema via a 'tags' field for arbitrary eval types.
- Integrates into the routing engine as a capability source (like
  rankings/live AA data, reliability, bench, learned adjustments).
"""

import json
import os
import sqlite3
import threading
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ Schema

_EVAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run_name TEXT NOT NULL,
    model_key TEXT NOT NULL,
    use_case TEXT,
    score REAL,
    pass_fail TEXT,
    rubric_score REAL,
    llm_judge_score REAL,
    sample_count INTEGER,
    dataset TEXT,
    version TEXT,
    source TEXT,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_er_run ON eval_results(run_name, ts);
CREATE INDEX IF NOT EXISTS idx_er_model ON eval_results(model_key, ts);
CREATE INDEX IF NOT EXISTS idx_er_ds ON eval_results(dataset, ts);
"""

_lock = threading.Lock()


def db_path(config):
    """Resolve the eval_results SQLite path from config."""
    return os.path.join(
        BASE,
        (config.get("eval", {}) or {}).get("db_file", "eval_results.db"),
    )


def _connect(config):
    path = db_path(config)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_EVAL_SCHEMA)
    return conn


def _enabled(config):
    return (config.get("eval", {}) or {}).get("enabled", True)


# ------------------------------------------------------------------ Public API

def submit_result(config, run_name, model_key, score=None,
                  pass_fail=None, rubric_score=None, llm_judge_score=None,
                  sample_count=0, dataset=None, version=None,
                  source=None, tags=None, use_case=None, ts=None):
    """Submit one evaluation result.

    Args:
        run_name:  Unique identifier for this eval run (e.g. "coding-bench-v2").
        model_key: Fleet key of the model being evaluated.
        score:     General numeric score (0-100).
        pass_fail: "pass" | "fail" | None (for binary evals).
        rubric_score: Score from a rubric-based scorer (0-100).
        llm_judge_score: Score from an LLM-judge scorer (0-100).
        sample_count: Number of samples in this run.
        dataset:   Dataset name (e.g. "gsm8k", "humaneval").
        version:   Dataset/scorer version.
        source:    Source name (e.g. "bench", "llm-judge", "human").
        tags:      Arbitrary string tags for filtering (comma-separated).
        use_case:  Optional use case the eval targets.
        ts:        Timestamp (defaults to now).

    Appends to both route_log.jsonl (audit trail) and the SQLite store
    (dashboard queries). Best-effort: failures on either backend must
    not break the caller.
    """
    if not _enabled(config):
        return None

    entry = {
        "ts": ts if ts is not None else time.time(),
        "run_name": run_name,
        "model_key": model_key,
        "use_case": use_case,
        "score": score,
        "pass_fail": pass_fail,
        "rubric_score": rubric_score,
        "llm_judge_score": llm_judge_score,
        "sample_count": sample_count,
        "dataset": dataset,
        "version": version,
        "source": source,
        "tags": tags,
    }

    # Write to route_log.jsonl (audit trail, shared with learning.py)
    try:
        from . import learning
        learning.append_log(config, {
            "type": "eval_result",
            **entry,
        })
    except Exception:
        pass  # best-effort: never block on logging

    # Write to SQLite (dashboard queries)
    try:
        with _lock:
            conn = _connect(config)
            try:
                conn.execute(
                    "INSERT INTO eval_results "
                    "(ts, run_name, model_key, use_case, score, pass_fail, "
                    "rubric_score, llm_judge_score, sample_count, dataset, "
                    "version, source, tags) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry["ts"],
                        entry["run_name"],
                        entry["model_key"],
                        entry.get("use_case"),
                        entry.get("score"),
                        entry.get("pass_fail"),
                        entry.get("rubric_score"),
                        entry.get("llm_judge_score"),
                        entry.get("sample_count", 0),
                        entry.get("dataset"),
                        entry.get("version"),
                        entry.get("source"),
                        entry.get("tags"),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass  # best-effort: never block on SQLite

    return entry


def query_results(config, run_name=None, model_key=None, dataset=None,
                  source=None, tags=None, since_ts=None, window_seconds=None,
                  use_case=None, now=None):
    """Query eval results with optional filters.

    Returns a list of result dicts, oldest first.

    Filters (all optional):
        run_name:   Exact match on run name.
        model_key:  Fleet key of the model.
        dataset:    Dataset name.
        source:     Source name (e.g. 'bench', 'llm-judge').
        tags:       Comma-separated tags (substring match).
        since_ts:   Unix timestamp filter.
        window_seconds: If provided, computes since_ts = now - window_seconds.
        use_case:   Filter by use case.
        now:        Reference timestamp for window_seconds (defaults to time.time()).

    For windowed aggregation (summary), see window_summary().
    """
    with _lock:
        conn = _connect(config)
        try:
            conn.row_factory = sqlite3.Row
            clauses = []
            params = []

            if run_name is not None:
                clauses.append("run_name = ?")
                params.append(run_name)
            if model_key is not None:
                clauses.append("model_key = ?")
                params.append(model_key)
            if dataset is not None:
                clauses.append("dataset = ?")
                params.append(dataset)
            if source is not None:
                clauses.append("source = ?")
                params.append(source)
            if tags is not None:
                clauses.append("tags LIKE ?")
                params.append(f"%{tags}%")
            if use_case is not None:
                clauses.append("use_case = ?")
                params.append(use_case)

            if window_seconds is not None:
                if now is None:
                    now = time.time()
                since_ts = now - window_seconds
            if since_ts is not None:
                clauses.append("ts >= ?")
                params.append(since_ts)

            where = ""
            if clauses:
                where = "WHERE " + " AND ".join(clauses)

            cur = conn.execute(
                f"SELECT * FROM eval_results {where} ORDER BY ts ASC",
                params,
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def window_summary(config, window_seconds, run_name=None, model_key=None,
                   dataset=None, source=None, tags=None, use_case=None,
                   now=None):
    """Windowed summary of eval results.

    Returns aggregated stats over the trailing window_seconds:
      run_count, model_count, avg_score, avg_rubric_score,
      avg_llm_judge_score, pass_rate, total_samples,
      by_model: {model_key: {avg_score, sample_count, latest_score, latest_ts}},
      by_run: {run_name: {avg_score, sample_count, source, dataset}}.
    """
    if now is None:
        now = time.time()
    since_ts = now - window_seconds

    rows = query_results(
        config, run_name=run_name, model_key=model_key,
        dataset=dataset, source=source, tags=tags,
        since_ts=since_ts, use_case=use_case,
    )

    run_count = len(set(r["run_name"] for r in rows))
    model_count = len(set(r["model_key"] for r in rows))
    total_samples = sum(r.get("sample_count", 0) for r in rows)

    scores = [r["score"] for r in rows if r.get("score") is not None]
    rubric_scores = [r["rubric_score"] for r in rows if r.get("rubric_score") is not None]
    llm_judge_scores = [r["llm_judge_score"] for r in rows if r.get("llm_judge_score") is not None]

    avg_score = (sum(scores) / len(scores)) if scores else None
    avg_rubric = (sum(rubric_scores) / len(rubric_scores)) if rubric_scores else None
    avg_llm_judge = (sum(llm_judge_scores) / len(llm_judge_scores)) if llm_judge_scores else None

    passes = sum(1 for r in rows if r.get("pass_fail") == "pass")
    total_with_pf = sum(1 for r in rows if r.get("pass_fail") is not None)
    pass_rate = (passes / total_with_pf) if total_with_pf else None

    by_model = {}
    for r in rows:
        key = r["model_key"]
        m = by_model.setdefault(key, {
            "avg_score": 0.0, "scores": [],
            "sample_count": 0, "latest_score": None, "latest_ts": 0.0,
        })
        if r.get("score") is not None:
            m["scores"].append(r["score"])
            m["latest_score"] = r["score"]
            m["latest_ts"] = max(m["latest_ts"], r["ts"])
        m["sample_count"] += r.get("sample_count", 0)

    for key, m in by_model.items():
        if m["scores"]:
            m["avg_score"] = round(sum(m["scores"]) / len(m["scores"]), 2)
        else:
            m["avg_score"] = None
        del m["scores"]

    by_run = {}
    for r in rows:
        rn = r["run_name"]
        run = by_run.setdefault(rn, {
            "avg_score": 0.0, "scores": [],
            "sample_count": 0, "source": r.get("source"),
            "dataset": r.get("dataset"),
        })
        if r.get("score") is not None:
            run["scores"].append(r["score"])
        run["sample_count"] += r.get("sample_count", 0)

    for rn, run in by_run.items():
        if run["scores"]:
            run["avg_score"] = round(sum(run["scores"]) / len(run["scores"]), 2)
        else:
            run["avg_score"] = None
        del run["scores"]

    return {
        "window_seconds": window_seconds,
        "run_count": run_count,
        "model_count": model_count,
        "total_samples": total_samples,
        "avg_score": round(avg_score, 2) if avg_score is not None else None,
        "avg_rubric_score": round(avg_rubric, 2) if avg_rubric is not None else None,
        "avg_llm_judge_score": round(avg_llm_judge, 2) if avg_llm_judge is not None else None,
        "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
        "by_model": by_model,
        "by_run": by_run,
    }


def latest_for_model(config, model_key, window_seconds=None, use_case=None, now=None):
    """Latest eval result for a single model, fastest query path.

    Returns a single result dict or None.
    """
    rows = query_results(
        config, model_key=model_key, window_seconds=window_seconds,
        use_case=use_case, now=now,
    )
    if not rows:
        return None
    return rows[-1]
