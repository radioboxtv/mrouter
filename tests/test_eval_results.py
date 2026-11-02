"""Tests for mrouter.eval_results: submit, query, window_summary, latest_for_model."""

import json
import os
import sqlite3
import tempfile
import time

import pytest

# We import the module under test directly.
import mrouter.eval_results as ev


# ------------------------------------------------------------------ Helpers

def _make_config(tmpdir, **overrides):
    """Build a minimal config dict pointing to a temp dir."""
    base = str(tmpdir)
    cfg = {
        "eval": {
            "enabled": True,
            "db_file": os.path.join(base, "eval_results.db"),
            "log_file": os.path.join(base, "route_log.jsonl"),
            "learned_file": os.path.join(base, "learned.json"),
        },
        "learning": {
            "enabled": True,
            "log_file": os.path.join(base, "route_log.jsonl"),
            "learned_file": os.path.join(base, "learned.json"),
        },
        "use_cases": {"chat": {}, "coding": {}, "sensitive": {}, "thinking": {}},
    }
    cfg.update(overrides)
    return cfg


def _populate_rows(conn, rows):
    """Insert raw rows into the SQLite DB for query tests."""
    for r in rows:
        conn.execute(
            "INSERT INTO eval_results "
            "(ts, run_name, model_key, use_case, score, pass_fail, "
            "rubric_score, llm_judge_score, sample_count, dataset, "
            "version, source, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r.get("ts", 1000.0),
                r["run_name"],
                r["model_key"],
                r.get("use_case"),
                r.get("score"),
                r.get("pass_fail"),
                r.get("rubric_score"),
                r.get("llm_judge_score"),
                r.get("sample_count", 0),
                r.get("dataset"),
                r.get("version"),
                r.get("source"),
                r.get("tags"),
            ),
        )


# ------------------------------------------------------------------ submit_result

class TestSubmitResult:
    """submit_result: audit trail (route_log.jsonl) + SQLite write."""

    def test_submits_to_sqlite(self, tmpdir):
        cfg = _make_config(tmpdir)
        result = ev.submit_result(cfg, "test-run", "local/qwen3.8", score=85,
                                   dataset="gsm8k", source="bench")
        assert result is not None
        assert result["run_name"] == "test-run"
        assert result["model_key"] == "local/qwen3.8"
        assert result["score"] == 85

        # Verify SQLite
        conn = ev._connect(cfg)
        try:
            rows = conn.execute("SELECT * FROM eval_results").fetchall()
            assert len(rows) == 1
            assert rows[0][5] == 85  # score is 6th column (index 5)
        finally:
            conn.close()

    def test_submits_to_route_log(self, tmpdir):
        cfg = _make_config(tmpdir)
        ev.submit_result(cfg, "test-run", "local/qwen3.8", score=85)
        log_path = cfg["eval"]["log_file"]
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        entry = json.loads(lines[-1])
        assert entry["type"] == "eval_result"
        assert entry["run_name"] == "test-run"
        assert entry["score"] == 85

    def test_all_fields_preserved(self, tmpdir):
        cfg = _make_config(tmpdir)
        result = ev.submit_result(
            cfg, "rubric-v1", "aperture/claude-opus",
            score=92, pass_fail="pass", rubric_score=92,
            llm_judge_score=88, sample_count=100,
            dataset="humaneval", version="2.0", source="rubric",
            tags="coding,benchmark", use_case="coding",
        )
        assert result["pass_fail"] == "pass"
        assert result["rubric_score"] == 92
        assert result["llm_judge_score"] == 88
        assert result["sample_count"] == 100
        assert result["dataset"] == "humaneval"
        assert result["version"] == "2.0"
        assert result["source"] == "rubric"
        assert result["tags"] == "coding,benchmark"
        assert result["use_case"] == "coding"

    def test_disabled_no_write(self, tmpdir):
        cfg = _make_config(tmpdir, eval={"enabled": False})
        result = ev.submit_result(cfg, "x", "m", score=1)
        assert result is None

    def test_default_timestamp(self, tmpdir):
        before = time.time()
        cfg = _make_config(tmpdir)
        result = ev.submit_result(cfg, "x", "m", score=1)
        after = time.time()
        assert before <= result["ts"] <= after

    def test_explicit_timestamp(self, tmpdir):
        cfg = _make_config(tmpdir)
        result = ev.submit_result(cfg, "x", "m", score=1, ts=5000.0)
        assert result["ts"] == 5000.0

    def test_no_score_fields_is_valid(self, tmpdir):
        cfg = _make_config(tmpdir)
        result = ev.submit_result(cfg, "x", "m")
        assert result is not None
        assert result["score"] is None
        assert result["pass_fail"] is None


# ------------------------------------------------------------------ query_results

class TestQueryResults:
    """query_results: filter, order, and multi-filter."""

    def _insert_many(self, conn, rows):
        for r in rows:
            conn.execute(
                "INSERT INTO eval_results "
                "(ts, run_name, model_key, use_case, score, pass_fail, "
                "rubric_score, llm_judge_score, sample_count, dataset, "
                "version, source, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r.get("ts", 1000.0),
                    r["run_name"],
                    r["model_key"],
                    r.get("use_case"),
                    r.get("score"),
                    r.get("pass_fail"),
                    r.get("rubric_score"),
                    r.get("llm_judge_score"),
                    r.get("sample_count", 0),
                    r.get("dataset"),
                    r.get("version"),
                    r.get("source"),
                    r.get("tags"),
                ),
            )

    def test_returns_all_when_no_filters(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80, "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90, "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg)
        assert len(results) == 2
        assert results[0]["score"] == 80
        assert results[1]["score"] == 90

    def test_filter_by_run_name(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "run-a", "model_key": "m1", "score": 80, "ts": 1000.0},
                {"run_name": "run-b", "model_key": "m2", "score": 90, "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg, run_name="run-a")
        assert len(results) == 1
        assert results[0]["run_name"] == "run-a"

    def test_filter_by_model_key(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80, "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90, "ts": 2000.0},
                {"run_name": "c", "model_key": "m1", "score": 70, "ts": 3000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg, model_key="m1")
        assert len(results) == 2
        assert all(r["model_key"] == "m1" for r in results)

    def test_filter_by_dataset(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "dataset": "gsm8k", "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "dataset": "humaneval", "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg, dataset="gsm8k")
        assert len(results) == 1
        assert results[0]["dataset"] == "gsm8k"

    def test_filter_by_source(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "source": "bench", "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "source": "llm-judge", "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg, source="llm-judge")
        assert len(results) == 1
        assert results[0]["source"] == "llm-judge"

    def test_filter_by_tags_substring(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "tags": "coding,benchmark", "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "tags": "chat", "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg, tags="coding")
        assert len(results) == 1
        assert results[0]["tags"] == "coding,benchmark"

    def test_filter_by_window_seconds(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80, "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90, "ts": 5000.0},
                {"run_name": "c", "model_key": "m3", "score": 95, "ts": 9000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        # 1h window from ts=10000: only ts=9000 is included
        results = ev.query_results(cfg, window_seconds=3600, now=10000.0)
        assert len(results) == 1
        assert results[0]["run_name"] == "c"

    def test_filter_by_use_case(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "use_case": "coding", "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "use_case": "chat", "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg, use_case="chat")
        assert len(results) == 1
        assert results[0]["use_case"] == "chat"

    def test_empty_results_when_no_match(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80, "ts": 1000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg, run_name="nonexistent")
        assert len(results) == 0

    def test_multiple_filters_combined(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "dataset": "gsm8k", "source": "bench",
                 "use_case": "coding", "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "dataset": "gsm8k", "source": "llm-judge",
                 "use_case": "chat", "ts": 2000.0},
                {"run_name": "c", "model_key": "m1", "score": 70,
                 "dataset": "gsm8k", "source": "bench",
                 "use_case": "coding", "ts": 3000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(
            cfg, model_key="m1", dataset="gsm8k", source="bench",
            use_case="coding",
        )
        assert len(results) == 2
        assert all(
            r["model_key"] == "m1" and r["dataset"] == "gsm8k"
            and r["source"] == "bench" and r["use_case"] == "coding"
            for r in results
        )

    def test_order_by_timestamp_ascending(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "b", "model_key": "m2", "score": 90, "ts": 2000.0},
                {"run_name": "a", "model_key": "m1", "score": 80, "ts": 1000.0},
                {"run_name": "c", "model_key": "m3", "score": 70, "ts": 3000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        results = ev.query_results(cfg)
        assert results[0]["run_name"] == "a"
        assert results[1]["run_name"] == "b"
        assert results[2]["run_name"] == "c"


# ------------------------------------------------------------------ window_summary

class TestWindowSummary:
    """window_summary: aggregations and breakdowns."""

    def _insert_many(self, conn, rows):
        for r in rows:
            conn.execute(
                "INSERT INTO eval_results "
                "(ts, run_name, model_key, use_case, score, pass_fail, "
                "rubric_score, llm_judge_score, sample_count, dataset, "
                "version, source, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r.get("ts", 1000.0),
                    r["run_name"],
                    r["model_key"],
                    r.get("use_case"),
                    r.get("score"),
                    r.get("pass_fail"),
                    r.get("rubric_score"),
                    r.get("llm_judge_score"),
                    r.get("sample_count", 0),
                    r.get("dataset"),
                    r.get("version"),
                    r.get("source"),
                    r.get("tags"),
                ),
            )

    def test_basic_summary(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "rubric_score": 85, "llm_judge_score": 78,
                 "sample_count": 50, "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "rubric_score": 92, "llm_judge_score": 88,
                 "sample_count": 100, "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        summary = ev.window_summary(cfg, window_seconds=3600, now=3000.0)
        assert summary["run_count"] == 2
        assert summary["model_count"] == 2
        assert summary["total_samples"] == 150
        assert summary["avg_score"] == 85.0
        assert summary["avg_rubric_score"] == 88.5
        assert summary["avg_llm_judge_score"] == 83.0

    def test_pass_rate(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "pass_fail": "pass", "sample_count": 50, "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "pass_fail": "pass", "sample_count": 100, "ts": 2000.0},
                {"run_name": "c", "model_key": "m3", "score": 70,
                 "pass_fail": "fail", "sample_count": 30, "ts": 2500.0},
            ])
            conn.commit()
        finally:
            conn.close()

        summary = ev.window_summary(cfg, window_seconds=3600, now=3000.0)
        assert summary["pass_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_by_model_breakdown(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "sample_count": 50, "ts": 1000.0},
                {"run_name": "b", "model_key": "m1", "score": 90,
                 "sample_count": 100, "ts": 2000.0},
                {"run_name": "c", "model_key": "m2", "score": 70,
                 "sample_count": 30, "ts": 2500.0},
            ])
            conn.commit()
        finally:
            conn.close()

        summary = ev.window_summary(cfg, window_seconds=3600, now=3000.0)
        assert "m1" in summary["by_model"]
        assert summary["by_model"]["m1"]["avg_score"] == 85.0
        assert summary["by_model"]["m1"]["sample_count"] == 150
        assert summary["by_model"]["m1"]["latest_score"] == 90
        assert "m2" in summary["by_model"]
        assert summary["by_model"]["m2"]["avg_score"] == 70.0

    def test_by_run_breakdown(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "source": "bench", "dataset": "gsm8k",
                 "sample_count": 50, "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "source": "llm-judge", "dataset": "humaneval",
                 "sample_count": 100, "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        summary = ev.window_summary(cfg, window_seconds=3600, now=3000.0)
        assert "a" in summary["by_run"]
        assert summary["by_run"]["a"]["avg_score"] == 80.0
        assert summary["by_run"]["a"]["source"] == "bench"
        assert summary["by_run"]["a"]["dataset"] == "gsm8k"
        assert summary["by_run"]["b"]["avg_score"] == 90.0

    def test_empty_results(self, tmpdir):
        cfg = _make_config(tmpdir)
        summary = ev.window_summary(cfg, window_seconds=3600, now=3000.0)
        assert summary["run_count"] == 0
        assert summary["model_count"] == 0
        assert summary["total_samples"] == 0
        assert summary["avg_score"] is None
        assert summary["pass_rate"] is None

    def test_filters_applied(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "use_case": "coding", "ts": 1000.0},
                {"run_name": "b", "model_key": "m2", "score": 90,
                 "use_case": "chat", "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        summary = ev.window_summary(
            cfg, window_seconds=3600, now=3000.0, use_case="chat",
        )
        assert summary["model_count"] == 1
        assert summary["avg_score"] == 90.0


# ------------------------------------------------------------------ latest_for_model

class TestLatestForModel:
    """latest_for_model: single-model lookup."""

    def _insert_many(self, conn, rows):
        for r in rows:
            conn.execute(
                "INSERT INTO eval_results "
                "(ts, run_name, model_key, use_case, score, pass_fail, "
                "rubric_score, llm_judge_score, sample_count, dataset, "
                "version, source, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r.get("ts", 1000.0),
                    r["run_name"],
                    r["model_key"],
                    r.get("use_case"),
                    r.get("score"),
                    r.get("pass_fail"),
                    r.get("rubric_score"),
                    r.get("llm_judge_score"),
                    r.get("sample_count", 0),
                    r.get("dataset"),
                    r.get("version"),
                    r.get("source"),
                    r.get("tags"),
                ),
            )

    def test_returns_latest(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80, "ts": 1000.0},
                {"run_name": "b", "model_key": "m1", "score": 90, "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        latest = ev.latest_for_model(cfg, "m1")
        assert latest is not None
        assert latest["score"] == 90
        assert latest["run_name"] == "b"

    def test_returns_none_when_no_results(self, tmpdir):
        cfg = _make_config(tmpdir)
        latest = ev.latest_for_model(cfg, "nonexistent")
        assert latest is None

    def test_filter_by_window(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80, "ts": 1000.0},
                {"run_name": "b", "model_key": "m1", "score": 90, "ts": 5000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        # 1h window from ts=8000: only ts=5000 is included
        latest = ev.latest_for_model(cfg, "m1", window_seconds=3600, now=8000.0)
        assert latest is not None
        assert latest["score"] == 90

    def test_filter_by_use_case(self, tmpdir):
        cfg = _make_config(tmpdir)
        conn = ev._connect(cfg)
        try:
            self._insert_many(conn, [
                {"run_name": "a", "model_key": "m1", "score": 80,
                 "use_case": "coding", "ts": 1000.0},
                {"run_name": "b", "model_key": "m1", "score": 90,
                 "use_case": "chat", "ts": 2000.0},
            ])
            conn.commit()
        finally:
            conn.close()

        latest = ev.latest_for_model(cfg, "m1", use_case="chat")
        assert latest is not None
        assert latest["score"] == 90
