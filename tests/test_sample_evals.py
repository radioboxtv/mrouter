"""Sample evals: realistic evaluation scenarios that exercise the eval_results
subsystem end-to-end.

These tests double as documentation — each scenario demonstrates a common
eval pattern (benchmark, rubric, LLM-judge, binary pass/fail, regression)
and shows how results flow through submit → query → window_summary →
latest_for_model, including time-windowed filtering and per-model
comparison.

Run:  python3 -m pytest tests/test_sample_evals.py -v
"""

import time

import mrouter.eval_results as ev


# ------------------------------------------------------------------ helpers

def _cfg(tmpdir):
    base = str(tmpdir)
    return {
        "eval": {
            "enabled": True,
            "db_file": base + "/eval_results.db",
        },
        "learning": {
            "enabled": True,
            "log_file": base + "/route_log.jsonl",
            "learned_file": base + "/learned.json",
        },
        "use_cases": {"chat": {}, "coding": {}, "sensitive": {}, "thinking": {}},
    }


def _submit_many(cfg, rows):
    for r in rows:
        ev.submit_result(cfg, **r)


# ------------------------------------------------------------------ Scenario 1: HumanEval-style coding benchmark

class TestSampleCodingBenchmark:
    """A coding benchmark (e.g. HumanEval) run across the fleet.

    Each model gets a pass@1 score; results are compared to pick the
    best coding model.
    """

    def test_coding_benchmark_submits_and_ranks(self, tmpdir):
        cfg = _cfg(tmpdir)
        now = time.time()

        # Simulate a HumanEval run: pass@1 per model (0-100 scale)
        _submit_many(cfg, [
            {"run_name": "humaneval-v2", "model_key": "local/qwen3.8-27b",
             "score": 72.0, "pass_fail": "pass", "sample_count": 164,
             "dataset": "humaneval", "version": "1.0", "source": "benchmark",
             "use_case": "coding", "ts": now},
            {"run_name": "humaneval-v2", "model_key": "aperture:deepseek-v4-pro",
             "score": 85.5, "pass_fail": "pass", "sample_count": 164,
             "dataset": "humaneval", "version": "1.0", "source": "benchmark",
             "use_case": "coding", "ts": now},
            {"run_name": "humaneval-v2", "model_key": "aperture:claude-sonnet",
             "score": 91.2, "pass_fail": "pass", "sample_count": 164,
             "dataset": "humaneval", "version": "1.0", "source": "benchmark",
             "use_case": "coding", "ts": now},
            {"run_name": "humaneval-v2", "model_key": "aperture:deepseek-v4-flash",
             "score": 68.3, "pass_fail": "pass", "sample_count": 164,
             "dataset": "humaneval", "version": "1.0", "source": "benchmark",
             "use_case": "coding", "ts": now},
        ])

        # Query all results for this run
        rows = ev.query_results(cfg, run_name="humaneval-v2")
        assert len(rows) == 4

        # Summary: best model by score
        summary = ev.window_summary(cfg, window_seconds=3600, now=now + 1)
        assert summary["model_count"] == 4
        assert summary["avg_score"] is not None

        # Claude Sonnet should be the top scorer
        best = max(summary["by_model"].items(), key=lambda kv: kv[1]["avg_score"])
        assert best[0] == "aperture:claude-sonnet"
        assert best[1]["avg_score"] == 91.2

        # latest_for_model returns the most recent result
        latest = ev.latest_for_model(cfg, "aperture:claude-sonnet", use_case="coding")
        assert latest is not None
        assert latest["score"] == 91.2
        assert latest["dataset"] == "humaneval"


# ------------------------------------------------------------------ Scenario 2: Rubric-based chat quality eval

class TestSampleChatRubric:
    """A rubric-based chat quality eval (e.g. MT-Bench style).

    Each model is scored on multiple dimensions (helpfulness, accuracy,
    coherence) and the rubric_score captures the aggregate.
    """

    def test_chat_rubric_eval(self, tmpdir):
        cfg = _cfg(tmpdir)
        now = time.time()

        _submit_many(cfg, [
            {"run_name": "mt-bench-v3", "model_key": "local/qwen3.8-27b",
             "score": 7.8, "rubric_score": 7.8, "sample_count": 80,
             "dataset": "mt-bench", "version": "3.0", "source": "rubric",
             "tags": "chat,quality", "use_case": "chat", "ts": now},
            {"run_name": "mt-bench-v3", "model_key": "aperture:claude-haiku",
             "score": 8.5, "rubric_score": 8.5, "sample_count": 80,
             "dataset": "mt-bench", "version": "3.0", "source": "rubric",
             "tags": "chat,quality", "use_case": "chat", "ts": now},
            {"run_name": "mt-bench-v3", "model_key": "aperture:gemini-flash",
             "score": 8.2, "rubric_score": 8.2, "sample_count": 80,
             "dataset": "mt-bench", "version": "3.0", "source": "rubric",
             "tags": "chat,quality", "use_case": "chat", "ts": now},
        ])

        # Filter by tags
        rows = ev.query_results(cfg, tags="chat")
        assert len(rows) == 3

        # Filter by source
        rows = ev.query_results(cfg, source="rubric")
        assert len(rows) == 3

        # Haiku should lead
        summary = ev.window_summary(cfg, window_seconds=3600, now=now + 1)
        best = max(summary["by_model"].items(), key=lambda kv: kv[1]["avg_score"])
        assert best[0] == "aperture:claude-haiku"


# ------------------------------------------------------------------ Scenario 3: LLM-judge eval

class TestSampleLLMJudge:
    """An LLM-judge eval where a strong model rates responses.

    The llm_judge_score captures the judge's rating; the general score
    is the average across all judge dimensions.
    """

    def test_llm_judge_eval(self, tmpdir):
        cfg = _cfg(tmpdir)
        now = time.time()

        _submit_many(cfg, [
            {"run_name": "judge-coding-v1", "model_key": "local/qwen3.8-27b",
             "score": 7.5, "llm_judge_score": 7.5, "rubric_score": 7.2,
             "sample_count": 50, "dataset": "code-review-tasks",
             "version": "1.0", "source": "llm-judge",
             "tags": "coding,judge", "use_case": "coding", "ts": now},
            {"run_name": "judge-coding-v1", "model_key": "aperture:deepseek-v4-pro",
             "score": 8.8, "llm_judge_score": 8.8, "rubric_score": 8.5,
             "sample_count": 50, "dataset": "code-review-tasks",
             "version": "1.0", "source": "llm-judge",
             "tags": "coding,judge", "use_case": "coding", "ts": now},
        ])

        summary = ev.window_summary(cfg, window_seconds=3600, now=now + 1,
                                    use_case="coding")
        assert summary["model_count"] == 2
        assert summary["avg_llm_judge_score"] is not None
        assert summary["avg_rubric_score"] is not None

        # Verify llm_judge_score is preserved in query results
        rows = ev.query_results(cfg, source="llm-judge")
        assert len(rows) == 2
        for r in rows:
            assert r["llm_judge_score"] is not None
            assert r["rubric_score"] is not None


# ------------------------------------------------------------------ Scenario 4: Binary pass/fail regression test

class TestSampleRegressionEval:
    """A binary pass/fail regression eval (e.g. 'does the model still
    handle this edge case correctly?').

    pass_fail tracks the outcome; score is 100 for pass, 0 for fail.
    """

    def test_regression_eval(self, tmpdir):
        cfg = _cfg(tmpdir)
        now = time.time()

        _submit_many(cfg, [
            {"run_name": "regression-suite", "model_key": "local/qwen3.8-27b",
             "score": 100.0, "pass_fail": "pass", "sample_count": 25,
             "dataset": "edge-cases-v1", "version": "1.0", "source": "regression",
             "tags": "regression,critical", "use_case": "coding", "ts": now},
            {"run_name": "regression-suite", "model_key": "aperture:deepseek-v4-flash",
             "score": 0.0, "pass_fail": "fail", "sample_count": 25,
             "dataset": "edge-cases-v1", "version": "1.0", "source": "regression",
             "tags": "regression,critical", "use_case": "coding", "ts": now},
        ])

        summary = ev.window_summary(cfg, window_seconds=3600, now=now + 1)
        assert summary["pass_rate"] == 0.5  # 1 pass, 1 fail

        # Filter by pass_fail via query (check tags + source)
        rows = ev.query_results(cfg, tags="critical")
        assert len(rows) == 2

        # Verify pass/fail values
        pass_rows = [r for r in rows if r["pass_fail"] == "pass"]
        fail_rows = [r for r in rows if r["pass_fail"] == "fail"]
        assert len(pass_rows) == 1
        assert len(fail_rows) == 1
        assert pass_rows[0]["model_key"] == "local/qwen3.8-27b"
        assert fail_rows[0]["model_key"] == "aperture:deepseek-v4-flash"


# ------------------------------------------------------------------ Scenario 5: Time-windowed eval comparison

class TestSampleTimeWindowed:
    """Demonstrates time-windowed eval queries: only recent results
    count, older results are excluded.

    Useful for tracking model improvement over time (e.g. 'did the
    new model version actually get better?').
    """

    def test_time_windowed_filtering(self, tmpdir):
        cfg = _cfg(tmpdir)
        old_ts = time.time() - 86400 * 2  # 2 days ago
        recent_ts = time.time()

        # Old run (should be excluded by 1h window)
        ev.submit_result(cfg, run_name="bench-old", model_key="local/qwen3.8-27b",
                         score=65.0, sample_count=100, dataset="gsm8k",
                         source="benchmark", use_case="chat", ts=old_ts)

        # Recent run (should be included)
        ev.submit_result(cfg, run_name="bench-new", model_key="local/qwen3.8-27b",
                         score=82.0, sample_count=100, dataset="gsm8k",
                         source="benchmark", use_case="chat", ts=recent_ts)

        # 1h window: only recent
        rows_1h = ev.query_results(cfg, window_seconds=3600, now=recent_ts + 1)
        assert len(rows_1h) == 1
        assert rows_1h[0]["run_name"] == "bench-new"
        assert rows_1h[0]["score"] == 82.0

        # 24h window: still only recent (old is 2 days ago)
        rows_24h = ev.query_results(cfg, window_seconds=86400, now=recent_ts + 1)
        assert len(rows_24h) == 1

        # 3-day window: both included
        rows_3d = ev.query_results(cfg, window_seconds=86400 * 3, now=recent_ts + 1)
        assert len(rows_3d) == 2

        # latest_for_model with window returns the recent one
        latest = ev.latest_for_model(cfg, "local/qwen3.8-27b", window_seconds=3600,
                                     now=recent_ts + 1)
        assert latest is not None
        assert latest["score"] == 82.0


# ------------------------------------------------------------------ Scenario 6: Multi-model comparison across eval types

class TestSampleMultiModelComparison:
    """A comprehensive comparison: multiple models evaluated across
    multiple eval types (benchmark + rubric + judge).

    Demonstrates querying by model_key to get a per-model profile.
    """

    def test_multi_model_profile(self, tmpdir):
        cfg = _cfg(tmpdir)
        now = time.time()

        # Model A: strong on benchmarks, weaker on rubric
        _submit_many(cfg, [
            {"run_name": "humaneval", "model_key": "model-a",
             "score": 90.0, "sample_count": 164, "dataset": "humaneval",
             "source": "benchmark", "use_case": "coding", "ts": now},
            {"run_name": "mt-bench", "model_key": "model-a",
             "score": 7.0, "rubric_score": 7.0, "sample_count": 80,
             "dataset": "mt-bench", "source": "rubric", "use_case": "chat",
             "ts": now},
        ])

        # Model B: balanced
        _submit_many(cfg, [
            {"run_name": "humaneval", "model_key": "model-b",
             "score": 82.0, "sample_count": 164, "dataset": "humaneval",
             "source": "benchmark", "use_case": "coding", "ts": now},
            {"run_name": "mt-bench", "model_key": "model-b",
             "score": 8.5, "rubric_score": 8.5, "sample_count": 80,
             "dataset": "mt-bench", "source": "rubric", "use_case": "chat",
             "ts": now},
        ])

        # Per-model profile
        profile_a = ev.query_results(cfg, model_key="model-a")
        assert len(profile_a) == 2

        profile_b = ev.query_results(cfg, model_key="model-b")
        assert len(profile_b) == 2

        # Model A leads on coding benchmark
        coding_a = [r for r in profile_a if r["use_case"] == "coding"][0]
        coding_b = [r for r in profile_b if r["use_case"] == "coding"][0]
        assert coding_a["score"] > coding_b["score"]

        # Model B leads on chat rubric
        chat_a = [r for r in profile_a if r["use_case"] == "chat"][0]
        chat_b = [r for r in profile_b if r["use_case"] == "chat"][0]
        assert chat_b["score"] > chat_a["score"]

        # Overall summary
        summary = ev.window_summary(cfg, window_seconds=3600, now=now + 1)
        assert summary["model_count"] == 2
        assert summary["run_count"] == 2  # humaneval + mt-bench


# ------------------------------------------------------------------ Scenario 7: Version tracking

class TestSampleVersionTracking:
    """Demonstrates tracking eval results across dataset/scorer versions.

    When a new version of a benchmark or scorer is released, results
    from different versions are stored separately and can be compared.
    """

    def test_version_tracking(self, tmpdir):
        cfg = _cfg(tmpdir)
        now = time.time()

        # v1 of the benchmark
        ev.submit_result(cfg, run_name="gsm8k-eval", model_key="model-x",
                         score=70.0, sample_count=100, dataset="gsm8k",
                         version="1.0", source="benchmark", use_case="chat",
                         ts=now - 100)

        # v2 of the benchmark (improved dataset)
        ev.submit_result(cfg, run_name="gsm8k-eval", model_key="model-x",
                         score=78.0, sample_count=150, dataset="gsm8k",
                         version="2.0", source="benchmark", use_case="chat",
                         ts=now)

        # Query by version
        v1_rows = ev.query_results(cfg, version="1.0")
        v2_rows = ev.query_results(cfg, version="2.0")
        assert len(v1_rows) == 1
        assert len(v2_rows) == 1
        assert v1_rows[0]["score"] == 70.0
        assert v2_rows[0]["score"] == 78.0

        # latest_for_model returns the most recent (v2)
        latest = ev.latest_for_model(cfg, "model-x")
        assert latest["version"] == "2.0"
        assert latest["score"] == 78.0


# ------------------------------------------------------------------ Scenario 8: Disabled eval (graceful no-op)

class TestSampleDisabledEval:
    """When eval is disabled in config, all operations are no-ops."""

    def test_disabled_eval(self, tmpdir):
        cfg = _cfg(tmpdir)
        cfg["eval"]["enabled"] = False

        result = ev.submit_result(cfg, run_name="x", model_key="m", score=1)
        assert result is None

        rows = ev.query_results(cfg)
        assert rows == []

        summary = ev.window_summary(cfg, window_seconds=3600)
        assert summary["run_count"] == 0
        assert summary["avg_score"] is None
