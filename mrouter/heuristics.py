"""Heuristic scoring: turn model features into per-use-case scores (0-100).

Multiplexes across the LOCAL fleet (LM Studio, $0) and the Aperture cloud
fleet (39 models with live pricing). Cost enters as `cost_efficiency`
(log-scaled estimated cost per representative request: free local = 100,
priciest flagship ~ 0) and `value` (capability per dollar). Learned
adjustments from observed routing outcomes are layered on top.
"""

import json
import math
import os
import time

from . import learning, rankings, reliability as reliability_mod
from . import providers as providers_mod

BENCH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_results.json")

REF_INPUT_TOKENS = 4000
REF_OUTPUT_TOKENS = 1000
REF_CACHED_TOKENS = 2000  # half the prompt assumed cache-hit when cache pricing exists

# Neutral score when NO real measurement exists at all (no live traffic, no
# bench entry) -- used for TTFT, which has no curated per-family fallback
# the way throughput has `speed_est`. Deliberately a flat, honest "unknown"
# rather than an invented number: a brand-new model should neither win nor
# lose on latency it's never actually produced.
NEUTRAL_PERF_SCORE = 50.0


# ---------------------------------------------------------------- family KB

def _cap(model, config, kind):
    """Family capability score (0-10) from the knowledge base, matched on name/arch."""
    kb = config.get("family_kb", {})
    upstream_id = model.get("upstream_id") or model.get("key") or ""
    hay = " ".join([upstream_id, model.get("key", ""), model.get("arch", "")]).lower()
    for fam, info in kb.items():
        if any(pat.lower() in hay for pat in info.get("match", [])):
            val = info.get(f"capability_{kind}")
            if val is None:
                val = config["default_family_scores"][f"capability_{kind}"]
            return float(val), fam, info
    return float(config["default_family_scores"][f"capability_{kind}"]), None, {}


def _live_capability(model, config, kind, live_cache):
    """Live-records capability for `kind`; live applies to cloud models only.
    Returns ((value, source) | None). Local models keep curated scores:
    benchmarks measure hosted/full-precision weights, not local quants."""
    if model.get("is_local"):
        return None
    mode, records = live_cache
    if mode not in ("live", "blend") or not records:
        return None
    rec = rankings.find_ranking(records, model)
    if rec is None:
        return None
    scores = rankings._cached_scores(rec)
    v = scores.get(f"capability_{kind}")
    if v is None:
        return None
    return float(v), "artificial-analysis"


def _resolve_capability(model, config, kind, live_cache):
    """Final capability for `kind` -> (value_0_to_10, fam_key, fam_info, source).

    Precedence per config `family_kb_mode` / `rankings.mode`:
      curated -> hand-written KB only
      live    -> AA scores win for cloud models (KB remains the vision/notes source)
      blend   -> weighted average of KB and AA for cloud models that match both
    Local models always use the KB; learned adjustments apply later, on top.
    """
    cur_val, fam, info = _cap(model, config, kind)
    mode = live_cache[0]
    if mode == "curated":
        return cur_val, fam, info, "curated"
    live = _live_capability(model, config, kind, live_cache)
    if live is None:
        return cur_val, fam, info, "curated"
    live_val, source = live
    if mode == "live":
        return live_val, fam, info, source
    if mode == "blend":
        w = float((config.get("rankings", {}) or {}).get("blend_weight", 0.5))
        w = max(0.0, min(1.0, w))
        return cur_val * (1 - w) + live_val * w, fam, info, "blend"
    return cur_val, fam, info, "curated"


# ---------------------------------------------------------------- feature scores

def _ctx_score(model):
    ctx = model.get("max_ctx") or 0
    if ctx >= 524288:
        return 100
    if ctx >= 262144:
        return 92
    if ctx >= 131072:
        return 80
    if ctx >= 65536:
        return 60
    if ctx >= 32768:
        return 40
    return 25


def _mem_fit_score(model, config):
    """Local: how comfortably weights fit in RAM. Cloud: resident by definition."""
    if not model.get("is_local"):
        return 100
    ram = config.get("hardware", {}).get("total_ram_gb", 36)
    size = model.get("size_gb") or 0
    budget = 0.75 * ram
    ratio = size / budget if budget else 1.0
    if ratio <= 0.35:
        return 100
    return max(0.0, round(100 - (ratio - 0.35) * 180, 1))


def _speed_score(model, fleet, bench, fam_info, live_perf=None, live_min_samples=5):
    """Real-world throughput (tokens/sec) score for ANY model, local or cloud.

    Precedence, honest about provenance via the returned source label:
      1. 'live'      -- real p90 tokens/sec from metrics_store (this fleet's
                        own measured traffic), once >= live_min_samples
                        requests have been logged for this model. Applies to
                        BOTH local and cloud models -- a cloud model that has
                        actually been routed enough times gets scored on what
                        it really does, not a guessed number.
      2. 'bench'     -- LOCAL ONLY: the synthetic micro-bench
                        (bench_results.json) from `mrouter bench`/autobench.
      3. 'estimated' -- the curated family `speed_est` (cloud) or a
                        params-based relative guess (local with neither live
                        nor bench data, e.g. brand new / never benched).

    Returns (score_0_to_100, source). live_perf: optional
    {model_key: {tokens_per_sec_p90, samples}} from
    metrics_store.window_summary()['models']; None/omitted means no live data
    was loaded (caller's choice -- score_fleet auto-loads it by default).
    """
    live = (live_perf or {}).get(model["key"])
    if live and (live.get("samples") or 0) >= live_min_samples and live.get("tokens_per_sec_p90"):
        # Real traffic is normalized across the WHOLE fleet (local + cloud
        # together), not split by locality: once we have honest measurements,
        # a fast cloud model and a fast local model are directly comparable,
        # and that comparison is exactly what routing needs.
        best = max(
            (((live_perf or {}).get(m["key"]) or {}).get("tokens_per_sec_p90") or 0)
            for m in fleet
        ) or 1
        return round(100 * live["tokens_per_sec_p90"] / max(best, 1e-9), 1), "live"
    if model.get("is_local"):
        measured = _bench_lookup(bench, model["key"])
        if measured and measured.get("tokens_per_sec"):
            best = max((b.get("tokens_per_sec") or 0) for b in (bench or {}).values()) or 1
            return round(100 * measured["tokens_per_sec"] / max(best, 1e-9), 1), "bench"
        params = [m["params_b"] or 0.1 for m in fleet if m.get("is_local")]
        if params:
            return round(100 * min(params) / (model["params_b"] or 0.1), 1), "estimated"
    est = (fam_info or {}).get("speed_est", config_default_speed(model))
    return float(est), "estimated"


def _ttft_score(model, fleet, bench, live_perf=None, live_min_samples=5):
    """Real-world time-to-first-token (latency) score for ANY model, local or
    cloud. Lower TTFT is better; score is normalized against the best (lowest)
    TTFT observed among comparable peers, same precedence/provenance shape as
    `_speed_score`:

      1. 'live'      -- real p90 TTFT from metrics_store, >= live_min_samples.
                        Local and cloud both qualify.
      2. 'bench'     -- LOCAL ONLY: the synthetic micro-bench's measured ttft_s.
      3. 'estimated' -- NEUTRAL_PERF_SCORE flat prior. There is no curated
                        per-family TTFT guess in family_kb (unlike speed_est)
                        -- inventing one would defeat the point of this
                        component, so an honestly-unknown model gets a flat,
                        non-penalizing score instead of a fabricated number.

    Returns (score_0_to_100, source).
    """
    live = (live_perf or {}).get(model["key"])
    if live and (live.get("samples") or 0) >= live_min_samples and live.get("ttft_p90_s") is not None:
        # Same whole-fleet normalization rationale as _speed_score: local vs.
        # cloud TTFT are directly comparable once both are real measurements.
        ttfts = [((live_perf or {}).get(m["key"]) or {}).get("ttft_p90_s") for m in fleet]
        ttfts = [t for t in ttfts if t is not None and t > 0]
        best = min(ttfts) if ttfts else live["ttft_p90_s"]
        return round(100 * best / max(live["ttft_p90_s"], 1e-9), 1), "live"
    if model.get("is_local"):
        measured = _bench_lookup(bench, model["key"])
        if measured and measured.get("ttft_s"):
            candidates = [b.get("ttft_s") for b in (bench or {}).values() if b.get("ttft_s")]
            best = min(candidates) if candidates else measured["ttft_s"]
            return round(100 * best / max(measured["ttft_s"], 1e-9), 1), "bench"
    return NEUTRAL_PERF_SCORE, "estimated"


def config_default_speed(model):
    return 70.0


def _params_quality_score(model, fleet, config):
    """Local models: fleet-relative params. Cloud: proxied by thinking capability."""
    if not model.get("is_local"):
        think, _, _ = _cap(model, config, "thinking")
        return think * 10
    locals_ = [m for m in fleet if m.get("is_local")]
    params = [(m["params_b"] or 0.1) for m in locals_] or [0.1]
    return round(100 * (model["params_b"] or 0.1) / max(params), 1)


def _multimodal_score(model, fam_info):
    vis = model.get("multimodal")
    if vis is None:
        vis = (fam_info or {}).get("vision")
    if vis is None:
        vis = model.get("vision")
    return 100 if vis else 0


def _tool_use_score(model):
    return 100 if model.get("tool_use") else 30


def _cost_efficiency_score(model):
    """100 = free (local). Decays log-wise with estimated request cost."""
    cost = providers_mod.estimate_cost_usd(model, REF_INPUT_TOKENS, REF_OUTPUT_TOKENS, REF_CACHED_TOKENS)
    if cost <= 0:
        return 100.0
    # $0.001 -> ~97, $0.01 -> ~78, $0.05 -> ~55, $0.10 -> ~42, $0.20 -> ~28, $0.14(Opus-ish ref) ~34
    return round(max(0.0, 100 - 22 * math.log10(cost / 0.001 + 1) * 10 / 3.7), 1)


def _value_score(model, config):
    """Capability-average per dollar, normalized: local (free, decent) sets the bar."""
    cost = providers_mod.estimate_cost_usd(model, REF_INPUT_TOKENS, REF_OUTPUT_TOKENS, REF_CACHED_TOKENS)
    caps = [_cap(model, config, k)[0] for k in ("chat", "coding", "thinking")]
    avg = sum(caps) / 3
    if cost <= 0:
        # Free: value anchored to capability (local models are the efficiency baseline)
        return round(min(100.0, avg * 11), 1)
    # capability points per cent: great value (DeepSeek Flash) >> flagship
    per_cent = avg / (cost * 100)
    return round(min(100.0, per_cent * 2.2), 1)


def _privacy_score(model):
    return 100 if model.get("is_local") else 0


def _bench_lookup(bench, key):
    if not bench:
        return None
    entry = bench.get(key)
    if not entry:
        return None
    age_h = (time.time() - entry.get("ts", 0)) / 3600
    return entry if age_h < 7 * 24 else None


def stale_local_models(models, bench, max_age_hours: float = 7 * 24):
    """Local models with no fresh bench entry (missing, failed, or expired).

    Powers automatic background benchmarking in the proxy: `serve` calls this
    on every fleet refresh so newly-downloaded local models get measured
    without a human ever running `mrouter bench`.
    """
    out = []
    for m in models:
        if not m.get("is_local"):
            continue
        entry = (bench or {}).get(m["key"])
        if not entry or not entry.get("tokens_per_sec"):
            out.append(m)
            continue
        age_h = (time.time() - entry.get("ts", 0)) / 3600
        if age_h >= max_age_hours:
            out.append(m)
    return out


def load_bench():
    if not os.path.exists(BENCH_PATH):
        return {}
    try:
        with open(BENCH_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------- scoring engine

def _component_scores(model, fleet, config, bench, live_cache, live_perf=None):
    chat_cap, fam, fam_info, chat_src = _resolve_capability(model, config, "chat", live_cache)
    code_cap, _, _, code_src = _resolve_capability(model, config, "coding", live_cache)
    think_cap, _, _, think_src = _resolve_capability(model, config, "thinking", live_cache)
    capability_source = next((s for s in (chat_src, code_src, think_src) if s != "curated"), "curated")
    notes = (config.get("family_kb", {}).get(fam, {}) or {}).get("notes", "") if fam else ""
    est_cost = providers_mod.estimate_cost_usd(model, REF_INPUT_TOKENS, REF_OUTPUT_TOKENS, REF_CACHED_TOKENS)
    live_min_samples = int((config.get("metrics", {}) or {}).get("live_speed_min_samples", 5))
    speed, speed_source = _speed_score(model, fleet, bench, fam_info, live_perf, live_min_samples)
    ttft, ttft_source = _ttft_score(model, fleet, bench, live_perf, live_min_samples)
    comps = {
        "speed": speed,
        "ttft": ttft,
        "capability_chat": chat_cap * 10,
        "capability_coding": code_cap * 10,
        "capability_thinking": think_cap * 10,
        "capability_avg": round(((chat_cap + code_cap + think_cap) / 3) * 10, 1),
        "context": _ctx_score(model),
        "multimodal": _multimodal_score(model, fam_info),
        "tool_use": _tool_use_score(model),
        "mem_fit": _mem_fit_score(model, config),
        "params_quality": _params_quality_score(model, fleet, config),
        "cost_efficiency": _cost_efficiency_score(model),
        "value": _value_score(model, config),
        "privacy": _privacy_score(model),
        "reliability": None,  # filled in by score_fleet once stats are resolved
    }
    sources = {"capability": capability_source, "speed": speed_source, "ttft": ttft_source}
    return comps, notes, est_cost, sources


def _load_live_perf(config):
    """Best-effort {model_key: {tokens_per_sec_p90, ttft_p90_s, samples}} from
    metrics_store over the configured default window, for BOTH local and
    cloud models. Never raises: scoring must work even if the metrics db is
    missing/corrupt/disabled -- it's a feed-in, not a dependency."""
    if not (config.get("metrics", {}) or {}).get("enabled", True):
        return {}
    try:
        from . import metrics_store
        window_s = metrics_store.resolve_window_seconds(config)
        summary = metrics_store.window_summary(config, window_seconds=window_s)
        return {
            key: {
                "tokens_per_sec_p90": m.get("tokens_per_sec_p90"),
                "ttft_p90_s": m.get("ttft_p90_s"),
                "samples": m.get("requests", 0),
            }
            for key, m in (summary.get("models") or {}).items()
        }
    except Exception:
        return {}


def eligible_models(models, config, use_case):
    """Hard filters before scoring. `sensitive` NEVER leaves the machine."""
    routing = config.get("routing", {}) or {}
    if use_case == "sensitive" and routing.get("sensitive_local_only", True):
        return [m for m in models if m.get("is_local")]
    allow_cloud = (routing.get("allow_cloud") or {}).get(use_case, True)
    if not allow_cloud:
        return [m for m in models if m.get("is_local")]
    return list(models)


def score_fleet(models, config, bench=None, learned=None, live_records=None, reliability=None, live_perf=None):
    """Return {use_case: {label, description, models:[ranked...]}}.

    live_records: optional raw ranking records (Artificial Analysis shape).
    reliability:  optional {model_key: {samples, success_rate, score}} from
                  reliability.compute_reliability(); loaded from log when None.
    live_perf:    optional {model_key: {tokens_per_sec_p90, ttft_p90_s, samples}}
                  feeding the speed/ttft components from real traffic instead
                  of guessed/curated numbers, for LOCAL AND CLOUD models alike.
                  Loaded from metrics_store (the configured default window)
                  when None, unless metrics are disabled, in which case it
                  stays empty and speed/ttft fall back to bench/estimated
                  values as before this feature existed.
    """
    bench = bench if bench is not None else load_bench()
    learned = learned if learned is not None else learning.load_learned(config)
    if live_records is None:
        live_records = rankings.load_rankings(config)
    if reliability is None:
        reliability = reliability_mod.compute_reliability(config)
    if live_perf is None:
        live_perf = _load_live_perf(config)
    mode = (config.get("rankings", {}) or {}).get("mode", "curated")
    live_cache = (mode, live_records)
    routing = config.get("routing", {}) or {}
    rel_floor = float(routing.get("reliability_floor", 0.0))
    rel_min_n = int(routing.get("reliability_min_samples", 5))
    result = {}
    for uc, uc_cfg in config["use_cases"].items():
        pool = eligible_models(models, config, uc)
        # Hard reliability floor: exclude chronically failing models (proven flaky only)
        pool = [m for m in pool if not reliability_mod.is_below_floor(
            reliability, m["key"], floor=rel_floor, min_samples=rel_min_n)]
        ranked = []
        for m in pool:
            comps, notes, est_cost, sources = _component_scores(m, models, config, bench, live_cache, live_perf)
            comps["reliability"] = reliability_mod.reliability_score(reliability, m["key"])
            weights = uc_cfg["weights"]
            base = sum(comps[k] * w for k, w in weights.items() if k in comps)
            adj = learning.adjustment_for(learned, uc, m["key"])
            score = round(base + adj, 1)
            top = sorted(weights.items(), key=lambda kv: -kv[1])[:3]
            contrib = ", ".join(f"{k}={comps.get(k)}" for k, _ in top)
            ranked.append({
                "key": m["key"], "api_id": m.get("api_id"), "provider": m.get("provider"),
                "is_local": bool(m.get("is_local")), "display_name": m["display_name"],
                "score": score, "base_score": round(base, 1), "learned_adj": adj,
                "components": comps, "reason": contrib, "notes": notes,
                "capability_source": sources["capability"],
                "speed_source": sources["speed"], "ttft_source": sources["ttft"],
                "est_cost_usd": round(est_cost, 6),
                "cost_input_per_m": round((m.get("cost_input_per_token") or 0) * 1e6, 4),
                "cost_output_per_m": round((m.get("cost_output_per_token") or 0) * 1e6, 4),
            })
        # tie-break: cheaper first, then local first
        ranked.sort(key=lambda r: (-r["score"], r["est_cost_usd"], not r["is_local"]))
        result[uc] = {"label": uc_cfg["label"], "description": uc_cfg["description"], "models": ranked}
    return result


def suggest(models, config, use_case, bench=None, learned=None, reliability=None):
    """Best model for one use case: (model_dict, score_info)."""
    ranked = score_fleet(models, config, bench, learned, reliability=reliability)[use_case]["models"]
    if not ranked:
        return None, None
    best = ranked[0]
    match = next((m for m in models if m["key"] == best["key"]), None)
    return match, best