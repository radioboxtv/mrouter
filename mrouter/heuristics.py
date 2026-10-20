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

from . import learning, providers as providers_mod

BENCH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_results.json")

REF_INPUT_TOKENS = 4000
REF_OUTPUT_TOKENS = 1000
REF_CACHED_TOKENS = 2000  # half the prompt assumed cache-hit when cache pricing exists


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


def _speed_score(model, fleet, bench, fam_info):
    """Measured tok/s for locals when benched; family estimate for cloud."""
    if model.get("is_local"):
        measured = _bench_lookup(bench, model["key"])
        if measured and measured.get("tokens_per_sec"):
            best = max((b.get("tokens_per_sec") or 0) for b in (bench or {}).values()) or 1
            return round(100 * measured["tokens_per_sec"] / max(best, 1e-9), 1)
        params = [m["params_b"] or 0.1 for m in fleet if m.get("is_local")]
        if params:
            return round(100 * min(params) / (model["params_b"] or 0.1), 1)
    est = (fam_info or {}).get("speed_est", config_default_speed(model))
    return float(est)


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


def load_bench():
    if not os.path.exists(BENCH_PATH):
        return {}
    try:
        with open(BENCH_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------- scoring engine

def _component_scores(model, fleet, config, bench):
    chat_cap, fam, fam_info = _cap(model, config, "chat")
    code_cap, _, _ = _cap(model, config, "coding")
    think_cap, _, _ = _cap(model, config, "thinking")
    notes = (config.get("family_kb", {}).get(fam, {}) or {}).get("notes", "") if fam else ""
    est_cost = providers_mod.estimate_cost_usd(model, REF_INPUT_TOKENS, REF_OUTPUT_TOKENS, REF_CACHED_TOKENS)
    comps = {
        "speed": _speed_score(model, fleet, bench, fam_info),
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
    }
    return comps, notes, est_cost


def eligible_models(models, config, use_case):
    """Hard filters before scoring. `sensitive` NEVER leaves the machine."""
    routing = config.get("routing", {}) or {}
    if use_case == "sensitive" and routing.get("sensitive_local_only", True):
        return [m for m in models if m.get("is_local")]
    allow_cloud = (routing.get("allow_cloud") or {}).get(use_case, True)
    if not allow_cloud:
        return [m for m in models if m.get("is_local")]
    return list(models)


def score_fleet(models, config, bench=None, learned=None):
    """Return {use_case: {label, description, models:[ranked...]}}."""
    bench = bench if bench is not None else load_bench()
    learned = learned if learned is not None else learning.load_learned(config)
    result = {}
    for uc, uc_cfg in config["use_cases"].items():
        pool = eligible_models(models, config, uc)
        ranked = []
        for m in pool:
            comps, notes, est_cost = _component_scores(m, models, config, bench)
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
                "est_cost_usd": round(est_cost, 6),
                "cost_input_per_m": round((m.get("cost_input_per_token") or 0) * 1e6, 4),
                "cost_output_per_m": round((m.get("cost_output_per_token") or 0) * 1e6, 4),
            })
        # tie-break: cheaper first, then local first
        ranked.sort(key=lambda r: (-r["score"], r["est_cost_usd"], not r["is_local"]))
        result[uc] = {"label": uc_cfg["label"], "description": uc_cfg["description"], "models": ranked}
    return result


def suggest(models, config, use_case, bench=None, learned=None):
    """Best model for one use case: (model_dict, score_info)."""
    ranked = score_fleet(models, config, bench, learned)[use_case]["models"]
    if not ranked:
        return None, None
    best = ranked[0]
    match = next((m for m in models if m["key"] == best["key"]), None)
    return match, best