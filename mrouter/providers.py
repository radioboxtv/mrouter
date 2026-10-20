"""Cloud provider support: catalog + pricing from an OpenAI-compatible gateway, cost estimation.

Answers 'are cost inputs available?': gateways that publish a per-model
`pricing` block in GET /v1/models give mrouter everything it needs:
  `pricing` {input, input_cache_read, output} as USD-per-token strings,
  plus context_window_tokens / max_output_tokens.
Chat responses carry `usage` token counts, so actual cost per request is
computable exactly. The gateway returns no dollar-cost field itself.
"""

import json
import os
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE, "aperture_cache.json")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def resolve_provider_key(cfg):
    """API key for a cloud provider config: api_key_env wins over static api_key."""
    env_var = cfg.get("api_key_env")
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
    return cfg.get("api_key")


def _fetch_models(config):
    cfg = config["providers"]["aperture"]
    headers = {"Content-Type": "application/json"}
    key = resolve_provider_key(cfg)
    if key and key != "no-key-required":
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(cfg["api_base"].rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def discover_aperture_models(config):
    """Return model dicts in the same normalized shape as local models."""
    cfg = config["providers"]["aperture"]
    if not cfg.get("enabled"):
        return []
    ttl = cfg.get("catalog_cache_seconds", 300)

    raw = None
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cached = json.load(f)
            if time.time() - cached.get("ts", 0) < ttl:
                raw = cached["data"]
        except Exception:
            raw = None
    if raw is None:
        try:
            raw = _fetch_models(config)
            with open(CACHE_PATH, "w") as f:
                json.dump({"ts": time.time(), "data": raw}, f)
        except Exception:
            # fall back to stale cache if present
            if os.path.exists(CACHE_PATH):
                try:
                    with open(CACHE_PATH) as f:
                        raw = json.load(f)["data"]
                except Exception:
                    return []
            else:
                return []

    models = []
    for e in raw.get("data", []):
        pricing = e.get("pricing") or {}
        p_in = _num(pricing.get("input"))
        p_out = _num(pricing.get("output"))
        models.append({
            "key": f"aperture:{e['id']}",
            "api_id": e["id"],
            "upstream_id": e["id"],
            "provider": "aperture",
            "is_local": False,
            "display_name": e.get("display_name") or e["id"],
            "publisher": ((e.get("metadata") or {}).get("provider") or {}).get("id", "aperture"),
            "arch": "cloud",
            "params_b": None,
            "size_gb": 0.0,
            "quant_bits": None,
            "quant_name": None,
            "max_ctx": e.get("context_window_tokens"),
            "max_output": e.get("max_output_tokens"),
            "vision": None,  # resolved from family KB (None -> inferred there)
            "tool_use": True,  # Aperture chat models are tool-capable endpoints
            "hybrid_attention": False,
            "multimodal": None,
            "cost_input_per_token": p_in,
            "cost_output_per_token": p_out,
            "cost_cache_read_per_token": _num(pricing.get("input_cache_read")),
            "upstream": ((e.get("metadata") or {}).get("provider") or {}).get("id"),
        })
    return models


def estimate_cost_usd(model, est_input_tokens=4000, est_output_tokens=1000, cached_input_tokens=0):
    """Estimated USD for one representative request. Local models: $0."""
    if model.get("is_local"):
        return 0.0
    p_in = model.get("cost_input_per_token") or 0.0
    p_out = model.get("cost_output_per_token") or 0.0
    p_cache = model.get("cost_cache_read_per_token")
    if p_cache is None:
        p_cache = p_in
    non_cached = max(0, est_input_tokens - cached_input_tokens)
    return non_cached * p_in + cached_input_tokens * p_cache + est_output_tokens * p_out


def actual_cost_usd(model, usage):
    """Exact USD from a response's usage block. None if inputs are missing."""
    if model.get("is_local"):
        return 0.0
    if not usage:
        return None
    p_in = model.get("cost_input_per_token")
    p_out = model.get("cost_output_per_token")
    if p_in is None or p_out is None:
        return None
    p_cache = model.get("cost_cache_read_per_token")
    if p_cache is None:
        p_cache = p_in
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    cached = min(cached, prompt_tokens)
    return (prompt_tokens - cached) * p_in + cached * p_cache + completion_tokens * p_out
