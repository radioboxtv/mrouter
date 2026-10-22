"""Live model rankings from Artificial Analysis (optional capability prior).

Artificial Analysis publishes per-model evaluation data via a REST API
(base: https://artificialanalysis.ai/api/v2, auth via the `x-api-key`
header, free tier available via GET /language/models/free). Records carry
headline indices (Intelligence, Coding, Agentic) on a ~0-100 scale, plus
median performance (tokens/sec, TTFT).

This module turns those records into mrouter capability scores (0-10) and a
speed estimate, used by heuristics when `rankings.mode` is `live` or `blend`.
Local curated scores + measured benchmarks stay authoritative for cost and
local-model speed; rankings are a prior, not a bill.
"""

import json
import os
import re
import time
import urllib.request

from .providers import resolve_provider_key

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENDPOINT = "https://artificialanalysis.ai/api/v2/language/models/free"
CACHE_PATH = None  # overridable for tests


def _cache_file():
    return CACHE_PATH or os.path.join(BASE, "aa_cache.json")


def norm_slug(text):
    """Normalize a model name or slug for matching: lowercase, [ . _ space] -> '-'."""
    return re.sub(r"[._\s]+", "-", (text or "").lower()).strip("-")


def candidates_for_model(model):
    """Normalized match candidates for a fleet model id, most specific first."""
    key = model.get("upstream_id") or model.get("api_id") or model.get("key") or ""
    key = key.split("aperture:", 1)[-1] if key.startswith("aperture:") else key
    out = []
    tail = key.split("/")[-1]
    if tail:
        out.append(norm_slug(tail))
    full = norm_slug(key.replace("/", "-"))
    if full and full not in out:
        out.append(full)
    return out


def find_ranking(records, model):
    """Best-matching raw AA record for a fleet model, or None. Tolerant to
    publisher prefixes, case, and dot/dash version variants."""
    if not records:
        return None
    index = {}
    for rec in records:
        slug = rec.get("slug") or rec.get("model_name") or rec.get("name")
        if slug:
            index[norm_slug(slug)] = rec
    for cand in candidates_for_model(model):
        if cand in index:
            return index[cand]
    return None


def _fetch(rcfg, key):
    """Fetch raw AA records over HTTP. Raises on any failure."""
    req = urllib.request.Request(
        rcfg.get("endpoint", DEFAULT_ENDPOINT),
        headers={"x-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        payload = json.loads(r.read().decode())
    if isinstance(payload, dict):
        return payload.get("data") or []
    return payload or []


def load_rankings(config):
    """Live ranking records for this config. Cache-first: a fresh disk cache
    (per `cache_seconds`) is served without a key or fetch. Stale or missing
    cache triggers one fetch when a key exists; on fetch failure the stale
    cache is served (better than nothing). Disabled or key-less and
    cache-less -> [] (routing degrades to curated scores)."""
    rcfg = config.get("rankings", {}) or {}
    if not rcfg.get("enabled"):
        return []
    ttl = float(rcfg.get("cache_seconds", 86400))
    cache_file = _cache_file()
    cached = None
    try:
        with open(cache_file) as f:
            cached = json.load(f)
    except Exception:
        cached = None
    if cached and time.time() - cached.get("ts", 0) < ttl:
        return cached.get("data") or []
    key = resolve_provider_key(rcfg)
    if not key:
        return (cached or {}).get("data") or []
    try:
        data = _fetch(rcfg, key)
        with open(cache_file, "w") as f:
            json.dump({"ts": time.time(), "data": data}, f)
        return data
    except Exception:
        return (cached or {}).get("data") or []


def _scores_cache_key(rec):
    return rec.get("slug") or rec.get("model_name") or rec.get("name") or "?"


_scores_cache = {}


def _cached_scores(rec):
    key = _scores_cache_key(rec)
    hit = _scores_cache.get(key)
    if hit is None or hit[0] is not rec:
        scores = scores_from_record(rec)
        _scores_cache[key] = (rec, scores)
        return scores
    return hit[1]


def clear_scores_cache():
    _scores_cache.clear()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_10(v):
    if v is None:
        return None
    return round(max(0.0, min(100.0, v)) / 10, 2)


def scores_from_record(rec):
    """Map a raw AA record to mrouter's shape {capability_*, speed_est}
    with None for anything missing. Indices ~0-100 -> 0-10."""
    core = dict(rec or {})
    idx = {}
    for k in list(core.keys()):
        if k.startswith("artificial_analysis_"):
            idx.setdefault(k[len("artificial_analysis_"):], core[k])

    def val(*names):
        for n in names:
            v = _num(core.get(n))
            if v is not None:
                return v
            v = _num(idx.get(n))
            if v is not None:
                return v
        return None

    intel = val("intelligence_index", "intelligence")
    coding = val("coding_index", "coding")
    agentic = val("agentic_index", "agentic")
    chat = intel if intel is not None else agentic
    think = max([v for v in (intel, agentic) if v is not None], default=None)
    tps = val("median_output_tokens_per_second", "output_tokens_per_second")
    speed_est = None
    if tps is not None:
        speed_est = round(min(98.7, tps / 3.0), 1)
    return {
        "capability_chat": _to_10(chat),
        "capability_coding": _to_10(coding),
        "capability_thinking": _to_10(think),
        "speed_est": speed_est,
        "aa_slug": core.get("slug"),
    }