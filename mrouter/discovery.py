"""Model discovery: enumerate LM Studio models and extract routing heuristics.
Also provides the unified fleet (local + Aperture cloud) used for multiplexing.
"""

import json
import os
import re
import subprocess
import urllib.request

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")


def load_config():
    """Load config: $MROUTER_CONFIG (absolute), then config.json (gitignored,
    machine-local), then config.example.json (safe public defaults)."""
    env_config = os.environ.get("MROUTER_CONFIG")
    if env_config:
        with open(os.path.expanduser(env_config)) as f:
            return json.load(f)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("config.json", "config.example.json"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        "No config found: set MROUTER_CONFIG or provide config.json / config.example.json"
    )


def _expand(path):
    return os.path.expanduser(path)


def _run_lms(config):
    """Run `lms ls --json` and return parsed entries, or None if unavailable."""
    lms_bin = _expand(config["lm_studio"]["lms_bin"])
    if not os.path.exists(lms_bin):
        return None
    try:
        out = subprocess.run(
            [lms_bin, "ls", "--json"], capture_output=True, text=True, timeout=30
        )
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


def _model_config_path(config, indexed_id):
    """Locate the HF-style config.json under the LM Studio models root for a model."""
    root = _expand(config["lm_studio"]["models_root"])
    candidate = os.path.join(root, indexed_id)
    if os.path.isfile(os.path.join(candidate, "config.json")):
        return os.path.join(candidate, "config.json")
    # Fall back to a directory scan keyed on the tail of the identifier.
    tail = indexed_id.split("/")[-1]
    for dirpath, dirnames, filenames in os.walk(root):
        if "config.json" in filenames and dirpath.lower().endswith(tail.lower()):
            return os.path.join(dirpath, "config.json")
    for dirpath, dirnames, filenames in os.walk(root):
        if "config.json" in filenames and tail.lower() in dirpath.lower():
            return os.path.join(dirpath, "config.json")
    return None


def _extra_from_hf_config(config, indexed_id):
    """Read heuristic extras from the model's config.json (modality tokens, hybrid attention, ctx)."""
    extras = {}
    path = _model_config_path(config, indexed_id)
    if not path:
        return extras
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception:
        return extras

    keys = ("image_token_id", "boi_token_id", "audio_token_id", "boa_token_id")
    extras["multimodal_raw"] = any(k in cfg for k in keys)

    def find_key(obj, key):
        """Depth-first search for a key in nested dicts (configs vary per architecture)."""
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                r = find_key(v, key)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = find_key(v, key)
                if r is not None:
                    return r
        return None

    layers = find_key(cfg, "layer_types")
    if isinstance(layers, list) and layers:
        n_linear = sum(1 for t in layers if "linear_attention" in str(t))
        extras["hybrid_attention"] = 0 < n_linear < len(layers)
        extras["layer_count"] = len(layers)

    mpe = find_key(cfg, "max_position_embeddings")
    if isinstance(mpe, int) and mpe > 0:
        extras["max_position_embeddings"] = mpe
    return extras


def discover_models(config):
    """Return local model dicts with normalized heuristic features."""
    entries = _run_lms(config)
    models = []
    if entries is None:
        return models

    for e in entries:
        if e.get("type") not in ("llm",):
            continue  # embedding models are not chat candidates
        m = {
            "key": e.get("modelKey") or e.get("indexedModelIdentifier"),
            "api_id": e.get("modelKey"),
            "upstream_id": e.get("modelKey"),
            "provider": "lmstudio",
            "is_local": True,
            "display_name": e.get("displayName", e.get("modelKey", "?")),
            "publisher": e.get("publisher", "?"),
            "arch": e.get("architecture", "?"),
            "params_b": _parse_params(e.get("paramsString")),
            "size_gb": round((e.get("sizeBytes") or 0) / 2**30, 2),
            "quant_bits": (e.get("quantization") or {}).get("bits"),
            "quant_name": (e.get("quantization") or {}).get("name"),
            "max_ctx": e.get("maxContextLength"),
            "max_output": None,
            "vision": bool(e.get("vision")),
            "tool_use": bool(e.get("trainedForToolUse")),
            "format": e.get("format"),
            "cost_input_per_token": 0.0,
            "cost_output_per_token": 0.0,
            "cost_cache_read_per_token": 0.0,
        }
        extras = _extra_from_hf_config(config, e.get("indexedModelIdentifier") or m["key"])
        m["hybrid_attention"] = extras.get("hybrid_attention", False)
        if m.get("max_ctx") is None:
            m["max_ctx"] = extras.get("max_position_embeddings")
        m["multimodal"] = m["vision"] or extras.get("multimodal_raw", False)
        models.append(m)
    return models


def discover_fleet(config, include_cloud=True):
    """Unified fleet: local LM Studio models + Aperture cloud models."""
    fleet = discover_models(config)
    if include_cloud and (config.get("providers", {}).get("aperture", {}) or {}).get("enabled"):
        from . import providers
        fleet = fleet + providers.discover_aperture_models(config)
    return fleet


def _parse_params(s):
    """'27B' -> 27.0, 'E4B' -> 4.0, '8x7B' -> 56.0; None if unparseable."""
    if not s:
        return None
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([KMB])", s.upper()):
        total += float(num) * {"K": 1e-3, "M": 1e-2, "B": 1.0}[unit]
    return round(total, 2) or None


def server_alive(config):
    try:
        with urllib.request.urlopen(config["lm_studio"]["api_base"] + "/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(config):
    lms_bin = _expand(config["lm_studio"]["lms_bin"])
    try:
        subprocess.run([lms_bin, "server", "start"], capture_output=True, text=True, timeout=60)
        return server_alive(config)
    except Exception:
        return False