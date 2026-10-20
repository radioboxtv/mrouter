"""Micro-benchmark: measure TTFT and tokens/sec for each model via LM Studio's API."""

import json
import re
import time
import urllib.request

from .heuristics import BENCH_PATH


def _stream_one(api_base, api_key, model_key, prompt, max_tokens):
    t0 = time.time()
    body = json.dumps({
        "model": model_key,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        api_base + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    ttft = None
    n_tokens = 0
    t_last = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token_text = delta.get("content") or delta.get("reasoning_content")
            if token_text:
                if ttft is None:
                    ttft = time.time() - t0
                n_tokens += 1
                t_last = time.time()
    total = (t_last or time.time()) - t0
    return ttft, n_tokens, total


def run_bench(models, config, api_key=None):
    api_base = config["lm_studio"]["api_base"]
    prompt = config["bench"]["prompt"]
    max_tokens = config["bench"]["max_tokens"]
    results = {}
    for m in models:
        key = m["key"]
        print(f"  benchmarking {m['display_name']} ({m['size_gb']} GB)...", flush=True)
        try:
            ttft, n_tokens, total = _stream_one(api_base, api_key, key, prompt, max_tokens)
            gen_secs = max(1e-9, (total - (ttft or 0)))
            tps = n_tokens / gen_secs if n_tokens else 0
            results[key] = {
                "ts": time.time(), "ttft_s": round(ttft, 3) if ttft else None,
                "tokens": n_tokens, "tokens_per_sec": round(tps, 1),
                "total_s": round(total, 3),
            }
            print(f"    ttft={round(ttft, 2) if ttft else 'n/a'}s  ~{tps:.1f} tok/s  "
                  f"({round(total, 1)}s incl. load)")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[key] = {"ts": time.time(), "error": str(e)}
    # Keep prior entries for models that failed this run, but drop stale successes.
    prior = {}
    try:
        with open(BENCH_PATH) as f:
            prior = json.load(f)
    except Exception:
        pass
    for k, v in prior.items():
        if k not in results and v.get("tokens_per_sec"):
            results.setdefault(k, v)
    with open(BENCH_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results