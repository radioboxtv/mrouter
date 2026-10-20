"""mrouter CLI: scan | suggest | bench | serve | classify | feedback | learn | stats"""

import argparse
import json
import sys

from . import bench as bench_mod
from . import categorize, discovery, heuristics, learning


def _load(include_cloud=True):
    config = discovery.load_config()
    models = discovery.discover_fleet(config, include_cloud=include_cloud)
    return config, models


def cmd_scan(args):
    config, models = _load(include_cloud=not args.local_only)
    if not models:
        print("No models discovered. Is LM Studio installed / Aperture reachable?")
        return 1
    local = [m for m in models if m["is_local"]]
    cloud = [m for m in models if not m["is_local"]]
    print(f"Discovered {len(models)} model(s): {len(local)} local (LM Studio), {len(cloud)} Aperture cloud.\n")

    print("LOCAL (LM Studio, $0):")
    hdr = f"{'MODEL':32} {'PARAMS':>7} {'QUANT':>6} {'SIZE':>7} {'CTX':>8} {'VISION':>6} {'TOOLS':>6} {'HYBRID':>7}"
    print(hdr)
    print("-" * len(hdr))
    for m in local:
        ctx = f"{m['max_ctx'] // 1024}k" if m["max_ctx"] else "?"
        print(f"{m['display_name']:32} {str(m['params_b'] or '?') + 'B':>7} "
              f"{str(m['quant_name'] or '?'):>6} {str(m['size_gb']) + 'G':>7} {ctx:>8} "
              f"{str(m['vision']):>6} {str(m['tool_use']):>6} {str(m['hybrid_attention']):>7}")

    print("\nCLOUD GATEWAY (priced per token):")
    hdr2 = f"{'MODEL ID':46} {'CTX':>8} {'IN $/M':>8} {'OUT $/M':>8} {'UPSTREAM':>12}"
    print(hdr2)
    print("-" * len(hdr2))
    for m in sorted(cloud, key=lambda x: (x.get("cost_output_per_token") or 0)):
        ctx = f"{(m['max_ctx'] or 0) // 1024}k" if m["max_ctx"] else "?"
        print(f"{m['api_id']:46} {ctx:>8} "
              f"{(m.get('cost_input_per_token') or 0) * 1e6:>8.2f} "
              f"{(m.get('cost_output_per_token') or 0) * 1e6:>8.2f} "
              f"{str(m.get('upstream') or ''):>12}")

    alive = discovery.server_alive(config)
    print(f"\nLM Studio server: {'UP' if alive else 'DOWN (run: lms server start)'}")
    print("Cost inputs: YES — Aperture GET /v1/models returns pricing {input, input_cache_read,")
    print("output} per token per model; responses return usage, so mrouter computes exact $/request.")
    return 0


def cmd_suggest(args):
    config, models = _load(include_cloud=not args.local_only)
    if not models:
        print("No models discovered.")
        return 1
    bench = heuristics.load_bench()
    ranked = heuristics.score_fleet(models, config, bench)

    if args.use_case:
        ranked = {args.use_case: ranked[args.use_case]}
    if args.json:
        print(json.dumps(ranked, indent=2))
        return 0

    for uc, r in ranked.items():
        print(f"\n=== {r['label']}  [{uc}] ===")
        print(f"    {r['description']}")
        for i, m in enumerate(r["models"][: args.top]):
            badge = "BEST" if i == 0 else f" #{i+1}"
            where = "local $0" if m["is_local"] else f"Aperture ~${m['est_cost_usd']:.4f}/req"
            adj = f" learned{m['learned_adj']:+.1f}" if m["learned_adj"] else ""
            print(f"  {badge:5} {m['display_name']:38} score {m['score']:>5}  [{where}]{adj}")
            print(f"        top factors: {m['reason']}")
            if m["notes"]:
                print(f"        family: {m['notes']}")
    if "sensitive" in ranked:
        print("\nNote on 'sensitive': HARD local-only filter — cloud models are excluded before")
        print("scoring, so sensitive prompts never leave this machine regardless of weights.")
    return 0


def cmd_bench(args):
    config, models = _load(include_cloud=False)
    if not models:
        print("No local LLMs discovered.")
        return 1
    if not discovery.server_alive(config):
        print("LM Studio server is down — starting it...")
        if not discovery.start_server(config):
            print("Could not reach LM Studio server. Open LM Studio or run: lms server start")
            return 1
    print("Running micro-benchmark (first run includes model load time)...\n")
    results = bench_mod.run_bench(models, config)
    ok = [r for r in results.values() if r.get("tokens_per_sec")]
    if ok:
        print("\nResults (saved to bench_results.json; used to replace params-based speed estimates):")
    return 0 if ok else 1


def cmd_serve(args):
    from . import proxy
    config = discovery.load_config()
    proxy.serve(config, host=args.host, port=args.port)
    return 0


def cmd_classify(args):
    config = discovery.load_config()
    text = " ".join(args.text) if args.text else sys.stdin.read()
    print(json.dumps(categorize.classify_task(text, config), indent=2))
    return 0


def cmd_feedback(args):
    config = discovery.load_config()
    rating = 1 if args.rating in ("+1", "1", "good") else -1
    entry = learning.append_feedback(config, args.use_case, args.model, rating, args.note or "")
    learning.save_learned(config)
    print(f"Recorded feedback ({'+' if rating > 0 else '-'}1) for {args.model} in '{args.use_case}'.")
    print("Adjusted scores will apply on the next suggest/serve. Run: python3 -m mrouter learn")
    return 0


def cmd_learn(args):
    config = discovery.load_config()
    out = learning.save_learned(config)
    entries = learning.load_log(config)
    print(f"Aggregated {len(entries)} log entries -> {len(out['adjustments'])} learned adjustment(s):")
    for k, v in out["adjustments"].items():
        print(f"  {k:70} adj {v['adj']:+.2f}  samples={v['samples']} "
              f"succ={v['success_rate']} fb +{v['pos']}/-{v['neg']} cost ${v['total_cost_usd']:.4f}")
    print(f"\nSaved: {learning.learned_path(config)}")
    return 0


def cmd_stats(args):
    config = discovery.load_config()
    entries = learning.load_log(config)
    total = sum(float(e.get("actual_cost_usd") or 0) for e in entries)
    by_uc = {}
    for e in entries:
        if e.get("type") == "request" and e.get("use_case"):
            by_uc.setdefault(e["use_case"], []).append(e)
    print(f"Logged entries: {len(entries)}   total actual spend: ${total:.4f}\n")
    for uc, rows in sorted(by_uc.items()):
        cost = sum(float(x.get("actual_cost_usd") or 0) for x in rows)
        print(f"  {uc:10} requests={len(rows):4}  spend=${cost:.4f}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="mrouter", description="Multiplexing heuristic model router (LM Studio + Aperture, Jev-aware)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scan", help="list discovered local + Aperture models and pricing")
    ps.add_argument("--local-only", action="store_true")
    pg = sub.add_parser("suggest", help="recommend best model per use case across the fleet")
    pg.add_argument("--use-case", choices=["chat", "coding", "sensitive", "thinking"])
    pg.add_argument("--top", type=int, default=3)
    pg.add_argument("--local-only", action="store_true")
    pg.add_argument("--json", action="store_true")
    sub.add_parser("bench", help="micro-benchmark local TTFT + tok/s")
    pv = sub.add_parser("serve", help="start multiplexing routing proxy")
    pv.add_argument("--host", default=None)
    pv.add_argument("--port", type=int, default=None)
    pc = sub.add_parser("classify", help="classify a prompt (Jev if key present, else local)")
    pc.add_argument("text", nargs="*")
    pf = sub.add_parser("feedback", help="record explicit feedback to improve routing")
    pf.add_argument("use_case", choices=["chat", "coding", "sensitive", "thinking"])
    pf.add_argument("model", help="fleet key, e.g. aperture:deepseek/deepseek-v4-flash")
    pf.add_argument("rating", choices=["+1", "-1"])
    pf.add_argument("--note", default="")
    sub.add_parser("learn", help="recompute learned adjustments from route_log.jsonl")
    sub.add_parser("stats", help="show routing log totals and spend")

    args = p.parse_args(argv)
    fn = {"scan": cmd_scan, "suggest": cmd_suggest, "bench": cmd_bench, "serve": cmd_serve,
          "classify": cmd_classify, "feedback": cmd_feedback, "learn": cmd_learn,
          "stats": cmd_stats}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())