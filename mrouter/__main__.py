"""mrouter CLI: init | serve | doctor | feedback  (+ debug: scan/suggest/bench/classify/learn/stats/reliability)"""

import argparse
import json
import sys
import time

from . import bench as bench_mod
from . import categorize, discovery, heuristics, learning
from . import eval_results as eval_mod


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
    # serve_forever restarts the HTTP server on crashes — matters for launchd/daemon use,
    # where there's no human watching to notice a dead process and restart it by hand.
    proxy.serve_forever(config, host=args.host, port=args.port)
    return 0


def cmd_classify(args):
    config = discovery.load_config()
    text = " ".join(args.text) if args.text else sys.stdin.read()
    print(json.dumps(categorize.classify_task(text, config), indent=2))
    return 0


def cmd_feedback(args):
    config = discovery.load_config()
    rating = 1 if args.rating in ("+1", "1", "good") else -1
    learning.append_feedback(config, args.use_case, args.model, rating, args.note or "")
    print(f"Recorded feedback ({'+' if rating > 0 else '-'}1) for {args.model} in '{args.use_case}'.")
    print("Learned adjustments were recomputed automatically — already live for the next request.")
    return 0


def cmd_learn(args):
    # Manual override only: normal operation never needs this — `serve` recomputes
    # learned.json automatically after every logged request and every feedback vote.
    # Useful right after hand-editing route_log.jsonl, or to force a recompute now.
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


def cmd_reliability(args):
    from mrouter import reliability as rel_mod
    config = discovery.load_config()
    stats = rel_mod.compute_reliability(config)
    if not stats:
        print("No request data yet. Route some traffic, then run: mrouter reliability")
        return 0
    routing = config.get("routing", {}) or {}
    floor = float(routing.get("reliability_floor", 0.5))
    min_n = int(routing.get("reliability_min_samples", 5))
    print(f"{'MODEL':50} {'SCORE':>6} {'SUCCESS':>8} {'SAMPLES':>8}  STATUS")
    print("-" * 88)
    for k in sorted(stats, key=lambda k: -stats[k]["score"]):
        v = stats[k]
        sr = f"{v['success_rate']*100:.0f}%" if v["success_rate"] is not None else "?"
        below = rel_mod.is_below_floor(stats, k, floor=floor, min_samples=min_n)
        flag = "  ⚠ BELOW FLOOR" if below else ""
        print(f"  {k:48} {v['score']:>6.1f} {sr:>8} {v['samples']:>8}{flag}")
    print(f"\nFloor: success_rate < {floor:.0%} with ≥{min_n} samples → excluded from routing pools")
    return 0


def cmd_classifier_reliability(args):
    """Decision-provider (baseten/jev) reliability — the circuit breaker that
    skips a proven-bad provider before paying its network timeout, distinct
    from `mrouter reliability` which covers routed MODELS, not classifiers."""
    from mrouter import reliability as rel_mod
    config = discovery.load_config()
    stats = rel_mod.compute_classifier_reliability(config)
    if not stats:
        print("No classifier attempts logged yet. Route some 'auto' traffic, then re-run.")
        return 0
    decision = config.get("decision", {}) or {}
    floor = float(decision.get("reliability_floor", 0.5))
    min_n = int(decision.get("reliability_min_samples", 5))
    print(f"{'PROVIDER':20} {'SCORE':>6} {'SUCCESS':>8} {'SAMPLES':>8}  STATUS")
    print("-" * 58)
    for k in sorted(stats, key=lambda k: -stats[k]["score"]):
        v = stats[k]
        sr = f"{v['success_rate']*100:.0f}%" if v["success_rate"] is not None else "?"
        below = rel_mod.is_below_floor(stats, k, floor=floor, min_samples=min_n)
        flag = "  ⚠ CIRCUIT OPEN" if below else ""
        print(f"  {k:18} {v['score']:>6.1f} {sr:>8} {v['samples']:>8}{flag}")
    print(f"\nBreaker: success_rate < {floor:.0%} with ≥{min_n} samples → skipped without a network call")
    return 0


def cmd_tray(args):
    from . import trayapp
    return trayapp.main()


def cmd_doctor(args):
    from . import doctor
    config = discovery.load_config()
    report = doctor.run(config, verbose=args.verbose)
    doctor.print_report(report, verbose=args.verbose)
    if args.verbose:
        print("\n--- scan ---")
        cmd_scan(argparse.Namespace(local_only=False))
        print("\n--- suggest ---")
        cmd_suggest(argparse.Namespace(use_case=None, top=3, local_only=False, json=False))
        print("\n--- stats ---")
        cmd_stats(argparse.Namespace())
        print("\n--- reliability ---")
        cmd_reliability(argparse.Namespace())
        print("\n--- classifier-reliability ---")
        cmd_classifier_reliability(argparse.Namespace())
    return 1 if report.has_fail else 0


def cmd_init(args):
    from . import init as init_mod
    return init_mod.run(
        gateway_url=args.gateway_url,
        gateway_key_env=args.gateway_key_env,
        skip_launchd=args.no_launchd,
        skip_hermes=args.no_hermes,
        non_interactive=args.yes,
    )


def cmd_eval(args):
    config = discovery.load_config()
    sub = getattr(args, "eval_cmd", None)

    if sub == "submit":
        result = eval_mod.submit_result(
            config,
            run_name=args.run_name,
            model_key=args.model,
            score=args.score,
            pass_fail=args.pass_fail,
            rubric_score=args.rubric_score,
            llm_judge_score=args.llm_judge_score,
            sample_count=args.samples or 0,
            dataset=args.dataset,
            version=args.version,
            source=args.source or "cli",
            tags=args.tags,
            use_case=args.use_case,
        )
        if result is None:
            print("Eval disabled in config.")
            return 1
        print(f"Submitted eval result: run={result['run_name']} model={result['model_key']} "
              f"score={result['score']} n={result['sample_count']}")
        return 0

    if sub == "list":
        rows = eval_mod.query_results(
            config,
            run_name=args.run_name,
            model_key=args.model,
            dataset=args.dataset,
            source=args.source,
            tags=args.tags,
            use_case=args.use_case,
            version=args.version,
            window_seconds=args.window,
        )
        if not rows:
            print("No eval results found.")
            return 0
        print(f"{'RUN':24} {'MODEL':32} {'SCORE':>6} {'N':>6} {'SOURCE':12} {'TS'}")
        print("-" * 100)
        for r in rows:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r["ts"]))
            print(f"{r['run_name']:24} {r['model_key']:32} "
                  f"{str(r.get('score') or ''):>6} {str(r.get('sample_count') or 0):>6} "
                  f"{str(r.get('source') or ''):12} {ts}")
        return 0

    if sub == "summary":
        from . import metrics_store
        ws = metrics_store.resolve_window_seconds(config, args.window)
        summary = eval_mod.window_summary(config, window_seconds=ws)
        print(f"Eval summary (window={args.window or 'default'}):")
        print(f"  runs={summary['run_count']} models={summary['model_count']} "
              f"samples={summary['total_samples']} avg_score={summary['avg_score']} "
              f"pass_rate={summary['pass_rate']}")
        if summary["by_model"]:
            print("\nBy model:")
            for k, v in sorted(summary["by_model"].items()):
                print(f"  {k:32} avg={v['avg_score']} n={v['sample_count']} "
                      f"latest={v['latest_score']}")
        if summary["by_run"]:
            print("\nBy run:")
            for k, v in sorted(summary["by_run"].items()):
                print(f"  {k:24} avg={v['avg_score']} n={v['sample_count']} "
                      f"source={v['source']} dataset={v['dataset']}")
        return 0

    # default: show help
    print("Usage: mrouter eval {submit|list|summary}")
    return 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="mrouter",
        description="Multiplexing heuristic model router (LM Studio + Aperture, Jev-aware). "
                     "Day-to-day surface: init (once), serve (daemon), doctor (health check).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- primary surface: the only things a human should need in normal operation
    pi = sub.add_parser("init", help="one-time setup: configure endpoints, install launchd agent, wire up Hermes")
    pi.add_argument("--gateway-url", default=None, help="cloud gateway base URL, e.g. https://gw.example.com/v1")
    pi.add_argument("--gateway-key-env", default=None, help="env var name holding the gateway API key")
    pi.add_argument("--no-launchd", action="store_true", help="skip installing the launchd agent (macOS)")
    pi.add_argument("--no-hermes", action="store_true", help="skip wiring up a local Hermes Agent profile")
    pi.add_argument("--yes", action="store_true", help="non-interactive: accept defaults, don't prompt")

    pv = sub.add_parser("serve", help="start the multiplexing routing proxy (auto-starts LM Studio, "
                         "auto-benches new local models, auto-learns from traffic)")
    pv.add_argument("--host", default=None)
    pv.add_argument("--port", type=int, default=None)

    pd = sub.add_parser("doctor", help="health check: endpoints, pricing, classifier, bench freshness, "
                         "reliability floor, autolearn — the only maintenance command")
    pd.add_argument("--verbose", "-v", action="store_true", help="also print scan/suggest/stats/reliability detail")

    pf = sub.add_parser("feedback", help="record explicit feedback to improve routing (teaches the router)")
    pf.add_argument("use_case", choices=["chat", "coding", "sensitive", "thinking"])
    pf.add_argument("model", help="fleet key, e.g. aperture:deepseek/deepseek-v4-flash")
    pf.add_argument("rating", choices=["+1", "-1"])
    pf.add_argument("--note", default="")

    pe = sub.add_parser("eval", help="submit, list, or summarize custom evaluation results")
    pe_sub = pe.add_subparsers(dest="eval_cmd", required=True)
    pe_submit = pe_sub.add_parser("submit", help="submit one eval result")
    pe_submit.add_argument("--run-name", required=True, dest="run_name")
    pe_submit.add_argument("--model", required=True)
    pe_submit.add_argument("--score", type=float, default=None)
    pe_submit.add_argument("--pass-fail", choices=["pass", "fail"], default=None, dest="pass_fail")
    pe_submit.add_argument("--rubric-score", type=float, default=None, dest="rubric_score")
    pe_submit.add_argument("--llm-judge-score", type=float, default=None, dest="llm_judge_score")
    pe_submit.add_argument("--samples", type=int, default=0)
    pe_submit.add_argument("--dataset", default=None)
    pe_submit.add_argument("--version", default=None)
    pe_submit.add_argument("--source", default=None)
    pe_submit.add_argument("--tags", default=None)
    pe_submit.add_argument("--use-case", choices=["chat", "coding", "sensitive", "thinking"], default=None, dest="use_case")

    pe_list = pe_sub.add_parser("list", help="list eval results")
    pe_list.add_argument("--run-name", default=None, dest="run_name")
    pe_list.add_argument("--model", default=None)
    pe_list.add_argument("--dataset", default=None)
    pe_list.add_argument("--source", default=None)
    pe_list.add_argument("--tags", default=None)
    pe_list.add_argument("--version", default=None)
    pe_list.add_argument("--use-case", choices=["chat", "coding", "sensitive", "thinking"], default=None, dest="use_case")
    pe_list.add_argument("--window", default=None)

    pe_summary = pe_sub.add_parser("summary", help="summarize eval results")
    pe_summary.add_argument("--window", default=None)

    sub.add_parser("tray", help="launch the macOS menu-bar tray app (live perf/spend/models; "
                   "requires: pip install mrouter[tray])")

    # --- debug/inspection surface: not needed day-to-day; doctor --verbose covers most of this
    ps = sub.add_parser("scan", help="[debug] list discovered local + Aperture models and pricing")
    ps.add_argument("--local-only", action="store_true")
    pg = sub.add_parser("suggest", help="[debug] recommend best model per use case across the fleet")
    pg.add_argument("--use-case", choices=["chat", "coding", "sensitive", "thinking"])
    pg.add_argument("--top", type=int, default=3)
    pg.add_argument("--local-only", action="store_true")
    pg.add_argument("--json", action="store_true")
    sub.add_parser("bench", help="[debug] force a synchronous micro-benchmark now (serve does this automatically)")
    pc = sub.add_parser("classify", help="[debug] classify a prompt (Baseten/Jev if key present, else local)")
    pc.add_argument("text", nargs="*")
    sub.add_parser("learn", help="[debug] force an immediate learned.json recompute (serve does this automatically)")
    sub.add_parser("stats", help="[debug] show routing log totals and spend")
    sub.add_parser("reliability", help="[debug] show per-model reliability scores from route log")
    sub.add_parser("classifier-reliability",
                    help="[debug] show per-provider classifier reliability + circuit breaker status")

    args = p.parse_args(argv)
    fn = {"init": cmd_init, "serve": cmd_serve, "doctor": cmd_doctor, "feedback": cmd_feedback,
          "scan": cmd_scan, "suggest": cmd_suggest, "bench": cmd_bench,
          "classify": cmd_classify, "learn": cmd_learn,
          "stats": cmd_stats, "reliability": cmd_reliability,
          "classifier-reliability": cmd_classifier_reliability, "tray": cmd_tray,
          "eval": cmd_eval}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())