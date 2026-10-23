"""mrouter doctor: one command, a full health report.

Replaces `scan` + a chunk of `suggest`/`stats`/`reliability` as the day-to-day
surface a human actually needs to run. Everything it checks is either already
automatic (bench, learn) or informational (pricing, classifier, reliability
floor) — doctor's job is to say "is this healthy" and "here's exactly what to
fix" in one pass, not to be yet another manual step in the loop.

Exit code: 0 if no FAIL, 1 if any FAIL (WARN never fails the run — it's used
for things that are fine in degraded/offline mode by design, e.g. no decision
model key -> local classifier).
"""

import os
import time

from . import categorize, discovery, heuristics, learning
from . import reliability as reliability_mod


class Report:
    def __init__(self):
        self.rows = []  # (status, section, message)

    def ok(self, section, msg):
        self.rows.append(("OK", section, msg))

    def warn(self, section, msg):
        self.rows.append(("WARN", section, msg))

    def fail(self, section, msg):
        self.rows.append(("FAIL", section, msg))

    @property
    def has_fail(self):
        return any(r[0] == "FAIL" for r in self.rows)

    @property
    def has_warn(self):
        return any(r[0] == "WARN" for r in self.rows)


def _check_lm_studio(config, report):
    alive = discovery.server_alive(config)
    models = discovery.discover_models(config)
    if alive and models:
        report.ok("lm_studio", f"server up, {len(models)} local model(s) discovered")
    elif alive and not models:
        report.warn("lm_studio", "server up but no local models found (nothing loaded in LM Studio?)")
    else:
        report.fail("lm_studio", "server unreachable — `serve` auto-starts it, but it's not "
                     "running right now; local + sensitive routing is unavailable until it is")
    return models


def _check_cloud(config, report):
    pcfg = (config.get("providers", {}) or {}).get("aperture", {}) or {}
    if not pcfg.get("enabled"):
        report.warn("cloud", "cloud gateway disabled in config — only local models are routable")
        return []
    from . import providers as providers_mod
    try:
        models = providers_mod.discover_aperture_models(config)
    except Exception as e:
        report.fail("cloud", f"gateway discovery raised {type(e).__name__}: {e}")
        return []
    if not models:
        report.fail("cloud", f"gateway enabled ({pcfg.get('api_base')}) but returned 0 models — "
                     "check api_base / auth")
        return []
    priced = sum(1 for m in models if m.get("cost_input_per_token") or m.get("cost_output_per_token"))
    if priced == len(models):
        report.ok("cloud", f"{len(models)} model(s) from gateway, all with pricing — "
                   "cost scoring + exact spend accounting active")
    elif priced:
        report.warn("cloud", f"{len(models)} model(s) from gateway, only {priced} carry pricing — "
                     "the rest route fine but skip cost features and log null spend")
    else:
        report.warn("cloud", f"{len(models)} model(s) from gateway, none carry pricing — "
                     "gateway's /v1/models has no `pricing` block; cost scoring disabled fleet-wide")
    return models


def _check_classifier(config, report):
    decision = (config.get("decision", {}) or {}).get("provider", "auto")
    try:
        result = categorize.classify_task("quick doctor probe: is this sensitive or just chat?", config)
    except Exception as e:
        report.fail("classifier", f"classify_task raised unexpectedly ({type(e).__name__}) — "
                     "this should never happen (it's designed to fall back to local)")
        return
    source = result.get("source", "?")
    if decision == "auto" and source == "local-heuristic":
        report.warn("classifier", "decision.provider=auto but no Baseten/Jev key resolved — "
                     "using the free local keyword classifier (works, but less accurate on edge cases)")
    else:
        report.ok("classifier", f"active provider: {source}")


def _check_bench_freshness(config, local_models, report):
    bench = heuristics.load_bench()
    max_age_h = float((config.get("bench", {}) or {}).get("max_age_hours", 7 * 24))
    stale = heuristics.stale_local_models(local_models, bench, max_age_hours=max_age_h)
    if not local_models:
        return
    if not stale:
        report.ok("bench", f"all {len(local_models)} local model(s) have fresh speed measurements")
    else:
        names = ", ".join(m["display_name"] for m in stale[:5])
        more = f" (+{len(stale) - 5} more)" if len(stale) > 5 else ""
        report.warn("bench", f"{len(stale)} local model(s) missing/stale bench data: {names}{more} — "
                     "`serve` benches these automatically in the background on next run; "
                     "falls back to params-based speed estimate until then")


def _check_reliability(config, report):
    stats = reliability_mod.compute_reliability(config)
    if not stats:
        report.ok("reliability", "no routed traffic logged yet — nothing to flag")
        return
    routing = config.get("routing", {}) or {}
    floor = float(routing.get("reliability_floor", 0.5))
    min_n = int(routing.get("reliability_min_samples", 5))
    below = [k for k in stats if reliability_mod.is_below_floor(stats, k, floor=floor, min_samples=min_n)]
    if below:
        report.fail("reliability", f"{len(below)} model(s) below the reliability floor "
                     f"(excluded from routing pools): {', '.join(below)}")
    else:
        report.ok("reliability", f"{len(stats)} model(s) observed, none below the floor "
                   f"({floor:.0%} success w/ ≥{min_n} samples)")


def _check_learning(config, report):
    log_entries = learning.load_log(config)
    learned_path = learning.learned_path(config)
    if not log_entries:
        report.ok("learning", "no requests logged yet — learned.json starts neutral")
        return
    if not os.path.exists(learned_path):
        report.fail("learning", f"{len(log_entries)} log entries but no learned.json — "
                     "autolearn should have created this; run `mrouter learn` once to force it")
        return
    age_s = time.time() - os.path.getmtime(learned_path)
    if age_s < 3600:
        report.ok("learning", f"learned.json fresh ({int(age_s)}s old, {len(log_entries)} log entries) "
                   "— autolearn is keeping up")
    else:
        report.warn("learning", f"learned.json is {age_s / 3600:.1f}h old with {len(log_entries)} log "
                     "entries — fine if traffic has been quiet; if the proxy is running and serving "
                     "requests this would indicate autolearn is stuck")


def _check_config_sanity(config, report):
    problems = []
    for uc, uc_cfg in (config.get("use_cases", {}) or {}).items():
        weights = uc_cfg.get("weights", {}) or {}
        total = sum(float(w) for w in weights.values())
        if not (0.9 <= total <= 1.1):
            problems.append(f"'{uc}' weights sum to {total:.2f} (expected ~1.0)")
    seen_patterns = {}
    for fam, info in (config.get("family_kb", {}) or {}).items():
        for pat in info.get("match", []):
            if pat in seen_patterns:
                problems.append(f"family_kb pattern '{pat}' matched by both "
                                 f"'{seen_patterns[pat]}' and '{fam}' — first-match-wins order matters")
            seen_patterns[pat] = fam
    if problems:
        for p in problems:
            report.warn("config", p)
    else:
        report.ok("config", "use-case weights sane, no duplicate family_kb patterns")


def run(config=None, verbose=False):
    config = config or discovery.load_config()
    report = Report()

    local_models = _check_lm_studio(config, report)
    _check_cloud(config, report)
    _check_classifier(config, report)
    _check_bench_freshness(config, local_models, report)
    _check_reliability(config, report)
    _check_learning(config, report)
    _check_config_sanity(config, report)

    return report


def print_report(report, verbose=False):
    icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}
    width = max((len(r[1]) for r in report.rows), default=8)
    for status, section, msg in report.rows:
        print(f"  {icon[status]} {section:<{width}}  {msg}")
    n_ok = sum(1 for r in report.rows if r[0] == "OK")
    n_warn = sum(1 for r in report.rows if r[0] == "WARN")
    n_fail = sum(1 for r in report.rows if r[0] == "FAIL")
    print(f"\n{n_ok} ok, {n_warn} warning(s), {n_fail} failure(s).")
    if report.has_fail:
        print("Fix FAILs above — routing is degraded or broken right now.")
    elif report.has_warn:
        print("No failures. WARNs are informational (often fine in a minimal/offline setup).")
    else:
        print("All green. Nothing for a human to do.")
