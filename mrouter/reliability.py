"""Reliability scoring: time-decayed dependability from observed outcomes.

Two independent consumers share this module:

1. **Model reliability** — a model's dependability from route-log request
   outcomes (status < 400 = success). Feeds the `reliability` component into
   the scoring engine and the below-floor swap-out in eligibility.
2. **Classifier reliability** — a decision-model provider's (baseten/jev)
   dependability from classifier-attempt outcomes logged in categorize.py.
   Feeds a circuit breaker in `classify_task`: a provider with a proven-bad
   track record is skipped before paying its network timeout, instead of
   being retried on every single request and only failing after the full
   `timeout_seconds` wait.

Why: capability/cost scores pick a winner, but a model that intermittently
rate-limits or 5xxs (observed as Hermes fallbacks) should lose to a slightly
less capable but dependable one — and a chronically failing model should be
swapped out entirely. The same logic applies one layer up: a classifier
provider that's been timing out for the last hour shouldn't cost every
request a multi-second wait before falling back.

Design constants:
- PRIOR (85): score for an unobserved entity — trust until proven flaky.
- PSEUDO (3 pseudo-samples): shrinkage strength. 1 failure in 1 request
  lands near the prior; 10 failures in 10 crater it.
- HALFLIFE_H (48h): each log entry's weight halves every two days, so a
  bad morning outweighs a good week only while it is fresh.
"""

import time

PRIOR = 85.0
PSEUDO = 3.0
HALFLIFE_H = 48.0


def _weight(age_h):
    """Exponential recency weight: half-life HALFLIFE_H hours."""
    return 0.5 ** (max(0.0, age_h) / HALFLIFE_H)


def _compute_from_outcomes(outcomes):
    """Shared math: outcomes is an iterable of (key, success: bool, age_h: float).
    Returns {key: {samples, success_rate, score}}."""
    stats = {}
    for key, ok, age_h in outcomes:
        if not key:
            continue
        w = _weight(age_h)
        st = stats.setdefault(key, {"w_ok": 0.0, "w_err": 0.0, "n": 0})
        st["n"] += 1
        if ok:
            st["w_ok"] += w
        else:
            st["w_err"] += w

    out = {}
    for key, st in stats.items():
        tot = st["w_ok"] + st["w_err"]
        rate = (st["w_ok"] / tot) if tot > 0 else None
        if rate is None:
            score = PRIOR
        else:
            score = round((PSEUDO * PRIOR + tot * rate * 100.0) / (PSEUDO + tot), 1)
        out[key] = {
            "samples": st["n"],
            "success_rate": round(rate, 3) if rate is not None else None,
            "score": score,
        }
    return out


def compute_reliability(config, entries=None):
    """Return {model_key: {samples, success_rate, score}} from route-log entries.

    `success_rate` is the recency-weighted rate; `score` is it shrunk toward
    PRIOR with PSEUDO pseudo-samples, on the 0-100 component scale.
    """
    if entries is None:
        from . import learning
        entries = learning.load_log(config)
    now = time.time()

    def _outcomes():
        for e in entries or []:
            if e.get("type") != "request":
                continue
            key = e.get("routed_model_key")
            if not key:
                continue
            status = e.get("status_code") or 0
            age_h = (now - (e.get("ts") or now)) / 3600.0
            yield key, bool(status and status < 400), age_h

    return _compute_from_outcomes(_outcomes())


def compute_classifier_reliability(config, entries=None):
    """Return {provider: {samples, success_rate, score}} from classifier-attempt
    log entries (type == 'classifier_attempt', written by categorize.py).

    Same shrinkage/recency math as model reliability, over a different log
    entry type and key space (provider name: 'baseten' | 'jev') so the two
    never collide even though they share route_log.jsonl.
    """
    if entries is None:
        from . import learning
        entries = learning.load_log(config)
    now = time.time()

    def _outcomes():
        for e in entries or []:
            if e.get("type") != "classifier_attempt":
                continue
            provider = e.get("provider")
            if not provider:
                continue
            age_h = (now - (e.get("ts") or now)) / 3600.0
            yield provider, bool(e.get("ok")), age_h

    return _compute_from_outcomes(_outcomes())


def reliability_score(stats, key):
    """Component score for a model; unobserved models get the neutral prior."""
    entry = (stats or {}).get(key)
    return float(entry.get("score", PRIOR)) if entry else PRIOR


def is_below_floor(stats, key, floor, min_samples):
    """True when a model has enough samples and still fails too often.

    Unknown models and thin samples are never excluded — flakiness must be
    proven, not guessed."""
    entry = (stats or {}).get(key)
    if not entry:
        return False
    if (entry.get("samples") or 0) < min_samples:
        return False
    rate = entry.get("success_rate")
    if rate is None:
        return False
    return rate < floor