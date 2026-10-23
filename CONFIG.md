# mrouter configuration reference

mrouter reads `config.json` (or `$MROUTER_CONFIG`). `mrouter init` writes this
file for you (prompting for the gateway endpoint/key) — manual setup below is
for anyone who wants to skip the wizard. Nothing secret belongs in this file
— keys come from environment variables.

## Setup, day-to-day, and maintenance

```bash
mrouter init      # one-time: config.json, launchd agent (macOS), Hermes Agent wiring
mrouter serve     # the only process that runs continuously — auto-starts LM Studio,
                  # auto-benchmarks new local models in the background, auto-recomputes
                  # learned.json from traffic and feedback
mrouter doctor    # the only "maintenance" command — run it whenever something feels off
```

`feedback` is the one routine human action (`mrouter feedback <use_case>
<model_key> +1|-1`); everything else (`scan`, `suggest`, `bench`, `classify`,
`learn`, `stats`, `reliability`) is debug/inspection tooling that `doctor
--verbose` already surfaces — none of it is required maintenance.

## Section map

| Section | What it controls |
|---|---|
| `lm_studio` | Local runtime: CLI binary path, API base, model storage root |
| `providers.aperture` | Cloud gateway: endpoint, auth, catalog cache TTL |
| `proxy` | Listening host/port for the routing proxy |
| `hardware` | Total RAM (GB) — powers memory-fit scoring |
| `routing` | Hard eligibility rules (local-only, cloud allowlist, privacy floor) |
| `resilience` | Retry-with-backoff, cascading fallback to runner-up models, and the in-process circuit breaker |
| `jev` | Optional Jev (TypeSafe AI) classifier endpoint/auth |
| `learning` | Feedback & outcome logging, adjustment bounds (recomputed automatically by `serve`) |
| `rankings` | Optional live rankings prior (Artificial Analysis): mode, key, cache TTL |
| `use_cases` | Per-use-case weight blends — the main tuning surface |
| `family_kb` | Curated capability knowledge base, matched by model name |
| `default_family_scores` | Scores for models that match no family_kb entry |
| `bench` | Local micro-benchmark prompt + result freshness (`max_age_hours` also governs when `serve` auto-re-benchmarks a model in the background) |

## Endpoints & secrets

```json
"lm_studio": {
  "lms_bin": "~/.lmstudio/bin/lms",
  "api_base": "http://localhost:1234/v1",
  "models_root": "~/.lmstudio/models"
},
"providers": {
  "aperture": {
    "enabled": true,
    "api_base": "https://your-gateway.example.com/v1",
    "api_key_env": "APERTURE_API_KEY",
    "api_key": "",
    "catalog_cache_seconds": 300
  }
}
```

- **api_key_env** wins over **api_key**; leave `api_key` empty and export
  `APERTURE_API_KEY` (or whatever you name it). `"no-key-required"` disables auth.
- Any OpenAI-compatible gateway works. If its `GET /v1/models` publishes a
  per-model `pricing` object (`input`, `input_cache_read`, `output` — USD per
  token), mrouter uses it for cost scoring and exact spend attribution; if not,
  cloud routing still works without cost features.
- The gateway's `/v1/models` response is cached for `catalog_cache_seconds`.

## Use cases & weights

Four routes ship: `chat`, `coding`, `sensitive`, `thinking`. Each blends
feature scores with weights (should sum to ≈1). Feature keys:

| Key | Meaning | Range |
|---|---|---|
| `speed` | Measured tok/s (local) or family estimate (cloud) | 0–100 |
| `capability_chat` / `_coding` / `_thinking` | Family-knowledge-base capability | 0–100 |
| `capability_avg` | Mean of the three capabilities | 0–100 |
| `context` | Context-window tiering | 25–100 |
| `multimodal` | Vision/audio support | 0/100 |
| `tool_use` | Tool-call training (local) / endpoint capability | 30/100 |
| `mem_fit` | Weights fit comfortably in RAM (local only) | 0–100 |
| `params_quality` | Params vs. local fleet; capability proxy for cloud | 0–100 |
| `cost_efficiency` | −log(est. cost of a 4k-in/2k-cached/1k-out reference request); free = 100 | 0–100 |
| `value` | Capability per dollar, normalized to local baseline | 0–100 |
| `privacy` | 100 = local-only | 0/100 |

Example — heavier cost sensitivity for chat:

```json
"chat": {
  "weights": {
    "speed": 0.20, "capability_chat": 0.25, "cost_efficiency": 0.35,
    "multimodal": 0.05, "context": 0.05, "tool_use": 0.05, "value": 0.05
  }
}
```

Add your own use case: new key under `use_cases` with a `label`,
`description`, and `weights` — it appears in the CLI and as a proxy route
automatically.

## Routing rules (hard filters)

```json
"routing": {
  "sensitive_local_only": true,
  "allow_cloud": { "chat": true, "coding": true, "thinking": true, "sensitive": false },
  "sensitive_probability_floor": 0.25
}
```

- `sensitive_local_only` removes every non-local model from the `sensitive`
  pool before scoring. `allow_cloud.*` does the same per use case.
- `sensitive_probability_floor`: in `auto` mode, any prompt whose sensitive
  probability reaches this value is forced onto the local-only route —
  independent of which label "won".

## Resilience: retry, fallback, circuit breaker

```json
"resilience": {
  "retry": { "max_attempts": 2, "backoff_s": 0.5, "on_status": [429, 500, 502, 503, 504] },
  "fallback": { "enabled": true, "max_models": 3 },
  "circuit_breaker": { "failure_threshold": 5, "cooldown_s": 60 },
  "upstream_timeout_s": 600
}
```

Previously the proxy made exactly one pick and one upstream attempt per
request: an upstream 5xx/429 or a connection error was surfaced straight to
the client even when healthy runner-up models were ranked right behind the
one that failed. This section closes that gap. All keys are optional and
default to the values shown above — an existing `config.json` written before
this feature shipped is resilient with zero edits.

- **`retry`**: on a status in `on_status`, the **same** model is retried up
  to `max_attempts` times, sleeping `backoff_s * attempt` between tries
  (so 0.5s, then 1.0s, ... at the default). A non-retryable status (e.g. a
  client's bad-request 400) is never retried. A connection/timeout error is
  always treated as retryable.
- **`fallback`**: once a candidate model exhausts its retries, `enabled`
  controls whether the proxy cascades to the **next-ranked eligible**
  candidate instead of surfacing the failure, up to `max_models` total model
  attempts for the request. Eligibility is computed exactly like a normal
  pick — in particular the **`sensitive` hard local-only filter still
  applies to every fallback candidate**, so a sensitive request can exhaust
  every local model and return a clean error, but will never fall back to a
  cloud model. `auto` mode that lands on `sensitive` via the probability
  floor gets the same guarantee.
- **`circuit_breaker`**: an in-process, per-model consecutive-failure
  counter (resets on process restart; distinct from the persisted, log-based
  `routing.reliability_floor`). A model with `failure_threshold` or more
  *consecutive* failures is deprioritized — moved behind every currently
  healthy candidate in the ranking, not hard-excluded — until `cooldown_s`
  elapses, after which it gets one "half-open" trial request; a success
  resets its failure count immediately. `failure_threshold: 0` disables the
  breaker.
- **`upstream_timeout_s`**: replaces the old hardcoded `urlopen(..., timeout=600)`.
- Every attempt (not just the final outcome) is logged as its own
  `route_log.jsonl` entry (`fallback_attempt`/`fallback_total` fields) so
  `mrouter reliability`/`learn` see fallback events, not just the request
  that ultimately succeeded or failed. The full attempt chain for a request
  is also returned to the client in the `X-Routed-Attempts` response header
  (JSON array) so callers can see what happened.

## Decision model (optional, Baseten or Jev)

```json
"decision": {
  "provider": "auto",
  "reliability_floor": 0.5,
  "reliability_min_samples": 5,
  "note": "Which categorizer 'auto' uses: auto (baseten key -> jev key -> local), baseten, jev, local."
},
"baseten": {
  "enabled": true,
  "endpoint": "https://inference.baseten.co/v1/systemone",
  "model": "decision-models/baseten-mammoth-27b-latest",
  "api_key_env": "BASETEN_API_KEY",
  "timeout_seconds": 5
}
```

Decision models return typed, calibrated probabilities instead of text. mrouter
ships two providers and a deterministic local fallback (offline, free):

- `decision.provider`: `auto` (default — Baseten key → Jev key → local),
  `baseten`, `jev`, or `local` (pinned).
- Baseten defaults to **mammoth-1** (27B; #1 of 208 on Jev Decision Index 0.4).
  Set `baseten.model` to `decision-models/baseten-elephant-9b-latest` for the
  cheaper 9B. Auth: `BASETEN_API_KEY` env (or `api_key_env` of your choosing).
  No key/config → auto falls back to Jev/local.

Without a key, `auto` uses the built-in deterministic classifier (offline,
free). With a key, prompts are classified by the decision model and the
calibrated probabilities are blended across use-case rankings;
`classify_task` never raises — provider failure falls back to local.

### Classifier circuit breaker

Every Baseten/Jev attempt (success, exception, or an unparseable response —
all three count) is logged under `type: "classifier_attempt"` in the same
`route_log.jsonl` used for routing, and scored with the identical
time-decay/shrinkage math as model reliability
(`mrouter.reliability.compute_classifier_reliability`), just over a
`provider` key space (`baseten` | `jev`) instead of model keys.

- **`decision.reliability_floor`** / **`decision.reliability_min_samples`**:
  same semantics as `routing.reliability_floor`/`reliability_min_samples`
  below, but for the decision-provider layer. A provider whose recent
  classifier-attempt success rate falls below the floor over at least
  `min_samples` attempts is skipped — straight to the next provider in the
  chain (or local) — **without even attempting the network call**, saving
  `timeout_seconds` on every request instead of waiting out a dead endpoint
  each time. Default 0.5 / 5, matching model reliability; `0.0` disables the
  breaker entirely.
- Unobserved or thin-sample providers are never skipped — a provider must
  prove itself unreliable first, same as the model-level floor.
- `mrouter classifier-reliability` CLI prints the current per-provider table;
  `mrouter doctor` surfaces a WARN (not a FAIL — the fallback chain still
  routes traffic) when a breaker is currently open.

## Reliability scoring (routed models)

```json
"routing": {
  "reliability_floor": 0.5,
  "reliability_min_samples": 5
}
```

Every routed request is logged with its status code. `mrouter/reliability.py`
turns that log into a per-model score (0–100) using exponential time-decay
(48 h half-life, so fresh failures matter more than stale ones) shrunk toward
an 85-point neutral prior (3 pseudo-samples — new models aren't penalised until
their track record is real).

- **`reliability_floor`**: models whose observed success-rate falls below this
  threshold AND have at least `reliability_min_samples` are excluded from the
  eligible pool before scoring. Default 0.5 (50 %); 0.0 disables floor-exclusion.
- **`reliability_min_samples`**: minimum logged requests before floor-exclusion
  can fire. Default 5 — avoids excluding a model on one bad request.
- Unobserved models (no log entries) score the neutral prior (85) — trusted
  until proven flaky.
- `mrouter reliability` CLI prints the current per-model table with flags.

Reliability feeds `routing.reliability` weight in each use case (default 8 %
of the composite score). Tune the weight like any other component.



```json
"learning": {
  "log_file": "route_log.jsonl",
  "learned_file": "learned.json",
  "max_adjustment": 8.0,
  "min_samples_for_auto_adj": 5
}
```

- `route_log.jsonl` — append-only; one line per routed request.
- `learned.json` — regenerated **automatically** whenever `serve` logs a
  request or records feedback (throttled to once per
  `AUTOLEARN_MIN_INTERVAL_S` [5s] for ordinary traffic; feedback forces an
  immediate, unthrottled recompute). Adjustments capped at ±`max_adjustment`
  score points. Feedback (±1) moves a model up to ±1.5 per vote immediately;
  automatic reliability adjustments need `min_samples_for_auto_adj` requests
  first. `mrouter learn` is a manual override for forcing a recompute from a
  script or after hand-editing the log — normal operation never needs it.
- Reset anytime: delete both files.

## Metrics (tray app & dashboard)

```json
"metrics": {
  "enabled": true,
  "db_file": "metrics.db",
  "default_window": "1h",
  "windows_available": ["1h", "24h"],
  "retention_days": 7,
  "tray_poll_seconds": 10,
  "tray_live_window_seconds": 300,
  "live_speed_min_samples": 5
}
```

Every routed request also writes one row to a local SQLite store
(`metrics.db`, WAL mode, gitignored like other machine-local state) — separate
from `route_log.jsonl`, which stays the audit/learning log unchanged.
`mrouter/metrics_store.py` answers windowed aggregate queries (percentiles,
per-model breakdown, bucketed volume series) without rescanning the whole
JSONL log on every poll.

- **`enabled`**: `false` disables metrics writes entirely (no `metrics.db`
  created, `/v1/metrics*` return empty windows, live-speed scoring falls back
  to bench/family estimates).
- **`default_window`** / **`windows_available`**: the time-window toggle shown
  in the tray dropdown and dashboard tab (`1h` / `24h` by default). Pass
  `?window=1h|24h|<seconds>` to `GET /v1/metrics` to query any window.
- **`retention_days`**: rows older than this are pruned on a throttled
  schedule (checked on every write, actually prunes at most once per 5
  minutes) — keeps `metrics.db` small indefinitely.
- **`tray_poll_seconds`**: how often `mrouter tray` polls `/v1/metrics/live`
  for the menu-bar title and dropdown.
- **`tray_live_window_seconds`**: the fixed window `/v1/metrics/live` uses for
  the tray's cheap per-tick snapshot (doesn't follow the 1h/24h toggle).
- **`live_speed_min_samples`**: a model's `speed` AND `ttft` scores prefer
  real p90 tokens/sec / p90 TTFT from live traffic over guessed numbers once
  at least this many requests have been logged for it in the default window
  — **for local and cloud models alike**. Below that: local models fall back
  to `bench_results.json` (from `mrouter bench`/autobench) then a
  params-based guess; cloud models fall back straight to the curated family
  `speed_est` (throughput) or a flat neutral prior (TTFT — there's no curated
  per-family latency guess to fall back to, by design; see below).
  `heuristics.score_fleet()`'s ranked rows report `speed_source` and
  `ttft_source` (`live` / `bench` / `estimated`) per model so routing
  decisions and the dashboard can show which numbers are real vs. guessed.
  `mrouter doctor` surfaces a `perf_sources` check summarizing the mix
  fleet-wide.

`GET /v1/metrics?window=1h|24h` returns requests, errors, error_rate,
`ttft_p50_s`/`ttft_p90_s`, `tokens_per_min_p90`, `spend_actual_usd` (computed
from real usage × pricing) vs. `spend_estimated_usd` (fallback for unpriced
models), a bucketed `series` for the volume sparkline, and a `models` map with
per-model request/error counts, use-case breakdown, p90 TTFT/tok-s, and
`live`/`recent` status. `GET /v1/metrics/live` returns the same shape over the
small fixed `tray_live_window_seconds` window, for cheap frequent polling.

### Real-world performance feeds model selection, not just the dashboard

Two scoring components use this data directly, with `ttft` new alongside
`speed` (each use case's weights in `config.example.json` split what used to
be a single `speed` weight roughly 2:1 into `speed` + `ttft`, so total weight
per use case is unchanged):

- **`speed`** — real-world throughput (tokens/sec). `live` > `bench`
  (local-only) > `estimated` (curated `speed_est` for cloud, params-based
  guess for local).
- **`ttft`** — real-world time-to-first-token (latency; lower is better,
  normalized against the fastest fleet member actually observed). `live` >
  `bench` (local-only, from the micro-bench's measured `ttft_s`) >
  `estimated` — a flat neutral prior (50/100), **never a fabricated number**:
  unlike throughput, there is no curated per-family TTFT guess in
  `family_kb`, because a plausible-looking invented latency number would be
  worse than admitting "unknown" for a model that's never actually been
  routed.

Both components normalize `live` measurements across the **whole fleet**
(local + cloud together) once there's real data for a model — a genuinely
fast cloud model and a genuinely fast local model become directly comparable
on fact, not on a local-measured-vs-cloud-guessed mismatch.

**Existing `config.json` from before this feature:** if your use-case
weights don't have a `ttft` entry, the `ttft` component simply contributes
nothing to scoring (missing weight keys are skipped, not treated as an
error) — routing still works, it just isn't latency-aware until you add one.
Copy the `ttft` weight from the matching use case in the current
`config.example.json` (each use case split its old `speed` weight ~2:1 into
`speed` + `ttft`, so totals stay ~1.0), or regenerate a fresh config via
`mrouter init` if you haven't customized yours.

`mrouter tray` launches a macOS menu-bar app showing these metrics (menu-bar
title: spend + p90 TTFT; dropdown: 1h/24h toggle, KPIs, and an active/recent
model list with each model's use case(s) — truncated to the first couple of
names plus "+N more" when a model serves several). This is the one place
mrouter takes on a dependency beyond the stdlib: install
[`rumps`](https://github.com/jaredks/rumps) (`python3 -m pip install rumps`)
first; without it, `mrouter tray` prints an install hint and exits — the core
proxy/CLI stay zero-dependency either way. The dashboard tab in the browser UI
(`mrouter/ui.html`, served at `/ui`) shows the same data in more detail and
needs no extra dependency.

## Family knowledge base

`family_kb` maps name patterns to capability scores; the first matching entry
wins (match against model id + architecture). Pins per family: capabilities
(0–10 scale, ×10 internally), `speed_est` (0–100 tok/s-ish blend score for
cloud), `vision`, and a human-readable `notes` field used in `suggest` output.
Unmatched models fall back to `default_family_scores`. Curated, opinionated,
fully editable — tune to your taste.

```json
"deepseek-v4-flash": {
  "match": ["deepseek-v4-flash", "deepseek-v4.1-flash"],
  "capability_coding": 8.3, "capability_thinking": 8.0, "capability_chat": 8.2,
  "speed_est": 92, "vision": false,
  "notes": "Very cheap, fast, capable. Strong value pick."
}
```

## Live rankings (optional, Artificial Analysis)

```json
"rankings": {
  "enabled": false,
  "mode": "curated",
  "blend_weight": 0.5,
  "api_key_env": "AA_API_KEY",
  "endpoint": "https://artificialanalysis.ai/api/v2/language/models/free",
  "cache_seconds": 86400
}
```

Live per-model intelligence/coding/agentic indices as a capability prior:

- **`mode`**: `curated` (default — hand-written `family_kb` only) | `live`
  (AA indices replace curated capabilities for **cloud** models) | `blend`
  (weighted average; `blend_weight` is the AA share, default 0.5).
- Cloud models only: local models always keep curated + bench-measured
  scores — public benchmarks measure hosted full-precision weights, not your
  local quants.
- Cache-first: a fresh `aa_cache.json` (TTL `cache_seconds`, 24h default) is
  served with no key and no network. Missing/stale cache triggers one fetch
  when `AA_API_KEY` is set; on failure the stale cache is served. No key and
  no cache → routing silently degrades to `curated`.
- Key resolution: `api_key_env` first (via `AA_API_KEY` by default), then a
  static `api_key`. Free-tier AA keys work — the `/free` endpoint includes
  the headline indices and median performance.
- The curated KB remains the vision/notes source in every mode; learned
  adjustments (±8) apply on top of whatever capability prior was used.

## Environment variables

| Var | Purpose |
|---|---|
| `MROUTER_CONFIG` | Absolute path to an alternate config file |
| `APERTURE_API_KEY` | Cloud gateway auth (name set by `providers.aperture.api_key_env`) |
| `BASETEN_API_KEY` | Decision-model key (name set by `baseten.api_key_env`) |
| `TYPESAFE_API_KEY` | Jev classifier key (name set by `jev.api_key_env`) |
| `AA_API_KEY` | Artificial Analysis key (name set by `rankings.api_key_env`) |

Machine-local files (`config.json`, `aperture_cache.json`, `aa_cache.json`,
`bench_results.json`, `learned.json`, `route_log.jsonl`) are gitignored by default.
