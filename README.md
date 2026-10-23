# mrouter — an intelligent model router for local + cloud LLM fleets

mrouter multiplexes requests across **local models (LM Studio, $0)** and **any
OpenAI-compatible cloud gateway with per-model pricing** — scoring capability,
speed, context window, memory fit, and **real per-request cost** — then routes
each request to the best model for the job. It classifies prompts into use
cases (`chat`, `coding`, `sensitive`, `thinking`), keeps sensitive data on
your machine by construction, logs actual spend, and learns from outcomes so
selection improves over time.

Built for Apple Silicon Macs running [LM Studio](https://lmstudio.ai), wired to
work with any OpenAI-compatible client — including
[Hermes Agent](https://hermes-agent.nousresearch.com), whose default model can
simply be `auto`.

**Zero dependencies. Python 3.9+ stdlib only.** No dotenv, no framework, no
build step — clone and run.

## Why

Storefront model pickers make you choose; that's the wrong default. Most
requests don't need a frontier model, sensitive requests shouldn't leave
your machine at all, and the right answer changes as you add models or as
pricing moves. mrouter's stance:

- **Classify the task, then pick the model.** You shouldn't have to think
  about which model you're talking to.
- **Cost is a first-class input.** Local is free; cloud models are scored on
  capability-per-dollar, not just capability. Every request's actual cost is
  computed from usage × pricing and logged.
- **Privacy is a filter, not a preference.** The `sensitive` use case removes
  cloud models from the pool *before* scoring — a sensitive prompt cannot be
  routed off-device because no cloud model is ever a candidate.
- **Learn from reality.** Success rates, latency, and your explicit feedback
  nudge future routing decisions within sane bounds.

## Quick start

```bash
git clone https://github.com/radioboxtv/mrouter && cd mrouter
python3 -m mrouter init     # one-time: detect LM Studio, configure the cloud gateway,
                            # install a launchd agent, wire up Hermes Agent
```

That's it — `init` writes `config.json`, installs `mrouter serve` to run at
login (macOS launchd; adapt `scripts/serve.sh` for systemd/pm2 elsewhere), and
can point a local Hermes Agent at the proxy automatically. Day-to-day, the
only command you should need is:

```bash
python3 -m mrouter doctor   # health check — endpoints, pricing, classifier,
                            # bench freshness, reliability, autolearn status
```

Everything else is automatic: `serve` auto-starts LM Studio if it's down,
auto-benchmarks any local model missing fresh speed data in the background,
and recomputes learned routing adjustments after every request and every
piece of feedback — there is no `mrouter learn` or `mrouter bench` step to
remember. `scan`, `suggest`, `stats`, `reliability`, `classify`, `bench`, and
`learn` still exist as debug/inspection commands (and `doctor --verbose`
prints all of them in one pass) but none of them are required maintenance.

Point any OpenAI-compatible client at `http://127.0.0.1:8730/v1` and send the
use case as the model name — or `auto` to have mrouter classify each prompt:

```bash
curl http://127.0.0.1:8730/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Refactor this regex…"}],"max_tokens":256}'
# -> routed to the best coding model; response headers tell you which:
#    X-Routed-Model, X-Routed-Provider, X-Use-Case, X-Classifier,
#    X-Routed-Attempts (present when a retry/fallback happened)
```

On a retryable upstream failure (429/5xx/connection error), mrouter retries
the same model with backoff, then cascades to the next-ranked model — never
crossing the `sensitive` local-only boundary — instead of failing the
request outright. See `resilience` in [CONFIG.md](CONFIG.md) to tune it.

Route names:

| model field | behavior |
|---|---|
| `auto` | classify the prompt, then route (blends classifier probabilities) |
| `chat` / `coding` / `thinking` | best model for that use case across the fleet |
| `sensitive` | best **local** model; cloud excluded before scoring |
| `router/<name>` | same as `<name>` (useful if a client pins names) |
| *anything else* | pass-through to the right upstream (local ids → LM Studio, others → cloud), still cost-logged |

## CLI

Day-to-day, this is the whole surface:

```bash
python3 -m mrouter init      # one-time: config.json, launchd agent, Hermes wiring
python3 -m mrouter serve     # the proxy — auto-starts LM Studio, auto-benches, auto-learns
python3 -m mrouter doctor    # health check; the only "maintenance" command
python3 -m mrouter tray      # macOS menu-bar app: live perf/spend/models (needs: pip install rumps)
```

`feedback` is the one routine human action — teach the router a routing was
good or bad (adjustments apply immediately, no separate recompute step):

```bash
python3 -m mrouter feedback <use_case> <model_key> +1|-1   # teach it
```

Everything below is debug/inspection tooling, not required reading. `doctor
--verbose` prints all of it in one pass:

```bash
python3 -m mrouter scan       # inventory: local models + catalog + pricing
python3 -m mrouter suggest    # ranked recommendations per use case
python3 -m mrouter bench      # force a synchronous re-benchmark now (serve does this automatically)
python3 -m mrouter classify "your prompt text"   # see the classifier's decision
python3 -m mrouter learn      # force an immediate learned.json recompute (serve does this automatically)
python3 -m mrouter stats      # request counts + actual spend
python3 -m mrouter reliability  # per-model reliability table
```

Inspect a live router: `curl :8730/v1/routes` (best + top 3 per use case),
`curl :8730/v1/stats` (spend + learned state), `curl :8730/v1/models` (fleet
with per-model $/M pricing), `curl :8730/v1/metrics?window=1h` (live perf —
p90 TTFT, p90 tokens/min, request volume, error rate, actual+estimated spend,
per-model reliability — over `1h`, `24h`, or raw seconds). Or open
`http://127.0.0.1:8730/ui` for the browser dashboard, or run `mrouter tray`
for the macOS menu-bar version (see [Metrics](#metrics-tray-app--dashboard)
below).

## How routing works

1. **Fleet discovery.** Local models from `lms ls --json` (params, quant,
   context, vision, tool-use training) enriched from each model's HF-style
   `config.json` (modality tokens, hybrid-attention layout). Cloud models from
   your gateway's `GET /v1/models` — including the per-token `pricing` block
   when the gateway publishes one (Aperture-style gateways do; see below).
2. **Classification** (`auto`): the configured **decision model** — Baseten's
   *mammoth-1* / *elephant-1* (System One API: typed, calibrated
   probabilities) or TypeSafe's *Jev* (same API shape) — if its key is set;
   else a deterministic local keyword classifier at $0. Classifier
   probabilities are **blended** across use-case rankings, and any prompt
   whose `sensitive` probability crosses the floor is forced onto the
   local-only route. Provider chain via `decision.provider`:
   `auto` (Baseten → Jev → local) | `baseten` | `jev` | `local`. Every
   Baseten/Jev attempt is logged and scored the same way model reliability
   is (time-decayed success rate, shrunk to a neutral prior) — a provider
   that's been failing consistently trips a **circuit breaker**
   (`decision.reliability_floor`/`reliability_min_samples`) and gets skipped
   straight to the next one in the chain instead of paying its network
   timeout on every request. `mrouter doctor` reports breaker status;
   `mrouter classifier-reliability` shows the per-provider table.
3. **Scoring.** Each model gets feature scores — capability (curated family
   knowledge base), speed (measured for local, family estimate for cloud),
   context tier, multimodality, tool-use, RAM fit, `cost_efficiency`
   (log-scaled cost of a reference request; free local = 100), and `value`
   (capability per dollar). Each use case blends them with weights from
   `config.json`; learned adjustments (±8) are added on top.
4. **Routing.** Forward to the winner, stream the response back, and log
   use case, model, latency, token usage, and **actual cost** to
   `route_log.jsonl`.
5. **Learning.** Every logged request and every `feedback` vote automatically
   recomputes bounded per-(use case, model) adjustments in `learned.json` —
   `serve` does this in-process (throttled for ordinary traffic, immediate
   for feedback), so there is no separate step to run. ≥5 samples before
   reliability moves a score; feedback counts more than volume; everything is
   plain JSON you can inspect or reset. `mrouter learn` still exists as a
   manual, forced recompute for scripts or after hand-editing the log.

## Configuration

Everything tunable lives in `config.json` (not code). Copy
`config.example.json`, then see **[CONFIG.md](CONFIG.md)** for the full
reference — endpoints, weights, the family knowledge base, hardware, decision
models, and learning bounds. Set `MROUTER_CONFIG=/path/to/config.json` to keep
configs out of the repo entirely. Secrets never live in the repo: keys come
from environment variables (`APERTURE_API_KEY`, `TYPESAFE_API_KEY`) or, for
setec-cli users, a secrets service.

## Cost inputs

If your gateway's `/v1/models` returns per-model `pricing` (`input`,
`input_cache_read`, `output` — USD per token) mrouter scores cost and
capability-per-dollar and computes each request's **actual** cost from
response `usage` (cache-read tokens billed at the cache rate). Without
pricing data, cloud models still route — they just don't get the cost
features, and spend logging reports `null` for them.

## Sensitive data

The `sensitive` route is a **hard local-only filter**: cloud models are
removed from the candidate pool before any scoring, and the proxy only ever
forwards sensitive traffic to `localhost`. In `auto`, the same filter arms
when the sensitive-classifier probability crosses
`routing.sensitive_probability_floor`. Your data can't be routed off-device
by a misconfiguration of weights.

## Use with Hermes Agent

mrouter was built alongside [Hermes Agent](https://hermes-agent.nousresearch.com)
(and any OpenAI client works — Codex, Cline, Aider, your own scripts).
`mrouter init` offers to do this for you automatically; the equivalent
manual commands are:

```bash
hermes config set model.provider custom
hermes config set model.default auto
hermes config set model.base_url http://127.0.0.1:8730/v1
hermes fallback add   # keep a cloud model up in case mrouter is down
```

Every new Hermes session now classifies its own prompts and picks models
across your fleet. `mrouter doctor` is also the right place to check that
this integration is still healthy — it reports LM Studio/gateway/classifier
status that Hermes depends on.

To run mrouter at login (macOS): `mrouter init` writes and loads a launchd
agent (`~/Library/LaunchAgents/com.mrouter.serve.plist`) pointing at
`scripts/serve.sh`, with `KeepAlive` so it restarts on crash. Elsewhere,
adapt `scripts/serve.sh` for systemd/pm2/your init system of choice.

## What it does not do

- No GPU-level scheduling — LM Studio owns local inference; mrouter only chooses.
- No fine-tuning or model serving.
- The family capability scores in `config.json` are curated heuristics —
  opinionated but transparent, editable in one place, and improved by
  learning over time.

## Live rankings prior (optional, Artificial Analysis)

Public benchmarks can replace or augment the curated `family_kb` for **cloud**
models. Set `rankings.enabled: true` and a mode: `curated` (default),
`live` (AA intelligence/coding/agentic indices win), or `blend` (weighted
average, `blend_weight` = AA share). Local models always keep curated +
bench-measured scores — public benchmarks measure hosted full-precision
weights, not local quants. Cache-first (24h `aa_cache.json`), needs a free
`AA_API_KEY` only to refresh; key-less or offline it degrades silently to
curated. See [CONFIG.md](CONFIG.md).

## Metrics (tray app & dashboard)

Every routed request writes one row to a lightweight on-device SQLite store
(`metrics.db`, WAL mode) separate from `route_log.jsonl` (which stays the
audit/learning log). It powers two surfaces over a configurable time window
(`1h` / `24h` by default, editable in `config.json`):

- **`mrouter tray`** — a macOS menu-bar app: spend + p90 TTFT in the title,
  and a dropdown with the 1h/24h toggle, KPIs (spend, requests, error rate,
  p90 TTFT, p90 tokens/min), and an active/recent model list. Each model
  shows its use case(s) — truncated to the first couple plus "+N more" for
  models that serve several. Needs `python3 -m pip install rumps` (the one
  dependency beyond the stdlib anywhere in mrouter, isolated to this feature);
  without it, `mrouter tray` prints an install hint and exits.
- **Dashboard tab** in the browser UI at `http://127.0.0.1:8730/ui` — the
  same metrics with more room: a request-volume/error sparkline, a
  reliability table, and a fuller per-model breakdown. No extra dependency.

`GET /v1/metrics?window=1h|24h|<seconds>` and the cheap-poll
`GET /v1/metrics/live` back both surfaces. Local models' speed score in
routing also prefers real p90 tokens/sec from this store over the synthetic
micro-bench once enough live samples exist (`metrics.live_speed_min_samples`,
default 5) — live traffic, not just a 24-token bench prompt, feeds routing
decisions. See [CONFIG.md](CONFIG.md#metrics-tray-app--dashboard) for every
knob (retention, poll interval, window choices).

## Tests

```bash
python3 -m unittest discover -s tests -v
```

172 hermetic tests: scoring, cost math, sensitive-invariant routing, classifier
shapes (incl. mocked decision models), full proxy integration against in-process fake
upstreams, config loading, learning bounds, live-rankings behavior
(slug matching, record mapping, modes, cache/fetch lifecycle), reliability
scoring, the automatic-maintenance layer (autolearn throttling/forcing,
autobench staleness detection + dedup, `doctor` report checks), and the
on-device metrics store (windowed percentiles, per-model breakdown, pruning,
live-speed-score feed into routing, tray-app menu rendering).

## License

MIT — see [LICENSE](LICENSE).
