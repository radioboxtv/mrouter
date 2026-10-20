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
cp config.example.json config.json   # then point it at your endpoints
python3 -m mrouter scan              # inventory: local models + catalog + pricing
python3 -m mrouter serve             # proxy on http://127.0.0.1:8730/v1
```

Point any OpenAI-compatible client at `http://127.0.0.1:8730/v1` and send the
use case as the model name — or `auto` to have mrouter classify each prompt:

```bash
curl http://127.0.0.1:8730/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Refactor this regex…"}],"max_tokens":256}'
# -> routed to the best coding model; response headers tell you which:
#    X-Routed-Model, X-Routed-Provider, X-Use-Case, X-Classifier
```

Route names:

| model field | behavior |
|---|---|
| `auto` | classify the prompt, then route (blends classifier probabilities) |
| `chat` / `coding` / `thinking` | best model for that use case across the fleet |
| `sensitive` | best **local** model; cloud excluded before scoring |
| `router/<name>` | same as `<name>` (useful if a client pins names) |
| *anything else* | pass-through to the right upstream (local ids → LM Studio, others → cloud), still cost-logged |

## CLI

```
python3 -m mrouter scan       # inventory: local models + catalog + pricing
python3 -m mrouter suggest    # ranked recommendations per use case
python3 -m mrouter bench      # measure local TTFT / tok/s (feeds speed scoring)
python3 -m mrouter serve      # the proxy (daemon-friendly via serve_forever)
python3 -m mrouter classify "your prompt text"   # see the classifier's decision
python3 -m mrouter feedback <use_case> <model_key> +1|-1   # teach it
python3 -m mrouter learn      # recompute learned adjustments from the log
python3 -m mrouter stats      # request counts + actual spend
```

Inspect a live router: `curl :8730/v1/routes` (best + top 3 per use case),
`curl :8730/v1/stats` (spend + learned state), `curl :8730/v1/models` (fleet
with per-model $/M pricing).

## How routing works

1. **Fleet discovery.** Local models from `lms ls --json` (params, quant,
   context, vision, tool-use training) enriched from each model's HF-style
   `config.json` (modality tokens, hybrid-attention layout). Cloud models from
   your gateway's `GET /v1/models` — including the per-token `pricing` block
   when the gateway publishes one (Aperture-style gateways do; see below).
2. **Classification** (`auto`): a [Jev](https://typesafe.ai) call (TypeSafe AI's
   System One classifier — typed, calibrated probabilities, $0.042/Mtok input,
   output free, ~70–500 ms) if `TYPESAFE_API_KEY` is set, else a deterministic
   local keyword classifier at $0. Classifier probabilities are **blended**
   across use-case rankings, and any prompt whose `sensitive` probability
   crosses the floor is forced onto the local-only route.
3. **Scoring.** Each model gets feature scores — capability (curated family
   knowledge base), speed (measured for local, family estimate for cloud),
   context tier, multimodality, tool-use, RAM fit, `cost_efficiency`
   (log-scaled cost of a reference request; free local = 100), and `value`
   (capability per dollar). Each use case blends them with weights from
   `config.json`; learned adjustments (±8) are added on top.
4. **Routing.** Forward to the winner, stream the response back, and log
   use case, model, latency, token usage, and **actual cost** to
   `route_log.jsonl`.
5. **Learning.** `mrouter learn` aggregates the log (+ your `feedback` entries)
   into bounded per-(use case, model) adjustments in `learned.json`. ≥5
   samples before reliability moves a score; feedback counts more than volume;
   everything is plain JSON you can inspect or reset.

## Configuration

Everything tunable lives in `config.json` (not code). Copy
`config.example.json`, then see **[CONFIG.md](CONFIG.md)** for the full
reference — endpoints, weights, the family knowledge base, hardware, Jev,
and learning bounds. Set `MROUTER_CONFIG=/path/to/config.json` to keep
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
(and any OpenAI client works — Codex, Cline, Aider, your own scripts):

```bash
hermes config set model.provider custom
hermes config set model.default auto
hermes config set model.base_url http://127.0.0.1:8730/v1
```

Every new Hermes session now classifies its own prompts and picks models
across your fleet. Keep a cloud fallback so Hermes survives the proxy being
down (`hermes fallback add`).

To run mrouter at login (macOS): adapt `scripts/serve.sh` (it also starts LM
Studio's server if down) in a launchd agent with `KeepAlive`.

## What it does not do

- No GPU-level scheduling — LM Studio owns local inference; mrouter only chooses.
- No fine-tuning or model serving.
- The family capability scores in `config.json` are curated heuristics —
  opinionated but transparent, editable in one place, and improved by
  learning over time.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

34 hermetic tests: scoring, cost math, sensitive-invariant routing, classifier
shapes (incl. mocked Jev), full proxy integration against in-process fake
upstreams, config loading, and learning bounds.

## License

MIT — see [LICENSE](LICENSE).
