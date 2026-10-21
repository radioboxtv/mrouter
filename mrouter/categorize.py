"""Task categorization: Jev (TypeSafe AI) when a key exists, else local heuristics.

Jev is TypeSafe AI's System One classifier: POST https://api.typesafe.ai/v1/systemone
with one `choice` question whose options are the use-case names. It returns
calibrated probabilities in ~70-500ms for $0.042/Mtok (output free). The router
uses Jev probabilities to weight/mix routing instead of a hard label, and gates
'sensitive' on confidence (low confidence -> stick to local, conservative).

Fallback (no TYPESAFE_API_KEY env and no setec secret): a deterministic local
classifier — code/reasoning/sensitive keyword scoring + input-length signals.
Zero cost, no network, keeps the privacy boundary safe by default.
"""

import json
import os
import re
import subprocess
import urllib.request

USE_CASES = ("chat", "coding", "sensitive", "thinking")

CODE_HINTS = re.compile(
    r"\b(code|function|bug|error|traceback|python|javascript|typescript|rust|golang|"
    r"sql|regex|compile|debug|refactor|implement|class |def |import |stack trace|"
    r"git|api|sdk|unit test|typescript|segfault)\b", re.I)
SENSITIVE_HINTS = re.compile(
    r"\b(sensitive|confidential|private|secret|password|ssn|social security|salary|"
    r"medical|patient|hipaa|do not share|internal only|nda|proprietary|tax return|"
    r"bank account|credit card)\b", re.I)
THINKING_HINTS = re.compile(
    r"\b(analyze|analyse|research|compare|evaluate|strategy|plan|reason|implications|"
    r"trade-?offs|pros and cons|literature|synthesize|argument|hypothesis|why does|"
    r"explain in depth|essay|report)\b", re.I)


def _local_classify(text):
    """Deterministic zero-cost classifier. Returns {use_case: prob} + confidence."""
    t = (text or "")[:4000]
    low = t.lower()
    scores = {"chat": 0.25, "coding": 0.12, "sensitive": 0.05, "thinking": 0.18}
    if CODE_HINTS.search(low):
        scores["coding"] += 0.55
        scores["thinking"] += 0.08
    if SENSITIVE_HINTS.search(low):
        scores["sensitive"] += 0.75
        scores["coding"] -= 0.03
    if THINKING_HINTS.search(low):
        scores["thinking"] += 0.45
    if len(t) > 1800:
        scores["thinking"] += 0.10
        scores["sensitive"] += 0.05
    # code fences / indentation strongly imply coding
    if "```" in t or re.search(r"^\s{4}\S", t, re.M):
        scores["coding"] += 0.25
    total = sum(max(0.0, v) for v in scores.values()) or 1.0
    probs = {k: round(max(0.0, v) / total, 4) for k, v in scores.items()}
    top = max(probs, key=probs.get)
    return {
        "probs": probs, "top": top, "confidence": probs[top],
        "source": "local-heuristic", "model": "local-heuristic-v1",
    }


def _attempt_baseten(text, config):
    """Run _baseten_classify if a Baseten key is configured; None otherwise.
    Errors are swallowed (logged) -> None so callers can fall back."""
    bcfg = config.get("baseten", {}) or {}
    if not bcfg.get("enabled", False):
        return None
    env_var = bcfg.get("api_key_env", "BASETEN_API_KEY")
    key = os.environ.get(env_var) if env_var else None
    if not key:
        return None
    try:
        return _baseten_classify(text, config, key)
    except Exception as e:
        print(f"[mrouter] Baseten classify failed ({type(e).__name__}); "
              "falling back", flush=True)
        return None


def _baseten_classify(text, config, key):
    """Classify via Baseten decision models (elephant-1 / mammoth-1).

    System One API: POST {endpoint} with {model, state, questions{use_case:
    {type:'choice', instructions, criteria:{uc: description}}}}. Typed answer:
    answers.use_case = {choice, probabilities, confidence}.
    """
    bcfg = config.get("baseten", {}) or {}
    model_id = bcfg.get("model", "decision-models/baseten-mammoth-27b-latest")
    endpoint = bcfg.get("endpoint", "https://inference.baseten.co/v1/systemone")
    criteria = {
        "chat": "General-purpose conversation, questions, and everyday assistance.",
        "coding": "Writing, debugging, reviewing, or explaining source code and technical implementation.",
        "sensitive": "Confidential, private, personal, regulated, or proprietary content that must remain local.",
        "thinking": "Deep reasoning, planning, research, analysis, comparison, or long-form synthesis.",
    }
    body = {
        "model": model_id,
        "state": (text or "")[:8000],
        "questions": {
            "use_case": {
                "type": "choice",
                "instructions": (
                    "Classify this request for LLM routing. "
                    "Choose the single best use case."
                ),
                "criteria": criteria,
            }
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(bcfg.get("timeout_seconds", 5))) as r:
        data = json.loads(r.read().decode())

    ans = (data.get("answers") or {}).get("use_case")
    if not isinstance(ans, dict):
        return None
    probs = ans.get("probabilities") or {}
    top = ans.get("choice") or ans.get("value") or ans.get("label")
    if top not in USE_CASES:
        if probs:
            top = max(probs, key=lambda k: probs[k] if k in USE_CASES else -1)
        else:
            return None
    if top not in USE_CASES:
        return None
    probs = {k: float((probs or {}).get(k, 0.0)) for k in USE_CASES}
    total = sum(probs.values()) or 1.0
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    conf = ans.get("confidence")
    conf = probs[top] if conf is None else float(conf)
    return {
        "probs": probs, "top": top, "confidence": conf,
        "source": "baseten", "model": data.get("model") or model_id,
    }


def _get_typesafe_key(config):
    jcfg = config.get("jev", {}) or {}
    env_var = jcfg.get("api_key_env", "TYPESAFE_API_KEY")
    key = os.environ.get(env_var)
    if key:
        return key.strip()
    secret = jcfg.get("setec_secret")
    if secret:
        env = dict(os.environ)
        server = jcfg.get("setec_server")
        if server:
            env["SETEC_SERVER"] = server
        try:
            out = subprocess.run(["setec", "get", secret], capture_output=True, text=True, timeout=10, env=env)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return None


def _jev_classify(text, config, key):
    jcfg = config.get("jev", {}) or {}
    state = (text or "")[:8000]
    body = {
        "model": jcfg.get("model", "jev-latest"),
        "state": state,
        "questions": {
            "use_case": {
                "type": "choice",
                "options": list(USE_CASES),
                "instructions": (
                    "Classify this request for LLM routing. "
                    "chat = everyday conversation/questions; "
                    "coding = writing/debugging/reviewing code or technical implementation; "
                    "sensitive = confidential, private, personal, or proprietary data that must stay local; "
                    "thinking = deep research, analysis, planning, or long-form reasoning."
                ),
            }
        },
    }
    req = urllib.request.Request(
        jcfg.get("endpoint", "https://api.typesafe.ai/v1/systemone"),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(jcfg.get("timeout_seconds", 5))) as r:
        data = json.loads(r.read().decode())

    ans = (data.get("answers") or {}).get("use_case") or (data.get("questions") or {}).get("use_case") or data.get("use_case")
    # Tolerate response-shape variants while preferring the documented choice shape.
    probs, top, conf = None, None, None
    if isinstance(ans, dict):
        probs = ans.get("probabilities") or ans.get("probs") or ans.get("distribution")
        top = ans.get("value") or ans.get("choice") or ans.get("key") or ans.get("label")
        conf = ans.get("confidence")
    elif isinstance(ans, str):
        top = ans
    if probs:
        probs = {k: float(probs.get(k, 0.0)) for k in USE_CASES}
        s = sum(probs.values()) or 1.0
        probs = {k: round(v / s, 4) for k, v in probs.items()}
        if top not in USE_CASES:
            top = max(probs, key=probs.get)
        if conf is None:
            conf = probs[top]
    elif top in USE_CASES:
        probs = {uc: (0.85 if uc == top else 0.05) for uc in USE_CASES}
        conf = 0.85 if conf is None else float(conf)
    else:
        return None
    return {
        "probs": probs, "top": top, "confidence": float(conf),
        "source": "jev", "model": data.get("model") or jcfg.get("model", "jev-latest"),
    }


def classify_task(text, config):
    """Return {probs, top, confidence, source, model}. Never raises.

    Provider selection via config `decision.provider`:
      auto (default) -> baseten key? then jev key? then local
      baseten | jev | local -> pinned
    """
    decision = config.get("decision", {}) or {}
    provider = decision.get("provider", "auto")
    if provider == "local":
        return _local_classify(text)
    if provider in ("baseten", "auto"):
        res = _attempt_baseten(text, config)
        if res is not None:
            return res
        if provider == "baseten":
            return _local_classify(text)
    if provider in ("jev", "auto"):
        jcfg = config.get("jev", {}) or {}
        if jcfg.get("enabled", True):
            key = _get_typesafe_key(config)
            if key:
                try:
                    result = _jev_classify(text, config, key)
                    if result:
                        return result
                except Exception as e:
                    print(f"[mrouter] Jev classify failed ({type(e).__name__}); using local fallback", flush=True)
        if provider == "jev":
            return _local_classify(text)
    return _local_classify(text)


def extract_user_text(payload):
    """User text from an OpenAI chat payload for classification.

    Classifies on USER messages only: agent frameworks (e.g. Hermes) prepend
    large boilerplate system prompts full of tool/coding keywords that would
    skew classification toward 'coding'. System text is used only as a
    fallback when there is no user content at all.
    """
    user_parts, sys_parts = [], []

    def _collect(m, sink):
        c = m.get("content")
        if isinstance(c, str):
            sink.append(c)
        elif isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                    sink.append(item.get("text", ""))

    for m in payload.get("messages") or []:
        if m.get("role") == "user":
            _collect(m, user_parts)
        elif m.get("role") == "system":
            _collect(m, sys_parts)
    parts = user_parts or sys_parts
    return "\n".join(p for p in parts if p)[:8000]