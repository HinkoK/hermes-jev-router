"""jev-router: Jev picks the model for every message in Hermes.

Light tasks go to the cheap model, ordinary work to the middle one, hard multi-step problems to the strongest.
Jev (TypeSafe's System One model) makes the call in ~0.3 s for ~$0.00005 and never writes text itself.

    /router on                  auto: Jev chooses light / standard / heavy for each message
    /router off                 no routing, the chat model answers
    /router light|standard|heavy   pin every message to one tier (aliases: luna, sol, astra)
    /router max standard        never go above a tier (e.g. no Astra); /router max off removes the cap
    /router log N               last N decisions
    /router check N             which model actually answered, time and $ per decision
    /router stats N             tier shares and money compared with sending everything to one model
    /router status

Terminal: `hermes jev-router <same words>` and `hermes jev-router route "text"` (decision only, no answer).

How it works: before each model request Hermes lets plugins edit it (llm_request middleware). The router
reads the latest user message plus the last few turns, asks Jev five narrow questions in one call, combines
the answers in code (see `decide`) and swaps the request's model name within the same provider.
Nothing else in the request changes. Cron runs are never routed.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PLUGIN_ID = "jev-router"
TIERS = ("light", "standard", "heavy")
TIER_RU = {"light": "лёгкое", "standard": "среднее", "heavy": "сложное"}
RANK = {t: i for i, t in enumerate(TIERS)}
ALIASES = {"luna": "light", "sol": "standard", "astra": "heavy", "лёгкое": "light", "легкое": "light",
           "среднее": "standard", "сложное": "heavy"}
MODES = ("on", "off", *TIERS)

# Default model per tier for each provider. Override: plugins.entries.jev-router.settings.models.<provider>.<tier>
DEFAULT_MODELS = {
    "openai-codex": {"light": "gpt-6-luna", "standard": "gpt-6-sol", "heavy": "gpt-6-astra"},
    "openrouter": {"light": "deepseek/deepseek-v4-flash", "standard": "openai/gpt-6-sol", "heavy": "openai/gpt-6-astra"},
}
API_MODES = {"openai-codex": "codex_responses", "openrouter": "chat_completions"}
PRICE_IDS = {"gpt-6-luna": "openai/gpt-6-luna", "gpt-6-sol": "openai/gpt-6-sol", "gpt-6-astra": "openai/gpt-6-astra"}
NAMES = {"gpt-6-luna": "GPT-6 Luna", "gpt-6-sol": "GPT-6 Sol", "gpt-6-astra": "GPT-6 Astra",
         "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash", "openai/gpt-6-sol": "GPT-6 Sol", "openai/gpt-6-astra": "GPT-6 Astra"}

JEV_OPENROUTER = ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13")
JEV_TYPESAFE = ("https://api.typesafe.ai/v1/systemone", "jev-latest")
JEV_PRICE_PER_TOKEN = 0.042 / 1_000_000  # estimate when the provider does not report cost

MSG_LIMIT = 12000
TURN_LIMIT = 400
CONTEXT_TURNS = 4
FOOTER_RE = re.compile(r"\n*— Ответила: [^\n]*\s*$")
LOG_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \w+ \[([^\]]+)\] agent\.conversation_loop: "
    r"API call #(\d+): model=(\S+) provider=(\S+).*?\bid=(gen-[A-Za-z0-9-]+)")
logger = logging.getLogger(__name__)

# ---- the questions Jev answers (eval/ uses exactly these) --------------------------------------------------------

TIER_QUESTION = {
    "type": "choice",
    "instructions": {
        "question": "Which model should answer `latest_message`? Choose the cheapest model that would answer as well as the strongest model.",
        "focus": "Judge what the user needs done now and how much careful multi-step reasoning a good answer takes. Ignore the length of the message: a long pasted text with a mechanical request is light; a short question that needs diagnosis, design or weighing many conditions is heavy.",
        "conversation": "If `latest_message` continues a task from `earlier_in_conversation`, judge the work that the task now requires. If it only closes the task (thanks, it works, bye), it is light.",
    },
    "criteria": {
        "light": {
            "what": "A short or mechanical answer that needs almost no thinking and whose mistakes are obvious at a glance.",
            "includes": ["greetings, thanks, small talk, confirmations like 'ok'",
                         "a short fact, a one-line definition, unit conversion, simple arithmetic",
                         "translating, fixing typos or grammar, reformatting, changing case, listing items from a given text — even when the pasted text is long",
                         "pulling dates, names, numbers or a list out of a given text",
                         "closing a task: 'thanks, it works', 'that's all for today'"],
            "not_for": "Writing new content, giving advice, planning, or anything that needs judgment.",
        },
        "standard": {
            "what": "Normal work that needs thought and good writing, but the path to the answer is clear.",
            "includes": ["writing a post, email, script, description, resume, greeting, reply to a customer",
                         "explaining a concept, comparing options, advice on life, work or purchases",
                         "making a plan: content, trip, workouts, study, event",
                         "routine code: a small function or script, an error with a clear message, setup by instructions",
                         "summarizing or critiquing a text, brainstorming, names and ideas",
                         "rewriting or extending earlier content in a different way"],
            "not_for": "Mechanical edits of given text (light) or problems with an unknown cause, system design or many interacting conditions (heavy).",
        },
        "heavy": {
            "what": "Multi-step reasoning: the cause is unknown, there are many interacting conditions, or a subtle mistake is costly.",
            "includes": ["finding the cause of a failure that has no obvious error: intermittent crashes, leaks, race conditions, 'sometimes', 'only in production', 'after the update'",
                         "designing a system, architecture, database schema, scaling or a migration without downtime",
                         "decisions with many conditions and a costly mistake: money, taxes, legal risk, medications, business strategy with numbers",
                         "math proofs, algorithms, optimization, load estimation, multi-scenario calculations",
                         "large agentic work: understanding a whole project, changing code across many files, security audits, deep reviews",
                         "continuing such a task: 'ok, do it', 'continue', 'it still fails', 'what if we get 10x users?'"],
            "not_for": "Simple one-step answers, even when the topic sounds technical.",
        },
    },
}
DEPTH_QUESTION = {
    "type": "score",
    "instructions": {
        "question": "How much careful multi-step reasoning does a high-quality answer to `latest_message` require?",
        "conversation": "If `latest_message` continues a task from `earlier_in_conversation`, rate the reasoning that task now needs.",
    },
    "criteria": [
        "None: a reply, a fact, or a mechanical transformation of text the user gave",
        "Some: ordinary writing, explaining, advising or planning where the path is clear",
        "Deep: finding an unknown cause, designing a system, rigorous math, or deciding under many interacting conditions where a mistake is costly",
    ],
}
SIGNAL_QUESTIONS = {
    "continues": {"type": "noul", "instructions": "Does `latest_message` ask to keep working on a task from `earlier_in_conversation` (do it, continue, change it, it still fails, what if ...), rather than start something new or just say thanks?"},
    "agentic": {"type": "noul", "instructions": "Does `latest_message` ask the assistant to carry out work across a project, many files, commands or web pages, rather than only answer in chat?"},
    "costly": {"type": "noul", "instructions": "Would a subtly wrong answer to `latest_message` cause real harm: lost money, legal trouble, a health risk, a security hole or a production outage?"},
}
QUESTIONS = {"tier": TIER_QUESTION, "depth": DEPTH_QUESTION, **SIGNAL_QUESTIONS}
# Decision rule, set on the dev set before the holdout check (eval/REPORT.md).
HEAVY_AT = 0.45
LIGHT_AT = 0.55


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + " […] " + text[-half:]


def build_state(latest: str, earlier: list[dict] | None = None) -> dict:
    """What Jev sees: the latest message and up to four earlier turns (clipped)."""
    state = {"latest_message": _clip(latest, MSG_LIMIT)}
    turns = [{"role": t["role"], "text": _clip(t["text"], TURN_LIMIT)} for t in (earlier or []) if t.get("text")]
    if turns:
        state["earlier_in_conversation"] = turns[-CONTEXT_TURNS:]
    return state


def decide(answers: dict, heavy_at: float = HEAVY_AT, light_at: float = LIGHT_AT) -> tuple[str, dict]:
    """Combine Jev's answers into a tier. Under-routing (hard task to a weak model) costs more than over-routing,
    so the depth score and the costly/agentic signals can only push towards heavy; light needs the tier choice
    and a shallow depth to agree."""
    tier = answers.get("tier") or {}
    p = {k: float(v) for k, v in (tier.get("probabilities") or {}).items()}
    depth = float((answers.get("depth") or {}).get("score") if (answers.get("depth") or {}).get("score") is not None else 1.0)
    n = {k: float((answers.get(k) or {}).get("noul") or 0.0) for k in SIGNAL_QUESTIONS}
    heavy = p.get("heavy", 0.0) + 0.5 * max(0.0, depth - 1.0) + 0.2 * n["costly"] + 0.2 * n["agentic"]
    light = p.get("light", 0.0) - 0.5 * max(0.0, depth - 0.5)
    if heavy >= heavy_at:
        chosen = "heavy"
    elif light >= light_at:
        chosen = "light"
    else:
        chosen = "standard"
    detail = {"p": {k: round(v, 3) for k, v in p.items()}, "depth": round(depth, 2),
              **{k: round(v, 2) for k, v in n.items()}, "heavy_score": round(heavy, 3), "light_score": round(light, 3),
              "jev_choice": tier.get("choice"), "jev_confidence": tier.get("confidence")}
    return chosen, detail


# ---- helpers ---------------------------------------------------------------------------------------------------------

def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _env_key(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    try:
        for line in (_hermes_home() / ".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _jev_route() -> tuple[str, str, str] | None:
    """(endpoint, model, key): TypeSafe direct when TYPESAFE_API_KEY is set, otherwise OpenRouter."""
    key = _env_key("TYPESAFE_API_KEY")
    if key:
        return (*JEV_TYPESAFE, key)
    key = _env_key("OPENROUTER_API_KEY")
    if key:
        return (*JEV_OPENROUTER, key)
    return None


def _post_jev(state: dict, *, timeout: float) -> dict:
    route = _jev_route()
    if route is None:
        raise RuntimeError("no TYPESAFE_API_KEY or OPENROUTER_API_KEY")
    endpoint, model, key = route
    body = json.dumps({"model": model, "state": state, "questions": QUESTIONS}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _jev_cost(data: dict) -> float | None:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    try:
        cost = float(usage.get("cost"))
        if math.isfinite(cost) and cost >= 0:
            return cost
    except (TypeError, ValueError):
        pass
    tokens = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    return tokens * JEV_PRICE_PER_TOKEN if tokens else None


def _part_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, dict) and isinstance(p.get("text"), str)).strip()
    return ""


def _conversation(request: dict) -> tuple[str, list[dict]]:
    """Latest user text and the earlier user/assistant turns, from chat `messages` or Responses `input`."""
    items = request.get("messages")
    if not isinstance(items, list):
        items = request.get("input")
    if not isinstance(items, list):
        return "", []
    turns = []
    for item in items:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        text = _part_text(item.get("content"))
        if item.get("role") == "assistant":
            text = FOOTER_RE.sub("", text).strip()
        if text:
            turns.append({"role": item["role"], "text": text})
    last_user = max((i for i, t in enumerate(turns) if t["role"] == "user"), default=-1)
    if last_user < 0:
        return "", []
    return turns[last_user]["text"], turns[:last_user]


def _strip_prior_reasoning(request: dict) -> int:
    """Drop encrypted reasoning/compaction items from earlier turns (before the last user item).
    Codex seals them to the model that produced them: replaying one model's blob to another is HTTP 400."""
    items = request.get("input")
    if not isinstance(items, list):
        return 0
    last_user = max((i for i, it in enumerate(items) if isinstance(it, dict) and it.get("role") == "user"), default=-1)
    kept, dropped = [], 0
    for i, it in enumerate(items):
        if i < last_user and isinstance(it, dict) and it.get("type") in ("reasoning", "compaction") and it.get("encrypted_content"):
            dropped += 1
            continue
        kept.append(it)
    if dropped:
        request["input"] = kept
    return dropped


def _has_prior_reasoning(request: dict) -> bool:
    items = request.get("input")
    if not isinstance(items, list):
        return False
    last_user = max((i for i, it in enumerate(items) if isinstance(it, dict) and it.get("role") == "user"), default=-1)
    return any(isinstance(it, dict) and it.get("type") in ("reasoning", "compaction") and it.get("encrypted_content")
               for it in items[:max(last_user, 0)])


def _same_model(served: Any, chosen: str) -> bool:
    """Servers echo the base slug or a dated snapshot of it (gpt-6-astra-20260903)."""
    served = str(served or "")
    return served == chosen or served.startswith(chosen + "-")


def _name(model: str) -> str:
    base = next((m for m in NAMES if _same_model(model, m)), None)
    return NAMES.get(base, model) if base else str(model)


def _http_json(url: str, key: str = "", timeout: float = 15) -> Any:
    headers = {"Authorization": "Bearer " + key} if key else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        return json.load(response)


def _openrouter_catalog() -> dict[str, dict]:
    try:
        rows = _http_json("https://openrouter.ai/api/v1/models", timeout=20).get("data", [])
        return {row["id"]: row for row in rows if isinstance(row, dict) and row.get("id")}
    except Exception:  # noqa: BLE001
        return {}


def _api_price(usage: dict, pricing: dict) -> float | None:
    """What a call would cost per API: fresh input + cached input + output (reasoning included)."""
    if not pricing:
        return None
    try:
        pin, pout = float(pricing.get("prompt", 0)), float(pricing.get("completion", 0))
        pread = float(pricing.get("input_cache_read") or pin)
        return (int(usage.get("input_tokens") or 0) * pin + int(usage.get("cache_read_tokens") or 0) * pread
                + int(usage.get("cache_write_tokens") or 0) * pin + int(usage.get("output_tokens") or 0) * pout)
    except (TypeError, ValueError):
        return None


def _generation(key: str, generation_id: str) -> dict | None:
    url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode({"id": generation_id})
    for attempt in range(4):
        try:
            obj = _http_json(url, key)
            return obj.get("data", obj) if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001: metadata lags a few seconds
            if attempt < 3:
                time.sleep(1.5)
    return None


def _openrouter_calls() -> list[dict]:
    rows = []
    for path in sorted((_hermes_home() / "logs").glob("agent.log*"), key=lambda p: p.stat().st_mtime):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = LOG_LINE.match(line)
            if m and m.group(5) == "openrouter":
                ts = time.mktime(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timetuple())
                rows.append({"t": ts, "session_id": m.group(2), "generation_id": m.group(6)})
    seen, unique = set(), []
    for row in sorted(rows, key=lambda r: r["t"]):
        if row["generation_id"] not in seen:
            seen.add(row["generation_id"])
            unique.append(row)
    return unique


# ---- router ----------------------------------------------------------------------------------------------------------

class Router:
    def __init__(self, ctx: Any, transport=None) -> None:
        self.ctx = ctx
        self.transport = transport or _post_jev
        self.memo: dict[str, dict] = {}            # "mode:turn_id" -> decision (tool loops reuse it)
        self.turn_info: dict[str, dict] = {}       # turn_id -> decision + served model (footer, journal)
        self.session_models: dict[str, set] = {}   # session_id -> models that answered earlier turns

    # settings
    def setting(self, name: str, default: Any) -> Any:
        return self.ctx.get_config(name, default)

    def mode(self) -> str:
        value = self.setting("mode", "off")
        if isinstance(value, bool):  # `hermes config set ... mode on/off` stores YAML booleans
            return "on" if value else "off"
        mode = ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())
        return mode if mode in MODES else "off"

    def models(self, provider: str) -> dict[str, str] | None:
        base = DEFAULT_MODELS.get(provider)
        custom = self.setting("models", {}) or {}
        custom = custom.get(provider) if isinstance(custom, dict) else None
        if base is None and not isinstance(custom, dict):
            return None
        merged = dict(base or {})
        if isinstance(custom, dict):
            merged.update({t: str(m) for t, m in custom.items() if t in TIERS and m})
        return merged if all(t in merged for t in TIERS) else None

    def max_tier(self) -> str | None:
        value = str(self.setting("max_tier", "") or "").strip().lower()
        value = ALIASES.get(value, value)
        return value if value in TIERS else None

    def _flag(self, name: str, default: bool) -> bool:
        value = self.setting(name, default)
        return value not in (False, "false", "False", "0", 0, "off")

    def data_dir(self) -> Path:
        path = getattr(getattr(self.ctx, "state", None), "data_dir", None)
        return Path(path) if path else _hermes_home() / "data" / PLUGIN_ID

    def audit(self, record: dict) -> None:
        try:
            directory = self.data_dir()
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "decisions.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "t": time.time(), **record}, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("jev-router journal unavailable: %s", type(exc).__name__)

    # decision
    def ask_jev(self, latest: str, earlier: list[dict]) -> dict:
        try:
            timeout = float(self.setting("timeout_s", 4.0))
            timeout = timeout if math.isfinite(timeout) and timeout > 0 else 4.0
        except (TypeError, ValueError):
            timeout = 4.0
        start = time.monotonic()
        try:
            data = self.transport(build_state(latest, earlier), timeout=timeout)
            tier, detail = decide(data.get("answers") or {}, float(self.setting("heavy_at", HEAVY_AT)),
                                  float(self.setting("light_at", LIGHT_AT)))
            return {"tier": tier, "reason": "jev", "detail": detail, "latency_ms": int((time.monotonic() - start) * 1000),
                    "jev_cost_usd": _jev_cost(data)}
        except Exception as exc:  # noqa: BLE001: routing never blocks a turn
            logger.warning("jev-router: Jev request failed: %s", type(exc).__name__)
            return {"tier": None, "reason": "jev_error", "detail": {}, "latency_ms": int((time.monotonic() - start) * 1000),
                    "jev_cost_usd": None}

    def _replay_guard(self, routed: dict, session_id: str, configured: str, target: str) -> int:
        """Codex: strip earlier-turn reasoning unless every earlier turn was answered by `target`."""
        earlier = self.session_models.get(session_id)
        if earlier is None:
            earlier = {configured} if _has_prior_reasoning(routed) else set()
            self.session_models[session_id] = set(earlier)
            if len(self.session_models) > 1000:
                self.session_models.pop(next(iter(self.session_models)))
        dropped = 0 if earlier <= {target} else _strip_prior_reasoning(routed)
        self.session_models[session_id].add(target)
        return dropped

    def _cap(self, routed: dict) -> int:
        """OpenRouter reserves max_tokens x price up front; keep the reservation bounded."""
        try:
            cap = max(256, min(int(self.setting("max_output_tokens", 8192)), 65536))
        except (ValueError, TypeError):
            cap = 8192
        field = "max_completion_tokens" if "max_completion_tokens" in routed else "max_tokens"
        current = routed.get(field)
        routed[field] = min(current, cap) if isinstance(current, int) and current > 0 else cap
        return routed[field]

    @staticmethod
    def _clamp_reasoning(routed: dict, model: str) -> None:
        reasoning = routed.get("reasoning")
        if not isinstance(reasoning, dict) or not reasoning.get("effort"):
            return
        try:
            from agent.reasoning_effort import clamp_effort, codex_supported_efforts
            supported = codex_supported_efforts(model)
            if supported:
                routed["reasoning"] = {**reasoning, "effort": clamp_effort(str(reasoning["effort"]), supported)}
        except Exception:  # noqa: BLE001
            pass

    def on_llm_request(self, request=None, original_request=None, *, provider="", api_mode="", model="", turn_id="",
                       session_id="", api_request_id="", platform="", api_call_count=0, **kwargs):
        try:
            provider = (provider or "").lower()
            tiers = self.models(provider)
            skip = self.setting("skip_platforms", ["cron"]) or []
            if tiers is None or (platform or "").lower() in [str(s).lower() for s in skip] or not isinstance(request, dict):
                return None
            codex = api_mode == "codex_responses"
            mode = self.mode()
            if mode == "off":
                # After routed turns the chat model must not receive another model's sealed reasoning.
                if codex and session_id in self.session_models:
                    routed = dict(request)
                    if self._replay_guard(routed, session_id, model, model):
                        return {"request": routed, "source": PLUGIN_ID, "reason": "strip_foreign_reasoning"}
                return None
            if API_MODES.get(provider) and api_mode != API_MODES[provider]:
                return None
            if model not in tiers.values():
                self.audit({"event": "skip", "reason": "chat_model_not_in_tiers", "model": model, "provider": provider})
                return None
            memo_key = f"{mode}:{turn_id}" if turn_id else ""
            replayed = bool(memo_key and memo_key in self.memo)
            if replayed:
                decision = self.memo[memo_key]
            else:
                latest, earlier = _conversation(request)
                if not latest:
                    return None
                if mode in TIERS:
                    decision = {"tier": mode, "reason": "pinned", "detail": {}, "latency_ms": 0, "jev_cost_usd": 0.0}
                else:
                    decision = self.ask_jev(latest, earlier)
                    cap = self.max_tier()
                    if decision["tier"] and cap and RANK[decision["tier"]] > RANK[cap]:
                        decision["detail"] = {**decision["detail"], "capped_from": decision["tier"]}
                        decision["tier"] = cap
                decision["chars"] = len(latest)
                if self._flag("log_text", False):
                    decision["preview"] = latest[:80]
                if memo_key:
                    self.memo[memo_key] = decision
                    if len(self.memo) > 256:
                        self.memo.pop(next(iter(self.memo)))
            target = tiers[decision["tier"]] if decision["tier"] else model
            routed = dict(request)
            routed["model"] = target
            dropped, cap = 0, None
            if codex:
                self._clamp_reasoning(routed, target)
                dropped = self._replay_guard(routed, session_id, model, target)
            elif provider == "openrouter" and target != model:
                cap = self._cap(routed)
            if not replayed:
                if turn_id and turn_id not in self.turn_info:
                    self.turn_info[turn_id] = {"tier": decision["tier"], "model": target, "reason": decision["reason"],
                                               "latency_ms": decision["latency_ms"], "detail": decision["detail"]}
                    if len(self.turn_info) > 512:
                        self.turn_info.pop(next(iter(self.turn_info)))
                self.audit({"event": "route", "mode": mode, "provider": provider, "tier": decision["tier"], "model": target,
                            "chat_model": model, "reason": decision["reason"], "detail": decision["detail"],
                            "latency_ms": decision["latency_ms"], "jev_cost_usd": decision["jev_cost_usd"],
                            "chars": decision.get("chars"), "preview": decision.get("preview"),
                            "reasoning_items_dropped": dropped, "max_output_tokens": cap,
                            "turn_id": turn_id, "session_id": session_id, "platform": platform})
            if target == model and not dropped:
                return None
            return {"request": routed, "source": PLUGIN_ID, "reason": target}
        except Exception as exc:  # noqa: BLE001: never break a turn
            logger.warning("jev-router routing failed: %s", type(exc).__name__)
            return None

    def on_post_api_request(self, *, turn_id="", session_id="", provider="", response_model=None, usage=None,
                            api_duration=0.0, **kwargs) -> None:
        try:
            info = self.turn_info.get(turn_id) if turn_id else None
            if info is None:
                return
            if response_model:
                info["served"] = str(response_model)
            self.audit({"event": "response", "turn_id": turn_id, "session_id": session_id, "provider": provider,
                        "sent_model": info["model"], "response_model": response_model,
                        "usage": usage if isinstance(usage, dict) else None, "api_duration_s": round(float(api_duration or 0), 3)})
        except Exception as exc:  # noqa: BLE001
            logger.debug("jev-router response journal failed: %s", exc)

    def footer(self, info: dict) -> str:
        served = info.get("served") or info["model"]
        name = _name(served)
        if not _same_model(served, info["model"]):
            name += f" (просили {_name(info['model'])})"
        if info["reason"] == "pinned":
            how = f"закреплено: {TIER_RU[info['tier']]}"
        elif info["reason"] == "jev" and (info.get("detail") or {}).get("capped_from"):
            how = (f"Джев: {TIER_RU[info['detail']['capped_from']]} задание, ограничено до «{TIER_RU[info['tier']]}» "
                   f"· {info.get('latency_ms')} мс")
        elif info["reason"] == "jev":
            how = f"Джев: {TIER_RU[info['tier']]} задание · {info.get('latency_ms')} мс"
        else:
            how = "Джев недоступен, модель чата"
        return f"Ответила: {name} · {how}"

    def on_transform_llm_output(self, response_text="", turn_id="", **kwargs):
        try:
            if not self._flag("show_model", True):
                return None
            info = self.turn_info.get(turn_id) if turn_id else None
            if info is None or not isinstance(response_text, str) or not response_text.strip():
                return None
            return response_text.rstrip() + "\n\n— " + self.footer(info)
        except Exception as exc:  # noqa: BLE001
            logger.debug("jev-router footer failed: %s", exc)
            return None

    # ---- journal views ---------------------------------------------------------------------------------------------

    def records(self) -> list[dict]:
        path = self.data_dir() / "decisions.jsonl"
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def decisions(self, count: int, records: list[dict] | None = None) -> list[dict]:
        rows = [r for r in (records if records is not None else self.records()) if r.get("event") == "route"]
        return rows[-max(1, min(int(count), 500)):]

    def verify(self, count: int) -> list[dict]:
        records = self.records()
        chosen = self.decisions(count, records)
        catalog = calls = key = None
        out = []
        for row in chosen:
            if row.get("provider") == "openrouter":
                if calls is None:
                    calls, key = _openrouter_calls(), _env_key("OPENROUTER_API_KEY")
                    catalog = catalog if catalog is not None else _openrouter_catalog()
                later = [r for r in chosen if r.get("session_id") == row.get("session_id") and float(r["t"]) > float(row["t"])]
                end = min((float(r["t"]) for r in later), default=float("inf"))
                gens = [c for c in calls if c["session_id"] == row.get("session_id") and float(row["t"]) - 2 <= c["t"] < end]
                slug = (catalog.get(row["model"]) or {}).get("canonical_slug") or row["model"]
                served, cost, secs, ok = [], 0.0, 0.0, bool(gens)
                for g in gens:
                    data = _generation(key, g["generation_id"]) if key else None
                    if not isinstance(data, dict):
                        ok = False
                        continue
                    served.append(data.get("model"))
                    cost += float(data.get("total_cost") or 0)
                    secs += (float(data.get("generation_time") or 0) + float(data.get("latency") or 0)) / 1000
                    ok = ok and data.get("model") == slug
                out.append({**row, "served": served, "answer_usd": cost, "cost_kind": "счёт OpenRouter", "seconds": secs, "verified": ok})
                continue
            catalog = catalog if catalog is not None else _openrouter_catalog()
            pricing = (catalog.get(PRICE_IDS.get(row["model"], "")) or {}).get("pricing", {})
            responses = [r for r in records if r.get("event") == "response" and r.get("turn_id") == row.get("turn_id")]
            served = [r.get("response_model") for r in responses]
            out.append({**row, "served": served, "answer_usd": sum(_api_price(r.get("usage") or {}, pricing) or 0 for r in responses),
                        "cost_kind": "как если бы по API", "seconds": sum(float(r.get("api_duration_s") or 0) for r in responses),
                        "usage": [r.get("usage") or {} for r in responses],
                        "verified": bool(responses) and all(_same_model(s, row["model"]) for s in served)})
        return out

    def text_status(self) -> str:
        try:
            from hermes_cli.config import load_config
            main = load_config().get("model") or {}
        except Exception:  # noqa: BLE001
            main = {}
        provider = str(main.get("provider", "")).lower()
        tiers = self.models(provider)
        lines = [f"Роутер: {self.mode()}", f"Модель чата: {main.get('default')} ({provider or '?'})"]
        if self.max_tier():
            lines.append(f"Потолок: не выше уровня «{TIER_RU[self.max_tier()]}»")
        if tiers:
            lines.append("Уровни: " + ", ".join(f"{TIER_RU[t]} → {_name(tiers[t])}" for t in TIERS))
            if main.get("default") not in tiers.values():
                lines.append("Внимание: модель чата не входит в уровни, роутер её не трогает.")
        else:
            lines.append(f"Для провайдера {provider or '?'} уровни не заданы: роутер не работает (см. README, settings.models).")
        route = _jev_route()
        lines.append("Джев: " + ("TypeSafe напрямую" if route and "typesafe.ai" in route[0] else "через OpenRouter" if route else "нет ключа (TYPESAFE_API_KEY или OPENROUTER_API_KEY)"))
        lines.append(f"Журнал: {self.data_dir() / 'decisions.jsonl'}")
        return "\n".join(lines)

    def text_log(self, count: int) -> str:
        rows = self.decisions(count)
        if not rows:
            return "Решений пока нет."
        out = ["# | время | уровень | модель | как решено | мс | $ Джева | сообщение"]
        for i, r in enumerate(rows, 1):
            how = {"jev": "Джев", "pinned": "закреплено"}.get(r.get("reason"), r.get("reason"))
            msg = repr(r["preview"][:40]) if r.get("preview") else f"{r.get('chars')} зн."
            cost = "—" if r.get("jev_cost_usd") is None else f"${r['jev_cost_usd']:.6f}"
            out.append(f"{i} | {str(r.get('ts', ''))[11:19]} | {TIER_RU.get(r.get('tier'), '—')} | {_name(r['model'])} | {how} | "
                       f"{r.get('latency_ms')} | {cost} | {msg}")
        return "\n".join(out)

    def text_check(self, count: int) -> str:
        rows = self.verify(count)
        if not rows:
            return "Решений пока нет."
        out = ["# | уровень | просили | ответила | сверка | $ ответа | сек"]
        for i, r in enumerate(rows, 1):
            served = ", ".join(sorted(set(str(s) for s in r["served"]))) or "нет ответа в журнале"
            out.append(f"{i} | {TIER_RU.get(r.get('tier'), '—')} | {_name(r['model'])} | {served} | "
                       f"{'совпало' if r['verified'] else 'НЕ СОВПАЛО'} | ${r['answer_usd']:.4f} | {r['seconds']:.1f}")
        kinds = ", ".join(sorted(set(r["cost_kind"] for r in rows)))
        out.append(f"Итого: ответы ${sum(r['answer_usd'] for r in rows):.4f} ({kinds}), Джев "
                   f"${sum(float(r.get('jev_cost_usd') or 0) for r in rows):.6f}; сверено {sum(r['verified'] for r in rows)} из {len(rows)}")
        return "\n".join(out)

    def text_stats(self, count: int) -> str:
        rows = self.verify(count)
        if not rows:
            return "Решений пока нет."
        shares = {t: sum(r.get("tier") == t for r in rows) for t in TIERS}
        lines = [f"Решений: {len(rows)} — " + ", ".join(f"{TIER_RU[t]} {shares[t]}" for t in TIERS)]
        codex = [r for r in rows if r.get("cost_kind") == "как если бы по API" and r.get("usage")]
        if codex:
            catalog = _openrouter_catalog()
            price = {m: (catalog.get(PRICE_IDS.get(m, "")) or {}).get("pricing", {}) for m in PRICE_IDS}
            actual = sum(r["answer_usd"] for r in codex)
            alt = {m: sum(_api_price(u, price[m]) or 0 for r in codex for u in r["usage"]) for m in PRICE_IDS}
            lines.append(f"Ответы с роутером: ${actual:.4f} (как если бы по API)")
            lines.append("Если бы всё шло на одну модель: " + ", ".join(f"{_name(m)} ${v:.4f}" for m, v in alt.items()))
        lines.append(f"Джев за эти решения: ${sum(float(r.get('jev_cost_usd') or 0) for r in rows):.6f}")
        return "\n".join(lines)

    def command(self, raw: str) -> str:
        parts = (raw or "").split()
        action = parts[0].lower() if parts else "status"
        action = ALIASES.get(action, action)
        number = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
        if action in MODES:
            self.ctx.set_config("mode", action)
            self.memo.clear()
            return {"on": "Роутер включён: Джев выбирает уровень для каждого сообщения.",
                    "off": "Роутер выключен: отвечает модель чата."}.get(
                action, f"Все сообщения идут на уровень «{TIER_RU.get(action, action)}».")
        if action == "max":
            value = ALIASES.get(parts[1].lower(), parts[1].lower()) if len(parts) > 1 else ""
            if value in ("off", "none", "нет"):
                self.ctx.set_config("max_tier", "")
                return "Потолок снят: Джев может выбрать любой уровень."
            if value not in TIERS:
                return "Использование: /router max light|standard|heavy (или luna|sol|astra), /router max off"
            self.ctx.set_config("max_tier", value)
            self.memo.clear()
            return f"Потолок: не выше уровня «{TIER_RU[value]}». Задания сложнее уходят на этот уровень."
        if action == "log":
            return self.text_log(number)
        if action == "check":
            return self.text_check(number)
        if action == "stats":
            return self.text_stats(number)
        if action == "status":
            return self.text_status()
        return "Команды: /router on | off | light | standard | heavy | max <уровень|off> | log N | check N | stats N | status"


def register(ctx: Any) -> None:
    router = Router(ctx)
    ctx.register_middleware("llm_request", router.on_llm_request)
    ctx.register_hook("post_api_request", router.on_post_api_request)
    ctx.register_hook("transform_llm_output", router.on_transform_llm_output)
    ctx.register_command("router", router.command,
                         description="Роутер Джева: on, off, light/standard/heavy, max, log, check, stats, status",
                         args_hint="on|off|light|standard|heavy|max <tier|off>|log N|check N|stats N|status")

    def setup(parser):
        sub = parser.add_subparsers(dest="jev_command")
        for name in (*MODES, "luna", "sol", "astra", "status"):
            sub.add_parser(name)
        route = sub.add_parser("route", help="show Jev's decision for a text (no model answer)")
        route.add_argument("text", nargs="+")
        for name in ("log", "check", "stats"):
            sub.add_parser(name).add_argument("count", nargs="?", type=int, default=10)
        sub.add_parser("max").add_argument("tier")

    def handle(args):
        action = args.jev_command or "status"
        if action == "route":
            decision = router.ask_jev(" ".join(args.text), [])
            print(json.dumps(decision, ensure_ascii=False))
            return 0 if decision["tier"] else 1
        if action in ("log", "check", "stats"):
            print(router.command(f"{action} {args.count}"))
            return 0
        if action == "max":
            print(router.command(f"max {args.tier}"))
            return 0
        print(router.command(action))
        return 0

    ctx.register_cli_command(name=PLUGIN_ID, help="Jev model router: on/off/tiers, log, check, stats, route",
                             setup_fn=setup, handler_fn=handle)
    ctx.register_hook("on_session_end", lambda **kwargs: router.memo.clear())
