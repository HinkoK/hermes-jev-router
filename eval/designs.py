"""Варианты вопросов к Джеву для роутинга light / standard / heavy.

У каждого дизайна: payload(item) -> запрос к Джеву, rules() -> {имя: decide(row) -> tier}.
"""
from __future__ import annotations

TIERS = ("light", "standard", "heavy")
JEV = "typesafe/jev-1.13"
MSG_LIMIT = 12000
TURN_LIMIT = 400


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + " […] " + text[-half:]


def context_state(item: dict, turns: int = 4) -> dict:
    state = {"latest_message": clip(item["message"], MSG_LIMIT)}
    earlier = [{"role": t["role"], "text": clip(t["text"], TURN_LIMIT)} for t in item.get("context", [])[-turns:]]
    if earlier:
        state["earlier_in_conversation"] = earlier
    return state


def answers(row: dict) -> dict:
    return (row["raw"] or {}).get("answers") or {}


def probs(ans: dict) -> dict:
    return {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}


# ---- D0: the demo router (last message only, one Choice, argmax, <0.5 -> standard) ----------------------------------
class D0:
    name = "D0"
    MODELS = (("gpt-6-luna", "Greetings, short facts, translation and typo-only correction even for a long text; not substantive rewriting or debugging."),
              ("gpt-6-sol", "Normal writing, explanations, content plans and routine code; safe default when no Jev decision is available."),
              ("gpt-6-astra", "Root-cause debugging, architecture and multi-step technical diagnosis even when the question is short."))

    def payload(self, item):
        return {"model": JEV, "state": {"user_message": clip(item["message"], 16000)},
                "questions": {"model_route": {"type": "choice",
                                              "instructions": "Choose the single best answer model for the actual user intent, not the length of the message. Prefer lower cost when it can do the work reliably.",
                                              "criteria": {str(i): f"{m}: {d}" for i, (m, d) in enumerate(self.MODELS, 1)}}}}

    def rules(self):
        def argmax(row):
            a = answers(row).get("model_route", {})
            if float(a.get("confidence") or 0) < 0.5:
                return "standard"
            return TIERS[int(a.get("choice")) - 1]
        return {"argmax": argmax}


# ---- D1: one structured Choice + conversation context ------------------------------------------------------------
TIER_CRITERIA = {
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
}
D1_INSTRUCTIONS = {
    "question": "Which model should answer `latest_message`? Choose the cheapest model that would answer as well as the strongest model.",
    "focus": "Judge what the user needs done now and how much careful multi-step reasoning a good answer takes. Ignore the length of the message: a long pasted text with a mechanical request is light; a short question that needs diagnosis, design or weighing many conditions is heavy.",
    "conversation": "If `latest_message` continues a task from `earlier_in_conversation`, judge the work that the task now requires. If it only closes the task (thanks, it works, bye), it is light.",
}


class D1:
    name = "D1"

    def payload(self, item):
        return {"model": JEV, "state": context_state(item),
                "questions": {"tier": {"type": "choice", "instructions": D1_INSTRUCTIONS, "criteria": TIER_CRITERIA}}}

    def rules(self):
        def argmax(row):
            return answers(row).get("tier", {}).get("choice") or "standard"

        def cautious(row):
            # Cheaper errors: go up when the heavier tier is plausible.
            p = probs(answers(row).get("tier", {}))
            if p.get("heavy", 0) >= 0.30:
                return "heavy"
            if p.get("light", 0) >= 0.60:
                return "light"
            return "standard"
        return {"argmax": argmax, "cautious": cautious}


# ---- D2: D1 + depth Score + Nouls, combined in code --------------------------------------------------------------
DEPTH = {
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
NOULS = {
    "continues": {"type": "noul", "instructions": "Does `latest_message` ask to keep working on a task from `earlier_in_conversation` (do it, continue, change it, it still fails, what if ...), rather than start something new or just say thanks?"},
    "agentic": {"type": "noul", "instructions": "Does `latest_message` ask the assistant to carry out work across a project, many files, commands or web pages, rather than only answer in chat?"},
    "costly": {"type": "noul", "instructions": "Would a subtly wrong answer to `latest_message` cause real harm: lost money, legal trouble, a health risk, a security hole or a production outage?"},
}


class D2:
    name = "D2"

    def payload(self, item):
        q = {"tier": {"type": "choice", "instructions": D1_INSTRUCTIONS, "criteria": TIER_CRITERIA}, "depth": DEPTH, **NOULS}
        return {"model": JEV, "state": context_state(item), "questions": q}

    def rules(self):
        def parts(row):
            a = answers(row)
            p = probs(a.get("tier", {}))
            depth = float((a.get("depth") or {}).get("score") or 1.0)
            n = {k: float((a.get(k) or {}).get("noul") or 0) for k in NOULS}
            return a, p, depth, n

        def tier_only(row):
            return answers(row).get("tier", {}).get("choice") or "standard"

        def depth_only(row):
            _, _, depth, _ = parts(row)
            return "light" if depth < 0.5 else ("heavy" if depth >= 1.5 else "standard")

        def combined(row):
            _, p, depth, n = parts(row)
            heavy = p.get("heavy", 0) + 0.5 * max(0.0, depth - 1.0) + 0.2 * n["costly"] + 0.2 * n["agentic"]
            light = p.get("light", 0) - 0.5 * max(0.0, depth - 0.5)
            if heavy >= 0.45:
                return "heavy"
            if light >= 0.55:
                return "light"
            return "standard"

        def vote(row):
            # Two independent views: tier choice and depth score. Disagreement goes to the safer (higher) side,
            # except that light needs both views to agree.
            a, p, depth, n = parts(row)
            t = a.get("tier", {}).get("choice") or "standard"
            d = "light" if depth < 0.6 else ("heavy" if depth >= 1.4 else "standard")
            if t == "heavy" or d == "heavy":
                return "heavy"
            if t == "light" and d == "light":
                return "light"
            return "standard"
        return {"tier": tier_only, "depth": depth_only, "combined": combined, "vote": vote}


# ---- D3: Score only ---------------------------------------------------------------------------------------------
class D3:
    name = "D3"

    def payload(self, item):
        return {"model": JEV, "state": context_state(item), "questions": {"depth": DEPTH}}

    def rules(self):
        def cut(lo, hi):
            def f(row):
                s = float((answers(row).get("depth") or {}).get("score") or 1.0)
                return "light" if s < lo else ("heavy" if s >= hi else "standard")
            return f
        return {"0.5/1.5": cut(0.5, 1.5), "0.6/1.3": cut(0.6, 1.3)}


ALL = [D0(), D1(), D2(), D3()]


# ---- PROD: exactly what the plugin sends and decides ----------------------------------------------------------------
import importlib.util as _ilu
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location("jev_router_plugin", _Path(__file__).resolve().parent.parent / "__init__.py")
plugin = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


class PROD:
    name = "PROD"

    def payload(self, item):
        return {"model": plugin.JEV_OPENROUTER[1], "state": plugin.build_state(item["message"], item.get("context", [])),
                "questions": plugin.QUESTIONS}

    def rules(self):
        return {"decide": lambda row: plugin.decide((row["raw"] or {}).get("answers") or {})[0]}


ALL.append(PROD())
