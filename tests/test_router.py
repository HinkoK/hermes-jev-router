"""Offline tests for jev-router (no network, no model). Run: python3 -m unittest discover -s tests"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))  # optional: Hermes' effort clamp
spec = importlib.util.spec_from_file_location("jev_router", ROOT / "__init__.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeState:
    def __init__(self, path):
        self.data_dir = Path(path)


class Ctx:
    def __init__(self, tmp, **config):
        self.state = FakeState(tmp)
        self.config = {"mode": "on", **config}

    def get_config(self, key, default):
        return self.config.get(key, default)

    def set_config(self, key, value):
        self.config[key] = value


def jev(tier="heavy", p=None, depth=1.0, costly=0.0, agentic=0.0, continues=0.0):
    """Fake Jev returning the five answers the plugin asks for."""
    calls = []
    probs = p or {t: (0.9 if t == tier else 0.05) for t in mod.TIERS}

    def transport(state, timeout):
        calls.append(state)
        return {"answers": {"tier": {"type": "choice", "choice": tier, "probabilities": probs, "confidence": 0.9},
                            "depth": {"type": "score", "score": depth},
                            "costly": {"type": "noul", "noul": costly}, "agentic": {"type": "noul", "noul": agentic},
                            "continues": {"type": "noul", "noul": continues}},
                "usage": {"input_tokens": 900, "output_tokens": 60, "cost": 0.00004}}
    return transport, calls


def codex_request(text="Debug a leak", history=(), effort="medium"):
    items = list(history) + [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
    return {"model": "gpt-6-sol", "instructions": "sys", "input": items, "store": False,
            "reasoning": {"effort": effort, "summary": "auto"}, "include": ["reasoning.encrypted_content"]}


OLD_TURN = [{"role": "user", "content": [{"type": "input_text", "text": "earlier question"}]},
            {"type": "reasoning", "encrypted_content": "SOL-BLOB", "summary": []},
            {"role": "assistant", "content": [{"type": "output_text", "text": "earlier answer\n\n— Ответила: GPT-6 Sol · Джев: среднее задание · 300 мс"}]}]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def router(self, transport=None, **config):
        transport = transport or jev()[0]
        return mod.Router(Ctx(self.tmp.name, **config), transport=transport)

    def codex(self, router, request, turn="t1", session="S1", platform="telegram"):
        return router.on_llm_request(request, provider="openai-codex", api_mode="codex_responses", model="gpt-6-sol",
                                     turn_id=turn, session_id=session, platform=platform)


class DecideTests(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(mod.decide({"tier": {"probabilities": {"light": 0.9, "standard": 0.1, "heavy": 0}}, "depth": {"score": 0.2}})[0], "light")
        self.assertEqual(mod.decide({"tier": {"probabilities": {"light": 0.1, "standard": 0.85, "heavy": 0.05}}, "depth": {"score": 1.0}})[0], "standard")
        self.assertEqual(mod.decide({"tier": {"probabilities": {"light": 0, "standard": 0.1, "heavy": 0.9}}, "depth": {"score": 1.8}})[0], "heavy")

    def test_signals_only_push_up(self):
        # standard by choice, but deep reasoning + costly mistake -> heavy
        a = {"tier": {"probabilities": {"light": 0, "standard": 0.8, "heavy": 0.2}}, "depth": {"score": 1.5},
             "costly": {"noul": 0.9}}
        self.assertEqual(mod.decide(a)[0], "heavy")
        # light choice but the depth score disagrees -> not light
        a = {"tier": {"probabilities": {"light": 0.7, "standard": 0.3, "heavy": 0}}, "depth": {"score": 1.0}}
        self.assertEqual(mod.decide(a)[0], "standard")

    def test_missing_answers_default_to_standard(self):
        self.assertEqual(mod.decide({})[0], "standard")


class RoutingTests(Base):
    def test_routes_each_tier_to_its_model(self):
        for tier, model in (("light", "gpt-6-luna"), ("standard", "gpt-6-sol"), ("heavy", "gpt-6-astra")):
            depth = {"light": 0.1, "standard": 1.0, "heavy": 1.9}[tier]
            router = self.router(jev(tier, depth=depth)[0])
            out = self.codex(router, codex_request())
            got = out["request"]["model"] if out else "gpt-6-sol"
            self.assertEqual(got, model, tier)

    def test_state_has_latest_and_earlier_turns_without_footer(self):
        transport, calls = jev("light", depth=0.1)
        self.codex(self.router(transport), codex_request(text="спасибо!", history=OLD_TURN))
        state = calls[0]
        self.assertEqual(state["latest_message"], "спасибо!")
        self.assertEqual([t["role"] for t in state["earlier_in_conversation"]], ["user", "assistant"])
        self.assertNotIn("Ответила", state["earlier_in_conversation"][1]["text"])

    def test_chat_messages_format(self):
        transport, calls = jev("light", depth=0.1)
        router = self.router(transport)
        req = {"model": "openai/gpt-6-sol", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]}
        out = router.on_llm_request(req, provider="openrouter", api_mode="chat_completions", model="openai/gpt-6-sol", turn_id="x", session_id="S")
        self.assertEqual(out["request"]["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(out["request"]["max_tokens"], 8192)
        self.assertEqual(calls[0]["latest_message"], "hi")

    def test_pinned_tier_skips_jev(self):
        transport, calls = jev("light")
        for alias in ("heavy", "astra"):
            router = self.router(transport, mode=alias)
            self.assertEqual(self.codex(router, codex_request())["request"]["model"], "gpt-6-astra")
        self.assertEqual(calls, [])

    def test_tool_loop_reuses_decision(self):
        transport, calls = jev("heavy", depth=1.9)
        router = self.router(transport)
        self.codex(router, codex_request(), turn="t9")
        self.codex(router, codex_request(), turn="t9")
        self.assertEqual(len(calls), 1)

    def test_untouched_cases(self):
        transport, calls = jev("heavy")
        router = self.router(transport)
        self.assertIsNone(self.codex(router, codex_request(), platform="cron"))
        self.assertIsNone(router.on_llm_request(codex_request(), provider="anthropic", api_mode="anthropic_messages", model="x"))
        req = codex_request()
        req["model"] = "gpt-5.5"
        self.assertIsNone(router.on_llm_request(req, provider="openai-codex", api_mode="codex_responses", model="gpt-5.5", turn_id="z"))
        self.assertEqual(calls, [])
        self.assertIsNone(self.codex(self.router(transport, mode="off"), codex_request(), turn="off1"))

    def test_jev_failure_keeps_chat_model(self):
        def broken(state, timeout):
            raise TimeoutError
        router = self.router(broken)
        self.assertIsNone(self.codex(router, codex_request()))
        self.assertEqual(router.decisions(1)[0]["reason"], "jev_error")

    def test_custom_models_setting(self):
        router = self.router(jev("light", depth=0.1)[0], models={"openai-codex": {"light": "gpt-5.6-luna"}})
        self.assertEqual(self.codex(router, codex_request())["request"]["model"], "gpt-5.6-luna")

    def test_no_message_text_in_journal_by_default(self):
        router = self.router(jev("light", depth=0.1)[0])
        self.codex(router, codex_request(text="секретный текст"))
        self.assertNotIn("секретный", (Path(self.tmp.name) / "decisions.jsonl").read_text())
        router = self.router(jev("light", depth=0.1)[0], log_text=True)
        self.codex(router, codex_request(text="можно показать"), turn="t2")
        self.assertIn("можно показать", (Path(self.tmp.name) / "decisions.jsonl").read_text())


class CodexReplayTests(Base):
    def test_strips_foreign_reasoning(self):
        router = self.router(jev("light", depth=0.1)[0])
        req = codex_request(history=OLD_TURN)
        out = self.codex(router, req)
        self.assertNotIn("reasoning", [it.get("type") for it in out["request"]["input"]])
        self.assertEqual(req["input"][1]["encrypted_content"], "SOL-BLOB")

    def test_keeps_reasoning_when_same_model(self):
        router = self.router(mode="heavy")
        self.codex(router, codex_request(), turn="a1")
        history = [{"role": "user", "content": "q"}, {"type": "reasoning", "encrypted_content": "ASTRA"}, {"role": "assistant", "content": "a"}]
        out = self.codex(router, codex_request(history=history), turn="a2")
        self.assertIn("reasoning", [it.get("type") for it in out["request"]["input"]])

    def test_current_turn_reasoning_kept(self):
        router = self.router(jev("light", depth=0.1)[0])
        self.codex(router, codex_request(history=OLD_TURN), turn="t5")
        req = codex_request(history=OLD_TURN)
        req["input"] += [{"type": "reasoning", "encrypted_content": "LUNA-NOW"}, {"type": "function_call", "name": "x"}]
        out = self.codex(router, req, turn="t5")
        self.assertEqual([it["encrypted_content"] for it in out["request"]["input"] if it.get("type") == "reasoning"], ["LUNA-NOW"])

    def test_off_after_routing_strips(self):
        router = self.router(jev("light", depth=0.1)[0])
        self.codex(router, codex_request(), turn="r1")
        router.ctx.config["mode"] = "off"
        history = [{"role": "user", "content": "q"}, {"type": "reasoning", "encrypted_content": "LUNA"}, {"role": "assistant", "content": "a"}]
        out = self.codex(router, codex_request(history=history), turn="r2")
        self.assertNotIn("reasoning", [it.get("type") for it in out["request"]["input"]])
        self.assertEqual(out["request"]["model"], "gpt-6-sol")

    def test_astra_effort_clamped(self):
        router = self.router(mode="heavy")
        self.assertEqual(self.codex(router, codex_request(effort="none"))["request"]["reasoning"]["effort"], "low")


class FooterAndJournalTests(Base):
    def test_footer(self):
        router = self.router(jev("light", depth=0.1)[0])
        self.codex(router, codex_request(), turn="f1")
        router.on_post_api_request(turn_id="f1", session_id="S1", response_model="gpt-6-luna", usage={})
        out = router.on_transform_llm_output(response_text="Hello.", turn_id="f1")
        self.assertTrue(out.startswith("Hello.\n\n— Ответила: GPT-6 Luna · Джев: лёгкое задание"))

    def test_footer_mismatch_and_pinned(self):
        router = self.router(mode="heavy")
        self.codex(router, codex_request(), turn="f2")
        router.on_post_api_request(turn_id="f2", response_model="gpt-6-sol", usage={})
        self.assertIn("GPT-6 Sol (просили GPT-6 Astra) · закреплено: сложное", router.on_transform_llm_output(response_text="x", turn_id="f2"))

    def test_no_footer_when_disabled_or_unrouted(self):
        router = self.router(jev("light", depth=0.1)[0], show_model=False)
        self.codex(router, codex_request(), turn="f3")
        self.assertIsNone(router.on_transform_llm_output(response_text="x", turn_id="f3"))
        self.assertIsNone(self.router().on_transform_llm_output(response_text="x", turn_id="nope"))

    def test_check_and_stats(self):
        router = self.router(jev("heavy", depth=1.9)[0])
        self.codex(router, codex_request(), turn="v1")
        usage = {"input_tokens": 1000, "cache_read_tokens": 10000, "output_tokens": 500}
        router.on_post_api_request(turn_id="v1", session_id="S1", response_model="gpt-6-astra-20260903", usage=usage, api_duration=3.0)
        catalog = {"openai/gpt-6-astra": {"pricing": {"prompt": "0.00001", "completion": "0.00005", "input_cache_read": "0.000001"}},
                   "openai/gpt-6-sol": {"pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000002"}},
                   "openai/gpt-6-luna": {"pricing": {"prompt": "0.0000001", "completion": "0.0000005", "input_cache_read": "0.00000001"}}}
        with patch.object(mod, "_openrouter_catalog", return_value=catalog):
            row = router.verify(1)[0]
            stats = router.text_stats(1)
        self.assertTrue(row["verified"])
        self.assertAlmostEqual(row["answer_usd"], 1000 * 1e-5 + 10000 * 1e-6 + 500 * 5e-5)
        self.assertIn("сложное 1", stats)
        self.assertIn("GPT-6 Sol $", stats)

    def test_commands(self):
        router = self.router()
        self.assertIn("выключен", router.command("off"))
        self.assertEqual(router.mode(), "off")
        router.command("luna")
        self.assertEqual(router.mode(), "light")
        router.ctx.config["mode"] = True
        self.assertEqual(router.mode(), "on")


class MaxTierTests(Base):
    def test_heavy_capped_to_standard(self):
        router = self.router(jev("heavy", depth=1.9)[0], max_tier="standard")
        out = self.codex(router, codex_request(), turn="m1")
        self.assertIsNone(out)  # chat model is Sol, so nothing to rewrite
        row = router.decisions(1)[0]
        self.assertEqual((row["tier"], row["detail"]["capped_from"]), ("standard", "heavy"))
        router.on_post_api_request(turn_id="m1", response_model="gpt-6-sol", usage={})
        self.assertIn("сложное задание, ограничено до «среднее»", router.on_transform_llm_output(response_text="x", turn_id="m1"))

    def test_light_not_affected_and_alias(self):
        router = self.router(jev("light", depth=0.1)[0], max_tier="sol")
        self.assertEqual(self.codex(router, codex_request())["request"]["model"], "gpt-6-luna")

    def test_pinned_mode_ignores_cap(self):
        router = self.router(mode="heavy", max_tier="standard")
        self.assertEqual(self.codex(router, codex_request())["request"]["model"], "gpt-6-astra")

    def test_max_command(self):
        router = self.router()
        self.assertIn("не выше", router.command("max sol"))
        self.assertEqual(router.max_tier(), "standard")
        self.assertIn("снят", router.command("max off"))
        self.assertIsNone(router.max_tier())


if __name__ == "__main__":
    unittest.main()
