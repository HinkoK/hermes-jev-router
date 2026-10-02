#!/usr/bin/env python3
"""Сколько лимита подписки ChatGPT (Codex) тратит один ответ Luna, Sol и Astra.

Как работает:
  1. Читает счётчик лимитов (`hermes usage --json`, это только чтение).
  2. Закрепляет роутер за уровнем (`hermes jev-router light` и т.д.).
  3. Шлёт один и тот же короткий запрос через `hermes chat -q`, каждый раз в новой сессии,
     и после каждого запроса перечитывает счётчик.
  4. Серия на уровне кончается, когда 5-часовое окно сдвинулось на --target процентов
     или отправлено --max запросов.
  5. Печатает, сколько процентов окна стоит один запрос, и сколько токенов ушло (из журнала роутера).
В конце роутер возвращается в режим, который был до запуска (даже при Ctrl+C).

Счётчик подписки показывает целые проценты, поэтому результат — диапазон, а не точное число.
Расход ограничен: примерно --target процентов 5-часового окна на каждый уровень (+1 из-за округления).

    python3 tools/measure_limits.py --dry-run                 # только показать счётчик и план
    python3 tools/measure_limits.py                           # Luna, потом Astra
    python3 tools/measure_limits.py --tiers light,standard,heavy --target 3 --max 30
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROMPT = "Дай 5 коротких советов, как лучше высыпаться. Каждый совет — одно предложение, без вступления."
NAMES = {"light": "Luna", "standard": "Sol", "heavy": "Astra"}
RESULTS = Path(__file__).resolve().parent / "limit_results"


def sh(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["hermes", *args], capture_output=True, text=True, timeout=timeout)


def usage() -> dict:
    """{'session': float, 'weekly': float, 'session_reset': str, 'plan': str} — only reads the counter."""
    out = sh("usage", "--json", timeout=60)
    if out.returncode != 0:
        raise SystemExit("Не удалось прочитать лимиты (hermes usage). Основная модель должна быть на подписке ChatGPT.")
    data = json.loads(out.stdout)
    if data.get("provider") != "openai-codex":
        raise SystemExit(f"Провайдер {data.get('provider')}: скрипт меряет только подписку ChatGPT (openai-codex).")
    win = {w.get("label", "").lower(): w for w in data.get("windows") or []}
    s, w = win.get("session") or {}, win.get("weekly") or {}
    return {"session": float(s.get("used_percent")), "weekly": float(w.get("used_percent")),
            "session_reset": s.get("resets_at"), "plan": data.get("plan")}


def router_status() -> tuple[str, Path | None]:
    out = sh("jev-router", "status", timeout=60).stdout
    mode = re.search(r"Роутер: (\S+)", out)
    journal = re.search(r"Журнал: (\S+)", out)
    if not mode:
        raise SystemExit("Плагин jev-router не отвечает: `hermes plugins enable jev-router`.")
    return mode.group(1), Path(journal.group(1)) if journal else None


def tokens_for(journal: Path | None, sessions: set[str]) -> dict:
    total = {"requests": 0, "input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "models": set()}
    if not journal or not journal.is_file():
        return total
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") != "response" or row.get("session_id") not in sessions:
            continue
        total["requests"] += 1
        total["models"].add(str(row.get("response_model")))
        for k in ("input_tokens", "cache_read_tokens", "output_tokens", "reasoning_tokens"):
            total[k] += int((row.get("usage") or {}).get(k) or 0)
    return total


def series(tier: str, target: float, max_requests: int, stop_at: float, journal: Path | None) -> dict:
    set_mode = sh("jev-router", tier, timeout=60)
    if set_mode.returncode != 0:
        raise SystemExit(f"Не удалось закрепить уровень {tier}")
    start = usage()
    if start["session"] >= stop_at:
        return {"tier": tier, "skipped": f"окно уже использовано на {start['session']:.0f}% (порог {stop_at:.0f}%)"}
    print(f"\n== {NAMES[tier]}: окно {start['session']:.0f}%, неделя {start['weekly']:.0f}%", flush=True)
    sessions, sent, now, invalid = set(), 0, start, None
    t0 = time.time()
    while sent < max_requests:
        if now["session"] >= stop_at:
            invalid = f"остановлено: окно дошло до {now['session']:.0f}%"
            break
        out = sh("chat", "-q", PROMPT, "--quiet", "--ignore-rules", "--source", "tool", "--max-turns", "1", "-t", "safe")
        if out.returncode != 0:
            invalid = f"запрос {sent + 1} завершился с ошибкой (код {out.returncode})"
            break
        sid = re.search(r"session_id:\s*(\S+)", out.stdout)
        if sid:
            sessions.add(sid.group(1))
        sent += 1
        now = usage()
        if now["session_reset"] != start["session_reset"] or now["session"] < start["session"]:
            invalid = "окно лимита сбросилось во время серии, результат недействителен"
            break
        delta = now["session"] - start["session"]
        print(f"   запрос {sent}: окно {now['session']:.0f}% (+{delta:.0f})", flush=True)
        if delta >= target:
            break
    delta_s, delta_w = now["session"] - start["session"], now["weekly"] - start["weekly"]
    # Both readings are rounded to whole percents, so the true spend lies within ±1 of the difference.
    lo, hi = max(0.0, delta_s - 1), delta_s + 1
    tok = tokens_for(journal, sessions)
    return {"tier": tier, "model": NAMES[tier], "requests": sent, "seconds": round(time.time() - t0),
            "session_before": start["session"], "session_after": now["session"], "session_delta": delta_s,
            "weekly_delta": delta_w, "per_request_low": round(lo / sent, 3) if sent else None,
            "per_request_high": round(hi / sent, 3) if sent else None, "invalid": invalid,
            "served_models": sorted(tok["models"]), "tokens": {k: v for k, v in tok.items() if k != "models"}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", default="light,heavy", help="light,standard,heavy (по умолчанию light,heavy)")
    ap.add_argument("--target", type=float, default=3.0, help="на сколько процентов окна сдвинуться за серию (по умолчанию 3)")
    ap.add_argument("--max", type=int, default=30, help="максимум запросов в серии (по умолчанию 30)")
    ap.add_argument("--stop-at", type=float, default=92.0, help="не слать, если окно использовано на столько %% (по умолчанию 92)")
    ap.add_argument("--dry-run", action="store_true", help="только показать счётчик и план, ничего не тратить")
    args = ap.parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if any(t not in NAMES for t in tiers):
        raise SystemExit("--tiers: только light, standard, heavy")

    before = usage()
    mode, journal = router_status()
    print(f"Подписка {before['plan']}: 5-часовое окно использовано на {before['session']:.0f}% "
          f"(сброс {before['session_reset']}), неделя {before['weekly']:.0f}%. Роутер сейчас: {mode}.")
    print(f"План: {', '.join(NAMES[t] for t in tiers)}; в каждой серии до {args.max} запросов или пока окно не сдвинется на "
          f"{args.target:.0f}%. Ожидаемый расход: около {(args.target + 1) * len(tiers):.0f}% 5-часового окна.")
    if before["session"] + (args.target + 1) * len(tiers) > args.stop_at:
        print(f"Внимание: окна может не хватить (порог {args.stop_at:.0f}%). Лучше запустить после сброса окна.")
    if args.dry_run:
        return

    results = []
    try:
        for tier in tiers:
            results.append(series(tier, args.target, args.max, args.stop_at, journal))
    except KeyboardInterrupt:
        print("\nПрервано.")
    finally:
        sh("jev-router", mode, timeout=60)
        print(f"Роутер возвращён в режим: {mode}")

    print("\nмодель | запросов | окно, % | неделя, % | % окна на запрос | токены вход/кэш/выход (из них рассуждения) | ответила")
    for r in results:
        if r.get("skipped"):
            print(f"{NAMES[r['tier']]} | пропущено: {r['skipped']}")
            continue
        t = r["tokens"]
        per = f"{r['per_request_low']:.2f}–{r['per_request_high']:.2f}" if r["requests"] else "—"
        print(f"{r['model']} | {r['requests']} | {r['session_before']:.0f}→{r['session_after']:.0f} (+{r['session_delta']:.0f}) | "
              f"+{r['weekly_delta']:.0f} | {per} | {t['input_tokens']}/{t['cache_read_tokens']}/{t['output_tokens']} ({t['reasoning_tokens']}) | "
              f"{', '.join(r['served_models']) or '?'}" + (f" | {r['invalid']}" if r["invalid"] else ""))
    ok = [r for r in results if not r.get("skipped") and not r["invalid"] and r["requests"] and r["session_delta"] > 0]
    if len(ok) >= 2:
        cheap, dear = ok[0], ok[-1]
        mid_c = (cheap["per_request_low"] + cheap["per_request_high"]) / 2
        mid_d = (dear["per_request_low"] + dear["per_request_high"]) / 2
        if mid_c > 0:
            print(f"\nОдин ответ {dear['model']} тратит примерно в {mid_d / mid_c:.1f} раза больше лимита, чем {cheap['model']} "
                  f"(грубая оценка из-за округления счётчика до целых процентов).")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"prompt": PROMPT, "plan": before["plan"], "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Результат сохранён: {path}")


if __name__ == "__main__":
    main()
