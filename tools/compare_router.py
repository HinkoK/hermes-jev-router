#!/usr/bin/env python3
"""Выгоднее ли роутер: одни и те же сообщения без роутера (всё на Sol) и с роутером (Luna или Sol).

Прогон A: роутер выключен, каждое сообщение отвечает модель чата (GPT-6 Sol по подписке).
Прогон B: роутер включён с потолком «среднее» (`/router max standard`): Джев выбирает Luna или Sol, Astra не используется.

Для каждого прогона считает:
  - сколько процентов 5-часового окна и недели подписки ушло (`hermes usage`, только чтение счётчика);
  - токены: новый вход, вход из кэша, выход;
  - сколько это стоило бы по API (цены тех же моделей на OpenRouter);
  - время.
Ответы обоих прогонов сохраняются рядом в tools/compare_results/<время>.md, чтобы сравнить качество.

Режим разговора:
  --conversation chat   (по умолчанию) все сообщения прогона идут в одну переписку, как в живом чате;
                        так видно и то, что смена модели сбрасывает кэш начала переписки;
  --conversation fresh  каждое сообщение в новой сессии.

Сообщения: лёгкие и средние из eval/dev.jsonl (без продолжений разговора), поровну, вперемешку.

    python3 tools/compare_router.py --dry-run          # счётчик, список сообщений и прогноз Джева (Джев стоит копейки, лимит не тратится)
    python3 tools/compare_router.py                    # 8 лёгких + 8 средних, оба прогона
    python3 tools/compare_router.py --light 12 --standard 12 --conversation fresh

Счётчик подписки показывает целые проценты и обновляется с задержкой (скрипт ждёт, пока он успокоится).
16 сообщений — это примерно 2% 5-часового окна, поэтому разница меньше 2 процентов — шум.
Для надёжного ответа по лимиту: --light 20 --standard 20, и повторить с --order ba.
Точное сравнение — по токенам и цене «как если бы по API». Режим роутера и потолок после теста возвращаются как были.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "compare_results"
LOGS = Path.home() / ".hermes" / "logs"
LOG_LINE = re.compile(
    r"^(\S+ \S+),\d+ \w+ \[([^\]]+)\] agent\.conversation_loop: API call #(\d+): model=(\S+) provider=(\S+) "
    r"in=(\d+) out=(\d+) total=\d+ latency=[\d.]+s(?: cache=(\d+)/\d+ \([^)]*\))?")
PRICE_IDS = {"gpt-6-luna": "openai/gpt-6-luna", "gpt-6-sol": "openai/gpt-6-sol", "gpt-6-astra": "openai/gpt-6-astra"}
TIER_MODEL = {"light": "gpt-6-luna", "standard": "gpt-6-sol"}


def sh(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["hermes", *args], capture_output=True, text=True, timeout=timeout)


def usage() -> dict:
    out = sh("usage", "--json", timeout=60)
    if out.returncode != 0:
        raise SystemExit("Не удалось прочитать лимиты (hermes usage).")
    data = json.loads(out.stdout)
    if data.get("provider") != "openai-codex":
        raise SystemExit(f"Провайдер {data.get('provider')}: скрипт сравнивает только на подписке ChatGPT (openai-codex).")
    win = {w.get("label", "").lower(): w for w in data.get("windows") or []}
    return {"session": float(win["session"]["used_percent"]), "weekly": float(win["weekly"]["used_percent"]),
            "session_reset": win["session"].get("resets_at"), "plan": data.get("plan")}


def router_state() -> dict:
    out = sh("jev-router", "status", timeout=60).stdout
    mode = re.search(r"Роутер: (\S+)", out)
    if not mode:
        raise SystemExit("Плагин jev-router не отвечает: `hermes plugins enable jev-router`.")
    cap = re.search(r"Потолок: не выше уровня «([^»]+)»", out)
    journal = re.search(r"Журнал: (\S+)", out)
    chat_model = re.search(r"Модель чата: (\S+) \((\S+)\)", out)
    ru = {"лёгкое": "light", "среднее": "standard", "сложное": "heavy"}
    return {"mode": mode.group(1), "max_tier": ru.get(cap.group(1)) if cap else None,
            "journal": Path(journal.group(1)) if journal else None,
            "chat_model": chat_model.group(1) if chat_model else None}


def pick_messages(n_light: int, n_standard: int) -> list[dict]:
    rows = [json.loads(x) for x in (ROOT / "eval" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    single = [r for r in rows if not r.get("context")]

    def spread(label: str, n: int) -> list[dict]:
        pool = [r for r in single if r["label"] == label]
        if n >= len(pool):
            return pool
        step = len(pool) / n
        return [pool[int(i * step)] for i in range(n)]
    light, standard = spread("light", n_light), spread("standard", n_standard)
    mixed = []
    for i in range(max(len(light), len(standard))):
        mixed += light[i:i + 1] + standard[i:i + 1]
    return mixed


MAX_TURNS = "4"  # like normal use: the agent may call a tool (e.g. web search) before answering


def ask(text: str, session: str | None) -> tuple[str, str | None, float, str | None]:
    """(answer, session_id, seconds, error). Hermes prints session_id on stderr in --quiet mode."""
    args = ["chat", "-q", text, "--quiet", "--ignore-rules", "--source", "tool", "--max-turns", MAX_TURNS, "-t", "safe"]
    if session:
        args += ["--resume", session]
    t = time.time()
    out = sh(*args)
    both = f"{out.stdout}\n{out.stderr}"
    sid = re.findall(r"session_id:\s*(\S+)", both)
    answer = re.sub(r"\n*session_id:\s*\S+\s*", "\n", out.stdout).strip()
    error = None
    if out.returncode != 0:
        tail = " ".join(out.stderr.strip().splitlines()[-2:])[:200]
        error = f"код {out.returncode}: {tail}"
    return answer, sid[-1] if sid else session, time.time() - t, error


def window_reset(start: dict, now: dict) -> bool:
    """The server's reset time jitters by a second between reads; a real reset moves it by hours or drops the counter."""
    if now["session"] < start["session"]:
        return True
    try:
        a = datetime.fromisoformat(str(start["session_reset"]))
        b = datetime.fromisoformat(str(now["session_reset"]))
        return abs((b - a).total_seconds()) > 600
    except (TypeError, ValueError):
        return False


def settled_usage(wait: int) -> dict:
    """The subscription counter lags a little: read until two readings `wait` seconds apart agree."""
    prev = usage()
    for _ in range(4):
        time.sleep(wait)
        cur = usage()
        if (cur["session"], cur["weekly"]) == (prev["session"], prev["weekly"]):
            return cur
        prev = cur
    return prev


def log_tokens(sessions: set[str], since: float) -> list[dict]:
    """Per API call tokens from Hermes' agent.log (model there is the chat model, not the routed one)."""
    calls, seen = [], set()
    for path in sorted(LOGS.glob("agent.log*"), key=lambda p: p.stat().st_mtime):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LOG_LINE.match(line)
            if not m or m.group(2) not in sessions:
                continue
            ts = time.mktime(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timetuple())
            key = (m.group(2), m.group(3), m.group(1))
            if ts < since - 5 or key in seen:
                continue
            seen.add(key)
            total_in, cache = int(m.group(6)), int(m.group(8) or 0)
            calls.append({"session": m.group(2), "model": m.group(4), "fresh_in": max(0, total_in - cache),
                          "cache_in": cache, "out": int(m.group(7))})
    return calls


def journal_rows(journal: Path | None, sessions: set[str], since: float) -> list[dict]:
    if not journal or not journal.is_file():
        return []
    rows = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("session_id") in sessions and float(row.get("t") or 0) >= since - 5:
            rows.append(row)
    return rows


def prices() -> dict:
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20) as r:
            rows = {row["id"]: row.get("pricing", {}) for row in json.load(r)["data"]}
        return {m: rows.get(pid, {}) for m, pid in PRICE_IDS.items()}
    except Exception:  # noqa: BLE001
        return {}


def cost(p: dict, fresh: int, cache: int, out: int) -> float | None:
    if not p:
        return None
    pin, pout = float(p.get("prompt", 0)), float(p.get("completion", 0))
    pread = float(p.get("input_cache_read") or pin)
    return fresh * pin + cache * pread + out * pout


def run(name: str, messages: list[dict], conversation: str, stop_at: float, journal: Path | None, settle: int) -> dict:
    start = settled_usage(settle)
    print(f"\n== Прогон {name}: окно {start['session']:.0f}%, неделя {start['weekly']:.0f}%", flush=True)
    t0, session, sessions, answers, note = time.time(), None, set(), [], None
    for i, item in enumerate(messages, 1):
        now = usage() if i > 1 else start
        if now["session"] >= stop_at:
            note = f"остановлено перед сообщением {i}: окно {now['session']:.0f}%"
            break
        if window_reset(start, now):
            note = "окно лимита сбросилось во время прогона, результат недействителен"
            break
        try:
            answer, sid, secs, error = ask(item["message"], session if conversation == "chat" else None)
        except subprocess.TimeoutExpired:
            answer, sid, secs, error = "", session, 600.0, "таймаут 10 минут"
        if sid:
            sessions.add(sid)
            session = sid
        answers.append({"id": item["id"], "label": item["label"], "answer": answer, "seconds": round(secs, 1),
                        "session": sid, "error": error})
        print(f"   {i}/{len(messages)} {item['id']} ({item['label']}) {secs:.0f} с" + (f"  ОШИБКА {error}" if error else ""), flush=True)
    end = settled_usage(settle)
    if window_reset(start, end):
        note = note or "окно лимита сбросилось во время прогона, результат недействителен"
    return {"name": name, "start": start, "end": end, "seconds": round(time.time() - t0), "answers": answers,
            "sessions": sorted(sessions), "note": note, "since": t0}


def summarize(res: dict, journal: Path | None, table: dict) -> dict:
    sessions = set(res["sessions"])
    calls = log_tokens(sessions, res["since"])
    rows = journal_rows(journal, sessions, res["since"])
    responses = [r for r in rows if r.get("event") == "response"]
    by_model: dict[str, dict] = {}
    if responses:  # routed run: attribute tokens to the model the server named
        for r in responses:
            u = r.get("usage") or {}
            model = next((m for m in PRICE_IDS if str(r.get("response_model", "")).startswith(m)), str(r.get("response_model")))
            b = by_model.setdefault(model, {"calls": 0, "fresh_in": 0, "cache_in": 0, "out": 0})
            b["calls"] += 1
            b["fresh_in"] += int(u.get("input_tokens") or 0)
            b["cache_in"] += int(u.get("cache_read_tokens") or 0)
            b["out"] += int(u.get("output_tokens") or 0)
    else:
        for c in calls:
            b = by_model.setdefault(c["model"], {"calls": 0, "fresh_in": 0, "cache_in": 0, "out": 0})
            b["calls"] += 1
            for k in ("fresh_in", "cache_in", "out"):
                b[k] += c[k]
    usd = 0.0
    for model, b in by_model.items():
        b["usd"] = cost(table.get(model, {}), b["fresh_in"], b["cache_in"], b["out"])
        usd += b["usd"] or 0
    tiers = {}
    for r in rows:
        if r.get("event") == "route":
            tiers[r.get("tier")] = tiers.get(r.get("tier"), 0) + 1
    jev = sum(float(r.get("jev_cost_usd") or 0) for r in rows if r.get("event") == "route")
    return {"window": res["end"]["session"] - res["start"]["session"], "weekly": res["end"]["weekly"] - res["start"]["weekly"],
            "by_model": by_model, "usd": usd, "jev_usd": jev, "tiers": tiers,
            "messages": len(res["answers"]), "errors": sum(1 for a in res["answers"] if a.get("error")),
            "seconds": res["seconds"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--light", type=int, default=8, help="сколько лёгких сообщений (по умолчанию 8)")
    ap.add_argument("--standard", type=int, default=8, help="сколько средних сообщений (по умолчанию 8)")
    ap.add_argument("--conversation", choices=("chat", "fresh"), default="chat")
    ap.add_argument("--order", choices=("ab", "ba"), default="ab", help="какой прогон первым")
    ap.add_argument("--stop-at", type=float, default=92.0, help="не слать, если окно использовано на столько %%")
    ap.add_argument("--settle", type=int, default=45, help="секунд ждать, пока счётчик подписки обновится (по умолчанию 45)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    messages = pick_messages(args.light, args.standard)
    before, state = usage(), router_state()
    if state["chat_model"] != "gpt-6-sol":
        raise SystemExit(f"Модель чата {state['chat_model']}; для сравнения нужна gpt-6-sol по подписке.")
    print(f"Подписка {before['plan']}: окно {before['session']:.0f}% (сброс {before['session_reset']}), неделя {before['weekly']:.0f}%.")
    print(f"Роутер сейчас: {state['mode']}, потолок: {state['max_tier'] or 'нет'}.")
    print(f"Сообщений: {len(messages)} ({args.light} лёгких + {args.standard} средних), разговор: {args.conversation}. "
          f"Всего запросов к модели: {2 * len(messages)} (по одному на сообщение в каждом прогоне).")
    if args.dry_run:
        print("\nПрогноз роутера для прогона B (решает Джев; сложное ограничено до Sol):")
        luna = 0
        for m in messages:
            out = sh("jev-router", "route", m["message"], timeout=60).stdout.strip().splitlines()
            tier = json.loads(out[-1]).get("tier") if out else None
            model = "Luna" if tier == "light" else "Sol"
            luna += model == "Luna"
            print(f"   {m['id']} ({m['label']}) → {model}  | {m['message'][:70].replace(chr(10), ' ')!r}")
        print(f"В прогоне B на Luna уйдут {luna} из {len(messages)} сообщений, остальные на Sol.")
        return

    table, results = prices(), {}
    try:
        for step in args.order:
            if step == "a":
                sh("jev-router", "off", timeout=60)
                results["A"] = run("A (без роутера, всё Sol)", messages, args.conversation, args.stop_at, state["journal"], args.settle)
            else:
                sh("jev-router", "max", "standard", timeout=60)
                sh("jev-router", "on", timeout=60)
                results["B"] = run("B (роутер: Luna или Sol)", messages, args.conversation, args.stop_at, state["journal"], args.settle)
    except KeyboardInterrupt:
        print("\nПрервано.")
    finally:
        sh("jev-router", "max", state["max_tier"] or "off", timeout=60)
        sh("jev-router", state["mode"], timeout=60)
        print(f"\nРоутер возвращён: режим {state['mode']}, потолок {state['max_tier'] or 'нет'}.")

    summary = {k: summarize(v, state["journal"], table) for k, v in results.items()}
    print("\nпрогон | сообщений | окно, % | неделя, % | по API, $ | Джев, $ | время | по моделям (вызовов, токены новый/кэш/выход)")
    for k in ("A", "B"):
        if k not in summary:
            continue
        s, r = summary[k], results[k]
        models = "; ".join(f"{m}: {b['calls']} выз., {b['fresh_in']}/{b['cache_in']}/{b['out']}" for m, b in s["by_model"].items())
        errs = f" | ошибок: {s['errors']}" if s["errors"] else ""
        print(f"{r['name']} | {s['messages']} | +{s['window']:.0f} | +{s['weekly']:.0f} | ${s['usd']:.4f} | ${s['jev_usd']:.5f} | "
              f"{s['seconds']} с | {models}{errs}" + (f" | {r['note']}" if r["note"] else ""))
    if "A" in summary and "B" in summary:
        a, b = summary["A"], summary["B"]
        print(f"\nРешения роутера в B: {b['tiers']}")
        complete = (a["messages"] == b["messages"] == len(messages) and not a["errors"] and not b["errors"]
                    and not any(r["note"] for r in results.values()))
        if not complete:
            print(f"Сравнение не делаю: прогоны неполные (A: {a['messages']} из {len(messages)}, ошибок {a['errors']}; "
                  f"B: {b['messages']} из {len(messages)}, ошибок {b['errors']}). Причина — в последней колонке таблицы.")
        else:
            per_a, per_b = a["usd"] / a["messages"], b["usd"] / b["messages"]
            if a["usd"] > 0:
                change = 100 * (1 - b["usd"] / a["usd"])
                word = "дешевле" if change >= 0 else "дороже"
                print(f"По цене API прогон B {word} на {abs(change):.0f}%: ${a['usd']:.4f} → ${b['usd']:.4f} "
                      f"(в среднем ${per_a:.4f} → ${per_b:.4f} за ответ).")
            diff = a["window"] - b["window"]
            if abs(diff) < 2:
                print(f"По лимиту подписки: A +{a['window']:.0f}%, B +{b['window']:.0f}% окна — разница в пределах погрешности "
                      "(счётчик показывает целые проценты). Чтобы её увидеть, нужно больше сообщений.")
            else:
                word = "сэкономил" if diff > 0 else "потратил больше на"
                print(f"По лимиту подписки роутер {word} {abs(diff):.0f}% окна ({a['window']:.0f}% → {b['window']:.0f}%, погрешность ±2).")

    RESULTS.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    (RESULTS / f"{stamp}.json").write_text(json.dumps({"args": vars(args), "summary": summary,
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "since"} for k, v in results.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Сравнение ответов {stamp}", ""]
    ans = {k: {x["id"]: x for x in v["answers"]} for k, v in results.items()}
    for m in messages:
        md += [f"## {m['id']} ({m['label']})", "", "> " + m["message"][:500].replace("\n", "\n> "), ""]
        for k in ("A", "B"):
            if m["id"] in ans.get(k, {}):
                a_ = ans[k][m["id"]]
                md += [f"**{results[k]['name']}** ({a_['seconds']} с){' — ОШИБКА ' + a_['error'] if a_.get('error') else ''}:", "", a_["answer"], ""]
    (RESULTS / f"{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Цифры: {RESULTS / (stamp + '.json')}\nОтветы рядом для сравнения качества: {RESULTS / (stamp + '.md')}")


if __name__ == "__main__":
    main()
