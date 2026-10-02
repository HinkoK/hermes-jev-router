#!/usr/bin/env python3
"""Стенд проверки роутера: гоняет набор через Джева, кэширует сырые ответы, считает метрики.

    python3 harness.py dev D1            # дизайн D1 на рабочем наборе
    python3 harness.py dev all           # все дизайны
    python3 harness.py holdout D2 --rule best
Ключ OpenRouter берётся из ~/.hermes/.env и не печатается.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "cache"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
JEV = "typesafe/jev-1.13"
TIERS = ("light", "standard", "heavy")
RANK = {t: i for i, t in enumerate(TIERS)}
# Цена ошибки: [правильная][выбранная]. Занизить дороже, чем завысить.
COST = {"light": {"light": 0, "standard": 0.5, "heavy": 1.0},
        "standard": {"light": 2.0, "standard": 0, "heavy": 1.0},
        "heavy": {"light": 5.0, "standard": 2.0, "heavy": 0}}


def key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if k:
        return k
    for line in (Path.home() / ".hermes/.env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY absent")


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in (HERE / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def post(payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(ENDPOINT, data=body, method="POST",
                                 headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            t = time.time()
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.load(r)
            out["_ms"] = int((time.time() - t) * 1000)
            return out
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


def ask(design, item: dict) -> dict:
    """Cached Jev call for one item under one design."""
    payload = design.payload(item)
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / design.name / f"{item['id']}-{digest}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = post(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def run(design, items: list[dict], workers: int = 8) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(lambda it: ask(design, it), items))
    return [{"item": it, "raw": ans} for it, ans in zip(items, answers)]


def metrics(rows: list[dict], decide) -> dict:
    """rows: [{"item", "raw"}]; decide(row) -> tier."""
    res = []
    for row in rows:
        it = row["item"]
        pred = decide(row)
        good = {it["label"], *it.get("ok", [])}
        res.append({"id": it["id"], "label": it["label"], "pred": pred, "correct": pred in good,
                    "under": pred not in good and RANK[pred] < RANK[it["label"]],
                    "over": pred not in good and RANK[pred] > RANK[it["label"]],
                    "severe": it["label"] == "heavy" and pred == "light",
                    "cost": 0 if pred in good else COST[it["label"]][pred], "trap": it.get("trap"),
                    "ms": row["raw"].get("_ms", 0), "usd": float((row["raw"].get("usage") or {}).get("cost") or 0)})
    n = len(res)
    by_trap = {}
    for trap in ("short_heavy", "long_light", "followup", None):
        sub = [r for r in res if r["trap"] == trap]
        if sub:
            by_trap[str(trap)] = f"{sum(r['correct'] for r in sub)}/{len(sub)}"
    conf = Counter((r["label"], r["pred"]) for r in res)
    ms = sorted(r["ms"] for r in res)
    return {"n": n, "accuracy": round(sum(r["correct"] for r in res) / n, 3),
            "under": sum(r["under"] for r in res), "over": sum(r["over"] for r in res),
            "severe_heavy_to_light": sum(r["severe"] for r in res), "weighted_cost": round(sum(r["cost"] for r in res), 1),
            "by_trap": by_trap, "confusion": {f"{a}->{b}": c for (a, b), c in sorted(conf.items())},
            "share": dict(Counter(r["pred"] for r in res)), "p50_ms": ms[n // 2], "p95_ms": ms[int(n * 0.95) - 1],
            "jev_usd_total": round(sum(r["usd"] for r in res), 6), "errors": [r for r in res if not r["correct"]]}


def show(name: str, m: dict, items: dict[str, dict] | None = None, errors: bool = True) -> None:
    print(f"\n=== {name}: точность {m['accuracy']:.1%} ({m['n']}), занизил {m['under']} (сложное→Luna {m['severe_heavy_to_light']}), "
          f"завысил {m['over']}, цена ошибок {m['weighted_cost']}")
    print(f"    ловушки {m['by_trap']} | доли {m['share']} | {m['p50_ms']}/{m['p95_ms']} мс p50/p95 | Джев ${m['jev_usd_total']}")
    print(f"    {m['confusion']}")
    if errors and items:
        for e in m["errors"]:
            msg = items[e["id"]]["message"].replace("\n", " ")
            print(f"      {e['id']} {e['label']}→{e['pred']} [{e['trap'] or ''}] {msg[:90]}")


if __name__ == "__main__":
    import designs  # noqa: E402

    dataset, which = sys.argv[1], sys.argv[2]
    items = load(dataset)
    index = {it["id"]: it for it in items}
    chosen = designs.ALL if which == "all" else [d for d in designs.ALL if d.name == which]
    for d in chosen:
        rows = run(d, items)
        for rule_name, rule in d.rules().items():
            show(f"{d.name}/{rule_name}", metrics(rows, rule), index, errors="--errors" in sys.argv)
