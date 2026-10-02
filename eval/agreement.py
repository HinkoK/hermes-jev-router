#!/usr/bin/env python3
"""Согласие двух разметчиков и точность роутера по каждому из них.
    python3 agreement.py dev        # dev.jsonl vs dev_relabel.jsonl
"""
import json
import sys

import designs
import harness

name = sys.argv[1]
a = {r["id"]: r for r in harness.load(name)}
b = {r["id"]: r for r in harness.load(f"{name}_relabel")}
strict = sum(a[i]["label"] == b[i]["label"] for i in a)
lenient = sum(bool({a[i]["label"], *a[i]["ok"]} & {b[i]["label"], *b[i]["ok"]}) for i in a)
print(f"{name}: метки совпали {strict}/{len(a)}, совместимы (с учётом допустимых) {lenient}/{len(a)}")
for i in a:
    if not ({a[i]["label"], *a[i]["ok"]} & {b[i]["label"], *b[i]["ok"]}):
        print(f"  расходятся {i}: автор {a[i]['label']} / второй {b[i]['label']} | {a[i]['message'][:80]!r}")
rows = harness.run(designs.PROD(), list(a.values()))
decide = designs.PROD().rules()["decide"]
by_b = [{"item": {**r["item"], "label": b[r["item"]["id"]]["label"], "ok": b[r["item"]["id"]]["ok"]}, "raw": r["raw"]} for r in rows]
both = [{"item": {**r["item"], "ok": list({*a[r['item']['id']]['ok'], b[r['item']['id']]['label'], *b[r['item']['id']]['ok']} - {a[r['item']['id']]['label']})}, "raw": r["raw"]} for r in rows]
for title, data in (("по автору набора", rows), ("по второму разметчику", by_b), ("верно, если согласен хоть один", both)):
    m = harness.metrics(data, decide)
    print(f"PROD {title}: точность {m['accuracy']:.1%}, занизил {m['under']} (сложное→лёгкое {m['severe_heavy_to_light']}), завысил {m['over']}")
