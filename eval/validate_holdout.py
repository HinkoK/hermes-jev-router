import json
from pathlib import Path
import re
import sys
from collections import Counter

PATH = str(Path(__file__).with_name("holdout.jsonl"))

LABELS = {"light", "standard", "heavy"}
DOMAINS = {"life", "health", "money", "travel", "relationships", "study", "office", "content",
           "marketing", "business", "code", "devops", "data", "creative", "science_math",
           "legal_tax", "home", "ai_agents"}
TRAPS = {None, "short_heavy", "long_light", "followup"}
KEYS = ["id", "context", "message", "label", "ok", "domain", "trap", "why"]
BANNED = ["привет, ты тут?", "спасибо, получил, посмотрю вечером", "aiogram", "барбершоп",
          "barbershop", "каждое утро перестаёт", "каждое утро перестает"]
CYR = re.compile(r"[А-Яа-яЁё]")

errors = []
rows = []
with open(PATH, encoding="utf-8") as f:
    lines = f.read().split("\n")
if lines and lines[-1] == "":
    lines = lines[:-1]
if len(lines) != 120:
    errors.append(f"line count {len(lines)} != 120")

for n, line in enumerate(lines, 1):
    try:
        r = json.loads(line)
    except Exception as e:
        errors.append(f"line {n}: bad json {e}")
        continue
    rows.append(r)
    if list(r.keys()) != KEYS:
        errors.append(f"line {n}: keys {list(r.keys())}")
    if r.get("id") != f"h{n:03d}":
        errors.append(f"line {n}: id {r.get('id')}")
    if r["label"] not in LABELS:
        errors.append(f"{r['id']}: label {r['label']}")
    if not isinstance(r["ok"], list) or any(x not in LABELS or x == r["label"] for x in r["ok"]):
        errors.append(f"{r['id']}: ok {r['ok']}")
    if r["domain"] not in DOMAINS:
        errors.append(f"{r['id']}: domain {r['domain']}")
    if r["trap"] not in TRAPS:
        errors.append(f"{r['id']}: trap {r['trap']}")
    if not isinstance(r["message"], str) or not r["message"].strip():
        errors.append(f"{r['id']}: empty message")
    if not isinstance(r["why"], str) or not r["why"].strip() or not CYR.search(r["why"]):
        errors.append(f"{r['id']}: why must be Russian")
    ctx = r["context"]
    if not isinstance(ctx, list):
        errors.append(f"{r['id']}: context not list")
        ctx = []
    for t in ctx:
        if set(t.keys()) != {"role", "text"} or t["role"] not in {"user", "assistant"}:
            errors.append(f"{r['id']}: bad turn {t}")
        if t.get("role") == "assistant" and len(t.get("text", "")) > 300:
            errors.append(f"{r['id']}: assistant turn > 300 chars")
    if (r["trap"] == "followup") != bool(ctx):
        errors.append(f"{r['id']}: followup/context mismatch")
    if r["trap"] == "short_heavy":
        if r["label"] != "heavy":
            errors.append(f"{r['id']}: short_heavy not heavy")
        if len(r["message"].split()) > 30:
            errors.append(f"{r['id']}: short_heavy too long ({len(r['message'].split())} words)")
    if r["trap"] == "long_light":
        if r["label"] != "light":
            errors.append(f"{r['id']}: long_light not light")
        if len(r["message"].split()) < 130:
            errors.append(f"{r['id']}: long_light too short ({len(r['message'].split())} words)")
    low = r["message"].lower()
    for b in BANNED:
        if b in low:
            errors.append(f"{r['id']}: banned phrase '{b}'")
    if ("10 мин" in low or "10-мин" in low) and "видео" in low and "бот" in low:
        errors.append(f"{r['id']}: near-duplicate of banned 10-minute video about bots")

msgs = [r["message"].strip().lower() for r in rows]
dups = [m for m, c in Counter(msgs).items() if c > 1]
if dups:
    errors.append(f"duplicate messages: {dups}")
ids = [r["id"] for r in rows]
if len(set(ids)) != len(ids):
    errors.append("duplicate ids")

lang = Counter("ru" if CYR.search(r["message"]) else "en" for r in rows)
labels = Counter(r["label"] for r in rows)
traps = Counter(str(r["trap"]) for r in rows)
domains = Counter(r["domain"] for r in rows)
ok_n = sum(1 for r in rows if r["ok"])
fu_labels = Counter(r["label"] for r in rows if r["trap"] == "followup")
ll_words = [len(r["message"].split()) for r in rows if r["trap"] == "long_light"]
dom_label = {d: Counter(r["label"] for r in rows if r["domain"] == d) for d in sorted(domains)}

for d, c in domains.items():
    if c < 4:
        errors.append(f"domain {d} has {c} < 4")
for d in DOMAINS:
    if d not in domains:
        errors.append(f"domain {d} missing")
if traps["short_heavy"] < 12:
    errors.append("short_heavy < 12")
if traps["long_light"] < 10:
    errors.append("long_light < 10")
if traps["followup"] < 24:
    errors.append("followup < 24")

print("rows:", len(rows))
print("labels:", dict(labels))
print("traps:", dict(traps))
print("followup labels:", dict(fu_labels))
print("lang:", dict(lang))
print("items with ok:", ok_n)
print("long_light word counts (min/max):", min(ll_words), max(ll_words))
print("domains:")
for d in sorted(domains):
    c = dom_label[d]
    print(f"  {d:14s} {domains[d]:2d}  L{c['light']} S{c['standard']} H{c['heavy']}")
print("heavy domains:", sum(1 for d in dom_label if dom_label[d]["heavy"]), "/ 18;",
      "light domains:", sum(1 for d in dom_label if dom_label[d]["light"]), "/ 18")

if errors:
    print("\nERRORS:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
print("\nVALID")
