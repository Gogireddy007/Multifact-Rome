import json, os, sys
from collections import defaultdict

DATA = sys.argv[1] if len(sys.argv) > 1 else "counterfact.json"
OUT = "results/related_groups/pilot_groups.json"

PILOT = {
    "Japan": [1508, 21238], "Helsinki": [1304, 3182], "Kamchatka Krai": [2738, 18096],
    "Renault Trucks": [2130, 15759], "Sega Wow": [793, 11657], "Windows Vista": [3718, 15463],
    "iPod": [5837, 8947], "Google Drive": [335, 12146], "How I Met Your Mother": [700, 9037],
    "Orphan Black": [431, 1516], "The Decameron": [861, 4621], "Ma vie en rose": [9990, 13304],
    "Roy Hargrove": [299, 13300], "Andreas Ivanschitz": [11, 1221], "Lionel Messi": [6765, 9617],
    "Marie Curie": [10886, 14918], "Karl Marx": [3165, 11116], "Gregory XVI": [603, 18175],
    "Vladimir Putin": [8192, 11627], "Margaret Atwood": [11144, 16286],
}
SPARES = {
    "Brown University": [964, 20091], "Qing dynasty": [13949, 17921], "Gilad Atzmon": [44, 10924],
    "Cat Stevens": [3666, 13713], "Denver International Airport": [5015, 15695], "Ericsson": [2053, 9182],
}

data = json.load(open(DATA))
by_subject = defaultdict(list)
for r in data:
    rw = r["requested_rewrite"]
    by_subject[rw["subject"]].append((r["case_id"], rw["relation_id"]))

usable = sum(1 for recs in by_subject.values() if len({rel for _, rel in recs}) >= 2)

# sanity check: every picked case_id belongs to its subject, with different relations
for name, picks in {**PILOT, **SPARES}.items():
    found = dict(by_subject[name])
    assert all(c in found for c in picks), f"{name}: bad case_id"
    assert len({found[c] for c in picks}) == len(picks), f"{name}: duplicate relation"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({
    "pilot": [{"subject": s, "case_ids": c} for s, c in PILOT.items()],
    "spares": [{"subject": s, "case_ids": c} for s, c in SPARES.items()],
}, open(OUT, "w"), indent=2)

print(f"subjects: {len(by_subject)} | usable groups: {usable}")