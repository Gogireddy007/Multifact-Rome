"""Build groups_v2.json: multi-edit groups from CounterFact with relation-similarity labels.

Usage: python3 scripts/build_groups_v2.py /path/to/counterfact.json
Reads only the dataset and the pilot file; writes results/related_groups/groups_v2.json.
"""
import hashlib, itertools, json, os, random, sys
from collections import defaultdict

DATA = sys.argv[1] if len(sys.argv) > 1 else "counterfact.json"
OUT = "results/related_groups/groups_v2.json"
PILOT = "results/related_groups/pilot_groups.json"
SEED = 32
N_PAIRS = 120            # 2-edit groups (plan asks for >= 100)
N_TRIPLES = 12           # 3-edit groups
# 4-edit groups: take every available subject (only a handful exist)

# Fine cluster: same cluster => "similar". Domain: same domain, different cluster => "related".
# Everything else => "distinct".
CLUSTER = {
    "language": ["P1412", "P103", "P37", "P364", "P407"],
    "geo_container": ["P30", "P17", "P131", "P276", "P36", "P190", "P159"],
    "life_place": ["P19", "P20", "P937"],
    "creation_place": ["P495", "P740"],
    "citizenship": ["P27"],
    "creator": ["P176", "P178", "P127"],
    "affiliation": ["P108", "P463"],
    "occupation": ["P106", "P101", "P39", "P413", "P1303", "P136", "P641"],
    "release": ["P449"], "religion": ["P140"], "naming": ["P138"], "label": ["P264"],
}
DOMAIN = {
    "language": "language",
    "geo_container": "geography", "life_place": "geography",
    "creation_place": "geography", "citizenship": "geography",
    "creator": "creation", "affiliation": "creation",
    "occupation": "occupation",
    "release": "release", "religion": "religion", "naming": "naming", "label": "label",
}
REL2CLUSTER = {rel: c for c, rels in CLUSTER.items() for rel in rels}


def label(r1, r2):
    c1, c2 = REL2CLUSTER[r1], REL2CLUSTER[r2]
    if c1 == c2:
        return "similar"
    return "related" if DOMAIN[c1] == DOMAIN[c2] else "distinct"


raw = open(DATA, "rb").read()
data = json.loads(raw)
rng = random.Random(SEED)

by_subject = defaultdict(lambda: defaultdict(list))   # subject -> relation -> [case_ids]
case_info = {}
for r in data:
    rw = r["requested_rewrite"]
    by_subject[rw["subject"]][rw["relation_id"]].append(r["case_id"])
    case_info[r["case_id"]] = (rw["subject"], rw["relation_id"])

pilot = json.load(open(PILOT))
pilot_subjects = {g["subject"] for g in pilot["pilot"]}
pilot_ids = {c for g in pilot["pilot"] for c in g["case_ids"]}
spare_subjects = {g["subject"] for g in pilot["spares"]}

unknown = {r for s in by_subject.values() for r in s} - set(REL2CLUSTER)
assert not unknown, f"unlabelled relations: {unknown}"

eligible = sorted(s for s, rels in by_subject.items()
                  if len(rels) >= 2 and s not in pilot_subjects)


def make_group(subject, rels):
    rels = list(rels)
    cases = [(rng.choice(sorted(by_subject[subject][rel])), rel) for rel in rels]
    pairs = [{"a": i, "b": j, "label": label(cases[i][1], cases[j][1])}
             for i, j in itertools.combinations(range(len(cases)), 2)]
    return {"subject": subject, "size": len(cases),
            "case_ids": [c for c, _ in cases], "relation_ids": [r for _, r in cases],
            "pair_labels": pairs}


used = set()
# --- 3- and 4-edit groups first (scarce), then 2-edit groups from the remaining subjects ---
def pick(k, limit):
    cands = [s for s in eligible if len(by_subject[s]) >= k and s not in used]
    rng.shuffle(cands)
    out = []
    for s in cands[:limit]:
        used.add(s)
        rels = rng.sample(sorted(by_subject[s]), k)
        out.append(make_group(s, rels))
    return out

groups4 = pick(4, 10 ** 6)
groups3 = pick(3, N_TRIPLES)

# --- 2-edit groups, balanced across similar / related / distinct ---
want = {"similar": N_PAIRS // 3, "related": N_PAIRS // 3, "distinct": N_PAIRS - 2 * (N_PAIRS // 3)}
buckets = defaultdict(list)   # label -> [(subject, rel1, rel2)]
for s in eligible:
    if s in used:
        continue
    for r1, r2 in itertools.combinations(sorted(by_subject[s]), 2):
        buckets[label(r1, r2)].append((s, r1, r2))
groups2 = []
for lab in ("similar", "related", "distinct"):
    cands = buckets[lab][:]
    rng.shuffle(cands)
    n = 0
    for s, r1, r2 in cands:
        if n == want[lab]:
            break
        if s in used:
            continue
        used.add(s)
        rels = [r1, r2]
        rng.shuffle(rels)            # random A/B order assignment
        groups2.append(make_group(s, rels))
        n += 1
    assert n == want[lab], f"only {n} {lab} groups available"

groups = groups2 + groups3 + groups4
for i, g in enumerate(groups, 1):
    g["group_id"] = f"g{i:03d}"

# --- verification of every case ID ---
for g in groups:
    assert len(set(g["case_ids"])) == g["size"] == len(g["relation_ids"])
    assert len(set(g["relation_ids"])) == g["size"], g["subject"]
    for c, rel in zip(g["case_ids"], g["relation_ids"]):
        assert case_info[c] == (g["subject"], rel), f"{g['group_id']}: bad case {c}"
    assert not set(g["case_ids"]) & pilot_ids
assert len({g["subject"] for g in groups}) == len(groups)

out = {
    "dataset_sha256": hashlib.sha256(raw).hexdigest(),
    "seed": SEED,
    "label_rule": "similar = same relation cluster; related = same domain, different cluster; "
                  "distinct = different domain",
    "clusters": CLUSTER, "domains": DOMAIN,
    "excluded_pilot_subjects": sorted(pilot_subjects),
    "groups": groups,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)

print(f"groups: {len(groups)}  by size: " +
      ", ".join(f"{k}-edit={sum(g['size']==k for g in groups)}" for k in (2, 3, 4)))
lab_counts = defaultdict(int)
for g in groups:
    for p in g["pair_labels"]:
        lab_counts[(g["size"], p["label"])] += 1
for k in sorted(lab_counts):
    print("pair labels (size, label):", k, lab_counts[k])
print(f"eligible subjects: {len(eligible)}; wrote {OUT}")
