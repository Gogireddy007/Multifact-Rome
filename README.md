# NLP 544 — Group 32 ROME baseline

Rohith's completed single-edit baseline and shared setup for the team's model-editing project. The supplied run completed 200 independent CounterFact edits on GPT-2 XL with a Tesla T4.

- Efficacy: **100.00%** (paper reference: 100.0%).
- Paraphrase: **94.75%** (paper reference: 96.4%).
- Neighborhood: **74.45%** (paper reference: 75.4%).

These are probability-based success rates, not exact-match generation accuracy. The run uses the first 200 CounterFact records; [ROME Table 4](https://arxiv.org/pdf/2202.05262) uses 7,500 GPT-2 XL cases. This is a subset replication, not an exact reproduction. Fluency and generation consistency are not evaluated here.

## Start in Colab

- [Open baseline notebook](https://colab.research.google.com/github/Gogireddy007/Multifact-Rome/blob/main/notebooks/Group32_ROME_Baseline.ipynb)
- [Open shared model setup for Soham](https://colab.research.google.com/github/Gogireddy007/Multifact-Rome/blob/main/notebooks/Shared_Model_Setup.ipynb)

Choose a GPU runtime and run cells in order. Setup installs a separate Python 3.11 environment even when Colab's notebook kernel uses Python 3.13. Model weights, CounterFact and covariance statistics download automatically when needed. Use a fresh session with sufficient RAM/disk; the observed GPU had 14.6 GiB memory. Setup downloads several GB. The original 200-case experiment took about an hour; runtime varies. New runs show live progress after each completed case.

The shared notebook only installs the environment and loads the model; it does not run the baseline. Model variables live inside its Python 3.11 `EXPERIMENT` block. Soham should put tracing code in that same block or run a script with `.venv/bin/python` so it shares the correct environment. Do not import these packages into Colab's separate Python 3.13 kernel.

## Linux / NVIDIA installation

```bash
git clone https://github.com/Gogireddy007/Multifact-Rome.git
cd Multifact-Rome
python3 scripts/setup.py
```

The setup uses `uv` to install Python 3.11, creates `.venv`, installs `torch==2.5.1` from the CUDA 12.1 wheel index, installs `requirements.txt`, and checks ROME imports and a GPU calculation. It sets `MPLBACKEND=Agg` to prevent Colab's inline-backend import failure. PyArrow 17.0.0 and pandas 2.2.3 are pinned alongside NumPy 1.26.4 to prevent the observed compatibility failure.

Upstream [kmeng01/rome](https://github.com/kmeng01/rome) is cloned into ignored `vendor/rome` at commit `0874014cd9837e4365f3e6f3c71400ef11509e04`. Its source remains unchanged. This repository does not upload model weights, virtual environments, or the full CounterFact dataset.

## Load the model for causal tracing

```bash
.venv/bin/python scripts/model_setup.py
```

In a Python script run with `.venv/bin/python`, import the helper (add the repository's `scripts` directory to `sys.path` if necessary):

```python
from model_setup import load_model_and_tokenizer
model, tokenizer = load_model_and_tokenizer()
from experiments.causal_trace import ModelAndTokenizer
mt = ModelAndTokenizer(model=model, tokenizer=tokenizer)
# Use mt with the official causal tracing functions.
```

The helper loads clean GPT-2 XL revision `15ea56dee5df4983c59b2538573817e1667135e2`, float32 weights, eager attention, evaluation mode, frozen parameters, and EOS padding. It changes the working directory to `vendor/rome` because upstream configuration uses relative paths. Use absolute paths for your own outputs. Causal tracing experiments and heatmaps remain Soham's work.

## Rerun baseline

From the repository root, run each stage after checking the previous one:

```bash
.venv/bin/python scripts/run_stage.py --limit 1 --output runs/baseline_v1
.venv/bin/python scripts/run_stage.py --limit 5 --output runs/baseline_v1
.venv/bin/python scripts/run_stage.py --limit 200 --output runs/baseline_v1
```

Each fact starts from original model weights. Weights are restored in a `finally` block and checked for exact equality. Official ROME hyperparameters and scoring functions are retained; evaluation batches four prefixes at a time to limit memory use. Completed JSON results are saved atomically and reused on resume. Configuration, GPU, environment or source changes require a new output folder. Do not resume the new runner into the archived results: its source hash differs due to the corrected package-report interpreter and pinned model revision.

## Files for teammates

- `scripts/setup.py`, `requirements.txt`: complete installation commands.
- `scripts/model_setup.py`: reusable model/tokenizer loading.
- `notebooks/Shared_Model_Setup.ipynb`: Colab setup and editable experiment block.
- `scripts/run_baseline.py`, `baseline_support.py`, `run_stage.py`: evaluation and progress reporting.
- `results/baseline_200/`: all 200 raw records, comparison CSV, summary, original logs and manifests.
- `notebooks/archive/Executed_Baseline.ipynb`: original uploaded notebook and outputs, preserved as evidence. Its setup cell is only a repair/check; **do not use it as a fresh-session entry point**.
- `provenance/`: exact original runner/support/test sources, whose hashes match the archived manifest.
- `scripts/build_groups.py`, `results/related_groups/`: related-fact groups for the multi-edit pilot (20 subjects plus 6 spares, 2 edits each).

## Validation and provenance

```bash
python3 scripts/verify_results.py
python3 -m unittest discover -s scripts -p 'test_*.py' -v
```

Local verification checks all 200 unique cases, matching run keys, recorded restoration flags, finite metrics, source hashes and independently recomputed pre/post scores. This verifies saved evidence; it does not independently rerun model edits or prove saved restoration flags beyond the original run's assertions.

The uploaded notebook records successful GPU execution and completed results. The new fresh-session setup, live-progress wrapper and standalone model loader have been syntax-checked and reviewed, but have **not yet been executed end-to-end on a fresh GPU**. There is no claim that the packaged setup has passed that separate test.

**Known provenance issue:** archived `packages.txt` captured Colab's outer Python environment because the original runner invoked bare `python -m pip freeze`. Do not install from that file. The manifest recorded the actual experiment's core packages through its running interpreter. The new runner uses `sys.executable`; `requirements.txt` additionally includes the confirmed PyArrow/pandas/backend fixes. Original evidence is preserved unchanged.

## Related-fact groups (multi-edit pilot)

`results/related_groups/pilot_groups.json` holds the pilot groups. It has two keys, `pilot` (20 subjects) and `spares` (6), each a list of `{subject, case_ids}`. Look records up by `case_id`, not by list position in `counterfact.json`.

Selection rule: each subject has 2 edits with different `relation_id`s, so two edits never overwrite each other, and the subjects come from different domains. Out of N CounterFact subjects, M have 2+ distinct relations.

Regenerate with:

    python3 scripts/build_groups.py path/to/counterfact.json

The script reads CounterFact only to verify that every `case_id` belongs to its subject and to count usable groups. The dataset itself is not committed.

## Team scope

Rohith: shared setup and single-edit baseline. Soham: causal tracing. Arin: related-fact grouping. Nico: multi-edit pilot. Yuga: comparison/slides. This repository delivers the baseline and reusable setup; it does not claim the team's multi-edit or causal-tracing experiments are finished.

## What is in `groups_v2.json`

`results/related_groups/groups_v2.json` holds the larger set of fact groups: 141 groups, one subject each, with no pilot subjects.

### Top-level keys

| Key | What it holds |
|---|---|
| `dataset_sha256` | Fingerprint of the `counterfact.json` used |
| `seed` | Random seed (32). Same data and script give the same groups |
| `label_rule` | Definition of `similar`, `related` and `distinct` |
| `clusters` | Cluster name to relation IDs, for example `language` to P1412, P103, P37, P364, P407 |
| `domains` | Cluster name to broader domain, for example `life_place` and `citizenship` to `geography` |
| `excluded_pilot_subjects` | The 20 pilot subjects left out |
| `groups` | The list of groups |

### One group

```json
{
  "group_id": "g121",
  "subject": "Sesamstraat",
  "size": 3,
  "case_ids": [5596, 21197, 13694],
  "relation_ids": ["P407", "P495", "P364"],
  "pair_labels": [
    {"a": 0, "b": 1, "label": "distinct"},
    {"a": 0, "b": 2, "label": "similar"},
    {"a": 1, "b": 2, "label": "distinct"}
  ]
}
```

- `group_id`: g001 to g141.
- `subject`: the entity every fact in the group is about.
- `size`: number of edits in the group.
- `case_ids`: CounterFact `case_id` values, not list positions. Prompts and old and new answers are not copied here; look them up in `counterfact.json`. The listed order is the default edit order (A, then B).
- `relation_ids`: Wikidata property codes, in the same order as `case_ids`.
- `pair_labels`: one entry per pair of facts (1, 3 or 6 entries for 2, 3 or 4 edits). `a` and `b` are positions in `case_ids`, counted from 0.

### Labels

- `similar`: both relations are in the same cluster.
- `related`: same domain, different cluster.
- `distinct`: different domains.

The cluster table is hand-made.

### Counts

| Group IDs | Edits per group | Groups | Similar pairs | Related pairs | Distinct pairs |
|---|---|---|---|---|---|
| g001 to g120 | 2 | 120 | 40 | 40 | 40 |
| g121 to g132 | 3 | 12 | 6 | 0 | 30 |
| g133 to g141 | 4 | 9 | 11 | 0 | 43 |

The 2-edit groups were balanced on purpose. The 3 and 4-edit groups have no `related` pairs.

`run_multi_edit.py` currently reads only `pilot_groups.json` and handles two edits, so it needs changes to run this file.

Regenerate with:

```bash
python3 scripts/build_groups_v2.py path/to/counterfact.json
```
