# Final-phase multi-edit handoff

## Status

Completed November 24, 2026 on an NVIDIA A100-SXM4-40GB. All 20 pilot groups ran in both orders (40 results); the five-group resume check preserved all result hashes. The exported files passed the verifier again locally. Forward both-held: 15/20 (75%); reverse: 16/20 (80%). First-edit loss: 5/20 forward and 4/20 reverse, with all 20 first edits initially successful in each order. All archived forward-stage ES/PS/NS values match exactly. See [verification record](FINAL_VERIFICATION.md), [raw results](../results/multi_pilot_verified_20261124/), and [executed notebook](../notebooks/archive/Executed_Final_Multi_Edit.ipynb). Existing `results/multi_pilot/` is preserved as historical, score-only evidence. GPU verification covered two-edit groups; 3–4-edit protocol coverage is CPU-tested, not claimed as a completed GPU experiment.

## Run in Colab

Open `notebooks/Group32_Final_Multi_Edit.ipynb`, select an NVIDIA GPU (prefer A100 to match the old pilot), and run top to bottom. The notebook clones the implementation branch, installs the pinned environment, mounts Drive, and stages 1, 5, and 20 groups in both orders. Each stage verifies its outputs before continuing. Downloads include CounterFact, GPT-2 XL, and official covariance statistics; no dataset upload is needed.

Use a new Drive output folder for this schema. Do not reuse `results/multi_pilot` or the old baseline output. If the runtime disconnects, rerun setup and select the SAME source commit, GPU, package versions, groups, and output folder. A changed environment is deliberately rejected. A partial group/order restarts from original weights. Verified finished group/orders are reused. An advisory lock prevents concurrent writers to the same folder.

To reproduce this exact completed run, use experiment commit `44907cdcfa0fc0c0c4cc7026deca1c754cf9c2d5`, as recorded in the result folder. Later documentation/result commits do not change that recorded provenance.

## Nico: expanded groups

From the repository root, after `python3 scripts/setup.py`:

```bash
python3 scripts/run_multi_stage.py --groups results/related_groups/groups_v2.json --limit 100 --order both --output runs/final_groups_v2
python3 scripts/verify_multi_results.py --output runs/final_groups_v2 --limit 100
```

`--limit` counts groups, not edits or orders; it must not exceed the input length. `--order` is forward, reverse, or both. The default is forward. For 3–4-edit groups, reverse means reversing the whole sequence, not every permutation. A top-level group list or an object with a `pilot` list is supported. Each group needs `subject` and 2–4 distinct integer `case_ids`, all matching the subject and distinct relations in CounterFact. Extra fields, including relation labels, are preserved. Duplicate groups are rejected; reusing a fact in a different group is allowed.

Within each sequence, every fact is evaluated before edits and after each edit. Independent single-edit controls are run from clean weights for each fact. All sequences and controls reset the editable weights, with exact equality checked. Case lookup is by ID, never dataset position.

## Output contract

- `group_001_forward.json` / `group_001_reverse.json`: schema 2, group metadata, ordered case IDs, seeds, complete requested rewrites and prompts, pre/post-stage raw evaluations, per-fact scores, independent controls, retention, timings, UTC completion, restoration flag, and run key.
- `facts.json`, `groups.json`, `context_templates.json`, `manifest.json`: input snapshots and fingerprints. Manifest includes all groups so increasing 1 → 5 → 20 does not invalidate resume. Changed group lists require a new run.
- `sources/`: exact Python sources whose hashes are in the manifest. `packages.txt` uses the experiment interpreter.
- `summary_N.json`: recomputed verification report, completion scope, result-file hashes, retention denominators, and archived forward-pilot comparison.
- `comparison_N.csv`: one row per group/order/stage/fact, including independent controls and preserved labels.
- Per-sequence `.log` files contain upstream optimization output. Colab streams progress at each edit/evaluation boundary and completion with an ETA based on completed sequences.

Upstream `target_new` and `target_true` are mean target-token negative log-likelihoods, not probabilities: lower is better. `p_new` and `p_old` are `exp(-mean_NLL)`, geometric-mean token probabilities, not full-sequence likelihoods. Preserve the original NLL for scoring, including ties (failure). ES/PS prefer new; NS prefers old. Scores average prompts within each fact; the CSV keeps per-fact detail for team statistics.

First-edit loss requires that the first edit succeeded immediately after its application but fails after the full sequence. Reports include loss over all groups and conditional loss over initially successful first edits. For 3–4 facts, prior-edit retention includes earlier edits that succeeded immediately after application; the last edit is excluded from that denominator. Zero eligible edits yields null, not zero percent.

## Reproducibility and archived comparison

Keep GPT-2 XL revision `15ea56dee5df4983c59b2538573817e1667135e2`, ROME commit `0874014cd9837e4365f3e6f3c71400ef11509e04`, official layer-17 hyperparameters, float32, eager attention, and evaluation batches of four prefixes. Reuse the archived pilot context templates for both orders. Each edit is seeded with `32 + case_id`, including controls. This makes order comparisons and interrupted-run recovery explicit; the old pilot seeded the sequence only once, so that is a documented protocol difference rather than an exact replay claim.

The old pilot reported 15/20 both-held and 5/20 first-edit loss. Its archived JSON lacks raw evaluations. The new verifier compares forward scores and flags to matching archived cases, but cannot certify old raw scores retroactively. A different GPU is a separate run and a disclosed comparison limitation. Do not adjust hyperparameters to match the old numbers.

## Validation and release gate

```bash
python3 -m unittest discover -s scripts -p 'test_*.py' -v
python3 scripts/verify_results.py
python3 scripts/verify_multi_results.py --output RUN_FOLDER --limit 20
```

Completion requires all 40 pilot group/order results with controls, correct raw-score recomputation, and all restoration assertions. The verifier checks saved evidence; it does not independently prove the original GPU computation. To regenerate a summary from valid raw data, use `--write-summary`. The export cell rechecks evidence immediately before creating a ZIP.

Rohith owns pipeline/setup and this pilot verification. Arin owns group construction; Nico owns expanded runs; Jaivardhan owns consistency/fluency; Soham owns post-edit tracing; Yugandhar owns final statistics. No MEMIT or new fluency/tracing claims are included. Verification after November 21 must be identified as a late verification run, not represented as having met the earlier cutoff.
