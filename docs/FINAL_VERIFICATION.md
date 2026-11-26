# Completed pilot verification

Completed November 24, 2026. This is Rohith's pipeline verification on the existing 20 two-edit groups, not the expanded 100+ group experiment owned by Nico.

## Observed results

- 20 groups, both orders, 40 complete group/order records.
- Forward: both edits held in 15/20 groups (75%); first-edit loss in 5/20 (25%).
- Reverse: both edits held in 16/20 groups (80%); first-edit loss in 4/20 (20%).
- Initially successful first edits: 20/20 in each order, so conditional and unconditional first-edit loss rates coincide here.
- All 20 historical forward comparisons have zero ES/PS/NS differences at every comparable saved stage and no changed survival/loss flags.
- These descriptive pilot outcomes do not establish a statistically significant order advantage.

## Execution and verification

Experiment source commit: `44907cdcfa0fc0c0c4cc7026deca1c754cf9c2d5`.

Run key: `b908f54972fd42f2a172183ed512c9d59a83b4dc2f86eaa9a3bae2d211fcee79`.

GPU: `NVIDIA A100-SXM4-40GB`; experiment Python `3.11.17`; PyTorch `2.5.1+cu121`; CUDA `12.1`.

Fresh Colab setup passed its dependency/import checks and actual GPU calculation. All 20 unit tests passed inside the isolated environment. The notebook completed 1 → 5 → resume verification → 20 groups. The five-group resume check preserved all ten result file hashes. The final export cell verified all 40 records before creating the ZIP. The downloaded ZIP passed its integrity test, and local offline recomputation matched `summary_20.json` exactly. Current runner/support/verifier source hashes match the archived manifest. The executed notebook contains all completion markers and no error outputs.

Each result includes unedited evaluation, evaluation of both facts after each edit, two independent single-edit controls, raw NLL and derived geometric-mean token probabilities, weight restoration assertions, timing and provenance. Across both orders this produces 80 control evaluations and 40 completed sequences. Every sequence starts from original weights.

Sum of per-sequence elapsed times: 566.63 seconds (excludes setup, model download/loading and pauses between stages). Result completion timestamps span `2026-11-24T21:14:45.021585+00:00` through `2026-11-24T21:24:46.059891+00:00`.

## Files and checksums

- Raw evidence: `results/multi_pilot_verified_20261124/`.
- Completed notebook: `notebooks/archive/Executed_Final_Multi_Edit.ipynb`.
- ZIP SHA-256: `2844aa434e24ef38dfaa13b599c9cba6cbafdbb0f39413868abee5aa75e42a2d`.
- Executed notebook SHA-256: `fbcc57fd6c39f24d85dc5bd3d565d0c6c1cce61e877172b3e8a632612220f574`.
- Per-result hashes: `summary_20.json` → `result_sha256`.

Recheck from the repository root:

```bash
python3 scripts/verify_multi_results.py --output results/multi_pilot_verified_20261124 --limit 20
```

The offline verifier recomputes saved evidence; the live Colab run supplies the GPU execution evidence. It does not independently rerun model editing during the offline check.

## Limits and handoff

The run was completed after the November 21 experiment cutoff and is explicitly dated as late verification. It uses an A100 40GB rather than the historical A100 80GB, and per-edit seeding rather than a single seed per sequence; model revision and official ROME settings were fixed. Three- and four-edit protocol paths are CPU-tested, but no new GPU results for those group sizes are claimed. Expanded groups, consistency/fluency, post-edit tracing, and final statistical inference remain with the teammates assigned in the plan. MEMIT was not run.
