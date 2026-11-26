# Slide 1 — Reproducible setup

- GPT-2 XL + official ROME, pinned source and model revisions
- Official layer-17 parameters; float32 and eager attention
- Isolated Python 3.11 environment in Colab
- Automatic model/data/statistics downloads and persistent Drive outputs

Speaker notes: The setup is shared across the team. We hold model settings fixed so edit order is the experimental variable.

# Slide 2 — Sequential-edit protocol

- Check 2–4 facts for the same subject using CounterFact case IDs
- Start each order from original weights
- Evaluate every fact before editing and after every edit
- Compare forward and reverse order; include independent single-edit controls
- Restore and verify weights after each sequence, including failures

Speaker notes: For pairs, the sequences are A then B and B then A. Longer groups use forward and reverse sequences, not all permutations.

# Slide 3 — Evidence and validation

- Save prompt-level old/new NLL and geometric-mean token probabilities
- Recompute ES, PS, NS and edit-retention flags from raw values
- Report loss among initially successful edits with its denominator
- Check identity, stage coverage, source/configuration hashes on resume
- Show progress and export reproducible raw evidence

Speaker notes: The November 24 A100 40GB run completed all 40 group/order records and passed local re-verification after export. Both-held counts were 15/20 forward and 16/20 reverse; all archived forward scores matched. These are descriptive pilot findings, not evidence of a significant order advantage. The explicit per-edit seeds and GPU memory variant are documented differences from the old pilot. Three- and four-edit GPU experiments remain outside this verification run.
