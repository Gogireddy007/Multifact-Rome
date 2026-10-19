"""Result validation and aggregation; no GPU dependencies."""
import json
import math
import os
from pathlib import Path

METRICS = {'efficacy': 'rewrite_prompts_probs', 'paraphrase': 'paraphrase_prompts_probs', 'neighborhood': 'neighborhood_prompts_probs'}
PAPER = {'efficacy': 100.0, 'paraphrase': 96.4, 'neighborhood': 75.4}

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)

def scores(result, phase):
    out = {}
    for metric, key in METRICS.items():
        entries = result[phase][key]
        if not entries:
            raise ValueError(f'Empty {phase}/{key}')
        successes = []
        for pair in entries:
            true, new = pair['target_true'], pair['target_new']
            if not all(math.isfinite(x) for x in (true, new)):
                raise ValueError('Nonfinite evaluation probability')
            successes.append(true < new if metric == 'neighborhood' else new < true)
        out[metric] = 100 * sum(successes) / len(successes)
    return out

def validate_result(result, record, run_key):
    if result['run_key'] != run_key or result['case_id'] != record['case_id']:
        raise ValueError('Result belongs to another run/case')
    if result['requested_rewrite'] != record['requested_rewrite']:
        raise ValueError('Requested rewrite mismatch')
    if result.get('weights_restored') is not True:
        raise ValueError('Weight restoration was not verified')
    for phase in ('pre', 'post'):
        scores(result, phase)
    return result

def aggregate(results):
    if not results:
        raise ValueError('No results; cannot report scores')
    ids = [r['case_id'] for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate cases')
    output = {'num_cases': len(results), 'case_ids': ids}
    for phase in ('pre', 'post'):
        rows = [scores(r, phase) for r in results]
        output[phase] = {m: sum(row[m] for row in rows) / len(rows) for m in METRICS}
    output['paper_gpt2_xl_7500_cases'] = PAPER
    output['difference_percentage_points'] = {m: output['post'][m] - PAPER[m] for m in METRICS}
    return output
