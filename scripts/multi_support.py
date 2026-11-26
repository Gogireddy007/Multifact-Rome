"""CPU-only multi-edit protocol, schema validation, and recomputation."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

from baseline_support import scores

SCHEMA = 2
KINDS = ('rewrite', 'paraphrase', 'neighborhood', 'attribute')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def load_groups(data, records):
    groups = data.get('pilot') if isinstance(data, dict) else data
    require(isinstance(groups, list) and bool(groups), 'Expected a nonempty group list or pilot list')
    seen = set()
    result = []
    for i, raw in enumerate(groups, 1):
        group = copy.deepcopy(raw)
        ids = group.get('case_ids', [])
        require(isinstance(ids, list) and 2 <= len(ids) <= 4, 'Each group needs 2-4 case IDs')
        require(all(type(cid) is int for cid in ids), 'Case IDs must be integers')
        require(len(set(ids)) == len(ids), 'Duplicate case ID within group')
        require(all(cid in records for cid in ids), 'Missing CounterFact case ID')
        subject = group.get('subject')
        require(isinstance(subject, str) and bool(subject), 'Missing subject')
        require(all(records[c]['requested_rewrite']['subject'] == subject for c in ids), 'Subject mismatch')
        relations = [records[c]['requested_rewrite']['relation_id'] for c in ids]
        require(len(set(relations)) == len(ids), 'Repeated relation within group')
        signature = tuple(sorted(ids))
        require(signature not in seen, 'Duplicate group')
        seen.add(signature)
        group['group_id'] = f'group_{i:03d}'
        result.append(group)
    return result


def fact_spec(record):
    rewrite = record['requested_rewrite']
    prompts = {'rewrite': [rewrite['prompt'].format(rewrite['subject'])]}
    prompts.update({kind: record[kind + '_prompts'] for kind in KINDS[1:]})
    for kind, rows in prompts.items():
        require(isinstance(rows, list) and all(isinstance(p, str) for p in rows), f'Invalid {kind} prompts')
        if kind != 'attribute':
            require(bool(rows), f'Empty {kind} prompts')
    return {'case_id': record['case_id'], 'requested_rewrite': copy.deepcopy(rewrite), 'prompts': prompts}


def orders_for(ids, order):
    choices = {'forward': list(ids), 'reverse': list(reversed(ids))}
    require(order in ('forward', 'reverse', 'both'), 'Invalid order')
    return choices if order == 'both' else {order: choices[order]}


def pack_evaluation(raw, spec):
    rows = {}
    for kind in KINDS:
        entries = raw[kind + '_prompts_probs']
        require(len(entries) == len(spec['prompts'][kind]), 'Prompt count mismatch')
        rows[kind + '_prompts_probs'] = [
            dict(pair, prompt=prompt, p_new=math.exp(-pair['target_new']), p_old=math.exp(-pair['target_true']))
            for pair, prompt in zip(entries, spec['prompts'][kind])
        ]
    rows['scores'] = scores({'post': rows}, 'post')
    return rows


def validate_eval(value, spec):
    for kind in KINDS:
        rows = value[kind + '_prompts_probs']
        require(len(rows) == len(spec['prompts'][kind]), 'Prompt coverage mismatch')
        for row, prompt in zip(rows, spec['prompts'][kind]):
            require(row['prompt'] == prompt, 'Prompt identity mismatch')
            for raw, prob in [('target_new', 'p_new'), ('target_true', 'p_old')]:
                nll, p = row[raw], row[prob]
                require(type(nll) in (int, float) and math.isfinite(nll) and nll >= 0, 'Invalid NLL')
                require(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1, 'Invalid probability')
                require(math.isclose(p, math.exp(-nll), rel_tol=1e-10, abs_tol=1e-15), 'Probability conversion mismatch')
    require(value['scores'] == scores({'post': value}, 'post'), 'Cached scores mismatch')


def retention(stages, edit_order):
    final = stages[-1]['evaluations']
    held = {str(cid): final[str(cid)]['scores']['efficacy'] == 100.0 for cid in edit_order}
    eligible = []
    lost = []
    # The last edit has no subsequent edit, so cannot contribute to retention loss.
    for step, cid in enumerate(edit_order[:-1], 1):
        if stages[step]['evaluations'][str(cid)]['scores']['efficacy'] == 100.0:
            eligible.append(cid)
            if not held[str(cid)]:
                lost.append(cid)
    first = edit_order[0]
    return {'all_edits_hold': all(held.values()), 'final_held': held,
            'eligible_prior_case_ids': eligible, 'lost_prior_case_ids': lost,
            'first_edit_initially_successful': first in eligible,
            'prior_edit_loss': first in lost,
            'double_hold': all(held.values()) if len(edit_order) == 2 else None}


def execute_protocol(case_ids, edit_order, evaluate, edit, reset, reseed, progress=lambda message: None):
    """Callbacks make isolation/restoration and stage coverage testable without a GPU."""
    stages, controls, times = [], {}, []
    def evaluate_all():
        return {str(cid): evaluate(cid) for cid in case_ids}
    try:
        reset()
        progress('evaluating unedited model')
        stages.append({'applied_case_ids': [], 'evaluations': evaluate_all()})
        applied = []
        for step, cid in enumerate(edit_order, 1):
            progress(f'edit {step}/{len(edit_order)}: case {cid}')
            reseed(cid)
            started = time.perf_counter()
            edit(cid)
            times.append(time.perf_counter() - started)
            applied.append(cid)
            progress(f'evaluating all facts after edit {step}')
            stages.append({'applied_case_ids': list(applied), 'evaluations': evaluate_all()})
        for cid in case_ids:
            reset()
            reseed(cid)
            progress(f'independent control: case {cid}')
            edit(cid)
            controls[str(cid)] = evaluate(cid)
    finally:
        reset()
    return stages, controls, times


def validate_result(result, group, specs, run_key, order):
    ids = group['case_ids']
    edit_order = orders_for(ids, order)[order]
    require(result['schema'] == SCHEMA, 'Unsupported result schema')
    require(result['run_key'] == run_key and result['group'] == group, 'Run/group identity mismatch')
    require(result['order'] == order and result['edit_order'] == edit_order, 'Edit order mismatch')
    require(result['facts'] == {str(c): specs[str(c)] for c in ids}, 'Fact specification mismatch')
    require(result['weights_restored'] is True, 'Weight restoration not verified')
    require(result['seeds'] == {str(c): 32 + c for c in ids}, 'Seed mismatch')
    stages = result['stages']
    require(len(stages) == len(ids) + 1, 'Incomplete stages')
    for index, stage in enumerate(stages):
        require(stage['applied_case_ids'] == edit_order[:index], 'Stage sequence mismatch')
        require(set(stage['evaluations']) == {str(c) for c in ids}, 'Missing/extra evaluated fact')
        for cid in ids:
            validate_eval(stage['evaluations'][str(cid)], specs[str(cid)])
    require(set(result['single_edit_controls']) == {str(c) for c in ids}, 'Missing/extra independent control')
    for cid in ids:
        validate_eval(result['single_edit_controls'][str(cid)], specs[str(cid)])
    require(result['retention'] == retention(stages, edit_order), 'Retention mismatch')
    require(len(result['edit_seconds']) == len(ids), 'Missing edit timings')
    for t in result['edit_seconds'] + [result['elapsed_seconds']]:
        require(type(t) in (int, float) and math.isfinite(t) and t >= 0, 'Invalid timing')
    return result


def check_manifest(previous, current):
    require(previous == current, 'Environment, input or source changed. Use a NEW output folder.')


def summarize(results, expected_groups, order):
    require(bool(results), 'No results')
    output = {'schema': SCHEMA, 'num_groups': expected_groups, 'num_group_orders': len(results),
              'order': order, 'verified': True, 'orders': {}}
    for name in ('forward', 'reverse'):
        subset = [r for r in results if r['order'] == name]
        if not subset:
            continue
        n = len(subset)
        eligible = sum(len(r['retention']['eligible_prior_case_ids']) for r in subset)
        lost = sum(len(r['retention']['lost_prior_case_ids']) for r in subset)
        first_eligible = sum(r['retention']['first_edit_initially_successful'] for r in subset)
        first_lost = sum(r['retention']['prior_edit_loss'] for r in subset)
        output['orders'][name] = {
            'num_groups': n, 'all_edits_hold_count': sum(r['retention']['all_edits_hold'] for r in subset),
            'all_edits_hold_percent': 100 * sum(r['retention']['all_edits_hold'] for r in subset) / n,
            'first_edit_loss_count': first_lost, 'first_edit_eligible_count': first_eligible,
            'first_edit_loss_percent_all_groups': 100 * first_lost / n,
            'first_edit_loss_percent_eligible': 100 * first_lost / first_eligible if first_eligible else None,
            'prior_edits_lost': lost, 'prior_edits_eligible': eligible,
            'prior_edit_loss_percent_eligible': 100 * lost / eligible if eligible else None,
        }
    return output
