"""Verify raw multi-edit evidence without torch, CUDA, or network access."""
import argparse
import csv
import json
from pathlib import Path

from baseline_support import atomic_json
from multi_support import (SCHEMA, digest, file_hash, orders_for, read_json, require,
                           summarize, validate_result)

ROOT = Path(__file__).resolve().parents[1]


def verify_folder(folder, limit):
    folder = Path(folder)
    manifest = read_json(folder/'manifest.json')
    require(manifest['schema'] == SCHEMA, 'Legacy score-only pilot is not raw-verified evidence')
    groups = read_json(folder/'groups.json')
    specs = read_json(folder/'facts.json')
    require(groups == manifest['groups'], 'Group catalog mismatch')
    require(digest(specs) == manifest['facts_sha256'], 'Fact catalog mismatch')
    require(digest(read_json(folder/'context_templates.json')) == manifest['context_templates_sha256'], 'Context hash mismatch')
    for name, expected in manifest['sources'].items():
        require(Path(name).name == name, 'Invalid archived source path')
        require(file_hash(folder/'sources'/name) == expected, 'Source hash mismatch')
    require(type(limit) is int and 1 <= limit <= len(groups), 'Invalid verification limit')
    run_key = digest(manifest)
    expected = {}
    for group in groups:
        for order in orders_for(group['case_ids'], manifest['order']):
            expected[f"{group['group_id']}_{order}.json"] = (group, order)
    # Validate all existing results, even those beyond the requested prefix.
    for path in folder.glob('group_*.json'):
        require(path.name in expected, f'Unexpected result {path.name}')
        group, order = expected[path.name]
        validate_result(read_json(path), group, specs, run_key, order)
    results = []
    for group in groups[:limit]:
        for order in orders_for(group['case_ids'], manifest['order']):
            path = folder/f"{group['group_id']}_{order}.json"
            require(path.exists(), f'Missing result: {path.name}')
            results.append(validate_result(read_json(path), group, specs, run_key, order))
    report = summarize(results, limit, manifest['order'])
    report['run_key'] = run_key
    report['complete_requested_prefix'] = True
    report['complete_all_groups'] = limit == len(groups)
    report['result_sha256'] = {f"{r['group']['group_id']}_{r['order']}.json":
        file_hash(folder/f"{r['group']['group_id']}_{r['order']}.json") for r in results}
    report['verification_scope'] = 'Saved raw evidence, source fingerprints and recorded restoration; not an independent GPU rerun'
    legacy = ROOT/'results/multi_pilot'
    if manifest['order'] in ('forward', 'both'):
        comparisons = []
        for r in results:
            old_path = legacy/f"group_{int(r['group']['group_id'].split('_')[1]):02d}.json"
            if r['order'] != 'forward' or len(r['edit_order']) != 2 or not old_path.exists():
                continue
            old = read_json(old_path)
            if old['case_ids'] != r['group']['case_ids'] or old['subject'] != r['group']['subject']:
                continue
            a, b = map(str, r['edit_order'])
            stages = r['stages']
            mapping = {'pre_A': stages[0]['evaluations'][a], 'pre_B': stages[0]['evaluations'][b],
                       'A_edit_A': stages[1]['evaluations'][a], 'B_edit_A': stages[1]['evaluations'][b],
                       'A_edit_AB': stages[2]['evaluations'][a], 'B_edit_AB': stages[2]['evaluations'][b],
                       'Just_B': r['single_edit_controls'][b]}
            comparisons.append({'group_id': r['group']['group_id'], 'legacy_score_only': True,
                'double_hold_changed': old['double_hold'] != r['retention']['double_hold'],
                'prior_edit_loss_changed': old['prior_edit_loss'] != r['retention']['prior_edit_loss'],
                'score_deltas_percentage_points': {s: {m: v['scores'][m]-old['scores'][s][m] for m in v['scores']} for s, v in mapping.items()}})
        report['legacy_forward_comparison'] = comparisons
        report['legacy_comparison_note'] = 'Old pilot lacks raw probabilities. New per-edit seeding is explicit; GPU and source may differ. Differences are not tuned away.'
    return report, results


def write_exports(folder, report, results):
    folder = Path(folder)
    limit = report['num_groups']
    atomic_json(folder/f'summary_{limit}.json', report)
    with (folder/f'comparison_{limit}.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group_id','subject','order','stage','case_id','efficacy','paraphrase','neighborhood','relation_labels'])
        for r in results:
            labels = json.dumps({k: v for k,v in r['group'].items() if k not in ('group_id','subject','case_ids')}, sort_keys=True)
            stages = [(str(i), s['evaluations']) for i,s in enumerate(r['stages'])]
            stages.append(('single_edit_control', r['single_edit_controls']))
            for stage, evaluations in stages:
                for cid, row in evaluations.items():
                    writer.writerow([r['group']['group_id'],r['group']['subject'],r['order'],stage,cid,
                                     *[row['scores'][m] for m in ('efficacy','paraphrase','neighborhood')],labels])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--write-summary', action='store_true')
    args = parser.parse_args()
    report, results = verify_folder(args.output, args.limit)
    if args.write_summary:
        write_exports(args.output, report, results)
    else:
        saved = args.output/f'summary_{args.limit}.json'
        if saved.exists():
            require(read_json(saved) == report, 'Saved summary differs from independently recomputed evidence')
    print(json.dumps(report, indent=2))
    print('VERIFIED')


if __name__ == '__main__':
    main()
