"""Verified sequential ROME experiments. Launch with run_multi_stage.py."""
import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

os.environ['MPLBACKEND'] = 'Agg'
from baseline_support import atomic_json
from multi_support import (SCHEMA, check_manifest, digest, execute_protocol, fact_spec,
                           file_hash, load_groups, orders_for, pack_evaluation, read_json,
                           require, retention, validate_result)

ROOT = Path(__file__).resolve().parents[1]
ROME_COMMIT = '0874014cd9837e4365f3e6f3c71400ef11509e04'
MODEL_REVISION = '15ea56dee5df4983c59b2538573817e1667135e2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=Path, default=ROOT/'results/related_groups/pilot_groups.json')
    parser.add_argument('--order', choices=['forward', 'reverse', 'both'], default='forward')
    args = parser.parse_args()
    output, groups_path = args.output.resolve(), args.groups.resolve()
    require(args.limit > 0, 'Limit must be positive')
    vendor = ROOT/'vendor/rome'
    sys.path.insert(0, str(vendor))
    os.chdir(vendor)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dsets import CounterFactDataset
    from rome import ROMEHyperParams, apply_rome_to_model
    import rome.rome_main as rome_main
    from util import nethook
    from util.globals import DATA_DIR, STATS_DIR, REMOTE_ROOT_URL
    import experiments.py.eval_utils_counterfact as evaluation

    require(torch.cuda.is_available(), 'A CUDA GPU is required')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == ROME_COMMIT, 'ROME commit mismatch')
    require(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], text=True).strip(), 'ROME source changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def reseed(cid=0):
        random.seed(32 + cid)
        np.random.seed(32 + cid)
        torch.manual_seed(32 + cid)
        torch.cuda.manual_seed_all(32 + cid)
    reseed()
    dataset = list(CounterFactDataset(DATA_DIR))
    records = {r['case_id']: r for r in dataset}
    require(len(records) == len(dataset), 'Duplicate dataset case IDs')
    groups = load_groups(read_json(groups_path), records)
    require(args.limit <= len(groups), 'Limit exceeds available groups')
    ids = sorted({c for group in groups for c in group['case_ids']})
    specs = {str(c): fact_spec(records[c]) for c in ids}
    hparams_path = Path('hparams/ROME/gpt2-xl.json')
    hp = read_json(hparams_path)
    hparams = ROMEHyperParams.from_json(hparams_path)
    require(hp['layers'] == [17], 'Unexpected edit layer')
    # A prefix extension (1 -> 5 -> 20) is not a configuration change.
    config = {
        'schema': SCHEMA, 'rome_commit': commit, 'model': 'gpt2-xl', 'model_revision': MODEL_REVISION,
        'dtype': 'float32', 'attention': 'eager', 'hparams': hp, 'seed': 32,
        'seed_policy': '32 + case_id before each sequential or independent edit',
        'order': args.order, 'groups': groups, 'groups_sha256': file_hash(groups_path),
        'dataset_sha256': file_hash(Path(DATA_DIR)/'counterfact.json'), 'facts_sha256': digest(specs),
        'sources': {name: file_hash(ROOT/'scripts'/name) for name in
                    ('run_multi_edit.py', 'multi_support.py', 'baseline_support.py', 'run_multi_stage.py', 'verify_multi_results.py')},
        'requirements_sha256': file_hash(ROOT/'requirements.txt'),
        'packages': {x: importlib.metadata.version(x) for x in ('torch','transformers','tokenizers','datasets',
                     'numpy','scipy','scikit-learn','huggingface-hub','pyarrow','pandas')},
        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'evaluation_prefix_batch_size': 4, 'probability_definition': 'exp(-mean target-token NLL); not joint sequence probability',
        'context_policy': 'reuse archived pilot templates for paired-order comparability',
    }
    output.mkdir(parents=True, exist_ok=True)
    # An OS advisory lock is released even after a crash. Prevent two writers to one run.
    import fcntl
    lock = (output/'.run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another process is writing this run folder')
    context_path = output/'context_templates.json'
    original_context = ROOT/'results/multi_pilot/context_templates.json'
    contexts = read_json(original_context)
    require(isinstance(contexts, list) and contexts and all(isinstance(t, str) for t in contexts), 'Invalid templates')
    manifest = dict(config, context_templates_sha256=digest(contexts))
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        check_manifest(read_json(manifest_path), manifest)
        require(read_json(output/'facts.json') == specs, 'Fact catalog changed')
        require(read_json(output/'groups.json') == groups, 'Group catalog changed')
        require(read_json(context_path) == contexts, 'Context templates changed')
    else:
        require(not list(output.glob('group_*.json')), 'Unmanifested results; use a new folder')
        atomic_json(context_path, contexts)
        atomic_json(output/'facts.json', specs)
        atomic_json(output/'groups.json', groups)
        (output/'sources').mkdir(exist_ok=True)
        for name in config['sources']:
            (output/'sources'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
        atomic_json(manifest_path, manifest)
    for name, expected in manifest['sources'].items():
        require(file_hash(output/'sources'/name) == expected, 'Archived source hash mismatch')
    run_key = digest(manifest)
    pending = []
    for group in groups[:args.limit]:
        for order, edit_order in orders_for(group['case_ids'], args.order).items():
            path = output/f"{group['group_id']}_{order}.json"
            if path.exists():
                validate_result(read_json(path), group, specs, run_key, order)
                print(f"Resume: verified {path.name}", flush=True)
            else:
                pending.append((group, order, edit_order, path))
    if pending:
        stats_rel = 'gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz'
        stats_path = Path(STATS_DIR)/stats_rel
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if not stats_path.exists():
            print('Downloading official covariance statistics...', flush=True)
            temp = stats_path.with_suffix('.download')
            torch.hub.download_url_to_file(f'{REMOTE_ROOT_URL}/data/stats/{stats_rel}', temp)
            with np.load(temp, allow_pickle=False) as z:
                require(bool(z.files), 'Empty covariance archive')
            temp.replace(stats_path)
        print('Loading pinned GPT-2 XL on GPU...', flush=True)
        model = AutoModelForCausalLM.from_pretrained('gpt2-xl', revision=MODEL_REVISION,
                    torch_dtype=torch.float32, attn_implementation='eager', use_safetensors=True).eval().cuda()
        tok = AutoTokenizer.from_pretrained('gpt2-xl', revision=MODEL_REVISION)
        tok.pad_token = tok.eos_token
        nethook.set_requires_grad(False, model)
        original_prediction = evaluation.test_batch_prediction
        def chunked_prediction(model, tok, prefixes, target_new, target_true):
            values = []
            for start in range(0, len(prefixes), 4):
                values.extend(original_prediction(model, tok, prefixes[start:start+4], target_new, target_true))
            return values
        evaluation.test_batch_prediction = chunked_prediction
        rome_main.CONTEXT_TEMPLATES_CACHE = contexts
        with (output/'packages.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        weights = {f'{hparams.rewrite_module_tmp.format(layer)}.weight':
                   nethook.get_parameter(model, f'{hparams.rewrite_module_tmp.format(layer)}.weight') for layer in hparams.layers}
        originals = {name: value.detach().cpu().clone() for name, value in weights.items()}
        def reset():
            with torch.no_grad():
                for name, value in weights.items():
                    value.copy_(originals[name].to(value.device))
                for name, value in weights.items():
                    require(torch.equal(value.detach().cpu(), originals[name]), 'Weight restoration failed')
        def evaluate(cid):
            return pack_evaluation(evaluation.compute_rewrite_quality_counterfact(model, tok, records[cid], None, None), specs[str(cid)])
        def edit(cid):
            apply_rome_to_model(model, tok, [records[cid]['requested_rewrite']], hparams, copy=False, return_orig_weights=False)
        console = sys.stdout
        start_all = time.perf_counter()
        for done, (group, order, edit_order, path) in enumerate(pending, 1):
            started = time.perf_counter()
            def progress(message):
                print(f"[{done}/{len(pending)} pending] {group['group_id']} {group['subject']} {order}: {message}", file=console, flush=True)
            with path.with_suffix('.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                stages, controls, times = execute_protocol(group['case_ids'], edit_order, evaluate, edit, reset, reseed, progress)
            result = {'schema': SCHEMA, 'group': group, 'order': order, 'edit_order': edit_order,
                      'facts': {str(c): specs[str(c)] for c in group['case_ids']},
                      'run_key': run_key, 'stages': stages, 'single_edit_controls': controls,
                      'seeds': {str(c): 32+c for c in group['case_ids']}, 'edit_seconds': times,
                      'retention': retention(stages, edit_order), 'weights_restored': True,
                      'elapsed_seconds': time.perf_counter()-started,
                      'completed_at_utc': datetime.now(timezone.utc).isoformat()}
            validate_result(result, group, specs, run_key, order)
            atomic_json(path, result)
            elapsed = time.perf_counter()-start_all
            print(f'SAVED {path.name}; {done}/{len(pending)} pending completed; elapsed {elapsed/60:.1f} min; ETA {(len(pending)-done)*elapsed/done/60:.1f} min', flush=True)
    from verify_multi_results import verify_folder, write_exports
    report, results = verify_folder(output, args.limit)
    write_exports(output, report, results)
    print(json.dumps(report, indent=2), flush=True)
    print('VERIFIED MULTI-EDIT STAGE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
