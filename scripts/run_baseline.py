"""Run from the pinned official ROME checkout. Uses original editing/evaluation code."""
import argparse
import os
import sys
os.environ["MPLBACKEND"] = "Agg"
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path

from baseline_support import atomic_json, aggregate, validate_result

ROME_COMMIT = '0874014cd9837e4365f3e6f3c71400ef11509e04'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, choices=[1, 5, 200], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info
    from dsets import CounterFactDataset
    from rome import ROMEHyperParams, apply_rome_to_model
    import rome.rome_main as rome_main
    from util import nethook
    from util.globals import DATA_DIR, STATS_DIR, REMOTE_ROOT_URL
    import experiments.py.eval_utils_counterfact as evaluation

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU runtime is required. Choose GPU in Colab runtime settings.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if commit != ROME_COMMIT:
        raise RuntimeError('ROME version mismatch')
    if subprocess.check_output(['git', 'diff', '--name-only'], text=True).strip():
        raise RuntimeError('Tracked ROME source files were modified')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = 32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    model_id = 'gpt2-xl'
    revision = previous['model_revision'] if previous else '15ea56dee5df4983c59b2538573817e1667135e2'
    hp_path = Path('hparams/ROME/gpt2-xl.json')
    hp = json.loads(hp_path.read_text())
    hparams = ROMEHyperParams.from_json(hp_path)
    ds = CounterFactDataset(DATA_DIR, size=200)
    if len(ds) != 200:
        raise RuntimeError('Expected exactly 200 dataset records')
    records = list(ds)
    ids = [r['case_id'] for r in records]
    if len(set(ids)) != 200:
        raise RuntimeError('Duplicate dataset case IDs')
    packages = {x: importlib.metadata.version(x) for x in ['torch','transformers','tokenizers','datasets','numpy','scipy','scikit-learn','huggingface-hub']}
    manifest = {
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': hashlib.sha256(Path(__file__).with_name('baseline_support.py').read_bytes()).hexdigest(),
        'schema': 1, 'rome_commit': commit, 'model': model_id, 'model_revision': revision,
        'dtype': 'float32', 'attention': 'eager', 'seed': seed, 'hparams': hp,
        'selection': 'first 200 CounterFact records, in published order; no success filtering',
        'case_ids': ids, 'dataset_sha256': hashlib.sha256((Path(DATA_DIR)/'counterfact.json').read_bytes()).hexdigest(),
        'packages': packages, 'python': platform.python_version(),
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'metrics': 'official probability-based ES/PS/NS; generation metrics omitted',
        'evaluation_prefix_batch_size': 4,
    }
    if previous and previous != manifest:
        raise RuntimeError('Environment/config changed. Use a NEW output folder to avoid mixing results.')
    atomic_json(manifest_path, manifest)
    run_key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    pending = []
    for record in records[:args.limit]:
        path = output / f"case_{record['case_id']}.json"
        if path.exists():
            validate_result(json.loads(path.read_text()), record, run_key)
        else:
            pending.append(record)
    if pending:
        # Download official covariance first. Never silently launch 100k-document recomputation.
        rel = f"gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz"
        stats_path = Path(STATS_DIR) / rel
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if not stats_path.exists():
            temp = stats_path.with_suffix('.download')
            torch.hub.download_url_to_file(f'{REMOTE_ROOT_URL}/data/stats/{rel}', temp)
            with np.load(temp, allow_pickle=False) as z:
                assert z.files, 'Empty covariance archive'
            temp.replace(stats_path)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
            torch_dtype=torch.float32, attn_implementation='eager', use_safetensors=True).eval().cuda()
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        tok.pad_token = tok.eos_token
        # Freeze model parameters; ROME optimizes an internal activation, not these weights.
        nethook.set_requires_grad(False, model)
        # Keep original probability calculations; bound the number of prompts per forward pass.
        original_prediction = evaluation.test_batch_prediction
        def chunked_prediction(model, tok, prefixes, target_new, target_true):
            results = []
            for start in range(0, len(prefixes), 4):
                results.extend(original_prediction(model, tok, prefixes[start:start+4], target_new, target_true))
            return results
        evaluation.test_batch_prediction = chunked_prediction
        context_path = output / 'context_templates.json'
        if context_path.exists():
            rome_main.CONTEXT_TEMPLATES_CACHE = json.loads(context_path.read_text())
        else:
            if list(output.glob('case_*.json')):
                raise RuntimeError('Context templates are missing for an existing run')
            templates = rome_main.get_context_templates(model, tok, hparams.context_template_length_params)
            atomic_json(context_path, templates)
        atomic_json(output / 'environment.json', manifest)
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(output / 'packages.txt').open('w'), check=True)
        weights = {f'{hparams.rewrite_module_tmp.format(layer)}.weight':
                   nethook.get_parameter(model, f'{hparams.rewrite_module_tmp.format(layer)}.weight')
                   for layer in hparams.layers}
        originals = {name: value.detach().cpu().clone() for name, value in weights.items()}
        for record in pending:
            case_id = record['case_id']
            # Per-case seed plus persisted context templates makes resuming reproducible.
            random.seed(seed + case_id)
            np.random.seed(seed + case_id)
            torch.manual_seed(seed + case_id)
            torch.cuda.manual_seed_all(seed + case_id)
            started = time.perf_counter()
            with (output / f'case_{case_id}.log').open('w') as log, contextlib.redirect_stdout(log):
                pre = evaluation.compute_rewrite_quality_counterfact(model, tok, record, None, None)
                try:
                    edit_start = time.perf_counter()
                    edited, _ = apply_rome_to_model(model, tok, [record['requested_rewrite']], hparams,
                                                   copy=False, return_orig_weights=False)
                    edit_seconds = time.perf_counter() - edit_start
                    post = evaluation.compute_rewrite_quality_counterfact(edited, tok, record, None, None)
                finally:
                    # Restore even if editing or evaluation raises an exception.
                    with torch.no_grad():
                        for name, value in weights.items():
                            value.copy_(originals[name].to(value.device))
                    for name, value in weights.items():
                        if not torch.equal(value.detach().cpu(), originals[name]):
                            raise RuntimeError('Original weights were not restored')
            result = {'case_id': case_id, 'requested_rewrite': record['requested_rewrite'],
                      'run_key': run_key, 'pre': pre, 'post': post, 'time': edit_seconds,
                      'elapsed_seconds': time.perf_counter()-started, 'weights_restored': True}
            validate_result(result, record, run_key)
            atomic_json(output / f'case_{case_id}.json', result)
            print(f'Case {case_id} saved; original weights verified; {result["elapsed_seconds"]:.1f}s', flush=True)
    results = [validate_result(json.loads((output/f"case_{r['case_id']}.json").read_text()), r, run_key)
               for r in records[:args.limit]]
    summary = aggregate(results)
    summary['complete_200'] = args.limit == 200 and len(results) == 200
    summary['comparison_note'] = 'Paper Table 4 uses 7,500 GPT-2 XL cases. This is a 200-case subset replication, not an exact full reproduction.'
    atomic_json(output / f'summary_{args.limit}.json', summary)
    import csv
    with (output / f'comparison_{args.limit}.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric','pre_percent','post_percent','paper_percent','difference_percentage_points','n'])
        for metric in summary['post']:
            writer.writerow([metric, summary['pre'][metric], summary['post'][metric],
                             summary['paper_gpt2_xl_7500_cases'][metric], summary['difference_percentage_points'][metric], len(results)])
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
