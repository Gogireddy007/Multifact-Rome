"""Verify archived records and independently recompute the three reported scores."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    folder = ROOT/'results/baseline_200'
    manifest = json.loads((folder/'manifest.json').read_text())
    summary = json.loads((folder/'summary_200.json').read_text())
    run_key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    paths = sorted(folder.glob('case_*.json'), key=lambda p:int(p.stem.split('_')[1]))
    assert len(paths) == 200
    records = [json.loads(p.read_text()) for p in paths]
    assert [r['case_id'] for r in records] == list(range(200)) == summary['case_ids']
    for r in records:
        assert r['run_key'] == run_key and r['weights_restored'] is True
        assert r['elapsed_seconds'] >= 0 and r['time'] >= 0
    for phase in ['pre','post']:
        for metric,key in [('efficacy','rewrite_prompts_probs'),('paraphrase','paraphrase_prompts_probs'),('neighborhood','neighborhood_prompts_probs')]:
            values=[]
            for r in records:
                pairs=r[phase][key]
                assert pairs
                assert all(math.isfinite(v) for p in pairs for v in p.values())
                values.append(sum((p['target_true']<p['target_new']) if metric=='neighborhood' else (p['target_new']<p['target_true']) for p in pairs)/len(pairs))
            actual=sum(values)*100/len(values)
            assert math.isclose(actual,summary[phase][metric],abs_tol=1e-9),(metric,actual)
    for name,key in [('run_baseline.py','runner_sha256'),('baseline_support.py','support_sha256')]:
        assert hashlib.sha256((ROOT/'provenance'/name).read_bytes()).hexdigest()==manifest[key]
    assert summary['complete_200'] is True and summary['num_cases']==200
    print('PASS: 200 unique cases; run keys; restoration flags; finite metrics; archived source hashes; recomputed scores.')
    print('Post-edit percentages:',summary['post'])
    print('Recorded case execution minutes:',round(sum(r['elapsed_seconds'] for r in records)/60,2))

if __name__=='__main__':main()
