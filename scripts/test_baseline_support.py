import copy
import json
import tempfile
import unittest
from pathlib import Path
from baseline_support import aggregate, atomic_json, scores, validate_result

def fixture(case_id=1):
    phase = {'rewrite_prompts_probs':[{'target_new':1.,'target_true':2.}],
             'paraphrase_prompts_probs':[{'target_new':1.,'target_true':2.},{'target_new':2.,'target_true':1.}],
             'neighborhood_prompts_probs':[{'target_new':2.,'target_true':1.}]}
    return {'case_id':case_id,'requested_rewrite':{'subject':'X'}, 'run_key':'abc',
            'weights_restored':True,'pre':copy.deepcopy(phase),'post':copy.deepcopy(phase)}

class ResultTests(unittest.TestCase):
    def test_directions_and_ties(self):
        result = fixture()
        self.assertEqual(scores(result, 'post'), {'efficacy':100.,'paraphrase':50.,'neighborhood':100.})
        result['post']['rewrite_prompts_probs'][0]['target_true'] = 1.
        self.assertEqual(scores(result,'post')['efficacy'],0.)
    def test_average_per_fact_not_pooled_prompts(self):
        a,b = fixture(1),fixture(2)
        b['post']['paraphrase_prompts_probs'] = [{'target_new':2.,'target_true':1.}]*8
        self.assertEqual(aggregate([a,b])['post']['paraphrase'],25.)
    def test_bad_resume_rejected(self):
        r = fixture()
        for change in [{'run_key':'other'}, {'weights_restored':False}, {'case_id':3}]:
            broken = dict(r,**change)
            with self.assertRaises(ValueError): validate_result(broken,r,'abc')
    def test_invalid_metrics_rejected(self):
        r = fixture()
        r['post']['rewrite_prompts_probs'][0]['target_new'] = float('nan')
        with self.assertRaises(ValueError): scores(r,'post')
    def test_empty_and_duplicate_results(self):
        with self.assertRaises(ValueError): aggregate([])
        with self.assertRaises(ValueError): aggregate([fixture(),fixture()])
    def test_atomic_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'case.json'
            atomic_json(p, fixture())
            self.assertEqual(json.loads(p.read_text()),fixture())
            self.assertFalse(p.with_suffix('.json.tmp').exists())

if __name__ == '__main__': unittest.main()
