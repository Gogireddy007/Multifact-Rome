import copy
import json
from pathlib import Path
import tempfile
import unittest

from baseline_support import atomic_json
from multi_support import (SCHEMA, check_manifest, digest, execute_protocol, fact_spec,
                           file_hash, load_groups, orders_for, pack_evaluation, retention,
                           summarize, validate_result)
from verify_multi_results import verify_folder, write_exports


def record(cid):
    return {'case_id': cid, 'requested_rewrite': {'subject': 'Test subject', 'prompt': '{} is',
            'relation_id': f'P{cid}', 'target_new': {'str': 'new', 'id': 'Q1'},
            'target_true': {'str': 'old', 'id': 'Q2'}},
            'paraphrase_prompts': ['Who?'], 'neighborhood_prompts': ['Other?'], 'attribute_prompts': []}


class MultiTests(unittest.TestCase):
    def setUp(self):
        self.records = {i: record(i) for i in range(1, 5)}
        self.group = load_groups([{'subject': 'Test subject', 'case_ids': [1,2], 'similarity': 'related'}], self.records)[0]
        self.specs = {str(c): fact_spec(self.records[c]) for c in [1,2]}

    def evaluate(self, cid, successful=True):
        pair = {'target_new': 1.0 if successful else 3.0, 'target_true': 2.0}
        raw = {kind+'_prompts_probs': [dict(pair) for _ in prompts] for kind,prompts in fact_spec(self.records[cid])['prompts'].items()}
        return pack_evaluation(raw, fact_spec(self.records[cid]))

    def result(self, order='forward'):
        ids = self.group['case_ids']
        edit_order = orders_for(ids, order)[order]
        state = []
        def reset(): state.clear()
        def edit(cid): state.append(cid)
        def evaluate(cid): return self.evaluate(cid, cid in state)
        stages, controls, times = execute_protocol(ids, edit_order, evaluate, edit, reset, lambda cid: None)
        return {'schema': SCHEMA, 'group': copy.deepcopy(self.group), 'order': order, 'edit_order': edit_order,
                'facts': copy.deepcopy(self.specs), 'run_key': 'test', 'stages': stages, 'single_edit_controls': controls,
                'weights_restored': True, 'seeds': {str(c):32+c for c in ids}, 'edit_seconds': times,
                'elapsed_seconds': 1.0, 'retention': retention(stages, edit_order)}

    def validate(self, r):
        return validate_result(r, self.group, self.specs, 'test', r['order'])

    def test_valid_both_orders(self):
        for order in ('forward','reverse'):
            r = self.validate(self.result(order))
            self.assertTrue(r['retention']['all_edits_hold'])
            self.assertEqual(r['group']['similarity'], 'related')

    def test_groups_reject_bad_inputs(self):
        for ids in ([1], [1,1], [1,9], [1,2,3,4,5], [True,2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                load_groups([{'subject':'Test subject','case_ids':ids}], self.records)
        with self.assertRaises(ValueError):
            load_groups([{'subject':'Wrong subject','case_ids':[1,2]}], self.records)
        with self.assertRaises(ValueError):
            load_groups([self.group,self.group], self.records)
        self.records[2]['requested_rewrite']['relation_id'] = 'P1'
        with self.assertRaises(ValueError):
            load_groups([self.group], self.records)

    def test_pilot_wrapper_and_three_four_edits(self):
        for count in (3,4):
            ids = list(range(1,count+1))
            group = load_groups({'pilot':[{'subject':'Test subject','case_ids':ids}]},self.records)[0]
            for order in ('forward','reverse'):
                calls, state, seeds = [], [], []
                def reset(): state.clear()
                def edit(cid): state.append(cid); calls.append(cid)
                def evaluate(cid): return self.evaluate(cid,cid in state)
                sequence = orders_for(ids,order)[order]
                stages, controls, _ = execute_protocol(ids,sequence,evaluate,edit,reset,seeds.append)
                self.assertEqual(calls,sequence+ids)
                self.assertEqual(seeds,sequence+ids)
                self.assertEqual(stages[-1]['applied_case_ids'],sequence)
                self.assertEqual(len(stages),count+1)
                self.assertEqual(len(controls),count)
                self.assertEqual(state,[])
                self.assertTrue(retention(stages,sequence)['all_edits_hold'])

    def test_restore_on_edit_failure(self):
        state = []
        def edit(cid):
            state.append(cid)
            raise RuntimeError('edit failed')
        with self.assertRaises(RuntimeError):
            execute_protocol([1,2],[1,2],self.evaluate,edit,state.clear,lambda c:None)
        self.assertEqual(state,[])

    def test_restore_on_evaluation_failure(self):
        state = []
        def evaluate(cid):
            if state: raise RuntimeError('evaluation failed')
            return self.evaluate(cid)
        with self.assertRaises(RuntimeError):
            execute_protocol([1,2],[1,2],evaluate,state.append,state.clear,lambda c:None)
        self.assertEqual(state,[])

    def test_restore_failure_prevents_result(self):
        def reset(): raise RuntimeError('restore failed')
        with self.assertRaises(RuntimeError):
            execute_protocol([1,2],[1,2],self.evaluate,lambda c:None,reset,lambda c:None)

    def test_corrupt_evaluations_rejected(self):
        for field,value in [('target_new',float('nan')),('target_true',-1),('p_new',0.99),('prompt','Wrong')]:
            r=self.result()
            r['stages'][0]['evaluations']['1']['rewrite_prompts_probs'][0][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(r)

    def test_cached_scores_and_flags_rejected(self):
        r=self.result(); r['stages'][0]['evaluations']['1']['scores']['efficacy']=88
        with self.assertRaises(ValueError): self.validate(r)
        r=self.result(); r['retention']['all_edits_hold']=False
        with self.assertRaises(ValueError): self.validate(r)

    def test_identity_coverage_and_restoration_rejected(self):
        mutations = [lambda r:r.update(run_key='other'),lambda r:r.update(weights_restored=False),
                     lambda r:r['stages'].pop(),lambda r:r['single_edit_controls'].pop('1'),
                     lambda r:r['stages'][0]['evaluations'].pop('2'),lambda r:r.update(edit_order=[2,1]),
                     lambda r:r['facts']['1']['requested_rewrite'].update(subject='Wrong')]
        for mutation in mutations:
            r=self.result(); mutation(r)
            with self.assertRaises(ValueError): self.validate(r)

    def test_ties_are_failures_and_neighborhood_direction(self):
        r=self.evaluate(1)
        self.assertEqual(r['scores']['efficacy'],100)
        self.assertEqual(r['scores']['neighborhood'],0)
        raw={kind+'_prompts_probs':[{'target_new':2.,'target_true':2.} for _ in prompts]
             for kind,prompts in self.specs['1']['prompts'].items()}
        self.assertEqual(pack_evaluation(raw,self.specs['1'])['scores']['efficacy'],0)

    def test_conditional_loss_denominator(self):
        r=self.result()
        r['stages'][-1]['evaluations']['1']=self.evaluate(1,False)
        r['retention']=retention(r['stages'],r['edit_order'])
        self.validate(r)
        s=summarize([r],1,'forward')['orders']['forward']
        self.assertEqual(s['first_edit_loss_percent_eligible'],100)
        r['stages'][1]['evaluations']['1']=self.evaluate(1,False)
        r['retention']=retention(r['stages'],r['edit_order'])
        s=summarize([r],1,'forward')['orders']['forward']
        self.assertIsNone(s['first_edit_loss_percent_eligible'])
        self.assertEqual(s['first_edit_loss_count'],0)

    def test_manifest_mismatch(self):
        check_manifest({'a':1},{'a':1})
        with self.assertRaises(ValueError): check_manifest({'a':1},{'a':2})

    def test_folder_verifier_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            manifest={'schema':SCHEMA,'groups':[self.group],'facts_sha256':digest(self.specs),
                      'context_templates_sha256':digest(['{}']),'sources':{},'order':'both'}
            for name,value in [('manifest',manifest),('groups',[self.group]),('facts',self.specs),('context_templates',['{}'])]:
                atomic_json(folder/(name+'.json'),value)
            with self.assertRaises(ValueError): verify_folder(folder,1)
            for order in ('forward','reverse'):
                r=self.result(order); r['run_key']=digest(manifest)
                atomic_json(folder/f'group_001_{order}.json',r)
            report,results=verify_folder(folder,1)
            self.assertEqual(report['num_group_orders'],2)
            write_exports(folder,report,results)
            self.assertEqual(verify_folder(folder,1)[0],report)
            # Existing result is checked again, never blindly skipped.
            r['stages'][-1]['evaluations']['1']['scores']['efficacy']=42
            atomic_json(folder/'group_001_reverse.json',r)
            with self.assertRaises(ValueError): verify_folder(folder,1)

    def test_unexpected_result_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            manifest={'schema':SCHEMA,'groups':[self.group],'facts_sha256':digest(self.specs),
                      'context_templates_sha256':digest(['{}']),'sources':{},'order':'forward'}
            for name,value in [('manifest',manifest),('groups',[self.group]),('facts',self.specs),('context_templates',['{}']),('group_999_forward',{})]:
                atomic_json(folder/(name+'.json'),value)
            with self.assertRaises(ValueError): verify_folder(folder,1)


if __name__ == '__main__': unittest.main()
