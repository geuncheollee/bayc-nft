"""Public, data-free consistency checks. Not a raw-data scientific replication."""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, math, re, sys, unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

def read(rel):
    return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))

def rows(rel):
    with (ROOT/rel).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

RUNS = read('provenance/run_index.json')

class PublicReleaseTests(unittest.TestCase):
    def test_integrity(self):
        records = read('MANIFEST_SHA256.json')
        self.assertGreater(len(records), 100)
        for record in records:
            p = ROOT/record['path']
            self.assertTrue(p.is_file(), str(p))
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), record['sha256'], record['path'])

    def test_original_source_hashes(self):
        for record in read('provenance/source_files.json'):
            digest = hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()
            self.assertEqual(digest, record['source_sha256'], record['path'])
            self.assertEqual(digest, record['public_sha256'], record['path'])

    def test_python_syntax_and_public_boundary(self):
        forbidden = {'.jsonl', '.npy', '.npz', '.joblib', '.pkl', '.parquet', '.docx', '.pdf'}
        secret = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
        for p in ROOT.rglob('*'):
            if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:
                continue
            if 'reproduced_tables' in p.parts:
                continue
            self.assertNotIn(p.suffix.lower(), forbidden, str(p))
            self.assertNotIn('response_source', p.name.lower())
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            if p.suffix in {'.py', '.json', '.md', '.csv', '.txt', '.cff', '.yml'}:
                self.assertIsNone(secret.search(p.read_text(encoding='utf-8-sig')), str(p))

    def test_frozen_before_evaluation(self):
        for group in ('metadata', 'image', 'early', 'late'):
            record = read(RUNS[group]+'/selection_freeze.json')
            value = record.get('evaluation_labels_loaded_before_freeze',
                               record.get('evaluation_predictions_loaded_before_freeze'))
            self.assertIs(value, False, group)
        audit = read(RUNS['supplemental']+'/stage1_evaluation_audit.json')
        self.assertIs(audit['claim_policy']['study_level_single_use_untouched_test'], False)

    def test_development_selected_representatives(self):
        names = {'metadata': 'fixed_out_of_time_evaluation_32_combinations.csv',
                 'image': 'fixed_out_of_time_image_evaluation_112_combinations.csv',
                 'early': 'fixed_out_of_time_early_evaluation_112_combinations.csv',
                 'late': 'fixed_out_of_time_late_evaluation_112_combinations.csv'}
        summary = read('summary/performance.json')
        self.assertEqual(len(summary), 10)
        for group, name in names.items():
            candidates = rows(RUNS[group]+'/'+name)
            if group == 'metadata':
                candidates = [r for r in candidates if r['encoding'] == 'TF-IDF']
            self.assertEqual(len(candidates), 16 if group == 'metadata' else 112)
            for collection in ('BAYC', 'MAYC'):
                candidates_c = [r for r in candidates if r['collection'] == collection]
                best = min(candidates_c, key=lambda r: float(r['pre2025_validation_transaction_rmse']))
                reported = next(r for r in summary if r['collection'] == collection and r['group'] == group)
                for metric in ('rmse', 'mae', 'r2'):
                    self.assertAlmostEqual(float(best['evaluation_transaction_'+metric]),
                                           reported['transaction_'+metric], places=11)

    def test_grid_and_weights(self):
        spec = read(RUNS['early']+'/experiment_specification.json')
        self.assertEqual(len(spec['families']), 8)
        self.assertEqual(len(spec['encoders']), 7)
        self.assertEqual(sum(len(spec['grids'][f]) for f in spec['families']), 59)
        weight_rows = rows(RUNS['late']+'/pre2025_weight_grid_1232_combinations.csv')
        self.assertEqual(len(weight_rows), 1232)
        self.assertEqual({float(r['image_weight']) for r in weight_rows}, {i/10 for i in range(11)})

    def test_uncertainty_and_strict_refits(self):
        pairs = rows('summary/paired_comparisons.csv')
        self.assertEqual(len(pairs), 18)  # 2 collections x 3 contrasts x 3 metric/scheme combinations.
        for r in pairs:
            self.assertAlmostEqual(float(r['candidate_rmse'])-float(r['reference_rmse']),
                                   float(r['delta_rmse_candidate_minus_reference']), places=12)
            self.assertLessEqual(float(r['delta_rmse_ci95_low']), float(r['delta_rmse_ci95_high']))
        zero = read('summary/zero_uncertainty.json')
        self.assertTrue(all(float(r['delta_rmse_ci99_375_high']) < 0 for r in zero))
        strict = rows('summary/stage2_refit_token_intervals.csv')
        self.assertEqual(len(strict), 6)
        for r in strict:
            self.assertAlmostEqual(float(r['early_rmse'])-float(r['metadata_rmse']),
                                   float(r['early_minus_metadata_rmse']), places=12)
            if r['training_cohort'] != 'full':
                self.assertLess(float(r['paired_token_ci95_low']), 0)
                self.assertGreater(float(r['paired_token_ci95_high']), 0)

    def test_past_only_target_unit_checks(self):
        path = ROOT/'revision/code/build_past_only_targets.py'
        spec = importlib.util.spec_from_file_location('public_target', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(all(module.tests().values()))

if __name__ == '__main__':
    unittest.main(verbosity=2)
