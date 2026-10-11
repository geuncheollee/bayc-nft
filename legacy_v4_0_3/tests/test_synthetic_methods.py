"""Exercise actual modelling helpers on synthetic data; no NFT inputs or refits."""
from pathlib import Path
import importlib.metadata, importlib.util, json, sys, unittest
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'results/image_asinh_transaction_2022_2024_eight_regressors_20260920/code'))
import run_image_experiment as image
common = image.common

class SyntheticMethodsTests(unittest.TestCase):
    def test_asinh_round_trip(self):
        training = np.array([-1.0, -.2, 0., .1, .5, 2.])
        params = common.fit_target_transform(training)
        values = np.array([-12., -.5, 0., .8, 15.])
        np.testing.assert_allclose(common.target_inverse(common.target_forward(values, params), params), values,
                                   rtol=1e-13, atol=1e-13)
        with self.assertRaises(ValueError):
            common.fit_target_transform(np.ones(5))

    def test_unique_token_tfidf_and_unseen_traits(self):
        data = {'tokens': np.array([1, 2, 3, 4]),
                'X': np.array([['red','hat'], ['blue','hat'], ['red','none'], ['unseen','new']], dtype=object)}
        prep = common.prepare_tfidf(data, np.array([0,1,2]), np.array([3]))
        duplicated = {'tokens': np.array([1,1,1,2,3,4]),
                      'X': data['X'][[0,0,0,1,2,3]]}
        again = common.prepare_tfidf(duplicated, np.arange(5), np.array([5]))
        self.assertEqual(prep['unique_training_tokens'], 3)
        np.testing.assert_allclose(prep['idf'], again['idf'])
        np.testing.assert_allclose(prep['raw_valid'], 0.)
        np.testing.assert_allclose(again['raw_valid'], 0.)
        # Neither validation vocabulary nor its repetition can change training IDFs.
        np.testing.assert_allclose(prep['idf'], np.log(4/np.array([2.,3.,3.,2.]))+1)

    def test_count_weighted_squared_loss_matches_expanded_ridge(self):
        features = np.array([[0.,1.,4.], [2.,3.,4.], [5.,-1.,4.], [1.,0.,4.]])
        index = np.array([0,0,1,2,2,2,3])
        targets = np.array([.1,.5,1.2,-.2,.2,.6,.7])
        train, valid = np.arange(6), np.array([6])
        prep = image.prepare_image({'index': index}, features, train, valid, targets)
        expanded = features[index[train]][:,prep['mask']]
        scaler = StandardScaler().fit(expanded)
        np.testing.assert_allclose(scaler.mean_, prep['scaler'].mean_)
        np.testing.assert_allclose(scaler.scale_, prep['scaler'].scale_)
        grouped = Ridge(alpha=2., solver='cholesky').fit(prep['z_unique'], prep['means'], sample_weight=prep['counts'])
        direct = Ridge(alpha=2., solver='cholesky').fit(scaler.transform(expanded), targets[train])
        np.testing.assert_allclose(grouped.predict(prep['z_valid']),
                                   direct.predict(scaler.transform(features[index[valid]][:,prep['mask']])),
                                   atol=1e-12)
        prediction = grouped.predict(prep['z_unique'])
        within = np.sum((targets[train]-prep['means'][prep['inverse']])**2)
        expanded_sse = np.sum((targets[train]-prediction[prep['inverse']])**2)
        grouped_sse = np.dot(prep['counts'], (prep['means']-prediction)**2)
        self.assertAlmostEqual(expanded_sse, grouped_sse+within, places=12)

if __name__ == '__main__':
    print(json.dumps({'python': sys.version, 'packages': {n: importlib.metadata.version(n)
          for n in ['numpy','scipy','scikit-learn','joblib','psutil','threadpoolctl','xgboost','lightgbm']}}))
    unittest.main(verbosity=2)
