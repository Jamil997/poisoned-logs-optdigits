"""Tests specific to the new candidate construction and information boundary."""
import json, unittest
import numpy as np
from experiment import ROOT, build_specs, sha
from prepare import interpolate, probabilities

class BoundaryChecks(unittest.TestCase):
    def test_analytic_mixture_hits_target_and_clips(self):
        for gap in [-.01, .01]:
            w, clipped = interpolate(.75, .95, .89, gap)
            self.assertFalse(clipped)
            self.assertAlmostEqual((1-w)*.75+w*.95-.89, gap)
        self.assertEqual(interpolate(.75,.95,.89,.1), (1.,True))

    def test_frozen_selection_and_evaluation_are_consistent(self):
        s = json.loads((ROOT/'selection/selection.json').read_text())
        d = dict(np.load(ROOT/'frozen_models/models.npz'))
        meta = json.loads((ROOT/'frozen_models/models.json').read_text())
        self.assertEqual(s['protocol_sha256'],sha(ROOT/'protocol.json'))
        self.assertEqual(s['construction_source_sha256'],sha(ROOT/'prepare.py'))
        self.assertEqual(meta['selection_metadata_sha256'],sha(ROOT/'selection/selection.json'))
        self.assertLess(s['frozen_utc'],meta['evaluation_utc'])
        x = (d['features']-d['scaler_mean'])/d['scaler_scale']
        for j,item in enumerate(s['mixtures']):
            low = s['validation_parent_values'][item['weak']]
            high = s['validation_parent_values'][item['strong']]
            base = s['validation_parent_values'][s['incumbent']]
            w,clipped = interpolate(low,high,base,item['target_validation_advantage'])
            self.assertEqual(w,item['strong_weight']); self.assertEqual(clipped,item['clipped'])
            reconstructed=(1-w)*probabilities(d,x,item['weak'])+w*probabilities(d,x,item['strong'])
            np.testing.assert_allclose(d['p'][j+1],reconstructed,atol=1e-14)
        old = np.load(ROOT.parent/'security/frozen_models/models.npz')
        np.testing.assert_array_equal(d['p'][0],old['p'][0])
        np.testing.assert_array_equal(d['q'],old['q'])
        self.assertFalse(set(d['fit_indices']) & set(d['validation_indices']))

    def test_defender_requires_no_test_labels(self):
        d = dict(np.load(ROOT/'frozen_models/models.npz'))
        d.pop('y'); config=json.loads((ROOT/'protocol.json').read_text())
        mu,specs,_=build_specs(d,5000,.2,config)
        self.assertEqual(mu.shape,(1797,10));self.assertEqual(len(specs),6)
        for s in specs.values(): self.assertTrue(np.isfinite(s['rad']).all())

if __name__=='__main__': unittest.main(verbosity=2)
