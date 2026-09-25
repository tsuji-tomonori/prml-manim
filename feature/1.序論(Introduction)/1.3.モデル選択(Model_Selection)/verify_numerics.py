"""Check statistical claims, held-out independence, and exact audio provenance."""
import json
from pathlib import Path
import unittest
import numpy as np
import model_selection as m
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry


class ModelSelectionChecks(unittest.TestCase):
    def test_ols_and_story_choices(self):
        self.assertTrue(np.all(np.diff(m.TRAIN) <= 1e-10))
        self.assertLess(m.TRAIN[-1],1e-15)
        self.assertEqual(m.SELECTED,3)
        self.assertGreater(m.VALID[-1],m.VALID[2])
        for degree,w in zip(m.DEGREES,m.WEIGHTS):
            residual=m.predict(w,m.X)-m.T
            np.testing.assert_allclose(m.design(m.X,int(degree)).T@residual,0,atol=1e-7)

    def test_cross_validation_held_out_independence(self):
        np.testing.assert_array_equal(np.sort(np.concatenate(m.FOLDS)),np.arange(24))
        self.assertEqual([len(x) for x in m.FOLDS],[6]*4)
        for held,w in zip(m.FOLDS,m.CV_WEIGHTS):
            train=np.setdiff1d(np.arange(24),held)
            changed=m.TC.copy();changed[held]+=1000
            np.testing.assert_allclose(m.fit(m.XC[train],changed[train],3),w)
        self.assertAlmostEqual(float(m.CV_SCORES.mean()),float(np.mean((m.OOF_PREDICTIONS-m.TC)**2)))
        self.assertEqual(m.CV_SELECTED,3)
        self.assertEqual(len(m.LOO_WEIGHTS),24)
        self.assertAlmostEqual(float(m.LOO_SCORES.mean()),float(np.mean((m.LOO_PREDICTIONS-m.TC)**2)))

    def test_aic_parameter_count_and_sign(self):
        for d,w,ll in zip(m.DEGREES,m.WEIGHTS,m.LOG_LIKELIHOOD):
            log_densities=-np.log(m.SIGMA*np.sqrt(2*np.pi))-(m.T-m.predict(w,m.X))**2/(2*m.SIGMA**2)
            self.assertAlmostEqual(float(log_densities.sum()),float(ll))
        self.assertEqual(int(m.DEGREES[np.argmax(m.AIC_SCORE)]),3)
        np.testing.assert_allclose(-2*m.AIC_SCORE,-2*m.LOG_LIKELIHOOD+2*(m.DEGREES+1))

    def test_curves_fit_visible_range(self):
        for w in [*m.WEIGHTS,*m.CV_WEIGHTS,*m.LOO_WEIGHTS,m.fit(m.XC,m.TC,3)]:
            self.assertLess(float(np.max(np.abs(m.predict(w,m.GRID)))),1.5)
        for t in [m.T,m.TV,m.TT,m.TC]:
            self.assertLess(float(np.max(np.abs(t))),1.5)

    def test_narration_manifest(self):
        entries=json.loads(MANIFEST.read_text())['scenes']
        self.assertEqual({p.stem for p in OUTPUT_DIR.glob('*.wav')},{s['id'] for s in SCENES})
        for s,e in zip(SCENES,entries):
            self.assertTrue(valid_entry(s,e),s['id'])
            for i,b in enumerate(s['beats']):
                self.assertGreater(e['beat_durations'][i]-e['beat_speech_ends'][i],.3)
                self.assertLess(e['beat_durations'][i]-e['beat_speech_ends'][i],.43)
        self.assertEqual(sum(len(b['segments']) for s in SCENES for b in s['beats']),122)


if __name__=='__main__':
    unittest.main()
