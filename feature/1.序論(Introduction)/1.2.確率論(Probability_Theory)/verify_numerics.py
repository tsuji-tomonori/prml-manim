"""Check independent probability identities, fits, and final narration provenance."""
import json
import re
import unittest
import numpy as np
from probability_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry


class Verification(unittest.TestCase):
    def test_bayes(self):
        for p in np.linspace(.05,.95,21):
            j=box_joint(p)
            self.assertAlmostEqual(j.sum(),1)
            np.testing.assert_allclose(j.sum(axis=1),[p,1-p])
            np.testing.assert_allclose(box_posterior(p)*j[:,0].sum(),j[:,0])
            np.testing.assert_allclose(box_posterior(p,.75,.75),[p,1-p])
        self.assertAlmostEqual(box_posterior()[0],45/73)

    def test_density_and_moments(self):
        x=np.linspace(-12,12,20001)
        for mu,sig in [(-.7,.6),(0,.28),(.7,1.15)]:
            p=gaussian(x,mu,sig)
            self.assertAlmostEqual(np.trapezoid(p,x),1,places=9)
            self.assertAlmostEqual(np.trapezoid(x*p,x),mu,places=9)
            self.assertAlmostEqual(np.trapezoid((x-mu)**2*p,x),sig**2,places=9)
        u=np.linspace(0,1,10001)
        self.assertAlmostEqual(np.trapezoid(coin_posterior(u),u),1,places=6)
        self.assertAlmostEqual(np.trapezoid(u*coin_posterior(u),u),5/7,places=6)
        np.testing.assert_allclose(coin_posterior(u),5*coin_prior(u)*u**3)
        xx=np.linspace(-1.7,1.7,31); yy=xx**2-1
        self.assertAlmostEqual(np.mean((xx-xx.mean())*(yy-yy.mean())),0)
        self.assertGreater(np.ptp(yy),2)

    def test_maximum_likelihood_bias(self):
        mu=OBS.mean();var=OBS.var()
        best=np.log(gaussian(OBS,mu,np.sqrt(var))).sum()
        for a in np.linspace(mu-.5,mu+.5,15):
            for s in np.linspace(.5,1.4,15):
                self.assertLessEqual(np.log(gaussian(OBS,a,s)).sum(),best+1e-9)
        self.assertLess(abs(VAR_RUNNING[-1]-.5),.035)
        np.testing.assert_allclose(PAIR_VARIANCES,(PAIRS[:,0]-PAIRS[:,1])**2/4)

    def test_regression(self):
        design=phi(X)
        for alpha in [ALPHA,.1,8]:
            mean,cov=posterior(alpha)
            # Compare posterior mean with augmented least squares (independent solver).
            expected=np.linalg.lstsq(np.vstack([np.sqrt(BETA)*design,np.sqrt(alpha)*np.eye(DEGREE+1)]),np.r_[np.sqrt(BETA)*T,np.zeros(DEGREE+1)],rcond=None)[0]
            np.testing.assert_allclose(mean,expected,rtol=1e-10,atol=1e-10)
            self.assertGreater(np.linalg.eigvalsh(cov).min(),0)
        x=np.linspace(0,1,501);m,v=predictive(x)
        self.assertTrue(np.all(v>=1/BETA))
        self.assertLess(np.max(np.abs(m)+np.sqrt(v)),1.4)
        self.assertLess(np.max(np.abs(phi(x)@SAMPLE_W.T)),1.4)

    def test_narration(self):
        entries=json.loads(MANIFEST.read_text())['scenes']
        self.assertEqual(len(entries),len(SCENES))
        self.assertEqual({p.stem for p in OUTPUT_DIR.glob('*.wav')},{s['id'] for s in SCENES})
        ids=[]
        for s,e in zip(SCENES,entries):
            self.assertTrue(valid_entry(s,e),s['id'])
            for b in s['beats']:
                for cue in b['segments']:
                    ids.append(cue['id'])
                    self.assertNotRegex(cue['display'],r'エックス|ミュー|シグマ|ラムダ|ベータ|アルファ|ダブリュー|エヌ|ファイ')
        self.assertEqual(len(ids),len(set(ids)))


if __name__=='__main__':
    print(json.dumps(dict(box_posterior=box_posterior().tolist(),gaussian_mean=float(OBS.mean()),gaussian_variance=float(OBS.var()),mean_variance_4000=float(VAR_RUNNING[-1]),coin_prediction=5/7),indent=2))
    unittest.main()
