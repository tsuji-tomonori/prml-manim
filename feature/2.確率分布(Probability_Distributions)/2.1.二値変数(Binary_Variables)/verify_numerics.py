"""Independent identities and final asset consistency checks."""
import itertools
import json
import unittest
import numpy as np
from binary_model import *
from make_voicevox_narration import MANIFEST, valid_entry
from narration_content import SCENES

class Numerics(unittest.TestCase):
    def test_bernoulli_and_binomial(self):
        for mu in [.1,.25,.5,.7,.95]:
            p=bernoulli(mu);x=np.arange(2)
            self.assertAlmostEqual(p@x,mu)
            self.assertAlmostEqual(p@((x-mu)**2),mu*(1-mu))
            q=binomial(8,mu);m=np.arange(9)
            enum=np.zeros(9)
            for obs in itertools.product([0,1],repeat=8):
                enum[sum(obs)]+=np.prod([p[x] for x in obs])
            np.testing.assert_allclose(q,enum,atol=1e-14)
            self.assertAlmostEqual(q.sum(),1)
            self.assertAlmostEqual(q@m,8*mu)
            self.assertAlmostEqual(q@((m-8*mu)**2),8*mu*(1-mu))
    def test_likelihood(self):
        xs=np.linspace(0,1,10001)
        self.assertAlmostEqual(xs[np.argmax(likelihood(xs))],5/8)
        for mu in [.1,.5,.8]:
            self.assertAlmostEqual(likelihood(mu),np.prod(bernoulli(mu)[OBS]))
        self.assertEqual(np.argmax(likelihood(xs,3,3)),len(xs)-1)
    def test_beta_and_prediction(self):
        xs=np.linspace(0,1,100001)
        for a,b in [(1,1),(3,2),(9,6),(8,5),(9,1),(9,2),(10,1)]:
            p=beta_pdf(xs,a,b)
            self.assertAlmostEqual(np.trapezoid(p,xs),1,places=8)
            self.assertAlmostEqual(np.trapezoid(xs*p,xs),beta_mean(a,b),places=8)
            self.assertAlmostEqual(np.trapezoid((xs-beta_mean(a,b))**2*p,xs),beta_var(a,b),places=8)
        # Cosine substitution resolves the integrable singularities at both ends.
        nodes=200000
        t=(np.arange(nodes)+.5)*np.pi/nodes
        x=(1-np.cos(t))/2
        mass=np.sum(beta_pdf(x,.7,.7)*np.sin(t)/2)*np.pi/nodes
        self.assertAlmostEqual(mass,1,places=7)
        self.assertGreater(beta_pdf(.005,.7,.7),beta_pdf(.5,.7,.7))
        self.assertAlmostEqual(beta_mean(*posterior()),8/13)
    def test_sequential_conjugacy(self):
        a,b=PRIOR
        for x in OBS: a+=x;b+=1-x
        np.testing.assert_allclose((a,b),posterior())
        xs=np.linspace(.001,.999,10000)
        product=beta_pdf(xs,*PRIOR)*likelihood(xs)
        ratio=product/beta_pdf(xs,a,b)
        np.testing.assert_allclose(ratio,ratio[0],rtol=1e-12)
        np.testing.assert_allclose(posterior(OBS[::-1]),(a,b))
    def test_total_variance(self):
        r=variance_decomposition(9,1)
        self.assertGreater(r['variances'][0],r['prior'])
        self.assertAlmostEqual(r['mean'],.9)
        self.assertAlmostEqual(r['remaining']+r['moved'],r['prior'])
        self.assertLess(r['remaining'],r['prior'])
    def test_audio(self):
        entries=json.loads(MANIFEST.read_text())['scenes']
        self.assertEqual(len(entries),len(SCENES))
        for s,e in zip(SCENES,entries): self.assertTrue(valid_entry(s,e),s['id'])

if __name__=='__main__': unittest.main()
