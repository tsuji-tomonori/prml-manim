"""Verify every numerical claim and a distributional check of the sampler."""
import json, math
import numpy as np
from scipy.integrate import quad
from slice_model import f,boundary,expanded_interval,fixed_demo,slice_step
from pathlib import Path
root=Path(__file__).resolve().parent
Z=.9*math.sqrt(2*math.pi)
assert abs(quad(f,-math.inf,math.inf)[0]-Z)<1e-10
assert abs(f(0)-1.25)<1e-12
assert abs(f(1.4)-.19265878380400744)<1e-12
assert abs(f(2.)-.13490324234059642)<1e-12
assert expanded_interval()==(-2.2,2.3)
assert f(1.4)>.15>f(2.)
assert boundary(.95)<boundary(.6)<boundary(.15)<boundary(.05)
assert abs(quad(lambda u:1/Z,0,float(f(1.4)))[0]-f(1.4)/Z)<1e-12
rng=np.random.default_rng(114)
z=0.; samples=[]
for i in range(32000):
    z=slice_step(z,rng)
    if i>=2000:samples.append(z)
samples=np.array(samples)
assert abs(samples.mean())<.07,samples.mean()
assert abs(samples.var()-1.72125)<.13,samples.var()
result=dict(Z=Z,demo=fixed_demo(),f0=float(f(0)),f14=float(f(1.4)),f20=float(f(2.)),boundaries={str(h):boundary(h) for h in [.95,.6,.15,.05]},sample_count=len(samples),sample_mean=float(samples.mean()),sample_variance=float(samples.var()),expected_variance=1.72125,distribution_check='passed')
(root/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result)
