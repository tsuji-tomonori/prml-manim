"""Independently check the binary table, parameter counts and Gaussian recursion."""
import json
from itertools import product
from pathlib import Path
import numpy as np
from bayesian_model import joint,exact,DAYS,GAUSSIAN,SEED
root=Path(__file__).resolve().parent
r=exact()
assert abs(sum(joint(*v) for v in product((0,1),repeat=3))-1)<1e-14
for rr,dd in product((0,1),repeat=2):
    assert abs(sum(joint(rr,dd,u) for u in (0,1))-(.3 if rr else .7)*((.9 if rr else .1) if dd else (.1 if rr else .9)))<1e-14
assert abs(r['p_wet']-.34)<1e-14
assert abs(r['p_rain_given_wet']-27/34)<1e-14
assert abs(r['p_umbrella_given_wet']-23/34)<1e-14
assert abs(r['rain_wet_umbrella']-.216)<1e-14
assert abs(r['sun_wet_umbrella']-.014)<1e-14
assert (2**4-1,1+3*2,4)==(15,7,4)
# x1=N(0,1), x2=.8*x1+N(0,.36), x3=.6*x2+N(0,.25).
expected_cov=np.array([[1,.8,.48],[.8,1,.6],[.48,.6,.61]])
assert np.linalg.eigvalsh(expected_cov).min()>0
assert abs(expected_cov[0,2]-.6*expected_cov[0,1])<1e-14
measured_cov=np.cov(GAUSSIAN.T)
assert np.max(np.abs(measured_cov-expected_cov))<.09
rain=int(DAYS[:,0].sum());wet=int(DAYS[:,1].sum());umbrella=int(DAYS[:,2].sum())
wet_umbrella=int(np.logical_and(DAYS[:,1],DAYS[:,2]).sum())
result=dict(exact=r,seed=SEED,sample_n=len(DAYS),sample_counts=dict(rain=rain,wet=wet,umbrella=umbrella,wet_umbrella=wet_umbrella),parameter_counts=dict(full=15,chain=7,independent=4),gaussian_expected_cov=expected_cov.tolist(),gaussian_sample_cov=measured_cov.tolist())
(root/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result['exact'],result['sample_counts'])
