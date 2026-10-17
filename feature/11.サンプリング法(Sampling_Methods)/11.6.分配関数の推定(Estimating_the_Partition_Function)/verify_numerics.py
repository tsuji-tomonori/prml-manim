"""Independent analytic and Monte Carlo checks for the numbers on screen."""
import json
from math import isclose, log, pi, sqrt
import numpy as np
from partition_model import ALPHAS, EXPERIMENT, SAMPLES, SEED, energy, partition, weight

assert isclose(partition(0),sqrt(2*pi),rel_tol=1e-12)
assert isclose(partition(1),3*sqrt(2*pi),rel_tol=1e-12)
for alpha in ALPHAS:
    z=4*alpha
    assert isclose(energy(alpha,z+1)-energy(alpha,z),.5,rel_tol=1e-12)
    assert isclose(energy(alpha,z-1)-energy(alpha,z),.5,rel_tol=1e-12)
for a,b in zip(ALPHAS[:-1],ALPHAS[1:]):
    # Exact Gaussian moment of exp(-E_b+E_a), independently from partition().
    analytical=np.exp(-8*b*(1-b)+b*log(3)+8*a*(1-a)-a*log(3))
    assert isclose(analytical,partition(b)/partition(a),rel_tol=1e-12)
assert isclose(np.prod([partition(b)/partition(a) for a,b in zip(ALPHAS[:-1],ALPHAS[1:])]),3,rel_tol=1e-12)
assert len(EXPERIMENT['direct_samples'])==SAMPLES
assert max(EXPERIMENT['direct_samples'])<3
assert isclose(EXPERIMENT['direct'],.2411166458670652,rel_tol=1e-9)
assert isclose(EXPERIMENT['bridge'],2.963201834905863,rel_tol=1e-9)
result=dict(seed=SEED,samples_per_stage=SAMPLES,alpha=ALPHAS.tolist(),reference_partition=partition(0),target_partition=partition(1),exact_ratio=3.,direct_ratio=EXPERIMENT['direct'],bridge_ratio=EXPERIMENT['bridge'],direct_sample_max=float(max(EXPERIMENT['direct_samples'])),stage_ratios=EXPERIMENT['ratios'].tolist(),stage_exact_ratios=[partition(b)/partition(a) for a,b in zip(ALPHAS[:-1],ALPHAS[1:])])
open('numerical_validation.json','w').write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result)
