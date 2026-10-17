"""Independent checks on the model values shown on screen."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from sampling_model import *

root=Path(__file__).resolve().parent
true=truth()
independent=quad(lambda z:float(target(z)),THRESHOLD,np.inf,epsabs=1e-12)[0]
normalization=quad(lambda z:float(target(z)),-np.inf,np.inf,epsabs=1e-12)[0]
assert abs(true-independent)<1e-10
assert abs(normalization-1)<1e-10
assert .179<true<.181
intro=introductory_draws()
assert len(intro)==24 and np.count_nonzero(intro>1)==0
z,w=importance_draws()
assert len(z)==200 and abs(w.sum()-1)<1e-12
weighted=float(np.dot(w,z>1))
assert .16<weighted<.17
k,where,peak=envelope()
grid=np.linspace(-20,20,200001)
assert np.max(target(grid)/proposal(grid))<k
assert 8<k<9 and 2<where<2.5
rz,u,accepted=rejection_draws()
assert np.all(u[accepted]<target(rz[accepted])/(k*proposal(rz[accepted])))
resampled=resampled_draws()
assert all(np.any(z==v) for v in resampled)
result=dict(true_probability=true,independent_integral=independent,target_integral=normalization,introductory_count=len(intro),introductory_right_count=int(np.count_nonzero(intro>1)),importance_count=len(z),importance_right_count=int(np.count_nonzero(z>1)),importance_estimate=weighted,importance_effective_sample_size=float(1/np.dot(w,w)),rejection_envelope=k,rejection_envelope_peak=peak,rejection_envelope_at=where,theoretical_acceptance=1/k,rejection_proposals=len(rz),rejection_accepted=int(accepted.sum()),rejection_estimate=float(np.mean(rz[accepted]>1)),resampling_count=len(resampled),resampling_right_count=int(np.count_nonzero(resampled>1)),resampling_distinct_count=len(np.unique(resampled)),high_dimension_acceptance=float(1/1.01**1000))
(root/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(result)
