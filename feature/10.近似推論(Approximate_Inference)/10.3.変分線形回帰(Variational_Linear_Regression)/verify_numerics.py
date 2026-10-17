"""Independent checks of the displayed updates and prediction."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from regression_model import (N,DIM,A0,B0,BETA,X,T,PROBE,TRUE_W,
                              phi,step,fit,prediction,elbo)

history=fit()
last=history[-1]
design=phi(X)
assert len(X)==len(T)==N==10 and DIM==4
assert A0>0 and B0>0 and BETA>0
for s in history:
    assert np.all(np.linalg.eigvalsh(s['covariance'])>0)
    assert np.isclose(s['a'],A0+DIM/2)
    assert np.isclose(s['b'],B0+.5*(s['mean']@s['mean']+np.trace(s['covariance'])))
    assert np.allclose(np.linalg.inv(s['covariance']),s['input_alpha']*np.eye(DIM)+BETA*design.T@design)
assert min(np.diff([s['bound'] for s in history])) > -1e-10
assert abs(last['alpha']-1.8824003996369507)<1e-7
assert abs(last['bound']+10.274660530339965)<1e-7
p=prediction(last)
assert abs(p['mean']-2.3468890698152314)<1e-7
assert abs(p['weight_variance']-.5031294778067548)<1e-7
assert abs(p['noise_variance']-.09)<1e-12
total=p['weight_variance']+p['noise_variance']
assert abs(quad(lambda y:norm.pdf(y,loc=p['mean'],scale=np.sqrt(total)), -np.inf,np.inf)[0]-1)<1e-9
assert abs(p['true_mean']-(phi([PROBE])[0]@TRUE_W))<1e-12
# The variance sum also follows directly from independent Gaussian draws.
rng=np.random.default_rng(10304)
draws=rng.multivariate_normal(last['mean'],last['covariance'],size=250000)@phi([PROBE])[0]
draws+=rng.normal(0,1/np.sqrt(BETA),len(draws))
assert abs(draws.var()-total)<.012
result=dict(n=N,dimension=DIM,beta=BETA,seed=1030,probe=PROBE,
            fixed_weak=prediction(step(.01)),fixed_strong=prediction(step(100.)),
            mean_alpha=last['alpha'],bound_initial=history[0]['bound'],bound_final=last['bound'],
            prediction=p,total_variance=total,standard_deviation=np.sqrt(total),
            monte_carlo_variance=float(draws.var()),
            max_negative_bound_step=float(min(0,min(np.diff([s['bound'] for s in history])))))
Path('numerical_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('PASS',result)
