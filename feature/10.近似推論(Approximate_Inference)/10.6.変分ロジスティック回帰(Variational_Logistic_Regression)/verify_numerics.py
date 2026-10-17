"""Independent quadrature and algebra checks for the self-made experiment."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, roots_hermitenorm
from variational_logistic_model import (
    ALPHA, PHI, QUERY, NEAR, T, hierarchical, iterate, lam,
    local_bound, lower_bound, posterior, prediction,
)

root=Path(__file__).resolve().parent
fit=iterate()
xi,mean,covariance=fit['xi'],fit['mean'],fit['covariance']
assert np.min(np.diff(fit['history'])) > -1e-10
assert np.max(abs(xi**2 - np.sum((PHI@covariance)*PHI,axis=1) - (PHI@mean)**2)) < 1e-10
assert np.max(abs(covariance@np.linalg.inv(covariance)-np.eye(3))) < 1e-10
z=np.linspace(-12,12,4001)
bound_gap=expit(z)-local_bound(z,2.3)
assert bound_gap.min()>-1e-12
assert abs(float(local_bound(2.3,2.3))-float(expit(2.3)))<1e-12
assert abs(float(local_bound(-2.3,2.3))-float(expit(-2.3)))<1e-12

# A 3D Gauss-Hermite integral under the fixed Gaussian prior, independent
# of the analytic bound calculation, checks the evidence inequality.
nodes,weights=roots_hermitenorm(55)
weights=weights/np.sqrt(2*np.pi)
grid=np.stack(np.meshgrid(nodes,nodes,nodes,indexing='ij'),axis=-1).reshape(-1,3)/np.sqrt(ALPHA)
mass=np.prod(np.stack(np.meshgrid(weights,weights,weights,indexing='ij'),axis=-1),axis=-1).ravel()
scores=grid@PHI.T
log_likelihood=np.sum(np.where(T==1,-np.logaddexp(0,-scores),-np.logaddexp(0,scores)),axis=1)
log_evidence=float(np.log(np.dot(mass,np.exp(log_likelihood))))
assert fit['history'][-1] < log_evidence + 1e-8

near=prediction(NEAR,mean,covariance)
far=prediction(QUERY,mean,covariance)
far_again=prediction(QUERY,mean,covariance,order=240)
assert abs(far['gaussian_average']-far_again['gaussian_average'])<1e-8
assert abs(far['gaussian_average']-.7435306300)<1e-8
assert abs(far['representative']-.9598903163)<1e-6
assert near['gaussian_average']>far['gaussian_average']
assert abs(far['kappa_approximation']-far['gaussian_average'])<.003
hyper=hierarchical()
assert np.max(abs(hyper[-1]-hyper[-2]))<1e-10
data=dict(n=10,alpha=ALPHA,iterations=fit['iterations'],mean=mean.tolist(),covariance=covariance.tolist(),
          xi=xi.tolist(),lower_bound_history=fit['history'].tolist(),log_evidence_quadrature=log_evidence,
          near=near,far=far,hyperparameter_example=dict(a0=2,b0=2,final_alpha_mean=float(hyper[-1,0]),
                                                        shape=float(hyper[-1,1]),rate=float(hyper[-1,2])))
(root/'numerical_validation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:v for k,v in data.items() if k not in ('xi','covariance','lower_bound_history')})
