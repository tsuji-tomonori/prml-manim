"""Recompute all displayed numbers and verify PPCA identities."""
import json
from pathlib import Path
import numpy as np
from pca_model import POINTS, N, D, MEAN, COV, EIGENVALUES, EIGENVECTORS
from ppca_model import *

assert N == 13 and D == 2 and M == 1
assert np.allclose(MEAN, [1.15, .35])
assert np.allclose(MODEL_COV, COV, atol=1e-12)
assert np.isclose(SIGMA2, LAMBDA2)
assert np.isclose(np.linalg.norm(W)**2+SIGMA2, LAMBDA1)
assert np.isclose(np.linalg.norm(A-MEAN), 0)
assert np.isclose(np.linalg.norm(B-MEAN), 6)
assert np.isclose(density(B)/density(A), DENSITY_RATIO, rtol=1e-12)
assert np.isclose(DENSITY_RATIO, .010282158690519924)
assert np.isclose(posterior(B)[1], POSTERIOR_VARIANCE)
assert posterior(B)[0] < 6/np.linalg.norm(W)
assert np.allclose(EIGENVECTORS.T @ EIGENVECTORS, np.eye(2))
assert max((POINTS-MEAN) @ U) < 6
records = em_fit()
assert abs(records[-1]['angle']-np.degrees(np.arctan2(U[1],U[0]))) < .05
assert abs(records[-1]['sigma2']-SIGMA2) < .01
loglik=[]
for record in records:
    em_cov=np.outer(record['w'],record['w'])+record['sigma2']*np.eye(D)
    sign,logdet=np.linalg.slogdet(em_cov)
    assert sign>0
    quadratic=np.einsum('ni,ij,nj->',POINTS-MEAN,np.linalg.inv(em_cov),POINTS-MEAN)
    loglik.append(float(-.5*(N*(D*np.log(2*np.pi)+logdet)+quadratic)))
assert np.all(np.diff(loglik)>=-1e-9)
assert np.isclose(missing_x2_given_x1(MEAN[0]), MEAN[1])
result = dict(count=N, mean=MEAN.tolist(), eigenvalues=EIGENVALUES.tolist(),
              W=W.tolist(), sigma2=SIGMA2, model_covariance=MODEL_COV.tolist(),
              A=A.tolist(), B=B.tolist(), reconstruction_error_A=0., reconstruction_error_B=0.,
              density_A=density(A), density_B=density(B), density_ratio=DENSITY_RATIO,
              posterior_B=dict(mean=posterior(B)[0],variance=posterior(B)[1]),
              em=[dict(step=r['step'], angle=r['angle'],sigma2=r['sigma2'],log_likelihood=loglik[i]) for i,r in enumerate(records)],
              missing_example=dict(x1=2.5,mean_x2=missing_x2_given_x1(2.5)), check='passed')
(Path(__file__).parent/'numerical_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
