"""Independent numerical checks of the story's ridge/dual calculations."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from dual_model import X,T,LAMBDA,PROBE,PHI,K,W,A,K_PROBE,PREDICTION,CONTRIBUTIONS,kernel,phi,primal,dual

assert [F(str(v)) for v in X] == [F(-1),F(0),F(1)]
assert [F(str(v)) for v in T] == [F(1),F(0),F(1)]
expected_w=np.array([F(4,19), F(0), F(12,19)],dtype=float)
expected_a=np.array([F(6,19), F(-8,19), F(6,19)],dtype=float)
expected_k=np.array([F(3,4),F(1),F(7,4)],dtype=float)
expected_prediction=float(F(7,19))
assert np.allclose(PHI,[[1,-1,1],[1,0,0],[1,1,1]],atol=1e-14)
assert np.allclose(K,[[3,1,1],[1,1,1],[1,1,3]],atol=1e-14)
assert np.allclose(W,expected_w,atol=1e-12)
assert np.allclose(A,expected_a,atol=1e-12)
assert np.allclose(K_PROBE,expected_k,atol=1e-12)
assert abs(PREDICTION-expected_prediction)<1e-12
assert abs(float(K_PROBE@A)-PREDICTION)<1e-12
assert np.linalg.norm((K+LAMBDA*np.eye(3))@A-T)<1e-12
assert np.linalg.norm(PHI.T@A-W)<1e-12
assert np.linalg.norm(PHI.T@(PHI@W-T)+LAMBDA*W)<1e-12
for x in np.linspace(-1.5,1.5,31):
    assert abs(float(phi(x)@W)-float(kernel(X,x)@A))<1e-12
for lam in [.1,.5,1,2]:
    assert np.allclose(primal(lam),PHI.T@dual(lam),atol=1e-12)
assert np.all(np.linalg.eigvalsh(K)>=-1e-12)
result=dict(x=X.tolist(),target=T.tolist(),lambda_value=LAMBDA,phi=PHI.tolist(),K=K.tolist(),w=W.tolist(),a=A.tolist(),probe=PROBE,k_probe=K_PROBE.tolist(),contributions=CONTRIBUTIONS.tolist(),prediction=PREDICTION,expected_fraction='7/19',primal_dual_max_error=max(abs(float(phi(x)@W)-float(kernel(X,x)@A)) for x in np.linspace(-1.5,1.5,31)))
Path('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result['prediction'],result['primal_dual_max_error'])
