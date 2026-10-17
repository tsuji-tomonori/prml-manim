"""Recompute displayed margins and the box-constrained SVM solutions."""
import json
import numpy as np
from pathlib import Path
from margin_model import X,T,EXTRA,hard_margin,soft_svm,signed_distance,warp_inputs,nonlinear_features,nonlinear_kernel
assert np.array_equal(np.sign(X[:,0]-.4),T)
# The reversed-label point is strictly inside the positive-class triangle.
assert np.allclose(np.array([.6,.2,.2])@X[3:],EXTRA)
assert np.array_equal(np.sign(X[:,0]),T)
assert abs(hard_margin(.4)-.6)<1e-12
assert abs(hard_margin(0.)-1.)<1e-12
assert np.array_equal(signed_distance(X,T,0.),[1,2,2,1,2,2])
a=np.array([.5,0,0,.5,0,0]);w=(a*T)@X
assert np.allclose(w,[1,0]) and abs(a@T)<1e-12
assert np.allclose(a*(signed_distance(X,T,0.)-1),0)
soft={}
for c in [.2,.5]:
    s=soft_svm(c)
    assert abs(s['primal']-s['dual'])<1e-7
    assert abs(s['alpha']@s['t'])<1e-7
    assert np.all(s['alpha']>=-1e-8) and np.all(s['alpha']<=c+1e-8)
    assert np.all(s['t']*(s['x']@s['w']+s['b'])>=1-s['slack']-1e-8)
    assert s['slack'][-1]>1
    soft[str(c)]={k:(v.tolist() if isinstance(v,np.ndarray) else float(v)) for k,v in s.items() if k not in ('x','t')}
result=dict(base_points=X.tolist(),labels=T.tolist(),candidate_margin=hard_margin(.4),optimum_margin=hard_margin(0.),support_coefficients=a.tolist(),w=w.tolist(),b=0,extra_point=EXTRA.tolist(),soft=soft)
warped=warp_inputs(X)
assert np.allclose(nonlinear_features(warped),X)
k=nonlinear_kernel(warped,warped)
assert np.allclose(k,X@X.T)
assert np.min(np.linalg.eigvalsh(k))>-1e-12
assert np.allclose((a*T)@k, X[:,0])
assert abs(.5*(w@w)-(a.sum()-.5*(a*T)@k@(a*T)))<1e-12
# The two opposite-class points are 2 units apart, so margin <= 1 for any line.
assert np.isclose(np.linalg.norm(X[3]-X[0])/2,1)
result['nonlinear']=dict(warped_points=warped.tolist(),feature_reconstruction_error=float(np.max(abs(nonlinear_features(warped)-X))),gram_error=float(np.max(abs(k-X@X.T))),minimum_kernel_eigenvalue=float(np.min(np.linalg.eigvalsh(k))))
Path(__file__).with_name('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS hard and soft margins')
