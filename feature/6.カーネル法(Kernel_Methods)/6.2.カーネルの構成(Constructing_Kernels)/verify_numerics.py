"""Check the counterexample and both sound constructions against independent algebra."""
import json
from pathlib import Path
import numpy as np
from kernel_model import *
assert np.allclose(K_BAD,[[1,.75,0],[.75,1,.75],[0,.75,1]])
assert np.all((K_BAD>=0)&(K_BAD<=1))
assert np.isclose(BAD_QUADRATIC,-.25)
assert np.min(np.linalg.eigvalsh(K_BAD))<0
assert np.allclose(K_POLY,feature(X)@feature(X).T)
assert np.all(np.linalg.eigvalsh(K_POLY)>0)
assert np.all(np.linalg.eigvalsh(K_GAUSS)>0)
assert np.isclose(Y_POLY,7/19) and np.isclose(Y_OLD,7/19)
assert np.isclose(Y_GAUSS,.43104018971796787)
width_predictions={str(sigma):prediction(lambda x,z:gaussian(x,z,sigma)) for sigma in [.7,1.,1.2]}
assert np.isclose(width_predictions['0.7'],.37444412043549813)
assert np.isclose(width_predictions['1.2'],.47144090221088814)
for x in np.linspace(-1.2,1.2,49):
    assert np.isclose(polynomial(X,x)@A_POLY,feature(x)@np.linalg.solve(feature(X).T@feature(X)+LAMBDA*np.eye(3),feature(X).T@T))
for sigma in [.5,.7,1.,1.2,2.]:
    k=lambda x,z:gaussian(x,z,sigma)
    for pts in [X,np.array([-.9,-.4,0,.4,.9]),np.linspace(-1,1,11)]:
        assert np.min(np.linalg.eigvalsh(gram(k,pts)))>-1e-11
result=dict(bad_gram=K_BAD.tolist(),bad_quadratic=BAD_QUADRATIC,bad_eigenvalues=np.linalg.eigvalsh(K_BAD).tolist(),polynomial_gram=K_POLY.tolist(),polynomial_coefficients=A_POLY.tolist(),polynomial_probe=polynomial(X,PROBE).tolist(),polynomial_prediction=Y_POLY,old_prediction=Y_OLD,gaussian_gram=K_GAUSS.tolist(),gaussian_coefficients=A_GAUSS.tolist(),gaussian_probe=gaussian(X,PROBE).tolist(),gaussian_prediction=Y_GAUSS,gaussian_width_predictions=width_predictions,conditions=dict(x=X.tolist(),targets=T.tolist(),probe=PROBE,lambda_=LAMBDA,sigma=SIGMA))
Path('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',BAD_QUADRATIC,Y_POLY,Y_GAUSS)
