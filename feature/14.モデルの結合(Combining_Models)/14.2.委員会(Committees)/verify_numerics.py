"""Independent cross-checks of the quantities shown in the film."""
import json
from pathlib import Path
import numpy as np
from committee_model import *

c=comparison()
q=predict(np.array([QUERY_X]))[0]
assert abs(float(q.mean())-float(np.mean(q)))<1e-14
assert abs(float(truth(QUERY_X))-.9485259786005206)<1e-12
assert abs(float(q.mean())-.9654085994763207)<1e-10
assert abs(c['average_mse']-c['committee_mse']-c['diversity'])<1e-12
assert c['committee_mse']<=c['average_mse']
assert abs(c['average_mse']-.08451203867225368)<1e-10
assert abs(c['committee_mse']-.02584318601996055)<1e-10
for rho in (0.,.2,.8,1.):
    errors=orthogonal_errors(rho)
    # Integral (not pointwise) of Fourier modes is measured by trapezoids.
    av=np.trapezoid(np.mean(errors**2,axis=1),GRID)/2
    com=np.trapezoid(np.mean(errors,axis=1)**2,GRID)/2
    assert abs(av-1.)<1e-12
    assert abs(com-orthogonal_expectation(rho))<1e-12
result=dict(query_x=QUERY_X,truth=float(truth(QUERY_X)),predictions=q.tolist(),
            committee_prediction=float(q.mean()),**c,
            ideal_ratio=orthogonal_expectation(0.),correlated_ratio=orthogonal_expectation(.8),
            bootstrap_indices=BOOTSTRAP_INDICES.tolist(),train_x=TRAIN_X.tolist(),train_y=TRAIN_Y.tolist())
(Path(__file__).resolve().parent/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:v for k,v in result.items() if k not in ('bootstrap_indices','train_x','train_y','predictions','individual_mse')})
