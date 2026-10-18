"""Recalculate narrative values and identities independently."""
import json
from pathlib import Path
import numpy as np
import nonlinear_model as m
assert m.POINTS.shape == (13, 2)
assert np.allclose(m.COV, np.cov(m.POINTS.T, bias=True))
assert np.allclose(m.COV @ m.AXIS, m.EIGENVALUES[-1] * m.AXIS)
assert np.allclose(m.LINEAR_MSE, m.EIGENVALUES[0])
assert m.LINEAR_MSE > 20*m.CURVED_MSE
assert np.allclose(m.CURVED[:,0],m.POINTS[:,0])
assert abs(np.mean(m.DEPENDENT_X*m.DEPENDENT_Y)) < 1e-14
assert not np.allclose(m.DEPENDENT_Y, m.DEPENDENT_Y[0])
assert abs(np.linalg.det(m.MIXING)) > .1
assert np.allclose(m.RESTORED,np.stack([m.SOURCE_1,m.SOURCE_2]))
report=dict(n=13,mean=m.MEAN.tolist(),eigenvalues=m.EIGENVALUES.tolist(),
            principal_axis=m.AXIS.tolist(),linear_mse=m.LINEAR_MSE,
            fixed_curved_mse=m.CURVED_MSE,ratio=m.LINEAR_MSE/m.CURVED_MSE,
            first_point=m.POINTS[0].tolist(),first_linear=m.LINEAR[0].tolist(),
            first_curved=m.CURVED[0].tolist(),
            uncorrelated_dependent_covariance=float(np.mean(m.DEPENDENT_X*m.DEPENDENT_Y)),
            mixing_determinant=float(np.linalg.det(m.MIXING)),
            recovery_max_error=float(np.max(np.abs(m.RESTORED-np.stack([m.SOURCE_1,m.SOURCE_2])))))
Path('numerical_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS',report)
