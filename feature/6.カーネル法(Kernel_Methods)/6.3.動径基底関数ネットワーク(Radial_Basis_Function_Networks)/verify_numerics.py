"""Recompute shown RBF figures from independent direct formulas."""
import json
from pathlib import Path
import numpy as np
from rbf_model import X,T,SIGMA,PROBE,H,W,HEIGHTS,Q_PROBE,RAW_PROBE,NW_PROBE,design,bumps,raw,nw
assert np.linalg.eigvalsh(H).min() > 0
assert np.allclose(H @ W, T, atol=1e-12)
assert np.allclose(raw(X), T, atol=1e-12)
assert np.allclose(HEIGHTS, np.exp(-(PROBE-X)**2/(2*SIGMA**2)))
assert np.isclose(Q_PROBE.sum(), 1)
assert np.isclose(NW_PROBE, np.sum(HEIGHTS*T)/np.sum(HEIGHTS))
assert 0 < RAW_PROBE < .2 and .98 < NW_PROBE < 1.
# Two fixed centres: least squares is overdetermined, illustrating M<N.
centers=np.array([-1.,1.]); D=design(centers)
small_w=np.linalg.lstsq(D,T,rcond=None)[0]
small_pred=D @ small_w
result=dict(x=X.tolist(),t=T.tolist(),sigma=SIGMA,probe=PROBE,H=H.tolist(),interpolation_weights=W.tolist(),training_prediction=raw(X).tolist(),heights_at_probe=HEIGHTS.tolist(),normalized_weights_at_probe=Q_PROBE.tolist(),raw_probe=RAW_PROBE,nw_probe=NW_PROBE,small_centers=centers.tolist(),small_weights=small_w.tolist(),small_training_prediction=small_pred.tolist(),small_training_squared_error=float(np.sum((small_pred-T)**2)))
(Path(__file__).resolve().parent/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',RAW_PROBE,NW_PROBE,result['small_training_squared_error'])
