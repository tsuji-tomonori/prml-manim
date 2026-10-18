"""Independent identities and values used in the story."""
import json
from pathlib import Path
import numpy as np
import kernel_pca_model as m
r=Path(__file__).resolve().parent
assert m.POINTS.shape==(16,2) and set(m.LABELS)=={-1,1}
assert np.allclose(m.MEAN,0)
assert np.allclose(m.LINEAR_EIGENVALUES,[1.4464,1.0025])
assert set(m.POINTS[m.LABELS==1,0])==set(m.POINTS[m.LABELS==-1,0])
assert np.allclose(m.FEATURES@m.FEATURES.T,m.GRAM)
assert np.allclose(m.CENTERED_FEATURES@m.CENTERED_FEATURES.T,m.CENTERED_GRAM)
assert np.allclose(m.CENTERED_GRAM.sum(axis=0),0,atol=1e-12)
assert np.allclose(m.FEATURE_EIGENVALUES,[2.900032,.036864,.01])
assert np.allclose(m.GRAM_EIGENVALUES[:3],m.N*m.FEATURE_EIGENVALUES)
assert np.allclose(m.CENTERED_GRAM@m.ALPHA,m.N*m.FEATURE_EIGENVALUES[0]*m.ALPHA)
assert np.isclose(m.ALPHA@m.CENTERED_GRAM@m.ALPHA,1)
assert np.allclose(m.CENTERED_GRAM@m.ALPHA,m.SCORES)
assert np.allclose(m.SCORES,np.sqrt(2)*m.POINTS[:,0]*m.POINTS[:,1])
assert m.SCORES[m.LABELS==1].min()>0 and m.SCORES[m.LABELS==-1].max()<0
for x in np.r_[m.POINTS,np.array([[.9,.8],[-1.1,.7],[0,0]])]:
    assert np.isclose(m.project(x),(m.phi(x)-m.FEATURE_MEAN)@m.FEATURE_EIGENVECTORS[:,0])
assert np.isclose(m.NEW_SCORE,1.0182337649086286)
assert m.PREIMAGE_LHS>m.PREIMAGE_RHS
# Kernel-centering example cell from the actual 16-point table.
assert np.isclose(m.CENTERED_GRAM[0,0],m.GRAM[0,0]-2*m.GRAM[0].mean()+m.GRAM.mean())
report=dict(n=m.N,linear_eigenvalues=m.LINEAR_EIGENVALUES.tolist(),feature_mean=m.FEATURE_MEAN.tolist(),
            feature_eigenvalues=m.FEATURE_EIGENVALUES.tolist(),gram_eigenvalues=m.GRAM_EIGENVALUES[:3].tolist(),
            score_positive_range=[float(m.SCORES[m.LABELS==1].min()),float(m.SCORES[m.LABELS==1].max())],
            score_negative_range=[float(m.SCORES[m.LABELS==-1].min()),float(m.SCORES[m.LABELS==-1].max())],
            gram_cell_00=dict(raw=float(m.GRAM[0,0]),row_mean=float(m.GRAM[0].mean()),grand_mean=float(m.GRAM.mean()),centered=float(m.CENTERED_GRAM[0,0])),
            new_point=m.NEW_POINT.tolist(),new_score=m.NEW_SCORE,
            preimage=dict(point=m.PREIMAGE_POINT.tolist(),score_squared=m.PREIMAGE_LHS,required= m.PREIMAGE_RHS))
(r/'numerical_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS',report)
