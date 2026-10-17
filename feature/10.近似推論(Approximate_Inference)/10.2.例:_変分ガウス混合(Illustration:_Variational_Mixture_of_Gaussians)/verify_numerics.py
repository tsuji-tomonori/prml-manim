"""Independent invariants for the PRML 10.2 example and the same opening/ending."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import t,norm
from mixture_model import fit,posterior,responsibilities,expected_logs,K,D,ALPHA0,NU0

x,h=fit(steps=50)
assert x.shape==(80,2) and len(h)==51
assert np.allclose(h[0]['r'].sum(axis=1),1,atol=1e-14)
for s in h:
    assert np.all(s['r']>=0) and np.allclose(s['r'].sum(axis=1),1,atol=1e-13)
    assert np.isclose(s['nk'].sum(),80,atol=1e-12)
    assert np.isclose(s['weights'].sum(),1,atol=1e-12)
    assert np.all(np.linalg.eigvalsh(s['covariance'])>0)
step_changes=np.diff([s['bound'] for s in h])
assert np.min(step_changes)>-1e-8
final=h[50]
active=np.flatnonzero(final['weights']>.02)
assert active.tolist()==[1,4]
assert np.allclose(final['nk'][active],[40,40],atol=.01)
assert np.max(final['weights'][[0,2,3,5]])<1.3e-5
assert np.allclose(final['weights'],(ALPHA0+final['nk'])/(80+K*ALPHA0),atol=1e-14)
assert abs(final['bound']-h[30]['bound'])<1e-5

initial_p=posterior(x,h[0]['r'])
from_formula=responsibilities(x,initial_p)
assert np.max(np.abs(from_formula-h[1]['r']))<1e-13
for k in active:
    # Equation (10.61): prior center plus the responsibility-weighted sample center.
    weighted=np.dot(final['r'][:,k],x)/final['nk'][k]
    p=posterior(x,final['r'])
    expected=(.05*0+final['nk'][k]*weighted)/p['beta'][k]
    assert np.allclose(p['mean'][k],expected,atol=1e-12)
    assert np.isclose(p['alpha'][k],ALPHA0+final['nk'][k])
    assert np.isclose(p['nu'][k],NU0+final['nk'][k])

_,strong=fit(alpha0=10.,steps=50)
assert np.all(strong[-1]['weights']>.1)
assert np.min(np.diff([s['bound'] for s in strong]))>-1e-8

# 1-D marginal of each multivariate Student component integrates to one.
p=posterior(x,final['r'])
integrals=[]
tail_predicted=tail_fixed=0.
for k in active:
    df=p['nu'][k]+1-D
    scale_matrix=np.linalg.inv(p['w'][k])*(1+p['beta'][k])/(df*p['beta'][k])
    value=quad(lambda z:t.pdf(z,df,loc=p['mean'][k,0],scale=np.sqrt(scale_matrix[0,0])),
               -np.inf,np.inf,epsabs=1e-9)[0]
    integrals.append(value)
    weight=p['alpha'][k]/p['alpha'].sum()
    tail_predicted+=weight*t.pdf(3.5,df,loc=p['mean'][k,0],scale=np.sqrt(scale_matrix[0,0]))
    plug=np.sqrt(np.linalg.inv(p['nu'][k]*p['w'][k])[0,0])
    tail_fixed+=weight*norm.pdf(3.5,loc=p['mean'][k,0],scale=plug)
assert np.allclose(integrals,1,atol=1e-8)
assert 1.9<tail_predicted/tail_fixed<2.
result=dict(seed=1023,n=80,k=K,alpha0=ALPHA0,
            initial_bound=h[0]['bound'],final_bound=final['bound'],
            minimum_bound_increment=float(np.min(step_changes)),
            initial_weights=h[0]['weights'].tolist(),
            final_weights=final['weights'].tolist(),
            final_effective_counts=final['nk'].tolist(),
            strong_prior_weights=strong[-1]['weights'].tolist(),
            active_components=(active+1).tolist(),
            predictive_marginal_integrals=integrals,
            tail_density_x1_3_5=dict(variational=tail_predicted,plugin=tail_fixed,
                                    ratio=tail_predicted/tail_fixed))
(Path(__file__).resolve().parent/'numerical_validation.json').write_text(
    json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result)
