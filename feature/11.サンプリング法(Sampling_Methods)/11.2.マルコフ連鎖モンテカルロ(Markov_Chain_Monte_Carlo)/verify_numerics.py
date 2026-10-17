"""Independent quadrature, path replay, and detailed balance checks."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from mcmc_model import *

p=truth()
independent=quad(lambda z:float(target(z)),1,np.inf,epsabs=1e-12)[0]
whole=quad(lambda z:float(target(z)),-np.inf,np.inf,epsabs=1e-12)[0]
assert abs(p-independent)<1e-10 and abs(whole-1)<1e-10
rows=[]
for scale in SCALES:
    states,candidates,uniforms,accepted=walk(scale)
    assert len(states)==STEPS+1 and np.all(states[1:][~accepted]==states[:-1][~accepted])
    assert np.all(states[1:][accepted]==candidates[accepted])
    assert np.all((uniforms>=0)&(uniforms<1))
    assert np.all(accepted==(uniforms<np.minimum(1,target(candidates)/target(states[:-1]))))
    row=summary(scale); rows.append(row)
assert rows[0]['first_right_step']==1183 and rows[1]['first_right_step']==421 and rows[2]['first_right_step']==37
assert rows[0]['first_24_right']==0 and rows[2]['first_24_right']==0
assert rows[0]['right_estimate']>.4 and abs(rows[2]['right_estimate']-p)<.01
# q(x|y) is a shifted normal; the MH correction balances both directional flows.
for x,y in [(-1.,0.),(-1.,2.),(0.,1.4)]:
    qxy=norm.pdf(y,x+.4,1.1); qyx=norm.pdf(x,y+.4,1.1)
    axy=min(1,mh_ratio(x,y)); ayx=min(1,mh_ratio(y,x))
    assert abs(target(x)*qxy*axy-target(y)*qyx*ayx)<1e-12
result=dict(truth=p,quadrature=independent,normalization=whole,seed=SEED,start=START,steps=STEPS,burn=BURN,walks=rows,asymmetric_pair_ratio=mh_ratio(-1.,0.))
Path('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(result)
