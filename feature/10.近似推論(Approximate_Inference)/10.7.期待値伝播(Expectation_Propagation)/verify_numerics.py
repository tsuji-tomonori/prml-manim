"""Independent checks of the small clutter example and EP update identities."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from ep_model import OBS, ORDER, REVERSE, run, exact_moments, true_factor, gaussian

root=Path(__file__).resolve().parent
exact=exact_moments()
forward=run(ORDER,1)
reverse=run(REVERSE,1)
ep=run(ORDER,8)
ep_reverse=run(REVERSE,8)
assert abs(exact['mean']-4.865049007) < 1e-6
assert abs(forward['mean']-5.131327944) < 1e-6
assert abs(reverse['mean']-4.326951382) < 1e-6
assert abs(ep['mean']-4.879328716) < 1e-6
assert abs(forward['mean']-reverse['mean']) > .8
assert abs(ep['mean']-exact['mean']) < .02
assert abs(ep['variance']-exact['variance']) > .1  # Do not claim exact variance.
assert ep['variance'] > 0
assert abs(ep_reverse['mean']-ep['mean']) < 1e-8
assert any(v<0 for v in forward['site_precisions'])
# Direct integration of two tilted densities checks the analytic mixture update.
for row in (forward['history'][0],ep['history'][-1]):
    cm,cv,x=row['cavity_mean'],row['cavity_variance'],row['observation']
    unnorm=lambda t: gaussian(t,cm,cv)*true_factor(t,x)
    z=quad(unnorm,-math.inf,math.inf,epsabs=1e-12)[0]
    m=quad(lambda t:t*unnorm(t),-math.inf,math.inf,epsabs=1e-12)[0]/z
    v=quad(lambda t:(t-m)**2*unnorm(t),-math.inf,math.inf,epsabs=1e-12)[0]/z
    assert abs(z-row['tilted_normalizer']) < 1e-10
    assert abs(m-row['tilted_mean']) < 1e-9
    assert abs(v-row['tilted_variance']) < 1e-9
    assert abs(1/v-1/cv-row['site_precision']) < 1e-9
sweep_means=[ep['history'][5*i+4]['posterior_mean'] for i in range(8)]
assert abs(sweep_means[-1]-sweep_means[-2]) < 1e-8
result=dict(observations=OBS.tolist(),model=dict(a=10,b=100,w=.5),
            exact=exact,adf_forward=dict(mean=forward['mean'],variance=forward['variance']),
            adf_reverse=dict(mean=reverse['mean'],variance=reverse['variance']),
            ep=dict(mean=ep['mean'],variance=ep['variance'],sweep_means=sweep_means,
                    site_precisions=ep['site_precisions'],reverse_order_mean=ep_reverse['mean']),
            quadrature='scipy.integrate.quad over real line, epsabs=1e-12, epsrel=1e-11')
(root/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
