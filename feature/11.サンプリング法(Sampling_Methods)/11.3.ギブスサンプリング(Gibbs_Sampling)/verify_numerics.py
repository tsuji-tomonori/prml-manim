"""Verify the narration's fixed example and symbolic invariance checks."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from gibbs_model import *
row=checks()
for y in [-2.,0.,1.]:
    f=lambda x:joint_density(x,y)
    marginal=quad(f,-12,12,epsabs=1e-11)[0]
    expected=np.exp(-.5*y*y)/np.sqrt(2*np.pi)
    assert abs(marginal-expected)<1e-10
    conditional=quad(lambda x:f(x)/expected,-12,12,epsabs=1e-11)[0]
    assert abs(conditional-1)<1e-10
# Exactly sample a one-coordinate conditional: target = marginal * conditional.
for x,y in [(0.,0.),(.7,-.2),(-1.5,-1.4)]:
    m=np.exp(-.5*y*y)/np.sqrt(2*np.pi)
    c=np.exp(-.5*((x-RHO*y)/COND_SD)**2)/(COND_SD*np.sqrt(2*np.pi))
    assert np.isclose(joint_density(x,y),m*c,rtol=1e-11)
Path('numerical_validation.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
print('PASS',row)
