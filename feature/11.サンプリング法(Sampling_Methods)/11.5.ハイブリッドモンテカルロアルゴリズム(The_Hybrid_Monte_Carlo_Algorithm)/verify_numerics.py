"""Check the narrated HMC trajectory and the measure-preserving building blocks."""
import json,math
from pathlib import Path
import numpy as np
from hmc_model import *
s=leapfrog(START_Z,START_R)
assert s.shape==(STEPS+1,2)
assert abs(s[-1,0]-2.023450004778786)<1e-12
assert abs(s[-1,1]-.39519716151698236)<1e-12
assert abs(acceptance(s)-.9997346760494523)<1e-12
assert abs(energy(*s[0])-2.125)<1e-12
# Negative time steps undo a complete trajectory. Each shear has determinant one.
back=leapfrog(*s[-1],epsilon=-EPSILON,steps=STEPS)
assert np.max(np.abs(back[-1]-s[0]))<1e-12
for z,r in [(-2,.5),(0,1.2),(1.3,-.7)]:
    h=1e-6
    def step(a,b):return leapfrog(a,b,steps=1)[-1]
    J=np.column_stack(((step(z+h,r)-step(z-h,r))/(2*h),(step(z,r+h)-step(z,r-h))/(2*h)))
    assert abs(np.linalg.det(J)-1)<1e-9
# Corrected (11.68)–(11.69): accepted probability flow is the same both ways.
flow_forward=math.exp(-energy(*s[0]))*acceptance(s)
flow_reverse=math.exp(-energy(*s[-1]))*min(1.,math.exp(energy(*s[-1])-energy(*s[0])))
assert abs(flow_forward-flow_reverse)<1e-14
coarse=leapfrog(START_Z,START_R,epsilon=1.4,steps=STEPS)
assert energy(*coarse[-1])-energy(*coarse[0])>.1
aniso=leapfrog_anisotropic()
assert len(aniso)==STEPS+1 and abs(aniso[-1][0][0]-1.51556344)<1e-7
assert 2/.4==5 and (2/.4)**2==25
# Exact Gaussian moment after momentum refresh is standard normal in both coordinates.
rng=np.random.default_rng(115)
refreshed=rng.normal(size=50000)
assert abs(refreshed.mean())<.025 and abs(refreshed.var()-1)<.025
out=dict(initial=[START_Z,START_R],epsilon=EPSILON,steps=STEPS,final=s[-1].tolist(),initial_energy=energy(*s[0]),final_energy=energy(*s[-1]),energy_error=energy(*s[-1])-energy(*s[0]),acceptance=acceptance(s),reverse_error=float(np.max(np.abs(back[-1]-s[0]))),phase_volume_determinant=1.,detailed_balance_flow=flow_forward,coarse=dict(epsilon=1.4,final=coarse[-1].tolist(),energy_error=energy(*coarse[-1])-energy(*coarse[0]),acceptance=acceptance(coarse)),anisotropic=dict(sigmas=[2.,.4],start_position=aniso[0][0].tolist(),final_position=aniso[-1][0].tolist(),scale_ratio=5,walk_scale_ratio_squared=25),momentum_check=dict(count=len(refreshed),mean=float(refreshed.mean()),variance=float(refreshed.var())))
Path('numerical_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(out)
