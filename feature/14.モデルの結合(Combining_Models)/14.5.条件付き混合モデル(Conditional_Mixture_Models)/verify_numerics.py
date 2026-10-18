"""Independent numerical checks for the story's fixed example."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.integrate import trapezoid
from mixture_model import X,T,PHI,UPPER_X,LOWER_X,QUERY,fit_mixture,fit_gate,lower_gate,normal_pdf,predictive_density

states,responsibility=fit_mixture()
last=states[-1]
gate=fit_gate(responsibility)
assert len(X)==20 and len(UPPER_X)==13 and len(LOWER_X)==7
assert np.all(np.diff([s['log_likelihood'] for s in states])>=-1e-7)
assert np.allclose(responsibility.sum(axis=1),1)
assert np.allclose(last['mixing'],responsibility.mean(axis=0))
assert np.allclose(last['weights'],[[1.006923076923077,.1778021978021978],[-.9710714285714286,-.0557142857142857]],atol=1e-10)
assert abs(last['precision']-94.4048281326388)<1e-6
assert np.allclose(last['mixing'],[.65,.35],atol=1e-12)
prob=float(lower_gate(QUERY,gate))
assert abs(prob-.076855)<1e-5
# Integral of each component density equals its mixture coefficient.
t=np.linspace(-2,2,120001)
means=last['weights'][:,0]+last['weights'][:,1]*QUERY
areas=np.array([trapezoid(last['mixing'][k]*normal_pdf(t,means[k],last['precision']),t) for k in range(2)])
assert np.allclose(areas,last['mixing'],atol=1e-6)
assert abs(trapezoid(predictive_density(t,QUERY,last['weights'],last['mixing'],last['precision']),t)-1)<1e-6
# Gate has the promised direction and keeps normalized weights.
assert np.all(np.diff(lower_gate(np.linspace(-1,1,101),gate))<0)
assert abs(prob+(1-prob)-1)<1e-12
# E-step: a point next to the lower line has near-unit lower responsibility.
assert responsibility[13,1]>.999 and responsibility[0,0]>.999
result=dict(n=20,query_x=QUERY,weights=last['weights'].tolist(),mixing=last['mixing'].tolist(),precision=last['precision'],sigma=float(last['precision']**-.5),log_likelihood_initial=states[0]['log_likelihood'],log_likelihood_final=last['log_likelihood'],gate=gate.tolist(),query_lower_gate=prob,query_upper_gate=1-prob,query_component_means=means.tolist(),component_density_areas=areas.tolist(),max_responsibility_sum_error=float(np.max(abs(responsibility.sum(axis=1)-1))))
(Path(__file__).parent/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result)
