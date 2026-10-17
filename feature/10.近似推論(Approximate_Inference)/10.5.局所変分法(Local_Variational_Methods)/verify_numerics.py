"""Independent scalar quadrature and grid checks of the local bound."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from local_bound_model import (MEAN,SD,bound,conjugate_exp,density,exact_expectation,
                               integrated_bound,integrated_bound_quad,lam,optimum,tangent_exp)
root=Path(__file__).resolve().parent
xi, best=optimum()
exact=exact_expectation()
assert abs(exact-quad(lambda z:expit(MEAN+SD*z)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-12,12)[0])<1e-12
assert abs(expit(MEAN)-0.7310585786300049)<1e-12
assert abs(exact-0.6477264385258688)<1e-9
assert abs(xi-2.3319123)<1e-4 and abs(best-0.599663413)<1e-8
assert best<exact<expit(MEAN)
for contact in (.5,1.,2.5,xi):
    grid=np.linspace(-12,12,2401)
    difference=expit(grid)-bound(grid,contact)
    assert difference.min()>-2e-14
    assert abs(bound(contact,contact)-expit(contact))<1e-14
    assert abs(bound(-contact,contact)-expit(-contact))<1e-14
    assert abs(integrated_bound(contact)-integrated_bound_quad(contact))<1e-11
for contact in (.5,1.,2.5):
    grid=np.linspace(-2,6,801)
    assert np.min(np.exp(-grid)-tangent_exp(grid,contact))>-1e-14
    eta=-np.exp(-contact)
    assert abs(eta*contact-conjugate_exp(eta)-np.exp(-contact))<1e-12
assert abs(lam(0)-.125)<1e-12
result=dict(mean=MEAN,sd=SD,plug_in=float(expit(MEAN)),exact=exact,xi_opt=xi,bound_opt=best,
            xi_1=integrated_bound(1),xi_2_5=integrated_bound(2.5),
            contact_2_5=[dict(x=x,true=float(expit(x)),bound=float(bound(x,2.5))) for x in [-2.5,-1,0,1,2.5]],
            grid_bound_min_gap=float(np.min(expit(np.linspace(-12,12,2401))-bound(np.linspace(-12,12,2401),xi))))
(root/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result)
