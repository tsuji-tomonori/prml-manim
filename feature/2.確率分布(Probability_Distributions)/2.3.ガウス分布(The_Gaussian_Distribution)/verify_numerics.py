"""Independent identities and quadrature checks for displayed experiments."""
import json
import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma
from gaussian_model import *

checks={}
assert abs(quad(lambda x: normal(x,.7,1.3),-np.inf,np.inf)[0]-1)<1e-9
checks['normal_integral']=1.
for c in [COV,covariance(),np.diag([2.,.6]),np.eye(2)]:
    pts=ellipse(c)
    delta=np.einsum('ni,ij,nj->n',pts,np.linalg.inv(c),pts)
    assert np.max(abs(delta-1))<1e-12
    for b in [-1.,0.,1.]:
        m,v=conditional(c,b)
        precision=np.linalg.inv(c)
        assert np.isclose(v,1/precision[0,0])
        assert np.isclose(m,-precision[0,1]*b/precision[0,0])
checks['conditional_b1']=conditional(COV,1.)
for n,data in CLT_MEANS.items():
    assert abs(data.mean()-.5)<.01
    assert abs(data.var()*12*n-1)<.04
checks['clt_means_variances']={n:[float(d.mean()),float(d.var())] for n,d in CLT_MEANS.items()}
assert abs(BIAS.mean()-.75)<.025
checks['variance_bias']=[float(BIAS.mean()),float(BIAS.mean()*4/3)]
mean=0.
for n,x in enumerate(DATA,1):
    mean+=(x-mean)/n
    assert np.isclose(mean,DATA[:n].mean())
checks['mle']=[float(DATA.mean()),float(DATA.var())]
for n in [0,1,2,10]:
    m,v=posterior(n)
    assert np.isclose(v,1/(1+n/.36))
    if n: assert np.isclose(m,np.linalg.solve(np.array([[1+n/.36]]),[BAYES_DATA[:n].sum()/.36])[0])
checks['posterior_n10']=posterior(10)
for a,noise,y in [(1.4,.9,1.3),(1.4,.35,1.3)]:
    joint=np.array([[1,a],[a,a*a+noise*noise]])
    assert np.allclose(conditional(joint,y),linear_posterior(a,noise,y))
for x in [-3.,0.,2.]:
    integral=quad(lambda t: normal(x,0,1/np.sqrt(t))*gamma.pdf(t,1.5,scale=1/1.5),0,np.inf)[0]
    assert np.isclose(integral,student(x),rtol=1e-7)
checks['robust_fit_start_end']=[ROBUST_FITS[0].tolist(),ROBUST_FITS[-1].tolist()]
assert abs(ROBUST_FITS[-1,2]-ROBUST_FITS[0,2]) < abs(ROBUST_FITS[-1,0]-ROBUST_FITS[0,0])
for m in [0,1,5]:
    assert abs(quad(lambda t: von_mises(t,.3,m),0,2*np.pi)[0]-1)<1e-10
    assert np.isclose(von_mises(0,.3,m),von_mises(2*np.pi,.3,m))
angles=np.deg2rad([5,355]); angle=np.arctan2(np.sin(angles).mean(),np.cos(angles).mean())
assert abs(angle)<1e-12
checks['circular_mean_degrees']=float(np.rad2deg(angle))
for w in [.2,.45,.8]:
    assert abs(quad(lambda x: mixture(x,w),-np.inf,np.inf)[0]-1)<1e-10
    r=responsibility(np.linspace(-4,4,51),w)
    assert np.all((r>=0)&(r<=1))
checks['responsibility_x0']=float(responsibility(0))
print(json.dumps(checks,indent=2,ensure_ascii=False))
print('PASS: normalization, covariance geometry, conditional/linear Bayes, CLT, MLE, bias, sequential update, posterior, t mixture, robustness, circular mean, mixture responsibilities')
