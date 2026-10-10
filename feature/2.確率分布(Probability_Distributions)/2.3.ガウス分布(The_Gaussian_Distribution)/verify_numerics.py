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
# The opening and ending reuse these 100 observations and the same two fits.
middle_count=int(np.count_nonzero((MIX_DATA>=-.55)&(MIX_DATA<.55)))
single_mid=float(normal(0,MIX_DATA.mean(),MIX_DATA.std()))
two_mid=float(mixture(0))
single_ll=float(np.log(normal(MIX_DATA,MIX_DATA.mean(),MIX_DATA.std())).sum())
two_ll=float(np.log(mixture(MIX_DATA)).sum())
assert len(MIX_DATA)==100 and middle_count==2
assert np.isclose(single_mid,.22372912131409675)
assert np.isclose(two_mid,.028870790587310416)
assert two_ll>single_ll
assert np.isclose(MIX_WEIGHT,.4493772872323348,atol=1e-8)
checks['opening_return']={'observations':len(MIX_DATA),'middle_count':middle_count,
    'single_mid_density':single_mid,'two_mid_density':two_mid,
    'single_log_likelihood':single_ll,'two_log_likelihood':two_ll,
    'fitted_weight':MIX_WEIGHT,'fitted_means':MIX_MEANS.tolist(),
    'fitted_scales':MIX_SCALES.tolist()}
# V10: use polygon area and Gaussian quadrature independently of the drawing.
b=np.diag([2.,1.]);cov=b@b.T
square=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])@b.T
area=.5*abs(np.dot(square[:,0],np.roll(square[:,1],-1))-np.dot(square[:,1],np.roll(square[:,0],-1)))
assert np.isclose(area,2) and np.isclose(np.sqrt(np.linalg.det(cov)),area)
assert np.isclose(normal(0,0,2)/normal(0),.5)
assert abs(quad(lambda x: normal(x,0,2),-np.inf,np.inf)[0]-1)<1e-9
product_area=quad(lambda u:6*u*(1-u)*u**3,0,1)[0]
assert np.isclose(product_area,.2)
assert np.isclose(quad(lambda u:30*u**4*(1-u),0,1)[0],1)
checks['visual_aid']={'area_multiplier':area,'covariance_determinant':float(np.linalg.det(cov)),
                      'density_ratio':.5,'bayes_product_area':product_area,'bayes_normalized_area':1.}
print(json.dumps(checks,indent=2,ensure_ascii=False))
print('PASS: normalization, covariance geometry, conditional/linear Bayes, CLT, MLE, bias, sequential update, posterior, t mixture, robustness, circular mean, mixture responsibilities, opening/ending fit')
