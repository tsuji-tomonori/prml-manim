"""Reproducible numerical experiments; no digitised PRML figures."""
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats import gamma


def normal(x, mu=0., sigma=1.):
    return np.exp(-.5*((np.asarray(x)-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def covariance(s1=1.6, s2=.65, theta=.6):
    a=rotation(theta) @ np.diag([s1,s2])
    return a @ a.T


def ellipse(cov, radius=1., mu=(0.,0.)):
    vals, vecs=np.linalg.eigh(cov)
    t=np.linspace(0,2*np.pi,161)
    return np.asarray(mu)+(vecs @ (np.sqrt(vals)[:,None]*radius*np.array([np.cos(t),np.sin(t)]))).T


def conditional(cov, b, mu=(0.,0.)):
    return mu[0]+cov[0,1]/cov[1,1]*(b-mu[1]), cov[0,0]-cov[0,1]**2/cov[1,1]


COV=np.array([[1.4,.85],[.85,1.]])
BASE_POINTS=np.random.default_rng(2303).normal(size=(100,2))
CLT_U=np.random.default_rng(2302).uniform(size=(20000,32))
CLT_MEANS={n:CLT_U[:,:n].mean(axis=1) for n in [1,2,10,32]}
DATA=np.random.default_rng(2307).normal(.6,.85,16)
BIAS=np.random.default_rng(2317).normal(size=(6000,4)).var(axis=1)
BAYES_DATA=np.random.default_rng(2308).normal(.8,.6,10)


def posterior(n):
    n=int(n); v=1/(1+n/.36)
    return v*BAYES_DATA[:n].sum()/.36, v


def linear_posterior(a, noise, y):
    # x~N(0,1), y=a*x+epsilon, epsilon~N(0,noise^2)
    v=1/(1+a*a/noise**2)
    return v*a*y/noise**2,v


def student(x,mu=0.,scale=1.,nu=3.):
    z=(np.asarray(x)-mu)/scale
    c=np.exp(gammaln((nu+1)/2)-gammaln(nu/2))/np.sqrt(nu*np.pi)/scale
    return c*(1+z*z/nu)**(-(nu+1)/2)


ROBUST_DATA=np.random.default_rng(2309).normal(0,.65,24)
OUTLIER_GRID=np.linspace(0,7,71)


def robust_fit(outlier):
    d=np.r_[ROBUST_DATA,outlier]
    def loss(p):
        return -np.log(student(d,p[0],np.exp(p[1]),3)).sum()
    result=minimize(loss,[np.median(d),np.log(.65)],method='BFGS',tol=1e-7)
    # BFGS may report precision loss at numerical tolerance; check gradient independently.
    if np.linalg.norm(result.jac)>1e-4:
        raise RuntimeError(result.message)
    return np.array([d.mean(),d.std(),result.x[0],np.exp(result.x[1])])


ROBUST_FITS=np.array([robust_fit(o) for o in OUTLIER_GRID])


def fit_at(outlier):
    # Interpolated drawing path; endpoints are genuine fixed-nu MLE fits.
    return np.array([np.interp(outlier,OUTLIER_GRID,ROBUST_FITS[:,i]) for i in range(4)])


def gamma_posterior(x,n):
    n=int(n)
    a=2+n/2; b=1+np.sum((BAYES_DATA[:n]-.8)**2)/2
    return gamma.pdf(x,a,scale=1/b)


def von_mises(theta,center=0.,concentration=1.):
    return np.exp(concentration*np.cos(np.asarray(theta)-center))/(2*np.pi*np.i0(concentration))


MIX_DATA=np.r_[np.random.default_rng(2311).normal(-1.7,.55,45),np.random.default_rng(2312).normal(1.5,.7,55)]


def fit_two_gaussians(data=MIX_DATA):
    """EM fit of two 1D Gaussians to the fixed opening/closing observations."""
    weight, means, scales = .45, np.array([-1.7, 1.5]), np.array([.55, .7])
    previous = -np.inf
    for _ in range(200):
        a = weight * normal(data, means[0], scales[0])
        b = (1 - weight) * normal(data, means[1], scales[1])
        density = a + b
        likelihood = float(np.log(density).sum())
        if likelihood - previous < 1e-11:
            break
        previous = likelihood
        r = a / density
        mass = np.array([r.sum(), (1-r).sum()])
        weight = mass[0] / len(data)
        means = np.array([(r*data).sum()/mass[0], ((1-r)*data).sum()/mass[1]])
        scales = np.sqrt(np.array([
            (r*(data-means[0])**2).sum()/mass[0],
            ((1-r)*(data-means[1])**2).sum()/mass[1],
        ]))
    return float(weight), means, scales


MIX_WEIGHT, MIX_MEANS, MIX_SCALES = fit_two_gaussians()


def mixture(x,weight=None):
    if weight is None:
        weight = MIX_WEIGHT
    return weight*normal(x,MIX_MEANS[0],MIX_SCALES[0])+(1-weight)*normal(x,MIX_MEANS[1],MIX_SCALES[1])


def responsibility(x,weight=None):
    if weight is None:
        weight = MIX_WEIGHT
    return weight*normal(x,MIX_MEANS[0],MIX_SCALES[0])/mixture(x,weight)
