"""Reproducible numerical experiments for PRML 1.2 (no drawing dependencies)."""
import numpy as np


def gaussian(x, mu=0., sigma=1.):
    return np.exp(-.5*((np.asarray(x)-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))


def box_joint(prior=.3, orange_red=.75, orange_blue=.2):
    """Rows: red/blue boxes; columns: orange/apple."""
    return np.array([prior, 1-prior])[:,None]*np.array([[orange_red,1-orange_red],[orange_blue,1-orange_blue]])


def box_posterior(prior=.3, orange_red=.75, orange_blue=.2):
    j=box_joint(prior,orange_red,orange_blue)
    return j[:,0]/j[:,0].sum()


OBS=np.array([-1.15,-.55,-.2,.15,.55,.9,1.15])
PAIRS=np.random.default_rng(124).normal(size=(4000,2))
PAIR_VARIANCES=PAIRS.var(axis=1)
VAR_RUNNING=np.cumsum(PAIR_VARIANCES)/np.arange(1,len(PAIRS)+1)
X=np.array([.08,.17,.28,.42,.56,.65,.78,.92])
T=.85*np.sin(2*np.pi*X)+np.random.default_rng(1206).normal(0,.15,len(X))
BETA=1/.15**2
ALPHA=.015
DEGREE=3


def phi(x):
    return np.asarray(x)[...,None]**np.arange(DEGREE+1)


def posterior(alpha=ALPHA):
    design=phi(X)
    precision=alpha*np.eye(DEGREE+1)+BETA*design.T@design
    cov=np.linalg.solve(precision,np.eye(DEGREE+1))
    mean=np.linalg.solve(precision,BETA*design.T@T)
    return mean,cov


def predictive(x,alpha=ALPHA):
    mean,cov=posterior(alpha)
    f=phi(x)
    return f@mean,1/BETA+np.einsum('...i,ij,...j->...',f,cov,f)


W,S=posterior()
SAMPLE_W=np.random.default_rng(731).multivariate_normal(W,S,8)


def coin_prior(theta):
    return 6*np.asarray(theta)*(1-np.asarray(theta))


def coin_posterior(theta):
    return 30*np.asarray(theta)**4*(1-np.asarray(theta))
