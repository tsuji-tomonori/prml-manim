"""Fixed educational mixture and reproducible experiments for PRML 11.1."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

THRESHOLD=1.0

def target(z):
    z=np.asarray(z)
    return .82*norm.pdf(z,-1.0,.45)+.18*norm.pdf(z,2.0,.35)

def proposal(z):
    return norm.pdf(z,-1.0,1.4)

def truth():
    return .82*norm.sf((THRESHOLD+1)/.45)+.18*norm.sf((THRESHOLD-2)/.35)

def envelope():
    # q has broader Gaussian tails than either component: p/q -> 0 at both ends.
    vals=[]
    for a,b in [(-8,-.2),(-.2,6)]:
        result=minimize_scalar(lambda x:-target(x)/proposal(x),bounds=(a,b),method='bounded',options={'xatol':1e-13})
        vals.append((-result.fun,result.x))
    peak,z=max(vals)
    return 1.002*peak,z,peak

def introductory_draws():
    return np.random.default_rng(0).normal(-1.,1.4,24)

def importance_draws():
    z=np.random.default_rng(11).normal(-1.,1.4,200)
    raw=target(z)/proposal(z)
    w=raw/raw.sum()
    return z,w

def rejection_draws(n=2000):
    rng=np.random.default_rng(1101)
    k,_,_=envelope()
    z=rng.normal(-1.,1.4,n)
    u=rng.random(n)
    accepted=u<target(z)/(k*proposal(z))
    return z,u,accepted

def resampled_draws():
    z,w=importance_draws()
    return np.random.default_rng(17).choice(z,size=200,replace=True,p=w)
