"""Numerical source for the original PRML 2.1 experiments."""
import math
import numpy as np

OBS = np.array([1,0,1,1,0,1,0,1],dtype=int)
PRIOR = (3.,2.)

def bernoulli(mu):
    return np.array([1-mu,mu])

def likelihood(mu, m=5, n=8):
    mu=np.asarray(mu)
    return mu**m*(1-mu)**(n-m)

def binomial(n,mu):
    return np.array([math.comb(n,m)*mu**m*(1-mu)**(n-m) for m in range(n+1)])

def beta_pdf(x,a,b):
    x=np.asarray(x,dtype=float)
    norm=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b))
    return norm*x**(a-1)*(1-x)**(b-1)

def beta_mean(a,b): return a/(a+b)
def beta_var(a,b): return a*b/((a+b)**2*(a+b+1))
def posterior(obs=OBS,prior=PRIOR):
    return np.array(prior)+[np.sum(obs),len(obs)-np.sum(obs)]

def variance_decomposition(a,b):
    weights=bernoulli(beta_mean(a,b))
    means=np.array([beta_mean(a,b+1),beta_mean(a+1,b)])
    variances=np.array([beta_var(a,b+1),beta_var(a+1,b)])
    return dict(prior=beta_var(a,b),mean=float(weights@means),
        remaining=float(weights@variances),moved=float(weights@((means-beta_mean(a,b))**2)),
        weights=weights.tolist(),means=means.tolist(),variances=variances.tolist())
