"""Numerical experiments for PRML 2.2; all examples are original."""
import itertools
import math
import numpy as np
from scipy.special import gammaln, xlogy

SEQUENCE = np.array([0,1,0,2,0,1,0,0,1,0])
COUNTS = np.bincount(SEQUENCE, minlength=3)
PERMUTATIONS = sorted(set(itertools.permutations([0,0,1,2])))


def likelihood(mu, counts=COUNTS):
    return np.exp(np.sum(xlogy(counts, mu), axis=-1))


def coefficient(counts):
    return math.factorial(sum(counts)) // math.prod(math.factorial(int(c)) for c in counts)


def multinomial(mu, counts):
    return coefficient(counts)*likelihood(mu, counts)


def mle_slice(t):
    t=np.asarray(t)
    return np.stack([t, .75*(1-t), .25*(1-t)], axis=-1)


def relative_likelihood(t):
    return likelihood(mle_slice(t))/likelihood(COUNTS/COUNTS.sum())


def dirichlet_density(mu, alpha):
    mu, alpha=np.asarray(mu), np.asarray(alpha)
    return np.exp(gammaln(alpha.sum())-gammaln(alpha).sum()+np.sum(xlogy(alpha-1, mu),axis=-1))


def predictive(alpha, counts):
    a=np.asarray(alpha)+np.asarray(counts)
    return a/a.sum()


def density_raster(alpha, width=256, height=222):
    """Fixed logarithmic color scale of density w.r.t. dmu1 dmu2, capped at 20.

    Raster coordinates are an affine barycentric map. The Jacobian is constant;
    values show PRML's density, not probability per screen pixel.
    """
    y,x=np.mgrid[0:height,0:width]
    w3=1-(y+.5)/height
    w2=(x+.5)/width-w3/2
    w1=1-w2-w3
    mu=np.stack([w1,w2,w3],axis=-1)
    inside=np.all(mu>0,axis=-1)
    d=dirichlet_density(np.maximum(mu,1e-12), alpha)
    t=np.clip(np.log1p(d)/np.log(21),0,1)
    low=np.array([24,35,74]); high=np.array([253,231,136])
    rgb=low[None,None,:]*(1-t[:,:,None])+high[None,None,:]*t[:,:,None]
    return np.dstack([rgb.astype(np.uint8),inside.astype(np.uint8)*255])
