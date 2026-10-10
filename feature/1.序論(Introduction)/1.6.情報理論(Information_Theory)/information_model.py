"""Numerical source of truth for the original visual experiments."""
import math
import numpy as np

P = np.array([.5, .25, .125, .125])
Q = np.array([.1, .2, .3, .4])
P_CODES = ('0', '10', '110', '111')
Q_CODES = ('000', '001', '01', '1')
DATA = np.array([1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1])

def mean_code_length(p, codes):
    return float(np.asarray(p) @ np.array([len(code) for code in codes]))

def entropy(p, base=np.e):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p*np.log(p))/np.log(base))

def kl(p, q):
    p,q = np.asarray(p),np.asarray(q)
    good = p > 0
    if np.any(q[good] == 0):
        return float('inf')
    return float(np.sum(p[good]*np.log(p[good]/q[good])))

def gaussian_pdf(x, sigma=1):
    return np.exp(-.5*(np.asarray(x)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def gaussian_entropy(sigma):
    return .5*np.log(2*np.pi*np.e*sigma*sigma)

def spread(sigma):
    p = np.exp(-.5*((np.arange(30)-14)/sigma)**2)
    return p/p.sum()

def joint(r):
    return np.array([[1+r,1-r],[1-r,1+r]])/4

def conditional(r):
    return entropy([(1+r)/2,(1-r)/2])

def mutual(r):
    return kl(joint(r).ravel(),np.full(4,.25))

def nll(theta):
    return float(-np.mean(DATA*np.log(theta)+(1-DATA)*np.log1p(-theta)))

def multiplicity(n, k):
    return math.comb(n,k)
