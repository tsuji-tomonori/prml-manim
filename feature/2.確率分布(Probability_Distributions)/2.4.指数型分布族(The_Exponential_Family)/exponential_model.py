"""Numerical values used by the film; all observations are original examples."""
import math
import numpy as np

COINS = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
POINTS = np.array([-1.2, -.4, .1, .6, 1.0, 1.7])
COLORS = ['#FF7986', '#58B5ED', '#77D49A']

def sigmoid(eta):
    return 1/(1+np.exp(-np.asarray(eta)))

def softmax(eta):
    e=np.exp(np.asarray(eta)-np.max(eta))
    return e/e.sum()

def normal(x, mu=0., sigma=1.):
    return np.exp(-.5*((np.asarray(x)-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)

def gaussian_natural(mu, sigma):
    return np.array([mu/sigma**2, -1/(2*sigma**2)])

def gaussian_g(eta):
    a,b=eta
    return np.sqrt(-2*b)*np.exp(a*a/(4*b))

def beta_pdf(x, a, b):
    x=np.asarray(x)
    log_norm=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
    return np.exp(log_norm)*x**(a-1)*(1-x)**(b-1)

def log_likelihood(eta, successes=7, n=10):
    return successes*np.asarray(eta)-n*np.logaddexp(0, eta)

def transformed_density(eta, power):
    return power*np.asarray(eta)**(power-1)
