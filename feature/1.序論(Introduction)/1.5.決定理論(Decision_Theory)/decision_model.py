"""Numerical experiments, independent of Manim. All distributions are synthetic."""
import math
import numpy as np


def gaussian(x, mean=0., sigma=1.):
    return np.exp(-.5*((np.asarray(x)-mean)/sigma)**2)/(sigma*np.sqrt(2*np.pi))


def joint(x, k, prior=.5):
    return (prior if k == 1 else 1-prior)*gaussian(x, -1 if k == 1 else 1)


def posterior(x, prior=.5):
    a, b = joint(x, 1, prior), joint(x, 2, prior)
    return a/(a+b)


def cdf(x, mean=0., sigma=1.):
    return .5*(1+math.erf((x-mean)/(sigma*np.sqrt(2))))


def mistake_parts(boundary):
    return .5*cdf(boundary, 1), .5*(1-cdf(boundary, -1))


def optimal_boundary(cost=1., prior=.5):
    return .5*np.log(cost*prior/(1-prior))


def risks(probability, cost):
    return np.array([1-probability, cost*probability])


def reject_bounds(theta):
    if theta < .5:
        return 0., 0.
    if theta >= 1:
        return -np.inf, np.inf
    b = .5*np.log(theta/(1-theta))
    return -b, b


def reject_fraction(theta):
    a, b = reject_bounds(theta)
    return .5*(cdf(b,-1)-cdf(a,-1)+cdf(b,1)-cdf(a,1))


def corrected_posterior(old, train_prior, new_prior):
    weights = np.array([old/train_prior*new_prior,
                        (1-old)/(1-train_prior)*(1-new_prior)])
    return weights/weights.sum()


def combine_posteriors(a, b, prior):
    weights = np.array([a*b/prior, (1-a)*(1-b)/(1-prior)])
    return weights/weights.sum()


def mixture(t):
    return .65*gaussian(t,-1,.45)+.35*gaussian(t,1.6,.6)


T = np.linspace(-4, 4.5, 8501)
DENSITY = mixture(T)
MEAN = .65*(-1)+.35*1.6
VARIANCE = .65*(.45**2+(-1-MEAN)**2)+.35*(.6**2+(1.6-MEAN)**2)
CDF = np.r_[0,np.cumsum((DENSITY[1:]+DENSITY[:-1])*.5*np.diff(T))]
MEDIAN = float(np.interp(.5,CDF,T))
MODE = float(T[np.argmax(DENSITY)])


def regression_risk(y, q=2):
    return float(np.trapezoid(np.abs(y-T)**q*DENSITY,T))
