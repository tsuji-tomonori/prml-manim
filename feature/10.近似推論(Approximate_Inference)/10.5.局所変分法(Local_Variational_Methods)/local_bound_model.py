"""Numerical example for PRML 10.5, using the corrected positive JJ coefficient."""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit

MEAN = 1.0
SD = 2.0

def sigmoid(x):
    return expit(x)

def lam(xi):
    xi = float(xi)
    return np.tanh(xi / 2) / (4 * xi) if abs(xi) > 1e-8 else 1 / 8

def bound(x, xi):
    x = np.asarray(x)
    coefficient = lam(xi)
    return np.exp(log_expit(xi) + (x - xi) / 2 - coefficient * (x*x - xi*xi))

def upper_sigmoid(x, xi):
    """Exponential of a tangent to concave log sigmoid."""
    return np.exp(log_expit(xi) + sigmoid(-xi) * (np.asarray(x)-xi))

def density(x):
    return np.exp(-0.5 * ((x-MEAN)/SD)**2) / (SD*np.sqrt(2*np.pi))

def exact_expectation():
    return quad(lambda x: sigmoid(x) * density(x), -20, 20, epsabs=1e-13)[0]

def integrated_bound(xi):
    """Analytic E[bound(A,xi)] for A ~ N(MEAN, SD**2)."""
    c = lam(xi)
    q = 1 + 2*c*SD**2
    log_factor = -0.5*np.log(q) + (-c*MEAN**2 + MEAN/2)
    log_factor += SD**2 * (0.5 - 2*c*MEAN)**2 / (2*q)
    return float(np.exp(log_expit(xi) - xi/2 + c*xi**2 + log_factor))

def integrated_bound_quad(xi):
    return quad(lambda x: bound(x, xi)*density(x), -20, 20, epsabs=1e-13)[0]

def optimum():
    result = minimize_scalar(lambda xi: -integrated_bound(xi), bounds=(1e-6, 8), method='bounded', options={'xatol':1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x), float(-result.fun)

def tangent_exp(x, xi):
    return np.exp(-xi) * (1 + xi - np.asarray(x))

def conjugate_exp(eta):
    if eta >= 0:
        raise ValueError('eta must be negative')
    return eta - eta * np.log(-eta)
