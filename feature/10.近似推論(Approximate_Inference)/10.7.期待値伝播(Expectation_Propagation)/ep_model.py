"""Deterministic one-dimensional EP experiment for PRML 10.7.

The five observations are authored for this film. A site stores Gaussian natural
parameters, not a normalized density; its precision may be negative.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

OBS = np.array([4.2, 4.8, 5.1, 5.4, -3.0])
A, B, W = 10.0, 100.0, 0.5
ORDER = tuple(range(len(OBS)))
REVERSE = tuple(reversed(ORDER))


def true_factor(theta: float | np.ndarray, x: float) -> float | np.ndarray:
    return (1-W)*norm.pdf(x, theta, 1.0) + W*norm.pdf(x, 0.0, math.sqrt(A))


def exact_unnormalized(theta: float | np.ndarray) -> float | np.ndarray:
    value = norm.pdf(theta, 0.0, math.sqrt(B))
    for x in OBS:
        value = value*true_factor(theta, float(x))
    return value


def exact_moments() -> dict:
    # Independent quadrature of the actual five-factor density, rather than
    # reusing the moment-matching algebra that drives EP.
    integrals = [quad(lambda t, k=k: (t**k)*exact_unnormalized(t),
                      -math.inf, math.inf, epsabs=1e-12, epsrel=1e-11)[0]
                 for k in range(3)]
    z = integrals[0]
    mean = integrals[1]/z
    return dict(evidence=z, mean=mean, variance=integrals[2]/z-mean**2)


def update(precision: np.ndarray, natural_mean: np.ndarray, index: int) -> dict:
    tau = 1/B + float(np.sum(precision)) - precision[index]
    eta = float(np.sum(natural_mean)) - natural_mean[index]
    if tau <= 0:
        raise ArithmeticError('cavity must be normalizable')
    cavity_var = 1/tau
    cavity_mean = eta/tau
    x = float(OBS[index])
    clean_weight = (1-W)*norm.pdf(x, cavity_mean, math.sqrt(cavity_var+1))
    clutter_weight = W*norm.pdf(x, 0.0, math.sqrt(A))
    z = clean_weight+clutter_weight
    rho = clean_weight/z
    clean_var = cavity_var/(cavity_var+1)
    clean_mean = cavity_mean + clean_var*(x-cavity_mean)
    new_mean = (1-rho)*cavity_mean + rho*clean_mean
    new_var = ((1-rho)*(cavity_var+(cavity_mean-new_mean)**2)
               + rho*(clean_var+(clean_mean-new_mean)**2))
    precision[index] = 1/new_var - 1/cavity_var
    natural_mean[index] = new_mean/new_var - cavity_mean/cavity_var
    return dict(index=index, observation=x, cavity_mean=cavity_mean,
                cavity_variance=cavity_var, clean_probability=rho,
                tilted_normalizer=z, tilted_mean=new_mean,
                tilted_variance=new_var, posterior_mean=new_mean,
                posterior_variance=new_var,
                site_precision=float(precision[index]),
                site_natural_mean=float(natural_mean[index]))


def run(order=ORDER, passes=1) -> dict:
    precision = np.zeros(len(OBS))
    natural_mean = np.zeros(len(OBS))
    history = []
    for sweep in range(passes):
        for index in order:
            row = update(precision, natural_mean, index)
            row['sweep'] = sweep+1
            history.append(row)
    total_precision = 1/B + np.sum(precision)
    return dict(mean=float(np.sum(natural_mean)/total_precision),
                variance=float(1/total_precision), history=history,
                site_precisions=precision.tolist(),
                site_natural_means=natural_mean.tolist())


def gaussian(theta: float | np.ndarray, mean: float, variance: float):
    return norm.pdf(theta, mean, math.sqrt(variance))


if __name__ == '__main__':
    for label, result in [('exact', exact_moments()), ('ADF forward', run()),
                          ('ADF reverse', run(REVERSE)), ('EP', run(passes=6))]:
        print(label, result['mean'], result['variance'])
