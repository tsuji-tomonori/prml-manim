"""Deterministic one-dimensional example for PRML 9.2.

Each state contains the parameters before the E step, its responsibilities,
and its observed-data log likelihood. M uses those fixed responsibilities.
"""
from __future__ import annotations
import numpy as np

POINTS = np.array([-2.2, -1.8, -1.6, -1.3, -1.0, -.8, -.55, -.25,
                   .15, .4, .75, 1.05, 1.3, 1.6, 1.9, 2.3])
PROBE_INDEX = 8
INITIAL = dict(mu=np.array([-1.1, 1.1]), sigma=np.array([.9, 1.2]),
               pi=np.array([.65, .35]))


def components(x, mu, sigma, pi):
    values = np.asarray(x, dtype=float)[..., None]
    return pi / (sigma * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((values - mu) / sigma) ** 2)


def evaluate(mu, sigma, pi):
    weighted = components(POINTS, mu, sigma, pi)
    totals = weighted.sum(axis=1)
    return weighted / totals[:, None], float(np.log(totals).sum())


def em_history(cycles=20):
    mu, sigma, pi = (INITIAL[k].copy() for k in ('mu', 'sigma', 'pi'))
    states = []
    for _ in range(cycles + 1):
        gamma, log_likelihood = evaluate(mu, sigma, pi)
        states.append(dict(mu=mu.copy(), sigma=sigma.copy(), pi=pi.copy(),
                           gamma=gamma.copy(), log_likelihood=log_likelihood))
        counts = gamma.sum(axis=0)
        new_mu = (gamma * POINTS[:, None]).sum(axis=0) / counts
        new_sigma = np.sqrt((gamma * (POINTS[:, None] - new_mu) ** 2).sum(axis=0) / counts)
        mu, sigma, pi = new_mu, new_sigma, counts / len(POINTS)
    return states


def collapsed_example(sigma):
    """A separate parameter path: the blue component stays broad; red pins one point."""
    broad = .9 / (1.55 * np.sqrt(2 * np.pi)) * np.exp(-.5 * (POINTS / 1.55) ** 2)
    narrow = .1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((POINTS - .15) / sigma) ** 2)
    return float(np.log(broad + narrow).sum()), float((broad + narrow)[PROBE_INDEX])
