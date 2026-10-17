"""Fixed 16-point Gaussian mixture and exact finite-latent EM bound."""
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


def posterior_and_log_likelihood(mu, sigma, pi):
    weighted = components(POINTS, mu, sigma, pi)
    totals = weighted.sum(axis=1)
    return weighted / totals[:, None], float(np.log(totals).sum())


def expected_complete_log(q, mu, sigma, pi):
    log_components = (np.log(pi)[None, :] - np.log(sigma)[None, :]
                      - .5 * np.log(2 * np.pi)
                      - .5 * ((POINTS[:, None] - mu[None, :]) / sigma[None, :]) ** 2)
    return float(np.sum(q * log_components))


def entropy(q):
    return float(-np.sum(q * np.log(q)))


def decomposition(q, mu, sigma, pi):
    posterior, observed = posterior_and_log_likelihood(mu, sigma, pi)
    q_value = expected_complete_log(q, mu, sigma, pi)
    lower = q_value + entropy(q)
    kl = float(np.sum(q * np.log(q / posterior)))
    return dict(q=q_value, entropy=entropy(q), lower=lower, kl=kl,
                observed=observed, posterior=posterior)


def em_history(cycles=3):
    mu, sigma, pi = (INITIAL[k].copy() for k in ('mu', 'sigma', 'pi'))
    states = []
    for _ in range(cycles + 1):
        q, observed = posterior_and_log_likelihood(mu, sigma, pi)
        states.append(dict(mu=mu.copy(), sigma=sigma.copy(), pi=pi.copy(),
                           gamma=q.copy(), log_likelihood=observed))
        counts = q.sum(axis=0)
        mu = (q * POINTS[:, None]).sum(axis=0) / counts
        sigma = np.sqrt((q * (POINTS[:, None] - mu) ** 2).sum(axis=0) / counts)
        pi = counts / len(POINTS)
    return states


def pi_slice(pi_red, q=None):
    """Only the red mixing coefficient changes; both component shapes stay initial."""
    pi = np.array([pi_red, 1 - pi_red])
    if q is None:
        q = posterior_and_log_likelihood(**INITIAL)[0]
    return decomposition(q, INITIAL['mu'], INITIAL['sigma'], pi)
