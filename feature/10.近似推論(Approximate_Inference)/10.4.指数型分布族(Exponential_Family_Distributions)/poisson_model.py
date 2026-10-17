"""Two-component Poisson mixture with fixed equal mixing weights.

Gamma distributions use shape/rate coordinates throughout.  The coordinate
updates are the specialization of the corrected PRML (10.115)--(10.121).
"""
import numpy as np
from scipy.special import digamma, gammaln, logsumexp

COUNTS = np.array([0, 1, 0, 2, 1, 0, 2, 3, 5, 6, 7, 8], dtype=float)
PROBE_INDEX = 7
A0 = np.array([1., 6.])
B0 = np.array([1., 1.])
PI = np.array([.5, .5])


def initial_responsibilities():
    low = 1 / (1 + np.exp((COUNTS - 3.2) * 2.3))
    return np.column_stack([low, 1 - low])


def update_rates(r):
    return A0 + (r * COUNTS[:, None]).sum(axis=0), B0 + r.sum(axis=0)


def update_responsibilities(a, b):
    # E[log lambda] is not log E[lambda].  -log(x!) cancels across components.
    log_scores = (np.log(PI)[None, :] + COUNTS[:, None] *
                  (digamma(a) - np.log(b))[None, :] - (a / b)[None, :])
    return np.exp(log_scores - logsumexp(log_scores, axis=1, keepdims=True))


def history(steps=18):
    r = initial_responsibilities()
    states = []
    for _ in range(steps + 1):
        a, b = update_rates(r)
        states.append(dict(r=r.copy(), a=a.copy(), b=b.copy(), mean=a / b,
                           effective_n=r.sum(axis=0),
                           effective_sum=(r * COUNTS[:, None]).sum(axis=0)))
        r = update_responsibilities(a, b)
    return states


def exact_probe_probability():
    """Integrate rates and sum all 2^12 assignment vectors independently."""
    masks = np.arange(1 << len(COUNTS))
    high = ((masks[:, None] >> np.arange(len(COUNTS))) & 1)
    assigned = np.stack([1 - high, high], axis=2)
    n = assigned.sum(axis=1)
    s = (assigned * COUNTS[None, :, None]).sum(axis=1)
    log_mass = (gammaln(A0 + s) - gammaln(A0) + A0 * np.log(B0)
                - (A0 + s) * np.log(B0 + n)).sum(axis=1)
    weights = np.exp(log_mass - logsumexp(log_mass))
    return float(weights @ high[:, PROBE_INDEX])
