"""Deterministic polynomial example for PRML 10.3; Gamma uses shape/rate."""
import numpy as np
from scipy.special import digamma, gammaln
N = 10
DIM = 4
BETA = 1 / 0.09
A0 = 0.05
B0 = 0.05
SEED = 1030
X = np.linspace(-1.0, 1.0, N)
TRUE_W = np.array([0.2, 0.8, -0.1, 0.55])
T = np.vander(X, DIM, increasing=True) @ TRUE_W + np.random.default_rng(SEED).normal(0, np.sqrt(1 / BETA), N)
PROBE = 1.35


def phi(x):
    return np.vander(np.atleast_1d(x), DIM, increasing=True)


def step(alpha):
    design = phi(X)
    covariance = np.linalg.inv(alpha * np.eye(DIM) + BETA * design.T @ design)
    mean = BETA * covariance @ design.T @ T
    a = A0 + DIM / 2
    b = B0 + .5 * (mean @ mean + np.trace(covariance))
    return dict(alpha=a / b, input_alpha=alpha, a=a, b=b, mean=mean, covariance=covariance)


def elbo(state):
    """Equation 10.107 with corrected 10.110; proper Gamma prior."""
    m, s, a, b = (state[k] for k in ('mean', 'covariance', 'a', 'b'))
    design = phi(X)
    ea = a / b
    eloga = digamma(a) - np.log(b)
    ew2 = m @ m + np.trace(s)
    residual = np.sum((T - design @ m) ** 2) + np.trace(design @ s @ design.T)
    likelihood = N / 2 * np.log(BETA / (2 * np.pi)) - BETA / 2 * residual
    weight_prior = DIM / 2 * (eloga - np.log(2 * np.pi)) - ea / 2 * ew2
    alpha_prior = A0 * np.log(B0) - gammaln(A0) + (A0 - 1) * eloga - B0 * ea
    _, logdet = np.linalg.slogdet(s)
    weight_entropy = .5 * (DIM * (1 + np.log(2 * np.pi)) + logdet)
    alpha_entropy = a - np.log(b) + gammaln(a) + (1 - a) * digamma(a)
    return float(likelihood + weight_prior + alpha_prior + weight_entropy + alpha_entropy)


def fit(start=1., iterations=25):
    history = []
    alpha = float(start)
    for _ in range(iterations):
        state = step(alpha)
        state['bound'] = elbo(state)
        history.append(state)
        alpha = state['alpha']
    return history


def prediction(state, x=PROBE):
    f = phi([x])[0]
    return dict(mean=float(f @ state['mean']), noise_variance=1 / BETA,
                weight_variance=float(f @ state['covariance'] @ f),
                true_mean=float(f @ TRUE_W))
