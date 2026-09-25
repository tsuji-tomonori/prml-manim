"""Reproducible numerical experiment; no dependency on Manim."""
from functools import lru_cache
import numpy as np

NOISE_STD = 0.25
TRAIN_SEED = 2
TEST_SEED = 17
EXTRA_SEED = 6


def sine(x):
    return np.sin(2 * np.pi * np.asarray(x))


def design_matrix(x, degree):
    return np.vander(np.atleast_1d(x), degree + 1, increasing=True)


def fit_polynomial(x, t, degree, lam=0.0):
    phi = design_matrix(x, degree)
    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    if lam:
        # Augmented least squares avoids squaring the condition number.
        # Eq.(1.4): penalize ALL coefficients, including w0.
        phi = np.vstack([phi, np.sqrt(lam) * np.eye(degree + 1)])
        t = np.r_[t, np.zeros(degree + 1)]
    return np.linalg.lstsq(phi, t, rcond=None)[0]


def eval_poly(w, x):
    return np.polynomial.polynomial.polyval(x, w)


def rms_error(w, x, t):
    return float(np.sqrt(np.mean((eval_poly(w, x) - t) ** 2)))


def data():
    x = np.linspace(0, 1, 10)
    t = sine(x) + np.random.default_rng(TRAIN_SEED).normal(0, NOISE_STD, 10)
    xt = np.linspace(0, 1, 100)
    tt = sine(xt) + np.random.default_rng(TEST_SEED).normal(0, NOISE_STD, 100)
    rng = np.random.default_rng(EXTRA_SEED)
    xe = rng.uniform(0, 1, 90)
    te = sine(xe) + rng.normal(0, NOISE_STD, 90)
    return x, t, xt, tt, np.r_[x, xe], np.r_[t, te]


X, T, XT, TT, X_ALL, T_ALL = data()
WEIGHTS = [np.pad(fit_polynomial(X, T, m), (0, 9 - m)) for m in range(10)]
TRAIN_RMS = np.array([rms_error(w, X, T) for w in WEIGHTS])
TEST_RMS = np.array([rms_error(w, XT, TT) for w in WEIGHTS])


def degree_weights(value):
    """Visual morph only: M remains integer at each fitted endpoint."""
    low = int(np.clip(np.floor(value), 0, 9))
    high = min(low + 1, 9)
    a = float(value - low)
    return (1 - a) * WEIGHTS[low] + a * WEIGHTS[high]


@lru_cache(maxsize=2048)
def ridge_weights(log_lambda):
    return fit_polynomial(X, T, 9, np.exp(log_lambda))


@lru_cache(maxsize=100)
def sample_weights(n):
    return fit_polynomial(X_ALL[:n], T_ALL[:n], 9)


def growing_weights(value):
    low = int(np.clip(np.floor(value), 10, 100))
    high = min(low + 1, 100)
    a = value - low
    return (1 - a) * sample_weights(low) + a * sample_weights(high)
