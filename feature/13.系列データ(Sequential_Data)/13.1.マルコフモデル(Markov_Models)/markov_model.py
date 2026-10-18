"""Fixed pedagogical two-state state-space model for PRML 13.1."""
import numpy as np

TRANSITION = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
RAIN = np.array([0.2, 0.8], dtype=float)
INITIAL = np.array([0.5, 0.5], dtype=float)


def filter_history(history):
    """Return state posterior after each observation and next-state prior."""
    prior = INITIAL.copy()
    steps = []
    for observation in history:
        likelihood = RAIN if observation else 1.0 - RAIN
        unnormalized = prior * likelihood
        posterior = unnormalized / unnormalized.sum()
        next_prior = posterior @ TRANSITION
        steps.append(dict(prior=prior.copy(), posterior=posterior.copy(),
                          next_prior=next_prior.copy(), observation=observation))
        prior = next_prior
    return steps


def predict_rain(history):
    steps = filter_history(history)
    prior = steps[-1]['next_prior'] if steps else INITIAL
    return float(prior @ RAIN)


def parameters(k, order):
    return k ** order * (k - 1)
