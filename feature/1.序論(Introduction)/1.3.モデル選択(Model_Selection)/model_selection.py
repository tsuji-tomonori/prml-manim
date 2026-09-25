"""Reproducible experiments; no clipped curves, scores, or copied PRML data."""
import numpy as np

SIGMA = .25
DEGREES = np.arange(1, 10)
GRID = np.linspace(0, 1, 241)


def sample(n, seed):
    x = np.linspace(0, 1, n)
    return x, np.sin(2*np.pi*x) + np.random.default_rng(seed).normal(0, SIGMA, n)


def design(x, degree):
    return np.vander(np.atleast_1d(x), degree + 1, increasing=True)


def fit(x, t, degree, lam=0):
    a = design(x, degree)
    if lam:
        a = np.vstack([a, np.sqrt(lam)*np.eye(degree+1)])
        t = np.r_[t, np.zeros(degree+1)]
    return np.linalg.lstsq(a, t, rcond=None)[0]


def predict(w, x):
    return design(x, len(w)-1) @ w


def mse(w, x, t):
    return float(np.mean((predict(w, x)-t)**2))


X, T = sample(10, 7)
XV, TV = sample(24, 17)
XT, TT = sample(80, 37)
WEIGHTS = [np.pad(fit(X, T, int(d)), (0, 9-int(d))) for d in DEGREES]
TRAIN = np.array([mse(w, X, T) for w in WEIGHTS])
VALID = np.array([mse(w, XV, TV) for w in WEIGHTS])
SELECTED = int(DEGREES[np.argmin(VALID)])
# Only the selected model is evaluated against the final test set.
TEST_MSE = mse(WEIGHTS[SELECTED-1], XT, TT)
LOG_LIKELIHOOD = -len(X)*np.log(SIGMA*np.sqrt(2*np.pi)) - len(X)*TRAIN/(2*SIGMA**2)
PARAMETERS = DEGREES + 1  # sigma is known, not estimated.
AIC_SCORE = LOG_LIKELIHOOD - PARAMETERS


def interpolated_weights(d):
    d = float(np.clip(d, 1, 9))
    low = int(np.floor(d))
    high = min(9, low+1)
    return (1-(d-low))*WEIGHTS[low-1] + (d-low)*WEIGHTS[high-1]


XC, TC = sample(24, 52)
PERMUTATION = np.random.default_rng(91).permutation(24)
FOLDS = np.array_split(PERMUTATION, 4)


def cross_validate(degree=3, groups=FOLDS):
    models, scores, predictions = [], [], np.empty(len(XC))
    for held_out in groups:
        training = np.setdiff1d(np.arange(len(XC)), held_out)
        w = fit(XC[training], TC[training], degree)
        models.append(w)
        predictions[held_out] = predict(w, XC[held_out])
        scores.append(mse(w, XC[held_out], TC[held_out]))
    return models, np.array(scores), predictions


CV_WEIGHTS, CV_SCORES, OOF_PREDICTIONS = cross_validate()
CV_DEGREES = np.arange(1, 6)
CV_MEANS = np.array([cross_validate(int(d))[1].mean() for d in CV_DEGREES])
CV_SELECTED = int(CV_DEGREES[np.argmin(CV_MEANS)])
LOO_WEIGHTS, LOO_SCORES, LOO_PREDICTIONS = cross_validate(groups=np.array_split(PERMUTATION, 24))
