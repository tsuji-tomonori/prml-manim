"""Reproducible independent-data experiment; all displayed values are computed."""
from functools import lru_cache
import numpy as np

SEED, L, N, SIGMA = 3202, 100, 25, .25
CENTERS = np.linspace(0, 1, 24)
GRID = np.linspace(0, 1, 201)
LOG_LAMBDAS = np.linspace(-3, 3, 61)

def truth(x):
    return np.sin(2*np.pi*np.asarray(x))

def design(x):
    x = np.asarray(x)
    return np.concatenate([np.ones((*x.shape, 1)), np.exp(-.5*((x[..., None]-CENTERS)/.08)**2)], axis=-1)

rng = np.random.default_rng(SEED)
X = np.sort(rng.uniform(size=(L, N)), axis=1)
T = truth(X) + rng.normal(0, SIGMA, X.shape)
PHI = design(X)
GRAM = np.einsum('lni,lnj->lij', PHI, PHI)
RHS = np.einsum('lni,ln->li', PHI, T)
EIGENVALUES, EIGENVECTORS = np.linalg.eigh(GRAM)
ROTATED_RHS = np.einsum('lij,li->lj', EIGENVECTORS, RHS)
EVAL_X = rng.uniform(size=2000)
TEST_X = rng.uniform(size=1000)
TEST_T = truth(TEST_X) + rng.normal(0, SIGMA, TEST_X.shape)
EVAL_PHI, TEST_PHI = design(EVAL_X), design(TEST_X)
G_EVAL = EVAL_PHI.T @ EVAL_PHI / len(EVAL_X)
H_EVAL = EVAL_PHI.T @ truth(EVAL_X) / len(EVAL_X)
H2_EVAL = np.mean(truth(EVAL_X)**2)
G_TEST = TEST_PHI.T @ TEST_PHI / len(TEST_X)
T_TEST = TEST_PHI.T @ TEST_T / len(TEST_X)
T2_TEST = np.mean(TEST_T**2)

@lru_cache(maxsize=4)
def experiment(log_lambda):
    weights = np.einsum('lij,lj->li', EIGENVECTORS, ROTATED_RHS/(EIGENVALUES+np.exp(log_lambda)))
    mean_w = weights.mean(axis=0)
    delta = weights-mean_w
    bias2 = float(mean_w @ G_EVAL @ mean_w - 2*mean_w@H_EVAL + H2_EVAL)
    variance = float(np.einsum('li,ij,lj->', delta, G_EVAL, delta)/L)
    test = float(np.einsum('li,ij,lj->', weights,G_TEST,weights)/L-2*mean_w@T_TEST+T2_TEST)
    curves = weights @ design(GRID).T
    return dict(weights=weights, curves=curves, mean=curves.mean(axis=0), bias2=bias2, variance=variance, test=test,
                expected=bias2+variance+SIGMA**2)

METRICS = np.array([[experiment(float(l))[k] for k in ['bias2','variance','expected','test']] for l in LOG_LAMBDAS])
BEST = float(LOG_LAMBDAS[np.argmin(METRICS[:,0]+METRICS[:,1])])
