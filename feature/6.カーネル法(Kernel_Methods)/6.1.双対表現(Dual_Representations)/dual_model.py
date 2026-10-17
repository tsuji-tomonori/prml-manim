"""Reproducible three-observation example for PRML 6.1."""
from __future__ import annotations
import numpy as np

X = np.array([-1., 0., 1.])
T = np.array([1., 0., 1.])
LAMBDA = 0.5
PROBE = 0.5

def phi(x):
    x = np.asarray(x, dtype=float)
    return np.stack((np.ones_like(x), x, x*x), axis=-1)

def kernel(x, z):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    return 1 + x*z + x*x*z*z

PHI = phi(X)
K = PHI @ PHI.T
W = np.linalg.solve(PHI.T @ PHI + LAMBDA*np.eye(PHI.shape[1]), PHI.T @ T)
A = np.linalg.solve(K + LAMBDA*np.eye(len(X)), T)
K_PROBE = kernel(X, PROBE)
PREDICTION = float(phi(PROBE) @ W)
CONTRIBUTIONS = K_PROBE * A


def primal(lam=LAMBDA):
    return np.linalg.solve(PHI.T @ PHI + lam*np.eye(PHI.shape[1]), PHI.T @ T)


def dual(lam=LAMBDA):
    return np.linalg.solve(K + lam*np.eye(len(X)), T)

# A second valid polynomial kernel used only in the closing comparison.
def alternative_kernel(x,z):
    x, z = np.asarray(x,dtype=float), np.asarray(z,dtype=float)
    return (1+x*z)**3

ALT_K = alternative_kernel(X[:,None],X[None,:])
ALT_A = np.linalg.solve(ALT_K+LAMBDA*np.eye(len(X)),T)
ALT_PREDICTION = float(alternative_kernel(X,PROBE)@ALT_A)
