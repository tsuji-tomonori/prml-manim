"""The same three-point example under raw interpolation and normalized regression."""
import numpy as np
X = np.array([-1., 0., 1.])
T = np.array([1., 0., 1.])
SIGMA = .7
PROBE = 2.5

def bumps(x, centers=X, sigma=SIGMA):
    return np.exp(-.5*((np.asarray(x)[..., None]-centers)/sigma)**2)

def design(centers=X, sigma=SIGMA):
    return bumps(X, centers, sigma)

H = design()
W = np.linalg.solve(H, T)

def raw(x):
    return bumps(x) @ W

def normalized_weights(x):
    values = bumps(x)
    return values / np.sum(values, axis=-1, keepdims=True)

def nw(x):
    return normalized_weights(x) @ T

RAW_PROBE = float(raw(PROBE))
NW_PROBE = float(nw(PROBE))
HEIGHTS = bumps(PROBE)
Q_PROBE = normalized_weights(PROBE)
