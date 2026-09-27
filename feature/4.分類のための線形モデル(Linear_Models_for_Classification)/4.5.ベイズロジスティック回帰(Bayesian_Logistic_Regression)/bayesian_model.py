"""Reproducible NumPy computations for PRML (4.140)--(4.155)."""
import math
import numpy as np

# Deliberately small, overlapping, invented binary observations.
X = np.array([-2.1, -1.7, -1.15, -.65, -.2, .25, .7, 1.1, 1.6, 2.05]) + .65
T = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
ALPHA = .55
PHI = np.column_stack((np.ones(len(X)), X))


def sigmoid(a):
    return 1 / (1 + np.exp(-np.clip(a, -700, 700)))


def normal(x, mean=0., var=1.):
    return np.exp(-.5 * (np.asarray(x) - mean)**2 / var) / np.sqrt(2*np.pi*var)


def cdf(x):
    return np.vectorize(lambda z: .5 * (1 + math.erf(z / np.sqrt(2))))(x)


def design(x):
    x = np.atleast_1d(x)
    return np.column_stack((np.ones(len(x)), x))


def objective(w, phi=PHI):
    a = phi @ np.atleast_1d(w)
    return .5*ALPHA*np.sum(np.asarray(w)**2) + np.sum(np.logaddexp(0, a) - T*a)


def derivatives(w, phi=PHI):
    y = sigmoid(phi @ np.atleast_1d(w))
    return ALPHA*np.atleast_1d(w) + phi.T@(y-T), ALPHA*np.eye(phi.shape[1]) + (phi.T*(y*(1-y)))@phi


def fit(phi=PHI):
    w = np.zeros(phi.shape[1])
    for _ in range(50):
        g, h = derivatives(w, phi)
        step = np.linalg.solve(h, g)
        scale = 1.
        while objective(w-scale*step, phi) > objective(w, phi) and scale > 1e-8:
            scale *= .5
        w -= scale*step
        if np.linalg.norm(g) < 1e-12:
            break
    return w, np.linalg.inv(derivatives(w, phi)[1])


MAP, COV = fit()
SCALAR_MAP, SCALAR_COV = fit(X[:, None])
SW = float(SCALAR_MAP[0])
SV = float(SCALAR_COV[0, 0])
GRID = np.linspace(-5, 7, 2401)
# Extend integration tails well beyond the plotted range.
INTEGRATION_GRID = np.linspace(-15, 15, 12001)


def scalar_density(grid=GRID, n=len(X)):
    # Fractional n smoothly introduces one observation's log likelihood.
    weights = np.clip(float(n) - np.arange(len(X)), 0, 1)
    def log_density(w):
        a = np.asarray(w)[:, None] * X
        return -.5*ALPHA*np.asarray(w)**2 + ((T*a-np.logaddexp(0, a))*weights).sum(axis=1)
    logz = log_density(INTEGRATION_GRID)
    peak = logz.max()
    z = np.trapezoid(np.exp(logz-peak), INTEGRATION_GRID)
    return np.exp(log_density(np.asarray(grid))-peak)/z


def scalar_energy(w):
    a = np.asarray(w)[..., None] * X
    return .5*ALPHA*np.asarray(w)**2 + np.sum(np.logaddexp(0, a)-T*a, axis=-1) - objective([SW], X[:, None])


def stats(x):
    phi = design(x)
    return phi@MAP, np.einsum('ij,jk,ik->i', phi, COV, phi)


def kappa(var):
    return 1 / np.sqrt(1 + np.pi*np.asarray(var)/8)


GH_X, GH_W = np.polynomial.hermite.hermgauss(100)


def predictive_integral(mu, var):
    mu, var = np.broadcast_arrays(mu, var)
    return np.sum(sigmoid(mu[..., None] + np.sqrt(2*var[..., None])*GH_X)*GH_W, axis=-1)/np.sqrt(np.pi)


def predict(x, scale=1):
    mu, var = stats(x)
    return sigmoid(kappa(scale*var)*mu)


SAMPLES = np.random.default_rng(45).multivariate_normal(MAP, COV, 10)
