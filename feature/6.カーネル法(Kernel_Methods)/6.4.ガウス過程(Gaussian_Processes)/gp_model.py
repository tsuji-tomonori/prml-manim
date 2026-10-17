"""Small, reproducible Gaussian-process example shared by all scenes."""
import numpy as np

X = np.array([-1., 0., 1.])
T = np.array([1., 0., 1.])
PROBE = 2.5
ELL = .7
NOISE_SD = .2


def kernel(a, b, ell=ELL):
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    return np.exp(-.5 * ((a[:, None] - b[None, :]) / ell) ** 2)


def covariance(ell=ELL, noise_sd=NOISE_SD):
    return kernel(X, X, ell) + noise_sd**2 * np.eye(len(X))


def predict(xs, ell=ELL, noise_sd=NOISE_SD, observed=True):
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    c = covariance(ell, noise_sd)
    cross = kernel(X, xs, ell)
    mean = cross.T @ np.linalg.solve(c, T)
    reduction = np.sum(cross * np.linalg.solve(c, cross), axis=0)
    variance = 1. - reduction + (noise_sd**2 if observed else 0.)
    return mean, np.maximum(variance, 0.)


def log_evidence(ell=ELL, noise_sd=NOISE_SD):
    c = covariance(ell, noise_sd)
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0:
        raise ValueError('covariance must be positive definite')
    return float(-.5 * logdet - .5 * T @ np.linalg.solve(c, T)
                 - len(T) / 2 * np.log(2 * np.pi))


def prior_samples(grid, count=4, seed=6404, ell=ELL):
    rng = np.random.default_rng(seed)
    gram = kernel(grid, grid, ell) + 1e-9 * np.eye(len(grid))
    return (np.linalg.cholesky(gram) @ rng.standard_normal((len(grid), count))).T


MEAN_PROBE, VAR_PROBE = predict([PROBE])
MEAN_PROBE = float(MEAN_PROBE[0])
SD_PROBE = float(np.sqrt(VAR_PROBE[0]))
