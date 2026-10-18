"""Deterministic illustrative regression and orthogonal error experiments."""
import numpy as np

TRAIN_X = np.linspace(-1., 1., 12)
SEED_NOISE = 3
SEED_BOOTSTRAP = 203
M = 5
QUERY_X = .45
GRID = np.linspace(-1., 1., 501)
RIDGE = np.array([0., 0., .01, .02, .04, .08])


def truth(x):
    return np.sin(2.2 * np.asarray(x)) + .25 * np.asarray(x)


TRAIN_Y = truth(TRAIN_X) + np.random.default_rng(SEED_NOISE).normal(0., .18, 12)
BOOTSTRAP_INDICES = np.random.default_rng(SEED_BOOTSTRAP).integers(0, 12, (M, 12))


def design(x):
    return np.polynomial.legendre.legvander(np.asarray(x), 5)


def fit(indices):
    matrix = design(TRAIN_X[indices])
    return np.linalg.solve(matrix.T @ matrix + np.diag(RIDGE), matrix.T @ TRAIN_Y[indices])


COEFFICIENTS = np.array([fit(ids) for ids in BOOTSTRAP_INDICES])


def predict(x):
    return design(x) @ COEFFICIENTS.T


def comparison(grid=GRID):
    candidates = predict(grid)
    actual = truth(grid)
    individual = np.mean((candidates - actual[:, None]) ** 2, axis=0)
    mean_error = float(np.mean((np.mean(candidates, axis=1) - actual) ** 2))
    average_error = float(np.mean(individual))
    diversity = float(np.mean((candidates - np.mean(candidates, axis=1)[:, None]) ** 2))
    return dict(individual_mse=individual.tolist(), average_mse=average_error,
                committee_mse=mean_error, diversity=diversity,
                ratio=mean_error / average_error)


def orthogonal_errors(rho, grid=GRID):
    """Five unit-variance zero-mean Fourier errors, with a common component."""
    u = (np.asarray(grid) + 1.) / 2.
    common = np.sqrt(2.) * np.cos(2. * np.pi * 6. * u)
    unique = np.array([np.sqrt(2.) * np.cos(2. * np.pi * m * u)
                       for m in range(1, M + 1)]).T
    return np.sqrt(rho) * common[:, None] + np.sqrt(1. - rho) * unique


def orthogonal_expectation(rho):
    """Continuous uniform input on [-1,1]; exact Fourier orthogonality."""
    return rho + (1. - rho) / M
