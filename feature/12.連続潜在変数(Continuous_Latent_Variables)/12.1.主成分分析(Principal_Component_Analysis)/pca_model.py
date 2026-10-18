"""Fixed, illustrative 13-point PCA example; PRML uses the 1/N covariance."""
import numpy as np

T = np.linspace(-2.6, 2.6, 13)
NOISE = np.array([.24, -.37, .48, -.52, .31, -.13, .08, .33, -.26, .46, -.44, .22, -.40])
POINTS = np.column_stack((T + 1.15, .72 * T + .75 * NOISE + .35))
N, D = POINTS.shape
MEAN = POINTS.mean(axis=0)
CENTERED = POINTS - MEAN
COV = CENTERED.T @ CENTERED / N
EIGENVALUES_ASC, EIGENVECTORS_ASC = np.linalg.eigh(COV)
EIGENVALUES = EIGENVALUES_ASC[::-1]
EIGENVECTORS = EIGENVECTORS_ASC[:, ::-1]
for col in range(D):
    EIGENVECTORS[:, col] *= np.sign(EIGENVECTORS[0, col])
THETA = float(np.arctan2(EIGENVECTORS[1, 0], EIGENVECTORS[0, 0]))


def direction(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def projected(theta):
    u = direction(theta)
    return MEAN + np.outer(CENTERED @ u, u)


def variance(theta):
    u = direction(theta)
    return float(u @ COV @ u)


def error(theta):
    return float(np.mean(np.sum((POINTS - projected(theta)) ** 2, axis=1)))


def reconstruction(m):
    if not 0 <= m <= D:
        raise ValueError(m)
    basis = EIGENVECTORS[:, :m]
    return MEAN + (CENTERED @ basis) @ basis.T


def reconstruction_error(m):
    return float(np.mean(np.sum((POINTS - reconstruction(m)) ** 2, axis=1)))
