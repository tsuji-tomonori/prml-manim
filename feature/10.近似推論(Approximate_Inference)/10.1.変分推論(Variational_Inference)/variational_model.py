"""A fixed correlated Gaussian posterior for the §10.1 mean-field story."""
import numpy as np

PRECISION = np.array([[2.0, -1.6], [-1.6, 2.0]])
COVARIANCE = np.linalg.inv(PRECISION)
EVIDENCE = 0.2
INITIAL_MEAN = np.array([1.5, 0.8])
FACTOR_VARIANCE = 1 / PRECISION[0, 0]


def q_covariance(variance=FACTOR_VARIANCE):
    return np.diag([variance, variance])


def kl_qp(mean, covariance=None):
    """KL[N(mean,covariance) || N(0,COVARIANCE)]."""
    mean = np.asarray(mean, dtype=float)
    covariance = q_covariance() if covariance is None else np.asarray(covariance)
    return float((np.trace(PRECISION @ covariance) + mean @ PRECISION @ mean - 2
                  + np.log(np.linalg.det(COVARIANCE) / np.linalg.det(covariance))) / 2)


def lower_bound(mean, covariance=None):
    return float(np.log(EVIDENCE) - kl_qp(mean, covariance))


def coordinate_steps(cycles=8):
    mean = INITIAL_MEAN.copy()
    rows = [dict(which='initial', mean=mean.copy(), kl=kl_qp(mean), lower=lower_bound(mean))]
    for _ in range(cycles):
        mean[0] = 0.8 * mean[1]
        rows.append(dict(which='z1', mean=mean.copy(), kl=kl_qp(mean), lower=lower_bound(mean)))
        mean[1] = 0.8 * mean[0]
        rows.append(dict(which='z2', mean=mean.copy(), kl=kl_qp(mean), lower=lower_bound(mean)))
    return rows


def gaussian_density(points, mean, covariance):
    points = np.asarray(points)
    diff = points - np.asarray(mean)
    inv = np.linalg.inv(covariance)
    d2 = np.einsum('...i,ij,...j->...', diff, inv, diff)
    return np.exp(-d2 / 2) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))


def contour(mean, covariance, radius, count=101):
    theta = np.linspace(0, 2 * np.pi, count)
    circle = np.stack((np.cos(theta), np.sin(theta)))
    return (np.asarray(mean)[:, None] + radius * np.linalg.cholesky(covariance) @ circle).T
