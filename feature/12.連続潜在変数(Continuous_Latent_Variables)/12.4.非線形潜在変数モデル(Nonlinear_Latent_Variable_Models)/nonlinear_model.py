"""Deterministic examples for PRML 12.4; no learned neural net is implied."""
import numpy as np

T = np.linspace(-1.8, 1.8, 13)
OFFSETS = np.array([-.09, .06, -.04, .08, -.07, .05, 0., -.05, .07, -.08, .04, -.06, .09])

def curve(t):
    t = np.asarray(t)
    return np.stack([t, .48 * (t*t - 1.08)], axis=-1)

CENTERS = curve(T)
NORMALS = np.stack([-.96*T, np.ones_like(T)], axis=1)
NORMALS /= np.linalg.norm(NORMALS, axis=1)[:, None]
POINTS = CENTERS + OFFSETS[:, None] * NORMALS
MEAN = POINTS.mean(axis=0)
COV = (POINTS-MEAN).T @ (POINTS-MEAN) / len(T)
EIGENVALUES, EIGENVECTORS = np.linalg.eigh(COV)
AXIS = EIGENVECTORS[:, -1]
if AXIS[0] < 0:
    AXIS = -AXIS
LINEAR = MEAN + np.outer((POINTS-MEAN) @ AXIS, AXIS)
# Fixed illustrative encoder/decoder, not trained weights or a principal curve.
LATENT = POINTS[:, 0]
CURVED = curve(LATENT)
LINEAR_MSE = float(np.mean(np.sum((POINTS-LINEAR)**2, axis=1)))
CURVED_MSE = float(np.mean(np.sum((POINTS-CURVED)**2, axis=1)))
# Dependence with zero covariance: E[x(x^2-E[x^2])]=0 by symmetry.
DEPENDENT_X = np.linspace(-1.5, 1.5, 9)
DEPENDENT_Y = DEPENDENT_X**2 - np.mean(DEPENDENT_X**2)
# Sources are illustrative time series. Known, nonsingular mixing matrix.
TIME = np.arange(72)
SOURCE_1 = np.sin(2*np.pi*TIME/18)
SOURCE_2 = np.zeros(72)
SOURCE_2[[7, 20, 34, 47, 63]] = [1.4, -1.1, 1.2, -1.3, 1.5]
MIXING = np.array([[1., .55], [.45, 1.]])
MIXED = MIXING @ np.stack([SOURCE_1, SOURCE_2])
RESTORED = np.linalg.solve(MIXING, MIXED)
