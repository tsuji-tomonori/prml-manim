"""A 1D probabilistic PCA fit to the self-contained 13-point example."""
import numpy as np
from pca_model import POINTS, N, D, MEAN, CENTERED, COV, EIGENVALUES, EIGENVECTORS

M = 1
U = EIGENVECTORS[:, 0]
LAMBDA1, LAMBDA2 = map(float, EIGENVALUES)
SIGMA2 = LAMBDA2
W = U * np.sqrt(LAMBDA1 - SIGMA2)
MODEL_COV = np.outer(W, W) + SIGMA2 * np.eye(D)
A = MEAN.copy()
B = MEAN + 6 * U
SCORE_B = 6.0
DENSITY_RATIO = float(np.exp(-SCORE_B**2 / (2*LAMBDA1)))
POSTERIOR_VARIANCE = SIGMA2 / LAMBDA1


def posterior(x, w=W, sigma2=SIGMA2):
    """Scalar latent Gaussian from (12.42), corrected covariance sigma²/M."""
    den = float(w @ w + sigma2)
    return float(w @ (np.asarray(x) - MEAN) / den), float(sigma2 / den)


def density(x):
    delta = np.asarray(x) - MEAN
    sign, logdet = np.linalg.slogdet(MODEL_COV)
    assert sign > 0
    return float(np.exp(-.5*(D*np.log(2*np.pi)+logdet+delta @ np.linalg.solve(MODEL_COV, delta))))


def em_fit(iterations=24):
    """Fixed-mean PPCA EM, Bishop (12.54)–(12.57), D=2,M=1."""
    w = np.sqrt(LAMBDA1-LAMBDA2) * np.array([1., 0.])
    sigma2 = .3
    records = []
    for step in range(iterations+1):
        records.append(dict(step=step, w=w.copy(), sigma2=float(sigma2),
                            angle=float(np.degrees(np.arctan2(w[1],w[0])))))
        if step == iterations:
            break
        den = float(w @ w + sigma2)
        ez = CENTERED @ w / den
        ezz = sigma2 / den + ez*ez
        w_new = CENTERED.T @ ez / ezz.sum()
        residual = (np.sum(CENTERED*CENTERED) - 2*np.sum(ez*(CENTERED @ w_new))
                    + np.sum(ezz)*(w_new @ w_new))
        sigma2 = float(residual/(N*D))
        w = w_new
    return records


def missing_x2_given_x1(x1):
    return float(MEAN[1]+MODEL_COV[1,0]/MODEL_COV[0,0]*(x1-MEAN[0]))
