"""A symmetric 16-point example of kernel PCA in PRML 12.3."""
import numpy as np

POINTS = np.array([[sx*1.2+dx, sy*1.0+dy]
                   for sx in (-1, 1) for sy in (-1, 1)
                   for dx,dy in ((-.08,-.05),(-.08,.05),(.08,-.05),(.08,.05))], dtype=float)
LABELS = np.array([sx*sy for sx in (-1,1) for sy in (-1,1) for _ in range(4)])
N,D = POINTS.shape
MEAN = POINTS.mean(axis=0)
COV = (POINTS-MEAN).T@(POINTS-MEAN)/N
LINEAR_EIGENVALUES = np.linalg.eigvalsh(COV)[::-1]

def phi(x):
    x=np.asarray(x)
    return np.stack((x[...,0]**2,np.sqrt(2)*x[...,0]*x[...,1],x[...,1]**2),axis=-1)

def kernel(x,z):
    return (np.asarray(x)@np.asarray(z).T)**2

FEATURES = phi(POINTS)
FEATURE_MEAN = FEATURES.mean(axis=0)
CENTERED_FEATURES = FEATURES-FEATURE_MEAN
FEATURE_COV = CENTERED_FEATURES.T@CENTERED_FEATURES/N
FEATURE_EIGENVALUES,FEATURE_EIGENVECTORS = np.linalg.eigh(FEATURE_COV)
order=np.argsort(FEATURE_EIGENVALUES)[::-1]
FEATURE_EIGENVALUES=FEATURE_EIGENVALUES[order]
FEATURE_EIGENVECTORS=FEATURE_EIGENVECTORS[:,order]
if FEATURE_EIGENVECTORS[1,0]<0:
    FEATURE_EIGENVECTORS[:,0]*=-1
SCORES=CENTERED_FEATURES@FEATURE_EIGENVECTORS[:,0]
GRAM=kernel(POINTS,POINTS)
H=np.eye(N)-np.ones((N,N))/N
CENTERED_GRAM=H@GRAM@H
GRAM_EIGENVALUES=np.linalg.eigvalsh(CENTERED_GRAM)[::-1]
# PRML (12.76), (12.80) and (12.81): v=sum_n a_n phi_c(x_n), v^T v=1.
ALPHA=SCORES/(N*FEATURE_EIGENVALUES[0])
NEW_POINT=np.array([.9,.8])

def centered_kernel_vector(x):
    raw=kernel(POINTS,x)
    return raw-GRAM.mean(axis=1)-raw.mean()+GRAM.mean()

def project(x):
    return float(centered_kernel_vector(x)@ALPHA)

NEW_SCORE=project(NEW_POINT)
PREIMAGE_POINT=POINTS[-1]
PREIMAGE_SCORE=project(PREIMAGE_POINT)
PREIMAGE_LHS=PREIMAGE_SCORE**2
PREIMAGE_RHS=2*FEATURE_MEAN[0]*FEATURE_MEAN[2]
