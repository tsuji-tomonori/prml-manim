"""Numerical source for all plots; synthetic data, no textbook figure copying."""
import numpy as np
ALPHA, BETA = 2., 25.
FORECAST_X = 1.5
HELD_OUT_T = .15 + .65 * FORECAST_X + np.random.default_rng(3304).normal(0, .2)
rng = np.random.default_rng(3303)
X = np.r_[.65, -.65, rng.uniform(-1, 1, 18)]
T = .15 + .65 * X + rng.normal(0, .2, len(X))
XR = np.r_[.35, .75, .15, .9, np.random.default_rng(331).uniform(0, 1, 21)]
TR = np.sin(2*np.pi*XR) + np.random.default_rng(332).normal(0, .2, 25)
CENTERS = np.linspace(0, 1, 9)
Z = np.random.default_rng(337).normal(size=(6, 2))
ZR = np.random.default_rng(339).normal(size=(6, 9))

def phi(x, kind='line'):
    x = np.atleast_1d(x)
    return np.column_stack([np.ones_like(x), x]) if kind=='line' else np.exp(-.5*((x[:,None]-CENTERS)/.14)**2)

def posterior(n, alpha=ALPHA, beta=BETA, kind='line'):
    x,t = (X,T) if kind=='line' else (XR,TR)
    f = phi(x,kind)
    weights = np.clip(n-np.arange(len(x)),0,1)
    precision = alpha*np.eye(f.shape[1]) + beta*f.T @ (weights[:,None]*f)
    cov = np.linalg.solve(precision,np.eye(f.shape[1]))
    mean = np.linalg.solve(precision,beta*f.T@(weights*t))
    return mean,cov

def predict(x, n, alpha=ALPHA, beta=BETA, kind='line'):
    mean,cov=posterior(n,alpha,beta,kind)
    f=phi(x,kind)
    latent=np.einsum('ij,jk,ik->i',f,cov,f)
    return f@mean, latent, latent+1/beta

def samples(n, alpha=ALPHA, beta=BETA, kind='line', phase=0):
    mean,cov=posterior(n,alpha,beta,kind)
    z=Z if kind=='line' else ZR
    # Orthogonal mixing of independent standard normals preserves each marginal.
    other=np.roll(z,1,axis=0)
    return mean+(np.cos(phase)*z+np.sin(phase)*other)@np.linalg.cholesky(cov).T

def kernel(x, z, n=25, alpha=ALPHA, beta=BETA):
    _,cov=posterior(n,alpha,beta,'rbf')
    return beta*phi(x,'rbf')@cov@phi(z,'rbf').T
