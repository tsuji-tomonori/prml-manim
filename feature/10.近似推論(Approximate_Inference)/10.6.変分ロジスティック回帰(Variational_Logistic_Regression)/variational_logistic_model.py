"""Self-contained 10 point experiment for PRML 10.6 (corrected errata)."""
from __future__ import annotations
import numpy as np
from scipy.special import expit, log_expit, roots_hermitenorm

Y = np.array([-.8, -.4, 0., .4, .8])
X = np.array([-1.45, -1.30, -1.50, -1.35, -1.40,
               1.45, 1.30, 1.50, 1.35, 1.40])
XY = np.column_stack((X, np.r_[Y, Y]))
PHI = np.column_stack((np.ones(10), XY))
T = np.r_[np.zeros(5), np.ones(5)]
ALPHA = .4
QUERY = np.array([1., 1.4, 4.])
NEAR = np.array([1., 1.4, 0.])

def lam(xi):
    xi = np.asarray(xi, dtype=float)
    return np.where(abs(xi) < 1e-8, 1/8, np.tanh(xi/2)/(4*np.maximum(abs(xi), 1e-8)))

def local_bound(z, xi):
    return expit(xi) * np.exp((np.asarray(z)-xi)/2-lam(xi)*(np.asarray(z)**2-xi**2))

def posterior(xi, alpha=ALPHA):
    weights = lam(xi)
    precision = alpha*np.eye(3) + 2*PHI.T@(weights[:,None]*PHI)
    covariance = np.linalg.inv(precision)
    mean = covariance@(PHI.T@(T-.5))
    return mean, covariance

def lower_bound(xi, mean, covariance, alpha=ALPHA):
    # PRML Eq. (10.164), with the signs corrected in the official errata.
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    precision = np.linalg.inv(covariance)
    return (.5*(logdet+3*np.log(alpha)) + .5*mean@precision@mean
            + np.sum(log_expit(xi)-xi/2+lam(xi)*xi**2))

def iterate(xi_initial=1., alpha=ALPHA, count=100, tolerance=1e-12):
    xi=np.full(10,xi_initial,dtype=float)
    history=[]
    for _ in range(count):
        mean,covariance=posterior(xi,alpha)
        value=lower_bound(xi,mean,covariance,alpha)
        history.append(value)
        new=np.sqrt(np.sum((PHI@covariance)*PHI,axis=1)+(PHI@mean)**2)
        if np.max(abs(new-xi))<tolerance:
            xi=new
            break
        xi=new
    mean,covariance=posterior(xi,alpha)
    history.append(lower_bound(xi,mean,covariance,alpha))
    return dict(xi=xi,mean=mean,covariance=covariance,history=np.array(history),iterations=len(history)-1)

def prediction(phi,mean,covariance,order=320):
    phi=np.asarray(phi)
    mu=float(phi@mean)
    variance=float(phi@covariance@phi)
    nodes,weights=roots_hermitenorm(order)
    average=float(np.dot(weights,expit(mu+np.sqrt(variance)*nodes))/np.sqrt(2*np.pi))
    kappa=float(expit(mu/np.sqrt(1+np.pi*variance/8)))
    return dict(score_mean=mu,score_variance=variance,representative=float(expit(mu)),
                gaussian_average=average,kappa_approximation=kappa)

def hierarchical(a0=2.,b0=2.,count=80):
    """Factorized q(w)q(alpha): the 10.6.3 extension, separate from main example."""
    xi=np.ones(10)
    alpha=a0/b0
    rows=[]
    for _ in range(count):
        mean,covariance=posterior(xi,alpha)
        shape=a0+3/2
        rate=b0+.5*(np.trace(covariance)+mean@mean)
        alpha=shape/rate
        xi=np.sqrt(np.sum((PHI@covariance)*PHI,axis=1)+(PHI@mean)**2)
        rows.append((alpha,shape,rate,float(np.mean(xi))))
    return np.array(rows)
