"""Reproducible 20-point conditional Gaussian mixture for PRML 14.5.

The gate is an explanatory second-stage weighted logistic fit, rather than a
joint maximum-likelihood fit of the full hierarchical mixture of experts.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

UPPER_X = np.array([-.9,-.75,-.6,-.45,-.3,-.15,0,.15,.3,.45,.6,.75,.9])
LOWER_X = np.array([-.9,-.75,-.6,-.45,-.3,-.15,0])
UPPER_T = 1+.18*UPPER_X+3*np.array([.03,-.06,.02,.05,-.04,.01,-.03,.04,-.02,.05,-.01,.03,-.04])
LOWER_T = -1-.12*LOWER_X+3*np.array([-.02,.03,-.05,.01,.04,-.03,.02])
X = np.r_[UPPER_X, LOWER_X]
T = np.r_[UPPER_T, LOWER_T]
PHI = np.c_[np.ones(len(X)), X]
QUERY = .75


def expectation(weights, mixing, precision):
    delta = T[:,None]-PHI@weights.T
    logp = np.log(mixing)[None,:]+.5*np.log(precision/(2*np.pi))-.5*precision*delta**2
    responsibilities = np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
    return responsibilities, float(logsumexp(logp,axis=1).sum())


def fit_mixture(iterations=20):
    weights=np.array([[.6,0.],[-.6,0.]])
    mixing=np.array([.5,.5])
    precision=4.
    history=[dict(weights=weights.copy(),mixing=mixing.copy(),precision=precision,
                  log_likelihood=expectation(weights,mixing,precision)[1])]
    for _ in range(iterations):
        responsibility,_=expectation(weights,mixing,precision)
        mixing=responsibility.mean(axis=0)
        weights=np.array([np.linalg.solve(PHI.T@(responsibility[:,k,None]*PHI),
                                          PHI.T@(responsibility[:,k]*T)) for k in range(2)])
        precision=len(X)/np.sum(responsibility*(T[:,None]-PHI@weights.T)**2)
        history.append(dict(weights=weights.copy(),mixing=mixing.copy(),precision=float(precision),
                            log_likelihood=expectation(weights,mixing,precision)[1]))
    # The initial upper component remains component 0 for these fixed data.
    assert weights[0,0]>weights[1,0]
    responsibility,_=expectation(weights,mixing,precision)
    return history,responsibility


def fit_gate(responsibility, penalty=.02):
    """Fit P(lower component | x) from the final EM responsibilities."""
    target=responsibility[:,1]
    def objective(parameters):
        z=parameters[0]+parameters[1]*X
        return float(np.logaddexp(0,z).sum()-target@z+penalty*parameters[1]**2)
    result=minimize(objective,[-1.,-2.],method='BFGS',tol=1e-11)
    if not result.success and np.linalg.norm(result.jac)>1e-5:
        raise RuntimeError(result.message)
    return result.x


def lower_gate(x,parameters):
    return expit(parameters[0]+parameters[1]*np.asarray(x))


def normal_pdf(t,mean,precision):
    t=np.asarray(t)
    return np.sqrt(precision/(2*np.pi))*np.exp(-.5*precision*(t-mean)**2)


def predictive_density(t,x,weights,mixing,precision):
    mean=weights[:,0]+weights[:,1]*x
    return sum(mixing[k]*normal_pdf(t,mean[k],precision) for k in range(2))


if __name__=='__main__':
    states,responsibility=fit_mixture()
    gate=fit_gate(responsibility)
    print('mixing',states[-1]['mixing'],'weights',states[-1]['weights'],'precision',states[-1]['precision'])
    print('gate',gate,'lower at x=0.75',lower_gate(QUERY,gate))
