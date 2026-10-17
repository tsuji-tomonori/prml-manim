"""Reproducible hard- and soft-margin examples for PRML 7.1."""
import numpy as np
from scipy.optimize import minimize

X=np.array([[-1.,0.],[-2.,-1.],[-2.,1.],[1.,0.],[2.,-1.],[2.,1.]])
T=np.array([-1.,-1.,-1.,1.,1.,1.])
EXTRA=np.array([1.4,0.])

def signed_distance(x,t,position):
    return np.asarray(t)*(np.asarray(x)[...,0]-position)

def hard_margin(position):
    return float(np.min(signed_distance(X,T,position)))

def soft_svm(c):
    x=np.vstack([X,EXTRA]);t=np.r_[T,-1.]
    q=np.outer(t,t)*(x@x.T)
    result=minimize(lambda a:.5*a@q@a-a.sum(),np.zeros(len(t)),
                    jac=lambda a:q@a-np.ones(len(t)),method='SLSQP',
                    bounds=[(0.,c)]*len(t),
                    constraints=[dict(type='eq',fun=lambda a:float(a@t),jac=lambda a:t)],
                    options=dict(ftol=1e-12,maxiter=3000))
    if not result.success: raise RuntimeError(result.message)
    a=result.x;w=(a*t)@x
    free=(a>1e-6)&(a<c-1e-6)
    if not free.any(): raise RuntimeError('No free support vector')
    b=float(np.mean(t[free]-x[free]@w))
    xi=np.maximum(0.,1.-t*(x@w+b))
    primal=.5*w@w+c*xi.sum();dual=a.sum()-.5*a@q@a
    return dict(x=x,t=t,alpha=a,w=w,b=b,slack=xi,primal=primal,dual=dual,free=free,decision_x=-b/w[0])
