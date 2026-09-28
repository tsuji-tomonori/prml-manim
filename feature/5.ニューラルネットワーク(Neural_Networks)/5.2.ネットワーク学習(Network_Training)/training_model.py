"""Deterministic numerical experiments; no hand-drawn optimization paths."""
import numpy as np
X=np.linspace(-1.6,1.6,13)
T=.8*np.tanh(1.8*X)-.35*np.tanh(3*(X-.5))+np.random.default_rng(52).normal(0,.07,len(X))
def prediction(x,w): return w*np.tanh(1.8*x)-.35*np.tanh(3*(x-.5))
def regression_error(w): return .5*np.sum((prediction(X,w)-T)**2)
REG_W=float(np.linalg.lstsq(np.tanh(1.8*X)[:,None],T+.35*np.tanh(3*(X-.5)),rcond=None)[0][0])
VARIANCE=float(2*regression_error(REG_W)/len(X))
def sigmoid(a): return 1/(1+np.exp(-np.asarray(a)))
def softmax(a):
    a=np.asarray(a); z=np.exp(a-np.max(a)); return z/z.sum()
def binary_loss(a,t): return np.logaddexp(0,a)-t*a
# Explicit educational error surface, distinct from the network above.
def landscape(w): return .18*(np.asarray(w)**2-2.2)**2+.16*np.asarray(w)+.45
def landscape_grad(w): return .72*w*(w*w-2.2)+.16
def landscape_hess(w): return 2.16*w*w-1.584
STATIONARY=np.sort(np.roots([.72,0,-1.584,.16]))
def rotation(theta): return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
def hessian(lam=9,theta=.4):
    r=rotation(theta); return r@np.diag([1.,lam])@r.T
H=hessian()
def descent(eta,steps=20,start=None):
    w=rotation(.4)@np.array([1.65,.34]) if start is None else np.array(start,dtype=float)
    rows=[w.copy()]
    for _ in range(steps):
        w=w-eta*H@w; rows.append(w.copy())
    return np.array(rows)
def sample_path(path,u):
    u=np.clip(u,0,len(path)-1); i=min(int(u),len(path)-2)
    return (1-(u-i))*path[i]+(u-i)*path[i+1]
TARGETS=np.array([-.9,.1,1.4,.6])
def batch_online(online=False,steps=24,start=-1.4):
    w=start; rows=[w]
    for n in range(steps):
        w-=.3*(w-TARGETS[n%4]) if online else .1*np.sum(w-TARGETS)
        rows.append(w)
    return np.array(rows)
def scalar_error(w): return .5*np.sum((np.asarray(w)[...,None]-TARGETS)**2,axis=-1)
