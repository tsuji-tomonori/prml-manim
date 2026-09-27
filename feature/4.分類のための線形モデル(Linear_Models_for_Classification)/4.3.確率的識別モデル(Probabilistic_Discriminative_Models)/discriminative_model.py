"""Deterministic original experiments; no traced PRML data."""
import math
import numpy as np

def sigmoid(a):
    return np.exp(-np.logaddexp(0., -np.asarray(a)))

def softmax(a):
    a = np.asarray(a)
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def normal_pdf(a):
    return np.exp(-np.asarray(a)**2/2)/np.sqrt(2*np.pi)

def normal_cdf(a):
    return np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2)))(a)

X = np.array([-2.6,-2.1,-1.6,-1.1,-.65,-.2,.25,.7,1.15,1.6,2.1,2.6])
T = np.array([0,0,0,1,0,0,1,0,1,1,1,1.])
PHI = np.column_stack([np.ones_like(X),X])
SEP_T = (X>0).astype(float)

def loss(w, targets=T, lam=0.):
    a = PHI@w
    return float(np.sum(np.logaddexp(0,a)-targets*a)+lam*np.dot(w,w)/2)

def derivatives(w, targets=T, lam=0.):
    y = sigmoid(PHI@w)
    r = y*(1-y)
    return PHI.T@(y-targets)+lam*w, PHI.T@(r[:,None]*PHI)+lam*np.eye(2)

def fit(targets=T, lam=0., steps=7):
    w=np.zeros(2); history=[w.copy()]
    for _ in range(steps):
        g,h=derivatives(w,targets,lam)
        w=w-np.linalg.solve(h,g)
        history.append(w.copy())
    return np.array(history)

HISTORY=fit()

def interpolated_weight(step):
    i=min(int(step),len(HISTORY)-2)
    return HISTORY[i]+(step-i)*(HISTORY[i+1]-HISTORY[i])

def ring_data():
    theta=np.linspace(0,2*np.pi,18,endpoint=False)+.12
    inner=np.column_stack([.63*np.cos(theta),.63*np.sin(theta)])
    outer=np.column_stack([1.28*np.cos(theta+.08),1.28*np.sin(theta+.08)])
    return np.vstack([inner,outer]), np.r_[np.ones(18),np.zeros(18)]

RING,RING_T=ring_data()

def transformed_points(alpha):
    return (1-alpha)*RING+alpha*RING**2

def noisy_probability(a,epsilon):
    return epsilon+(1-2*epsilon)*sigmoid(a)
