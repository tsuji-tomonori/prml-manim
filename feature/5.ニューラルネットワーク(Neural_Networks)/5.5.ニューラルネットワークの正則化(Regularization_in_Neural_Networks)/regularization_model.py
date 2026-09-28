"""Deterministic numerical experiments; no hand-drawn learning curves."""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize

SEED = 5506
X = np.linspace(.03, .97, 10)
T = np.sin(2*np.pi*X) + np.random.default_rng(SEED).normal(0, .22, len(X))
XV = np.linspace(.01, .99, 60)
TV = np.sin(2*np.pi*XV) + np.random.default_rng(SEED+1).normal(0, .22, len(XV))
GRID = np.linspace(0, 1, 241)


def unpack(w):
    m=(len(w)-1)//3
    return w[:m],w[m:2*m],w[2*m:3*m],w[-1]


def predict(w,x):
    a,b,v,c=unpack(w)
    return np.tanh(np.asarray(x)[...,None]*a+b)@v+c


def objective(w,lam=0,x=X,t=T):
    a,b,v,c=unpack(w)
    z=np.tanh(x[:,None]*a+b);r=z@v+c-t
    dz=r[:,None]*v*(1-z*z)
    grad=np.r_[np.sum(dz*x[:,None],axis=0),dz.sum(0),z.T@r,r.sum()]
    return .5*(r@r+lam*(w@w)),grad+lam*w


def initial(m,seed):
    rng=np.random.default_rng(seed)
    return np.r_[rng.normal(0,6,m),rng.normal(0,2,m),rng.normal(0,.3,m),0.]


@lru_cache(None)
def fit(m=10,seed=550,lam=1e-5):
    return minimize(objective,initial(m,seed),args=(lam,),jac=True,method='L-BFGS-B',
                    options={'maxiter':1600,'ftol':1e-12,'gtol':1e-8}).x


@lru_cache(None)
def ridge_path():
    logs=np.linspace(-5,1,49)
    weights=[]
    w=fit().copy()
    for z in logs:
        w=minimize(objective,w,args=(10**z,),jac=True,method='L-BFGS-B',
                   options={'maxiter':800,'ftol':1e-11,'gtol':1e-7}).x
        weights.append(w.copy())
    return logs,np.array(weights)


def ridge_weights(loglam):
    logs,w=ridge_path()
    q=np.clip((loglam-logs[0])/(logs[1]-logs[0]),0,len(logs)-1)
    i=min(int(q),len(logs)-2);return (1-(q-i))*w[i]+(q-i)*w[i+1]


def mse(w,x=X,t=T):
    return float(np.mean((predict(w,x)-t)**2))


@lru_cache(None)
def training_history():
    w=initial(10,552); mom=np.zeros_like(w);v=np.zeros_like(w)
    weights=[w.copy()]
    for k in range(1,4001):
        _,g=objective(w)
        mom=.9*mom+.1*g;v=.999*v+.001*g*g
        w-=.025*(mom/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
        if k%20==0:weights.append(w.copy())
    weights=np.array(weights)
    return weights,np.array([mse(w) for w in weights]),np.array([mse(w,XV,TV) for w in weights])


H=np.array([.35,3.]); WML=np.array([2.5,1.6]);ETA=.25

def early_point(t):
    return (1-(1-ETA*H)**t)*WML

def ridge_point(lam):
    return H/(H+lam)*WML


def rotate(x,angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])@x

def tangent(x):
    return np.array([-x[1],x[0]])

def radial(x,c):
    return np.dot(x,x)+c*x[0]

def sensitivity(x,c):
    return np.dot(2*np.asarray(x)+[c,0],tangent(x))


def noise_example(eps,slope=1.2,residual=.3):
    # x=0, target=-residual, y(x)=slope*x+x², symmetric ±epsilon.
    exact=.25*((residual+slope*eps+eps**2)**2+(residual-slope*eps+eps**2)**2)
    approx=.5*residual**2+.5*eps**2*(slope**2+2*residual)
    return exact,approx


DIGIT=np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],
                [0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]],float)
IMAGE=np.zeros((10,10));IMAGE[1:8,2:7]=DIGIT
KERNEL=np.array([[-1,1,0],[-1,1,0],[-1,1,0]],float)

def convolution(image,kernel=KERNEL):
    # Cross-correlation convention used for learned CNN filters.
    windows=np.lib.stride_tricks.sliding_window_view(image,kernel.shape)
    return np.einsum('ijab,ab->ij',windows,kernel)

def sigmoid(x):return 1/(1+np.exp(-x))

def subsample(feature):
    return sigmoid(feature.reshape(4,2,4,2).mean(axis=(1,3)))


def mixture_components(x,mu=np.array([-1.2,1.2]),sigma=np.array([.6,.6]),pi=np.array([.5,.5])):
    return pi*np.exp(-.5*((np.asarray(x)[...,None]-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)

def responsibilities(x,mu=np.array([-1.2,1.2]),sigma=np.array([.6,.6]),pi=np.array([.5,.5])):
    v=mixture_components(x,mu,sigma,pi);return v/v.sum(axis=-1,keepdims=True)

def soft_gradient(x,mu=np.array([-1.2,1.2]),sigma=np.array([.6,.6]),pi=np.array([.5,.5])):
    return np.sum(responsibilities(x,mu,sigma,pi)*(np.asarray(x)[...,None]-mu)/sigma**2,axis=-1)


@lru_cache(None)
def soft_history():
    # Joint gradient descent on .5*4*||w-target||² + Omega.
    target=np.array([-2.2,-1.6,-.9,-.3,.25,.8,1.5,2.1])
    w=target.copy();mu=np.array([-1.2,1.2]);logvar=np.log([.6**2,.6**2]);logits=np.zeros(2)
    rows=[]
    for k in range(101):
        sigma=np.exp(logvar/2);pi=np.exp(logits-logits.max());pi/=pi.sum()
        rows.append((w.copy(),mu.copy(),sigma.copy(),pi.copy()))
        gamma=responsibilities(w,mu,sigma,pi);delta=w[:,None]-mu
        gw=4*(w-target)+np.sum(gamma*delta/sigma**2,axis=1)
        gm=-np.sum(gamma*delta/sigma**2,axis=0)
        gv=.5*np.sum(gamma*(1-delta**2/sigma**2),axis=0)
        gp=len(w)*pi-gamma.sum(0)
        w-=.008*gw;mu-=.008*gm;logvar-=.008*gv;logits-=.008*gp
    return rows
