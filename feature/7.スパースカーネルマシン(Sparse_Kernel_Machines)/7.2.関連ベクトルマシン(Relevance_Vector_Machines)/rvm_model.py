"""Deterministic 11-point regression RVM demonstration with fixed noise precision."""
from __future__ import annotations
import numpy as np

X=np.linspace(0.,1.,11)
NOISE=np.array([.01,-.02,.015,0.,-.01,.01,-.015,.005,.01,-.01,.015])
WIDTH=.3
BETA=40.
T=1.1*np.exp(-.5*((X-.25)/WIDTH)**2)-np.exp(-.5*((X-.75)/WIDTH)**2)+NOISE
CENTERS=X.copy()

def basis(z):
    z=np.asarray(z,dtype=float).reshape(-1)
    return np.column_stack([np.ones(len(z)),np.exp(-.5*((z[:,None]-CENTERS[None,:])/WIDTH)**2)])
PHI=basis(X)

def covariance(alpha):
    live=np.isfinite(alpha)
    return np.eye(len(X))/BETA+(PHI[:,live]/alpha[live])@PHI[:,live].T

def evidence(alpha):
    c=covariance(alpha)
    sign,logdet=np.linalg.slogdet(c)
    assert sign>0
    return float(-.5*(len(X)*np.log(2*np.pi)+logdet+T@np.linalg.solve(c,T)))

def one_basis_terms(alpha,index):
    others=alpha.copy();others[index]=np.inf
    c=covariance(others)
    v=PHI[:,index]
    return float(v@np.linalg.solve(c,v)),float(v@np.linalg.solve(c,T))

def optimum_alpha(s,q):
    return s*s/(q*q-s) if q*q>s else np.inf

def fit():
    alpha=np.full(PHI.shape[1],np.inf)
    history=[]
    for sweep in range(80):
        old=evidence(alpha)
        for i in range(len(alpha)):
            s,q=one_basis_terms(alpha,i)
            alpha[i]=optimum_alpha(s,q)
        now=evidence(alpha)
        assert now>=old-1e-10,(sweep,old,now)
        history.append(dict(sweep=sweep+1,evidence=now,active=np.flatnonzero(np.isfinite(alpha)).tolist()))
        if now-old<1e-10:break
    live=np.isfinite(alpha)
    p=PHI[:,live]
    sigma=np.linalg.inv(np.diag(alpha[live])+BETA*p.T@p)
    mean=BETA*sigma@p.T@T
    return dict(alpha=alpha,active=np.flatnonzero(live),sigma=sigma,mean=mean,history=history)

MODEL=fit()

def predict(z,model=MODEL):
    one=np.isscalar(z)
    a=basis([z] if one else z)[:,model['active']]
    mean=a@model['mean']
    weight_var=np.einsum('ni,ij,nj->n',a,model['sigma'],a)
    total_var=1/BETA+weight_var
    if one:return float(mean[0]),float(total_var[0]),float(weight_var[0])
    return mean,total_var,weight_var
