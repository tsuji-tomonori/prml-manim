"""Reproducible 2-D variational Gaussian mixture, PRML (10.46)--(10.66)."""
from __future__ import annotations
import numpy as np
from scipy.special import digamma, logsumexp, gammaln, multigammaln

SEED = 1023
K = 6
D = 2
ALPHA0 = .001
BETA0 = .05
NU0 = 3.
M0 = np.zeros(D)
W0 = np.eye(D) * 1.5
INITIAL_CENTERS = np.array([[-2.7,-1.0],[-1.6,-.6],[-.7,.15],
                            [.65,-.25],[1.65,.75],[2.8,1.25]])


def observations():
    rng = np.random.default_rng(SEED)
    left = rng.multivariate_normal([-1.65,-.65], [[.28,.13],[.13,.24]], 40)
    right = rng.multivariate_normal([1.75,.75], [[.31,-.08],[-.08,.22]], 40)
    return np.vstack((left,right))


def initial_responsibilities(x):
    logits = -.5 * np.sum((x[:,None,:]-INITIAL_CENTERS[None,:,:])**2,axis=2)/.40
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def log_b(w,nu):
    """Log Wishart normalizer B(W,nu), PRML (B.79)."""
    return -.5*nu*np.linalg.slogdet(w)[1]-.5*nu*D*np.log(2)-multigammaln(.5*nu,D)


def posterior(x,r,alpha0=ALPHA0):
    nk = r.sum(axis=0)
    beta = BETA0+nk
    nu = NU0+nk
    mean = (BETA0*M0+np.einsum('nk,nd->kd',r,x))/beta[:,None]
    alpha = alpha0+nk
    w = np.empty((K,D,D))
    for k in range(K):
        delta=x-mean[k]
        # Equivalent to Nk*Sk + beta0*Nk/(beta0+Nk)*(xbar-m0)(...)^T.
        scatter=np.einsum('n,ni,nj->ij',r[:,k],delta,delta)
        scatter += BETA0*np.outer(mean[k]-M0,mean[k]-M0)
        w[k]=np.linalg.inv(np.linalg.inv(W0)+scatter)
    return dict(nk=nk,beta=beta,nu=nu,mean=mean,alpha=alpha,w=w)


def expected_logs(p):
    elnpi=digamma(p['alpha'])-digamma(p['alpha'].sum())
    elnlam=np.array([sum(digamma((v+1-i)/2) for i in range(1,D+1))
                      +D*np.log(2)+np.linalg.slogdet(w)[1]
                      for v,w in zip(p['nu'],p['w'])])
    return elnpi,elnlam


def responsibilities(x,p):
    elnpi,elnlam=expected_logs(p)
    distances=np.einsum('nkd,kde,nke->nk',x[:,None,:]-p['mean'][None,:,:],
                        p['w'],x[:,None,:]-p['mean'][None,:,:])
    logits=elnpi[None,:]+.5*elnlam[None,:]-.5*(D*np.log(2*np.pi)
             +D/p['beta'][None,:]+p['nu'][None,:]*distances)
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def lower_bound(x,r,p,alpha0=ALPHA0):
    """Full seven-term bound (10.70), independently usable for monotonicity."""
    nk=p['nk']; beta=p['beta'];nu=p['nu'];w=p['w'];mean=p['mean']
    elnpi,elnlam=expected_logs(p)
    epx=0.
    epprior=0.
    eqparams=0.
    for k in range(K):
        delta=x-mean[k]
        quad=np.einsum('ni,ij,nj->n',delta,w[k],delta)
        epx+=.5*np.dot(r[:,k],elnlam[k]-D*np.log(2*np.pi)-D/beta[k]-nu[k]*quad)
        mdelta=mean[k]-M0
        epprior+=.5*(D*np.log(BETA0/(2*np.pi))+elnlam[k]
                    -D*BETA0/beta[k]-BETA0*nu[k]*(mdelta@w[k]@mdelta))
        epprior+=log_b(W0,NU0)+.5*(NU0-D-1)*elnlam[k]
        epprior-=.5*nu[k]*np.trace(np.linalg.solve(W0,w[k]))
        h_w=-log_b(w[k],nu[k])-.5*(nu[k]-D-1)*elnlam[k]+.5*nu[k]*D
        eqparams+=.5*elnlam[k]+.5*D*np.log(beta[k]/(2*np.pi))-.5*D-h_w
    epz=np.sum(r*elnpi[None,:])
    eppi=gammaln(K*alpha0)-K*gammaln(alpha0)+(alpha0-1)*elnpi.sum()
    eqz=np.sum(np.where(r>0,r*np.log(np.maximum(r,1e-300)),0))
    a=p['alpha']
    eqpi=gammaln(a.sum())-gammaln(a).sum()+np.dot(a-1,elnpi)
    return float(epx+epz+eppi+epprior-eqz-eqpi-eqparams)


def fit(alpha0=ALPHA0,steps=130):
    x=observations();r=initial_responsibilities(x)
    history=[]
    for t in range(steps+1):
        p=posterior(x,r,alpha0)
        history.append(dict(iteration=t,r=r.copy(),mean=p['mean'].copy(),
                            nk=p['nk'].copy(),weights=(p['alpha']/p['alpha'].sum()).copy(),
                            covariance=np.array([np.linalg.inv(w)/max(v-D-1,1)
                                                 for w,v in zip(p['w'],p['nu'])]),
                            bound=lower_bound(x,r,p,alpha0)))
        r=responsibilities(x,p)
    return x,history


if __name__=='__main__':
    for a in (ALPHA0,.1,1.,10.):
        x,h=fit(a)
        print('alpha',a,'bound',round(h[-1]['bound'],3),'weights',np.round(h[-1]['weights'],4),
              'nk',np.round(h[-1]['nk'],2),'min bound step',np.min(np.diff([s['bound'] for s in h])))
