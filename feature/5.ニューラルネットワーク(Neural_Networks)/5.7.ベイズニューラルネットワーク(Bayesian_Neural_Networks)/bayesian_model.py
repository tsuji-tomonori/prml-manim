"""Small, fully trainable tanh networks; NumPy derivatives and Laplace calculations.

Regression: 1-2-1 (7 weights including biases), classification: 2-4-1 (17).
All displayed curves/densities/contours come from these computations.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import expit

ROOT=Path(__file__).resolve().parent
X=np.array([-1.7,-1.4,-1.1,-.75,-.4,-.1,.2,.45,.8,1.1,1.4,1.7])[:,None]
T=np.sin(1.4*X[:,0]) + np.array([.09,-.08,.12,-.09,.05,-.12,.08,-.04,.10,-.1,.07,-.03])
ALPHA=.8
BETA=64.
GRID=np.linspace(-2.6,2.6,181)

def unpack(w,d):
    m=(len(w)-1)//(d+2)
    return w[:m*d].reshape(m,d),w[m*d:m*(d+1)],w[m*(d+1):-1],w[-1]

def output(w,x):
    x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
    u,b,v,c=unpack(w,x.shape[1]);h=np.tanh(x@u.T+b)
    return h@v+c

def jacobian(w,x):
    x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
    u,b,v,c=unpack(w,x.shape[1]);h=np.tanh(x@u.T+b);q=(1-h*h)*v
    return np.concatenate([(q[:,:,None]*x[:,None,:]).reshape(len(x),-1),q,h,np.ones((len(x),1))],axis=1)

def objective(w,x,t,alpha,beta=1.,classification=False):
    y=output(w,x);j=jacobian(w,x)
    if classification:
        return np.sum(np.logaddexp(0,y)-t*y)+alpha*(w@w)/2,j.T@(expit(y)-t)+alpha*w
    r=y-t
    return beta*(r@r)/2+alpha*(w@w)/2,beta*j.T@r+alpha*w

def hessian(w,x,t,alpha,beta=1.,classification=False):
    # Central differentiation of the exact gradient includes residual Hessians.
    step=1e-4
    eye=np.eye(len(w))*step
    a=np.column_stack([(objective(w+v,x,t,alpha,beta,classification)[1]-objective(w-v,x,t,alpha,beta,classification)[1])/(2*step) for v in eye])
    return (a+a.T)/2

def fit(x,t,alpha,beta=1.,classification=False,m=2,initial=None):
    size=m*(x.shape[1]+2)+1
    starts=[initial] if initial is not None else [np.random.default_rng(i+57).normal(0,.7,size) for i in range(6)]
    best=None
    for start in starts:
        if classification:
            result=minimize(lambda w:objective(w,x,t,alpha,beta,True),start,jac=True,method='BFGS',options=dict(gtol=1e-8,maxiter=1400))
            w=result.x
        else:
            result=least_squares(lambda w:np.r_[np.sqrt(beta)*(output(w,x)-t),np.sqrt(alpha)*w], start,
                jac=lambda w:np.vstack([np.sqrt(beta)*jacobian(w,x),np.sqrt(alpha)*np.eye(size)]),gtol=1e-11,ftol=1e-11,xtol=1e-11,max_nfev=2500)
            w=result.x
        cost=objective(w,x,t,alpha,beta,classification)[0]
        if best is None or cost<best[0]:best=(cost,w)
    w=best[1];a=hessian(w,x,t,alpha,beta,classification)
    eig=np.linalg.eigvalsh(a)
    if eig.min()<=0:raise ValueError(f'Non-positive curvature: {eig.min()}')
    cov=np.linalg.inv(a)
    evidence=-best[0]-.5*np.linalg.slogdet(a)[1]+len(w)/2*np.log(alpha)
    if not classification:evidence+=len(t)/2*np.log(beta/(2*np.pi))
    return dict(w=w,A=a,cov=cov,alpha=alpha,beta=beta,evidence=evidence,
                gamma=float(np.trace((a-alpha*np.eye(len(w)))@cov)),grad=float(np.linalg.norm(objective(w,x,t,alpha,beta,classification)[1])))

def variance(w,cov,x):
    j=jacobian(w,x)
    return np.einsum('ij,jk,ik->i',j,cov,j)

def gaussian(x,mu,variance):return np.exp(-.5*(np.asarray(x)-mu)**2/variance)/np.sqrt(2*np.pi*variance)
def kappa(v):return 1/np.sqrt(1+np.pi*np.asarray(v)/8)
def integrated_sigmoid(mu,var):
    z,h=np.polynomial.hermite.hermgauss(80)
    return np.sum(h*expit(mu+np.sqrt(2*var)*z))/np.sqrt(np.pi)

def evidence_example(sd):
    # t=1.2, Gaussian likelihood sd=.35; normalized N(w|0,sd^2).
    return float(gaussian(1.2,0,sd*sd+.35**2))

def build():
    r=fit(X,T,ALPHA,BETA)
    history=[r]
    for _ in range(4):
        old=history[-1];w=old['w'];gam=old['gamma']
        alpha=gam/(w@w);beta=(len(T)-gam)/np.sum((output(w,X)-T)**2)
        history.append(fit(X,T,alpha,beta,initial=w))
    rng=np.random.default_rng(571)
    cx=rng.uniform(-2,2,(50,2))
    true=2.8*(cx[:,0]-.6*np.sin(1.5*cx[:,1]))
    ct=(rng.random(len(cx))<expit(true)).astype(float)
    candidates=[fit(cx,ct,a,classification=True,m=4) for a in [.015,.06,.2,.6,1.5]]
    best=max(candidates,key=lambda z:z['evidence'])
    z=rng.normal(size=(9,len(r['w'])))
    samples=r['w']+z@np.linalg.cholesky(r['cov']).T
    np.savez(ROOT/'model_data.npz',w=r['w'],A=r['A'],cov=r['cov'],samples=samples,
       history_w=np.array([h['w'] for h in history]),history_alpha=[h['alpha'] for h in history],
       history_beta=[h['beta'] for h in history],history_gamma=[h['gamma'] for h in history],
       cx=cx,ct=ct,cw=best['w'],cA=best['A'],ccov=best['cov'],calpha=best['alpha'],weak=candidates[0]['w'],
       class_alphas=[h['alpha'] for h in candidates],class_evidence=[h['evidence'] for h in candidates],
       class_grad=[h['grad'] for h in candidates])
    print('regression', {k:v for k,v in r.items() if np.isscalar(v)})
    print('updates',[(h['alpha'],h['beta'],h['gamma']) for h in history])
    print('class',[(h['alpha'],h['evidence'],h['grad']) for h in candidates])

if __name__=='__main__':build()
