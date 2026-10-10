"""Numerical experiments for PRML 3.5. All plotted quantities are computed here."""
import numpy as np
from functools import lru_cache

X = np.linspace(0, 1, 18)
SIGMA = .25
T = np.sin(2*np.pi*X) + np.random.default_rng(35).normal(0, SIGMA, len(X))
XT = np.linspace(0, 1, 200)
TT = np.sin(2*np.pi*XT) + np.random.default_rng(351).normal(0, SIGMA, len(XT))
BETA = 1/SIGMA**2

def design(x):
    x=np.atleast_1d(x)
    return np.column_stack([np.ones(len(x)),np.exp(-.5*((x[:,None]-np.linspace(0,1,9))/.13)**2)])
PHI=design(X)
EIGS=np.maximum(np.linalg.eigvalsh(PHI.T@PHI),0)
U=np.linspace(0,1,181)
PU=design(U)

def stats(alpha,beta,phi=PHI,t=T):
    n,m=phi.shape
    A=alpha*np.eye(m)+beta*phi.T@phi
    mean=np.linalg.solve(A,beta*phi.T@t)
    cov=np.linalg.solve(A,np.eye(m))
    rss=float(np.sum((t-phi@mean)**2)); norm=float(mean@mean)
    eig=np.maximum(np.linalg.eigvalsh(phi.T@phi)*beta,0)
    fractions=eig/(alpha+eig); gamma=float(fractions.sum())
    terms=np.array([m/2*np.log(alpha),n/2*np.log(beta),-beta*rss/2,-alpha*norm/2,-np.linalg.slogdet(A)[1]/2,-n/2*np.log(2*np.pi)])
    return dict(alpha=alpha,beta=beta,A=A,mean=mean,cov=cov,rss=rss,norm=norm,gamma=gamma,fractions=fractions,terms=terms,logev=float(terms.sum()))

@lru_cache(maxsize=2048)
def at(logalpha,beta=BETA):
    return stats(float(np.exp(logalpha)),float(beta))

GRID=np.linspace(-6,6,241)
EV=np.array([at(float(z))['logev'] for z in GRID])
# Bisection of the exact derivative d log evidence / d log alpha.
a,b=-6.,6.
for _ in range(60):
    c=(a+b)/2;s=at(c)
    if s['gamma']-s['alpha']*s['norm']>0:a=c
    else:b=c
OPT=(a+b)/2
TEST=np.array([np.sqrt(np.mean((design(XT)@at(float(z))['mean']-TT)**2)) for z in GRID])

def iterations(alpha=20.,beta=2.,steps=30):
    result=[]
    for _ in range(steps+1):
        s=stats(alpha,beta);result.append(s)
        # Both updates use the SAME old posterior and gamma, equations 3.92/3.95.
        alpha=s['gamma']/s['norm']; beta=(len(T)-s['gamma'])/s['rss']
    return result
ITER=iterations()

POLY_ALPHA=.005
POLY=[stats(POLY_ALPHA,BETA,np.vander(2*X-1,d+1,increasing=True)) for d in range(10)]
POLY_CURVES=[np.vander(2*U-1,d+1,increasing=True)@s['mean'] for d,s in enumerate(POLY)]

def normal(x,mean,variance):
    return np.exp(-.5*(np.asarray(x)-mean)**2/variance)/np.sqrt(2*np.pi*variance)

def scalar(logalpha):
    alpha=np.exp(logalpha)
    grid=np.linspace(-10,10,16001)
    prior=normal(grid,0,1/alpha);likelihood=normal(1.2,grid,.35**2)
    return float(np.trapezoid(prior*likelihood,grid)),float(normal(1.2,0,1/alpha+.35**2))

# The unseen input used at both ends of the story. The target is the noiseless
# generating wave, held out from evidence selection and fitting.
HIDDEN_X = 0.75
HIDDEN_TRUE = float(np.sin(2*np.pi*HIDDEN_X))
def hidden_prediction(logalpha):
    return float(design([HIDDEN_X])[0] @ at(float(logalpha))['mean'])
HIDDEN_WEAK = hidden_prediction(-6)
HIDDEN_STRONG = hidden_prediction(6)
HIDDEN_SELECTED = hidden_prediction(OPT)
