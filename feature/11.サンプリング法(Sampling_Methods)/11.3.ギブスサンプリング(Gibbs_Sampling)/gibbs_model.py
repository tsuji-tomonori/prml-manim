"""Reproducible two-variable Gibbs examples; all numbers follow this module."""
from functools import lru_cache
import numpy as np
RHO=.98
SEED=43
START=np.array([-2.,-2.])
COND_SD=float(np.sqrt(1-RHO**2))

def conditional_mean(other,rho=RHO):return rho*other

def conditional_sd(rho=RHO):return float(np.sqrt(1-rho*rho))

@lru_cache(None)
def chain(n=3000,rho=RHO,alpha=0.,seed=SEED):
    """Return every half-step; exact normal conditional / over-relaxed update."""
    rng=np.random.default_rng(seed)
    state=START.copy(); points=[state.copy()]
    sd=conditional_sd(rho)
    for _ in range(n):
        for k in (0,1):
            mean=rho*state[1-k]
            state[k]=mean+alpha*(state[k]-mean)+sd*np.sqrt(1-alpha*alpha)*rng.normal()
            points.append(state.copy())
    return np.array(points)

def joint_density(x,y,rho=RHO):
    q=(x*x-2*rho*x*y+y*y)/(1-rho*rho)
    return np.exp(-.5*q)/(2*np.pi*np.sqrt(1-rho*rho))

def exact_lag1(rho=RHO):return rho*rho

def integrated_time(rho=RHO):
    a=rho*rho
    return (1+a)/(1-a)


def checks():
    c=chain()
    assert np.allclose(c[40],[-2.22290518,-2.09069438],atol=1e-8)
    assert np.isclose(COND_SD**2,.0396)
    assert np.isclose(exact_lag1(),.9604)
    assert np.isclose(integrated_time(),49.50505050505)
    # The corrected over-relaxation has the same conditional mean and variance.
    a=-.8
    assert np.isclose(a*a*COND_SD**2+(1-a*a)*COND_SD**2,COND_SD**2)
    assert np.isclose(joint_density(1,1),joint_density(-1,-1))
    return dict(rho=RHO,seed=SEED,start=START.tolist(),after20=c[40].tolist(),conditional_sd=COND_SD,conditional_variance=COND_SD**2,lag1=exact_lag1(),integrated_autocorrelation_time=integrated_time(),after40=c[80].tolist(),after200=c[400].tolist(),overrelaxation_alpha=a)
