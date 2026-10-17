"""Reproducible 11.2 walk on the 11.1 educational mixture."""
from functools import lru_cache
import numpy as np
from scipy.stats import norm

THRESHOLD=1.0
SEED=112
START=-1.0
STEPS=12000
BURN=2000
SCALES=(.35,.8,2.2)

def target(z):
    z=np.asarray(z)
    return .82*norm.pdf(z,-1.,.45)+.18*norm.pdf(z,2.,.35)

def truth():
    return float(.82*norm.sf(1.,-1.,.45)+.18*norm.sf(1.,2.,.35))

@lru_cache(maxsize=None)
def walk(scale):
    rng=np.random.default_rng(SEED)
    states=np.empty(STEPS+1);states[0]=START
    candidates=np.empty(STEPS); uniforms=np.empty(STEPS);accepted=np.empty(STEPS,dtype=bool)
    for i in range(STEPS):
        old=states[i]
        candidate=old+rng.normal(0,scale)
        u=rng.random()
        candidates[i]=candidate;uniforms[i]=u
        accepted[i]=u<min(1.,float(target(candidate)/target(old)))
        states[i+1]=candidate if accepted[i] else old
    return states,candidates,uniforms,accepted

def summary(scale):
    states,candidates,uniforms,accepted=walk(scale)
    indicator=(states[BURN+1:]>THRESHOLD).astype(float)
    first=np.flatnonzero(states[1:]>THRESHOLD)
    lag=float(np.corrcoef(indicator[:-1],indicator[1:])[0,1]) if indicator.std()>0 else 1.
    return dict(scale=scale, first_right_step=int(first[0]+1) if len(first) else None,
                first_24_right=int(np.count_nonzero(states[1:25]>THRESHOLD)),
                acceptance_rate=float(accepted.mean()), right_estimate=float(indicator.mean()),
                right_lag1_correlation=lag)

def mh_ratio(old,candidate,shift=.4,scale=1.1):
    # q(candidate|old)=N(candidate|old+shift,scale^2).
    forward=norm.pdf(candidate,old+shift,scale)
    reverse=norm.pdf(old,candidate+shift,scale)
    return float(target(candidate)*reverse/(target(old)*forward))
