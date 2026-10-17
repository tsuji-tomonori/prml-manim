"""Exact binary example and seeded sampling for the PRML 8.1 visuals."""
from itertools import product
import numpy as np

P_RAIN=.3
P_WET=(.1,.9)
P_UMBRELLA=(.2,.8)
SEED=8101

def joint(r,d,u):
    rain=P_RAIN if r else 1-P_RAIN
    wet=P_WET[r] if d else 1-P_WET[r]
    umbrella=P_UMBRELLA[r] if u else 1-P_UMBRELLA[r]
    return rain*wet*umbrella

def exact():
    p_wet=sum(joint(r,1,u) for r,u in product((0,1),repeat=2))
    p_rain_wet=sum(joint(1,1,u) for u in (0,1))
    p_umbrella_wet=sum(joint(r,1,1) for r in (0,1))
    return dict(p_wet=p_wet,p_rain_given_wet=p_rain_wet/p_wet,
                p_umbrella_given_wet=p_umbrella_wet/p_wet,
                rain_wet=p_rain_wet,sun_wet=p_wet-p_rain_wet,
                rain_wet_umbrella=joint(1,1,1),sun_wet_umbrella=joint(0,1,1))

def sample_days(n=1000,seed=SEED):
    rng=np.random.default_rng(seed)
    r=rng.random(n)<P_RAIN
    d=rng.random(n)<np.where(r,P_WET[1],P_WET[0])
    u=rng.random(n)<np.where(r,P_UMBRELLA[1],P_UMBRELLA[0])
    return np.column_stack([r,d,u]).astype(np.int8)

def gaussian_chain(n=2000,seed=8102):
    rng=np.random.default_rng(seed)
    x1=rng.normal(0,1,n)
    x2=.8*x1+rng.normal(0,.6,n)
    x3=.6*x2+rng.normal(0,.5,n)
    return np.column_stack([x1,x2,x3])

RESULT=exact()
DAYS=sample_days()
GAUSSIAN=gaussian_chain()
