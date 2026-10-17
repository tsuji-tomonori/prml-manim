"""Deterministic example and a tested one-dimensional slice sampler."""
import math
import numpy as np
from scipy.optimize import brentq

WIDTH=.5
INITIAL=(-.2,.3)
HEIGHT=.15

def f(z):
    x=np.asarray(z)
    return np.exp(-x*x/(2*.45**2))+.25*np.exp(-x*x/(2*1.8**2))

def boundary(height):
    if not 0<height<1.25: raise ValueError(height)
    return float(brentq(lambda z:f(z)-height,0.,10.))

def expanded_interval(left=INITIAL[0],right=INITIAL[1],height=HEIGHT,width=WIDTH):
    if not left<=0<=right or abs(right-left-width)>1e-12: raise ValueError((left,right,width))
    while f(left)>height: left-=width
    while f(right)>height: right+=width
    return round(left,10),round(right,10)

def fixed_demo():
    left,right=expanded_interval()
    trial1=2.
    if f(trial1)>=HEIGHT:raise RuntimeError('Expected first trial outside slice')
    right=trial1
    trial2=1.4
    if not left<trial2<right or f(trial2)<=HEIGHT:raise RuntimeError('Expected second trial inside slice')
    return dict(start=0.,height=HEIGHT,initial=list(INITIAL),expanded=[left,2.3],
                rejected=trial1,shrunk=[left,right],accepted=trial2)

def slice_step(z,rng,width=WIDTH):
    height=float(rng.uniform(0,f(z)))
    left=float(z-rng.uniform(0,width));right=left+width
    for _ in range(200):
        if f(left)<=height:break
        left-=width
    else:raise RuntimeError('Left step-out did not terminate')
    for _ in range(200):
        if f(right)<=height:break
        right+=width
    else:raise RuntimeError('Right step-out did not terminate')
    for _ in range(1000):
        trial=float(rng.uniform(left,right))
        if f(trial)>=height:return trial
        if trial<z:left=trial
        else:right=trial
    raise RuntimeError('Shrinkage did not terminate')
