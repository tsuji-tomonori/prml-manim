"""Reproducible one-dimensional Hamiltonian example and a 2D Gaussian comparison."""
from __future__ import annotations
import math
import numpy as np

START_Z=-2.0
START_R=0.5
EPSILON=0.15
STEPS=18

def energy(z,r):
    return .5*(float(z)*float(z)+float(r)*float(r))

def leapfrog(z,r,epsilon=EPSILON,steps=STEPS):
    """Leapfrog for E(z)=z²/2; returns every full position/momentum state."""
    states=[(float(z),float(r))]
    for _ in range(steps):
        r-=epsilon*z/2
        z+=epsilon*r
        r-=epsilon*z/2
        states.append((float(z),float(r)))
    return np.array(states)

def acceptance(states):
    return min(1.,math.exp(energy(*states[0])-energy(*states[-1])))

def gaussian_density(z):
    return math.exp(-.5*z*z)/math.sqrt(2*math.pi)

def anisotropic_potential(x,y,sigma_x=2.,sigma_y=.4):
    return .5*((x/sigma_x)**2+(y/sigma_y)**2)


def leapfrog_anisotropic(position=(-2.,-.2), momentum=(1.,.2), epsilon=.15, steps=18, sigmas=(2.,.4)):
    """Leapfrog for an independent 2D Gaussian with standard deviations sigmas."""
    z=np.array(position,dtype=float);r=np.array(momentum,dtype=float)
    scales=np.square(np.asarray(sigmas,dtype=float))
    states=[(z.copy(),r.copy())]
    for _ in range(steps):
        r-=epsilon*z/scales/2
        z+=epsilon*r
        r-=epsilon*z/scales/2
        states.append((z.copy(),r.copy()))
    return states
