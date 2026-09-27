"""Deterministic examples; NumPy calculations, not values copied from PRML."""
import numpy as np
K, B, SCALE = 4.0, .8, 1.1

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -np.asarray(x)))

def logf(z):
    z=np.asarray(z)
    return -.5*(z/SCALE)**2 - np.logaddexp(0, -(K*z+B))

def grad(z):
    return -z/SCALE**2 + K*sigmoid(-(K*z+B))

def precision(z):
    s=sigmoid(K*z+B)
    return 1/SCALE**2 + K*K*s*(1-s)

def mode():
    z=0.
    for _ in range(30):
        z+=grad(z)/precision(z)
    return float(z)

Z0=mode()
A=float(precision(Z0))
F0=float(np.exp(logf(Z0)))
GRID=np.linspace(-12,12,48001)
Z=float(np.trapezoid(np.exp(logf(GRID)),GRID))
ZL=F0*np.sqrt(2*np.pi/A)
MEAN=float(np.trapezoid(GRID*np.exp(logf(GRID))/Z,GRID))

def f(z): return np.exp(logf(z))
def local(z): return F0*np.exp(-.5*A*(np.asarray(z)-Z0)**2)
def pdf(z): return f(z)/Z
def gaussian(z,mu=Z0,a=A):
    return np.sqrt(a/(2*np.pi))*np.exp(-.5*a*(np.asarray(z)-mu)**2)
def bowl(z): return logf(Z0)-logf(z)
def rotation(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s],[s,c]])
def matrix(l1,l2,theta):
    r=rotation(theta)
    return r@np.diag([l1,l2])@r.T

def evidence(width, prior_width):
    # Uniform prior on [-prior_width/2,prior_width/2]. L(0)=1.
    x=np.linspace(-prior_width/2,prior_width/2,20001)
    return float(np.trapezoid(np.exp(-.5*(x/width)**2),x)/prior_width)

def mixture(z, weight=.6):
    return (1-weight)*gaussian(z,-1.5,5)+weight*gaussian(z,1.6,7)

def mixture_laplace(which, weight=.6):
    # Actual numerical mode and log-curvature of the mixture.
    z=-1.5 if which==0 else 1.6
    h=1e-4
    for _ in range(8):
        l,c,r=np.log(mixture(np.array([z-h,z,z+h]),weight))
        g=(r-l)/(2*h); a=-(r-2*c+l)/h**2
        z+=g/a
    return z,a
