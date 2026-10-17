"""Numerical example for PRML 6.2. All displayed values derive from these functions."""
import numpy as np
X=np.array([-1.,0.,1.]); T=np.array([1.,0.,1.]); PROBE=.5; LAMBDA=.5; SIGMA=1.

def tempting(x,z):
    x,z=np.asarray(x,dtype=float),np.asarray(z,dtype=float)
    return 1-(x-z)**2/4

def polynomial(x,z):
    x,z=np.asarray(x,dtype=float),np.asarray(z,dtype=float)
    return (1+x*z)**2

def feature(x):
    x=np.asarray(x,dtype=float)
    return np.stack((np.ones_like(x),np.sqrt(2)*x,x*x),axis=-1)

def gaussian(x,z,sigma=SIGMA):
    x,z=np.asarray(x,dtype=float),np.asarray(z,dtype=float)
    return np.exp(-(x-z)**2/(2*sigma*sigma))

def old_kernel(x,z):
    x,z=np.asarray(x,dtype=float),np.asarray(z,dtype=float)
    return 1+x*z+x*x*z*z

def gram(k,points=X):
    return k(points[:,None],points[None,:])

def coefficients(k,lam=LAMBDA):
    return np.linalg.solve(gram(k)+lam*np.eye(len(X)),T)

def prediction(k,x=PROBE,lam=LAMBDA):
    return float(k(X,x)@coefficients(k,lam))

K_BAD=gram(tempting); C_BAD=np.array([1.,-1.5,1.]); BAD_QUADRATIC=float(C_BAD@K_BAD@C_BAD)
K_POLY=gram(polynomial); A_POLY=coefficients(polynomial); Y_POLY=prediction(polynomial)
K_GAUSS=gram(gaussian); A_GAUSS=coefficients(gaussian); Y_GAUSS=prediction(gaussian)
Y_OLD=prediction(old_kernel)
