"""Independent NumPy experiments for PRML 5.4; no autodiff dependency."""
import numpy as np

X=.8
TARGET=.25
WEIGHTS=np.array([.7,1.2])
B=np.array([[1.4,.3],[.2,1.3],[1.,-.8]])
ALPHA=.3

def net(w=WEIGHTS,x=X,t=TARGET):
    u,v=np.asarray(w); a=u*x; z=np.tanh(a); hp=1-z*z; hpp=-2*z*hp
    y=v*z; r=y-t
    b=np.array([v*hp*x,z])
    d2y=np.array([[v*hpp*x*x,hp*x],[hp*x,0.]])
    outer=np.outer(b,b); residual=r*d2y
    return dict(a=a,z=z,hp=hp,hpp=hpp,y=y,r=r,b=b,E=.5*r*r,g=r*b,outer=outer,residual=residual,H=outer+residual)

def hvp(w,direction,x=X,t=TARGET):
    """R-forward/backward, (5.98)-(5.111); never constructs a Hessian."""
    u,v=np.asarray(w); du,dv=np.asarray(direction)
    a=u*x; z=np.tanh(a); hp=1-z*z; hpp=-2*z*hp; y=v*z
    Ra=du*x; Rz=hp*Ra; Ry=dv*z+v*Rz
    delta=y-t; hidden=hp*v*delta
    Rdelta=Ry; Rhidden=hpp*Ra*v*delta+hp*dv*delta+hp*v*Rdelta
    grad=np.array([hidden*x,delta*z])
    result=np.array([x*Rhidden,Rdelta*z+delta*Rz])
    return result,dict(a=a,z=z,y=y,Ra=Ra,Rz=Rz,Ry=Ry,delta=delta,hidden=hidden,Rdelta=Rdelta,Rhidden=Rhidden,g=grad)

def finite_hessian(w,eps=1e-4):
    w=np.asarray(w); H=np.zeros((2,2)); eye=np.eye(2)
    for i in range(2):
        for j in range(2):
            if i==j:
                H[i,i]=(net(w+eps*eye[i])['E']-2*net(w)['E']+net(w-eps*eye[i])['E'])/eps**2
            else:
                H[i,j]=sum(s*t*net(w+eps*(s*eye[i]+t*eye[j]))['E'] for s in [-1,1] for t in [-1,1])/(4*eps**2)
    return H

def gradient_difference(w,eps=1e-5):
    return np.column_stack([(net(w+eps*e)['g']-net(w-eps*e)['g'])/(2*eps) for e in np.eye(2)])

def precision(n,alpha=ALPHA):
    weights=np.clip(n-np.arange(len(B)),0,1)
    return alpha*np.eye(2)+np.einsum('n,ni,nj->ij',weights,B,B)

def inverse_update(n,alpha=ALPHA):
    C=np.eye(2)/alpha
    for weight,b in zip(np.clip(n-np.arange(len(B)),0,1),B):
        q=np.sqrt(weight)*b; v=C@q
        C-=np.outer(v,v)/(1+q@v)
    return C

def ellipse_points(H,level=1,count=181):
    vals,vecs=np.linalg.eigh(H)
    angle=np.linspace(0,2*np.pi,count)
    return (vecs@((np.sqrt(level/vals))[:,None]*np.array([np.cos(angle),np.sin(angle)]))).T
