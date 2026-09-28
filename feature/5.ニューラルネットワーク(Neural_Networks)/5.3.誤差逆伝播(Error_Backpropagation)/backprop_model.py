"""Independent NumPy calculations for all visual experiments (bias column first)."""
import numpy as np
W1=np.array([[.1,.7,-.4],[-.2,-.3,.8]])
W2=np.array([[.05,1.1,-.6],[-.1,.4,.9]])
X=np.array([.6,-.4]); T=np.array([.3,-.2])
DATA=[(X,T),(np.array([-.5,.7]),np.array([-.2,.5])),(np.array([.5,.6]),np.array([-.5,.2]))]

def forward(x=X,w1=W1,w2=W2):
    xb=np.r_[1.,x];a=w1@xb;z=np.tanh(a);zb=np.r_[1.,z];y=w2@zb
    return dict(x=xb,a=a,z=z,zb=zb,y=y)
def loss(x=X,t=T,w1=W1,w2=W2):
    r=forward(x,w1,w2)['y']-t
    return float(.5*r@r)
def backward(x=X,t=T,w1=W1,w2=W2):
    f=forward(x,w1,w2);d2=f['y']-t;d1=(1-f['z']**2)*(w2[:,1:].T@d2)
    return dict(**f,d2=d2,d1=d1,g1=np.outer(d1,f['x']),g2=np.outer(d2,f['zb']))
def jacobian(x=X,w1=W1,w2=W2):
    f=forward(x,w1,w2)
    return (w2[:,1:]*(1-f['z']**2))@w1[:,1:]
def scalar(w,x=.8,t=.7):
    a=w*x+.1;z=np.tanh(a);y=1.2*z-.1;E=.5*(y-t)**2
    delta=(y-t)*1.2*(1-z*z)
    return dict(a=a,z=z,y=y,E=E,delta=delta,g=delta*x)
def central(w,eps):
    return (scalar(w+eps)['E']-scalar(w-eps)['E'])/(2*eps)
def total(w1=W1,w2=W2):
    return sum(loss(x,t,w1,w2) for x,t in DATA)
def batch():
    b=[backward(x,t) for x,t in DATA]
    return sum(v['g1'] for v in b),sum(v['g2'] for v in b)
def softmax(a):
    v=np.exp(a-np.max(a));return v/v.sum()
