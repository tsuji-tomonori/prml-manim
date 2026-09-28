"""Deterministic NumPy MDN experiment; 1 input, 16 tanh units, 3 Gaussian components.

PRML (5.148)-(5.157), with the author's correction for log-sigma derivatives.
The experiment uses sigma=0.012+exp(a_sigma), an explicit numerical safeguard.
"""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'assets' / 'mdn_training.npz'
H = 16
K = 3
FLOOR = .012


def forward_function(u):
    return u + .28 * np.sin(2 * np.pi * u)


def data(seed=5606, n=480):
    rng = np.random.default_rng(seed)
    t = rng.uniform(.02, .98, n)
    x = forward_function(t) + rng.uniform(-.045, .045, n)
    return x, t


def unpack(w):
    a=H
    return w[:a][None,:], w[a:2*a], w[2*a:11*a].reshape(H,9), w[11*a:]


def raw_outputs(w, x):
    W,b,V,c=unpack(w)
    h=np.tanh((2*np.asarray(x).reshape(-1,1)-1) @ W + b)
    return h @ V + c, h


def parameters(w,x):
    a,_=raw_outputs(w,x)
    pi=np.exp(a[:,:3]-logsumexp(a[:,:3],axis=1,keepdims=True))
    return pi,a[:,3:6],FLOOR+np.exp(a[:,6:9])


def normal(t,mu,sigma):
    return np.exp(-.5*((np.asarray(t)-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)


def density(t,pi,mu,sigma):
    return np.sum(pi*normal(np.asarray(t)[...,None],mu,sigma),axis=-1)


def responsibility(t,pi,mu,sigma):
    logp=np.log(pi)-np.log(sigma)-.5*np.log(2*np.pi)-.5*((t-mu)/sigma)**2
    return np.exp(logp-logsumexp(logp))


def loss_grad(w,x,t):
    a,h=raw_outputs(w,x)
    logpi=a[:,:3]-logsumexp(a[:,:3],axis=1,keepdims=True)
    mu=a[:,3:6]; expa=np.exp(a[:,6:]); sigma=FLOOR+expa
    z=(t[:,None]-mu)/sigma
    logparts=logpi-np.log(sigma)-.5*np.log(2*np.pi)-.5*z*z
    logp=logsumexp(logparts,axis=1,keepdims=True)
    gamma=np.exp(logparts-logp)
    da=np.c_[np.exp(logpi)-gamma, gamma*(mu-t[:,None])/sigma**2,
             gamma*(1-z*z)*expa/sigma]/len(x)
    W,b,V,c=unpack(w)
    dh=(da @ V.T)*(1-h*h)
    grad=np.r_[((2*x[:,None]-1).T @ dh).ravel(),dh.sum(0),(h.T @ da).ravel(),da.sum(0)]
    return float(-logp.mean()), grad


def load_model():
    return np.load(MODEL_PATH)


def train():
    x,t=data()
    rng=np.random.default_rng(56)
    w=np.r_[rng.normal(0,1.4,H),rng.normal(0,.4,H),rng.normal(0,.03,H*9),
            [0,0,0,.16,.5,.84], np.log([.14,.14,.14])]
    snapshots=[w.copy()]; epochs=[0]; losses=[loss_grad(w,x,t)[0]]
    m=np.zeros_like(w); v=np.zeros_like(w)
    for step in range(1,5001):
        loss,g=loss_grad(w,x,t)
        g=np.clip(g,-20,20)
        m=.9*m+.1*g; v=.999*v+.001*g*g
        rate=.008 if step<2500 else .003
        w-=rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step % 50 == 0:
            snapshots.append(w.copy()); epochs.append(step); losses.append(loss_grad(w,x,t)[0])
    # A separate tanh least-squares network, fitted to the same inverse data.
    from scipy.optimize import least_squares
    p0=np.r_[rng.normal(0,1.5,H),rng.normal(0,.3,H),rng.normal(0,.1,H),.5]
    def mean_predict(p,z):
        return np.tanh((2*np.asarray(z)[:,None]-1)*p[:H]+p[H:2*H]) @ p[2*H:3*H]+p[-1]
    fit=least_squares(lambda p: mean_predict(p,x)-t,p0,max_nfev=350)
    xt,tt=data(seed=5607,n=1000)
    report=dict(initial_nll=losses[0],final_nll=losses[-1],test_nll=loss_grad(w,xt,tt)[0],
                mean_train_mse=float(np.mean((mean_predict(fit.x,x)-t)**2)),iterations=5000,
                observations=len(x),sigma_floor=FLOOR)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(MODEL_PATH,weights=np.array(snapshots),epochs=epochs,losses=losses,
                        x=x,t=t,mean_weights=fit.x)
    import json
    (ROOT/'training_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


def mean_prediction(p,x):
    return np.tanh((2*np.asarray(x)[:,None]-1)*p[:H]+p[H:2*H]) @ p[2*H:3*H]+p[-1]


def robot(theta1,theta2):
    elbow=np.array([np.cos(theta1),np.sin(theta1)])
    end=elbow+np.array([np.cos(theta1+theta2),np.sin(theta1+theta2)])
    return elbow,end


if __name__=='__main__':
    train()
