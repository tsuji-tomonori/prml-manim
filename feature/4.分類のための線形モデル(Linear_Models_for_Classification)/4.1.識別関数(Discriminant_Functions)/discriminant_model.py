"""Deterministic numerical experiments; all displayed scores derive from these data."""
import numpy as np

def augment(x):
    return np.column_stack([np.ones(len(x)),x])

def unit(x):
    return x/np.linalg.norm(x)

rng=np.random.default_rng(4101)
BASE=[rng.normal([1.1,.65],[.35,.4],(18,2)),rng.normal([-1,-.55],[.35,.4],(18,2))]
MULTI_W=np.array([[0,0,0],[1,-1,0],[.5,.5,-1.]])

def scores(x,W=MULTI_W):
    return augment(np.atleast_2d(x))@W

def clip_halfplane(poly,w,b):
    result=[]
    for a,c in zip(poly,np.roll(poly,-1,axis=0)):
        fa,fc=np.dot(w,a)+b,np.dot(w,c)+b
        if fa>=-1e-10:result.append(a)
        if (fa<0)!=(fc<0):result.append(a+(c-a)*fa/(fa-fc))
    return np.asarray(result)

def region(W,k,bounds=(-3,3,-2,2)):
    x0,x1,y0,y1=bounds
    p=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)
    for j in range(W.shape[1]):
        if j!=k and len(p):p=clip_halfplane(p,W[1:,k]-W[1:,j],W[0,k]-W[0,j])
    return p

def ls_data(amount=0):
    # Additional orange points stay on the correct side of the original boundary.
    extra=np.array([[-1.2,-.7],[-1.4,-.5],[-1.1,-.8]])+amount*np.array([-.65,-3.1])
    return [BASE[0],np.vstack([BASE[1],extra])]

def least_squares(groups,targets=None):
    X=np.vstack(groups)
    if targets is None:
        T=np.repeat(np.eye(len(groups)),[len(g) for g in groups],axis=0)
    else:T=targets
    return np.linalg.lstsq(augment(X),T,rcond=None)[0]

# Symmetric bands: the central class has fewer points, so its linear LS score
# remains 0.2 everywhere. No ties masquerade as a missing class.
OFF=np.array([[x,y] for x in [-.25,.25] for y in [-1.4,-.7,.7,1.4]])
BANDS=[OFF+[-1.8,0],OFF[[0,3,4,7]],OFF+[1.8,0]]
BAND_W=least_squares(BANDS)

rng=np.random.default_rng(4105)
COV=np.array([[1.0,.83],[.83,.85]])
FISH=[rng.multivariate_normal([-1.,0],COV,30),rng.multivariate_normal([1.,.2],COV,22)]
MEANS=np.array([g.mean(0) for g in FISH])
SW=sum((g-m).T@(g-m) for g,m in zip(FISH,MEANS))
DELTA=MEANS[1]-MEANS[0]
FISHER=unit(np.linalg.solve(SW,DELTA))
MEAN_DIR=unit(DELTA)
THETA_MEAN=np.arctan2(MEAN_DIR[1],MEAN_DIR[0])
THETA_FISHER=np.arctan2(FISHER[1],FISHER[0])
FISH_MEAN=np.vstack(FISH).mean(0)
CODE=np.r_[np.full(len(FISH[0]),sum(map(len,FISH))/len(FISH[0])),np.full(len(FISH[1]),-sum(map(len,FISH))/len(FISH[1]))]
CODE_W=least_squares(FISH,CODE)

def direction(theta):return np.array([np.cos(theta),np.sin(theta)])
def criterion(w):return (w@DELTA)**2/(w@SW@w)
def scatter(groups):
    m=np.vstack(groups).mean(0)
    sw=sum((g-g.mean(0)).T@(g-g.mean(0)) for g in groups)
    sb=sum(len(g)*np.outer(g.mean(0)-m,g.mean(0)-m) for g in groups)
    return sw,sb

# Four-dimensional source with exactly centered paired offsets. D > K.
rng=np.random.default_rng(4107)
noise=rng.normal(0,.25,(12,4));noise=np.vstack([noise,-noise])
CENTERS=np.array([[-1.3,-.6,.4,.2],[1.3,-.6,-.4,.2],[0,1.2,0,-.4]])
MULTI=[noise+c for c in CENTERS]
MSW,MSB=scatter(MULTI)
EIG=np.sort(np.linalg.eigvals(np.linalg.solve(MSW,MSB)).real)[::-1]

# A small linearly separable set with both positive and negative corrections.
P_X=np.array([[.3,1.1],[-.6,-.1],[1.4,.4],[-1.3,-.6],[.8,1.2],[-.2,-1.1],[1.3,-.2],[-1.1,.3]])
P_T=np.array([1,-1,1,-1,1,-1,1,-1])
P_INITIAL=np.array([-.1,-.7,.25])

def perceptron(x=P_X,t=P_T,initial=P_INITIAL,passes=100):
    w=initial.copy();history=[]
    for epoch in range(passes):
        updates=0
        for i,phi in enumerate(augment(x)):
            a=w@phi;pred=1 if a>=0 else -1
            if pred!=t[i]:
                after=w+phi*t[i]
                history.append({'index':i,'before':w.copy(),'after':after.copy(),'gain':float(phi@phi)})
                w=after;updates+=1
        if not updates:return history,w
    raise RuntimeError('No convergence within budget')
P_HISTORY,P_FINAL=perceptron()

def interpolate_history(s):
    i=min(int(s),len(P_HISTORY)-1);a=np.clip(s-i,0,1)
    return (1-a)*P_HISTORY[i]['before']+a*P_HISTORY[i]['after']
