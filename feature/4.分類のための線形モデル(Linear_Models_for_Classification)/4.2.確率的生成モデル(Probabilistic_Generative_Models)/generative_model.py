"""Reproducible numerical experiments for PRML 4.2 (no textbook data)."""
import numpy as np

MEANS = np.array([[-1.15, -.45], [1.15, -.45], [0., 1.25]])
COV = np.array([[.52, .16], [.16, .42]])
OTHER_COV = np.array([[1.1, -.27], [-.27, .30]])
RNG = np.random.default_rng(4202)
RED_POINTS = RNG.multivariate_normal(MEANS[0], COV, 30)
BLUE_POINTS = RNG.multivariate_normal(MEANS[1], COV, 20)
BINARY_MU = np.array([[.8, .65, .25], [.2, .35, .7]])


def sigmoid(a):
    a = np.asarray(a)
    return np.exp(-np.logaddexp(0, -a))


def softmax(scores):
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def normal1(x, mean, sd=.9):
    return np.exp(-.5*((np.asarray(x)-mean)/sd)**2) / (sd*np.sqrt(2*np.pi))


def log_gaussian(x, mean, cov):
    d = np.asarray(x)-mean
    return -.5*(len(mean)*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+np.einsum('...i,ij,...j->...', d, np.linalg.inv(cov), d))


def scores(x, means=MEANS[:2], covs=None, priors=None):
    k = len(means)
    covs = [COV]*k if covs is None else covs
    priors = np.full(k, 1/k) if priors is None else np.asarray(priors)
    return np.stack([log_gaussian(x,m,c)+np.log(p) for m,c,p in zip(means,covs,priors)], axis=-1)


def posterior(x, means=MEANS[:2], covs=None, priors=None):
    return softmax(scores(x,means,covs,priors))


def linear_params(means=MEANS[:2], cov=COV, prior=.5):
    w = np.linalg.solve(cov, means[0]-means[1])
    b = -.5*(means[0] @ np.linalg.solve(cov,means[0])-means[1] @ np.linalg.solve(cov,means[1])) + np.log(prior/(1-prior))
    return w, b


def covariances(t):
    return [COV, COV, (1-t)*COV+t*OTHER_COV]


def fit(red=RED_POINTS, blue=BLUE_POINTS):
    means = np.array([red.mean(0), blue.mean(0)])
    residuals = [red-means[0], blue-means[1]]
    cov = sum(r.T @ r for r in residuals)/(len(red)+len(blue))
    return means, cov, len(red)/(len(red)+len(blue))


def contaminated(t):
    red = RED_POINTS.copy()
    red[0] = (1-t)*red[0] + t*np.array([4.3,3.2])
    return red


def binary_scores(x):
    return (np.asarray(x)*np.log(BINARY_MU)+(1-np.asarray(x))*np.log1p(-BINARY_MU)).sum(axis=-1)+np.log(.5)


def zero_contours(xs, ys, z):
    """NumPy marching squares, joined along shared grid edges (no extra dependency)."""
    h = z[:, :-1]*z[:, 1:] < 0
    v = z[:-1, :]*z[1:, :] < 0
    # A tiny deterministic perturbation avoids ambiguous exact grid vertices.
    if np.any(z == 0):
        return zero_contours(xs, ys, np.where(z == 0, 1e-12, z))
    hi=np.full(h.shape,-1,int);vi=np.full(v.shape,-1,int)
    hi[h]=np.arange(h.sum());vi[v]=np.arange(v.sum())+h.sum()
    hy,hx=np.nonzero(h);vy,vx=np.nonzero(v)
    ht=z[hy,hx]/(z[hy,hx]-z[hy,hx+1]);vt=z[vy,vx]/(z[vy,vx]-z[vy+1,vx])
    points=np.r_[np.c_[xs[hx]+ht*(xs[hx+1]-xs[hx]),ys[hy]],np.c_[xs[vx],ys[vy]+vt*(ys[vy+1]-ys[vy])]]
    cells=(hi[:-1]>=0)|(hi[1:]>=0)|(vi[:,:-1]>=0)|(vi[:,1:]>=0)
    edges=[]
    for y,x in zip(*np.nonzero(cells)):
        nodes=[hi[y,x],vi[y,x+1],hi[y+1,x],vi[y,x]]
        found=[n for n in nodes if n>=0]
        if len(found)==2:edges.append(tuple(found))
        elif len(found)==4:
            # Resolve saddle cells using the bilinear center sign.
            if np.sign(z[y:y+2,x:x+2].mean())==np.sign(z[y,x]):
                edges.extend([(nodes[0],nodes[1]),(nodes[2],nodes[3])])
            else:edges.extend([(nodes[0],nodes[3]),(nodes[1],nodes[2])])
    neighbors={}
    for a,b in edges:
        neighbors.setdefault(a,set()).add(b);neighbors.setdefault(b,set()).add(a)
    lines=[]
    while neighbors:
        start=next((i for i,ns in neighbors.items() if len(ns)==1),next(iter(neighbors)))
        chain=[start];current=start
        while current in neighbors and neighbors[current]:
            nxt=neighbors[current].pop();neighbors[nxt].remove(current)
            if not neighbors[current]:del neighbors[current]
            chain.append(nxt);current=nxt
        if current in neighbors:del neighbors[current]
        if len(chain)>1:lines.append(points[chain])
    return lines


def decision_paths(means=MEANS[:2], covs=None, priors=None, xr=(-3.5,3.5), yr=(-2.5,2.5), resolution=100):
    """Pair equality contours restricted to the maximal two scores, inside viewport."""
    xs=np.linspace(*xr,resolution); ys=np.linspace(*yr,resolution)
    xx,yy=np.meshgrid(xs,ys); grid=np.stack([xx,yy],axis=-1)
    values=scores(grid,means,covs,priors); paths=[]
    for i in range(len(means)):
        for j in range(i+1,len(means)):
            for line in zero_contours(xs,ys,values[...,i]-values[...,j]):
                mid=(line[:-1]+line[1:])/2
                v=scores(mid,means,covs,priors)
                keep=np.minimum(v[:,i],v[:,j])>=v.max(axis=-1)-.003
                # Split at hidden segments; no equality extensions through a third class.
                run=[]
                for n,visible in enumerate(keep):
                    if visible: run.append(line[n])
                    if run and (not visible or n==len(keep)-1):
                        run.append(line[n+1] if visible else line[n]);paths.append(np.array(run));run=[]
    return paths
