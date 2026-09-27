"""Reproducible numerical experiments for PRML 2.5; no density clipping."""
import numpy as np

SEED = 2509
N = 60
rng = np.random.default_rng(SEED)
components = rng.random(N) >= .5
SAMPLES = np.sort(rng.normal(np.where(components, .71, .29), np.where(components, .075, .055)))
# All order-statistic breakpoints are sample locations or pairwise midpoints.
GRID = np.unique(np.r_[np.linspace(0, 1, 601), SAMPLES,
                      ((SAMPLES[:, None]+SAMPLES[None, :])/2).ravel()])

def gaussian(x, mean, sd):
    return np.exp(-.5*((np.asarray(x)-mean)/sd)**2)/(sd*np.sqrt(2*np.pi))

def truth(x):
    return .5*gaussian(x, .29, .055)+.5*gaussian(x, .71, .075)

def fitted_gaussian(x):
    return gaussian(x, SAMPLES.mean(), SAMPLES.std())

def kde(x, h, samples=SAMPLES):
    return gaussian(np.asarray(x)[..., None], samples, h).mean(axis=-1)

def histogram(width, offset=0):
    # Fixed support for plotting, with smaller boundary bins explicitly normalized.
    inner=np.arange(offset-width, 1+width, width)
    edges=np.unique(np.r_[0, inner[(inner>1e-9)&(inner<1-1e-9)], 1])
    counts,_=np.histogram(SAMPLES,edges)
    return counts/(N*np.diff(edges)), edges, counts

def local_count(x, width):
    return int(np.count_nonzero(np.abs(SAMPLES-x)<=width/2+1e-12))

def box_kde(x,h):
    return (np.abs(np.asarray(x)[...,None]-SAMPLES)<=h/2).sum(axis=-1)/(N*h)

def knn_radius(x,k):
    return np.partition(np.abs(np.asarray(x)[...,None]-SAMPLES),k-1,axis=-1)[...,k-1]

def knn_density(x,k):
    with np.errstate(divide='ignore'):
        return k/(N*2*knn_radius(x,k))

# Hand-designed teaching example, independent of PRML's oil-flow data.
RED_POINTS=np.array([[-1.7,.7],[-1.3,.4],[-.8,.75],[-.45,.35],
                     [0,.35],[.45,.3],[.85,.7],[1.4,.4],[1.65,.85],[-1.4,-.65]])
BLUE_POINTS=np.array([[-1.65,-.25],[-1,-.75],[-.6,-.4],[-.2,-.25],
                      [.3,-.35],[.7,-.7],[1.1,-.3],[1.5,-.75],[1.8,-.15],[1.1,.95]])
POINTS=np.vstack([RED_POINTS,BLUE_POINTS])
LABELS=np.r_[np.zeros(len(RED_POINTS),dtype=int),np.ones(len(BLUE_POINTS),dtype=int)]
QUERY=np.array([.12,.08])

def neighbours(point,k):
    distances=np.linalg.norm(POINTS-np.asarray(point),axis=1)
    order=np.argsort(distances,kind='stable')[:k]
    return order, float(distances[order[-1]])

def posterior(point,k):
    order,_=neighbours(point,k)
    return np.bincount(LABELS[order],minlength=2)/k

def classify_grid(k,nx=320,ny=192):
    xx,yy=np.meshgrid(np.linspace(-2,2,nx),np.linspace(1.2,-1.2,ny))
    q=np.stack([xx,yy],axis=-1)
    d=np.linalg.norm(q[...,None,:]-POINTS,axis=-1)
    ids=np.argsort(d,axis=-1,kind='stable')[...,:k]
    return (LABELS[ids].sum(axis=-1)>k/2).astype(int)
