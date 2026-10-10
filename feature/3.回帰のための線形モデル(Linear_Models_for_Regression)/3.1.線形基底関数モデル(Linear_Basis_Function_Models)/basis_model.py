"""Numerical experiments for PRML 3.1; all data are generated here."""
import numpy as np

X = np.linspace(.04, .96, 12)
CENTERS = np.linspace(.08, .92, 8)
SCALE = .16

def target(x):
    return np.sin(2*np.pi*np.asarray(x)) + .25*np.cos(5*np.pi*np.asarray(x))

T = target(X) + np.random.default_rng(31).normal(0, .18, len(X))
T2 = .7*np.cos(2*np.pi*X) + np.random.default_rng(32).normal(0,.18,len(X))

def gaussian(x, centers=CENTERS, scale=SCALE):
    return np.exp(-.5*((np.atleast_1d(x)[:,None]-np.asarray(centers)[None,:])/scale)**2)

def design(x, centers=CENTERS, scale=SCALE):
    x=np.atleast_1d(x)
    return np.column_stack([np.ones(len(x)),gaussian(x,centers,scale)])

PHI=design(X)

# One held-out observation for the opening question and the final comparison.
# It is never included in PHI or T when fitting the weights.
HOLDOUT_X = .68
HOLDOUT_T = float(target(HOLDOUT_X) + .06)
LINEAR_COEF = np.polyfit(X, T, 1)
HOLDOUT_LINEAR = float(np.polyval(LINEAR_COEF, HOLDOUT_X))

def fit(lam=0, t=T):
    """Augmented least squares avoids explicitly forming a normal-equation inverse.

    All coefficients, including w0, are regularized, as in Eq.(3.27).
    """
    t=np.asarray(t)
    if lam == 0:
        return np.linalg.lstsq(PHI,t,rcond=None)[0]
    pad=np.zeros((PHI.shape[1],)+t.shape[1:])
    return np.linalg.lstsq(np.vstack([PHI,np.sqrt(lam)*np.eye(PHI.shape[1])]),
                           np.concatenate([t,pad]),rcond=None)[0]

W_ML=fit()
HOLDOUT_RIDGE_LAMBDA = .1
HOLDOUT_RIDGE = float((design([HOLDOUT_X]) @ fit(HOLDOUT_RIDGE_LAMBDA)).item())

def error(w,t=T):
    return float(np.sum((t-PHI@w)**2)/2)

def lms_step(w, index, eta=.22):
    phi=PHI[index]
    return w+eta*(T[index]-phi@w)*phi

ORDER=[2,9,4,7,0,11,1,8,3,10,5,6]
LMS=[np.zeros(PHI.shape[1])]
for i in ORDER: LMS.append(lms_step(LMS[-1],i))
LMS=np.array(LMS)

def interpolate_steps(value):
    i=min(int(value),len(LMS)-2)
    alpha=min(1.,value-i)
    return (1-alpha)*LMS[i]+alpha*LMS[i+1]

CONSTRAINT_TARGET=np.array([1.8,.5])
L2_POINT=CONSTRAINT_TARGET/np.linalg.norm(CONSTRAINT_TARGET)
L1_POINT=np.array([1.,0.])

def q_boundary(q, count=361):
    a=np.linspace(0,2*np.pi,count)
    c,s=np.cos(a),np.sin(a)
    r=(np.abs(c)**q+np.abs(s)**q)**(-1/q)
    return np.column_stack([r*c,r*s])


def constraint_solution(q):
    """Closest point on the q-ball for this positive 2D target, 1 <= q <= 2.

    Golden-section minimization of the angle on the first-quadrant boundary;
    compare endpoints explicitly. No interpolation of the minimizers is used.
    """
    if not 1 <= q <= 2:
        raise ValueError('The convex constraint experiment uses 1 <= q <= 2')
    if q == 1:
        return L1_POINT.copy()
    if q == 2:
        return L2_POINT.copy()
    def point(a):
        v=np.array([np.cos(a),np.sin(a)])
        return v/np.sum(v**q)**(1/q)
    def cost(a):return np.sum((point(a)-CONSTRAINT_TARGET)**2)
    low,high=0.,np.pi/2
    ratio=(np.sqrt(5)-1)/2
    a=high-ratio*(high-low);b=low+ratio*(high-low)
    fa,fb=cost(a),cost(b)
    for _ in range(65):
        if fa<fb:
            high,b,fb=b,a,fa;a=high-ratio*(high-low);fa=cost(a)
        else:
            low,a,fa=a,b,fb;b=low+ratio*(high-low);fb=cost(b)
    angle=min([0.,np.pi/2,(low+high)/2],key=cost)
    return point(angle)
