"""All plotted quantities are computed, with explicit proper priors."""
import math
import numpy as np

NOISE=.18
TAU=1.
X=np.linspace(-1,1,12)
T=.25+.25*X+.9*X**2+np.random.default_rng(34).normal(0,NOISE,len(X))
GRID=np.linspace(-1,1,241)
DEGREES=np.arange(8)

def normal(x,mean=0.,std=1.):
    return np.exp(-.5*((np.asarray(x)-mean)/std)**2)/(np.sqrt(2*np.pi)*std)

def uniform_evidence(width,datum=1.,noise=.35):
    # Integral N(datum | w, noise²) / width over the prior support.
    return .5*(math.erf((width/2-datum)/(np.sqrt(2)*noise))-
               math.erf((-width/2-datum)/(np.sqrt(2)*noise)))/width

def design(x,degree):
    # Legendre basis keeps coefficient scales comparable on [-1,1].
    return np.polynomial.legendre.legvander(x,degree)

def regression(degree,tau=TAU):
    phi=design(X,degree)
    c=NOISE**2*np.eye(len(X))+tau**2*phi@phi.T
    sign,logdet=np.linalg.slogdet(c)
    assert sign>0
    evidence=-.5*(len(X)*np.log(2*np.pi)+logdet+T@np.linalg.solve(c,T))
    precision=np.eye(degree+1)/tau**2+phi.T@phi/NOISE**2
    mean=np.linalg.solve(precision,phi.T@T/NOISE**2)
    ml=np.linalg.lstsq(phi,T,rcond=None)[0]
    likelihood=-.5*len(X)*np.log(2*np.pi*NOISE**2)-np.sum((phi@ml-T)**2)/(2*NOISE**2)
    return dict(evidence=float(evidence),likelihood=float(likelihood),mean=mean,ml=ml,cov=np.linalg.inv(precision))

FITS=[regression(int(d)) for d in DEGREES]
LOG_EVIDENCE=np.array([f['evidence'] for f in FITS])
LOG_ML=np.array([f['likelihood'] for f in FITS])
CURVES=np.array([design(GRID,int(d))@f['mean'] for d,f in zip(DEGREES,FITS)])
ML_CURVES=np.array([design(GRID,int(d))@f['ml'] for d,f in zip(DEGREES,FITS)])
MODEL_STDS=np.array([.35,1.3,3.])
DATUM=1.4
EVIDENCES=normal(DATUM,std=MODEL_STDS)

def posterior(prior=np.ones(3)/3,datum=DATUM):
    joint=np.asarray(prior)*normal(datum,std=MODEL_STDS)
    return joint/joint.sum()

def interpolate_rows(rows,value):
    lo=min(int(value),len(rows)-1);hi=min(lo+1,len(rows)-1)
    return (1-value+lo)*rows[lo]+(value-lo)*rows[hi]

K=np.arange(7)
P_TRUE=.65
P_ALT=.35
COUNT_PROBS=np.array([math.comb(6,int(k))*P_TRUE**k*(1-P_TRUE)**(6-k) for k in K])
LOG_BF=K*np.log(P_TRUE/P_ALT)+(6-K)*np.log((1-P_TRUE)/(1-P_ALT))
EXPECTED_LOG_BF=float(COUNT_PROBS@LOG_BF)
