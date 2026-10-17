"""Reproducible Gaussian bridge used in the PRML 11.6 video."""
from math import exp, log, pi, sqrt
import numpy as np

SAMPLES = 80
SEED = 5
ALPHAS = np.linspace(0, 1, 9)

def energy(alpha, z):
    return (1-alpha)*np.asarray(z)**2/2 + alpha*((np.asarray(z)-4)**2/2-log(3))

def partition(alpha):
    return sqrt(2*pi)*exp(-8*alpha*(1-alpha)+alpha*log(3))

def weight(a, b, z):
    return np.exp(-energy(b,z)+energy(a,z))

def experiment():
    rng=np.random.default_rng(SEED)
    direct_samples=rng.normal(size=SAMPLES)
    direct_weights=weight(0,1,direct_samples)
    stage_samples=[];ratios=[]
    for a,b in zip(ALPHAS[:-1],ALPHAS[1:]):
        samples=rng.normal(4*a,1,SAMPLES)
        stage_samples.append(samples)
        ratios.append(float(np.mean(weight(a,b,samples))))
    return dict(direct_samples=direct_samples,direct_weights=direct_weights,
                direct=float(np.mean(direct_weights)),stage_samples=stage_samples,
                ratios=np.asarray(ratios),bridge=float(np.prod(ratios)))

EXPERIMENT=experiment()
