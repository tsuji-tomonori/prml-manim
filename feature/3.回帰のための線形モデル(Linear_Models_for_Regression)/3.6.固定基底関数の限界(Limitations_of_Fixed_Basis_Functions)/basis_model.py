"""Reproducible NumPy experiments; no PRML figure data are copied."""
import numpy as np

CENTERS = np.array([-.65, 0., .65])
WIDTH = .23
X = np.linspace(-1, 1, 25)

def design(x):
    x = np.atleast_1d(x)
    return np.column_stack([np.ones(len(x)), np.exp(-.5*((x[:, None]-CENTERS)/WIDTH)**2)])

T = design(X) @ np.array([.06, .90, .60, .85]) + np.random.default_rng(3601).normal(0,.035,len(X))
BEST = np.linalg.lstsq(design(X), T, rcond=None)[0]

def manifold(s):
    s = np.asarray(s)
    return np.stack([s, .68*np.sin(2.4*s)], axis=-1)

S = np.linspace(-.94, .94, 72)
CLEAN = manifold(S)
NOISE = np.random.default_rng(3603).normal(0,.022,CLEAN.shape)
DATA = CLEAN + NOISE
GRID = np.array([(x,y) for y in np.linspace(-1,1,9) for x in np.linspace(-1,1,9)])
DISTANCE = np.linalg.norm(GRID[:,None,:]-DATA[None,:,:],axis=-1).min(axis=1)
NEAR = DISTANCE < .18
LOCAL = DATA[np.round(np.linspace(0,len(DATA)-1,12)).astype(int)]
SQUARE = np.random.default_rng(3605).uniform(-.94,.94,(70,2))

def sigmoid(z):
    return 1/(1+np.exp(-np.asarray(z)))

def direction(angle):
    return np.array([np.cos(angle),np.sin(angle)])

def response(x,angle=np.pi/4,slope=3.,bias=0.):
    return sigmoid(slope*(np.asarray(x)@direction(angle))+bias)

TARGET = response(SQUARE)

def rmse(angle):
    return float(np.sqrt(np.mean((response(SQUARE,angle)-TARGET)**2)))

def radial(x,centers=LOCAL,h=.22):
    return np.exp(-np.sum((np.asarray(x)-centers)**2,axis=-1)/(2*h*h))
