"""Reproducible numerical experiments for PRML 1.4 (no copied textbook data)."""
import math
import numpy as np


def cell_count(d):
    return 5 ** int(d)


def empty_fraction(d, n=1000):
    return float(np.exp(n * np.log1p(-1 / cell_count(d))))


def coefficient_count(d):
    return math.comb(int(d) + 3, 3)


def shell_fraction(d, epsilon):
    return -np.expm1(d * np.log1p(-np.asarray(epsilon)))


def radial_pdf(r, d):
    r = np.asarray(r, dtype=float)
    safe = np.maximum(r, 1e-100)
    out = np.exp((d-1)*np.log(safe) - safe**2/2 - ((d/2-1)*np.log(2)+math.lgamma(d/2)))
    return np.where(r == 0, np.sqrt(2/np.pi) if d == 1 else 0., out)


def normalized_pdf(u, d):
    return np.sqrt(d) * radial_pdf(np.asarray(u)*np.sqrt(d), d)


def class_data():
    rng = np.random.default_rng(1401)
    centers = [[.40,.58],[.64,.60],[.68,.21]]
    pts=np.concatenate([rng.normal(c, [.13,.12], (30,2)) for c in centers])
    return np.clip(pts,.025,.975), np.repeat(np.arange(3),30)


GAUSSIAN_POINTS=np.random.default_rng(1406).normal(size=(600,2))
GAUSSIAN_RADII=np.linalg.norm(GAUSSIAN_POINTS,axis=1)


def object_pixels(x=0., y=0., angle=0., n=16):
    """Smooth asymmetric planar object, so subpixel shifts remain continuous."""
    a=np.linspace(-1.5,1.5,n)
    xx,yy=np.meshgrid(a,a[::-1]); xx-=x; yy-=y
    u=np.cos(angle)*xx+np.sin(angle)*yy
    v=-np.sin(angle)*xx+np.cos(angle)*yy
    body=np.exp(-((u/.65)**8+(v/.22)**8))
    head=np.exp(-(((u-.44)/.22)**4+((v-.24)/.25)**4))
    return np.maximum(body,head)
