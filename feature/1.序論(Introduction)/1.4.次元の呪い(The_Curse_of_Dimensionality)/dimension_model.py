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


def class_neighborhood_masks():
    """Same 90 observations and query, with eight independent nuisance inputs.

    The first two coordinates use the visible box [.2,.6) x [.4,.8).
    Each added coordinate uses a width-.4 interval [.3,.7) about query .5.
    The added measurements are independent of the class labels by construction.
    """
    points, labels = class_data()
    extra = np.random.default_rng(1402).uniform(0, 1, (len(points), 8))
    visible = np.all((points >= [.2, .4]) & (points < [.6, .8]), axis=1)
    masks = []
    for dimension in range(2, 11):
        nearby = visible & np.all((extra[:, :dimension-2] >= .3)
                                & (extra[:, :dimension-2] < .7), axis=1)
        masks.append(nearby)
    return np.asarray(masks)


def class_neighborhood_counts():
    _, labels = class_data()
    return np.asarray([np.bincount(labels[mask], minlength=3)
                       for mask in class_neighborhood_masks()])


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
