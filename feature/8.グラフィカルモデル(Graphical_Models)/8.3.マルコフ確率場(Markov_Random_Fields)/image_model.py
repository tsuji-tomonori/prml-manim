"""Deterministic binary-image example for PRML 8.3.

Black = -1, white = +1. Edges join only horizontal/vertical neighbours.
Each unordered edge appears once in the energy sum.
"""
from __future__ import annotations

import numpy as np

SIDE = 12
BETA = 1.0
ETA = 2.1
H = 0.0
TARGET = (5, 5)
FLIPS = ((0, 0), (0, 7), (0, 11), (2, 2), (2, 9), (3, 6),
         TARGET, (6, 6), (8, 5), (9, 2), (9, 9), (11, 0), (11, 7), (11, 11))


def original():
    image = np.ones((SIDE, SIDE), dtype=np.int8)
    image[1:11, 1:4] = -1
    image[1:11, 8:11] = -1
    image[4:8, 1:11] = -1
    return image


def observed():
    image = original().copy()
    for row, col in FLIPS:
        image[row, col] *= -1
    return image


def neighbours(row, col):
    return [(r, c) for r, c in ((row-1, col), (row+1, col),
                                 (row, col-1), (row, col+1))
            if 0 <= r < SIDE and 0 <= c < SIDE]


def energy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    unary = H * int(x.sum()) - ETA * int((x*y).sum())
    pairwise = -BETA * (int((x[:-1]*x[1:]).sum()) +
                        int((x[:, :-1]*x[:, 1:]).sum()))
    return unary + pairwise


def local_energy(x, y, row, col, candidate):
    """Terms depending on one candidate; the other terms cancel in comparison."""
    near = sum(int(x[r, c]) for r, c in neighbours(row, col))
    return H*candidate - BETA*candidate*near - ETA*candidate*int(y[row, col])


def icm(y, max_sweeps=50):
    x = y.copy()
    history = [x.copy()]
    changes = []
    for _ in range(max_sweeps):
        flipped = []
        for row in range(SIDE):
            for col in range(SIDE):
                old = int(x[row, col])
                new = min((-1, 1), key=lambda v: (local_energy(x, y, row, col, v), v != old))
                if new != old:
                    x[row, col] = new
                    flipped.append((row, col))
        history.append(x.copy())
        changes.append(flipped)
        if not flipped:
            return x, history, changes
    raise RuntimeError('ICM did not stop')


if __name__ == '__main__':
    y = observed()
    x, history, changes = icm(y)
    r, c = TARGET
    print('flips', len(FLIPS), 'pixels', SIDE**2)
    print('agreement', int((y == original()).sum()), int((x == original()).sum()))
    print('target y', int(y[r, c]), 'local white', local_energy(y, y, r, c, 1),
          'local black', local_energy(y, y, r, c, -1))
    print('sweep changes', [len(k) for k in changes])
    print('target restored', int(x[r, c]))
