"""Deterministic, self-contained 4x4 regression example for PRML 14.4."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
import numpy as np

X = np.array(list(itertools.product((.5, 1.5, 2.5, 3.5), repeat=2)), dtype=float)
T = np.where(X[:, 0] < 2, np.where(X[:, 1] < 2, 2., 6.), 8.)
T[(X[:, 0] == 3.5) & (X[:, 1] == 3.5)] = 14.
QUERY = np.array([3.5, 3.5])
NEXT_T = 8.


def sse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sum((values - values.mean()) ** 2)) if len(values) else 0.


@dataclass
class Leaf:
    members: tuple[int, ...]
    path: tuple[tuple[int, float, bool], ...] = ()

    @property
    def mean(self):
        return float(np.mean(T[list(self.members)]))

    @property
    def error(self):
        return sse(T[list(self.members)])


def candidates(leaf):
    for axis in (0, 1):
        values = sorted(set(X[list(leaf.members), axis]))
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left = tuple(i for i in leaf.members if X[i, axis] <= threshold)
            right = tuple(i for i in leaf.members if X[i, axis] > threshold)
            reduction = leaf.error - sse(T[list(left)]) - sse(T[list(right)])
            yield reduction, axis, threshold, Leaf(left, leaf.path + ((axis, threshold, False),)), Leaf(right, leaf.path + ((axis, threshold, True),))


def grow():
    leaves = [Leaf(tuple(range(len(T))))]
    stages = [(leaves.copy(), None)]
    while True:
        choices = [(gain, -j, -axis, -threshold, j, axis, threshold, left, right)
                   for j, leaf in enumerate(leaves) for gain, axis, threshold, left, right in candidates(leaf)
                   if gain > 1e-9]
        if not choices:
            break
        gain, _, _, _, index, axis, threshold, left, right = max(choices)
        leaves = leaves[:index] + [left, right] + leaves[index + 1:]
        stages.append((leaves.copy(), (axis, threshold, gain)))
    return stages


def predict(leaves, point):
    for leaf in leaves:
        if all((point[axis] > threshold) == right for axis, threshold, right in leaf.path):
            return leaf.mean
    raise ValueError('No leaf contains point')


def cost(leaves, lam):
    return sum(leaf.error for leaf in leaves) + lam * len(leaves)
