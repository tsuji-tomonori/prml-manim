"""Fixed, positive binary chain used throughout the 8.4 story."""
from itertools import product
import numpy as np

TABLES = np.array([[[1, 4], [4, 4]],
                   [[4, 1], [3, 3]],
                   [[3, 3], [1, 3]]], dtype=int)
OBSERVED = (3, 1)  # zero-based x4 = 1


def weight(state):
    return int(np.prod([TABLES[i, state[i], state[i + 1]] for i in range(3)]))


def candidates():
    return [(state, weight(state)) for state in product((0, 1), repeat=4)
            if state[OBSERVED[0]] == OBSERVED[1]]


def sum_messages():
    # mu_L(x2) = sum_x1 psi12(x1,x2)
    left = TABLES[0].sum(axis=0)
    # mu_R(x3) = psi34(x3, observed x4)
    right3 = TABLES[2][:, OBSERVED[1]]
    # mu_R(x2) = sum_x3 psi23(x2,x3) mu_R(x3)
    right = TABLES[1] @ right3
    return left, right, left * right


def max_trace():
    """Max-product with deterministic first-index tie breaking and backtracking."""
    scores = np.ones(2, dtype=int)
    previous = []
    for table in TABLES:
        products = scores[:, None] * table
        previous.append(products.argmax(axis=0))
        scores = products.max(axis=0)
    state = [None] * 4
    state[3] = OBSERVED[1]
    for i in (2, 1, 0):
        state[i] = int(previous[i][state[i + 1]])
    return tuple(state), int(scores[OBSERVED[1]]), previous
