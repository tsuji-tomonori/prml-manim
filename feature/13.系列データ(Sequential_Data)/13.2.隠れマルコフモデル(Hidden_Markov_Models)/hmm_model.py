"""Small, exact two-state HMM used throughout PRML 13.2."""
from __future__ import annotations

import itertools
import numpy as np

PI = np.array([.5, .5])
A = np.array([[.9, .1], [.1, .9]])
B = np.array([[.8, .2], [.2, .8]])  # columns: clear, rain
OBS = (0, 1, 0, 0, 0)
PATH_OBS = (0, 0, 1, 0, 1)


def infer(obs=OBS, pi=PI, a=A, b=B):
    """Scaled forward/backward, smoothing, pairwise posterior, and likelihood."""
    obs = tuple(obs)
    n = len(obs)
    f = np.zeros((n, 2))
    scale = np.zeros(n)
    for t, x in enumerate(obs):
        prior = pi if t == 0 else f[t-1] @ a
        unnormalized = prior * b[:, x]
        scale[t] = unnormalized.sum()
        f[t] = unnormalized / scale[t]
    back = np.ones((n, 2))
    for t in range(n-2, -1, -1):
        back[t] = a @ (b[:, obs[t+1]] * back[t+1]) / scale[t+1]
    gamma = f * back
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((n-1, 2, 2))
    for t in range(n-1):
        xi[t] = f[t, :, None] * a * (b[:, obs[t+1]] * back[t+1])[None, :]
        xi[t] /= xi[t].sum()
    return dict(filter=f, backward=back, gamma=gamma, xi=xi,
                scale=scale, likelihood=float(np.prod(scale)))


def em_step(obs=OBS, pi=PI, a=A, b=B):
    """One full discrete-emission EM update; no smoothing prior."""
    result = infer(obs, pi, a, b)
    g, x = result['gamma'], result['xi']
    new_pi = g[0].copy()
    counts = x.sum(axis=0)
    new_a = counts / counts.sum(axis=1, keepdims=True)
    new_b = np.zeros_like(b)
    for symbol in range(b.shape[1]):
        new_b[:, symbol] = g[np.array(obs) == symbol].sum(axis=0)
    new_b /= g.sum(axis=0)[:, None]
    return dict(pi=new_pi, a=new_a, b=new_b, old_likelihood=result['likelihood'],
                new_likelihood=infer(obs, new_pi, new_a, new_b)['likelihood'],
                transition_counts=counts)


def viterbi(obs=PATH_OBS, pi=PI, a=A, b=B):
    """Maximum joint probability path and its dynamic-programming backpointers."""
    obs = tuple(obs)
    score = np.log(pi) + np.log(b[:, obs[0]])
    rows = [score.copy()]
    pointers = []
    for symbol in obs[1:]:
        candidates = score[:, None] + np.log(a)
        pointer = np.argmax(candidates, axis=0)
        score = candidates[pointer, np.arange(2)] + np.log(b[:, symbol])
        pointers.append(pointer)
        rows.append(score.copy())
    path = [int(np.argmax(score))]
    for pointer in reversed(pointers):
        path.append(int(pointer[path[-1]]))
    path.reverse()
    return dict(path=tuple(path), probability=float(np.exp(score.max())),
                scores=np.array(rows), pointers=np.array(pointers))


def exhaustive(obs, pi=PI, a=A, b=B):
    """Independent direct path summation for verification only."""
    paths = []
    for states in itertools.product(range(2), repeat=len(obs)):
        weight = pi[states[0]] * b[states[0], obs[0]]
        for t in range(1, len(obs)):
            weight *= a[states[t-1], states[t]] * b[states[t], obs[t]]
        paths.append((weight, states))
    return paths
