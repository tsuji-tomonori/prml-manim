"""Actual NumPy network functions used for every plotted curve and readout."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
X = np.linspace(-1, 1, 41)
GRID = np.linspace(-1, 1, 401)


def sigmoid(x):
    x = np.asarray(x)
    return np.exp(-np.logaddexp(0., -x))


def softmax(x):
    e = np.exp(np.asarray(x) - np.max(x))
    return e / e.sum()


def components(x, weights):
    w = np.asarray(weights)
    return np.tanh(np.asarray(x)[..., None] * w[:3] + w[3:6]) * w[6:9]


def network(x, weights):
    return components(x, weights).sum(axis=-1) + weights[9]


def bump(x, center=.4, amplitude=.65):
    x = np.asarray(x)
    return amplitude * (np.tanh(3*(x+center))-np.tanh(3*(x-center)))


def targets(index, x):
    x = np.asarray(x)
    return [lambda: x*x, lambda: np.sin(np.pi*x), lambda: abs(x),
            lambda: np.where(x >= 0, 1., 0.)][index]()


def class_hidden(points, weight=2.):
    w = np.array([[weight, 1.], [-1., 2.]])
    return np.tanh(np.asarray(points) @ w.T + np.array([.3, -.4]))


def class_score(points, weight=2.):
    return class_hidden(points, weight) @ np.array([2., 1.3]) - .35


def decision_boundary(weight=2.):
    x = np.linspace(-1.6, 1.6, 181)
    low, high = np.full_like(x, -1.6), np.full_like(x, 1.6)
    mask = (class_score(np.c_[x, low], weight) <= 0) & (class_score(np.c_[x, high], weight) >= 0)
    for _ in range(45):
        middle = (low + high)/2
        negative = class_score(np.c_[x, middle], weight) < 0
        low = np.where(negative, middle, low)
        high = np.where(negative, high, middle)
    return np.c_[x[mask], ((low + high)/2)[mask]]


def fitted_weights():
    return [np.array(r['weights']) for r in json.loads((ROOT/'network_fits.json').read_text())['fits']]


def fit_examples():
    """Three tanh hidden units, linear output; numerical least squares, not a training tutorial."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(51)
    fits = []
    for index in range(3):
        best = None
        for _ in range(5):
            start = np.r_[rng.uniform(.7, 3, 3), [-1., 0., 1.], rng.normal(0,.6,3), 0.]
            fit = least_squares(lambda w: network(X,w)-targets(index,X), start,
                                bounds=(-12,12), max_nfev=2500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
                best = fit
        w = best.x
        fits.append(dict(target=['x^2','sin(pi*x)','abs(x)'][index], weights=w.tolist(),
                         rms=float(np.sqrt(np.mean(best.fun**2))), max_grid_error=float(np.max(abs(network(GRID,w)-targets(index,GRID))))))
    # The step illustrates a continuous approximation, not the continuous-function theorem.
    w = [18., 1., 1., 0., 0., 0., .5, 0., 0., .5]
    fits.append(dict(target='H(x)', weights=w, rms=float(np.sqrt(np.mean((network(X,w)-targets(3,X))**2))),
                     max_grid_error=float(np.max(abs(network(GRID,w)-targets(3,GRID)))), method='explicit sigmoid approximation'))
    (ROOT/'network_fits.json').write_text(json.dumps(dict(seed=51,n=41,hidden=3,fits=fits),indent=2)+'\n')
    print(json.dumps(fits,indent=2))


if __name__ == '__main__':
    fit_examples()
