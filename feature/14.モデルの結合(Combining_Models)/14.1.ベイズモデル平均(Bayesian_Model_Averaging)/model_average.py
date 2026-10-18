"""Exact finite example for PRML 14.1: local mixture versus global model index."""
from fractions import Fraction

OBSERVATIONS = ('赤', '赤', '赤', '青')
P_RED = {'A': Fraction(4, 5), 'B': Fraction(1, 5)}
PRIOR = {'A': Fraction(1, 2), 'B': Fraction(1, 2)}


def point_probability(colour, model):
    red = P_RED[model]
    return red if colour == '赤' else 1 - red


def mixture_probability(colour):
    return sum(PRIOR[h] * point_probability(colour, h) for h in PRIOR)


def data_likelihood(model, data=OBSERVATIONS):
    result = Fraction(1)
    for colour in data:
        result *= point_probability(colour, model)
    return result


def posterior(data=OBSERVATIONS):
    unscaled = {h: PRIOR[h] * data_likelihood(h, data) for h in PRIOR}
    total = sum(unscaled.values())
    return {h: weight / total for h, weight in unscaled.items()}


def predictive_red(data=OBSERVATIONS):
    return sum(weight * P_RED[h] for h, weight in posterior(data).items())


def local_data_probability(data=OBSERVATIONS):
    result = Fraction(1)
    for colour in data:
        result *= mixture_probability(colour)
    return result


def global_data_probability(data=OBSERVATIONS):
    return sum(PRIOR[h] * data_likelihood(h, data) for h in PRIOR)
