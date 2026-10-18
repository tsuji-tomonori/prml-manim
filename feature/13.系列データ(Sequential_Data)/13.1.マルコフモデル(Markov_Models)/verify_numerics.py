"""Independent enumeration checks for the pedagogical sequence model."""
import itertools
import json
from pathlib import Path
import numpy as np
from markov_model import INITIAL, RAIN, TRANSITION, filter_history, predict_rain, parameters


def joint(observations):
    total = 0.0
    for states in itertools.product(range(2), repeat=len(observations)):
        weight = INITIAL[states[0]]
        for n, (state, obs) in enumerate(zip(states, observations)):
            weight *= RAIN[state] if obs else 1 - RAIN[state]
            if n:
                weight *= TRANSITION[states[n-1], state]
        total += weight
    return total


def independent_prediction(history):
    return joint([*history, 1]) / joint(history)


def last_state_masses(observations):
    masses = np.zeros(2)
    for states in itertools.product(range(2), repeat=len(observations)):
        weight = INITIAL[states[0]]
        for n, (state, obs) in enumerate(zip(states, observations)):
            weight *= RAIN[state] if obs else 1 - RAIN[state]
            if n:
                weight *= TRANSITION[states[n-1], state]
        masses[states[-1]] += weight
    return masses


def main():
    assert np.allclose(TRANSITION.sum(axis=1), 1)
    assert np.allclose(INITIAL @ TRANSITION, INITIAL)
    assert np.all((RAIN > 0) & (RAIN < 1))
    histories = [[], [0], [1], [0, 1], [1, 1], [0, 1, 1], [1, 1, 1]]
    results, state_posteriors = {}, {}
    for history in histories:
        prediction = predict_rain(history)
        if history:
            assert abs(prediction - independent_prediction(history)) < 1e-12
            filtered = filter_history(history)[-1]
            masses = last_state_masses(history)
            direct_posterior = masses / masses.sum()
            assert np.allclose(filtered['posterior'], direct_posterior, atol=1e-12)
            assert np.allclose(filtered['next_prior'], direct_posterior @ TRANSITION,
                               atol=1e-12)
            state_posteriors[''.join(map(str, history))] = filtered['posterior'].tolist()
        results[''.join(map(str, history)) or 'empty'] = prediction
    assert abs(results['empty'] - 0.5) < 1e-12
    assert abs(results['1'] - 0.644) < 1e-12
    assert results['01'] < results['11']
    assert results['011'] < results['111']
    assert [parameters(2, m) for m in [1, 2, 3, 5]] == [2, 4, 8, 32]
    assert [parameters(3, m) for m in [1, 2, 3]] == [6, 18, 54]
    output = {'histories': results, 'state_posteriors': state_posteriors,
              'free_parameters_k2':
              {str(m): parameters(2, m) for m in [1, 2, 3, 5]},
              'free_parameters_k3': {str(m): parameters(3, m) for m in [1, 2, 3]}}
    Path('numerical_validation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
