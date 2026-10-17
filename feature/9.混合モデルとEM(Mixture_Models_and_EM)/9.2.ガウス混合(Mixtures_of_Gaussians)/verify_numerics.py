"""Recalculate all displayed values without calling the model's EM update."""
import json
from pathlib import Path
import numpy as np
from mixture_model import POINTS, INITIAL, PROBE_INDEX, em_history, collapsed_example

states = em_history(20)
assert len(POINTS) == 16 and POINTS[PROBE_INDEX] == .15
independent = []
for index, state in enumerate(states):
    mu, sigma, pi = (state[k] for k in ('mu', 'sigma', 'pi'))
    q = np.empty((len(POINTS), 2))
    for n, x in enumerate(POINTS):
        for k in range(2):
            q[n, k] = float(pi[k]) * np.exp(-(float(x)-float(mu[k]))**2/(2*float(sigma[k])**2))/(np.sqrt(2*np.pi)*float(sigma[k]))
    gamma = q / q.sum(axis=1)[:, None]
    ll = sum(np.log(q[n].sum()) for n in range(len(POINTS)))
    assert np.allclose(gamma, state['gamma'], atol=1e-12)
    assert abs(ll-state['log_likelihood']) < 1e-11
    assert np.allclose(gamma.sum(axis=1), 1)
    assert np.all(sigma > 0) and np.isclose(sum(pi), 1)
    if index + 1 < len(states):
        counts = gamma.sum(axis=0)
        new_mu = np.array([sum(gamma[:, k]*POINTS)/counts[k] for k in range(2)])
        new_sigma = np.array([np.sqrt(sum(gamma[:, k]*(POINTS-new_mu[k])**2)/counts[k]) for k in range(2)])
        next_state = states[index+1]
        assert np.allclose(new_mu, next_state['mu'])
        assert np.allclose(new_sigma, next_state['sigma'])
        assert np.allclose(counts/len(POINTS), next_state['pi'])
    independent.append(float(ll))
assert all(b >= a-1e-10 for a, b in zip(independent, independent[1:]))
assert np.isclose(states[0]['gamma'][PROBE_INDEX, 0], .5636, atol=.0001)
assert np.isclose(states[0]['log_likelihood'], -27.73748, atol=.00001)
collapse = {str(s): dict(log_likelihood=collapsed_example(s)[0], probe_density=collapsed_example(s)[1]) for s in (.03, .005, .001)}
assert collapse['0.03']['log_likelihood'] < collapse['0.005']['log_likelihood'] < collapse['0.001']['log_likelihood']
result = dict(point_count=len(POINTS), initial_probe_responsibility=states[0]['gamma'][PROBE_INDEX].tolist(),
              log_likelihood=independent, states=[dict(mu=s['mu'].tolist(), sigma=s['sigma'].tolist(),
              pi=s['pi'].tolist(), probe_responsibility=s['gamma'][PROBE_INDEX].tolist()) for s in states],
              separate_collapse_path=collapse, checks='passed')
(Path(__file__).parent/'numerical_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print('PASS', len(states), result['initial_probe_responsibility'], independent[0], independent[-1])
