"""Independent checks of the example and the corrected PRML update."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma
from poisson_model import (A0, B0, COUNTS, PROBE_INDEX, exact_probe_probability,
                           history, update_rates, update_responsibilities)

states = history()
last = states[-1]
assert len(COUNTS) == 12 and COUNTS[PROBE_INDEX] == 3
assert np.allclose(last['r'].sum(axis=1), 1)
assert np.allclose(last['a'], A0 + last['effective_sum'])
assert np.allclose(last['b'], B0 + last['effective_n'])
assert np.allclose(last['r'], update_responsibilities(last['a'], last['b']), atol=2e-6)
assert np.allclose(update_rates(last['r']), (last['a'], last['b']))
assert np.max(np.abs(states[-1]['mean'] - states[-2]['mean'])) < 1e-5
assert np.all(last['a'] > 0) and np.all(last['b'] > 0)
# Corrected (10.121): nu_N chi_N = nu_0 chi_0 + sum E[u].  For each
# Poisson component nu=b and nu*chi=a; the Jacobian d eta=d lambda/lambda
# makes this Gamma(shape=a, rate=b) in lambda coordinates.
for k in range(2):
    area = quad(lambda rate: gamma.pdf(rate, last['a'][k], scale=1/last['b'][k]),
                0, np.inf)[0]
    mean = quad(lambda rate: rate * gamma.pdf(rate, last['a'][k], scale=1/last['b'][k]),
                0, np.inf)[0]
    assert abs(area - 1) < 1e-8
    assert abs(mean - last['mean'][k]) < 1e-7
exact = exact_probe_probability()
assert 0 < exact < 1
result = dict(counts=COUNTS.astype(int).tolist(), prior_shape=A0.tolist(), prior_rate=B0.tolist(),
              initial_probe_high=float(states[0]['r'][PROBE_INDEX, 1]),
              final_probe_high=float(last['r'][PROBE_INDEX, 1]),
              exact_probe_high=exact,
              means=last['mean'].tolist(), shapes=last['a'].tolist(), rates=last['b'].tolist(),
              effective_n=last['effective_n'].tolist(),
              effective_sum=last['effective_sum'].tolist(), iterations=len(states)-1)
Path('numerical_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(result)
