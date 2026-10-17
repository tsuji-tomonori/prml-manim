"""Check the actual data, bound identity, EM inequality, and pi-only slice."""
import json
import numpy as np
from em_model import (POINTS, PROBE_INDEX, INITIAL, components, decomposition,
                      em_history, pi_slice)

states = em_history(3)
old, new = states[:2]
q = old['gamma']
a = decomposition(q, old['mu'], old['sigma'], old['pi'])
b = decomposition(q, new['mu'], new['sigma'], new['pi'])
assert len(POINTS) == 16 and np.allclose(q.sum(axis=1), 1)
assert abs(q[PROBE_INDEX, 0] - .5635502272848772) < 1e-12
assert abs(a['observed'] - a['lower']) < 1e-12 and abs(a['kl']) < 1e-12
assert abs(b['observed'] - b['lower'] - b['kl']) < 1e-12
assert b['observed'] > b['lower'] > a['observed']
assert abs(b['observed'] + 26.520584555897734) < 1e-11
assert abs(b['lower'] + 26.686791384115736) < 1e-11
assert abs(b['kl'] - .1662068282180016) < 1e-11
assert np.isclose(new['pi'][0], q[:, 0].mean())
assert np.allclose(new['mu'], (q * POINTS[:, None]).sum(axis=0) / q.sum(axis=0))
# Independently sum the 2 latent alternatives per point and their KL terms.
manual_log = sum(np.log(sum(components(x, old['mu'], old['sigma'], old['pi'])[k]
                            for k in range(2))) for x in POINTS)
assert abs(manual_log - a['observed']) < 1e-12
for pi in np.linspace(.1, .9, 81):
    item = pi_slice(pi)
    assert abs(item['lower'] + item['kl'] - item['observed']) < 1e-11
    assert item['kl'] >= -1e-12
old_slice = pi_slice(.65)
opt_pi = float(q[:, 0].mean())
new_slice = pi_slice(opt_pi)
assert new_slice['lower'] > old_slice['lower']
assert new_slice['observed'] >= new_slice['lower']
result = dict(points=POINTS.tolist(), initial={k: v.tolist() for k, v in INITIAL.items()},
              probe_red=float(q[PROBE_INDEX, 0]), first_update={k: new[k].tolist() for k in ('mu', 'sigma', 'pi')},
              before={k: a[k] for k in ('q', 'entropy', 'lower', 'kl', 'observed')},
              after={k: b[k] for k in ('q', 'entropy', 'lower', 'kl', 'observed')},
              pi_slice=dict(old=.65, bound_max=opt_pi, old_value={k: old_slice[k] for k in ('lower', 'kl', 'observed')},
                            new_value={k: new_slice[k] for k in ('lower', 'kl', 'observed')}))
with open('numerical_validation.json', 'w') as output:
    json.dump(result, output, ensure_ascii=False, indent=2)
    output.write('\n')
print('PASS', result['before'], result['after'], result['pi_slice']['bound_max'])
