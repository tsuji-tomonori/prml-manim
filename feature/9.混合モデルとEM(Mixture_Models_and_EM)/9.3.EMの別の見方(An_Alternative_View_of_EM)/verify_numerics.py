"""Independent identities and update checks for the fixed 9.3 examples."""
import json
import numpy as np
from mixture_model import (POINTS, PROBE_INDEX, INITIAL, em_history, evaluate,
                           expected_complete_log, common_variance_responsibility,
                           BINARY_PATTERNS, BINARY_INITIAL, bernoulli_step, bernoulli_components)

states=em_history(20)
first, second, third=states[:3]
gamma=first['gamma']
assert np.allclose(gamma.sum(axis=1),1)
assert abs(gamma[PROBE_INDEX,0]-.56355023)<1e-7
q0=expected_complete_log(gamma, first['mu'],first['sigma'],first['pi'])
q1=expected_complete_log(gamma, second['mu'],second['sigma'],second['pi'])
assert q1>=q0
count=gamma.sum(axis=0)
mu=(gamma*POINTS[:,None]).sum(axis=0)/count
sigma=np.sqrt((gamma*(POINTS[:,None]-mu)**2).sum(axis=0)/count)
assert np.allclose(mu,second['mu']) and np.allclose(sigma,second['sigma'])
assert np.allclose(count/len(POINTS),second['pi'])
assert all(b['log_likelihood']>=a['log_likelihood']-1e-10 for a,b in zip(states,states[1:]))
assert abs(second['log_likelihood']+26.5205845559)<1e-8
assert abs(third['log_likelihood']+26.2931403347)<1e-8
# The fixed-posterior Q is not the observed log likelihood.
assert abs(q0-first['log_likelihood'])>1
# A 1-D common-variance experiment: unequal distances converge to hard assignment.
soft=[common_variance_responsibility(.15,e).tolist() for e in [1.,.5,.2,.05]]
assert soft[-1][1]>.99 and soft[0][0]>soft[0][1]
# A 4-pixel Bernoulli mixture with positive, independent pixel probabilities.
b_gamma,b_mu,b_pi=bernoulli_step()
assert np.allclose(b_gamma.sum(axis=1),1) and np.allclose(b_pi.sum(),1)
assert np.all((b_mu>0)&(b_mu<1))
manual=bernoulli_components()
assert np.allclose(b_gamma,manual/manual.sum(axis=1,keepdims=True))
assert np.allclose(b_mu[0],(b_gamma[:,0,None]*BINARY_PATTERNS).sum(axis=0)/b_gamma[:,0].sum())
result=dict(probe_red=float(gamma[PROBE_INDEX,0]),q_initial=q0,q_after_m=q1,
            observed_log_likelihood=[s['log_likelihood'] for s in states],
            first_update={k:second[k].tolist() for k in ('mu','sigma','pi')},
            probe_after_m=float(second['gamma'][PROBE_INDEX,0]),
            common_variance=[dict(variance=v,responsibility=g) for v,g in zip([1.,.5,.2,.05],soft)],
            bernoulli=dict(patterns=BINARY_PATTERNS.astype(int).tolist(),initial_mu=BINARY_INITIAL['mu'].tolist(),
                           responsibilities=b_gamma.tolist(),updated_mu=b_mu.tolist(),updated_pi=b_pi.tolist()))
open('numerical_validation.json','w').write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:result[k] for k in ('probe_red','q_initial','q_after_m','probe_after_m')})
