"""Verify the example against a direct joint-Gaussian calculation."""
import json
from pathlib import Path
import numpy as np
from lds_model import OBS, infer, INITIAL_VARIANCE, STATE_VARIANCE, SENSOR_VARIANCE

r = infer()
n = len(OBS)
# Random-walk covariance: cov(z_i,z_j) = P0 + min(i,j)*Gamma.
indices = np.arange(n)
prior_cov = INITIAL_VARIANCE + np.minimum.outer(indices, indices)*STATE_VARIANCE
observation_cov = prior_cov + np.eye(n)*SENSOR_VARIANCE
exact_mean = prior_cov @ np.linalg.solve(observation_cov, OBS)
exact_cov = prior_cov - prior_cov @ np.linalg.solve(observation_cov, prior_cov)
assert np.allclose(r['smooth_mean'], exact_mean, atol=1e-12)
assert np.allclose(r['smooth_var'], np.diag(exact_cov), atol=1e-12)
assert np.all(r['var'] > 0) and np.all(r['smooth_var'] > 0)
assert np.all((r['gain'] > 0) & (r['gain'] < 1))
assert abs(r['mean'][3] - 1.0886) < .001
assert abs(r['smooth_mean'][3] - 1.0510) < .001
low_noise = infer(sensor_variance=.04)
assert low_noise['gain'][3] > r['gain'][3]
assert low_noise['mean'][3] > r['mean'][3]
summary = {'observations': OBS.tolist(), 'prior_mean': r['prior_mean'].tolist(),
           'prior_variance': r['prior_var'].tolist(), 'gain': r['gain'].tolist(),
           'filtered_mean': r['mean'].tolist(), 'filtered_variance': r['var'].tolist(),
           'smoothed_mean': r['smooth_mean'].tolist(),
           'smoothed_variance': r['smooth_var'].tolist(),
           'low_sensor_noise_gain_at_4': float(low_noise['gain'][3]),
           'low_sensor_noise_mean_at_4': float(low_noise['mean'][3]),
           'direct_gaussian_max_error': float(np.max(np.abs(exact_mean-r['smooth_mean'])))}
Path('numerical_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
