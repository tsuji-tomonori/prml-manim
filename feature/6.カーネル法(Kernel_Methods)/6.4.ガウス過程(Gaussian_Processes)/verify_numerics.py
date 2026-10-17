"""Check the example and the Gaussian-conditioning identities independently."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from gp_model import X, T, ELL, NOISE_SD, PROBE, covariance, kernel, predict, log_evidence

c = covariance()
assert np.allclose(c, c.T, atol=1e-14)
assert np.linalg.eigvalsh(c).min() > 0
factor = cho_factor(c)
k = kernel(X, [PROBE]).ravel()
alpha = cho_solve(factor, T)
mean = float(k @ alpha)
latent_var = float(1 - k @ cho_solve(factor, k))
observed_var = latent_var + NOISE_SD**2
m2, v2 = predict([PROBE])
assert np.isclose(mean, m2[0], atol=1e-12)
assert np.isclose(observed_var, v2[0], atol=1e-12)
assert np.isclose(mean, .12328525265504306, atol=1e-12)
assert np.isclose(observed_var, 1.028911864366959, atol=1e-12)
assert np.sqrt(predict([0])[1][0]) < np.sqrt(observed_var)
sign, ld = np.linalg.slogdet(c)
evidence = -.5 * ld - .5 * T @ alpha - len(T)/2*np.log(2*np.pi)
assert sign > 0 and np.isclose(evidence, log_evidence(), atol=1e-12)
result = dict(x=X.tolist(), targets=T.tolist(), probe=PROBE, ell=ELL,
              noise_sd=NOISE_SD, covariance=c.tolist(),
              mean=mean, latent_variance=latent_var, observed_variance=observed_var,
              observed_sd=float(np.sqrt(observed_var)),
              observed_two_sd=[float(mean-2*np.sqrt(observed_var)),float(mean+2*np.sqrt(observed_var))],
              log_evidence=float(evidence), minimum_eigenvalue=float(np.linalg.eigvalsh(c).min()),
              checks='passed')
Path('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', result)
