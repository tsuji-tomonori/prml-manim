"""Check the fixed story values and the coordinate update independently."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from variational_model import (COVARIANCE, EVIDENCE, FACTOR_VARIANCE, INITIAL_MEAN,
                               PRECISION, coordinate_steps, gaussian_density, kl_qp,
                               lower_bound, q_covariance)

assert np.all(np.linalg.eigvalsh(PRECISION) > 0)
assert np.allclose(PRECISION @ COVARIANCE, np.eye(2), atol=1e-13)
assert np.isclose(COVARIANCE[0, 0], 25 / 18)
assert np.isclose(FACTOR_VARIANCE, .5)
rows = coordinate_steps(10)
assert all(b['lower'] >= a['lower'] - 1e-12 for a, b in zip(rows, rows[1:]))
assert np.linalg.norm(rows[-1]['mean']) < .02
assert np.isclose(kl_qp([0, 0]), .5 * np.log(np.linalg.det(COVARIANCE) / .25))
assert np.isclose(lower_bound([0, 0]) + kl_qp([0, 0]), np.log(EVIDENCE))

# Independent 2-D quadrature of E_q[log(q/p)] for a shifted distribution.
grid = np.linspace(-7, 7, 501)
xx, yy = np.meshgrid(grid, grid, indexing='ij')
points = np.stack((xx, yy), axis=-1)
q = gaussian_density(points, INITIAL_MEAN, q_covariance())
p = gaussian_density(points, [0, 0], COVARIANCE)
integrand = q * np.log(np.maximum(q, 1e-300) / np.maximum(p, 1e-300))
quadrature = float(simpson(simpson(integrand, x=grid, axis=1), x=grid, axis=0))
assert abs(quadrature - kl_qp(INITIAL_MEAN)) < 1e-8

# The reverse direction among factorized Gaussians uses the exact marginals.
reverse_covariance = np.diag(np.diag(COVARIANCE))
assert np.isclose(reverse_covariance[0, 0], 25 / 18)
data = dict(precision=PRECISION.tolist(), covariance=COVARIANCE.tolist(),
            exact_marginal_variance=float(COVARIANCE[0, 0]), factor_variance=FACTOR_VARIANCE,
            variance_ratio=float(FACTOR_VARIANCE / COVARIANCE[0, 0]),
            log_evidence=float(np.log(EVIDENCE)), initial_kl=rows[0]['kl'],
            optimal_kl=kl_qp([0, 0]), initial_lower=rows[0]['lower'],
            optimal_lower=lower_bound([0, 0]), quadrature_kl=quadrature,
            coordinate_steps=[dict(which=r['which'], mean=r['mean'].tolist(),
                                   kl=r['kl'], lower=r['lower']) for r in rows])
(Path(__file__).resolve().parent / 'numerical_validation.json').write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print('PASS', {k: v for k, v in data.items() if k != 'coordinate_steps'})
