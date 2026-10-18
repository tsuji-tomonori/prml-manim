"""Recompute all numbers shown in the film from the original 13 points."""
import json
from pathlib import Path
import numpy as np
from pca_model import *

assert N == 13 and D == 2
assert np.allclose(MEAN, [1.15, .35])
assert np.allclose(COV, CENTERED.T @ CENTERED / N)
assert np.allclose(COV @ EIGENVECTORS, EIGENVECTORS * EIGENVALUES)
assert np.allclose(EIGENVECTORS.T @ EIGENVECTORS, np.eye(D))
assert np.isclose(variance(0), 2.6288888888888895)
assert np.isclose(error(0), 1.3490944615384617)
assert np.isclose(variance(THETA), EIGENVALUES[0])
assert np.isclose(error(THETA), EIGENVALUES[1])
for angle in np.linspace(-np.pi / 2, np.pi / 2, 129):
    assert np.isclose(variance(angle) + error(angle), np.trace(COV), atol=1e-12)
for angle in np.linspace(0, np.pi / 2, 129):
    assert 0 <= variance(angle) <= 4.15
    assert 0 <= error(angle) <= 4.15
for m in range(D + 1):
    assert np.isclose(reconstruction_error(m), EIGENVALUES[m:].sum(), atol=1e-12)
assert np.allclose(reconstruction(2), POINTS)
whitened = CENTERED @ EIGENVECTORS / np.sqrt(EIGENVALUES)
assert np.allclose(whitened.mean(axis=0), [0, 0], atol=1e-12)
assert np.allclose(whitened.T @ whitened / N, np.eye(D), atol=1e-12)
assert np.all((POINTS >= [-1.95, -2.12]) & (POINTS <= [4.12, 2.27]))
assert np.all(np.abs(whitened) < 3)
assert np.all(np.abs(CENTERED @ EIGENVECTORS[:, 0]) < 3.2)
result = dict(count=N, mean=MEAN.tolist(), covariance=COV.tolist(),
              eigenvalues=EIGENVALUES.tolist(), first_component=EIGENVECTORS[:, 0].tolist(),
              angle_degrees=float(np.degrees(THETA)), horizontal=dict(variance=variance(0), error=error(0)),
              first_principal_component=dict(variance=variance(THETA), error=error(THETA)),
              reconstruction_errors={str(m): reconstruction_error(m) for m in range(3)},
              check='passed')
(Path(__file__).parent / 'numerical_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
