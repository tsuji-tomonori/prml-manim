"""Independent checks of optimization, curvature, projection and prediction."""
import json
import re
from pathlib import Path
import numpy as np
from bayesian_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry

report = {}
g, h = derivatives(MAP)
report['map_gradient_norm'] = float(np.linalg.norm(g))
assert report['map_gradient_norm'] < 1e-10
eps = 1e-5
fd = np.column_stack([(derivatives(MAP+eps*e)[0]-derivatives(MAP-eps*e)[0])/(2*eps)
                      for e in np.eye(2)])
report['hessian_finite_difference_error'] = float(np.max(np.abs(fd-h)))
assert report['hessian_finite_difference_error'] < 1e-8
assert np.all(np.linalg.eigvalsh(COV)>0)
assert np.allclose(COV@h, np.eye(2), atol=1e-12)
report['map'] = MAP.tolist()
report['covariance'] = COV.tolist()
report['scalar_density_integral'] = float(np.trapezoid(scalar_density(INTEGRATION_GRID),INTEGRATION_GRID))
assert abs(report['scalar_density_integral']-1)<1e-10
scalar_fd = (scalar_energy(SW+eps)-2*scalar_energy(SW)+scalar_energy(SW-eps))/eps**2
report['scalar_curvature_error'] = float(abs(scalar_fd-1/SV))
assert report['scalar_curvature_error'] < 1e-4

# Compare direct 2D Gaussian integration to the projected 1D integration.
nodes, weights = np.polynomial.hermite.hermgauss(50)
zz = np.array(np.meshgrid(nodes,nodes)).reshape(2,-1).T*np.sqrt(2)
ww = MAP + zz@np.linalg.cholesky(COV).T
weight2 = np.outer(weights,weights).ravel()/np.pi
points = np.array([-.8,0,1,2])
direct = sigmoid(ww@design(points).T).T@weight2
projected = predictive_integral(*stats(points))
report['projection_integral_error'] = float(np.max(np.abs(direct-projected)))
assert report['projection_integral_error'] < 1e-8

# Displayed Laplace energy curves stay within the chosen vertical extent.
energy_x = np.linspace(-.6,2.8,1001)
assert scalar_energy(energy_x).max() < 6
assert (.5*(energy_x-SW)**2/SV).max() < 6
# Exact Gaussian-CDF identity (4.152), independently integrated on a grid.
z = np.linspace(-14,14,40001)
lam = np.sqrt(np.pi/8)
probit_numeric = np.trapezoid(cdf(lam*z)*normal(z,1.2,1.7),z)
probit_exact = cdf(1.2/np.sqrt(lam**-2+1.7))
report['probit_identity_error'] = float(abs(probit_numeric-probit_exact))
assert report['probit_identity_error'] < 1e-10

# A separate dense trapezoidal integral checks Gauss-Hermite quadrature.
xx = np.linspace(-35,35,100001)
vals = np.linspace(0,9,37)
truth = np.array([sigmoid(2) if v==0 else np.trapezoid(sigmoid(xx)*normal(xx,2,v),xx) for v in vals])
gh = predictive_integral(2,vals)
report['quadrature_error'] = float(np.max(abs(truth-gh)))
assert report['quadrature_error'] < 1e-7
report['kappa_max_error_mu2_var0to9'] = float(np.max(abs(truth-sigmoid(2*kappa(vals)))))
report['probability_at_mu2_var9'] = float(predictive_integral(2,9))
report['map_probability_mu2'] = float(sigmoid(2))
assert np.all(np.diff(gh)<0)
assert np.allclose(predictive_integral(0,vals),.5)
assert np.allclose(predictive_integral(2,vals)+predictive_integral(-2,vals),1)
b = -MAP[0]/MAP[1]
report['boundary_x'] = float(b)
assert abs(float(predict([b])[0])-.5)<1e-12
grid = np.linspace(-1.7,3,20001)
report['threshold08_map_x'] = float((np.log(4)-MAP[0])/MAP[1])
report['threshold08_bayes_x'] = float(np.interp(.8,predict(grid),grid))
assert report['threshold08_bayes_x'] > report['threshold08_map_x']

# The precision card is an illustrative observation, independent of the fitted data.
phi_example = np.array([1.,2.])
outer_example = np.outer(phi_example,phi_example)
assert np.array_equal(outer_example,[[1,2],[2,4]])
precision_example = .5*(1-.5)*outer_example
assert np.array_equal(precision_example,[[.25,.5],[.5,1]])
assert np.linalg.eigvalsh(np.eye(2)+precision_example).min()>0
assert np.allclose(np.linalg.inv(np.eye(2)+precision_example)@(np.eye(2)+precision_example),np.eye(2))
report['aid_observation_precision'] = precision_example.tolist()
for a in [1.,4.]:
 z=np.linspace(-10,10,20001)
 assert abs(np.trapezoid(z*z*normal(z,0,1/a),z)-1/a)<1e-10
report['recap_curvature_variance'] = [[1,1],[4,.25]]

entries = json.loads(MANIFEST.read_text())['scenes']
assert all(valid_entry(s,e) for s,e in zip(SCENES,entries))
assert len(entries)==len(SCENES)
assert {p.stem for p in Path('assets/voicevox').glob('*.wav')}=={s['id'] for s in SCENES}
segments = [seg for s in SCENES for b in s['beats'] for seg in b['segments']]
reading = json.loads(Path('reading_check.json').read_text())['sentences']
assert [(s['id'],s['speech']) for s in segments]==[(s['id'],s['speech']) for s in reading]
assert not any(re.search('エックス|ラムダ|ミュー|シグマ|ファイ|ダブリュー|カッパ',s['display']) for s in segments)
assert all(s['display'] in Path('narration_script.md').read_text() for s in segments)
report['sentences'] = len(segments)
report['symbol_captions'] = sum('$' in s['display'] for s in segments)
report['audio_duration'] = sum(s['duration'] for s in entries)
print(json.dumps(report,ensure_ascii=False,indent=2))
Path('numerical_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
