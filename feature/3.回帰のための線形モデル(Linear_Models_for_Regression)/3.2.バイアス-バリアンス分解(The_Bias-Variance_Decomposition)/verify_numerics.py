"""Independent checks of ridge solutions, decomposition, assets and plotting range."""
import json
import re
from pathlib import Path
import numpy as np
from bias_variance_model import *
from make_voicevox_narration import valid_entry, MANIFEST
from narration_content import SCENES

errors=[]
for lam in [-3,BEST,3]:
    d=experiment(lam)
    direct=np.linalg.solve(GRAM+np.exp(lam)*np.eye(25),RHS[...,None])[...,0]
    np.testing.assert_allclose(d['weights'],direct,atol=2e-12)
    predictions=direct@EVAL_PHI.T
    mean=predictions.mean(axis=0)
    b=np.mean((mean-truth(EVAL_X))**2)
    v=np.mean((predictions-mean)**2)
    np.testing.assert_allclose(np.mean((predictions-truth(EVAL_X))**2,axis=0),
        (mean-truth(EVAL_X))**2+np.mean((predictions-mean)**2,axis=0),atol=2e-14)
    lhs=np.mean((predictions-truth(EVAL_X))**2)
    errors.append(abs(lhs-b-v))
    np.testing.assert_allclose([b,v],[d['bias2'],d['variance']],atol=2e-14)
    np.testing.assert_allclose(lhs,b+v,atol=2e-14)
    test=np.mean((direct@TEST_PHI.T-TEST_T)**2)
    np.testing.assert_allclose(test,d['test'],atol=2e-14)
    assert np.max(np.abs(d['curves'][:20]))<1.75
# Numerical range throughout lambda animation, not just the three endpoints.
span=max(np.max(np.abs(experiment(float(l))['curves'][:20])) for l in np.linspace(-3,3,121))
assert span<1.75
assert np.max(np.abs(T[:8]))<1.75
assert np.all(METRICS[:,:2]>=0)
assert -3<BEST<3
m=json.loads(MANIFEST.read_text())
assert len(m['scenes'])==len(SCENES)==8
for story,entry in zip(SCENES,m['scenes']): assert valid_entry(story,entry)
assert {p.stem for p in MANIFEST.parent.glob('*.wav')}=={s['id'] for s in SCENES}
subs='\n'.join(s['display'] for sc in SCENES for b in sc['beats'] for s in b['segments'])
assert not re.search('エックス|ラムダ|ミュー|シグマ|ワイバー|ティー|エム',subs)
assert not any(ord(c)<32 and c!='\n' for c in subs)
print(json.dumps(dict(checks=['ridge solve','pointwise/integrated identity','test MSE','range','manifest/hash','caption symbols'],
max_decomposition_error=max(errors),curve_absolute_max=float(span),best_log_lambda=BEST,
noise=SIGMA**2,wav_seconds=sum(s['duration'] for s in m['scenes']),sentences=len(subs.splitlines())),indent=2))
