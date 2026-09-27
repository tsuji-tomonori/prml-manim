"""Independent identities, sequential update, variance floor, kernel, and audio checks."""
import json
import re
from pathlib import Path
import numpy as np
from bayesian_model import *
from narration_content import SCENES
from make_voicevox_narration import valid_entry

# Sequential Gaussian conditioning (rank-one update) versus batch precision solve.
m=np.zeros(2);s=np.eye(2)/ALPHA
max_error=0.
for i in range(len(X)):
    f=phi(X[i])[0];gain=s@f/(1/BETA+f@s@f)
    m=m+gain*(T[i]-f@m);s=s-np.outer(gain,f@s)
    mb,sb=posterior(i+1)
    max_error=max(max_error,float(np.max(abs(m-mb))),float(np.max(abs(s-sb))))
assert max_error<1e-12
print('sequential_batch_max_error',max_error)

for kind,count,grid in [('line',20,np.linspace(-1,1,201)),('rbf',25,np.linspace(0,1,201))]:
    prev=predict(grid,0,kind=kind)[2]
    for n in range(1,count+1):
        mu,latent,var=predict(grid,n,kind=kind)
        assert np.all(var<=prev+1e-12) and np.min(var)>=1/BETA
        prev=var
    print(kind,'variance_min_max',var.min(),var.max())

f=phi(X);m,s=posterior(20)
ridge=np.linalg.lstsq(np.vstack([f,np.sqrt(ALPHA/BETA)*np.eye(2)]),np.r_[T,[0,0]],rcond=None)[0]
assert np.allclose(m,ridge,atol=1e-12)
print('MAP_ridge_error',np.max(abs(m-ridge)), 'posterior_mean',m.tolist())

u=np.linspace(0,1,201);k=kernel(u,XR);mu=predict(u,25,kind='rbf')[0]
err=np.max(abs(k@TR-mu));assert err<1e-12
_,s=posterior(25,kind='rbf');cov=phi(u,'rbf')@s@phi(u,'rbf').T
assert np.allclose(kernel(u,u)/BETA,cov,atol=1e-12)
vals,vecs=np.linalg.eigh(s);sqrt=(vecs*np.sqrt(vals))@vecs.T
psi=np.sqrt(BETA)*phi(u,'rbf')@sqrt
assert np.allclose(kernel(u,u),psi@psi.T)
print('kernel_mean_error',err,'kernel_weight_min',k.min(),'weight_sum_range',k.sum(1).min(),k.sum(1).max())
mu,l,v=predict([-10,10],25,kind='rbf');assert np.allclose(mu,0) and np.allclose(v,.04)
print('far_field_mean_variance',mu.tolist(),v.tolist())

# A constant unpenalized basis reproduces constants; penalizing it breaks that identity.
f=phi(X);x=phi([.3]);h=x@np.linalg.solve(f.T@f,f.T)
assert np.allclose(h.sum(),1)
hp=x@np.linalg.solve(f.T@f+ALPHA/BETA*np.eye(2),f.T)
assert not np.isclose(hp.sum(),1)
print('constant_weights_unregularized_regularized',h.sum(),hp.sum())

entries=json.loads(Path('assets/voicevox/manifest.json').read_text())['scenes']
assert len(entries)==len(SCENES)==9
assert all(valid_entry(s,e) for s,e in zip(SCENES,entries))
expected={s['id']+'.wav' for s in SCENES}
assert {p.name for p in Path('assets/voicevox').glob('*.wav')}==expected
for s in SCENES:
    for b in s['beats']:
        for t in b['segments']:
            assert not re.search('エックス|ラムダ|ミュー|シグマ|アルファ|ベータ|ティー|エムエヌ|エスエヌ',t['display']),t
print('audio_hashes_and_display',len(entries),'scenes',sum(len(b['segments']) for s in SCENES for b in s['beats']),'sentences')
print('PASS: 7 verification groups')
