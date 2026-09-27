"""Independent identities, maximum-likelihood stationarity and asset checks."""
import itertools
import json
import re
from pathlib import Path
import numpy as np
from generative_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry

out={}
rng=np.random.default_rng(73);x=rng.normal(size=(1000,2))
w,b=linear_params()
p=posterior(x)[:,0]
out['gaussian_sigmoid_max_error']=float(np.max(np.abs(p-sigmoid(x@w+b))))
assert out['gaussian_sigmoid_max_error']<1e-12
for prior in [.2,.5,.8]:
 ww,bb=linear_params(prior=prior)
 assert np.allclose(w,ww)
 assert np.isclose(bb-b,np.log(prior/(1-prior)))
 assert np.allclose(posterior(x,priors=[prior,1-prior])[:,0],sigmoid(x@ww+bb))
# Scores after discarding class-independent quadratic terms yield the same softmax.
weights=np.linalg.solve(COV,MEANS.T).T
bias=-.5*np.einsum('ij,ij->i',MEANS,weights)+np.log(1/3)
out['softmax_common_term_error']=float(abs(posterior(x,MEANS)-softmax(x@weights.T+bias)).max())
assert out['softmax_common_term_error']<1e-12
for t in np.linspace(0,1,21):
 covs=covariances(t)
 assert all(np.linalg.eigvalsh(c).min()>0 for c in covs)
 paths=decision_paths(MEANS,covs)
 assert paths
 for line in paths:
  values=scores((line[:-1]+line[1:])/2,MEANS,covs)
  ordered=np.sort(values,axis=-1)
  assert np.max(ordered[:,-1]-ordered[:,-2])<.025
  assert np.all(np.abs(line[:,0])<=3.5+1e-8) and np.all(np.abs(line[:,1])<=2.5+1e-8)
out['covariance_states_checked']=21
means,cov,prior=fit();r=np.r_[RED_POINTS-means[0],BLUE_POINTS-means[1]]
assert np.allclose(cov,r.T@r/50)
assert np.allclose((RED_POINTS-means[0]).sum(0),0)
assert np.allclose((BLUE_POINTS-means[1]).sum(0),0)
assert np.isclose(prior,.6)
# Analytic MLE compared with likelihood under independent perturbations.
def likelihood(means,cov,prior):
 return sum(log_gaussian(a,m,cov).sum()+len(a)*np.log(p) for a,m,p in zip([RED_POINTS,BLUE_POINTS],means,[prior,1-prior]))
base=likelihood(means,cov,prior)
for _ in range(100):
 dm=rng.normal(scale=.02,size=(2,2));d=rng.normal(scale=.02,size=(2,2));cc=cov+d@d.T
 assert likelihood(means+dm,cc,prior+.02)<base
out['mle']=dict(prior=prior,means=means.tolist(),covariance=cov.tolist(),log_likelihood=float(base))
mc,cc,pc=fit(contaminated(1))
out['outlier']=dict(mean_shift=float(np.linalg.norm(mc[0]-means[0])),trace_before=float(np.trace(cov)),trace_after=float(np.trace(cc)))
assert np.trace(cc)>np.trace(cov)
# Enumerate all binary vectors. Check normalization and affine log scores independently.
total=np.zeros(2)
for bits in itertools.product([0,1],repeat=3):
 bits=np.array(bits)
 raw=np.prod(np.where(bits[None,:],BINARY_MU,1-BINARY_MU),axis=1)
 total+=raw
 a=bits@np.log(BINARY_MU/(1-BINARY_MU)).T+np.log1p(-BINARY_MU).sum(1)+np.log(.5)
 assert np.allclose(binary_scores(bits),a)
 assert np.allclose(softmax(a),raw/raw.sum())
assert np.allclose(total,1)
out['binary_normalization']=total.tolist()
out['binary_examples']={str(b):softmax(binary_scores(b)).tolist() for b in [[0,0,0],[1,0,0],[1,0,1]]}
# Eq.4.83–86 at s=1, a Poisson example (u(x)=x).
rate=np.array([2.,5.]);n=np.arange(8)
full=np.array([[v*np.log(l)-l-sum(np.log(np.arange(1,v+1)))+np.log(.5) for l in rate] for v in n])
linear=n[:,None]*np.log(rate)-rate+np.log(.5)
assert np.allclose(softmax(full),softmax(linear))
entries=json.loads(MANIFEST.read_text())['scenes']
assert len(entries)==len(SCENES)
assert all(valid_entry(s,e) for s,e in zip(SCENES,entries))
assert {p.stem for p in Path('assets/voicevox').glob('*.wav')}=={s['id'] for s in SCENES}
segments=[c for s in SCENES for b in s['beats'] for c in b['segments']]
assert not any(re.search('エックス|ミュー|シグマ|ラムダ|ダブリュー|エヌ|ディー|ピー、|シーケー',c['display']) for c in segments)
readings=json.loads(Path('reading_check.json').read_text())['sentences']
assert [(r['id'],r['display'],r['speech']) for r in readings]==[(c['id'],c['display'],c['speech']) for c in segments]
out['audio_scenes']=len(entries);out['sentences']=len(segments);out['wav_duration']=sum(e['duration'] for e in entries)
Path('numerical_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2));print('PASS: 8 groups of independent numerical / data checks')
