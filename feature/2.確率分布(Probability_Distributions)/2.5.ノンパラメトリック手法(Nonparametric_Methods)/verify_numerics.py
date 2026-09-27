"""Numerical identities, neighbourhood membership, audio and caption checks."""
import json
import math
import re
from pathlib import Path
import numpy as np
from nonparametric_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry

def main():
    results={}
    assert 0<SAMPLES.min()<SAMPLES.max()<1
    integral=lambda y,x:float(np.trapezoid(y,x))
    x=np.linspace(-3,4,30001)
    areas=[integral(kde(x,h),x) for h in [.012,.06,.25]]
    assert np.allclose(areas,1,atol=1e-10)
    for w in np.linspace(.025,.5,50):
        for offset in [0,.02,.04]:
            v,e,c=histogram(w,offset)
            assert c.sum()==N and abs(np.sum(v*np.diff(e))-1)<1e-12
    results['normalization']={'kde_areas':areas,'histogram_configurations':150}
    for h in [.03,.12,.25]:
        for q in np.linspace(.1,.9,101):
            assert abs(box_kde(q,h)-local_count(q,h)/(N*h))<1e-12
    results['box_window_equal']=303
    # Radius is the exact kth order statistic, never floored or clipped.
    for k in range(3,26):
        for q in np.linspace(.18,.88,61):
            r=knn_radius(q,k)
            assert np.count_nonzero(np.abs(SAMPLES-q)<=r+1e-12)==k
            assert abs(knn_density(q,k)*2*r-k/N)<1e-12
    assert np.isinf(knn_density(SAMPLES[0],1))
    ratios=[float(knn_density(q,7)/(7/(2*N*abs(q)))) for q in [1e3,1e5,1e7]]
    assert abs(ratios[-1]-1)<1e-6
    results['knn']={'membership_cases':23*61,'peaks':{str(k):float(knn_density(GRID,k).max()) for k in [3,7,25]},'tail_ratios':ratios}
    for n in [20,200]:
        p=np.array([math.comb(n,k)*.35**k*.65**(n-k) for k in range(n+1)])
        z=np.arange(n+1)/n
        assert abs(p.sum()-1)<1e-12
        assert abs(p@z-.35)<1e-12
        assert abs(p@((z-.35)**2)-.35*.65/n)<1e-12
    results['binomial_variance']={str(n):.35*.65/n for n in [20,200]}
    assert np.allclose(posterior(QUERY,5),[.6,.4])
    for k in [1,5,11]:
        ids,r=neighbours(QUERY,k)
        counts=np.bincount(LABELS[ids],minlength=2)
        nk=np.bincount(LABELS,minlength=2);volume=np.pi*r*r
        bayes=(counts/(nk*volume))*(nk/len(POINTS))/(k/(len(POINTS)*volume))
        assert np.allclose(bayes,posterior(QUERY,k))
    maps={k:classify_grid(k) for k in [1,5,11]}
    transitions={str(k):int(np.count_nonzero(v[:,1:]!=v[:,:-1])+np.count_nonzero(v[1:,:]!=v[:-1,:])) for k,v in maps.items()}
    assert transitions['1']>transitions['5']>transitions['11']
    results['classification']={'initial_posterior':posterior(QUERY,5).tolist(),'grid':list(maps[1].shape),'adjacent_label_changes':transitions}
    manifest=json.loads(MANIFEST.read_text())['scenes']
    assert len(manifest)==len(SCENES)
    assert all(valid_entry(s,e) for s,e in zip(SCENES,manifest))
    assert {p.stem for p in OUTPUT_DIR.glob('*.wav')}=={s['id'] for s in SCENES}
    displays=[q['display'] for s in SCENES for b in s['beats'] for q in b['segments']]
    assert not any(re.search('エックス|ラムダ|ミュー|シグマ|エヌ|エイチ|デルタ|ケー|ブイ|ディー',s) for s in displays)
    results['audio']={'scenes':len(manifest),'sentences':len(displays),'total_seconds':sum(s['duration'] for s in manifest)}
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
