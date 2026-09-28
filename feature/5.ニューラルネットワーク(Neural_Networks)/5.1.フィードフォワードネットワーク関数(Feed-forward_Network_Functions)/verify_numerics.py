"""Numerical and audio invariants for the actual visual experiments."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
import numpy as np
from network_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--captions',action='store_true');args=parser.parse_args()
    results={}
    w=np.array([3,3,1,1.2,-1.2,0,.65,-.65,0,0.])
    np.testing.assert_allclose(network(GRID,w),bump(GRID),atol=1e-15)
    results['bump_network_error']=float(np.max(abs(network(GRID,w)-bump(GRID))))
    variants=[]
    for signs in itertools.product([-1,1],repeat=2):
        for order in [(0,1),(1,0)]:
            v=w.copy()
            for j in range(2):
                for base in [0,3,6]:v[base+j]*=signs[j]
            for base in [0,3,6]:v[base:base+2]=v[base+np.array(order)]
            variants.append(v)
    errors=[float(np.max(abs(network(GRID,v)-network(GRID,w)))) for v in variants]
    assert len({tuple(v) for v in variants})==8 and max(errors)<1e-14
    results['symmetry']={'distinct_vectors':8,'max_error':max(errors)}
    for a in np.linspace(-5,5,31):
        p=softmax([a,.7,-.6]);assert abs(p.sum()-1)<1e-14
        np.testing.assert_allclose(p,softmax(np.array([a,.7,-.6])+100),atol=1e-14)
    assert sigmoid(0)==.5
    results['output_checks']='sigmoid(0)=0.5; softmax normalization and shift invariance: 31 cases'
    bnderr=0.
    for weight in np.linspace(.7,2,11):
        points=decision_boundary(weight)
        bnderr=max(bnderr,float(np.max(abs(sigmoid(class_score(points,weight))-.5))))
        xx=np.linspace(-.3,.3,11); yy=np.arctanh(.5)-.3-weight*xx
        np.testing.assert_allclose(class_hidden(np.c_[xx,yy],weight)[:,0],.5,atol=1e-14)
    assert bnderr<1e-12
    results['boundary_max_probability_error']=bnderr
    params=np.array([.65,1.5,.6,.8,.2,-.15])
    a,v,b,c=1.5,.6,.2,-.15
    np.testing.assert_allclose(v*(a*GRID+b)+c,(v*a)*GRID+(v*b+c),atol=1e-15)
    results['affine_composition']='passed on 401 inputs'
    fits=fitted_weights()
    results['fits']=[]
    for i,weights in enumerate(fits):
        err=float(np.sqrt(np.mean((network(X,weights)-targets(i,X))**2)))
        assert np.isfinite(err) and err<[.01,.01,.03,.1][i]
        assert np.max(abs(network(GRID,weights)))<1.3
        results['fits'].append(dict(target=i,rms=err,max_error=float(np.max(abs(network(GRID,weights)-targets(i,GRID))))))
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES)
    for scene,entry in zip(SCENES,entries):
        assert valid_entry(scene,entry),scene['id']
    assert {p.name for p in OUTPUT_DIR.glob('*.wav')}=={e['id']+'.wav' for e in entries}
    segments=[s for scene in SCENES for b in scene['beats'] for s in b['segments']]
    readings=json.loads((ROOT/'reading_check.json').read_text())['sentences']
    assert [(s['id'],s['display'],s['speech']) for s in segments]==[(s['id'],s['display'],s['speech']) for s in readings]
    forbidden=r'エックス|ファイ|ラムダ|ミュー|シグマ|ダブリュー|ハイパボリック|タンジェント'
    assert not any(re.search(forbidden,s['display']) for s in segments)
    assert not any('\t' in s['display'] for s in segments)
    results['audio']={'scenes':len(entries),'beats':sum(len(s['beats']) for s in SCENES),'sentences':len(segments),'duration':sum(e['duration'] for e in entries)}
    (ROOT/'media').mkdir(exist_ok=True)
    (ROOT/'media/display_check.txt').write_text('\n'.join(s['display'] for s in segments)+'\n')
    if args.captions:
        from manim import config
        from video_support import caption_mobject
        Path(config.media_dir,'Tex').mkdir(parents=True,exist_ok=True)
        boxes=[caption_mobject(s['display']) for s in segments]
        results['captions']={'count':len(boxes),'max_width':max(m.width for m in boxes),'max_height':max(m.height for m in boxes)}
    (ROOT/'numerical_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(results,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
