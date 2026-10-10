"""Check independent mathematical identities, audio integrity, and caption safety."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from basis_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--captions',action='store_true')
    args=parser.parse_args()
    phi=design(X)
    assert np.linalg.matrix_rank(phi)==4
    assert np.max(abs(phi.T@(phi@BEST-T)))<1e-12
    np.testing.assert_allclose(BEST,np.linalg.solve(phi.T@phi,phi.T@T),atol=1e-12)
    initial=np.array([0,.3,.2,.3])
    assert np.linalg.norm(phi@BEST-T)<np.linalg.norm(phi@initial-T)
    print('PASS least squares; RMSE=',np.sqrt(np.mean((phi@BEST-T)**2)))
    assert 5**10==9765625
    assert GRID.shape==(81,2) and LOCAL.shape==(12,2)
    assert np.all(np.linalg.norm(LOCAL[:,None]-DATA[None,:],axis=-1).min(axis=1)==0)
    assert DATA.shape==(72,2) and int(NEAR.sum())==17
    grid_dist=np.linalg.norm(DATA[:,None,:]-GRID[None,:,:],axis=2).min(axis=1)
    local_dist=np.linalg.norm(DATA[:,None,:]-LOCAL[None,:,:],axis=2).min(axis=1)
    assert np.all(grid_dist<.18) and np.all(local_dist<.18)
    assert np.all(np.linalg.norm(GRID[:,None,:]-DATA[None,:,:],axis=2).min(axis=1)[~NEAR]>=.18)
    assert np.max(radial(LOCAL[4]))==1
    print('PASS same 72 points covered by 81 and 12 centers at radius .18; max distances=',
          grid_dist.max(),local_dist.max(),'; near grid centers=',int(NEAR.sum()))
    direction_invariant=np.array([[u,-u] for u in np.linspace(-.8,.8,101)])
    np.testing.assert_allclose(response(direction_invariant),.5,atol=1e-15)
    assert rmse(np.pi/4)<1e-14 and rmse(0)>.1
    assert np.all(np.diff([rmse(a) for a in np.linspace(0,np.pi/4,101)])<0)
    print('PASS relevant direction; RMSE horizontal=',rmse(0),'aligned=',rmse(np.pi/4))
    # Full visible domains are checked, with no clipping of any curve.
    u=np.linspace(-1,1,2001)
    for weights in [initial,[0,1.2,.2,.3],[0,1.2,.8,.3],BEST]:
        y=design(u)@weights
        assert y.min()>=0 and y.max()<1.5
    assert np.max(abs(DATA))<1
    assert np.all((TARGET>=0)&(TARGET<=1))
    print('PASS displayed domains')
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES)
    for s,e in zip(SCENES,entries):
        assert valid_entry(s,e)
        assert abs(sum(e['beat_durations'])-e['duration'])<1e-8
        assert all(c['end']>c['start'] for c in e['subtitle_cues'])
    assert {v.name for v in OUTPUT_DIR.glob('scene*.wav')}=={s['id']+'.wav' for s in SCENES}
    display='\n'.join(seg['display'] for s in SCENES for b in s['beats'] for seg in b['segments'])
    assert not re.search('エックス|ラムダ|ミュー|シグマ|ファイ|ダブリュー|ディー|エイチ|ジェイ|エージェイ|ビージェイ',display)
    rows=json.loads((Path(__file__).parent/'reading_check.json').read_text())['sentences']
    assert [(r['id'],r['display'],r['speech']) for r in rows]==[
        (c['id'],c['display'],c['speech']) for s in SCENES for b in s['beats'] for c in b['segments']]
    print('PASS audio / script / reading audit;',len(rows),'sentences;',sum(e['duration'] for e in entries),'seconds')
    if args.captions:
        from manim import config
        config.media_dir='/tmp/prml36-caption-check'
        from caption_layout import caption_mobject,jp
        sizes=[(caption_mobject(r['display']).width,caption_mobject(r['display']).height) for r in rows]
        assert all(jp(s['title'],34).width<13 for s in SCENES)
        print('PASS captions; max width/height=',np.max(sizes,axis=0))

if __name__=='__main__':main()
