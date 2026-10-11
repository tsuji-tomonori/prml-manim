"""Numerical invariants and measured narration integrity for the actual scenes."""
import json
import re
from pathlib import Path
import numpy as np
from discriminant_model import *
from narration_content import SCENES
from make_voicevox_narration import valid_entry,MANIFEST,OUTPUT_DIR

def run():
    results={}
    # Normal projection and invariance to positive rescaling.
    u=direction(.65);b=-.4;foot=-b*u+.3*np.array([-u[1],u[0]])
    assert abs(u@foot+b)<1e-12
    for r in np.linspace(-1.1,1.8,50):
        x=foot+r*u
        for c in [1,1.5,2]:assert abs((c*u@x+c*b)/np.linalg.norm(c*u)-r)<1e-12
    results['distance_rescaling']='passed'
    # Convex clipping yields points that actually win the argmax.
    for bias in np.linspace(0,2,21):
        W=MULTI_W.copy();W[0,2]=bias
        for k in range(3):
            p=region(W,k)
            if len(p):assert np.min(scores(p,W)[:,k,None]-scores(p,W))>-1e-9
    results['argmax_regions']='passed (21 states)'
    # Actual least squares and its outside-range scores.
    max_res=0
    initial=least_squares(ls_data(0));c=initial[:,0]-initial[:,1]
    for a in np.linspace(0,1,41):
        groups=ls_data(a);W=least_squares(groups);X=augment(np.vstack(groups))
        T=np.repeat(np.eye(2),[len(g) for g in groups],axis=0)
        max_res=max(max_res,float(np.max(abs(X.T@(X@W-T)))))
        assert np.all(augment(groups[1][-3:])@c<0)
        assert np.max(abs(scores(np.array([[-2.2,-2.5],[2.2,-2.5]]),W).sum(1)-1))<1e-12
    results['least_squares_normal_residual']=max_res
    vals=scores(np.array([[-2.2,-2.5],[2.2,-2.5]]),W)
    assert vals.min()<0 and vals.max()>1
    results['least_squares_score_range']=[float(vals.min()),float(vals.max())]
    # The opening, scene 4, and ending use the identical white point and data states.
    white=np.array([0.,0.])
    opening=scores(white,least_squares(ls_data(0)))[0]
    ending=scores(white,least_squares(ls_data(1)))[0]
    assert np.argmax(opening)==1 and np.argmax(ending)==0
    assert np.allclose(opening,[.4919048,.5080952],atol=1e-7)
    assert np.allclose(ending,[.51180412,.48819588],atol=1e-7)
    assert np.all(augment(ls_data(1)[1][-3:])@c<0)
    results['white_point_scores']={'initial':opening.tolist(),'moved':ending.tolist()}
    pred=np.argmax(scores(np.vstack(BANDS),BAND_W),axis=1)
    assert not np.any(pred==1)
    results['middle_class_selected']=int(np.sum(pred==1))
    # Fisher optimum versus exhaustive angular sweep; exact LS relation.
    best=max(criterion(direction(t)) for t in np.linspace(-np.pi,np.pi,20001))
    assert criterion(FISHER)>=best-1e-9
    cosine=float(abs(unit(CODE_W[1:])@FISHER));assert abs(cosine-1)<1e-12
    assert abs(CODE_W[0]+CODE_W[1:]@FISH_MEAN)<1e-12
    results['fisher_J']=[criterion(MEAN_DIR),criterion(FISHER)]
    results['ls_fisher_absolute_cosine']=cosine
    assert np.linalg.matrix_rank(MSB)==2
    C=np.vstack(MULTI);st=(C-C.mean(0)).T@(C-C.mean(0))
    results['scatter_decomposition_error']=float(abs(st-MSW-MSB).max())
    assert results['scatter_decomposition_error']<1e-10
    results['multiclass_eigenvalues']=EIG.tolist()
    # Every update is on a truly wrong point; margin gain matches equation 4.56.
    for h in P_HISTORY:
        i=h['index'];phi=augment(P_X)[i];t=P_T[i]
        assert (1 if h['before']@phi>=0 else -1)!=t
        assert np.allclose(h['after'],h['before']+phi*t)
        assert np.isclose(t*(h['after']-h['before'])@phi,h['gain'])
    assert np.all(np.where(augment(P_X)@P_FINAL>=0,1,-1)==P_T)
    assert np.all(np.where(augment(P_X)@np.r_[-.05,P_FINAL[1:]]>=0,1,-1)==P_T)
    results['perceptron_updates']=len(P_HISTORY)
    # The aid uses centered residual pairs, not an uncentered second moment.
    residual_groups=[np.array([[2,1],[-2,-1]]),np.array([[1,-1],[-1,1]])]
    aid_sw=sum(g.T@g for g in residual_groups)
    assert all(np.allclose(g.mean(0),0) for g in residual_groups)
    assert np.array_equal(np.outer([2,1],[2,1]),[[4,2],[2,1]])
    assert np.array_equal(aid_sw,[[10,2],[2,4]])
    w=np.array([1,0])
    assert w@aid_sw@w == sum(np.sum((g@w)**2) for g in residual_groups) == 10
    results['aid_scatter']={'matrix':aid_sw.tolist(),'horizontal_squared_deviation_sum':10}
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES) and all(valid_entry(s,e) for s,e in zip(SCENES,entries))
    assert {p.stem for p in OUTPUT_DIR.glob('*.wav')}=={s['id'] for s in SCENES}
    for e in entries:
        cues=e['subtitle_cues'];assert all(a['end']<=b['start']+1e-8 for a,b in zip(cues,cues[1:]))
    sentences=[s for scene in SCENES for b in scene['beats'] for s in b['segments']]
    katakana=re.compile('エックス|ダブリュー|ファイ|ラムダ|ミュー|シグマ|ケーマイナス|エヌ割る')
    assert not any(katakana.search(s['display']) for s in sentences)
    results['audio_scenes']=len(entries);results['sentences']=len(sentences)
    results['audio_seconds']=sum(e['duration'] for e in entries)
    results['checks_passed']=9
    Path('validation_numerics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
