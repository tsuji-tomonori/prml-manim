"""Independent identities, analytic optima, and narration asset consistency."""
import itertools
import json
from pathlib import Path
import numpy as np
from information_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry

def main():
    results={}
    codes=P_CODES
    assert not any(b.startswith(a) for a,b in itertools.permutations(codes,2))
    assert not any(b.startswith(a) for a,b in itertools.permutations(Q_CODES,2))
    mistaken=mean_code_length(P,Q_CODES)
    assert np.isclose(mistaken,2.625) and mistaken>2
    assert np.isclose(mean_code_length(Q,Q_CODES),1.9)
    mean=mean_code_length(P,codes)
    assert np.isclose(mean,entropy(P,2)) and np.isclose(mean,1.75)
    results['coding_bits']=mean
    results['opening_code_bits']=mistaken
    results['ideal_cross_entropy_bits']=float(-P@np.log2(Q))
    assert np.isclose(results['ideal_cross_entropy_bits']-mean,kl(P,Q)/np.log(2))
    assert multiplicity(6,3)==len(list(itertools.combinations(range(6),3)))==20
    assert np.isclose(entropy(np.ones(30)/30),np.log(30))
    assert entropy(np.eye(30)[14])==0
    results['uniform30_nats']=entropy(np.ones(30)/30)
    for n in [4,8,16,32]:
        assert np.isclose(entropy(np.ones(n)/n),-np.log(1/n))
    x=np.linspace(-12,12,240001)
    for sigma in [.3,.6,1,1.3]:
        p=gaussian_pdf(x,sigma); positive=p>0
        assert abs(np.trapezoid(p,x)-1)<1e-10
        actual=-np.trapezoid(p[positive]*np.log(p[positive]),x[positive])
        assert np.isclose(actual,gaussian_entropy(sigma),atol=1e-9)
    results['gaussian_h_sigma1']=float(gaussian_entropy(1))
    results['uniform_same_variance_h']=float(np.log(2*np.sqrt(3)))
    for r in np.linspace(0,1,101):
        j=joint(r)
        assert np.allclose(j.sum(axis=0),.5) and np.allclose(j.sum(axis=1),.5)
        assert np.isclose(entropy(j),np.log(2)+conditional(r))
        assert np.isclose(mutual(r),np.log(2)-conditional(r))
    results['conditional_r06']=conditional(.6)
    results['mutual_r06']=mutual(.6)
    assert kl(P,Q)>0 and not np.isclose(kl(P,Q),kl(Q,P)) and kl(P,P)==0
    assert np.isinf(kl(P,[0,.5,.25,.25]))
    for t in np.linspace(0,1,101):
        q=(1-t)*P+t*Q
        assert np.isclose(-np.sum(P*np.log(q))-entropy(P),kl(P,q))
        assert kl(P,q)>=-1e-12
    results['kl_pq']=kl(P,Q);results['kl_qp']=kl(Q,P)
    for lam in np.linspace(0,1,101):
        a,b=.25,2
        assert -np.log(lam*a+(1-lam)*b)<=lam*(-np.log(a))+(1-lam)*(-np.log(b))+1e-12
    aid_p=np.array([.5,.5]);aid_q=np.array([.25,.75])
    weighted=aid_p*(aid_q/aid_p)
    assert np.allclose(weighted,[.25,.75]) and np.isclose(weighted.sum(),1)
    assert np.isclose(kl(aid_p,aid_q),-aid_p@np.log(aid_q/aid_p))
    results['ratio_aid_contributions']=weighted.tolist()
    results['ratio_aid_mean']=float(weighted.sum())
    assert DATA.sum()==14 and len(DATA)==20
    grid=np.linspace(.01,.99,981)
    optimum=grid[np.argmin([nll(t) for t in grid])]
    assert np.isclose(optimum,DATA.mean())
    results['mle_theta']=float(optimum);results['mle_nll']=nll(optimum)
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES)==10
    for s,e in zip(SCENES,entries):
        assert valid_entry(s,e),s['id']
        assert all(0<=b-a<.42 for a,b in zip(e['beat_speech_ends'],e['beat_durations']))
    assert {p.stem for p in OUTPUT_DIR.glob('*.wav')}=={s['id'] for s in SCENES}
    displays=[s['display'] for sc in SCENES for b in sc['beats'] for s in b['segments']]
    assert not any(word in s for s in displays for word in ['エックス','ワイ','シグマ','シータ','デルタ','ダブリュー','ラムダ','ミュー'])
    assert not any('\t' in s or '\r' in s for s in displays)
    results['scenes']=len(entries);results['sentences']=len(displays)
    results['audio_seconds']=sum(e['duration'] for e in entries)
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
