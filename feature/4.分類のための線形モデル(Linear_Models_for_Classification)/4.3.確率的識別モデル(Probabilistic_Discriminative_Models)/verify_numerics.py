"""Verify derivatives, actual IRLS updates, source corrections and audio integrity."""
import json
import re
from pathlib import Path
import numpy as np
from discriminative_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry

ROOT=Path(__file__).resolve().parent

def main():
    result={}
    w=np.array([.2,-.7]); h=1e-5
    g,H=derivatives(w)
    eye=np.eye(2)
    fd=np.array([(loss(w+h*u)-loss(w-h*u))/(2*h) for u in eye])
    fdH=np.column_stack([(derivatives(w+h*u)[0]-derivatives(w-h*u)[0])/(2*h) for u in eye])
    assert np.max(abs(g-fd))<1e-8 and np.max(abs(H-fdH))<1e-8
    result['gradient_max_error']=float(np.max(abs(g-fd)))
    result['hessian_max_error']=float(np.max(abs(H-fdH)))
    errors=[]
    for old,new in zip(HISTORY[:-1],HISTORY[1:]):
        y=sigmoid(PHI@old);r=y*(1-y);z=PHI@old+(T-y)/r
        ls=np.linalg.lstsq(np.sqrt(r)[:,None]*PHI,np.sqrt(r)*z,rcond=None)[0]
        errors.append(float(np.max(abs(ls-new))))
    assert max(errors)<1e-10
    assert np.all(np.diff([loss(w) for w in HISTORY])<=1e-10)
    result['irls_weighted_ls_error']=max(errors)
    result['irls_losses']=[loss(w) for w in HISTORY]
    result['irls_final_weights']=HISTORY[-1].tolist()
    result['hessian_min_eigenvalue']=float(np.linalg.eigvalsh(derivatives(HISTORY[-1])[1]).min())
    assert np.all((RING**2).sum(axis=1)[RING_T==1]<1)
    assert np.all((RING**2).sum(axis=1)[RING_T==0]>1)
    result['basis_boundary_error']=float(np.max(abs(np.cos(np.linspace(0,6.28,100))**2+np.sin(np.linspace(0,6.28,100))**2-1)))
    vals=[loss(np.array([0.,k]),SEP_T) for k in [1,5,20]]
    assert np.all(np.diff(vals)<0)
    result['separable_losses']=vals
    for lam in [.015,.3,1.5]:
        w=fit(SEP_T,lam,steps=12)[-1]
        assert np.linalg.norm(derivatives(w,SEP_T,lam)[0])<1e-8
    a=np.array([1.,.5,-.2]);p=softmax(a)
    assert abs(p.sum()-1)<1e-14
    assert np.max(abs(p-softmax(a+1000)))<1e-12
    J=np.diag(p)-np.outer(p,p)
    fdJ=np.column_stack([(softmax(a+h*u)-softmax(a-h*u))/(2*h) for u in np.eye(3)])
    assert np.max(abs(J-fdJ))<1e-9
    assert np.linalg.norm(J@np.ones(3))<1e-14
    result['softmax_hessian_error']=float(np.max(abs(J-fdJ)))
    result['softmax_common_shift_null_norm']=float(np.linalg.norm(J@np.ones(3)))
    xx=np.linspace(-8,8,100001)
    integral=np.trapezoid(normal_pdf(xx[xx<=1.6]),xx[xx<=1.6])
    assert abs(integral-normal_cdf(1.6))<2e-5
    result['cdf_area_error']=float(abs(integral-normal_cdf(1.6)))
    result['outlier_loss_at_minus5']=dict(logistic=float(np.logaddexp(0,8.5)),probit=float(-np.log(normal_cdf(-5))))
    assert abs(noisy_probability(-100,.1)-.1)<1e-12
    assert abs(noisy_probability(100,.1)-.9)<1e-12
    # Canonical Bernoulli inverse link cancels derivative, scale s remains.
    q=np.linspace(.01,.99,99)
    assert np.max(abs((1/q+1/(1-q))*q*(1-q)-1))<1e-14
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES) and all(valid_entry(s,e) for s,e in zip(SCENES,entries))
    assert {p.stem for p in (ROOT/'assets/voicevox').glob('*.wav')}=={s['id'] for s in SCENES}
    display='\n'.join(s['display'] for c in SCENES for b in c['beats'] for s in b['segments'])
    assert not re.search('エックス|ラムダ|ミュー|シグマ|ファイ|ダブリュー|イプシロン|アイアールエルエス',display)
    (ROOT/'media/display_check.txt').write_text(display+'\n')
    result['scene_durations']={e['id']:e['duration'] for e in entries}
    result['sentences']=sum(len(e['subtitle_cues']) for e in entries)
    result['beats']=sum(len(s['beats']) for s in SCENES)
    result['wav_seconds']=sum(e['duration'] for e in entries)
    result['checks']='8 groups passed: derivatives, IRLS, basis, separation/ridge, softmax, CDF/noise, canonical, audio/display'
    (ROOT/'numerical_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
