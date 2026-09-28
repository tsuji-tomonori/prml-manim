"""Independent finite differences, integrals, inverse kinematics, and asset checks."""
import json
import re
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from mdn_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry


def main():
    model=load_model();w=model['weights'][-1]
    result={}
    # Backprop chain, including the experimental sigma floor, against central differences.
    x,t=data(n=17);v=model['weights'][12];_,g=loss_grad(v,x,t)
    indices=np.linspace(0,len(v)-1,40,dtype=int); errors=[]
    for i in indices:
        d=np.zeros_like(v); d[i]=1e-6
        fd=(loss_grad(v+d,x,t)[0]-loss_grad(v-d,x,t)[0])/2e-6
        errors.append(abs(fd-g[i]))
    assert max(errors)<1e-6
    result['gradient_max_absolute_error']=max(errors)
    # Check the corrected original log-width derivative independently of the network floor.
    pi=np.array([.2,.5,.3]);mu=np.array([.1,.5,.9]);sigma=np.array([.15,.12,.2]);target=.38
    gamma=responsibility(target,pi,mu,sigma)
    analytic=gamma*(1-((target-mu)/sigma)**2)
    fd=[]
    for k in range(3):
        d=np.zeros(3);d[k]=1e-6
        fd.append((-np.log(density(target,pi,mu,sigma*np.exp(d)))+np.log(density(target,pi,mu,sigma*np.exp(-d))))/2e-6)
    result['corrected_5_157_max_error']=float(np.max(abs(analytic-fd)))
    assert result['corrected_5_157_max_error']<1e-7
    grid=np.linspace(-2,3,40001);integral_errors=[];moment_errors=[]
    for z in [.12,.35,.5,.65,.88]:
        p,m,s=(a[0] for a in parameters(w,[z]));pdf=density(grid,p,m,s)
        mean=p@m;variance=p@(s*s+(m-mean)**2)
        integral_errors.append(abs(trapezoid(pdf,grid)-1))
        moment_errors.extend([abs(trapezoid(grid*pdf,grid)-mean),abs(trapezoid((grid-mean)**2*pdf,grid)-variance)])
        assert abs(p.sum()-1)<1e-12 and np.all(s>FLOOR)
    result['normalization_max_error']=max(integral_errors);result['moment_max_error']=max(moment_errors)
    assert max(integral_errors+moment_errors)<1e-8
    modes=[]
    for z in [.12,.5,.88]:
        p,m,s=(a[0] for a in parameters(w,[z]));pdf=density(grid,p,m,s)
        modes.append(len(find_peaks(pdf,prominence=pdf.max()*.03)[0]))
    assert modes==[1,3,1];result['slice_mode_counts']=modes
    a=np.arccos(.6)
    end1=robot(a,-2*a)[1];end2=robot(-a,2*a)[1];meanend=robot(0,0)[1]
    assert np.allclose(end1,[1.2,0]) and np.allclose(end1,end2)
    result['averaged_arm_target_error']=float(np.linalg.norm(meanend-end1))
    p=np.array([.65,.35]);m=np.array([.04,.96]);s=np.array([.22,.055]);mode=grid[np.argmax(density(grid,p,m,s))]
    result['mode_counterexample']={'largest_weight_center':float(m[np.argmax(p)]),'numeric_mode':float(mode)}
    assert abs(mode-m[np.argmax(p)])>.8
    manifest=json.loads(MANIFEST.read_text()); entries={e['id']:e for e in manifest['scenes']}
    assert all(valid_entry(s,entries[s['id']]) for s in SCENES)
    assert {p.stem for p in (ROOT/'assets/voicevox').glob('*.wav')}==set(entries)
    segments=[c for s in SCENES for b in s['beats'] for c in b['segments']]
    for c in segments:
        assert not re.search('エックス|ミュー|シグマ|ガンマ|ラムダ|パイ、ケー',c['display'])
        assert '$' not in c['speech'] and '\\' not in c['speech']
    result['audio_scenes']=len(entries);result['sentences']=len(segments)
    if '--captions' in __import__('sys').argv:
        from video_support import caption_mobject
        dimensions=[(caption_mobject(c['display']).width,caption_mobject(c['display']).height) for c in segments]
        result['caption_max_width']=max(w for w,h in dimensions);result['caption_max_height']=max(h for w,h in dimensions)
    (ROOT/'numerical_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
