"""Independent numerical and narration integrity checks for the published example."""
import json
import math
import re
from pathlib import Path
import numpy as np
from laplace_model import *
from narration_content import SCENES
from make_voicevox_narration import valid_entry


def main():
    h=1e-4
    assert abs(grad(Z0))<1e-12
    fd=-(logf(Z0+h)-2*logf(Z0)+logf(Z0-h))/h**2
    assert abs(fd-A)<1e-6
    assert A>0
    # Independent dense-grid mode and integration refinement.
    g=np.linspace(-10,10,200001)
    assert abs(g[np.argmax(f(g))]-Z0)<1.1e-4
    assert abs(np.trapezoid(f(g),g)-Z)<1e-8
    assert abs(np.trapezoid(gaussian(g),g)-1)<1e-10
    assert abs(np.trapezoid(local(g),g)-ZL)<1e-10
    assert abs(np.trapezoid((g-Z0)**2*gaussian(g),g)-1/A)<1e-9
    # Local fit matches log value, slope, curvature; normalized heights need not.
    assert abs(local(Z0)-f(Z0))<1e-12
    assert abs(pdf(Z0)-gaussian(Z0))>.01
    for angle in [0,.65,1]:
        mat=matrix(5,1,angle);cov=np.linalg.inv(mat)
        assert np.allclose(mat@cov,np.eye(2))
        assert np.allclose(np.linalg.eigvalsh(mat),[1,5])
        assert abs(np.linalg.det(mat)-5)<1e-10
        v=rotation(angle)[:,0]/np.sqrt(5)
        assert abs(v@mat@v-1)<1e-12
    for sigma,w in [(.8,5),(.3,5),(.3,8)]:
        expected=sigma*np.sqrt(2*np.pi)*math.erf(w/(2*np.sqrt(2)*sigma))/w
        assert abs(evidence(sigma,w)-expected)<1e-8
    for i in [0,1]:
        mu,a=mixture_laplace(i)
        assert abs((np.log(mixture(mu+h))-np.log(mixture(mu-h)))/(2*h))<1e-7
        assert a>0
    # This pedagogical example preserves width/height/area relations exactly.
    areas=[h*s*np.sqrt(2*np.pi) for h,s in [(2,.75),(2,.25),(1.4,.85)]]
    assert areas[1]<areas[2]
    # Independent finite differences for the V11 cross derivative and V10 area.
    example=lambda x,y:.5*(x*x+x*y+y*y)
    cross=(example(h,1+h)-example(h,1-h)-example(-h,1+h)+example(-h,1-h))/(4*h*h)
    assert abs(cross-.5)<1e-8
    assert np.allclose(np.linalg.eigvalsh([[1,.5],[.5,1]]),[.5,1.5])
    assert abs(np.linalg.det(np.diag([4,1]))**-.5-.5)<1e-12
    assert abs(np.prod(1/np.sqrt([4,1]))-.5)<1e-12
    root=Path(__file__).resolve().parent
    entries=json.loads((root/'assets/voicevox/manifest.json').read_text())['scenes']
    assert len(entries)==len(SCENES)==9
    for scene,entry in zip(SCENES,entries):
        assert valid_entry(scene,entry),scene['id']
        assert len(entry['subtitle_cues'])==len(scene['beats'])
        assert abs(entry['duration']*15-round(entry['duration']*15))<1e-7
        for b in scene['beats']:
            for q in b['segments']:
                assert not re.search('エックス|ゼット|エー|エム|エヌ|ラムダ|ミュー|シグマ|キュー|パイ|ルート',q['display']),q['display']
                assert not any(ord(c)<32 for c in q['display'])
    assert {f.name for f in (root/'assets/voicevox').glob('*.wav')}=={e['id']+'.wav' for e in entries}
    rows=json.loads((root/'reading_check.json').read_text())['sentences']
    assert [(r['id'],r['speech'],r['display']) for r in rows]==[(q['id'],q['speech'],q['display']) for s in SCENES for b in s['beats'] for q in b['segments']]
    print(json.dumps(dict(check_groups=8,mode=Z0,precision=A,variance=1/A,Z=Z,Z_laplace=ZL,relative_integral_error=ZL/Z-1,mean=MEAN,peak_p=float(pdf(Z0)),peak_q=float(gaussian(Z0)),areas=areas,evidence=[evidence(.8,5),evidence(.3,5),evidence(.3,8)],bic_penalties={str(n):[float(m/2*np.log(n)) for m in [2,5]] for n in [20,200]},audio_seconds=sum(e['duration'] for e in entries)),indent=2))

if __name__=='__main__':main()
