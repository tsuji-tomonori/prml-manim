"""Independent central differences, error bounds and caption/audio integrity."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from backprop_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,valid_entry
ROOT=Path(__file__).parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--captions',action='store_true');args=parser.parse_args()
    eps=1e-5;errors=[]
    for x,t in DATA:
        b=backward(x,t)
        for layer,w in enumerate([W1,W2]):
            analytic=b['g1' if layer==0 else 'g2']
            for idx in np.ndindex(w.shape):
                up=w.copy();down=w.copy();up[idx]+=eps;down[idx]-=eps
                cd=((loss(x,t,up,W2)-loss(x,t,down,W2)) if layer==0 else (loss(x,t,W1,up)-loss(x,t,W1,down)))/(2*eps)
                errors.append(abs(cd-analytic[idx]))
    assert max(errors)<1e-8
    jerrors=[]
    for x,t in DATA:
        for i in range(2):
            d=np.zeros(2);d[i]=eps
            cd=(forward(x+d)['y']-forward(x-d)['y'])/(2*eps)
            jerrors.extend(abs(cd-jacobian(x)[:,i]))
    assert max(jerrors)<1e-8
    a=np.array([.2,-.4,.7]);y=softmax(a);seed=np.diag(y)-np.outer(y,y)
    seeds=[]
    for i in range(3):
        d=np.zeros(3);d[i]=eps
        seeds.extend(abs((softmax(a+d)-softmax(a-d))/(2*eps)-seed[:,i]))
    assert max(seeds)<1e-9
    # Branch with the original negative outgoing weight: cancellation at a computed target.
    z=np.tanh(.5);v=np.array([1.1,-.6]);y=v*z+np.array([.05,-.1]);t1=.3
    t2=y[1]+v[0]*(y[0]-t1)/v[1]
    assert abs(v@(y-np.array([t1,t2])))<1e-12
    g1,g2=batch();before=total();after=total(W1-.1*g1,W2-.1*g2)
    assert after<before
    batch_errors=[]
    for idx in np.ndindex(W2.shape):
        up=W2.copy();down=W2.copy();up[idx]+=eps;down[idx]-=eps
        batch_errors.append(abs((total(w2=up)-total(w2=down))/(2*eps)-g2[idx]))
    assert max(batch_errors)<1e-8
    cd_error=[abs(central(.2,e)-scalar(.2)['g']) for e in [.1,.01,.001]]
    assert cd_error[1]<cd_error[0]/90 and cd_error[2]<cd_error[1]/90
    direction=np.array([1,-.5]);y0=forward()['y'];J=jacobian()
    local_errors={str(s):float(np.linalg.norm(forward(X+s*direction)['y']-y0-s*J@direction)) for s in [.05,.2,1.1]}
    assert local_errors['0.05']<local_errors['0.2']<local_errors['1.1']
    for s in np.linspace(0,1.1,101):
        for point in [forward(X+s*direction)['y'],y0+s*J@direction]:
            assert .8<=point[0]<=2.15 and -.62<=point[1]<=-.3
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES) and all(valid_entry(s,e) for s,e in zip(SCENES,entries))
    segments=[v for s in SCENES for b in s['beats'] for v in b['segments']]
    display='\n'.join(v['display'] for v in segments)
    assert not re.search('エックス|ダブリュー|デルタ|ラムダ|ミュー|シグマ|ゼット|ジェー',display)
    (ROOT/'media').mkdir(exist_ok=True);(ROOT/'media/display.txt').write_text(display)
    result=dict(weight_derivatives=36,max_gradient_error=max(errors),jacobian_derivatives=12,max_jacobian_error=max(jerrors),max_softmax_seed_error=max(seeds),max_batch_error=max(batch_errors),loss_before=before,loss_after=after,central_difference_errors=cd_error,local_linear_errors=local_errors,branch_cancel_target=t2,scenes=len(SCENES),beats=sum(len(s['beats']) for s in SCENES),sentences=len(segments),math_captions=sum('$' in s['display'] for s in segments))
    if args.captions:
        from video_support import caption_mobject
        captions=[caption_mobject(v['display']) for v in segments]
        result.update(max_caption_width=max(m.width for m in captions),max_caption_height=max(m.height for m in captions))
    (ROOT/'numerical_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
