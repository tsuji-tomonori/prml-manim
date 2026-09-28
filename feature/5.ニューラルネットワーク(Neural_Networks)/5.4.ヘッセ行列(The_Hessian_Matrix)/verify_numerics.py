"""Cross-check analytic curvature, finite differences and matrix-free R recursion."""
import json
from pathlib import Path
import numpy as np
from hessian_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,valid_entry

def main():
    rng=np.random.default_rng(54)
    fd=[];gd=[];hv=[]
    for w in rng.uniform([-.9,.4],[1.3,1.7],(40,2)):
        result=net(w);H=result['H'];d=rng.normal(size=2)
        fd.append(float(np.max(np.abs(H-finite_hessian(w)))))
        gd.append(float(np.max(np.abs(H-gradient_difference(w)))))
        hv.append(float(np.max(np.abs(H@d-hvp(w,d)[0]))))
        assert np.allclose(H,H.T)
        assert np.allclose(np.column_stack([hvp(w,e)[0] for e in np.eye(2)]),H)
    assert max(fd)<2e-7 and max(gd)<2e-9 and max(hv)<2e-14
    inverse_error=max(float(np.max(np.abs(inverse_update(n)-np.linalg.inv(precision(n))))) for n in np.linspace(0,3,61))
    assert inverse_error<1e-13
    ellipse_error=max(float(np.max(np.abs(np.einsum('ni,ij,nj->n',ellipse_points(precision(n)),precision(n),ellipse_points(precision(n)))-1))) for n in [0,1,2,3])
    assert ellipse_error<1e-13
    zero=net(t=net()['y']);assert np.max(abs(zero['residual']))==0
    eig=np.linalg.eigvalsh([[3,2],[2,3]]);assert np.allclose(eig,[1,5])
    entries=json.loads(MANIFEST.read_text())['scenes']; assert len(entries)==len(SCENES)
    assert all(valid_entry(s,e) for s,e in zip(SCENES,entries))
    assert {p.stem for p in (Path(__file__).parent/'assets/voicevox').glob('*.wav')}=={s['id'] for s in SCENES}
    # Nonlinear logistic output: (5.85) uses the pre-sigmoid gradient.
    a=net()['y'];p=1/(1+np.exp(-a));b=net()['b']; y=.35
    logistic_H=p*(1-p)*np.outer(b,b)+(p-y)*net(t=net()['y']-1)['residual']
    def logistic_grad(w):
        q=net(w);prob=1/(1+np.exp(-q['y']));return (prob-y)*q['b']
    lfd=np.column_stack([(logistic_grad(WEIGHTS+1e-5*e)-logistic_grad(WEIGHTS-1e-5*e))/2e-5 for e in np.eye(2)])
    logistic_error=float(np.max(abs(lfd-logistic_H)));assert logistic_error<1e-9
    result=dict(seed=54,samples=40,max_function_difference_error=max(fd),max_gradient_difference_error=max(gd),max_hvp_error=max(hv),max_inverse_update_error=inverse_error,max_ellipse_level_error=ellipse_error,logistic_hessian_error=logistic_error,eigenvalues=eig.tolist(),example_H=net()['H'].tolist(),example_direction=[.6,.8],example_Hv=hvp(WEIGHTS,[.6,.8])[0].tolist(),scenes=len(entries),sentences=sum(len(b['segments']) for s in SCENES for b in s['beats']),audio_seconds=sum(e['duration'] for e in entries))
    Path('numerical_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
