"""Independent identities and finite differences for all displayed experiments."""
import json
from pathlib import Path
import numpy as np
from training_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,valid_entry

def main():
    eps=1e-5
    fd=(regression_error(REG_W+eps)-regression_error(REG_W-eps))/(2*eps)
    assert abs(fd)<1e-8
    assert np.isclose(VARIANCE,np.mean((prediction(X,REG_W)-T)**2))
    errors=[]
    for a in np.linspace(-4,4,23):
        for t in (0,1):
            d=(binary_loss(a+eps,t)-binary_loss(a-eps,t))/(2*eps)
            errors.append(abs(d-(sigmoid(a)-t)))
    assert max(errors)<1e-8
    a=np.array([1.2,-.7,.4]); p=softmax(a)
    assert np.allclose(p,softmax(a+100)) and np.isclose(p.sum(),1)
    for k in range(3):
        e=np.eye(3)[k]*eps
        d=(-np.log(softmax(a+e)[0])+np.log(softmax(a-e)[0]))/(2*eps)
        assert abs(d-(p[k]-(k==0)))<1e-8
    assert np.allclose(landscape_grad(STATIONARY),0)
    eig,u=np.linalg.eigh(H)
    assert np.allclose(eig,[1,9]) and np.allclose(H@u,u*eig)
    for theta in np.linspace(0,2*np.pi,91):
        v=rotation(.4)@(np.array([np.cos(theta),np.sin(theta)])/np.sqrt(eig))
        assert np.isclose(v@H@v,1)
    results={}
    for eta,n in ((.04,20),(.20,20),(.24,7)):
        path=descent(eta,n); e=np.einsum('ni,ij,nj->n',path,H,path)/2
        if eta<2/9: assert np.all(np.diff(e)<0)
        else: assert e[-1]>e[0]
        results[str(eta)]=dict(initial=float(e[0]),final=float(e[-1]),steps=n)
    b=batch_online(); o=batch_online(True)
    assert abs(b[-1]-TARGETS.mean())<1e-5
    assert np.isclose(np.sum(TARGETS.mean()-TARGETS),0)
    assert np.all(np.abs(TARGETS.mean()-TARGETS)>0)
    # Independent finite differences for the new explanatory slice at (1,-1).
    example=lambda w:w[0]**2+.5*w[1]**2
    point=np.array([1.,-1.])
    gradient=np.array([(example(point+np.eye(2)[i]*eps)-example(point-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
    assert np.allclose(gradient,[2,-1],atol=1e-9)
    residual=.45; beta=2.
    density=np.sqrt(beta/(2*np.pi))*np.exp(-.5*beta*residual**2)
    gaussian_error=abs(-np.log(density)-(.5*beta*residual**2+.5*np.log(2*np.pi/beta)))
    assert gaussian_error<1e-12
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES) and all(valid_entry(s,e) for s,e in zip(SCENES,entries))
    actual={p.stem for p in Path('assets/voicevox').glob('*.wav')}
    assert actual=={s['id'] for s in SCENES}
    result=dict(check_groups=7,regression_optimum=REG_W,noise_variance_ml=VARIANCE,
                binary_gradient_max_error=max(errors),stationary_points=STATIONARY.tolist(),
                hessian_eigenvalues=eig.tolist(),descent=results,batch_final=float(b[-1]),
                online_final=float(o[-1]),scalar_optimum=float(TARGETS.mean()),
                audio_seconds=sum(e['duration'] for e in entries),sentences=sum(len(e['subtitle_cues']) for e in entries),
                visual_aid_gradient=gradient.tolist(),visual_aid_gradient_max_error=float(max(abs(gradient-[2,-1]))),
                recap_gaussian_error=gaussian_error,recap_binary_losses=(-np.log([.9,.1])).tolist())
    Path('numerical_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
