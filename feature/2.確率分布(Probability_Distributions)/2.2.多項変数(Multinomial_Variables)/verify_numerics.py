"""Independent enumeration, quadrature and manifest checks."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.integrate import dblquad, quad
from scipy.optimize import minimize, minimize_scalar
from multinomial_model import *
from narration_content import SCENES
from make_voicevox_narration import valid_entry, MANIFEST, OUTPUT_DIR


def main():
    results={}
    assert OPENING_SEQUENCE.tolist()==[0,0,0,1]
    assert OPENING_COUNTS.tolist()==[3,1,0]
    assert SEQUENCE[:4].tolist()==OPENING_SEQUENCE.tolist()
    assert COUNTS.tolist()==[6,3,1]
    opening_prediction=predictive([1,1,1],OPENING_COUNTS)
    assert np.allclose(opening_prediction,[4/7,2/7,1/7])
    results['opening_and_return']={'sequence':OPENING_SEQUENCE.tolist(),
        'counts':OPENING_COUNTS.tolist(),'mle':(OPENING_COUNTS/4).tolist(),
        'predictive':opening_prediction.tolist(),'yellow':float(opening_prediction[2])}
    mu=np.array([.5,.3,.2]); counts=np.array([2,1,1])
    sequences=list(itertools.product(range(3),repeat=4))
    probs=np.array([np.prod(mu[list(s)]) for s in sequences])
    hit=[i for i,s in enumerate(sequences) if np.array_equal(np.bincount(s,minlength=3),counts)]
    assert len(hit)==coefficient(counts)==12
    assert abs(probs.sum()-1)<1e-12
    assert abs(probs[hit].sum()-multinomial(mu,counts))<1e-12
    results['enumeration']={'sequences':81,'matches':len(hit),'single':float(probs[hit[0]]),'count_probability':float(probs[hit].sum()),'total':float(probs.sum())}
    optimum=minimize(lambda x:-np.log(likelihood(x)),[.3,.4,.3],bounds=[(.0001,.9999)]*3,
                     constraints=[{'type':'eq','fun':lambda x:sum(x)-1}],tol=1e-11)
    assert optimum.success and np.allclose(optimum.x,COUNTS/10,atol=1e-6)
    results['mle']={'numerical':optimum.x.tolist(),'formula':(COUNTS/10).tolist()}
    integrals=[]
    for a in [[1,1,1],[2,2,2],[4,2,1],[8,5,3]]:
        f=lambda v,u:float(dirichlet_density([u,v,1-u-v],a))
        total,err=dblquad(f,0,1,lambda u:0,lambda u:1-u,epsabs=1e-9)
        first,_=dblquad(lambda v,u:u*f(v,u),0,1,lambda u:0,lambda u:1-u,epsabs=1e-9)
        assert abs(total-1)<1e-8 and abs(first-a[0]/sum(a))<1e-8
        integrals.append({'alpha':a,'integral':total,'mean_red':first})
    results['dirichlet_quadrature']=integrals
    yellow_integral,_=dblquad(
        lambda v,u:(1-u-v)*float(dirichlet_density([u,v,1-u-v],[4,2,1])),
        0,1,lambda u:0,lambda u:1-u,epsabs=1e-9)
    assert abs(yellow_integral-1/7)<1e-8
    results['opening_and_return']['yellow_by_quadrature']=float(yellow_integral)
    point=np.array([.4,.35,.25]);prior=np.array([2,2,2])
    ratio=dirichlet_density(point,prior+COUNTS)/(dirichlet_density(point,prior)*likelihood(point))
    point2=np.array([.2,.5,.3]);ratio2=dirichlet_density(point2,prior+COUNTS)/(dirichlet_density(point2,prior)*likelihood(point2))
    assert np.isclose(ratio,ratio2)
    assert np.allclose(prior+sum(np.eye(3)[SEQUENCE]),prior+COUNTS)
    results['conjugacy']={'density_ratio':float(ratio),'posterior':(prior+COUNTS).tolist()}
    assert np.allclose(predictive([1,1,1],OPENING_COUNTS),np.array([4,2,1])/7)
    results['predictive']={'four':predictive([1,1,1],OPENING_COUNTS).tolist(),'forty':predictive([1,1,1],[30,10,0]).tolist()}
    # Independently optimize and integrate the two visual-aid examples.
    opt2=minimize_scalar(lambda u:-(3*np.log(u)+2*np.log1p(-u)),bounds=(.001,.999),method='bounded',options={'xatol':1e-12})
    assert opt2.success and abs(opt2.x-.6)<1e-6
    grad=np.array([3/.6,2/.4]);normal=np.ones(2)
    assert abs(grad@np.array([1,-1]))<1e-12
    assert np.allclose(grad-5*normal,0)
    evidence,_=quad(lambda u:6*u*(1-u)*u**3,0,1)
    posterior_area,_=quad(lambda u:30*u**4*(1-u),0,1)
    assert abs(evidence-.2)<1e-12 and abs(posterior_area-1)<1e-12
    results['review_examples']={'constrained_optimum':opt2.x,'gradient':grad.tolist(),
        'lambda':-5,'tangent_derivative':float(grad@np.array([1,-1])),
        'prior_times_likelihood_area':evidence,'posterior_area':posterior_area}
    manifest=json.loads(MANIFEST.read_text())
    assert len(manifest['scenes'])==len(SCENES)==9
    assert all(valid_entry(s,e) for s,e in zip(SCENES,manifest['scenes']))
    assert {p.stem for p in OUTPUT_DIR.glob('*.wav')}=={s['id'] for s in SCENES}
    results['audio']={'scenes':9,'sentences':sum(len(b['segments']) for s in SCENES for b in s['beats']),
                      'total_seconds':sum(s['duration'] for s in manifest['scenes'])}
    Path('numerical_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    print('8 numerical / audio checks passed')

if __name__=='__main__':main()
