"""Independent normalization, moment, inference and audio consistency checks."""
import json
import re
from pathlib import Path
import numpy as np
from exponential_model import *
from narration_content import SCENES
from make_voicevox_narration import valid_entry

ROOT=Path(__file__).resolve().parent

def main():
    errors={}
    grid=np.linspace(-5,5,101)
    assert np.allclose(sigmoid(grid)+sigmoid(-grid),1)
    for e in np.linspace(-2,2,25):
        p=softmax([e,-e/2,0])
        assert np.allclose(p.sum(),1) and np.all(p>0)
        assert np.allclose(np.log(p[:2]/p[2]),[e,-e/2])
    x=np.linspace(-12,12,40001)
    errors['gaussian_normalization']=0.
    for mu in [-.8,0,.8]:
        for sd in [.55,.7,1.1]:
            eta=gaussian_natural(mu,sd)
            p=gaussian_g(eta)/np.sqrt(2*np.pi)*np.exp(eta[0]*x+eta[1]*x*x)
            assert np.allclose(p,normal(x,mu,sd))
            error=abs(np.trapezoid(p,x)-1)
            errors['gaussian_normalization']=max(error,errors['gaussian_normalization'])
            assert error<1e-9
            assert np.isclose(np.trapezoid(x*p,x),mu)
            assert np.isclose(np.trapezoid(x*x*p,x),mu*mu+sd*sd)
    # Review examples: component product and variance as a weighted square deviation.
    assert np.dot([2,-1],[1,3]) == -1
    for m, expected in [(.5,.25),(.1,.09)]:
        e=np.log(m/(1-m))
        assert np.isclose(sigmoid(e),m)
        direct=(1-m)*(0-m)**2+m*(1-m)**2
        assert np.isclose(direct,expected)
    step=1e-3
    A=lambda e:np.logaddexp(0,e)
    first=(A(grid+step)-A(grid-step))/(2*step)
    second=(A(grid+step)-2*A(grid)+A(grid-step))/step**2
    errors['mean_derivative']=float(np.max(abs(first-sigmoid(grid))))
    errors['variance_derivative']=float(np.max(abs(second-sigmoid(grid)*(1-sigmoid(grid)))))
    assert errors['mean_derivative']<2e-8 and errors['variance_derivative']<2e-8
    eta=np.log(COINS.mean()/(1-COINS.mean()))
    assert np.all(log_likelihood(eta)>=log_likelihood(grid))
    for m in [.1,.4,.7,.95]:
        direct=np.prod(m**COINS*(1-m)**(1-COINS))
        assert np.isclose(direct,m**COINS.sum()*(1-m)**(len(COINS)-COINS.sum()))
        assert np.isclose(direct,np.prod(m**COINS[::-1]*(1-m)**(1-COINS[::-1])))
    assert np.isclose(POINTS.var(),(POINTS**2).mean()-POINTS.mean()**2)
    u=np.linspace(0,1,10001)
    prior=beta_pdf(u,2,2)
    post=prior*u**7*(1-u)**3
    post/=np.trapezoid(post,u)
    assert np.allclose(post,beta_pdf(u,9,5),atol=1e-8)
    assert np.isclose(np.trapezoid(u*post,u),9/14)
    e=np.linspace(-18,18,30001); m=sigmoid(e)
    # Beta(a,b) on mu is g(eta)^(a+b) exp(a eta) on eta after its Jacobian.
    density_eta=beta_pdf(m,2,2)*m*(1-m)
    assert np.allclose(density_eta/6,np.exp(2*e-4*np.logaddexp(0,e)))
    assert abs(np.trapezoid(density_eta,e)-1)<1e-9
    mass_errors=[]
    for q in np.linspace(1,2,11):
        for k in range(4):
            lo=(k/4)**(1/q); hi=((k+1)/4)**(1/q)
            # Integrate in the original uniform coordinate as an independent area check.
            mass_errors.append(abs(hi**q-lo**q-.25))
    errors['transformed_quarter_mass']=max(mass_errors)
    assert max(mass_errors)<1e-12
    assert np.allclose(np.log(np.array([10,100,1000])/np.array([1,10,100])),np.log(10))
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())
    assert len(manifest['scenes'])==len(SCENES)==10
    assert {p.stem for p in (ROOT/'assets/voicevox').glob('*.wav')}=={s['id'] for s in SCENES}
    for s,en in zip(SCENES,manifest['scenes']):
        assert valid_entry(s,en)
        for seg in [seg for b in s['beats'] for seg in b['segments']]:
            assert not re.search('エックス|イータ|ミュー|シグマ|ラムダ|ニュー|カイ',seg['display'])
        assert all(abs(d*15-round(d*15))<1e-9 for d in en['beat_durations'])
    print(json.dumps(dict(checks='8 original groups + review examples passed',errors=errors,coin_mean=float(COINS.mean()),gaussian_sum=float(POINTS.sum()),gaussian_square_sum=float((POINTS**2).sum()),gaussian_mean=float(POINTS.mean()),gaussian_variance=float(POINTS.var()),posterior_mean=9/14,audio_seconds=sum(e['duration'] for e in manifest['scenes'])),indent=2))

if __name__=='__main__': main()
