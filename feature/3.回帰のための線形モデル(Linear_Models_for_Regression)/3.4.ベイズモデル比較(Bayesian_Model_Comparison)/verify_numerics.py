"""Independent integral/identity checks for the examples shown in the movie."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal
import bayesian_model as m
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,valid_entry

def verify():
    errors=[]
    for width in [4,8,16,40]:
        numerical=quad(lambda w:float(m.normal(1,w,.35))/width,-width/2,width/2,epsabs=1e-12)[0]
        errors.append(abs(numerical-m.uniform_evidence(width)))
        assert np.isclose(numerical,m.uniform_evidence(width),atol=1e-11)
    print('Uniform-prior quadrature: max error',max(errors))
    for sd in m.MODEL_STDS:
        tau=np.sqrt(sd**2-.2**2)
        marginal=quad(lambda w:float(m.normal(m.DATUM,w,.2)*m.normal(w,0,tau)),-np.inf,np.inf)[0]
        assert np.isclose(marginal,m.normal(m.DATUM,std=sd),rtol=1e-8)
    print('Three Gaussian latent-parameter integrals: passed')
    errors=[]
    for d,fit in enumerate(m.FITS):
        phi=m.design(m.X,d)
        c=m.NOISE**2*np.eye(len(m.X))+phi@phi.T
        errors.append(abs(fit['evidence']-multivariate_normal.logpdf(m.T,mean=np.zeros(12),cov=c)))
        # Eq 3.69 gives an independent coefficient-space evidence identity at w=0.
        loglike=-len(m.X)*np.log(np.sqrt(2*np.pi)*m.NOISE)-m.T@m.T/(2*m.NOISE**2)
        logprior=-(d+1)/2*np.log(2*np.pi)
        logposterior=multivariate_normal.logpdf(np.zeros(d+1),mean=fit['mean'],cov=fit['cov'])
        assert abs(loglike+logprior-logposterior-fit['evidence'])<1e-9
    assert max(errors)<1e-9
    assert np.argmax(m.LOG_EVIDENCE)==2
    assert np.all(np.diff(m.LOG_ML)>=-1e-10)
    assert m.CURVES.min()>0 and m.CURVES.max()<1.8
    assert m.ML_CURVES.min()>0 and m.ML_CURVES.max()<1.8
    print('Regression covariance/parameter identities: max error',max(errors))
    for p in [.1,.5,.85]:
        density=lambda x:p*m.normal(x,-1.35,.32)+(1-p)*m.normal(x,1.35,.32)
        assert abs(quad(density,-np.inf,np.inf)[0]-1)<1e-10
    assert m.normal(0,1.35,.32)<.001*m.normal(1.35,1.35,.32)
    prior=np.array([.2,.6,.2]);post=m.posterior(prior)
    assert abs(post.sum()-1)<1e-12
    assert np.isclose(post[1]/post[2],prior[1]/prior[2]*m.EVIDENCES[1]/m.EVIDENCES[2])
    print('Mixture normalization/bimodality and posterior odds: passed')
    expected=6*(m.P_TRUE*np.log(m.P_TRUE/m.P_ALT)+(1-m.P_TRUE)*np.log((1-m.P_TRUE)/(1-m.P_ALT)))
    assert np.isclose(expected,m.EXPECTED_LOG_BF)
    assert np.any(m.LOG_BF<0) and expected>0
    print('Expected log Bayes factor',expected,'wrong-model probability',m.COUNT_PROBS[:3].sum())
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES)==9
    assert all(valid_entry(s,e) for s,e in zip(SCENES,entries))
    assert {p.stem for p in Path('assets/voicevox').glob('*.wav')}=={s['id'] for s in SCENES}
    readings=json.loads(Path('reading_check.json').read_text())['sentences']
    sentences=[s for scene in SCENES for b in scene['beats'] for s in b['segments']]
    assert [(s['id'],s['speech'],s['display']) for s in sentences]==[(s['id'],s['speech'],s['display']) for s in readings]
    print('Narration/hash/reading alignment:',len(sentences),'sentences;',sum(e['duration'] for e in entries),'seconds')
    print('All 6 numerical/asset verification groups passed')

if __name__=='__main__':verify()
