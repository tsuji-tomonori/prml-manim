"""Independent identities, derivative checks, iteration consistency and artifact checks."""
import json
from pathlib import Path
import numpy as np
from evidence_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,valid_entry

def main():
    errors=[]
    for z in [-6.,0.,OPT,6.]:
        s=at(z);a=s['alpha'];b=s['beta'];C=np.eye(len(T))/b+PHI@PHI.T/a
        independent=-.5*(len(T)*np.log(2*np.pi)+np.linalg.slogdet(C)[1]+T@np.linalg.solve(C,T))
        errors.append(abs(s['logev']-independent))
        np.testing.assert_allclose(s['logev'],independent,atol=2e-9)
        np.testing.assert_allclose(s['A']@s['mean'],b*PHI.T@T,atol=1e-10)
        delta=np.arange(10)/11-.4;w=s['mean']+delta
        energy=.5*b*np.sum((T-PHI@w)**2)+.5*a*w@w
        np.testing.assert_allclose(energy,.5*b*s['rss']+.5*a*s['norm']+.5*delta@s['A']@delta)
        finite=(at(z+1e-5)['logev']-at(z-1e-5)['logev'])/2e-5
        np.testing.assert_allclose(finite,.5*(s['gamma']-a*s['norm']),atol=2e-6)
    print('Evidence / predictive covariance / square completion / derivative: PASS; max error',max(errors))
    scalar_errors=[abs(scalar(z)[0]-scalar(z)[1]) for z in [-2,0,2.5]]
    assert max(scalar_errors)<2e-10
    print('Scalar integral quadrature: PASS; max error',max(scalar_errors))
    for old,new in zip(ITER,ITER[1:]):
        np.testing.assert_allclose(new['alpha'],old['gamma']/old['norm'])
        np.testing.assert_allclose(new['beta'],(len(T)-old['gamma'])/old['rss'])
    assert np.all(np.diff([s['logev'] for s in ITER])>=-1e-10)
    s=ITER[-1]
    print('30 iterations: PASS', {k:s[k] for k in ['alpha','beta','gamma','logev']})
    assert abs(at(OPT)['gamma']-np.exp(OPT)*at(OPT)['norm'])<1e-10
    span=max(float(abs(PU@at(float(z))['mean']).max()) for z in GRID)
    assert span<1.6 and max(abs(T))<1.6
    assert max(max(abs(y)) for y in POLY_CURVES)<1.6
    assert TEST.min()>.2 and TEST.max()<.8
    print('Plot range: PASS; mean max',span,'RMS range',TEST.min(),TEST.max())
    assert int(np.argmax([s['logev'] for s in POLY]))==3
    print('Polynomial optimum d=3:',POLY[3]['logev'])
    assert HIDDEN_X not in X and HIDDEN_TRUE == -1.0
    for z,expected in [(-6,HIDDEN_WEAK),(6,HIDDEN_STRONG),(OPT,HIDDEN_SELECTED)]:
        np.testing.assert_allclose(hidden_prediction(z),expected,atol=1e-12)
    np.testing.assert_allclose([HIDDEN_WEAK,HIDDEN_STRONG,HIDDEN_SELECTED],
                               [-0.7143597518460014,-0.23049363245775223,-0.944805486915999],atol=1e-10)
    assert abs(HIDDEN_SELECTED-HIDDEN_TRUE) < abs(HIDDEN_WEAK-HIDDEN_TRUE)
    assert abs(HIDDEN_SELECTED-HIDDEN_TRUE) < abs(HIDDEN_STRONG-HIDDEN_TRUE)
    assert [round(v,2) for v in (HIDDEN_WEAK,HIDDEN_STRONG,HIDDEN_SELECTED,HIDDEN_TRUE)] == [-0.71,-0.23,-0.94,-1.0]
    print('Unseen x=0.75 callback and fixed-beta predictions: PASS',
          HIDDEN_WEAK,HIDDEN_STRONG,HIDDEN_SELECTED,HIDDEN_TRUE)
    # Independent small examples used by V13a and V10.
    w=np.linspace(-3,4,101)
    np.testing.assert_allclose(2*w*w-4*w+5,2*(w-1)**2+3)
    precision=np.diag([4.,1.]);sigma=1/np.sqrt(np.linalg.eigvalsh(precision))
    np.testing.assert_allclose(np.prod(sigma),1/np.sqrt(np.linalg.det(precision)))
    assert np.prod(sigma)==.5
    print('Visual examples: square completion and width product 0.5: PASS')
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES)==10
    assert all(valid_entry(sc,e) for sc,e in zip(SCENES,entries))
    assert set(p.stem for p in MANIFEST.parent.glob('*.wav'))=={s['id'] for s in SCENES}
    rows=json.loads((Path(__file__).parent/'reading_check.json').read_text())['sentences']
    segments=[s for sc in SCENES for b in sc['beats'] for s in b['segments']]
    assert [(r['id'],r['speech']) for r in rows]==[(s['id'],s['speech']) for s in segments]
    forbidden=['エックス','アルファ','ベータ','ガンマ','ラムダ','エムエヌ','シグマ','ダブリュー']
    assert not any(k in s['display'] for k in forbidden for s in segments)
    print(f'10 WAV, hashes, {len(segments)} readings / captions: PASS; total seconds',sum(e['duration'] for e in entries))
if __name__=='__main__':main()
