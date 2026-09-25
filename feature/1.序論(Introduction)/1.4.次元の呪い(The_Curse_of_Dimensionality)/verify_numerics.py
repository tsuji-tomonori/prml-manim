"""Numerical identities, empirical distributions, and asset consistency checks."""
import json
from pathlib import Path
import numpy as np
from dimension_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry


def main():
    assert [cell_count(d) for d in [1,2,3,6,10]] == [5,25,125,15625,9765625]
    assert 0.99989 < empty_fraction(10) < 0.99990
    # Count unique monomials independently, by enumerating exponent triples.
    for d in range(1,6):
        from itertools import combinations_with_replacement
        count=sum(len(list(combinations_with_replacement(range(d),k))) for k in range(4))
        assert coefficient_count(d)==count
    assert coefficient_count(100)==176851
    print('PASS grid/occupancy and polynomial coefficient counts')
    for d in [1,2,3,20,50,100]:
        e=1-2**(-1/d)
        assert abs(shell_fraction(d,e)-.5)<1e-13
    assert abs(shell_fraction(2,.1)-.19)<1e-14
    # Empirical radii in a uniform D-ball are U^(1/D).
    u=np.random.default_rng(1414).random(200000)
    assert abs(np.mean(u**(1/20)>.9)-shell_fraction(20,.1))<.003
    print('PASS sphere scaling, shell half-volume, Monte Carlo')
    r=np.linspace(0,20,40001)
    for d in [1,2,20,100]:
        p=radial_pdf(r,d)
        assert abs(np.trapezoid(p,r)-1)<3e-8
        assert abs(np.trapezoid(r*r*p,r)-d)<1e-7
        assert abs(np.trapezoid((r*r-d)**2*p,r)-2*d)<1e-6
        assert abs(r[np.argmax(p)]-np.sqrt(d-1))<.001
        u=np.linspace(0,20/np.sqrt(d),40001)
        assert abs(np.trapezoid(normalized_pdf(u,d),u)-1)<3e-8
    print('PASS Gaussian radial normalization, mode, second/fourth moments, change of variable')
    pts,labels=class_data();inside=np.all((pts>=[.2,.4])&(pts<[.6,.8]),axis=1)
    counts=np.bincount(labels[inside],minlength=3)
    assert counts.argmax()==0
    for a in np.linspace(-1,1,21):
        image=object_pixels(.4,-.3,a)
        assert image.shape==(16,16) and np.all((image>=0)&(image<=1))
    assert np.linalg.norm(object_pixels(angle=.2)-object_pixels(angle=.2001))<.01
    print('PASS majority class',counts.tolist(),'and smooth bounded pixel data')
    manifest=json.loads(MANIFEST.read_text())
    assert len(manifest['scenes'])==len(SCENES)==8
    for s,e in zip(SCENES,manifest['scenes']):
        assert valid_entry(s,e),s['id']
    assert {p.stem for p in MANIFEST.parent.glob('*.wav')}=={s['id'] for s in SCENES}
    print('PASS all WAV hashes, scripts, display/speech cues; duration',sum(e['duration'] for e in manifest['scenes']))
    for d in [2,3,20,50]: print('shell',d,float(shell_fraction(d,.1)))
    print('empty',empty_fraction(10),'half thickness D50',1-2**(-1/50))

if __name__=='__main__':main()
