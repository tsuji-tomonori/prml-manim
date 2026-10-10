"""Independent numerical identities, display notation, and narration integrity."""
import json
import re
from pathlib import Path
import numpy as np
from decision_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry

x=np.linspace(-9,9,18001)
a,b=joint(x,1),joint(x,2)
assert abs(np.trapezoid(a+b,x)-1)<1e-12
errors=[sum(mistake_parts(z)) for z in np.linspace(-3,3,601)]
assert np.argmin(errors)==300
assert abs(min(errors)-np.trapezoid(np.minimum(a,b),x))<1e-7
for c in (1,2,20,1000):
 z=optimal_boundary(c)
 assert abs(c*joint(z,1)-joint(z,2))<1e-12
 candidates=z+np.linspace(-.5,.5,101)
 risk=[mistake_parts(t)[0]+c*mistake_parts(t)[1] for t in candidates]
 assert np.argmin(risk)==50
# The opening prediction and the ending use the same posterior and costs.
assert np.allclose(risks(.08,1),[.92,.08])
assert np.allclose(risks(.08,1000),[.92,80])
assert np.isclose(posterior(.5*np.log(.92/.08)),.08)
assert reject_fraction(.4)==0 and reject_fraction(1)==1
assert reject_fraction(.9)>reject_fraction(.6)
for th in (.6,.9):
 left,right=reject_bounds(th)
 assert np.allclose([posterior(left),1-posterior(right)],th)
assert np.allclose(corrected_posterior(.8,.5,.01),[4/103,99/103])
assert np.allclose(combine_posteriors(.5,.6,.2),[6/7,1/7])
for y in [-2,MEAN,1.5]:
 assert abs(regression_risk(y)-((y-MEAN)**2+VARIANCE))<2e-5
assert abs(np.interp(MEDIAN,T,CDF)-.5)<1e-12
assert regression_risk(MEDIAN,1)<regression_risk(MEAN,1)
for s in SCENES:
 for beat in s['beats']:
  for seg in beat['segments']:
   assert not re.search(r'エックス|ラムダ|ミュー|シグマ|シータ|キュー|ティー|シーイチ|シーニ',seg['display'])
   assert not any(ord(c)<32 for c in seg['display'])
   assert seg['display'].count('$')%2==0
entries={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
for scene in SCENES:
 assert valid_entry(scene,entries[scene['id']]),scene['id']
assert {p.stem for p in (MANIFEST.parent).glob('scene*.wav')}==set(entries)
print(json.dumps(dict(mistake_min=min(errors),boundary_cost20=optimal_boundary(20),
    boundary_cost1000=optimal_boundary(1000),risk_008_cost1=risks(.08,1).tolist(),
    risk_008_cost1000=risks(.08,1000).tolist(),
    reject_06=reject_fraction(.6),reject_09=reject_fraction(.9),
    corrected=corrected_posterior(.8,.5,.01).tolist(),combined=combine_posteriors(.5,.6,.2).tolist(),
    mean=MEAN,median=MEDIAN,mode=MODE,variance=VARIANCE,
    scenes=len(SCENES),sentences=sum(len(b['segments']) for s in SCENES for b in s['beats']),
    audio_duration=sum(e['duration'] for e in entries.values())),indent=2))
print('PASS: density, error minimum, cost boundaries, reject, prior correction, fusion, regression, captions, audio integrity')
