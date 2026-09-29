"""Check small examples, review budgets and byte-identical reuse of old speech."""
import io
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from discriminative_model import sigmoid
from narration_content import SCENES

ROOT=Path(__file__).resolve().parent
BASELINE='276bec1'

def git_bytes(path):
    return subprocess.check_output(['git','show',f'{BASELINE}:{path}'],cwd=ROOT)

def main():
    h=1e-5
    error=lambda a:float(np.logaddexp(0,-a))
    derivative=(error(h)-error(-h))/(2*h)
    assert abs(derivative+.5)<1e-10
    dy=float(sigmoid(.04)-sigmoid(0))
    de=error(.04)-error(0)
    assert abs(dy-.01)<2e-6 and abs(de+.02)<.00021
    e=lambda w:1+2*w+2*w*w+.5*w**4
    g=(e(h)-e(-h))/(2*h)
    H=(e(h)-2*e(0)+e(-h))/h**2
    assert abs(g-2)<1e-8 and abs(H-4)<2e-6
    delta=-2/4
    assert abs(2+4*delta)<1e-12 and e(delta)<e(0)
    probs=np.array([.1,.5,.2])
    assert np.allclose(probs*(1-probs),[.09,.25,.16])
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
    prefix=ROOT.relative_to(ROOT.parents[2]).as_posix()
    previous=json.loads(git_bytes(f'{prefix}/assets/voicevox/manifest.json'))['scenes']
    unchanged=0;updated=[];cards=[]
    for scene,new,old in zip(SCENES,manifest,previous):
        oldc={c['id']:c for c in old['subtitle_cues']}
        def segments(blob,cues):
            with wave.open(io.BytesIO(blob),'rb') as f:
                rate=f.getframerate();pcm=f.readframes(f.getnframes());stride=f.getnchannels()*f.getsampwidth()
            return {c['id']:pcm[round(c['start']*rate)*stride:round(c['end']*rate)*stride] for c in cues}
        oldpcm=segments(git_bytes(f'{prefix}/{old["path"]}'),old['subtitle_cues'])
        newpcm=segments((ROOT/new['path']).read_bytes(),new['subtitle_cues'])
        assert set(oldc)<=set(newpcm)
        for c in new['subtitle_cues']:
            if c['id'] in oldc:
                if c['speech']==oldc[c['id']]['speech']:
                    assert newpcm[c['id']]==oldpcm[c['id']],c['id']
                    unchanged+=1
                else:
                    updated.append(c['id'])
        for beat,d in zip(scene['beats'],new['beat_durations']):
            key=beat['segments'][0]['id']
            if '-recap-' in key or '-aid-' in key:
                bounds=(10,30) if '-recap-' in key else (5,20)
                assert bounds[0]<=d<=bounds[1]
                cards.append(dict(id=key,seconds=d,note=beat['visual_note']))
    oldtime=sum(e['duration'] for e in previous);newtime=sum(e['duration'] for e in manifest)
    assert len(cards)==5 and newtime<=1.15*oldtime
    result=dict(baseline_commit=BASELINE,chain_derivative=derivative,chain_delta_y=dy,chain_delta_E=de,
        newton_gradient=g,newton_curvature=H,newton_step=delta,
        unchanged_sentences_byte_identical=unchanged,changed_existing_sentences=updated,cards=cards,
        baseline_wav_seconds=oldtime,new_wav_seconds=newtime,increase_percent=(newtime/oldtime-1)*100)
    (ROOT/'visual_aid_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
