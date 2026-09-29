"""Extract every symbolic caption, all beats, and before/after synchronization frames.

Run after rendering. Writes ignored media/review artifacts for human inspection.
"""
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from narration_content import SCENES
from make_voicevox_narration import MANIFEST

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_3_1_linear_basis_function_models/480p15/PRML31LinearBasisFunctionModels.mp4'
OUT=ROOT/'media/review'

def main():
    OUT.mkdir(exist_ok=True)
    timeline=json.loads((ROOT/'media/prml31_timeline.json').read_text())
    manifest=json.loads(MANIFEST.read_text())['scenes']
    frames={}
    symbolic=[]
    for sc in timeline:
        for i,b in enumerate(sc['beats']):
            frames[f"{sc['id']}-beat{i+1:02}"]=(b['start']+b['end'])/2
            for c in b['cues']:
                if '$' in c['display']:
                    key=c['id'];frames[key]=(c['start']+c['end'])/2;symbolic.append(key)
    sync=[]
    added=[]
    for si,sc in enumerate(timeline):
        for bi,b in enumerate(sc['beats']):
            if any('-recap-' in c['id'] or '-aid-' in c['id'] for c in b['cues']):
                added.append((si,bi))
                prefix=f"added-{sc['id']}"
                frames[prefix+'-before']=max(0,b['start']-.2)
                frames[prefix+'-after']=b['end']+.2
                frames[prefix+'-entry']=b['start']+.1
                frames[prefix+'-exit']=b['end']-.1
                for ai,action in enumerate(b.get('actions',[])):
                    if action['name']=='breath':continue
                    frames[f'{prefix}-action{ai:02}']=(action['start']+action['end'])/2
                    if action['name'] in ['R1.1 sum squares','V08b multiply negative term','V04b opposite small step']:
                        for fraction in [.15,.85]:
                            frames[f'{prefix}-action{ai:02}-{fraction}']=action['start']+(action['end']-action['start'])*fraction
            if any(c['id'] in ['scene07-02-02','scene08-02-01'] for c in b['cues']):
                for suffix,t in [('before',b['start']-.2),('during',(b['start']+b['end'])/2),('after',b['end']+.2)]:
                    frames[f'reference-{sc["id"]}-{suffix}']=t
    for si,bi in added+[(0,4),(4,3),(6,3)]:
        sc=timeline[si];b=sc['beats'][bi];entry=manifest[si]
        with wave.open(str(ROOT/entry['path']),'rb') as w:
            pcm=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(float)/32768;sr=w.getframerate()
        start=b['action_start']-sc['start'];end=b['action_end']-sc['start']
        region=pcm[int(start*sr):int(end*sr)]
        above=np.flatnonzero(np.abs(region)>10**(-45/20))
        onset=sc['start']+start+above[0]/sr
        sync.append(dict(scene=sc['id'],beat=bi+1,action_start=b['action_start'],action_end=b['action_end'],pcm_onset=onset))
        for frac in [.15,.85]:
            frames[f"sync-{sc['id']}-{frac}"]=b['action_start']+(b['action_end']-b['action_start'])*frac
    for key,t in frames.items():
        subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(t),'-i',str(VIDEO),'-frames:v','1',str(OUT/f'{key}.png')],check=True)
    keys=list(frames)
    for k in range(0,len(keys),6):
        sheet=Image.new('RGB',(1708,3*510),'#202020');draw=ImageDraw.Draw(sheet)
        for j,key in enumerate(keys[k:k+6]):
            x=(j%2)*854;y=(j//2)*510
            sheet.paste(Image.open(OUT/f'{key}.png'),(x,y+30));draw.text((x+10,y+8),f'{key} / {frames[key]:.3f}s',fill='white')
        sheet.save(OUT/f'sheet-{k//6+1:02}.png')
    (OUT/'frames.json').write_text(json.dumps(dict(frames=frames,symbolic=symbolic,sync=sync),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(count=len(frames),symbolic=len(symbolic),sync=sync),indent=2))

if __name__=='__main__': main()
