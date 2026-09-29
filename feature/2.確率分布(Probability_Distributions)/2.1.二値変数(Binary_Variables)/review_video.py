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
VIDEO=ROOT/'media/videos/prml_2_1_binary_variables/480p15/PRML21BinaryVariables.mp4'
OUT=ROOT/'media/review'

def main():
    OUT.mkdir(exist_ok=True)
    timeline=json.loads((ROOT/'media/prml21_timeline.json').read_text())
    manifest=json.loads(MANIFEST.read_text())['scenes']
    frames={}
    symbolic=[]
    for sc in timeline:
        for i,b in enumerate(sc['beats']):
            frames[f"{sc['id']}-beat{i+1:02}"]=(b['start']+b['end'])/2
            for c in b['cues']:
                if '$' in c['display']:
                    key=c['id'];frames[key]=(c['start']+c['end'])/2;symbolic.append(key)
    # New cards: include the preceding/following body and every action phase.
    additions=[]
    for si,bi in [(2,5),(4,1)]:
        sc=timeline[si];b=sc['beats'][bi]
        frames[f"context-{sc['id']}-before"]=b['start']-.4
        frames[f"context-{sc['id']}-after"]=b['end']+.4
        for j,a in enumerate(b['actions']):
            if a['name']=='breath':continue
            key=f"added-{sc['id']}-{j+1}"
            frames[key]=(a['start']+a['end'])/2
            additions.append(dict(key=key,**a))
        frames[f"added-{sc['id']}-last"]=b['end']-.15
    sync=[]
    for si,bi in [(0,2),(2,2),(5,2),(7,1),(2,5),(4,1)]:
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
    # Approximate phrase positions from audio_query / speedScale=1.08.
    # These are synthesis-clock checks, not forced alignment of the soundtrack.
    phrase_sync=[]
    for si,bi,sentence,offset,action_name,boundary in [
        (2,5,1,2.5519,'V04a negative slope on right','start'),
        (2,5,2,1.6859,'V04a return to zero slope','end'),
        (4,1,1,.0741,'R1.2 multiply prior by likelihood','start'),
        (4,1,2,.0741,'R1.2 normalize area from 0.2 to 1','start'),
    ]:
        beat=timeline[si]['beats'][bi]
        action=next(a for a in beat['actions'] if a['name']==action_name)
        speech=beat['cues'][sentence]['start']+offset
        phrase_sync.append(dict(scene=timeline[si]['id'],action=action_name,
            boundary=boundary,visual_time=action[boundary],phrase_time=speech,
            difference_seconds=action[boundary]-speech))
    for key,t in frames.items():
        subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(t),'-i',str(VIDEO),'-frames:v','1',str(OUT/f'{key}.png')],check=True)
    keys=list(frames)
    for k in range(0,len(keys),4):
        sheet=Image.new('RGB',(1708,2*510),'#202020');draw=ImageDraw.Draw(sheet)
        for j,key in enumerate(keys[k:k+4]):
            x=(j%2)*854;y=(j//2)*510
            sheet.paste(Image.open(OUT/f'{key}.png'),(x,y+30));draw.text((x+10,y+8),f'{key} / {frames[key]:.3f}s',fill='white')
        sheet.save(OUT/f'sheet-{k//4+1:02}.png')
    (OUT/'frames.json').write_text(json.dumps(dict(frames=frames,symbolic=symbolic,sync=sync,additions=additions,phrase_sync=phrase_sync),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(count=len(frames),symbolic=len(symbolic),sync=sync),indent=2))

if __name__=='__main__': main()
