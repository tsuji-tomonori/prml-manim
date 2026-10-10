"""Extract every beat, all math captions and PCM onset evidence from final MP4."""
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_4_1_discriminant_functions/480p15/PRML41DiscriminantFunctions.mp4'
OUT=ROOT/'media/review'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml41_timeline.json').read_text())
    manifest={s['id']:s for s in json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']}
    frames=[];sync=[]
    for scene in timeline:
        for i,b in enumerate(scene['beats']):
            frames.append({'label':f"{scene['id']} beat {i+1}",'time':b['start']+.72*(b['end']-b['start'])})
            for c in b['cues']:
                if '$' in c['display']:frames.append({'label':c['id']+' math','time':(c['start']+c['end'])/2})
    # Include every phase of each new card and its neighbouring original shots.
    for si,bi in [(0,0),(3,0),(3,3),(4,0),(4,7),(4,6),(4,8),(9,0)]:
        scene=timeline[si];b=scene['beats'][bi]
        times=[max(0,b['start']-.15),b['start']+.25,b['end']-.2,b['end']+.2]
        times += [(a['start']+a['end'])/2 for a in b.get('actions',[]) if a['name']!='breath']
        for j,t in enumerate(times):
            frames.append({'label':f"{scene['id']} review beat {bi+1} phase {j}",'time':t})
    for si,bi in [(0,0),(0,4),(3,3),(4,5),(7,2),(3,0),(4,0),(4,7),(9,0)]:
        s=timeline[si];b=s['beats'][bi];e=manifest[s['id']]
        with wave.open(str(ROOT/'assets/voicevox'/f"{s['id']}.wav")) as wav:
            rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(float)/32768
        onsets=[]
        for c in b['cues']:
            lo=int((c['start']-s['start'])*rate);hi=int((c['end']-s['start'])*rate)
            indices=np.flatnonzero(abs(pcm[lo:hi])>10**(-45/20))
            onsets.append(float(s['start']+(lo+indices[0])/rate) if len(indices) else None)
        actions=[]
        for action in b.get('actions',[]):
            if action['name']=='breath':continue
            lo=round((action['start']-s['start'])*rate);hi=round((action['end']-s['start'])*rate)
            active=np.flatnonzero(abs(pcm[lo:hi])>10**(-45/20))
            assert len(active)>0,action['name']
            actions.append(dict(action,voiced_samples=len(active),
                first_voice=s['start']+(lo+int(active[0]))/rate,
                last_voice=s['start']+(lo+int(active[-1]))/rate))
        sync.append({'scene':s['id'],'beat':bi+1,'action_start':b['action_start'],'action_end':b['action_end'],
                     'speech_onsets':onsets,'cues':b['cues'],'actions':actions})
        for part in [0.05,.5,.95]:frames.append({'label':f"{s['id']} sync {part}",'time':b['action_start']+part*(b['action_end']-b['action_start'])})
    for i,f in enumerate(frames):
        path=OUT/f'frame-{i:03}.png';f['path']=path.name
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(path)],check=True)
    for i in range(0,len(frames),6):
        canvas=Image.new('RGB',(1708,1512),'#202530');draw=ImageDraw.Draw(canvas)
        for j,f in enumerate(frames[i:i+6]):
            x=(j%2)*854;y=(j//2)*504
            draw.text((x+10,y+4),f"{i+j:03} {f['label']}  {f['time']:.3f}s",fill='white')
            canvas.paste(Image.open(OUT/f['path']),(x,y+24))
        canvas.save(OUT/f'sheet-{i//6:02}.png')
    results={'frames':frames,'sync':sync,'scenes':[{'id':s['id'],'start':s['start'],'duration':s['end']-s['start']} for s in timeline]}
    (ROOT/'validation_video.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(f'Extracted {len(frames)} frames; {sum("math" in f["label"] for f in frames)} math captions')
    print(json.dumps(sync,indent=2))

if __name__=='__main__':main()
