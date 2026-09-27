"""Extract every beat, every formula caption, and three synchronization pairs."""
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_2_4_exponential_family/480p15/PRML24ExponentialFamily.mp4'
OUT=ROOT/'media/review'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml24_timeline.json').read_text())
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())
    frames=[]
    for s in timeline:
        for bi,b in enumerate(s['beats']):
            frames.append(dict(name=f"{s['id']}-beat{bi+1}",time=b['cues'][-1]['start']+.7*(b['cues'][-1]['end']-b['cues'][-1]['start'])))
            for c in b['cues']:
                if '$' in c['display']:
                    frames.append(dict(name=c['id']+'-math',time=(c['start']+c['end'])/2))
    sync=[]
    for scene_i,beat_i,cue_i in [(2,1,1),(3,0,1),(7,3,0)]:
        s=timeline[scene_i]; c=s['beats'][beat_i]['cues'][cue_i]
        with wave.open(str(ROOT/f"assets/voicevox/{s['id']}.wav"),'rb') as w:
            rate=w.getframerate(); pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')/32768
        lo=round((c['audio_start']-s['start'])*rate);hi=round((c['audio_end']-s['start'])*rate)
        audible=np.flatnonzero(abs(pcm[lo:hi])>10**(-45/20))
        onset=s['start']+(lo+audible[0])/rate
        sync.append(dict(id=c['id'],display=c['display'],onset=onset,action_start=c['start'],action_end=c['end']))
        for fraction in [.15,.85]:
            frames.append(dict(name=c['id']+f'-sync{fraction}',time=c['start']+fraction*(c['end']-c['start'])))
    for i,f in enumerate(frames):
        f['file']=f'{i:03}-{f["name"]}.png'
        subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(OUT/f['file'])],check=True)
    (OUT/'frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2)+'\n')
    cue_error=max(abs(c['start']-c['audio_start']) for s in timeline for b in s['beats'] for c in b['cues'])
    duration_error=max(abs((s['end']-s['start'])-e['duration']) for s,e in zip(timeline,manifest['scenes']))
    result=dict(frame_count=len(frames),symbol_caption_count=sum('$' in c['display'] for s in timeline for b in s['beats'] for c in b['cues']),cue_boundary_max_error=cue_error,scene_duration_max_error=duration_error,sync=sync)
    (OUT/'sync.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for start in range(0,len(frames),4):
        sheet=Image.new('RGB',(854*2,510*2),'#222222');draw=ImageDraw.Draw(sheet)
        for j,f in enumerate(frames[start:start+4]):
            x=(j%2)*854;y=(j//2)*510
            sheet.paste(Image.open(OUT/f['file']),(x,y+30))
            draw.text((x+10,y+7),f"{f['name']}  {f['time']:.3f}s",fill='white')
        sheet.save(OUT/f'sheet-{start//4:02}.png')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
