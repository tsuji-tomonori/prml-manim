"""Extract every beat, math caption, visual aid, and PCM synchronization."""
import argparse
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_2_2_multinomial_variables/480p15/PRML22MultinomialVariables.mp4'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'media/review')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml22_timeline.json').read_text())
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
    frames=[];sync=[]
    for si,s in enumerate(timeline):
        for bi,b in enumerate(s['beats']):
            frames.append(dict(id=f"{s['id']}-beat{bi+1:02}",time=b['start']+.87*(b['end']-b['start'])))
            for c in b['cues']:
                if '$' in c['display']:
                    frames.append(dict(id=c['id']+'-math',time=(c['start']+c['end'])/2))
            # Inspect moving states as well as the end of the action.
            story_beats={1:{1,2,7},4:{2,3,4},6:{2,3,4},
                         7:{2,3,4,6},8:{2,4,5},9:{1,2,5,7,8}}
            if bi+1 in story_beats.get(si+1,set()):
                for fraction in (.15,.50,.85):
                    frames.append(dict(id=f"{s['id']}-beat{bi+1:02}-move-{fraction:.2f}",
                                       time=b['action_start']+fraction*(b['action_end']-b['action_start'])))
        if si in (1,3,7):
            anchor=f"{s['id']}-02-01"
            b=next(b for b in s['beats'] if any(c['id']==anchor for c in b['cues']))
            for fraction in (.12,.88):
                frames.append(dict(id=f"{s['id']}-sync-{fraction}",time=b['action_start']+fraction*(b['action_end']-b['action_start'])))
            with wave.open(str(ROOT/manifest[si]['path']),'rb') as wav:
                rate=wav.getframerate();audio=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')/32768
            starts=[]
            for c in b['cues']:
                offset=c['start']-s['start'];end=c['end']-s['start']
                samples=audio[round(offset*rate):round(end*rate)]
                hit=np.flatnonzero(np.abs(samples)>10**(-45/20))
                starts.append(c['start']+float(hit[0]/rate) if len(hit) else None)
            sync.append(dict(scene=s['id'],beat=2,action_start=b['action_start'],action_end=b['action_end'],pcm_voice_starts=starts))
    # Both visual aids: before/after, each sentence, and both ends of every action.
    for si,s in enumerate(timeline):
        for b in s['beats']:
            if not any('-aid-' in c['id'] or '-recap-' in c['id'] for c in b['cues']):
                continue
            prefix=b['cues'][0]['id'].rsplit('-',1)[0]
            frames.extend([dict(id=prefix+'-before',time=max(0,b['start']-.3)),
                           dict(id=prefix+'-after',time=b['end']+.3)])
            with wave.open(str(ROOT/manifest[si]['path']),'rb') as wav:
                rate=wav.getframerate();audio=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')/32768
            audit=[]
            for c,action in zip(b['cues'],[a for a in b['actions'] if a['name']!='breath']):
                for fraction in (.15,.85):
                    frames.append(dict(id=c['id']+f'-sync-{fraction}',time=action['start']+fraction*(action['end']-action['start'])))
                offset=c['start']-s['start'];end=c['end']-s['start']
                samples=audio[round(offset*rate):round(end*rate)]
                hit=np.flatnonzero(np.abs(samples)>10**(-45/20))
                onset=c['start']+float(hit[0]/rate) if len(hit) else None
                assert abs(action['start']-c['start']) <= 1/15+.001
                assert abs(action['end']-c['end']) <= 1/15+.001
                audit.append(dict(id=c['id'],action=action,caption_start=c['start'],caption_end=c['end'],pcm_voice_start=onset))
            sync.append(dict(scene=s['id'],card=prefix,sentences=audit))
    for row in frames:
        row['file']=row['id']+'.png'
        subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(row['time']),'-i',str(VIDEO),'-frames:v','1',str(out/row['file'])],check=True)
    for k in range(0,len(frames),6):
        sheet=Image.new('RGB',(854*2,510*3),'#20242f');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(frames[k:k+6]):
            x=(j%2)*854;y=(j//2)*510
            sheet.paste(Image.open(out/row['file']),(x,y+30))
            draw.text((x+12,y+8),f"{row['id']} / {row['time']:.3f}s",fill='white')
        sheet.save(out/f'sheet-{k//6+1:02}.jpg',quality=94)
    result=dict(frames=frames,synchronization=sync,scene_durations=[dict(id=s['id'],duration=s['end']-s['start']) for s in timeline],
                max_scene_audio_difference=max(abs(s['end']-s['start']-e['duration']) for s,e in zip(timeline,manifest)))
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'Extracted {len(frames)} frames; {sum("math" in r["id"] for r in frames)} mathematical subtitles; {len(sync)} sync groups.')
if __name__=='__main__':main()
