"""Extract each beat, every symbolic caption, and three PCM/action comparisons."""
import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from narration_content import SCENES

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_4_3_probabilistic_discriminative_models/480p15/PRML43ProbabilisticDiscriminativeModels.mp4'

def run(args):
    return subprocess.run(args,capture_output=True,text=True,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/tmp/prml43-review'));args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml43_timeline.json').read_text())
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
    probe=json.loads(run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]).stdout)
    silence=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af','silencedetect=noise=-45dB:d=3','-f','null','-']).stderr
    volume=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','volumedetect','-f','null','-']).stderr
    (args.output/'silence.log').write_text(silence);(args.output/'volume.log').write_text(volume)
    assert 'silence_start' not in silence
    streams={s['codec_type']:float(s['duration']) for s in probe['streams']}
    assert abs(streams['video']-streams['audio'])<.1
    assert abs(streams['video']-timeline[-1]['end'])<.1
    frames=[];sync=[];max_clock_error=0
    for scene,entry in zip(timeline,manifest):
        assert abs(scene['end']-scene['start']-entry['duration'])<.067
        actual=[c for b in scene['beats'] for c in b['cues']]
        for cue,expected in zip(actual,entry['subtitle_cues']):
            assert cue['id']==expected['id'] and cue['display']==expected['display']
            err=abs(cue['start']-scene['start']-expected['start']);max_clock_error=max(max_clock_error,err)
        for i,beat in enumerate(scene['beats']):
            for fraction in [.15, .50, .85]:
                frames.append(dict(id=f"{scene['id']}-beat{i+1:02}-{int(fraction*100):02}",
                                   time=beat['start']+fraction*(beat['end']-beat['start']),kind='beat'))
            for cue in beat['cues']:
                if '$' in cue['display']:
                    frames.append(dict(id=cue['id']+'-math',time=(cue['start']+cue['end'])/2,kind='math',display=cue['display']))
        if scene['id'] in ['scene01','scene05','scene07']:
            anchor={'scene01':'scene01-04-01','scene05':'scene05-04-01','scene07':'scene07-03-01'}[scene['id']]
            bi=next(i for i,b in enumerate(scene['beats']) if any(c['id']==anchor for c in b['cues']))
            beat=scene['beats'][bi];cue=beat['cues'][0]
            with wave.open(str(ROOT/entry['path']),'rb') as f:
                rate=f.getframerate();pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')/32768
            start=int((cue['start']-scene['start'])*rate);end=int((cue['end']-scene['start'])*rate)
            crossings=np.flatnonzero(abs(pcm[start:end])>10**(-45/20))
            onset=scene['start']+(start+int(crossings[0]))/rate
            sync.append(dict(scene=scene['id'],note=beat['note'],sentence=cue['id'],pcm_onset=onset,actions=beat['actions']))
            for fct in [.15,.85]:
                frames.append(dict(id=f"{scene['id']}-sync-{fct}",time=beat['start']+fct*(beat['end']-beat['start']),kind='sync'))
    review_sync=[]
    for scene,entry in zip(timeline,manifest):
        with wave.open(str(ROOT/entry['path']),'rb') as source:
            rate=source.getframerate()
            pcm=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2')/32768
        for beat in scene['beats']:
            key=beat['cues'][0]['id']
            review=('-recap-' in key or '-aid-' in key)
            label=any(c['id'] in ['scene02-04-02','scene04-04-02'] for c in beat['cues'])
            if not (review or label):
                continue
            for name,t in [('before',beat['start']-.2),('after',beat['end']+.2)]:
                frames.append(dict(id=key+'-'+name,time=t,kind='review-context'))
            for i,cue in enumerate(beat['cues']):
                for fraction in [.15,.85]:
                    frames.append(dict(id=f"{key}-sentence{i+1}-{fraction}",
                        time=cue['start']+fraction*(cue['end']-cue['start']),kind='review-sentence'))
            if review:
                comparisons=[]
                for action in beat['actions']:
                    if action.get('name')=='breath':
                        continue
                    lo=round((action['start']-scene['start'])*rate)
                    hi=round((action['end']-scene['start'])*rate)
                    active=np.flatnonzero(abs(pcm[lo:hi])>10**(-45/20))
                    assert len(active), action
                    comparisons.append(dict(**action,pcm_first_active=scene['start']+(lo+int(active[0]))/rate,
                        pcm_last_active=scene['start']+(lo+int(active[-1]))/rate))
                review_sync.append(dict(id=key,actions=comparisons,cues=beat['cues']))
    assert len(review_sync)==5
    for f in frames:
        path=args.output/(f['id']+'.png')
        run(['ffmpeg','-v','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(path)])
    for start in range(0,len(frames),6):
        batch=frames[start:start+6]
        sheet=Image.new('RGB',(1708,3*510),'#202633');draw=ImageDraw.Draw(sheet)
        for j,f in enumerate(batch):
            x=(j%2)*854;y=(j//2)*510
            sheet.paste(Image.open(args.output/(f['id']+'.png')),(x,y+28))
            draw.text((x+10,y+6),f"{f['id']}  {f['time']:.3f}s",fill='white')
        sheet.save(args.output/f'sheet-{start//6+1:02}.png')
    result=dict(probe=probe,silence_intervals=0,volume=[s.strip() for s in volume.splitlines() if 'mean_volume:' in s or 'max_volume:' in s],frames=frames,sync=sync,max_caption_clock_error=max_clock_error,
                review_sync=review_sync,mp4_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),visual_review='pending',full_listening=False)
    (ROOT/'validation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False,indent=2))
    print('frames',len(frames),'sheets',(len(frames)+5)//6)

if __name__=='__main__':main()
