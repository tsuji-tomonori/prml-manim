"""Collect final-MP4 review frames and audio/sync measurements; viewing is manual."""
import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from narration_content import SCENES

VIDEO=Path('media/videos/prml_3_4_bayesian_model_comparison/480p15/PRML34BayesianModelComparison.mp4')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('.working/review'))
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]))
    (out/'ffprobe.json').write_text(json.dumps(probe,indent=2)+'\n')
    streams={s['codec_type']:s for s in probe['streams']}
    assert 'audio' in streams and 'video' in streams
    assert abs(float(streams['audio']['duration'])-float(streams['video']['duration']))<.1
    for name,filter_ in [('silence','silencedetect=noise=-45dB:d=3'),('volume','volumedetect')]:
        result=subprocess.run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af',filter_,'-f','null','-'],capture_output=True,text=True,check=True)
        (out/f'{name}.log').write_text(result.stderr)
        if name=='silence':assert 'silence_start:' not in result.stderr
    timeline=json.loads(Path('media/prml34_timeline.json').read_text())
    manifest=json.loads(Path('assets/voicevox/manifest.json').read_text())['scenes']
    frames=[];sync=[]
    drift=[]
    for story,scene,entry in zip(SCENES,timeline,manifest):
        assert story['id']==scene['id']==entry['id']
        drift.append(abs(scene['end']-scene['start']-entry['duration']))
        cues=[c for b in scene['beats'] for c in b['cues']]
        assert [c['display'] for c in cues]==[s['display'] for b in story['beats'] for s in b['segments']]
        for cue,source in zip(cues,entry['subtitle_cues']):
            assert abs(cue['start']-scene['start']-source['start'])<1/15
        for i,beat in enumerate(scene['beats']):
            frames.append(dict(id=f"{scene['id']}-beat{i+1:02d}",time=beat['end']-.5,reason='beat-end'))
            for cue in beat['cues']:
                if '$' in cue['display']:
                    frames.append(dict(id=cue['id']+'-math',time=(cue['start']+cue['end'])/2,reason='math',display=cue['display']))
    for si,bi in [(3,1),(4,2),(6,4)]:
        scene=timeline[si];beat=scene['beats'][bi];entry=manifest[si]
        start,end=beat['action_start'],beat['action_end']
        with wave.open(entry['path'],'rb') as source:
            rate=source.getframerate();samples=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2')/32768
        first_cue=beat['cues'][0]
        lo=round((first_cue['start']-scene['start'])*rate);hi=round((first_cue['end']-scene['start'])*rate)
        active=np.flatnonzero(np.abs(samples[lo:hi])>10**(-45/20))
        speech_start=scene['start']+(lo+int(active[0]))/rate
        sync.append(dict(scene=scene['id'],beat=bi+1,action=[start,end],speech_start=speech_start,display=first_cue['display']))
        for fraction in [.2,.8]:
            frames.append(dict(id=f"{scene['id']}-sync-{fraction}",time=start+(end-start)*fraction,reason='sync'))
    for scene,entry in zip(timeline,manifest):
        for beat in scene['beats']:
            if not any('recap' in c['id'] for c in beat['cues']):
                continue
            key=beat['cues'][0]['id'].rsplit('-',1)[0]
            assert 10 <= beat['end']-beat['start'] <= 30
            times=[('before',max(0,beat['start']-.4)),('after',beat['end']+.4)]
            for i,cue in enumerate(beat['cues']):
                times.append((f'sentence{i+1}',cue['start']+.8*(cue['end']-cue['start'])))
            with wave.open(entry['path'],'rb') as source:
                rate=source.getframerate()
                samples=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2')/32768
            for i,action in enumerate(beat['actions']):
                if action['name']=='breath':continue
                lo=round((action['start']-scene['start'])*rate)
                hi=round((action['end']-scene['start'])*rate)
                active=np.flatnonzero(np.abs(samples[lo:hi])>10**(-45/20))
                assert len(active)>0,action
                sync.append(dict(scene=scene['id'],action=action['name'],
                                 interval=[action['start'],action['end']],
                                 audible=[scene['start']+(lo+int(active[0]))/rate,
                                          scene['start']+(lo+int(active[-1]))/rate]))
                times.append((f'action{i+1}',(action['start']+action['end'])/2))
            for label,t in times:
                frames.append(dict(id=f'{key}-{label}',time=t,reason='recap'))
    # The zero-card mixture reference has a changed first sentence.
    scene=timeline[6];beat=scene['beats'][0]
    for label,t in [('before',scene['start']-.4),('during',(beat['cues'][0]['start']+beat['cues'][0]['end'])/2),('after',beat['end']+.4)]:
        frames.append(dict(id=f'mixture-reference-{label}',time=t,reason='reference'))
    assert max(drift)<1/15
    for row in frames:
        destination=out/(row['id']+'.png')
        subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(row['time']),'-i',str(VIDEO),'-frames:v','1',str(destination)],check=True)
    for offset in range(0,len(frames),6):
        sheet=Image.new('RGB',(1708,1500),'#222222');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(frames[offset:offset+6]):
            x=j%2*854;y=j//2*500
            sheet.paste(Image.open(out/(row['id']+'.png')),(x,y))
            draw.text((x+10,y+480),f"{row['id']} / {row['time']:.3f} s",fill='white')
        sheet.save(out/f'sheet-{offset//6:02d}.png')
    summary=dict(frames=frames,sync=sync,max_scene_audio_drift=max(drift),sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest())
    (out/'review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='frames'},ensure_ascii=False,indent=2))
    print('Frames:',len(frames),'Math captions:',sum(f['reason']=='math' for f in frames))
    print('These frames still require manual inspection.')

if __name__=='__main__':main()
