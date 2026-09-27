"""Extract every beat, every math caption, and before/after synchronization frames.

Images and acoustic logs are evidence for human review, not automated layout approval.
"""
import argparse
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_2_5_nonparametric_methods/480p15/PRML25NonparametricMethods.mp4'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'media/review')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml25_timeline.json').read_text())
    manifest=json.loads(MANIFEST.read_text())['scenes']
    samples=[];sync=[];max_delta=0
    for story,t,e in zip(SCENES,timeline,manifest):
        assert story['id']==t['id']==e['id']
        max_delta=max(max_delta,abs(t['end']-t['start']-e['duration']))
        mc={c['id']:c for c in e['subtitle_cues']}
        for bi,b in enumerate(t['beats']):
            samples.append(dict(name=f"{t['id']}-beat{bi+1:02}",time=(b['start']+b['end'])/2,kind='beat'))
            for c in b['cues']:
                assert c['display']==mc[c['id']]['display']
                assert abs(c['start']-t['start']-mc[c['id']]['start'])<1/15
                if '$' in c['display']:
                    samples.append(dict(name=c['id']+'-math',time=(c['start']+c['end'])/2,kind='math',display=c['display']))
        selected={'scene03':1,'scene05':3,'scene07':6}
        if t['id'] in selected:
            b=t['beats'][selected[t['id']]]
            with wave.open(str(OUTPUT_DIR/f"{t['id']}.wav"),'rb') as wav:
                rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')/32768.
            c=b['cues'][0];off=round((c['start']-t['start'])*rate)
            end=round((c['end']-t['start'])*rate)
            nz=np.flatnonzero(np.abs(pcm[off:end])>10**(-45/20))
            onset=t['start']+(off+int(nz[0]))/rate
            sync.append(dict(scene=t['id'],beat=selected[t['id']]+1,sentence=c['id'],speech_onset=onset,
                             action_start=b['action_start'],action_end=b['action_end']))
            for f in [.15,.85]:
                samples.append(dict(name=t['id']+f'-sync-{f}',time=b['action_start']+f*(b['action_end']-b['action_start']),kind='sync'))
    assert max_delta<1/15
    for s in samples:
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(s['time']),'-i',str(VIDEO),'-frames:v','1',str(out/(s['name']+'.png'))],check=True)
    for start in range(0,len(samples),4):
        sheet=Image.new('RGB',(1708,1020),'#202020');draw=ImageDraw.Draw(sheet)
        for j,s in enumerate(samples[start:start+4]):
            image=Image.open(out/(s['name']+'.png')).convert('RGB')
            x=(j%2)*854;y=(j//2)*510
            sheet.paste(image,(x,y+30));draw.text((x+12,y+8),f"{s['name']}  {s['time']:.3f}s",fill='white')
        sheet.save(out/f'sheet-{start//4+1:02}.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size:stream=index,codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]))
    logs={}
    for name,flt in [('silence','silencedetect=noise=-45dB:d=3'),('volume','volumedetect')]:
        r=subprocess.run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af',flt,'-f','null','-'],capture_output=True,text=True,check=True)
        (out/f'{name}.log').write_text(r.stderr);logs[name]=[l.strip() for l in r.stderr.splitlines() if 'silence_' in l or 'mean_volume:' in l or 'max_volume:' in l]
    result=dict(frames=samples,sync=sync,scene_duration_max_delta=max_delta,probe=probe,audio_logs=logs)
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(frames=len(samples),math_frames=sum(s['kind']=='math' for s in samples),sync=sync,probe=probe,audio_logs=logs),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
