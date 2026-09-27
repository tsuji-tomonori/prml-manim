"""Reproduce media, timing and frame extraction checks for the 480p15 artifact."""
import argparse
import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_3_6_fixed_basis_limitations/480p15/PRML36FixedBasisLimitations.mp4'

def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True,help='Temporary review directory, outside version control')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    probe=json.loads(run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]).stdout)
    v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
    delta=abs(float(v['duration'])-float(a['duration']));assert delta<1/15
    sil=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af','silencedetect=noise=-45dB:d=3','-f','null','-']).stderr
    vol=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','volumedetect','-f','null','-']).stderr
    (out/'silence.log').write_text(sil);(out/'volume.log').write_text(vol)
    silence_count=len(re.findall('silence_start:',sil));assert silence_count==0
    volumes={k:float(re.search(k+r':\s*(-?[\d.]+) dB',vol)[1]) for k in ['mean_volume','max_volume']}
    timeline=json.loads((ROOT/'media/prml36_timeline.json').read_text())
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
    frames=[];sync=[];timing_errors=[]
    for s,e in zip(timeline,manifest):
        timing_errors.append(abs(s['end']-s['start']-e['duration']))
        for bi in [1,4,7]:
            b=s['beats'][bi];frames.append(dict(label=f"{s['id']}-beat{bi+1}",time=b['start']+.72*(b['end']-b['start'])))
        for b in s['beats']:
            for c in b['cues']:
                if '$' in c['display']:frames.append(dict(label=c['id']+'-math',time=(c['start']+c['end'])/2))
        flattened=[c for b in s['beats'] for c in b['cues']]
        assert len(flattened)==len(e['subtitle_cues'])
        for c,expected in zip(flattened,e['subtitle_cues']):
            assert c['id']==expected['id'] and c['display']==expected['display']
            assert abs(c['start']-s['start']-expected['start'])<1e-6
    assert max(timing_errors)<1e-6
    for si,bi,name in [(0,1,'weight raises local bump'),(2,4,'straighten manifold'),(4,2,'rotate relevant direction'),(5,2,'shift then sharpen sigmoid')]:
        s=timeline[si];b=s['beats'][bi];e=manifest[si]
        with wave.open(str(ROOT/'assets/voicevox'/f"{s['id']}.wav"),'rb') as w:
            rate=w.getframerate();assert w.getsampwidth()==2
            pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
        cue=e['subtitle_cues'][bi*2]
        window=pcm[round(cue['start']*rate):round(cue['end']*rate)]
        active=np.flatnonzero(abs(window)>10**(-45/20))
        onset=s['start']+cue['start']+active[0]/rate
        sync.append(dict(scene=s['id'],beat=bi+1,action=name,start=b['action_start'],end=b['action_end'],first_voice=onset,phases=b.get('actions',[])))
        for fraction in [.12,.82]:frames.append(dict(label=f"{s['id']}-sync-{fraction}",time=b['start']+fraction*(b['action_end']-b['start'])))
    for i,f in enumerate(frames):
        f['file']=f'frame-{i:02}.png'
        run(['ffmpeg','-loglevel','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(out/f['file'])])
    for start in range(0,len(frames),4):
        sheet=Image.new('RGB',(1708,1000),'#10141f')
        for j,f in enumerate(frames[start:start+4]):
            x=854*(j%2);y=500*(j//2)
            sheet.paste(Image.open(out/f['file']),(x,y))
            ImageDraw.Draw(sheet).text((x+10,y+482),f"{start+j:02} {f['label']} {f['time']:.3f}s",fill='white')
        sheet.save(out/f'sheet-{start//4:02}.png')
    result=dict(video_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),probe=probe,duration_difference=delta,
                silence_intervals_ge_3s=silence_count,**volumes,timeline_max_error=max(timing_errors),
                scenes=[dict(id=s['id'],title=s['title'],start=s['start'],duration=e['duration'],reference=s['reference']) for s,e in zip(timeline,manifest)],
                sync=sync,frames=frames,visual_review='pending',full_listening=False)
    (ROOT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['frames','scenes','sync']},ensure_ascii=False,indent=2))
    print('Extracted frames:',len(frames))

if __name__=='__main__':main()
