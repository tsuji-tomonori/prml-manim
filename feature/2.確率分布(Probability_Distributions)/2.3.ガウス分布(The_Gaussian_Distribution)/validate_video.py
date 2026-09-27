"""Extract review frames and audio diagnostics from the final MP4.

PNG files and logs go under ignored media/review. Extraction is not visual review.
"""
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_2_3_gaussian_distribution/480p15/PRML23GaussianDistribution.mp4'
OUT=ROOT/'media/review'
OUT.mkdir(parents=True,exist_ok=True)

def run(args):
    return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

probe=json.loads(run(['ffprobe','-v','error','-show_entries','format=duration,size:stream=index,codec_name,codec_type,width,height,r_frame_rate,duration','-of','json',str(VIDEO)]).stdout)
(OUT/'ffprobe.json').write_text(json.dumps(probe,indent=2)+'\n')
for name,filter_ in [('silence','silencedetect=noise=-45dB:d=3'),('volume','volumedetect')]:
    result=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af',filter_,'-f','null','-'])
    (OUT/f'{name}.log').write_text(result.stderr)

timeline=json.loads((ROOT/'media/prml23_timeline.json').read_text())
manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
frames=[];sync=[]
for scene,entry in zip(timeline,manifest):
    assert abs(scene['end']-scene['start']-entry['duration'])<1e-6
    actual=[c for b in scene['beats'] for c in b['cues']]
    for c,a in zip(actual,entry['subtitle_cues']):
        assert c['id']==a['id'] and c['display']==a['display']
        assert abs(c['start']-scene['start']-a['start'])<1e-6
    for j,b in enumerate(scene['beats'],1):
        cue=b['cues'][-1]
        frames.append(dict(name=f"{scene['id']}-{j:02}",time=(cue['start']+cue['end'])/2,display=cue['display']))
    for b in scene['beats']:
        for c in b['cues']:
            if '$' in c['display']:
                frames.append(dict(name=f"math-{c['id']}",time=(c['start']+c['end'])/2,display=c['display']))
    if scene['id'] in ['scene01','scene05','scene09']:
        j={'scene01':1,'scene05':2,'scene09':0}[scene['id']]
        b=scene['beats'][j];cue=b['cues'][0]
        with wave.open(str(ROOT/entry['path']),'rb') as wav:
            rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(float)/32768
        start=round((cue['start']-scene['start'])*rate);end=round((cue['end']-scene['start'])*rate)
        active=np.flatnonzero(abs(pcm[start:end])>10**(-45/20))
        assert len(active)
        onset=scene['start']+(start+active[0])/rate
        sync.append(dict(scene=scene['id'],beat=j+1,action_start=b['action_start'],action_end=b['action_end'],speech_onset=onset,first_sentence_end=cue['end']))
        for suffix,alpha in [('early',.15),('late',.85)]:
            frames.append(dict(name=f"sync-{scene['id']}-{suffix}",time=b['action_start']+alpha*(b['action_end']-b['action_start']),display=b['display']))

for f in frames:
    run(['ffmpeg','-v','error','-y','-ss',f"{f['time']:.6f}",'-i',str(VIDEO),'-frames:v','1',str(OUT/(f['name']+'.png'))])
for scene in timeline:
    subset=[f for f in frames if f['name'].startswith(scene['id']+'-')]
    canvas=Image.new('RGB',(1708,1512),'#252525')
    draw=ImageDraw.Draw(canvas)
    for i,f in enumerate(subset):
        x=i%2*854;y=i//2*504;canvas.paste(Image.open(OUT/(f['name']+'.png')),(x,y));draw.text((x+10,y+482),f"{f['name']} @ {f['time']:.3f}s",fill='white')
    canvas.save(OUT/(scene['id']+'-sheet.jpg'))
extra=[f for f in frames if f['name'].startswith(('math-','sync-'))]
for k in range(0,len(extra),4):
    canvas=Image.new('RGB',(1708,1008),'#252525');draw=ImageDraw.Draw(canvas)
    for i,f in enumerate(extra[k:k+4]):
        x=i%2*854;y=i//2*504;canvas.paste(Image.open(OUT/(f['name']+'.png')),(x,y));draw.text((x+10,y+482),f"{f['name']} @ {f['time']:.3f}s",fill='white')
    canvas.save(OUT/f'extra-{k//4+1}.jpg')
(OUT/'frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2)+'\n')
(OUT/'sync.json').write_text(json.dumps(sync,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'probe':probe,'frame_count':len(frames),'sync':sync},indent=2))
