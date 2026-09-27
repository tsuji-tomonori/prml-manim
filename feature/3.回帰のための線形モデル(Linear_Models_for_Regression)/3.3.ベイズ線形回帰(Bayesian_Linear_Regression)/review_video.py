"""Extract every beat, every formula caption, and three motion timing pairs."""
import json
import sys
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).parent
VIDEO=ROOT/'media/videos/prml_3_3_bayesian_linear_regression/480p15/PRML33BayesianLinearRegression.mp4'
OUT=Path('/tmp/prml33-review');OUT.mkdir(exist_ok=True)
timeline=json.loads((ROOT/'media/prml33_timeline.json').read_text())
manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
frames=[];sync=[]
for scene,entry in zip(timeline,manifest):
    assert abs(scene['end']-scene['start']-entry['duration'])<1/15
    for i,beat in enumerate(scene['beats']):
        frames.append(dict(id=f"{scene['id']}-beat{i+1}",time=(beat['start']+beat['end'])/2,reason='beat'))
        for cue in beat['cues']:
            actual=next(c for c in entry['subtitle_cues'] if c['id']==cue['id'])
            assert abs(cue['start']-scene['start']-actual['start'])<1/15
            if '--draft' not in sys.argv:
                assert cue['display']==actual['display']
            if '$' in cue['display']:
                frames.append(dict(id=cue['id'],time=(cue['start']+cue['end'])/2,reason='math',display=cue['display']))
    if scene['id'] in ['scene02','scene06','scene08']:
        bi={'scene02':4,'scene06':3,'scene08':1}[scene['id']]
        beat=scene['beats'][bi]
        with wave.open(str(ROOT/'assets/voicevox'/f"{scene['id']}.wav"),'rb') as w:
            pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')/32768
            sr=w.getframerate()
        for cue in beat['cues']:
            a=round((cue['start']-scene['start'])*sr);b=round((cue['end']-scene['start'])*sr)
            indices=np.flatnonzero(abs(pcm[a:b])>10**(-45/20))
            sync.append(dict(scene=scene['id'],cue=cue['id'],display=cue['display'],cue_start=cue['start'],voice_onset=scene['start']+(a+indices[0])/sr,action_start=beat['action_start'],action_end=beat['action_end']))
        for phase in [.15,.85]:
            frames.append(dict(id=f"{scene['id']}-sync-{phase}",time=beat['start']+phase*(beat['end']-beat['start']),reason='sync'))

def extract(f):
    dest=OUT/(f['id']+'.png')
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(dest)],check=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(extract,frames))
for index in range(0,len(frames),4):
    batch=frames[index:index+4];sheet=Image.new('RGB',(1708,1016),'#202020');draw=ImageDraw.Draw(sheet)
    for j,f in enumerate(batch):
        x=(j%2)*854;y=(j//2)*508
        sheet.paste(Image.open(OUT/(f['id']+'.png')),(x,y+28));draw.text((x+8,y+7),f"{f['id']}  {f['time']:.3f}s",fill='white')
    sheet.save(OUT/f'sheet-{index//4+1:02}.png')
(OUT/'frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2))
(OUT/'sync.json').write_text(json.dumps(sync,ensure_ascii=False,indent=2))
print('frames',len(frames),'math',sum(f['reason']=='math' for f in frames),'sheets',(len(frames)+3)//4)
print(json.dumps(sync,ensure_ascii=False,indent=2))
