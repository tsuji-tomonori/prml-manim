"""Extract each beat, every mathematical caption, and three before/after sync pairs."""
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_4_2_probabilistic_generative_models/480p15/PRML42ProbabilisticGenerativeModels.mp4'
OUT=ROOT.parents[2]/'.working/review/final'
OUT.mkdir(parents=True,exist_ok=True)
timeline=json.loads((ROOT/'media/prml42_timeline.json').read_text())
frames=[]
for s in timeline:
 for i,b in enumerate(s['beats']):
  frames.append(dict(name=f"{s['id']}-beat{i+1:02}",time=b['start']+.7*(b['end']-b['start']),reason='beat'))
  for c in b['cues']:
   if '$' in c['display']:
    frames.append(dict(name=c['id']+'-math',time=(c['start']+c['end'])/2,reason='math',display=c['display']))
for si,bi in [(3,1),(4,4),(6,1)]:
 s=timeline[si];b=s['beats'][bi]
 for fraction in [.15,.85]:
  frames.append(dict(name=f"{s['id']}-sync-{fraction}",time=b['action_start']+fraction*(b['action_end']-b['action_start']),reason='sync'))
for f in frames:
 subprocess.run(['ffmpeg','-v','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(OUT/(f['name']+'.png'))],check=True)
(OUT/'frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2)+'\n')
for page in range((len(frames)+5)//6):
 sheet=Image.new('RGB',(1708,3*504),'#10141F');draw=ImageDraw.Draw(sheet)
 for i,f in enumerate(frames[page*6:page*6+6]):
  x=(i%2)*854;y=(i//2)*504
  sheet.paste(Image.open(OUT/(f['name']+'.png')),(x,y+24))
  draw.text((x+8,y+5),f"{f['name']} / {f['time']:.3f}s",fill='white')
 sheet.save(OUT/f'sheet-{page+1:02}.jpg',quality=95)
print(f'{len(frames)} frames, {sum(f["reason"]=="math" for f in frames)} math captions, {(len(frames)+5)//6} sheets: {OUT}')
