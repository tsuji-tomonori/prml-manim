"""Extract every mathematical subtitle and two or more frames per scene."""
import json
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_5_5_regularization_in_neural_networks/480p15/PRML55RegularizationInNeuralNetworks.mp4'
OUT=ROOT/'media/review'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'scene_timeline.json').read_text())
    frames={}
    for scene in timeline:
        for i,b in enumerate(scene['beats']):
            # Every beat, plus every mathematical cue.
            c=b['cues'][-1];frames[c['id']]=(c['start']+c['end'])/2
            for c in b['cues']:
                if '$' in c['display']:frames[c['id']]=(c['start']+c['end'])/2
        for b in scene['beats']:
            first=b['cues'][0]['id']
            if 'recap-' in first or 'aid-' in first or first in ('scene06-03-01','scene07-03-01'):
                frames[first+'-before']=max(0,b['start']-.2)
                frames[first+'-after']=b['end']+.2
                for cue,action in zip(b['cues'],b['actions']):
                    for fraction in (.1,.5,.9):
                        frames[f"{cue['id']}-sync-{fraction}"]=action['start']+(action['end']-action['start'])*fraction
    rows=[]
    for name,t in sorted(frames.items(),key=lambda item:item[1]):
        output=OUT/f'{name}.png'
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',str(VIDEO),'-frames:v','1',str(output)],check=True)
        rows.append(dict(id=name,time=t,file=output.name))
    for start in range(0,len(rows),6):
        canvas=Image.new('RGB',(1708,1530),'#202020');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows[start:start+6]):
            x=i%2*854;y=i//2*510
            canvas.paste(Image.open(OUT/r['file']),(x,y+25))
            draw.text((x+10,y+5),f"{r['id']} / {r['time']:.3f}s",fill='white')
        canvas.save(OUT/f'sheet-{start//6:02}.png')
    (ROOT/'review_frames.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Extracted',len(rows),'frames')
if __name__=='__main__':main()
