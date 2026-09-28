"""Extract each beat, all math captions and early/late synchronization frames."""
import json
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).parent
VIDEO=ROOT/'media/videos/prml_5_4_hessian_matrix/480p15/PRML54HessianMatrix.mp4'
OUT=ROOT/'media/review'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml54_timeline.json').read_text())
    frames=[]
    for s in timeline:
        for i,b in enumerate(s['beats']):
            frames.append(dict(id=f"{s['id']}-beat{i+1}",time=(b['start']+b['end'])/2,kind='beat'))
            for c in b['cues']:
                if '$' in c['display']:
                    frames.append(dict(id=c['id'],time=(c['start']+c['end'])/2,kind='math',display=c['display']))
        if s['id'] in ('scene01','scene04','scene09'):
            b=s['beats'][1 if s['id']=='scene01' else 2 if s['id']=='scene04' else 3]
            action=b['actions'][0]
            for phase,alpha in [('early',.1),('late',.9)]:
                frames.append(dict(id=s['id']+'-'+phase,time=action['start']+alpha*(action['end']-action['start']),kind='sync'))
    for f in frames:
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(OUT/(f['id']+'.png'))],check=True)
    (OUT/'frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2)+'\n')
    # Original-resolution panels: do not lose subtitle readability to thumbnails.
    for i in range(0,len(frames),4):
        sheet=Image.new('RGB',(1708,1010),'#202020'); draw=ImageDraw.Draw(sheet)
        for j,f in enumerate(frames[i:i+4]):
            x=(j%2)*854;y=(j//2)*505
            sheet.paste(Image.open(OUT/(f['id']+'.png')),(x,y+25));draw.text((x+10,y+5),f"{f['id']}  {f['time']:.3f}s",fill='white')
        sheet.save(OUT/f'sheet-{i//4:02}.png')
    print(f'{len(frames)} frames; {sum(f["kind"]=="math" for f in frames)} math captions')
if __name__=='__main__':main()
