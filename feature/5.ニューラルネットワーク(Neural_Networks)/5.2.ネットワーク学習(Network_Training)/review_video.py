"""Extract each beat, all math captions and early/late synchronization frames."""
import json
import argparse
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).parent
VIDEO=ROOT/'media/videos/prml_5_2_network_training/480p15/PRML52NetworkTraining.mp4'
OUT=ROOT/'media/review'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recap",action="store_true",help="Review additions with context, all math captions and one frame per scene")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml52_timeline.json').read_text())
    frames=[]
    for s in timeline:
        for i,b in enumerate(s['beats']):
            if not args.recap or i==len(s["beats"])//2:
                frames.append(dict(id=f"{s['id']}-beat{i+1}",time=(b['start']+b['end'])/2,kind='beat'))
            # Check moving and emphasized objects while the corresponding words
            # are spoken, including the replay of the opening data at the end.
            if (s['id']=='scene01' or s['id']=='scene10' and i>=5
                    or s['id']=='scene09' and i in (1,2,3,4)
                    or s['id']=='scene04' and i in (2,3,4)):
                for phase,alpha in [('early',.12),('middle',.5),('late',.88)]:
                    frames.append(dict(id=f"{s['id']}-beat{i+1}-{phase}",
                                       time=b['start']+alpha*(b['end']-b['start']),kind='motion'))
                frames.append(dict(id=f"{s['id']}-beat{i+1}-after",time=b['end']+.1,kind='motion-after'))
            if any("-recap-" in c["id"] or "-aid-" in c["id"] for c in b["cues"]):
                for phase,t in [("before",b["start"]-.2),("after",b["end"]+.2)]:
                    frames.append(dict(id=b["cues"][0]["id"]+"-"+phase,time=t,kind="context"))
                for c in b["cues"]:
                    for phase,alpha in [("early",.15),("late",.85)]:
                        frames.append(dict(id=c["id"]+"-"+phase,time=c["start"]+alpha*(c["end"]-c["start"]),kind="addition-sync"))
            for c in b['cues']:
                if '$' in c['display']:
                    frames.append(dict(id=c['id'],time=(c['start']+c['end'])/2,kind='math',display=c['display']))
        if s['id']=='scene09':
            b=s['beats'][3]
            frames.append(dict(id='scene09-divergence-end',time=b['end']-.15,kind='endpoint'))
        if s['id'] in ('scene01','scene04','scene09'):
            b=s['beats'][2]
            for phase,alpha in [('early',.1),('late',.9)]:
                frames.append(dict(id=s['id']+'-'+phase,time=b['start']+alpha*(b['end']-b['start']),kind='sync'))
    # A final "after" sample can otherwise fall beyond the last video frame.
    last_frame_time=timeline[-1]['end']-.2
    for f in frames:
        f['time']=min(max(f['time'],0),last_frame_time)
        frame_path=OUT/(f['id']+'.png')
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(frame_path)],check=True)
        if not frame_path.exists():
            raise RuntimeError(f"No frame at {f['time']:.3f}s for {f['id']}")
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
