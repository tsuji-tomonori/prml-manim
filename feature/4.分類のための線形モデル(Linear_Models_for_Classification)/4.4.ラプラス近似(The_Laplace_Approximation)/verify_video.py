"""Collect video evidence; image inspection and listening are separate human steps."""
from pathlib import Path
import json
import subprocess
import wave
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_4_4_laplace_approximation/480p15/PRML44LaplaceApproximation.mp4'
OUT=ROOT/'.working/review'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=index,codec_name,codec_type,width,height,r_frame_rate,duration','-of','json',str(VIDEO)]))
    (OUT/'ffprobe.json').write_text(json.dumps(probe,indent=2))
    streams={s['codec_type']:s for s in probe['streams']}
    assert set(streams)>={'video','audio'}
    assert abs(float(streams['video']['duration'])-float(streams['audio']['duration']))<.1
    for key,flt in [('silence','silencedetect=noise=-45dB:d=3'),('volume','volumedetect')]:
        r=subprocess.run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af',flt,'-f','null','-'],capture_output=True,text=True,check=True)
        (OUT/(key+'.log')).write_text(r.stderr)
        if key=='silence':assert 'silence_start' not in r.stderr
    timeline=json.loads((ROOT/'media/prml44_timeline.json').read_text())
    entries={e['id']:e for e in json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']}
    records=[];sync=[]
    for scene in timeline:
        entry=entries[scene['id']]
        assert abs(scene['end']-scene['start']-entry['duration'])<1/15
        for index,b in enumerate(scene['beats']):
            cue=entry['subtitle_cues'][index]
            assert b['display']==cue['display']
            assert abs(b['start']-scene['start']-cue['start'])<1/15
            # Every beat is sampled near its end, after formula transitions.
            t=b['start']+.80*(b['end']-b['start'])
            records.append(dict(id=b['id'],time=t,display=b['display'],symbol='$' in b['display']))
        selected = [b for b in scene['beats'] if b['id'] in
                    ['scene03-03-01','scene06-04-01','scene08-04-01',
                     'scene05-aid-02','scene06-aid-01']]
        for b in selected:
            cue=next(c for c in entry['subtitle_cues'] if c['id']==b['id'])
            with wave.open(str(ROOT/'assets/voicevox'/f"{scene['id']}.wav"),'rb') as audio:
                rate=audio.getframerate();pcm=np.frombuffer(audio.readframes(audio.getnframes()),dtype=np.int16)/32768.
            region=pcm[round(cue['start']*rate):round(cue['end']*rate)]
            active=np.flatnonzero(abs(region)>10**(-45/20))
            sync.append(dict(id=b['id'],action=b['action'],animation_start=b['start'],voice_start=scene['start']+cue['start']+active[0]/rate,voice_end=scene['start']+cue['start']+active[-1]/rate,animation_end=b['end']))
        added = [i for i,b in enumerate(scene['beats']) if '-aid-' in b['id'] or '-recap-' in b['id']]
        for i in added:
            b=scene['beats'][i]
            for fraction in [.1,.5]:
                records.append(dict(id=b['id']+f'-sync-{fraction}',time=b['start']+fraction*(b['end']-b['start']),display=b['display'],symbol=False))
        if added:
            # Include both neighbours of each inserted card, or previous scene.
            first,last=scene['beats'][added[0]],scene['beats'][added[-1]]
            for suffix,t in [('before',max(0,first['start']-.15)),('after',last['end']+.15)]:
                records.append(dict(id=scene['id']+'-card-'+suffix,time=t,display='',symbol=False))
    for r in records:
        dst=OUT/(r['id']+'.png')
        subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(r['time']),'-i',str(VIDEO),'-frames:v','1',str(dst)],check=True)
    # Four native 854x480 frames on each sheet; readable without downscaling.
    for start in range(0,len(records),4):
        sheet=Image.new('RGB',(1708,1020),'#202020');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(records[start:start+4]):
            x=j%2*854;y=j//2*510
            sheet.paste(Image.open(OUT/(r['id']+'.png')),(x,y+28))
            draw.text((x+8,y+5),f"{r['id']}   {r['time']:.3f}s",fill='white')
        sheet.save(OUT/f'sheet-{start//4+1:02}.png')
    (OUT/'frames.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    (OUT/'sync.json').write_text(json.dumps(sync,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(probe=probe,frames=len(records),symbol_frames=sum(r['symbol'] for r in records),sync=sync),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
