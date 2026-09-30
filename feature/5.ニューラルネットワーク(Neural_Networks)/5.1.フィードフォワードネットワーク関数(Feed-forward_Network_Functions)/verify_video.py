"""Extract reproducible review frames and measure the final muxed audio/video."""
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
VIDEO=ROOT/'media/videos/prml_5_1_feed_forward_network_functions/480p15/PRML51FeedForwardNetworkFunctions.mp4'


def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    dest=args.output;dest.mkdir(parents=True,exist_ok=True)
    timeline=json.loads((ROOT/'media/prml51_timeline.json').read_text())
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
    probe=json.loads(run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]).stdout)
    audio=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af','silencedetect=noise=-45dB:d=3,volumedetect','-f','null','-']).stderr
    (dest/'ffprobe.json').write_text(json.dumps(probe,indent=2)+'\n');(dest/'audio.log').write_text(audio)
    silence=len(re.findall('silence_start:',audio));assert silence==0
    volumes={k:float(re.search(k+r':\s*([-\d.]+)',audio)[1]) for k in ['mean_volume','max_volume']}
    streams={s['codec_type']:s for s in probe['streams']}
    gap=abs(float(streams['video']['duration'])-float(streams['audio']['duration']));assert gap<1/15
    frames=[];sync=[];reviews=[];max_gap=0.;cue_count=0
    for si,(s,e) in enumerate(zip(timeline,manifest)):
        max_gap=max(max_gap,abs(s['end']-s['start']-e['duration']))
        assert abs(s['end']-s['start']-e['duration'])<1/15
        for bi,b in enumerate(s['beats']):
            frames.append(dict(label=f"{s['id']}-beat{bi+1:02}",time=b['start']+.7*(b['end']-b['start'])))
            for c in b['cues']:
                cue_count+=1
                expected=next(x for x in e['subtitle_cues'] if x['id']==c['id'])
                assert c['display']==expected['display']
                assert abs(c['start']-s['start']-expected['start'])<1e-5
                if '$' in c['display']:
                    frames.append(dict(label='math-'+c['id'],time=(c['start']+c['end'])/2))
        for b in s['beats']:
            if not b['note'].startswith(('復習:', '補足:')):
                continue
            with wave.open(str(ROOT/e['path'])) as f:
                rate=f.getframerate()
                pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')/32768.
            checks=[]
            assert len(b['cues'])==len(b['actions'])
            for c,action in zip(b['cues'],b['actions']):
                lo=round((c['start']-s['start'])*rate)
                hi=round((c['end']-s['start'])*rate)
                hits=np.flatnonzero(abs(pcm[lo:hi])>10**(-45/20))
                assert len(hits)>0
                onset=c['start']+float(hits[0])/rate
                end=c['start']+float(hits[-1])/rate
                assert abs(action['start']-c['start'])<=1/15
                assert action['start']<=onset<end<=action['end']+1/15
                checks.append(dict(sentence=c['id'],cue_start=c['start'],speech_onset=onset,speech_end=end,
                                   action_start=action['start'],action_end=action['end']))
                for q in [.15,.85]:
                    frames.append(dict(label=f"review-{c['id']}-{q}",
                                       time=c['start']+q*(c['end']-c['start'])))
            for label,t in [('before',b['start']-.2),('after',b['end']+.2)]:
                frames.append(dict(label=f"review-{s['id']}-{label}",time=t))
            reviews.append(dict(scene=s['id'],note=b['note'],start=b['start'],end=b['end'],
                                duration=b['end']-b['start'],sentence_sync=checks))
        if s['id']=='scene05':
            conversion=next(b for b in s['beats'] if b['cues'][0]['id']=='scene05-05-01')
            for label,t in [('before',conversion['start']-.2),
                            ('onset',conversion['start']+.2),
                            ('during',conversion['cues'][0]['end']-.2),
                            ('after',conversion['end']+.2)]:
                frames.append(dict(label='output-normalization-'+label,time=t))
        # Compare early/late animation states against the first sentence's PCM onset.
        selected={1:3,5:4,8:2}
        if si in selected:
            bi=selected[si];b=s['beats'][bi];c=b['cues'][0]
            with wave.open(str(ROOT/e['path'])) as f:
                rate=f.getframerate(); pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')/32768.
            lo=round((c['start']-s['start'])*rate);hi=round((c['end']-s['start'])*rate)
            hits=np.flatnonzero(abs(pcm[lo:hi])>10**(-45/20))
            onset=c['start']+float(hits[0])/rate
            times=[b['start']+.15*(b['end']-b['start']),b['start']+.85*(b['end']-b['start'])]
            for j,t in enumerate(times):frames.append(dict(label=f"sync-{s['id']}-{j}",time=t))
            sync.append(dict(scene=s['id'],beat=bi+1,note=b['note'],start=b['start'],end=b['end'],speech_onset=onset,frames=times))
    swap=timeline[8]['beats'][4]['cues'][0]
    for q in [.2,.5,.8]:
        frames.append(dict(label=f'swap-{q}',time=swap['start']+q*(swap['end']-swap['start'])))
    for i,f in enumerate(frames):
        f['file']=f'{i:03}-{f["label"]}.png'
        run(['ffmpeg','-v','error','-y','-ss',str(f['time']),'-i',str(VIDEO),'-frames:v','1',str(dest/f['file'])])
    # Keep each 854x480 frame at native resolution; labels outside video pixels.
    for start in range(0,len(frames),6):
        sheet=Image.new('RGB',(1708,1512),'#202020');draw=ImageDraw.Draw(sheet)
        for j,f in enumerate(frames[start:start+6]):
            x=(j%2)*854;y=(j//2)*504
            sheet.paste(Image.open(dest/f['file']),(x,y+24));draw.text((x+8,y+5),f['label']+f"  {f['time']:.3f}s",fill='white')
        sheet.save(dest/f'sheet-{start//6:02}.png')
    result=dict(video_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),probe=probe,stream_gap=gap,
                silence_intervals=silence,**volumes,scenes=len(timeline),cues=cue_count,max_scene_duration_gap=max_gap,
                frames=frames,sync=sync,review_cards=reviews,visual_review='Pending human inspection of extracted frames',full_listening=False)
    (ROOT/'validation_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['frames','probe']},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
