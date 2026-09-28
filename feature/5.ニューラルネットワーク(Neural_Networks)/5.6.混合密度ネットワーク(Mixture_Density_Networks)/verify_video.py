"""Extract every beat, every symbolic caption, and three synchronized motion pairs.

The contact sheets require a separate human/visual review; no script claims that review.
"""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,wave
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_5_6_mixture_density_networks/480p15/PRML56MixtureDensityNetworks.mp4'


def run(args):return subprocess.run(args,check=True,capture_output=True,text=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('/tmp/prml56-review'));args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    tl=json.loads((ROOT/'media/prml56_timeline.json').read_text())
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())
    streams=json.loads(run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]).stdout)
    silence=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af','silencedetect=noise=-45dB:d=3','-f','null','-']).stderr
    volume=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','volumedetect','-f','null','-']).stderr
    (args.output/'silence.log').write_text(silence);(args.output/'volume.log').write_text(volume)
    image_rows=[]; sync=[];max_error=0
    for scene,entry in zip(tl,manifest['scenes']):
        assert scene['audio'] and scene['id']==entry['id']
        cues=[c for b in scene['beats'] for c in b['cues']]
        for a,b in zip(cues,entry['subtitle_cues']):
            assert a['id']==b['id'] and a['display']==b['display']
            max_error=max(max_error,abs(a['start']-scene['start']-b['start']))
        for j,b in enumerate(scene['beats']):
            image_rows.append(dict(name=f"{scene['id']}-beat{j+1:02}",time=.3*b['start']+.7*b['end']))
            for c in b['cues']:
                if '$' in c['display']:image_rows.append(dict(name=c['id']+'-math',time=(c['start']+c['end'])/2))
    for scene_index,beat_index in [(1,1),(2,1),(6,2)]:
        scene=tl[scene_index];beat=scene['beats'][beat_index];entry=manifest['scenes'][scene_index]
        with wave.open(str(ROOT/entry['path'])) as wav:
            rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(float)/32768
        offset=beat['action_start']-scene['start']
        pcm_start=round(offset*rate);pcm_end=round((beat['action_end']-scene['start'])*rate)
        active=np.flatnonzero(np.abs(pcm[pcm_start:pcm_end])>10**(-45/20))
        onset=scene['start']+(pcm_start+int(active[0]))/rate
        rec=dict(scene=scene['id'],visual=beat['visual'],action_start=beat['action_start'],action_end=beat['action_end'],pcm_onset=onset)
        sync.append(rec)
        for fraction in [.15,.85]:
            image_rows.append(dict(name=f"{scene['id']}-sync-{fraction}",time=beat['action_start']+fraction*(beat['action_end']-beat['action_start'])))
    def extract(row):
        path=args.output/(row['name']+'.png')
        run(['ffmpeg','-v','error','-y','-ss',str(row['time']),'-i',str(VIDEO),'-frames:v','1',str(path)])
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(extract,image_rows))
    for start in range(0,len(image_rows),6):
        subset=image_rows[start:start+6]
        sheet=Image.new('RGB',(1708,1524),'#24262c');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(subset):
            x=(j%2)*854;y=(j//2)*508
            sheet.paste(Image.open(args.output/(row['name']+'.png')),(x,y+28))
            draw.text((x+12,y+7),f"{row['name']} / {row['time']:.3f}s",fill='white')
        sheet.save(args.output/f'sheet-{start//6:02}.jpg',quality=95)
    result=dict(video_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),probe=streams,
                long_silences=len(re.findall('silence_start:',silence)),
                mean_volume_db=float(re.search(r'mean_volume: ([\-\d.]+)',volume)[1]),
                peak_volume_db=float(re.search(r'max_volume: ([\-\d.]+)',volume)[1]),
                timeline_max_error=max_error,sync=sync,frames=image_rows,
                visual_review='pending',full_listen=False)
    durations={s['codec_type']:float(s['duration']) for s in streams['streams']}
    result['av_duration_difference']=abs(durations['video']-durations['audio'])
    assert result['long_silences']==0 and result['av_duration_difference']<.1 and max_error<1/15
    (ROOT/'validation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False,indent=2))
    print('Extracted',len(image_rows),'frames: visual review still required.')


if __name__=='__main__':main()
