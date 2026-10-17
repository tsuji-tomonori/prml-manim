"""Check the completed MP4 and extract review frames; human review is separate."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import wave
import numpy as np
from scipy.signal import correlate
from pathlib import Path
from PIL import Image, ImageDraw
from narration_content import SCENES
from make_voicevox_narration import valid_entry

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_10_4_exponential_family/480p15/PRML104ExponentialFamily.mp4'
TIMELINE=ROOT/'media/prml104_timeline.json'
FRAMES=ROOT/'media/review'
FRAMES.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())
timeline=json.loads(TIMELINE.read_text())
assert len(timeline)==len(SCENES)==len(manifest['scenes'])==8
segments=[segment for story in SCENES for beat in story['beats'] for segment in beat['segments']]
reading=json.loads((ROOT/'reading_check.json').read_text())['sentences']
captions=json.loads((ROOT/'caption_validation.json').read_text())['sentences']
assert [(r['id'],r['display'],r['speech']) for r in reading]==[(s['id'],s['display'],s['speech']) for s in segments]
assert all(r['reading'] and r['kana'] for r in reading)
assert [c['id'] for c in captions]==[s['id'] for s in segments]
errors=[]
for story,entry,scene in zip(SCENES,manifest['scenes'],timeline):
    assert valid_entry(story,entry)
    assert scene['id']==entry['id']==story['id']
    assert scene['audio'] is True
    assert len(scene['beats'])==len(story['beats'])==len(entry['beat_durations'])
    assert [(c['id'],c['display']) for b in scene['beats'] for c in b['cues']]==[(c['id'],c['display']) for c in entry['subtitle_cues']]
    errors.append(abs(scene['end']-scene['start']-entry['duration']))
    for beat in scene['beats']:
        assert beat['start']-1e-6<=beat['action_start']<=beat['action_end']<=beat['end']+1/15
        for cue in beat['cues']:
            assert beat['start']-1/15<=cue['start']<cue['end']<=beat['end']+1/15
assert max(errors)<1/15

def run(command):
    return subprocess.run(command,capture_output=True,text=True,check=True)
probe=json.loads(run(['ffprobe','-v','error','-show_entries','format=duration,size:stream=index,codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]).stdout)
video=next(s for s in probe['streams'] if s['codec_type']=='video')
audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
assert video['codec_name']=='h264' and audio['codec_name']=='aac'
assert int(video['width'])==854 and int(video['height'])==480 and video['r_frame_rate']=='15/1'
assert abs(float(video['duration'])-float(audio['duration']))<.1
assert abs(float(video['duration'])-timeline[-1]['end'])<.1
silence=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','silencedetect=noise=-45dB:d=3','-f','null','-']).stderr
volume=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','volumedetect','-f','null','-']).stderr
assert 'silence_start:' not in silence
run(['ffmpeg','-v','error','-i',str(VIDEO),'-f','null','-'])
# Match each spoken sentence in the compressed soundtrack to its source WAV.
rate=24000
pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(VIDEO),'-map','0:a:0','-ar',str(rate),'-ac','1','-f','s16le','-'])
soundtrack=np.frombuffer(pcm,dtype='<i2').astype(float)
alignment=[]
pad=round(.10*rate)
for entry,scene in zip(manifest['scenes'],timeline):
    with wave.open(str(ROOT/entry['path']),'rb') as wav:
        assert wav.getframerate()==rate and wav.getnchannels()==1 and wav.getsampwidth()==2
        source=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(float)
    for cue in entry['subtitle_cues']:
        local=cue['start']+.25
        end=min(cue['end']-.10,local+1.8)
        ref=source[round(local*rate):round(end*rate)].copy()
        assert len(ref)>rate*.25
        ref-=ref.mean()
        at=round((scene['start']+local)*rate)
        obs=soundtrack[at-pad:at+len(ref)+pad]
        sums=np.r_[0.,np.cumsum(obs)]
        squares=np.r_[0.,np.cumsum(obs*obs)]
        n=len(ref)
        energy=squares[n:]-squares[:-n]-(sums[n:]-sums[:-n])**2/n
        scores=correlate(obs,ref,mode='valid',method='fft')/np.sqrt(np.maximum(energy,1)*(ref@ref))
        best=int(np.argmax(scores))
        row=dict(id=cue['id'],offset_seconds=(best-pad)/rate,correlation=float(scores[best]))
        assert abs(row['offset_seconds'])<1/15 and row['correlation']>.97,row
        alignment.append(row)
shots=[]
for scene in timeline:
    for j,beat in enumerate(scene['beats'],1):
        # Onset and early samples expose entrance/highlight motion and its result.
        for phase,fraction in [('onset',.10),('middle',.55),('after',.94)]:
            time=beat['start']+fraction*(beat['end']-beat['start'])
            shots.append(dict(id=f"{scene['id']}-{j:02d}-{phase}",time=time,phase=phase))
    shots.append(dict(id=f"{scene['id']}-end",time=max(scene['start'],scene['end']-.15),phase='transition'))
for scene in timeline:
    for j,beat in enumerate(scene['beats'],1):
        for k,action in enumerate(beat.get('actions',[]),1):
            if action['name']=='breath':continue
            if action['name']=='panel':
                shots.append(dict(id=f"{scene['id']}-{j:02d}-action-{k}-mid",
                                  time=(action['start']+action['end'])/2,
                                  phase='during-action'))
            # Seeking past the last encoded frame can succeed without writing
            # an image.  Keep every action sample inside the final frame span.
            shots.append(dict(id=f"{scene['id']}-{j:02d}-action-{k}",
                              time=min(beat['end']-.15,action['end']+.03,
                                       timeline[-1]['end']-.15),phase='after-action'))
for shot in shots:
    path=FRAMES/(shot['id']+'.png')
    run(['ffmpeg','-v','error','-y','-ss',f"{shot['time']:.4f}",'-i',str(VIDEO),'-frames:v','1',str(path)])
    shot['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
for start in range(0,len(shots),16):
    sheet=Image.new('RGB',(1920,1184),'#16202b')
    draw=ImageDraw.Draw(sheet)
    for j,shot in enumerate(shots[start:start+16]):
        x=j%4*480;y=j//4*296
        with Image.open(FRAMES/(shot['id']+'.png')) as pic:
            sheet.paste(pic.resize((480,270)),(x,y+26))
        draw.text((x+8,y+6),f"{shot['id']}  {shot['time']:.1f}s",fill='white')
    sheet.save(FRAMES/f'sheet-{start//16+1:02d}.png')
result=dict(duration=float(video['duration']),audio_duration=float(audio['duration']),resolution=[854,480],fps=15,scene_timing_max_error=max(errors),long_silence_count=silence.count('silence_start:'),mean_volume_db=float(re.search(r'mean_volume: ([-0-9.]+)',volume)[1]),max_volume_db=float(re.search(r'max_volume: ([-0-9.]+)',volume)[1]),full_decode='passed',reading_count=len(reading),caption_count=len(captions),video_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),frame_count=len(shots),review_sheet_count=(len(shots)+15)//16,manual_review='Extraction only; see the work report for actual frame review.',audio_alignment=dict(count=len(alignment),minimum_correlation=min(a['correlation'] for a in alignment),maximum_absolute_offset=max(abs(a['offset_seconds']) for a in alignment),sentences=alignment),frames=shots)
(ROOT/'video_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:v for k,v in result.items() if k not in ('frames','video_sha256','audio_alignment')})
