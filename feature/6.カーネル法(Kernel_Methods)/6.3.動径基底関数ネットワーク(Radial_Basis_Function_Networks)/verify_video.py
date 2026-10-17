"""Check the completed MP4 and extract review frames; human review is separate."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
from narration_content import SCENES
from make_voicevox_narration import valid_entry

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_6_3_radial_basis_function_networks/480p15/PRML63RadialBasisFunctionNetworks.mp4'
TIMELINE=ROOT/'media/prml63_timeline.json'
FRAMES=ROOT/'media/review'
FRAMES.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())
timeline=json.loads(TIMELINE.read_text())
assert len(timeline)==len(SCENES)==len(manifest['scenes'])==8
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
assert int(video['width'])==854 and int(video['height'])==480 and video['r_frame_rate']=='15/1'
assert abs(float(video['duration'])-float(audio['duration']))<.1
assert abs(float(video['duration'])-timeline[-1]['end'])<.1
silence=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','silencedetect=noise=-45dB:d=3','-f','null','-']).stderr
volume=run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','volumedetect','-f','null','-']).stderr
assert 'silence_start:' not in silence
run(['ffmpeg','-v','error','-i',str(VIDEO),'-f','null','-'])
shots=[]
for scene in timeline:
    for j,beat in enumerate(scene['beats'],1):
        # The early and late samples expose entrances, highlights and exits.
        for phase,fraction in [('early',.18),('middle',.55),('late',.86)]:
            time=beat['start']+fraction*(beat['end']-beat['start'])
            shots.append(dict(id=f"{scene['id']}-{j:02d}-{phase}",time=time,phase=phase))
    shots.append(dict(id=f"{scene['id']}-end",time=max(scene['start'],scene['end']-.15),phase='transition'))
for shot in shots:
    path=FRAMES/(shot['id']+'.png')
    run(['ffmpeg','-v','error','-y','-ss',f"{shot['time']:.4f}",'-i',str(VIDEO),'-frames:v','1',str(path)])
    shot['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
for start in range(0,len(shots),4):
    sheet=Image.new('RGB',(1708,1012),'#16202b')
    draw=ImageDraw.Draw(sheet)
    for j,shot in enumerate(shots[start:start+4]):
        x=j%2*854;y=j//2*506
        with Image.open(FRAMES/(shot['id']+'.png')) as pic:sheet.paste(pic,(x,y+26))
        draw.text((x+8,y+6),f"{shot['id']}  {shot['time']:.2f}s",fill='white')
    sheet.save(FRAMES/f'sheet-{start//4+1:02d}.png')
result=dict(duration=float(video['duration']),audio_duration=float(audio['duration']),resolution=[854,480],fps=15,scene_timing_max_error=max(errors),long_silence_count=silence.count('silence_start:'),mean_volume_db=float(re.search(r'mean_volume: ([-0-9.]+)',volume)[1]),max_volume_db=float(re.search(r'max_volume: ([-0-9.]+)',volume)[1]),full_decode='passed',video_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),frame_count=len(shots),review_sheet_count=(len(shots)+3)//4,manual_review='Extraction only; see the work report for actual frame review.',frames=shots)
(ROOT/'video_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:v for k,v in result.items() if k not in ('frames','video_sha256')})
