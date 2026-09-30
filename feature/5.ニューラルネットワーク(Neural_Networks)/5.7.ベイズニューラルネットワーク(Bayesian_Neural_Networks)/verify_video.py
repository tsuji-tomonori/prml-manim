"""Collect media checks and review frames; does not claim human review."""
from pathlib import Path
import json, subprocess, wave, hashlib
import numpy as np
from PIL import Image, ImageDraw
d=Path(__file__).resolve().parent
out=d/'media/review';out.mkdir(parents=True,exist_ok=True)
video=d/'media/videos/prml_5_7_bayesian_neural_networks/480p15/PRML57BayesianNeuralNetworks.mp4'
timeline=json.loads((d/'media/prml57_timeline.json').read_text())
manifest=json.loads((d/'assets/voicevox/manifest.json').read_text())
from narration_content import SCENES
from make_voicevox_narration import valid_entry
assert len(timeline)==len(manifest['scenes'])==len(SCENES)
scene_timing_errors=[]
for scene, entry, story in zip(timeline,manifest['scenes'],SCENES):
 assert valid_entry(story,entry)
 assert [(c['id'],c['display']) for b in scene['beats'] for c in b['cues']]==[(c['id'],c['display']) for c in entry['subtitle_cues']]
 scene_timing_errors.append(abs(scene['end']-scene['start']-entry['duration']))
assert max(scene_timing_errors)<1/15
shots=[]
for s in timeline:
 for j,b in enumerate(s['beats']):
  cue=b['cues'][-1]
  shots.append(dict(id=f"{s['id']}-beat{j+1}",time=(cue['start']+cue['end'])/2,display=cue['display']))
  for c in b['cues']:
   if '$' in c['display']:
    shots.append(dict(id=c['id'],time=(c['start']+c['end'])/2,display=c['display']))
# Added cards: each sentence, early/middle/late action, and both transitions.
sync=[]
for scene,entry in zip(timeline,manifest['scenes']):
 for beat in scene['beats']:
  added=any('-recap-' in c['id'] or '-aid-' in c['id'] for c in beat['cues'])
  selected=[c for c in beat['cues'] if added or c['id'] in
            ['scene01-03-02','scene05-04-01','scene09-05-01']]
  if added:
   low,high=(5,20) if '-aid-' in beat['cues'][0]['id'] else (10,30)
   assert low<=beat['end']-beat['start']<=high
   for suffix,t in [('before',max(0,beat['start']-.2)),('after',beat['end']+.2)]:
    shots.append(dict(id=beat['cues'][0]['id']+'-'+suffix,time=t,display='transition'))
  for cue in selected:
   audio_cue=next(c for c in entry['subtitle_cues'] if c['id']==cue['id'])
   with wave.open(str(d/entry['path'])) as wav:
    rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16)/32768
   start=int(audio_cue['start']*rate);end=int(audio_cue['end']*rate)
   idx=np.flatnonzero(abs(pcm[start:end])>10**(-45/20))
   onset=scene['start']+(start+int(idx[0]))/rate
   if added:
    action=next(a for a in beat['actions'] if abs(a['start']-cue['start'])<1/15)
    assert abs(action['end']-cue['end'])<=1/15
   else:
    action=dict(start=beat['action_start'],end=beat['action_end'])
   sync.append(dict(id=cue['id'],sentence_start=cue['start'],pcm_onset=onset,
                    action_start=action['start'],action_end=action['end']))
   for suffix,fraction in [('early',.08),('middle',.5),('late',.94)]:
    shots.append(dict(id=cue['id']+'-'+suffix,time=action['start']+fraction*(action['end']-action['start']),display=cue['display']))
assert sum('-recap-' in c['id'] or '-aid-' in c['id'] for c in sync)==11
for s in shots:
 p=out/(s['id']+'.png')
 subprocess.run(['ffmpeg','-v','error','-y','-ss',str(s['time']),'-i',str(video),'-frames:v','1',str(p)],check=True)
 s['image_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
for start in range(0,len(shots),4):
 group=shots[start:start+4]
 sheet=Image.new('RGB',(1708,1012),'#18202d')
 draw=ImageDraw.Draw(sheet)
 for j,s in enumerate(group):
  x=(j%2)*854;y=(j//2)*506
  sheet.paste(Image.open(out/(s['id']+'.png')),(x,y+26))
  draw.text((x+10,y+5),f"{s['id']} / {s['time']:.3f}s",fill='white')
 sheet.save(out/f'sheet-{start//4+1:02}.png')
(out/'frames.json').write_text(json.dumps(shots,ensure_ascii=False,indent=2)+'\n')
(out/'sync.json').write_text(json.dumps(sync,ensure_ascii=False,indent=2)+'\n')
print('frames',len(shots),'sheets',(len(shots)+3)//4)
print(json.dumps(sync,indent=2))

probe = json.loads(subprocess.check_output([
    'ffprobe','-v','error','-show_entries',
    'format=duration,size:stream=index,codec_type,codec_name,duration,width,height,r_frame_rate',
    '-of','json',str(video)],text=True))
def audio_filter(name):
    result=subprocess.run(['ffmpeg','-hide_banner','-i',str(video),'-map','0:a:0',
                           '-af',name,'-f','null','-'],capture_output=True,text=True,check=True)
    return result.stderr
silence=audio_filter('silencedetect=noise=-45dB:d=3')
volume=audio_filter('volumedetect')
import re
audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
visual=next(s for s in probe['streams'] if s['codec_type']=='video')
difference=abs(float(audio['duration'])-float(visual['duration']))
silence_count=silence.count('silence_start:')
assert difference<.1 and silence_count==0
assert abs(float(visual['duration'])-timeline[-1]['end'])<.1
result=dict(scene_timing_max_error=max(scene_timing_errors),ffprobe=probe,stream_duration_difference=difference,
            long_silence_count=silence_count,
            mean_volume_db=float(re.search(r'mean_volume: ([-0-9.]+)',volume)[1]),
            max_volume_db=float(re.search(r'max_volume: ([-0-9.]+)',volume)[1]),
            video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
            frames=shots,sync=sync,
            manual_review='Frame extraction only; see the work report for actual review.',
            full_audio_listening=False)
# Locate each new sentence in the actual AAC stream, independently of the timeline.
from scipy.signal import correlate
rate=24000
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-map','0:a:0',
                             '-f','s16le','-ac','1','-ar',str(rate),'-'])
decoded=np.frombuffer(raw,dtype=np.int16).astype(float)
correlations=[]
for scene,entry in zip(timeline,manifest['scenes']):
 with wave.open(str(d/entry['path'])) as wav:
  assert wav.getframerate()==rate and wav.getnchannels()==1
  source=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16).astype(float)
 for cue in entry['subtitle_cues']:
  if '-recap-' not in cue['id'] and '-aid-' not in cue['id']:continue
  a,b=[round(cue[k]*rate) for k in ('start','end')]
  template=source[a:b];expected=scene['start']+cue['start']
  start=max(0,round((expected-.12)*rate));end=round((expected+.12)*rate)+len(template)
  window=decoded[start:end]
  scores=correlate(window,template,mode='valid',method='fft')
  k=int(np.argmax(scores));match=window[k:k+len(template)]
  coefficient=float(np.dot(match,template)/(np.linalg.norm(match)*np.linalg.norm(template)))
  error=(start+k)/rate-expected
  assert abs(error)<.1 and coefficient>.98
  correlations.append(dict(id=cue['id'],offset_seconds=error,correlation=coefficient))
result['added_sentence_audio_correlation']=correlations
print('AAC sync',len(correlations),'max offset',max(abs(c['offset_seconds']) for c in correlations),
      'min correlation',min(c['correlation'] for c in correlations))
(d/'video_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('media checks passed',difference,silence_count)
