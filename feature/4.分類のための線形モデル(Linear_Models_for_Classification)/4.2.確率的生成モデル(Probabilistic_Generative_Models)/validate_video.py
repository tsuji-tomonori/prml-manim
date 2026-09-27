"""Measure muxed audio/video, timing, and PCM onset; preserve actual evidence only."""
import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path
import numpy as np
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,valid_entry

ROOT=Path(__file__).resolve().parent
video=ROOT/'media/videos/prml_4_2_probabilistic_generative_models/480p15/PRML42ProbabilisticGenerativeModels.mp4'
review=ROOT.parents[2]/'.working/review';review.mkdir(parents=True,exist_ok=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(video)]))
v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
assert abs(float(v['duration'])-float(a['duration']))<1/15
logs={}
for name,filter_ in [('silence','silencedetect=noise=-45dB:d=3'),('volume','volumedetect')]:
 p=subprocess.run(['ffmpeg','-hide_banner','-i',str(video),'-map','0:a:0','-af',filter_,'-f','null','-'],capture_output=True,text=True,check=True)
 logs[name]=p.stderr;(review/(name+'.log')).write_text(p.stderr)
assert 'silence_start' not in logs['silence']
volume={k:float(re.search(k+r':\s*(-?[\d.]+)',logs['volume']).group(1)) for k in ['mean_volume','max_volume']}
timeline=json.loads((ROOT/'media/prml42_timeline.json').read_text())
entries=json.loads(MANIFEST.read_text())['scenes'];errors=[];sentence_errors=[]
for s,e,t in zip(SCENES,entries,timeline):
 assert valid_entry(s,e)
 errors.append(abs(t['end']-t['start']-e['duration']))
 actual=[c for b in t['beats'] for c in b['cues']]
 assert [(c['id'],c['display']) for c in actual]==[(c['id'],c['display']) for c in e['subtitle_cues']]
 for c,d in zip(actual,e['subtitle_cues']):
  sentence_errors.extend([abs(c['start']-t['start']-d['start']),abs(c['end']-t['start']-d['end'])])
assert max(errors)<1/15 and max(sentence_errors)<1/15
sync=[]
for si,bi in [(3,1),(4,4),(6,1)]:
 e=entries[si];t=timeline[si];b=t['beats'][bi]
 with wave.open(str(ROOT/'assets/voicevox'/f"{e['id']}.wav")) as wav:
  rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')/32768
 cues=[c for c in e['subtitle_cues'] if c['beat_index']==bi];onsets=[]
 for cue in cues:
  begin=round(cue['start']*rate);end=round(cue['end']*rate)
  indices=np.flatnonzero(np.abs(pcm[begin:end])>10**(-45/20))
  onsets.append(t['start']+(begin+int(indices[0]))/rate)
 sync.append(dict(scene=e['id'],beat=bi+1,action_start=b['action_start'],action_end=b['action_end'],speech_onsets=onsets,display=[c['display'] for c in cues]))
result=dict(probe=probe,stream_difference=abs(float(v['duration'])-float(a['duration'])),silences_over_3s=0,volume_db=volume,max_scene_clock_error=max(errors),max_sentence_clock_error=max(sentence_errors),sync=sync,sha256=hashlib.sha256(video.read_bytes()).hexdigest(),full_listening=False)
(ROOT/'validation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
