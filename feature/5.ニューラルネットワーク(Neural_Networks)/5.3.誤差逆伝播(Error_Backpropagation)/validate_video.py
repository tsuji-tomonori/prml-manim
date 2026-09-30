"""Check final streams, audio, narration clocks and save reproducible results."""
import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path
import numpy as np
from review_video import VIDEO,ROOT
from narration_content import SCENES
from make_voicevox_narration import MANIFEST

def main():
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]))
    durations={s['codec_type']:float(s['duration']) for s in probe['streams']}
    assert set(durations)=={'audio','video'} and abs(durations['audio']-durations['video'])<1/15
    proc=subprocess.run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af','silencedetect=noise=-45dB:d=3,volumedetect','-f','null','-'],capture_output=True,text=True,check=True)
    silence=proc.stderr.count('silence_start:');assert silence==0
    volumes={k:float(re.search(k+r':\s*([\d.-]+) dB',proc.stderr)[1]) for k in ('mean_volume','max_volume')}
    timeline=json.loads((ROOT/'media/prml53_timeline.json').read_text());entries=json.loads(MANIFEST.read_text())['scenes']
    errors=[];sync=[]
    for s,e in zip(timeline,entries):
        errors.append(abs(s['end']-s['start']-e['duration']))
        cues=[c for b in s['beats'] for c in b['cues']]
        for c,a in zip(cues,e['subtitle_cues']):
            assert c['id']==a['id'] and c['display']==a['display']
            assert abs(c['start']-s['start']-a['start'])<1e-7
        if s['id'] in ('scene01','scene04','scene09'):
            b=s['beats'][1 if s['id']=='scene01' else 2 if s['id']=='scene04' else 3]
            with wave.open(str(ROOT/e['path'])) as wav:
                rate=wav.getframerate(); data=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16)
            for ci,c in enumerate(b['cues']):
                start=round((c['start']-s['start'])*rate);end=round((c['end']-s['start'])*rate)
                indices=np.flatnonzero(np.abs(data[start:end].astype(float))/32768>10**(-45/20))
                action=b['actions'][min(ci,len(b['actions'])-1)]
                sync.append(dict(id=c['id'],display=c['display'],action_start=action['start'],action_end=action['end'],
                                 pcm_onset=s['start']+(start+int(indices[0]))/rate))
    # Audit every added sentence against actual PCM onset and its own action.
    added_sync=[]
    for scene,e in zip(timeline,entries):
        with wave.open(str(ROOT/e['path'])) as wav:
            rate=wav.getframerate();data=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16)
        for beat in scene['beats']:
            for cue,action in zip(beat['cues'],beat['actions']):
                if '-recap-' not in cue['id'] and '-aid-' not in cue['id']:continue
                start=round((cue['start']-scene['start'])*rate);end=round((cue['end']-scene['start'])*rate)
                audible=np.flatnonzero(np.abs(data[start:end].astype(float))/32768>10**(-45/20))
                onset=scene['start']+(start+int(audible[0]))/rate
                assert abs(action['start']-cue['start'])<=1/30+1e-7
                assert action['start']<=onset<action['end']
                added_sync.append(dict(id=cue['id'],cue_start=cue['start'],action_start=action['start'],
                                       action_end=action['end'],pcm_onset=onset,
                                       onset_after_action=onset-action['start']))
    assert len(added_sync)==5
    assert max(errors)<1e-6
    display='\n'.join(v['display'] for s in SCENES for b in s['beats'] for v in b['segments'])
    (ROOT/'media/display.txt').write_text(display)
    result=dict(probe=probe,stream_difference=abs(durations['audio']-durations['video']),silences_over_3s=silence,
                **volumes,added_sync=added_sync,max_scene_clock_error=max(errors),sync=sync,
                sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),full_listen=False)
    (ROOT/'validation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
