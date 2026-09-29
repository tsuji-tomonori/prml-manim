"""ffprobe/ffmpeg checks on the complete, final video; no listening claims."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from narration_content import SCENES

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_4_1_discriminant_functions/480p15/PRML41DiscriminantFunctions.mp4'

def main():
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,duration','-of','json',str(VIDEO)]))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
    delta=abs(float(video['duration'])-float(audio['duration']))
    assert delta<1/15
    logs={}
    for name,af in [('silence','silencedetect=noise=-45dB:d=3'),('volume','volumedetect')]:
        p=subprocess.run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af',af,'-f','null','-'],capture_output=True,text=True,check=True)
        logs[name]=p.stderr
        (ROOT/'media'/f'{name}.log').write_text(p.stderr)
    silence=len(re.findall('silence_start:',logs['silence']))
    assert silence==0
    volume={key:float(re.search(key+r':\s*([-\d.]+) dB',logs['volume']).group(1)) for key in ['mean_volume','max_volume']}
    timeline=json.loads((ROOT/'media/prml41_timeline.json').read_text())
    manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
    max_drift=0
    max_cue_drift=0
    for sc,t,e in zip(SCENES,timeline,manifest):
        assert sc['id']==t['id']==e['id']
        max_drift=max(max_drift,abs(t['end']-t['start']-e['duration']))
        assert len(sc['beats'])==len(t['beats'])
        for i,b in enumerate(t['beats']):
            assert b['start']-1e-6<=b['action_start']<=b['action_end']<=b['end']+1e-6
            assert [s['display'] for s in sc['beats'][i]['segments']]==[c['display'] for c in b['cues']]
            for c in b['cues']:
                expected=next(q for q in e['subtitle_cues'] if q['id']==c['id'])
                for edge in ['start','end']:
                    max_cue_drift=max(max_cue_drift,abs(c[edge]-t['start']-expected[edge]))
    assert max_drift<1/15
    assert max_cue_drift<1/15
    result={'probe':probe,'stream_duration_difference':delta,'long_silence_count':silence,**volume,
            'max_scene_wav_drift':max_drift,'max_cue_wav_drift':max_cue_drift,
            'sha256':hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
            'full_listening':False}
    (ROOT/'validation_media.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
