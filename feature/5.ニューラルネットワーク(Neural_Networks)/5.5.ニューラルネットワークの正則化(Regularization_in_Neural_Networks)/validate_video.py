"""Verify final media, clocks, hashes, captions and three animation/audio pairings."""
import hashlib,json,re,subprocess,wave
from pathlib import Path
import numpy as np
from review_video import ROOT,VIDEO
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,valid_entry

def main():
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,duration,width,height,r_frame_rate','-of','json',str(VIDEO)]))
    durations={s['codec_type']:float(s['duration']) for s in probe['streams']}
    assert set(durations)=={'audio','video'}
    difference=abs(durations['audio']-durations['video']);assert difference<1/15
    audio=subprocess.run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af','silencedetect=noise=-45dB:d=3,volumedetect','-f','null','-'],capture_output=True,text=True,check=True).stderr
    silences=audio.count('silence_start:');assert silences==0
    volume={key:float(re.search(key+r':\s*([\d.-]+) dB',audio)[1]) for key in ['mean_volume','max_volume']}
    timeline=json.loads((ROOT/'scene_timeline.json').read_text());entries=json.loads(MANIFEST.read_text())['scenes']
    errors=[];sync=[];count=0
    for scene,entry,story in zip(timeline,entries,SCENES):
        assert scene['id']==entry['id']==story['id'] and valid_entry(story,entry)
        errors.append(abs(scene['end']-scene['start']-entry['duration']))
        cues=[c for b in scene['beats'] for c in b['cues']]
        assert len(cues)==len(entry['subtitle_cues'])
        for cue,expected in zip(cues,entry['subtitle_cues']):
            assert cue['id']==expected['id'] and cue['display']==expected['display']
            errors.append(abs(cue['start']-scene['start']-expected['start']));count+=1
        if scene['id'] in ('scene02','scene06','scene08'):
            b=scene['beats'][1 if scene['id']=='scene08' else 3]
            with wave.open(str(ROOT/entry['path'])) as wav:
                rate=wav.getframerate();samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16)
            for cue,action in zip(b['cues'],b['actions']):
                start=round((cue['start']-scene['start'])*rate);end=round((cue['end']-scene['start'])*rate)
                active=np.flatnonzero(abs(samples[start:end].astype(float))/32768>10**(-45/20))
                onset=scene['start']+(start+int(active[0]))/rate
                assert action['start']<=onset<action['end']
                sync.append(dict(id=cue['id'],display=cue['display'],action=action['name'],action_start=action['start'],action_end=action['end'],pcm_onset=onset))
    assert max(errors)<1e-6
    display='\n'.join(s['display'] for scene in SCENES for b in scene['beats'] for s in b['segments'])
    (ROOT/'media/display.txt').write_text(display)
    assert not re.search('エックス|ラムダ|ミュー|シグマ|アルファ|ガンマ|タウ|クシー',display)
    readings=json.loads((ROOT/'reading_check.json').read_text())['sentences']
    expected=[s for scene in SCENES for b in scene['beats'] for s in b['segments']]
    assert [(r['id'],r['speech'],r['display']) for r in readings]==[(s['id'],s['speech'],s['display']) for s in expected]
    assert {p.stem for p in (ROOT/'assets/voicevox').glob('scene*.wav')}=={s['id'] for s in SCENES}
    result=dict(probe=probe,stream_difference=difference,silences_over_3s=silences,**volume,
                caption_count=count,math_caption_count=sum('$' in s['display'] for s in expected),max_clock_error=max(errors),sync=sync,
                sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),full_listen=False)
    (ROOT/'validation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
