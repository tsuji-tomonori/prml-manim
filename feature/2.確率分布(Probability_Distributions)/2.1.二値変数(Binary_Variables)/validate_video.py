"""Measure the final movie and reconcile PCM narration with the visual timeline."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from make_voicevox_narration import MANIFEST,valid_entry
from narration_content import SCENES

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_2_1_binary_variables/480p15/PRML21BinaryVariables.mp4'

def run(*args):
    p=subprocess.run(args,capture_output=True,text=True,check=True)
    return p.stdout+p.stderr

def main():
    probe=json.loads(run('ffprobe','-v','error','-show_entries','format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,duration','-of','json',str(VIDEO)))
    silence=run('ffmpeg','-hide_banner','-i',str(VIDEO),'-af','silencedetect=noise=-45dB:d=3','-f','null','-')
    volume=run('ffmpeg','-hide_banner','-i',str(VIDEO),'-map','0:a:0','-af','volumedetect','-f','null','-')
    entries=json.loads(MANIFEST.read_text())['scenes']
    timeline=json.loads((ROOT/'media/prml21_timeline.json').read_text())
    drift=[]
    for story,entry,scene in zip(SCENES,entries,timeline,strict=True):
        assert valid_entry(story,entry)
        drift.append(abs(entry['duration']-(scene['end']-scene['start'])))
        movie_cues=[c for b in scene['beats'] for c in b['cues']]
        for c,m in zip(movie_cues,entry['subtitle_cues'],strict=True):
            assert c['id']==m['id'] and c['display']==m['display']
            assert abs(c['start']-scene['start']-m['start'])<1/15
    streams={s['codec_type']:s for s in probe['streams']}
    av_difference=abs(float(streams['video']['duration'])-float(streams['audio']['duration']))
    assert av_difference<1/15
    count=len(re.findall(r'silence_start:',silence))
    assert count==0
    frames=json.loads((ROOT/'media/review/frames.json').read_text())
    results=dict(probe=probe,av_difference_seconds=av_difference,long_silence_count=count,
        mean_volume_db=float(re.search(r'mean_volume: ([-\d.]+)',volume)[1]),
        max_volume_db=float(re.search(r'max_volume: ([-\d.]+)',volume)[1]),
        max_scene_audio_drift_seconds=max(drift),sentence_count=sum(len(e['subtitle_cues']) for e in entries),
        scene_durations=[dict(id=e['id'],title=e['title'],seconds=e['duration']) for e in entries],
        extracted_frames=len(frames['frames']),symbolic_frames=len(frames['symbolic']),review=frames,
        full_listen=False,all_frames_reviewed=False,sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest())
    (ROOT/'validation_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'media/review/silencedetect.log').write_text(silence)
    (ROOT/'media/review/volumedetect.log').write_text(volume)
    print(json.dumps({k:v for k,v in results.items() if k!='review'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
