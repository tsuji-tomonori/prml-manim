"""Measure the delivered MP4 and compare every caption with PCM timing.

Run review_video.py first. Image extraction is not a substitute for visual review.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from make_voicevox_narration import MANIFEST

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'media/videos/prml_3_1_linear_basis_function_models/480p15/PRML31LinearBasisFunctionModels.mp4'

def main():
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,duration',
        '-of','json',str(VIDEO)]))
    streams={s['codec_type']:s for s in probe['streams']}
    assert {'video','audio'}<=streams.keys()
    gap=abs(float(streams['video']['duration'])-float(streams['audio']['duration']))
    assert gap<1/15
    audio=subprocess.run(['ffmpeg','-hide_banner','-i',str(VIDEO),'-af',
                          'silencedetect=noise=-45dB:d=3,volumedetect','-f','null','-'],capture_output=True,text=True,check=True).stderr
    silence=len(re.findall(r'silence_start:',audio))
    assert silence==0
    mean=float(re.search(r'mean_volume: ([\d.-]+)',audio)[1])
    peak=float(re.search(r'max_volume: ([\d.-]+)',audio)[1])
    (ROOT/'media/review/audio-analysis.log').write_text(audio)
    timeline=json.loads((ROOT/'media/prml31_timeline.json').read_text())
    entries=json.loads(MANIFEST.read_text())['scenes']
    max_gap=0.;cues=0
    for sc,e in zip(timeline,entries):
        max_gap=max(max_gap,abs(sc['end']-sc['start']-e['duration']))
        actual=[c for b in sc['beats'] for c in b['cues']]
        assert len(actual)==len(e['subtitle_cues'])
        for a,b in zip(actual,e['subtitle_cues']):
            assert (a['id'],a['display'])==(b['id'],b['display'])
            assert abs(a['start']-sc['start']-b['start'])<1/15
            assert abs(a['end']-sc['start']-b['end'])<1/15
            cues+=1
    assert max_gap<1/15
    review=json.loads((ROOT/'media/review/frames.json').read_text())
    out=dict(probe=probe,audio_video_duration_gap=gap,silence_over_3s=silence,
             mean_volume_db=mean,peak_volume_db=peak,caption_cues=cues,max_scene_clock_gap=max_gap,
             scene_durations=[dict(id=s['id'],start=s['start'],duration=s['end']-s['start']) for s in timeline],
             extracted_frames=len(review['frames']),symbolic_captions=len(review['symbolic']),
             sync=review['sync'],frame_times=review['frames'],
             mp4_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
             full_audio_listening=False)
    (ROOT/'validation_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='frame_times'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
