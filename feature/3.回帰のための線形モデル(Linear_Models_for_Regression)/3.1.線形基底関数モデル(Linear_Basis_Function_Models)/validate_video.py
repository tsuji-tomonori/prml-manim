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
    added=[]
    for sc in timeline:
        for b in sc['beats']:
            if any('-recap-' in c['id'] or '-aid-' in c['id'] for c in b['cues']):
                duration=b['end']-b['start']
                recap='-recap-' in b['cues'][0]['id']
                assert (10 if recap else 5)<=duration<=(30 if recap else 20)
                added.append(dict(scene=sc['id'],start=b['start'],duration=duration,actions=b['actions']))
    # audio_query mora/phrase durations divided by speedScale=1.08.
    # Compare spoken "足しました" and "二かける…" with the matching visual phases.
    timing=[]
    for cue_id,phrase_offset,action_name in [
        ('scene01-recap-01',4.711,'R1.1 sum squares'),
        ('scene04-aid-02',2.178,'V08b multiply negative term'),
    ]:
        beat=next(b for sc in timeline for b in sc['beats'] if any(c['id']==cue_id for c in b['cues']))
        cue=next(c for c in beat['cues'] if c['id']==cue_id)
        action=next(a for a in beat['actions'] if a['name']==action_name)
        spoken=cue['start']+phrase_offset
        delta=action['start']-spoken
        assert abs(delta)<1/15
        timing.append(dict(cue=cue_id,action=action_name,spoken_phrase_start=spoken,
                           animation_start=action['start'],difference=delta))
    out=dict(probe=probe,audio_video_duration_gap=gap,silence_over_3s=silence,
             mean_volume_db=mean,peak_volume_db=peak,caption_cues=cues,max_scene_clock_gap=max_gap,
             scene_durations=[dict(id=s['id'],start=s['start'],duration=s['end']-s['start']) for s in timeline],
             extracted_frames=len(review['frames']),symbolic_captions=len(review['symbolic']),
             sync=review['sync'],frame_times=review['frames'],
             visual_aids=added,phrase_timing_checks=timing,
             mp4_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
             full_audio_listening=False)
    (ROOT/'validation_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='frame_times'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
