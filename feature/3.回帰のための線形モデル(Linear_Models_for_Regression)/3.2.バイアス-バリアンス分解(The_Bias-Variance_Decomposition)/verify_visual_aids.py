"""Verify small-example arithmetic and export rendered action boundaries for review.

Run after the full render. PNGs and verification.json are generated under media/.
Inspect the exported images; this script does not judge their visual quality.
"""
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TIMELINE = ROOT / 'media/prml32_timeline.json'
VIDEO = ROOT / 'media/videos/prml_3_2_bias_variance_decomposition/480p15/PRML32BiasVarianceDecomposition.mp4'
OUT = ROOT / 'media/visual-aid-review'


def main():
    # A separate direct calculation checks the numbers displayed in the 3-by-2 card.
    predictions = np.array([[0., 1.], [2., 2.], [4., 3.]])
    means = predictions.mean(axis=0)
    squares = (predictions - means)**2
    variances = squares.mean(axis=0)
    np.testing.assert_allclose(means, [2, 2])
    np.testing.assert_allclose(variances, [8/3, 2/3])
    np.testing.assert_allclose(variances.mean(), 5/3)
    # Integrate the recalled N(0, .6^2) density to check its loss curve.
    ts = np.linspace(-8, 8, 16001)
    density = np.exp(-.5*(ts/.6)**2)/(.6*np.sqrt(2*np.pi))
    for pred in [0, .4, 1.6]:
        np.testing.assert_allclose(np.trapezoid((pred-ts)**2*density, ts), pred**2+.36, atol=1e-12)
    timeline = json.loads(TIMELINE.read_text())
    manifest = {s['id']:s for s in json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']}
    OUT.mkdir(parents=True, exist_ok=True)
    frames, records, max_drift = [], [], 0.
    for scene in timeline:
        frames.append((f"{scene['id']}-overview", scene['start']+.55*(scene['end']-scene['start'])))
        entry = manifest[scene['id']]
        with wave.open(str(ROOT/entry['path'])) as wav:
            assert wav.getsampwidth() == 2
            sample_rate = wav.getframerate()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        for index, beat in enumerate(scene['beats']):
            expected = scene['start']+sum(entry['beat_durations'][:index])
            max_drift = max(max_drift, abs(expected-beat['start']))
            assert abs(expected-beat['start']) < 1/15+1e-6
            if not any('-recap-' in c['id'] or '-aid-' in c['id'] for c in beat['cues']):
                continue
            tag = beat['cues'][0]['id'].rsplit('-',1)[0]
            frames.extend([(tag+'-before',beat['start']-.15),(tag+'-after',beat['end']+.15)])
            for i, action in enumerate(a for a in beat['actions'] if a['name'] != 'breath'):
                assert beat['start'] <= action['start'] < action['end'] <= beat['end']+1e-6
                frames.extend([(f'{tag}-action{i}-start',action['start']+.067),
                               (f'{tag}-action{i}-middle',(action['start']+action['end'])/2),
                               (f'{tag}-action{i}-end',action['end']-.067),
                               (f'{tag}-action{i}-after',action['end']+.067)])
            speech=[]
            for c in beat['cues']:
                lo = round((c['start']-scene['start'])*sample_rate)
                hi = round((c['end']-scene['start'])*sample_rate)
                active = np.flatnonzero(np.abs(pcm[lo:hi].astype(float))/32768 > 10**(-45/20))
                assert len(active)
                speech.append(dict(id=c['id'],display=c['display'],cue_start=c['start'],cue_end=c['end'],
                                   pcm_start=scene['start']+(lo+active[0])/sample_rate,
                                   pcm_end=scene['start']+(lo+active[-1])/sample_rate))
            records.append(dict(id=tag,start=beat['start'],end=beat['end'],speech=speech,actions=beat['actions']))
    for name, time in frames:
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(time),'-i',str(VIDEO),'-frames:v','1',str(OUT/f'{name}.png')],check=True)
    report=dict(example_means=means.tolist(),example_variances=variances.tolist(),example_total=float(variances.mean()),
                max_beat_audio_drift=max_drift,frame_count=len(frames),frames=[dict(name=n,time=t) for n,t in frames],cards=records)
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
