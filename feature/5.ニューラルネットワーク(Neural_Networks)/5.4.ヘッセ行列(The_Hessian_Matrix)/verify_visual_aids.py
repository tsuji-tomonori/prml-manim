"""Check the recap geometry, product-rule limit and PCM action alignment."""
import json
from pathlib import Path
import wave
import numpy as np
from narration_content import SCENES
from hessian_model import ellipse_points

ROOT = Path(__file__).parent
ADDITIONS = {'scene05': ('recap', 1), 'scene07': ('recap', 0), 'scene08': ('aid', 3)}

def main():
    # The red curves copied from 4.4 use standard deviations, not variances.
    theta = .65
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    errors = []
    for lam in (1., 2., 5.):
        A = Q @ np.diag([lam, 1.]) @ Q.T
        points = ellipse_points(A)
        errors.append(float(np.max(abs(np.einsum('ni,ij,nj->n', points, A, points)-1))))
        assert np.allclose(Q.T @ np.linalg.inv(A) @ Q, np.diag([1/lam, 1]))
    # Exact finite area = two strips + corner. After dividing by epsilon,
    # the corner is O(epsilon) and vanishes in the directional derivative.
    v, z, rv, rz = 2.8, 1.6, 1., .8
    expected = z*rv + v*rz
    limit_errors = []
    for eps in (.6, .1, .025, 1e-5):
        area = (v+eps*rv)*(z+eps*rz)-v*z
        strips = eps*(z*rv+v*rz)
        assert np.isclose(area, strips+eps**2*rv*rz)
        limit_errors.append(abs(area/eps-expected))
    assert all(a>b for a,b in zip(limit_errors, limit_errors[1:]))
    entries = json.loads((ROOT/'assets/voicevox/manifest.json').read_text())['scenes']
    timeline = json.loads((ROOT/'media/prml54_timeline.json').read_text())
    old_ids = {f'scene{s:02}-{b:02}-{j:02}' for s in range(1,10) for b in range(1,7) for j in (1,2)}
    ids = [c['id'] for s in SCENES for b in s['beats'] for c in b['segments']]
    assert len(ids) == len(set(ids)) == 116 and old_ids.issubset(ids)
    records, alignment = [], []
    for entry, scene, story in zip(entries,timeline,SCENES):
        assert abs(scene['end']-scene['start']-entry['duration']) < 1e-6
        if entry['id'] not in ADDITIONS:
            continue
        key, index = ADDITIONS[entry['id']]
        beat = scene['beats'][index]
        assert all(f'-{key}-' in c['id'] for c in beat['cues'])
        assert (10 if key=='recap' else 5) <= beat['end']-beat['start'] <= (30 if key=='recap' else 20)
        with wave.open(str(ROOT/entry['path'])) as f:
            rate = f.getframerate()
            pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        for cue, action in zip(beat['cues'], beat['actions']):
            assert cue['id']==action['sentence']
            assert abs(cue['start']-action['start']) <= 1/30 + 1e-7
            lo,hi = (round((cue[t]-scene['start'])*rate) for t in ('start','end'))
            voiced = np.flatnonzero(abs(pcm[lo:hi].astype(float))/32768 > 10**(-45/20))
            onset = scene['start']+(lo+int(voiced[0]))/rate
            end = scene['start']+(lo+int(voiced[-1]))/rate
            assert action['start'] <= onset < end <= action['end']
            alignment.append(dict(id=cue['id'],cue_start=cue['start'],pcm_onset=onset,
                                  pcm_end=end,action_start=action['start'],action_end=action['end']))
        records.append(dict(scene=scene['id'],kind=key,start=beat['start'],end=beat['end'],seconds=beat['end']-beat['start']))
    total = sum(e['duration'] for e in entries)
    baseline = 628.4666666666667
    assert 0 < total/baseline-1 < .15
    result = dict(max_recap_ellipse_error=max(errors),product_limit_errors=limit_errors,
                  baseline_seconds=baseline,total_seconds=total,increase_percent=(total/baseline-1)*100,
                  additions=records,alignment=alignment)
    (ROOT/'visual_aid_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
