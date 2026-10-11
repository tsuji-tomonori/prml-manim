"""Check the retained visual aids, recap beats, and their numerical examples."""
import json
from pathlib import Path

import numpy as np

from discriminative_model import HOLDOUT_P, HOLDOUT_X, HISTORY, X, sigmoid
from make_voicevox_narration import MANIFEST, valid_entry
from narration_content import SCENES

ROOT = Path(__file__).resolve().parent


def main():
    h = 1e-5
    error = lambda a: float(np.logaddexp(0, -a))
    derivative = (error(h) - error(-h)) / (2 * h)
    assert abs(derivative + .5) < 1e-10
    dy = float(sigmoid(.04) - sigmoid(0))
    de = error(.04) - error(0)
    assert abs(dy - .01) < 2e-6 and abs(de + .02) < .00021

    e = lambda w: 1 + 2*w + 2*w*w + .5*w**4
    g = (e(h) - e(-h)) / (2*h)
    curvature = (e(h) - 2*e(0) + e(-h)) / h**2
    assert abs(g - 2) < 1e-8 and abs(curvature - 4) < 2e-6
    step = -g / curvature
    assert abs(step + .5) < 1e-6 and e(step) < e(0)
    probs = np.array([.1, .5, .2])
    assert np.allclose(probs * (1-probs), [.09, .25, .16])

    entries = json.loads(MANIFEST.read_text())['scenes']
    assert len(entries) == len(SCENES)
    assert all(valid_entry(scene, entry) for scene, entry in zip(SCENES, entries))
    cards = []
    for scene, entry in zip(SCENES, entries):
        for beat, duration in zip(scene['beats'], entry['beat_durations']):
            key = beat['segments'][0]['id']
            if '-recap-' not in key and '-aid-' not in key:
                continue
            bounds = (10, 30) if '-recap-' in key else (5, 20)
            assert bounds[0] <= duration <= bounds[1], (key, duration)
            cards.append(dict(id=key, seconds=duration, note=beat['visual_note']))
    assert len(cards) == 5
    assert [c['id'] for c in cards] == [
        'scene03-recap-cost-01', 'scene03-aid-chain-01',
        'scene05-recap-irls-01', 'scene05-aid-newton-01',
        'scene09-recap-link-01']
    assert HOLDOUT_X not in X
    assert abs(float(sigmoid(HISTORY[-1] @ [1, HOLDOUT_X])) - HOLDOUT_P) < 1e-14

    result = dict(chain_derivative=derivative, chain_delta_y=dy, chain_delta_E=de,
                  newton_gradient=g, newton_curvature=curvature, newton_step=step,
                  cards=cards, held_out_probability=HOLDOUT_P,
                  wav_seconds=sum(e['duration'] for e in entries))
    (ROOT/'visual_aid_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
