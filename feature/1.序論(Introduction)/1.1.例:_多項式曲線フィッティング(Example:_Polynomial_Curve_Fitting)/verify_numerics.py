"""Run the reproducible numerical and audio-integrity checks without rendering."""
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

import make_voicevox_narration as voice
from narration_content import SCENES, estimated_duration, script_hash
from polynomial_model import (
    NOISE_STD, T, TT, T_ALL, WEIGHTS, X, XT, X_ALL, TRAIN_RMS, TEST_RMS,
    degree_weights, design_matrix, eval_poly, fit_polynomial, growing_weights,
    ridge_weights, rms_error, sample_weights,
)


class NumericalChecks(unittest.TestCase):
    def test_least_squares_and_interpolation(self):
        self.assertTrue(np.all(np.diff(TRAIN_RMS) <= 1e-8))
        np.testing.assert_allclose(eval_poly(WEIGHTS[9], X), T, atol=1e-8)
        self.assertGreater(TEST_RMS[9], 2 * TEST_RMS[3])
        for m, w in enumerate(WEIGHTS):
            np.testing.assert_allclose(degree_weights(m), w)
            r = eval_poly(w, X) - T
            self.assertAlmostEqual(rms_error(w, X, T), np.sqrt(2 * (.5 * (r @ r)) / len(X)))
        np.testing.assert_allclose(degree_weights(3.5), .5 * (WEIGHTS[3] + WEIGHTS[4]))

    def test_ridge_is_eq_1_4_with_bias(self):
        for loglam in [-32, -18, -8, 0]:
            lam = np.exp(loglam)
            w = ridge_weights(loglam)
            phi = design_matrix(X, 9)
            # Stationarity of Eq.(1.4), including the w0 penalty.
            np.testing.assert_allclose(phi.T @ (phi @ w - T) + lam * w, 0, atol=2e-8)
            self.assertTrue(np.isfinite(w).all())
        norms = [np.linalg.norm(ridge_weights(l)) for l in [-32, -18, -8, 0]]
        self.assertTrue(np.all(np.diff(norms) < 0))
        w = fit_polynomial(X, T, 0, lam=2)
        self.assertAlmostEqual(w[0], T.sum() / (len(T) + 2))

    def test_nested_data_and_contours(self):
        np.testing.assert_array_equal(X_ALL[:10], X)
        np.testing.assert_array_equal(T_ALL[:10], T)
        self.assertEqual(len(XT), 100)
        for n in [10, 15, 40, 100]:
            np.testing.assert_allclose(growing_weights(n), sample_weights(n))
        self.assertLess(rms_error(sample_weights(100), XT, TT), TEST_RMS[9])
        phi = design_matrix(X, 1)
        w = WEIGHTS[1][:2]
        delta = np.array([.2, -.4])
        diff = .5 * np.sum((phi @ (w + delta) - T)**2) - .5 * np.sum((phi @ w - T)**2)
        self.assertAlmostEqual(diff, .5 * delta @ (phi.T @ phi) @ delta)

    def test_displayed_curves_stay_in_the_plot(self):
        grid = np.linspace(0, 1, 2001)
        for w in WEIGHTS:
            self.assertGreaterEqual(eval_poly(w, grid).min(), -3)
            self.assertLessEqual(eval_poly(w, grid).max(), 2)
        for n in range(10, 101):
            y = eval_poly(sample_weights(n), grid)
            self.assertGreaterEqual(y.min(), -3.5)
            self.assertLessEqual(y.max(), 2)
        for l in np.linspace(-32, 0, 321):
            y = eval_poly(ridge_weights(float(l)), grid)
            self.assertGreaterEqual(y.min(), -3)
            self.assertLessEqual(y.max(), 2)


class AudioChecks(unittest.TestCase):
    def test_padding_hash_and_resume_manifest(self):
        # A local synthetic PCM fixture tests plumbing, not VOICEVOX synthesis.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'fixture.wav'
            with wave.open(str(fixture), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000)
                f.writeframes(b'\0\0' * 2400)
            wav_bytes = fixture.read_bytes()
            def fake_post(base, endpoint, params, body=None):
                return b'{}' if endpoint == 'audio_query' else wav_bytes
            with patch.object(voice, 'OUTPUT_DIR', root), patch.object(voice, 'SCENE_DIR', root), \
                 patch.object(voice, 'MANIFEST', root / 'manifest.json'), patch.object(voice, 'post_json', fake_post):
                scene = SCENES[0]
                entry = voice.generate_scene('fixture', scene)
                self.assertTrue(voice.valid_entry(scene, entry))
                self.assertAlmostEqual(sum(entry['beat_durations']), voice.wav_duration(root / 'scene01.wav'))
                voice.save_manifest([entry])
                prepared = voice.prepare_manifest()
                self.assertEqual(prepared[0]['status'], 'generated')
                self.assertTrue(all(e['status'] == 'pending' for e in prepared[1:]))
                changed = dict(scene, title='changed')
                self.assertFalse(voice.valid_entry(changed, entry))
                (root / 'scene01.wav').write_bytes(b'stale')
                self.assertFalse(voice.valid_entry(scene, entry))
                voice.prepare_manifest()
                self.assertFalse((root / 'scene01.wav').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
