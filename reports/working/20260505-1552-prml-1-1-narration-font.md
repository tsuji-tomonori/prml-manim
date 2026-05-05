# PRML 1.1 narration tempo and font update

## Summary

- WhiteCUL narration tempo was increased.
- Japanese text rendering was switched to `Noto Sans CJK JP`.
- The PRML 1.1 video was regenerated and checked for audio continuity and layout readability.

## Changed Files

- `feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/make_voicevox_narration.py`
  - Increased WhiteCUL `speed_scale` values.
  - Shortened inserted pauses from `0.45` seconds to `0.32` seconds.
  - Added `--from-scene` resume support.
  - Made per-scene WAV generation atomic through temporary output files.
- `feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/prml_1_1_polynomial_curve_fitting.py`
  - Added a default Japanese font wrapper for Manim `Text`.
  - Set Japanese text font to `Noto Sans CJK JP`.
- `feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/assets/voicevox/`
  - Regenerated WhiteCUL WAV narration assets and manifest.

## Verification

- `python3 -m py_compile prml_1_1_polynomial_curve_fitting.py make_voicevox_narration.py`
- `python3 make_voicevox_narration.py --from-scene scene07`
- `python3 make_voicevox_narration.py --from-scene scene11`
- `uv run manim --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting`
- `ffprobe` confirmed generated video duration: `403.799` seconds.
- `ffmpeg silencedetect=noise=-45dB:d=3` completed without reporting long silent gaps.
- `ffmpeg volumedetect` reported mean volume `-27.3 dB`, max volume `-8.2 dB`.
- Extracted frames at 00:00:32, 00:03:05, 00:04:15, and 00:05:45 and checked Japanese font/layout readability.

## Output

- `feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/media/videos/prml_1_1_polynomial_curve_fitting/480p15/PRML11PolynomialCurveFitting.mp4`
