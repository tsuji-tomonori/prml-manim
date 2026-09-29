"""Regenerate the reviewable script from the canonical content and measured WAVs."""
import json
from pathlib import Path
from narration_content import SCENES, script_hash
root = Path(__file__).resolve().parent
p = root / 'assets/voicevox/manifest.json'
entries = {s['id']: s for s in json.loads(p.read_text()).get('scenes', [])} if p.exists() else {}
lines = ['# PRML 2.3 ガウス分布：ナレーション台本', '',
         '原文は印刷 pp.78–113（PDF pp.98–133）。各シーンに対応する式と図を示す。節内に表はない。',
         '字幕は display、収録は speech。尺は生成済み音声の実測値。図は自作データ。', '']
for scene in SCENES:
    entry = entries.get(scene['id'], {})
    durations = entry.get('beat_durations', []) if entry.get('status') == 'generated' and entry.get('script_sha256') == script_hash(scene) else []
    lines += [f"## {scene['id']}: {scene['title']}", '', f"原文: {scene['reference']}", '']
    for i, beat in enumerate(scene['beats']):
        duration = f'{durations[i]:.3f} 秒' if durations else '音声生成後に確定'
        lines += [f"### {i+1}. {beat['visual_note']}（{duration}）", '']
        for s in beat['segments']:
            lines += [f"- `{s['id']}` 字幕: {s['display']}", f"  読み上げ: {s['speech']}"]
        lines += ['']
(root / 'narration_script.md').write_text('\n'.join(lines))
