"""Export the complete, reviewable narration source."""
import json
from pathlib import Path
from narration_content import SCENES
ROOT = Path(__file__).resolve().parent
manifest = ROOT / 'assets/voicevox/manifest.json'
entries = {s['id']:s for s in json.loads(manifest.read_text()).get('scenes', [])} if manifest.exists() else {}
lines = ['# PRML 4.4 ラプラス近似：全文台本', '', '原文: Bishop (2006) §4.4–4.4.1、印刷 pp.213–217（PDF pp.233–237）。図4.14、式(4.125)–(4.139)。本節に表はありません。', '', 'display は字幕、speech は合成音声の正本です。図は自作の数値例を使います。', '']
for s in SCENES:
    e=entries.get(s['id'], {})
    lines += [f"## {s['id']}: {s['title']}", '', f"原文: {s['reference']}。尺: {e.get('duration', '音声生成後に確定')} 秒。", '']
    for i,b in enumerate(s['beats'],1):
        lines += [f"### {i}. {b['visual_note']}", '']
        for q in b['segments']:
            lines += [f"- display: {q['display']}", f"- speech: {q['speech']}"]
        lines += ['']
(ROOT/'narration_script.md').write_text('\n'.join(lines)+'\n')
