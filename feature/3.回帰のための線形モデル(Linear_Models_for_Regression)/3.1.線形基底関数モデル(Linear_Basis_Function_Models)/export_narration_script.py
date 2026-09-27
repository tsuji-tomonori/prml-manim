"""Export the authoritative display and speech cues, optionally with measured durations."""
import json
from pathlib import Path
from narration_content import SCENES
p=Path(__file__).parent
f=p/'assets/voicevox/manifest.json'
m=json.loads(f.read_text()) if f.exists() else {}
entries={e['id']:e for e in m.get('scenes',[])} if isinstance(m,dict) else {}
lines=['# PRML 3.1 線形基底関数モデル 台本','', '原文：Bishop (2006)、印刷 pp.138–147、§3.1.1–3.1.5 を含む。該当節に表はない。', '問い → 連続変形 → 式による確認。各文の実測PCM尺で字幕と動作を同期する。','']
for sc in SCENES:
 lines += [f"## {sc['id']} {sc['title']}",'',f"原文：{sc['reference']}",f"音声尺：{entries.get(sc['id'],{}).get('duration','未生成')} 秒",'']
 for b in sc['beats']:
  lines += [f"### {b['visual_note']}",'']
  for s in b['segments']: lines += [f"- {s['id']} 字幕：{s['display']}",f"  読み上げ：{s['speech']}"]
  lines += ['']
(p/'narration_script.md').write_text('\n'.join(lines)+'\n')
