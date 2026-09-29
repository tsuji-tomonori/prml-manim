"""Rebuild the reviewable script from canonical content and measured timings."""
import json
from pathlib import Path
from narration_content import SCENES, estimated_duration, script_hash
ROOT=Path(__file__).resolve().parent
path=ROOT/'assets/voicevox/manifest.json'
raw=json.loads(path.read_text()) if path.exists() else {}
entries={s['id']:s for s in raw.get('scenes',[])} if isinstance(raw,dict) else {}
lines=['# PRML 3.6 固定基底関数の限界：台本', '', '原文: Bishop (2006), §3.6 印刷 pp.172–173（PDF pp.192–193）。この節には番号付きの式・図・表はない。式 (3.3)、(3.15)、(3.6) は前節の復習。それ以外の数式と図は本文の説明用に作った例。', '', '各 beat の文と動作が同じPCM時計を使う。全編 VOICEVOX:WhiteCUL（23）。', '']
for s in SCENES:
 entry=entries.get(s['id'],{})
 lines += [f"## {s['id']} {s['title']}", '', '原文参照: '+s['reference'], '']
 for i,b in enumerate(s['beats']):
  duration=(entry['beat_durations'][i] if entry.get('script_sha256')==script_hash(s)
            else estimated_duration(b))
  lines += [f"### Beat {i+1} / {duration:.3f} 秒 / {b['visual_note']}", '']
  for seg in b['segments']:
   lines += ['- 字幕: '+seg['display'], '- 音声: '+seg['speech']]
  lines += ['']
(ROOT/'narration_script.md').write_text('\n'.join(lines)+'\n')
