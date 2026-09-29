"""Generate reviewable script from the single source of narration truth."""
import json
from pathlib import Path
from narration_content import SCENES, script_hash
D=Path(__file__).resolve().parent
p=D/'assets/voicevox/manifest.json'
m=json.loads(p.read_text()) if p.exists() else {}
entries={s['id']:s for s in m.get('scenes',[])} if isinstance(m,dict) else {}
lines=['# PRML 3.5 エビデンス近似 — 全台本','', '原文: Bishop (2006), 印刷 pp.165–172（PDF pp.185–192）。式 (3.74)–(3.99)、図3.14–3.17。節内に表はありません。','', '次数を d、重みの個数を M と表記します。図3.14の次数 M は d に読み替えます。自作の訓練18点、ガウス基底9個＋定数。図の数値は原文の複製ではありません。','', '完全ベイズの同時積分と、固定した精度の下での厳密なガウス積分を区別します。再推定は停留条件であり大域解を保証しません。大標本極限では全方向がデータで決まる条件、さらに N が M より十分大きい条件で (3.98)–(3.99) を説明します。予測分散は前節の (3.59) を補助参照します。']
for sc in SCENES:
 e=entries.get(sc['id'],{})
 if e.get('script_sha256') != script_hash(sc): e={}
 dur=e.get('beat_durations',[b['seconds'] for b in sc['beats']])
 lines+=['',f"## {sc['id']} {sc['title']}",'',sc['reference'], '',f"尺: {sum(dur):.3f}秒（音声生成前は目安）"]
 for i,b in enumerate(sc['beats']):
  lines+=['',f"### {i+1}. {b['visual_note']} / {dur[i]:.3f}秒",'']
  for s in b['segments']:lines+=[f"- {s['id']} 字幕: {s['display']}",f"  音声: {s['speech']}"]
(D/'narration_script.md').write_text('\n'.join(lines)+'\n')
