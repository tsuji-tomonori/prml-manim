import json
from pathlib import Path
from narration_content import SCENES, script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 6.1 双対表現 台本','',
'原文: Bishop (2006), §6.1 印刷 pp.293–294 / PDF pp.313–314。節内に図表なし。',
'公式正誤表: (6.8) は (6.3) を (6.4) へ代入して導く。',
'字幕 display は記号、読み上げ speech は発音。各 beat は文 WAV の実測時間に同期する。','']
start=0.
for s in SCENES:
 e=entries.get(s['id'],{}); ds=e.get('beat_durations',[])
 current = e.get('script_sha256') == script_hash(s)
 duration=sum(ds) if current else 0
 lines += [f"## {s['id']} {s['title']}",'',f"原文: {s['reference']}",(f"開始 {start:.3f} 秒 / 尺 {duration:.3f} 秒（音声生成後の実測）" if current and start is not None else "時刻: 音声の再生成後に確定"),'']
 for i,b in enumerate(s['beats']):
  lines += [f"### beat {i+1}: {b['visual_note']}",'']
  for seg in b['segments']:lines += [f"- {seg['id']} 字幕: {seg['display']}",f"  音声: {seg['speech']}"]
  lines += ['']
 start = start + duration if current and start is not None else None
lines += ['## 内容の対応と条件','','- 自作3点と二次特徴、λ=0.5。近い2点の平均0.5は素朴な予想であり、学習結果ではない。','- カーネルは指定した特徴の内積。対称なら何でも有効とは主張しない。','- 計算は逆行列を明示的に作らず連立方程式として解く。','']
(root/'narration_script.md').write_text('\n'.join(lines))
