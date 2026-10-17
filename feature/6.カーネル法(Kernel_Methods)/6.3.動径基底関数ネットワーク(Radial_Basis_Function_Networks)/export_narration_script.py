import json
from pathlib import Path
from narration_content import SCENES, script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 6.3 動径基底関数ネットワーク 台本','',
'原文: Bishop (2006), §6.3 印刷 pp.299–303 / PDF pp.319–323。図6.2・6.3あり。',
'公式正誤表: p.300 の (6.40) 左辺は y(x)。',
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
lines += ['## 内容の対応と条件','','- 自作3点 x=(-1,0,1)、t=(1,0,1)、x*=2.5、σ=0.7。','- 生の基底を補間する w と、正規化した基底に掛ける t は別のモデル。','- 確率からの導出では各成分密度の目標方向の平均を0とする。','']
(root/'narration_script.md').write_text('\n'.join(lines))
