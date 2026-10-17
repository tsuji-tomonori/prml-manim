import json
from pathlib import Path
from narration_content import SCENES, script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 6.4 ガウス過程 台本','',
'原文: Bishop (2006), §6.4 印刷 pp.303–320 / PDF pp.323–340。図6.4–6.12を画像で確認。',
'公式正誤表: p.310 第2段落末文の削除を確認。',
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
lines += ['## 内容の対応と条件','','- 自作3点 x=(-1,0,1)、t=(1,0,1)、x*=2.5。','- ガウスカーネルの長さ尺度は0.7、信号分散は1、独立な観測ノイズの標準偏差は0.2。固定値であり、この3点から学習していない。','- 帯は観測値の予測平均±2標準偏差。潜在関数の分散とはノイズ分散0.04だけ異なる。','']
(root/'narration_script.md').write_text('\n'.join(lines))
