import json
from pathlib import Path
from narration_content import SCENES, script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 7.1 最大マージン分類器 台本','',
'原文: Bishop (2006), 章導入 p.325、§7.1 印刷 pp.326–345 / PDF pp.346–365。図7.1–7.8を画像で確認。',
'公式正誤表: p.329 の双対目的の上界、p.332 の Lagrangian の引数、p.333 の最大化への訂正を確認。',
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
lines += ['## 内容の対応と条件','','- 自作6点は負のクラス (-1,0),(-2,-1),(-2,1)、正のクラス (1,0),(2,-1),(2,1)。','- 冒頭と結末は同じ点を使う。境界 x1=0.4 と x1=0 の余白はそれぞれ0.6と1。','- シーン5は phi(x)=(x1-0.22 sin(2x2),x2) を用いた説明用カーネル。余白は特徴空間で測る。','- シーン6・7では青クラス (1.4,0) を追加。C=0.2 と0.5の線形SVMを比較する。','- 近似条件と数値の全文は README.md、numerical_validation.json を参照。','']
(root/'narration_script.md').write_text('\n'.join(lines))
