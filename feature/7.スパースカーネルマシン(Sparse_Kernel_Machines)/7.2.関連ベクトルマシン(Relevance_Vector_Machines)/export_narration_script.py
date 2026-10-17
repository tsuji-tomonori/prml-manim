import json
from pathlib import Path
from narration_content import SCENES, script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 7.2 関連ベクトルマシン 台本','',
'原文: Bishop (2006), §7.2 印刷 pp.345–356 / PDF pp.365–376。図7.9–7.12を画像で確認。',
'公式正誤表: (7.79)の\\beta引数、設計行列、(7.94)の絶対値、図7.10のキャプション、p.352の添字、(7.118)の引数を確認。',
'字幕 display は記号、読み上げ speech は発音。各 beat は文 WAV の実測時間に同期する。','']
start=0.
for s in SCENES:
 e=entries.get(s['id'],{}); ds=e.get('beat_durations',[])
 current=e.get('script_sha256')==script_hash(s)
 duration=sum(ds) if current else 0
 lines += [f"## {s['id']} {s['title']}",'',f"原文: {s['reference']}",(f"開始 {start:.3f} 秒 / 尺 {duration:.3f} 秒（音声生成後の実測）" if current and start is not None else "時刻: 音声の再生成後に確定"),'']
 for i,b in enumerate(s['beats']):
  lines += [f"### beat {i+1}: {b['visual_note']}",'']
  for seg in b['segments']:lines += [f"- {seg['id']} 字幕: {seg['display']}",f"  音声: {seg['speech']}"]
  lines += ['']
 start=start+duration if current and start is not None else None
lines += ['## 内容の対応と条件','','- 自作11点、固定ノイズ精度40、幅0.3のガウス基底を使う。','- 逐次1基底最適化で残る中心は x=0.3,0.7。外の場所で予測分散が小さくなる限界も示す。','- 分類場面は回帰曲線の得点をシグモイドへ写す説明用変換で、分類RVMの学習結果ではない。','- 近似条件と数値の全文は README.md、numerical_validation.json を参照。','']
(root/'narration_script.md').write_text('\n'.join(lines))
