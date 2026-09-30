import json
from pathlib import Path
from narration_content import SCENES, script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 5.7 ベイズニューラルネットワーク 台本','','原文: Bishop (2006), 印刷 pp.277–284 / PDF pp.297–304。図5.22・5.23。表なし。',
'公式正誤表に従い (5.181) の総和、(5.183) の不要な定数項、(5.190) の a_MAP を修正。',
'字幕 display と読み上げ speech は独立。各 beat の動作は文 WAV の実測時間で同期する。','']
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
lines += ['## 内容の対応と条件','','- scene02 の密度は他の直交座標を固定した条件付き一次元断面。全重みの周辺事後とは区別する。','- scene03 の H は二乗誤差のヘッセ行列。A は事前項を含む負の対数事後のヘッセ行列。','- scene04–05 の帯は局所線形化したネットワークの平均±2標準偏差。scene05 の共分散縮小は思考実験。','- scene06 の面積は一変数の説明例であり、ネットワーク全体のエビデンスとは区別する。','- scene07 の四方向の棒は非負の固有値を用いた説明例。更新値は実ネットワークのヘッセ行列から計算。一般の非線形 Hessian が半正定値とは限らない。','- Fig.5.22 は最尤とエビデンス正則化の比較。動画では数値発散を避けた弱い正則化を比較の起点に使う。','- Fig.5.23 の境界不変性は、同じ MAP とガウス・活性線形化・κ近似を使用する下での性質。任意の厳密ベイズ予測についての主張ではない。','- 隠れユニットの入れ替えと符号反転による M!2^M は原文の tanh 二層ネットワークの対称性。縮退や異なる山の重複に注意する。','']
(root/'narration_script.md').write_text('\n'.join(lines))
