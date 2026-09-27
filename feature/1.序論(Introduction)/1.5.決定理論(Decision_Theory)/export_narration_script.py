"""Export the reviewed display/speech pairs and source correspondence."""
import json
from pathlib import Path
from narration_content import SCENES
root = Path(__file__).resolve().parent
manifest = root / 'assets/voicevox/manifest.json'
entries = {e['id']: e for e in json.loads(manifest.read_text()).get('scenes', [])} if manifest.exists() else {}
lines = ['# PRML 1.5 決定理論：ナレーション台本', '',
         '問い → 動く図 → 式の順に進む。display は字幕、speech は WhiteCUL 用。数値は説明用の自作モデルであり、診断性能の実測値ではない。', '',
         '## 原文との照合', '',
         '- 印刷 pp.38–48（PDF pp.58–68）、Fig.1.29 は印刷 p.49（PDF p.69）。',
         '- Fig.1.25 は表の形をした図。行＝真のクラス k、列＝判断 j。損失 0/1000/1/0 は原文例。',
         '- Eq.(1.90) は著者の正誤表に従い、第2項を ∫Var[t|x]p(x)dx とする。',
         '- Fig.1.27 の密度と事後確率の違いは独自のガウス例で示す。原図の左側のモードは複製しない。',
         '- Eq.(1.91) は q=2 と q=1、および損失形状の連続変形を扱う。原文の q→0 と最頻値の記述は一般定理として用いない。連続分布の |y−t|^q は q→0 でほとんど至る所1となり、最小化の極限には別途検討が要る。最頻値は独立の狭い許容幅による判断として説明する。',
         '- Eq.(1.88) の変分微分は、高校数学向けに固定した x における二乗の展開と分散への分解で説明する。', '',
         '著者の正誤表: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf', '']
for s in SCENES:
 e=entries.get(s['id'],{}); duration=e.get('duration') if e.get('status')=='generated' else None
 lines += [f"## {s['id']}: {s['title']}", '', f"原文: {s['reference']}", f"視覚: {s['idea']}", f"実測尺: {duration:.3f}秒" if duration else '実測尺: 音声生成後に更新', '']
 for i,b in enumerate(s['beats']):
  lines += [f"### {i+1}. {b['visual_note']}", '']
  for seg in b['segments']:
   lines += [f"- `{seg['id']}`", f"  - display: {seg['display']}", f"  - speech: {seg['speech']}"]
  lines += ['']
(root/'narration_script.md').write_text('\n'.join(lines)+'\n')
