"""Export the full reviewed display/speech storyboard and measured timings."""
import json
from pathlib import Path
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry
entries = {e['id']:e for e in json.loads(MANIFEST.read_text()).get('scenes',[])} if MANIFEST.exists() else {}
lines=['# PRML 4.2 確率的生成モデル：ナレーション台本','',
 '原文：Bishop (2006), §4.2、印刷 pp.196–203（PDF pp.216–223）。原文を pdftotext -layout で抽出して確認。図4.9–4.11、式(4.57)–(4.86)。この節に表はありません。',
 '図はすべて自作データ。図4.9はシグモイドを採用し、probitとの比較は次節の話題として省略します。式(4.71)の尤度はラベルだけでなく観測とラベルの同時尤度として L と表記します。式(4.84)の尺度因子との混同を避け、(4.85)–(4.86)の表示は s=1 に限定します。', '',
 '字幕 display と発話 speech は別データです。$...$ は日本語の文字高と本体中心に合わせた MathTex として表示します。音声の各文PCM長を字幕と動作の共通時計に使います。','']
start=0
for s in SCENES:
 e=entries.get(s['id'],{}); duration=e.get('duration',0) if valid_entry(s,e) else 0
 lines += [f"## {s['id']}：{s['title']}",'',f"原文参照：{s['reference']}", f"実測：開始 {start:.3f} 秒 / 尺 {duration:.3f} 秒" if duration else '尺：音声生成後に実測値を出力。','']
 for i,b in enumerate(s['beats'],1):
  lines += [f"### Beat {i}：{b['visual_note']}",'']
  for c in b['segments']:
   lines += [f"- `{c['id']}` 字幕：{c['display']}",f"  読み上げ：{c['speech']}"]
  lines.append('')
 start+=duration
Path('narration_script.md').write_text('\n'.join(lines).rstrip()+'\n')
