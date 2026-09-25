"""Export the human-readable script from the independently authored cue pairs."""
import json
from pathlib import Path
from narration_content import SCENES
D=Path(__file__).resolve().parent
m=D/'assets/voicevox/manifest.json'
entries={s['id']:s for s in json.loads(m.read_text()).get('scenes',[])} if m.exists() else {}
lines=['# PRML 1.2 確率論：面積から予測の分布へ','',
'原文は Bishop (2006), 印刷 pp.12–32、§1.2–1.2.6。PDFを pdftotext -layout で抽出して本文・式・図説明を照合。該当節に表はない。',
'高校数学を前提とし、積分は細い面積の和、ベクトルは数の並びとして導入する。行列計算の導出、歴史的欄外記事、bootstrap の詳細は省略する。',
'図は自作。箱の例は赤=(リンゴ1,オレンジ3)、青=(リンゴ4,オレンジ1)、事前0.3であり、原文の数値例とは区別する。',
'ベイズ予測の行列式は §3.3 の形と整合する S⁻¹=αI+βΣφ(xₙ)φ(xₙ)ᵀ を使用。PDF の式(1.72)の第2因子には n が見えないが、訓練入力同士の外積が正しい。',
'動画の音声は VOICEVOX:WhiteCUL（23）。全文の API 読みは reading_check.md。字幕は display、音声は speech の独立データ。','',
'## 構成','', '| シーン | 題名 | 秒 | 原文 |','|---|---|---:|---|']
for s in SCENES:lines.append(f"| {s['id']} | {s['title']} | {entries.get(s['id'],{}).get('duration','生成後に確定')} | {s['reference']} |")
for s in SCENES:
 lines += ['',f"## {s['id']}: {s['title']}",'',f"原文: {s['reference']}",'']
 for i,b in enumerate(s['beats'],1):
  lines += [f"### {i}. {b['visual_note']}",'']
  for x in b['segments']:lines += [f"- `{x['id']}` 字幕: {x['display']}",f"  音声: {x['speech']}"]
  lines+=['']
(D/'narration_script.md').write_text('\n'.join(lines).rstrip()+'\n')
