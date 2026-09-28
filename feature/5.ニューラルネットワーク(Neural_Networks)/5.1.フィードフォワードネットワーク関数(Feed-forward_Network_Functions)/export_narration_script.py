"""Export the authoritative display/speech storyboard."""
import json
from pathlib import Path
from narration_content import SCENES
ROOT=Path(__file__).resolve().parent
m=ROOT/'assets/voicevox/manifest.json'
entries={x['id']:x for x in json.loads(m.read_text()).get('scenes',[])} if m.exists() else {}
lines=['# PRML 5.1 フィードフォワードネットワーク関数', '',
       '問い → 動く実験 → 式で確認。印刷 pp.227–232（PDF pp.247–252）、§5.1.1を含む。該当節に表はない。',
       '図は自作データ・自作配置で再構成。Fig.5.3 の三隠れユニットの構造を用いるが、サンプルと波の周期は独自。',
       'Fig.5.4 は隠れ出力 z=0.5 の等高線と最終出力 y=0.5 の境界を区別し、原図の最適境界は再現しない。', '',
       '字幕は display、音声は speech。時間は音声生成後の PCM 実測。', '']
for scene in SCENES:
 e=entries.get(scene['id'],{})
 lines += ['## '+scene['id']+' '+scene['title'], '', '原文参照: '+scene['reference'],
           '音声尺: '+str(e.get('duration','音声生成後に確定'))+' 秒', '']
 for i,b in enumerate(scene['beats']):
  lines += ['### '+str(i+1)+'. '+b['visual_note'], '']
  for q in b['segments']:
   lines += ['- '+q['id']+' 字幕: '+q['display'], '  音声: '+q['speech']]
  lines += ['']
(ROOT/'narration_script.md').write_text('\n'.join(lines)+'\n')
