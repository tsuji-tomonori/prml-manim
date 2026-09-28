"""Export the exact spoken/display text and measured timing from the source of truth."""
import json
from pathlib import Path
from narration_content import SCENES
root=Path(__file__).resolve().parent
p=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(p.read_text()).get('scenes',[])} if p.exists() else {}
lines=['# PRML 5.6 混合密度ネットワーク：収録台本','','原文：Bishop (2006) §5.6、印刷 pp.272–277（手元PDF pp.292–297）。図5.18–5.21、式5.148–5.160。本節に表はありません。','',
'一変数の答えに限定して説明します。成分数を K とし、出力数は 3K。多次元の等方ガウスでは K(D+2) 出力です。式5.157は著者正誤表の訂正後を使います。', '',
'自作データ：u∈[0.02,0.98]、v=u+0.28 sin(2πu)+一様ノイズ[-0.045,0.045]。逆問題は(x,t)=(v,u)。学習済みモデルと説明用つまみの実験は区別します。','',
'| シーン | 秒 | 原文参照 |','|---|---:|---|']
for s in SCENES:
    dur=entries.get(s['id'],{}).get('duration',0)
    lines.append(f"| {s['id']} {s['title']} | {dur:.3f} | {s['reference']} |")
for s in SCENES:
    lines += ['',f"## {s['id']}：{s['title']}",'',f"原文参照：{s['reference']}",'']
    for i,b in enumerate(s['beats'],1):
        lines += [f"### {i}. {b['visual_note']}",'']
        for seg in b['segments']:
            lines += [f"- 字幕：{seg['display']}",f"- 音声：{seg['speech']}"]
        lines.append('')
(root/'narration_script.md').write_text('\n'.join(lines).rstrip()+'\n')
