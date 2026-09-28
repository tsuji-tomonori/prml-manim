from pathlib import Path
from narration_content import SCENES
lines = ['# PRML 5.5 ニューラルネットワークの正則化', '', '印刷 pp.256–272（PDF pp.276–292）。本節に表はない。図5.18は次節5.6の内容として対象外。', '表示と読み上げは独立管理。数値実験は自作。式(5.132)等は公式正誤表を反映。', '']
for s in SCENES:
    lines += [f"## {s['id']} {s['title']}", '', f"原文: {s['reference']}", '']
    for i, b in enumerate(s['beats'], 1):
        lines += [f"### 操作{i}: {b['visual_note']}", '']
        for seg in b['segments']:
            lines += [f"- 表示: {seg['display']}", f"- 音声: {seg['speech']}"]
        lines += ['']
Path(__file__).with_name('narration_script.md').write_text('\n'.join(lines)+'\n')
