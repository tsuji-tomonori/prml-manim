"""Export the canonical display/speech and actual narration timings."""
import json
from pathlib import Path
from narration_content import SCENES
from make_voicevox_narration import MANIFEST

def main():
    entries = {s['id']:s for s in json.loads(MANIFEST.read_text())['scenes']}
    lines = ['# PRML 4.5 ベイズロジスティック回帰：制作台本', '',
             '原文: 本文 pp.217–220（PDF pp.237–240）、式 (4.140)–(4.155)。',
             '4.5 節に固有の図・表はない。Fig.4.9（p.211）を補助参照し、数値から独自に描く。',
             '公式正誤表に従い (4.143) の左辺は S_N^{-1}。Φ は標準正規 CDF（inverse probit）。予測近似には ≃ を使う。',
             '式 (4.146)–(4.148) のデルタ関数による表現は、候補点をロジットへ写す図で説明する。',
             '場面2–3は切片0のスカラーモデル。場面4以降は切片と傾きの2変数モデル。',
             '場面6–7の分散操作は平均2を固定した実験。場面8の曲線の変形中は共分散を補間する。',
             'display は記号付き字幕、speech は読み上げ専用。各文のPCM実測尺で字幕と操作を同期する。', '',
             '| シーン | 開始秒 | 尺（秒） | 原文参照 |', '|---|---:|---:|---|']
    start = 0.
    for s in SCENES:
        e = entries[s['id']]
        lines.append(f"| {s['id']} {s['title']} | {start:.3f} | {e['duration']:.3f} | {s['reference']} |")
        start += e['duration']
    for s in SCENES:
        e = entries[s['id']]
        lines += ['', f"## {s['id']}: {s['title']}", '', f"原文参照: {s['reference']}", '']
        for j,b in enumerate(s['beats']):
            lines += [f"### {j+1}. {b['visual_note']}（{e['beat_durations'][j]:.3f}秒）", '']
            for seg in b['segments']:
                lines += [f"- 字幕: {seg['display']}",f"  読み上げ: {seg['speech']}"]
            lines.append('')
    Path('narration_script.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
