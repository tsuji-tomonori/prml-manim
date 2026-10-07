"""Export the authored story with measured speech and subtitle boundaries."""
from __future__ import annotations

import json
from pathlib import Path

from make_voicevox_narration import MANIFEST, valid_entry
from narration_content import SCENES, SYNTHESIS_SETTINGS


def main():
    entries = {e['id']: e for e in json.loads(MANIFEST.read_text()).get('scenes', [])}
    for story in SCENES:
        if not valid_entry(story, entries.get(story['id'], {})):
            raise RuntimeError(f"Regenerate speech before exporting {story['id']}")
    count = sum(len(b['segments']) for s in SCENES for b in s['beats'])
    lines = [
        '# PRML 1.1 台本：完璧な曲線の、意外な失敗', '',
        '原文: Bishop (2006), Chapter 1, 印刷 pp.4–12。数値・点・図は自作。',
        '正本は `narration_content.py`。字幕は `display`、音声は `speech`。',
        '表示上の個数・番号・倍率はアラビア数字を使い、二次式・二乗などの数学用語は漢字を保つ。',
        f"VOICEVOX:WhiteCUL（23）、話速{SYNTHESIS_SETTINGS['speedScale']}、抑揚{SYNTHESIS_SETTINGS['intonationScale']}。",
        f'全{count}文。PCMの実測尺に字幕と動作を同期。読みのAPI結果は [reading_check.md](reading_check.md)。', '',
        '## 物語の軸', '',
        '10点すべてを通る赤い曲線が、新しい1点を大きく外す。',
        '曲線のつまみ、誤差の採点、過学習、係数の裏側を調べ、データ追加と正則化を試す。',
        '最後に冒頭と同じ1点へ戻り、改善と残る不確かさを見て、確率の問いへ進む。', '',
        '## シーン構成', '',
        '| シーン | 題名 | 尺（秒） | 原文参照 |',
        '|---|---|---:|---|',
    ]
    for story in SCENES:
        entry = entries[story['id']]
        lines.append(f"| {story['id']} | {story['title']} | {entry['duration']:.3f} | {story['reference']} |")
    lines += ['', f"合計 {sum(e['duration'] for e in entries.values()):.3f} 秒。"]
    total = 0.
    for story in SCENES:
        entry = entries[story['id']]
        lines += ['', f"## {story['id']}: {story['title']}", '',
                  f"原文参照: {story['reference']}", '',
                  f"開始 {total:.3f} 秒 / 尺 {entry['duration']:.3f} 秒"]
        offset = 0.
        for i, (item, duration) in enumerate(zip(story['beats'], entry['beat_durations'])):
            lines += ['', f"### Beat {i + 1}（シーン内 {offset:.3f}〜{offset + duration:.3f} 秒）", '',
                      f"演出メモ: {item['visual_note']}", '',
                      '| ID | 開始〜終了（シーン内秒） | 字幕 | 読み上げ |',
                      '|---|---|---|---|']
            for cue in entry['subtitle_cues']:
                if cue['beat_index'] == i:
                    lines.append(f"| {cue['id']} | {cue['start']:.3f}〜{cue['end']:.3f} | {cue['display']} | {cue['speech']} |")
            offset += duration
        total += entry['duration']
    (Path(__file__).resolve().parent / 'narration_script.md').write_text(
        '\n'.join(lines).rstrip() + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
