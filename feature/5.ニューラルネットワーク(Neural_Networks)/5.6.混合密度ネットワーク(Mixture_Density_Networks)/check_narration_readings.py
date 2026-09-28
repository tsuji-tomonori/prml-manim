"""Query every sentence with WhiteCUL and preserve the before/after reading audit.

First run: --baseline <JSON of {id,speech,reading,kana}>.
Later runs reuse the baseline in reading_check.json. This collects evidence;
human review of the complete Markdown table is still required.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from make_voicevox_narration import post_json
from narration_content import SCENES

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:50021')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    with urllib.request.urlopen(args.base_url + '/version', timeout=5) as response:
        version = json.loads(response.read())
    destination = ROOT / 'reading_check.json'
    previous = json.loads(destination.read_text()) if destination.exists() else {}
    baseline = json.loads(args.baseline.read_text()) if args.baseline else [r['before'] for r in previous.get('sentences', [])]
    before = {r['id']: r for r in baseline}
    segments = [s for scene in SCENES for b in scene['beats'] for s in b['segments']]
    def query(s):
        q = json.loads(post_json(args.base_url, 'audio_query', {'speaker': 23, 'text': s['speech']}))
        reading = ''.join(m['text'] for p in q['accent_phrases'] for m in p['moras'])
        old = before.get(s['id'], dict(id=s['id'], speech=s['speech'], reading=reading, kana=q['kana']))
        return dict(**s, before=old, reading=reading, kana=q['kana'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(query, segments))
    destination.write_text(json.dumps(dict(engine=version, speaker=23, sentences=rows), ensure_ascii=False, indent=2) + '\n')
    lines = ['# 全文の読み確認', '',
             f'VOICEVOX Engine {version} / WhiteCUL ノーマル（23）。全{len(rows)}文。',
             '`audio_query` の `accent_phrases[].moras[].text` を順に連結。句読点・アクセント位置は省略し、長母音は Engine の表記（ケエスウ等）を保つ。',
             '元の speech と修正前の読みを保存し、再実行時にも上書きしない。kana のアクセント情報は `reading_check.json` に保存。',
             '以下は API の結果であり、聴取済みを意味しない。全文の目視確認結果は今回の作業レポートを参照。', '',
             '## 変更した読み上げ', '', '| ID | 修正前の文 | 修正前の読み | 修正後の文 | 修正後の読み |', '|---|---|---|---|---|']
    for r in rows:
        b = r['before']
        if b['speech'] != r['speech']:
            lines.append(f"| {r['id']} | {b['speech']} | {b['reading']} | {r['speech']} | {r['reading']} |")
    lines += ['', '## 全文（字幕 → 読み上げ → 実際の読み）', '', '| ID | display | speech | 修正前の読み | 修正後の読み |', '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['display']} | {r['speech']} | {r['before']['reading']} | {r['reading']} |")
    (ROOT / 'reading_check.md').write_text('\n'.join(lines) + '\n')
    for r in rows:
        print(r['id'], r['speech'], '→', r['reading'])


if __name__ == '__main__':
    main()
