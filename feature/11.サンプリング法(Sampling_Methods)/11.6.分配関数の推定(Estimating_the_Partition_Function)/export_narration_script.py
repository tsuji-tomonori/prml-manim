"""Export the complete display/speech script and measured scene timings."""
import json
from pathlib import Path
from narration_content import SCENES,script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 11.6 分配関数の推定 台本','','原文: Bishop (2006), §11.6、印刷 pp.554–555 / PDF pp.574–575。','公式正誤表: pp.554–555式(11.72)–(11.73)の1/L補正を確認。','字幕 display は記号、読み上げ speech は発音。各 beat は文 WAV の実測時間に同期する。','']
start=0.
for s in SCENES:
    e=entries.get(s['id'],{});ds=e.get('beat_durations',[])
    current=e.get('script_sha256')==script_hash(s)
    duration=sum(ds) if current else 0
    lines += [f"## {s['id']} {s['title']}",'',f"原文: {s['reference']}",(f"開始 {start:.3f} 秒 / 尺 {duration:.3f} 秒（音声生成後の実測）" if current and start is not None else '時刻: 音声の再生成後に確定'),'']
    for i,b in enumerate(s['beats']):
        lines += [f"### beat {i+1}: {b['visual_note']}",'']
        for seg in b['segments']:lines += [f"- {seg['id']} 字幕: {seg['display']}",f"  音声: {seg['speech']}"]
        lines += ['']
    start=start+duration if current and start is not None else None
lines += ['## 内容の対応と条件','','- ガウス形の2つのエネルギーと中間分布の条件は partition_model.py と README.md に記載。','- 80個の直接標本と各段階の80個、8比の積は verify_numerics.py で照合。','- 元の PDF と公式正誤表の照合範囲は README.md を参照。','']
(root/'narration_script.md').write_text('\n'.join(lines))
