"""Export the complete display/speech script and measured scene timings."""
import json
from pathlib import Path
from narration_content import SCENES,script_hash
root=Path(__file__).resolve().parent
path=root/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text()).get('scenes',[])} if path.exists() else {}
lines=['# PRML 10.7 期待値伝播 台本','','原文: Bishop (2006), §10.7、印刷 pp.505–517 / PDF pp.525–537。','公式正誤表: p.509 の cavity 添字、(10.217)、(10.218)、(10.228) の訂正を確認。','字幕 display は記号、読み上げ speech は発音。各 beat は文 WAV の実測時間に同期する。','']
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
lines += ['## 内容の対応と条件','','- 自作5点とクラッタ混合モデルの条件は ep_model.py と README.md に記載。','- 順方向・逆順 ADF、8巡 EP、真の事後の数値積分は verify_numerics.py で照合。','- 元の PDF と公式正誤表の照合範囲は README.md を参照。','']
(root/'narration_script.md').write_text('\n'.join(lines))
