"""Check narration integrity and all rendered caption bounds, without video rendering."""
import json
import re
from pathlib import Path
from manim import config
from narration_content import SCENES
from make_voicevox_narration import valid_entry
from scene_support import caption_mobject, caption_math, jp

ROOT=Path(__file__).resolve().parent
config.media_dir=str(ROOT/'media'/'caption-check')
config.verbosity='ERROR'
manifest=json.loads((ROOT/'assets/voicevox/manifest.json').read_text())
assert len(manifest['scenes'])==len(SCENES)
assert all(valid_entry(s,e) for s,e in zip(SCENES,manifest['scenes']))
assert {f.name for f in (ROOT/'assets/voicevox').glob('*.wav')}=={s['id']+'.wav' for s in SCENES}
segments=[s for c in SCENES for b in c['beats'] for s in b['segments']]
readings=json.loads((ROOT/'reading_check.json').read_text())['sentences']
assert [(s['id'],s['speech'],s['display']) for s in segments]==[(s['id'],s['speech'],s['display']) for s in readings]
pattern=r'エックス|ミュー|シグマ|ラムダ|エヌ|ニュー|エム|ティー|エー'
assert not any(re.search(pattern,s['display']) for s in segments)
sizes=[(m.width,m.height) for s in segments for m in [caption_mobject(s['display'])]]
print(json.dumps({'wavs':len(SCENES),'sentences':len(segments),'inline_math':sum('$' in s['display'] for s in segments),'caption_max_width':max(w for w,h in sizes),'caption_max_height':max(h for w,h in sizes)},indent=2))
print('PASS: WAV/script hashes, cue identity, reading audit identity, no phonetic notation in display, all caption safe areas')
