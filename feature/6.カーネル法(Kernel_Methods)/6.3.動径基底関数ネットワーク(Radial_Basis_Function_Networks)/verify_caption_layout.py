"""Measure every display caption; retain evidence separate from visual review."""
import json,re
from pathlib import Path
from manim import tempconfig
from scene_support import caption_mobject
from narration_content import SCENES
root=Path(__file__).resolve().parent
with tempconfig({'media_dir':str(root/'media/caption-check')}):
    sizes=[]
    displays=[]
    for scene in SCENES:
        for beat in scene['beats']:
            for s in beat['segments']:
                m=caption_mobject(s['display']);sizes.append(dict(id=s['id'],width=m.width,height=m.height,math='$' in s['display']))
                displays.append(s['display'])
    forbidden=r'エックス|ラムダ|ミュー|シグマ|アルファ|ベータ|ガンマ|カッパ|ダブリュー|ティー|エヌ|エム'
    assert not re.search(forbidden,'\n'.join(displays))
    data=dict(count=len(sizes),math_count=sum(x['math'] for x in sizes),max_width=max(x['width'] for x in sizes),max_height=max(x['height'] for x in sizes),sentences=sizes)
    (root/'caption_validation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (root/'media/display_captions.txt').write_text('\n'.join(displays)+'\n')
    print({k:v for k,v in data.items() if k!='sentences'})
