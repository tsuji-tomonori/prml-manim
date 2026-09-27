"""Measure every display cue; visual review of extracted frames is separate."""
import json
from pathlib import Path
from manim import config
from narration_content import SCENES
from scene_support import caption_mobject

config.media_dir = '/tmp/prml45-caption-media'
Path(config.media_dir).mkdir(exist_ok=True)
rows = []
for scene in SCENES:
    for beat in scene['beats']:
        for segment in beat['segments']:
            m = caption_mobject(segment['display'])
            assert m.get_left()[0] >= -6.45 and m.get_right()[0] <= 6.45
            assert m.get_bottom()[1] >= -3.9 and m.get_top()[1] <= -2.9
            rows.append(dict(id=segment['id'],width=m.width,height=m.height))
result=dict(count=len(rows),max_width=max(r['width'] for r in rows),
            max_height=max(r['height'] for r in rows),cues=rows)
Path('caption_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['count'],result['max_width'],result['max_height'])
