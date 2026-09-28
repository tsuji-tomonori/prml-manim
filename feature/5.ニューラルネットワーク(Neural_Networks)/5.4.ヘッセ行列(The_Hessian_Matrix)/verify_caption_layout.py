"""Check every subtitle in an isolated TeX directory, safe during rendering."""
import json
from pathlib import Path
from manim import config
from narration_content import SCENES
from video_support import caption_mobject
config.media_dir=str(Path(__file__).parent/'media/caption-check')
Path(config.media_dir).mkdir(parents=True,exist_ok=True)
rows=[]
for s in SCENES:
    for b in s['beats']:
        for c in b['segments']:
            m=caption_mobject(c['display'])
            assert m.get_left()[0]>=-6.45 and m.get_right()[0]<=6.45
            assert m.get_bottom()[1]>=-3.9 and m.get_top()[1]<=-2.9
            rows.append(dict(id=c['id'],width=m.width,height=m.height))
result=dict(count=len(rows),max_width=max(r['width'] for r in rows),max_height=max(r['height'] for r in rows),cues=rows)
Path('caption_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['count'],result['max_width'],result['max_height'])
