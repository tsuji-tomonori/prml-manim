"""Measure every display caption; retain evidence separate from visual review."""
import ast,json,re
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
    tree=ast.parse((root/'prml_9_3_alternative_em.py').read_text())
    screen_text=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and ((isinstance(n.func,ast.Name) and n.func.id=='J') or
                      (isinstance(n.func,ast.Attribute) and n.func.attr in ('say','tag','note_anim'))) and n.args
                 and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)
                 and '\\' not in n.args[0].value]
    screen_text += [kw.value.value for n in ast.walk(tree) if isinstance(n,ast.Call)
                    for kw in n.keywords if kw.arg=='value' and isinstance(kw.value,ast.Constant)
                    and isinstance(kw.value.value,str)]
    all_text='\n'.join(displays+screen_text+[s['title'] for s in SCENES])
    assert not re.search(r'[一二三四五六七八九十][つ個本回日]',all_text)
    assert not re.search(r'\b(?:fixed|separable|backprop)\b',all_text,re.I)
    assert not any(word!='ICM' for note in screen_text for word in re.findall(r'[A-Za-z]{3,}',note))
    data=dict(count=len(sizes),math_count=sum(x['math'] for x in sizes),max_width=max(x['width'] for x in sizes),max_height=max(x['height'] for x in sizes),sentences=sizes)
    (root/'caption_validation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (root/'media/display_captions.txt').write_text('\n'.join(displays)+'\n')
    print({k:v for k,v in data.items() if k!='sentences'})
