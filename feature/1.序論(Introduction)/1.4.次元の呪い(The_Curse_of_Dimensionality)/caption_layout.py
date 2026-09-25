"""Inline Japanese/MathTex layout, inherited from PRML 1.1."""
import re
import numpy as np
from manim import *

def jp(text, size=25, color=WHITE):
    return Text(text, font="Noto Sans CJK JP", font_size=size, color=color)


def tex(text, size=30, color=WHITE):
    return MathTex(text, font_size=size, color=color)


def caption_math(formula, kana_height):
    """Scale to kana x-height and center the unscripted body on the text axis."""
    body = re.sub(r"[_^](?:\{[^}]*\}|\\[A-Za-z]+|.)", "", formula)
    reference = MathTex(formula, body, 'x', font_size=26)
    # Measure actual glyphs: font_size has different meanings in Pango and TeX.
    scale = .67 * kana_height / reference[2].height
    # Large operators and tall delimiters should remain inline-sized.
    scale = min(scale, 1.12 * kana_height / reference[1].height)
    reference.scale(scale)
    mob = reference[0].copy()
    mob.shift(LEFT * mob.get_center()[0] + DOWN * reference[1].get_center()[1])
    return mob


def caption_mobject(display):
    """Wrap Japanese and atomic inline math with a shared body centerline."""
    kana_height = jp('あ', 22).height
    gap = .35 * kana_height
    tokens = re.findall(r"\$[^$]+\$|.", display)
    # Include the enlarged formula and both side bearings when choosing a wrap.
    weights = [max(1, (caption_math(t[1:-1], kana_height).width + 2*gap) / .30)
               if t.startswith('$') else (.55 if t.isascii() else 1) for t in tokens]
    if sum(weights) > 32:
        cumulative = np.cumsum(weights)
        split = min(range(1, len(tokens)), key=lambda i:
                    abs(cumulative[i-1] - sum(weights)/2) + (0 if tokens[i-1] in '、。' else 5))
        lines = [tokens[:split], tokens[split:]]
    else:
        lines = [tokens]
    def line_mobject(parts):
        runs = re.split(r"(\$[^$]+\$)", ''.join(parts))
        mobs = []
        cursor = 0.
        for run in (r for r in runs if r.strip()):
            if run.startswith('$'):
                mob = caption_math(run[1:-1], kana_height)
            else:
                # Reference kana keep standalone punctuation in its natural place.
                reference = jp('あ' + run.strip() + 'あ', 22)
                mob = VGroup(*reference[1:-1])
                mob.shift(DOWN * reference[0].get_center()[1])
            # Horizontal placement must not recenter superscripts/subscripts.
            mob.shift(RIGHT * (cursor - mob.get_left()[0]))
            cursor = mob.get_right()[0] + gap
            mobs.append(mob)
        return VGroup(*mobs)
    caption = VGroup(*[line_mobject(line) for line in lines]).arrange(DOWN, buff=.13)
    caption.move_to([0, -3.4, 0])
    if caption.width > 12.9 or caption.height > .9:
        raise ValueError(f'Caption outside safe area ({caption.width}, {caption.height}): {display}')
    return caption


