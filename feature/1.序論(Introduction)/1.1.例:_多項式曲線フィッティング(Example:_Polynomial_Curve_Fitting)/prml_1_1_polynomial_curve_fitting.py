"""PRML 1.1 — continuous, linked visual experiments (Manim Community)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from manim import *

from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry
from narration_content import SCENES, estimated_duration
from depth_visuals import DepthDot, LossLandscape, raised_panel, relief_faces
from polynomial_model import (
    NOISE_STD, T, TT, T_ALL, WEIGHTS, X, XT, X_ALL, TRAIN_RMS, TEST_RMS,
    degree_weights, eval_poly, growing_weights, ridge_weights,
    rms_error, sine, linear_fit_error,
)

BLUE_DATA = ManimColor("#58B5ED")
TRUE_GREEN = ManimColor("#77D49A")
MODEL_RED = ManimColor("#FF6B77")
TEST_ORANGE = ManimColor("#FFB45B")
RESIDUAL_YELLOW = ManimColor("#FFE079")
REG_PURPLE = ManimColor("#C29AFF")
MUTED = ManimColor("#A8B2C5")
BG = "#10141F"
TERM_COLORS = [BLUE_DATA, RESIDUAL_YELLOW, REG_PURPLE, TEST_ORANGE]
# Shared visual-aid palette from the chapter 1–5 review plan.
AID_INPUT = ManimColor("#58C4DD")
AID_OPERATION = ManimColor("#FFFF00")
AID_RESULT = ManimColor("#83C167")
SCENE_DIR = Path(__file__).resolve().parent


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


def polyline(points, color, width=3, opacity=1):
    return VMobject().set_points_as_corners(points).set_stroke(color, width, opacity)


def graph_curve(ax, weights=None, values=None, color=MODEL_RED, opacity=1):
    u = np.linspace(0, 1, 241)
    v = eval_poly(weights, u) if values is None else values(u)
    # All experimental curves are displayed at their true values; no clipping.
    origin = ax.c2p(0, 0)
    points = origin + u[:, None] * (ax.c2p(1, 0) - origin) + v[:, None] * (ax.c2p(0, 1) - origin)
    return polyline(points, color, 3, opacity)


def data_dots(ax, x=X, t=T, color=BLUE_DATA, radius=.052):
    dots = VGroup(*[DepthDot(ax.c2p(a, b), radius=radius, color=color) for a, b in zip(x, t)])
    if getattr(ax, 'dynamic_span', False):
        for dot, a, b in zip(dots, x, t):
            dot.add_updater(lambda m, a=a, b=b: m.move_to(ax.c2p(a, b)))
    return dots


def residuals(ax, w, x=X, t=T, color=RESIDUAL_YELLOW):
    return VGroup(*[Line(ax.c2p(a, b), ax.c2p(a, c), color=color, stroke_width=2)
                    for a, b, c in zip(x, t, eval_poly(w, x))])


def readout(label, getter, position, color=WHITE, places=3, size=25):
    prefix = tex(label, size, color)
    number = DecimalNumber(getter(), num_decimal_places=places, font_size=size, color=color)
    group = VGroup(prefix, number).arrange(RIGHT, buff=.15).move_to(position)
    anchor = number.get_left().copy()
    number.add_updater(lambda m: m.set_value(getter()).move_to(anchor, aligned_edge=LEFT))
    return group


class PRML11PolynomialCurveFitting(Scene):
    scene_indices = range(9)
    timeline_name = 'prml11_timeline.json'

    def construct(self):
        self.camera.background_color = BG
        self.timeline = []
        self.manifest = {e['id']: e for e in json.loads(MANIFEST.read_text()).get('scenes', [])} if MANIFEST.exists() else {}
        methods = [self.question, self.knobs, self.squares, self.valley, self.degrees,
                   self.coefficients, self.more_data, self.regularization, self.uncertainty]
        for i in self.scene_indices:
            self.begin(i)
            methods[i]()
            if self.beat_index != len(self.story['beats']):
                raise RuntimeError(f"Unconsumed narration in {self.story['id']}")
            self.timeline[-1]['end'] = float(self.time)
        out = Path(config.media_dir) / self.timeline_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.timeline, ensure_ascii=False, indent=2) + '\n')

    def begin(self, index):
        self.clear()
        self.story = SCENES[index]
        self.beat_index = 0
        self.subtitle = None
        self.next_section(self.story['id'])
        title = jp(self.story['title'], 34).move_to([0, 3.35, 0])
        self.add(title)
        legend = VGroup(*[VGroup(Dot(radius=.045, color=c), jp(s, 18, c)).arrange(RIGHT, buff=.1)
                          for s, c in [('観測', BLUE_DATA), ('生成関数', TRUE_GREEN), ('モデル', MODEL_RED)]]).arrange(RIGHT, buff=.55)
        legend.move_to([0, 2.78, 0])
        self.add(legend)
        entry = self.manifest.get(self.story['id'], {})
        audio_valid = valid_entry(self.story, entry)
        self.audio_entry = entry if audio_valid else None
        self.durations = entry['beat_durations'] if audio_valid else [estimated_duration(b) for b in self.story['beats']]
        # Frame quantization is applied to cumulative boundaries, so audio does not drift.
        fps = config.frame_rate
        self.boundaries = np.ceil(np.cumsum(self.durations) * fps - 1e-6) / fps
        self.scene_start = float(self.time)
        if audio_valid:
            self.add_sound(str(OUTPUT_DIR / f"{self.story['id']}.wav"))
        self.timeline.append({'id': self.story['id'], 'title': self.story['title'],
                              'start': self.scene_start, 'reference': self.story['reference'],
                              'audio': audio_valid, 'beats': []})

    def beat_cues(self):
        offset = sum(self.durations[:self.beat_index])
        if self.audio_entry:
            return [dict(id=c['id'], display=c['display'], start=c['start'] - offset, end=c['end'] - offset)
                    for c in self.audio_entry['subtitle_cues'] if c['beat_index'] == self.beat_index]
        segments = self.story['beats'][self.beat_index]['segments']
        lengths = np.array([len(s['speech']) for s in segments], dtype=float)
        boundaries = np.r_[0, np.cumsum(lengths / lengths.sum() * self.durations[self.beat_index])]
        return [dict(id=s['id'], display=s['display'], start=float(a), end=float(b))
                for s, a, b in zip(segments, boundaries, boundaries[1:])]

    def sentence_duration(self, index):
        cue = self.beat_cues()[index]
        return cue['end'] - cue['start']

    def beat(self, *animations, moving=True, start_sentence=0, end_sentence=None, actions=None, phases=None):
        # Captions use display notation and the PCM duration of the paired speech.
        # The visual action shares this clock; no minimum-duration silent padding.
        item = self.story['beats'][self.beat_index]
        if self.subtitle is not None:
            self.remove(self.subtitle)
        cues = self.beat_cues()
        captions = VGroup()
        for cue in cues:
            caption = caption_mobject(cue['display']).set_opacity(0)
            captions.add(caption)
        self.subtitle = captions
        self.add(captions)
        start = float(self.time)
        fps = config.frame_rate
        frames = round((self.scene_start + self.boundaries[self.beat_index] - start) * fps)
        duration = frames / fps
        action_start = cues[start_sentence]['start']
        action_end = cues[end_sentence - 1]['end'] if end_sentence is not None else cues[-1]['end']
        def caption_at(m, alpha):
            clock = alpha * duration
            index = max(i for i, c in enumerate(cues) if c['start'] <= clock + 1e-8)
            for i, caption in enumerate(m):
                caption.set_opacity(1 if i == index else 0)
        caption_at(captions, 0)
        record = {'start': start, 'end': start + duration, 'display': ''.join(s['display'] for s in item['segments']),
                  'action_start': start + action_start, 'action_end': start + action_end,
                  'cues': [dict(c, start=start+c['start'], end=start+c['end']) for c in cues]}
        if actions:
            record['actions'] = [dict(a, start=start+a['start'], end=start+a['end']) for a in actions]
        self.timeline[-1]['beats'].append(record)
        if phases:
            # Build later animations only after the previous phase has finished.
            # Otherwise Manim can suspend a future target's updater from frame 0,
            # or add a not-yet-stamped RMS marker to the scene prematurely.
            elapsed_frames = 0
            elapsed_seconds = 0.
            record['actions'] = []
            for name, seconds, factory in [*phases, ('breath', duration - sum(p[1] for p in phases), lambda: Wait())]:
                elapsed_seconds += seconds
                end_frame = min(frames, round(elapsed_seconds * fps))
                phase_frames = end_frame - elapsed_frames
                if phase_frames <= 0:
                    continue
                self.update_mobjects(0)
                animation = factory()
                phase_start = float(self.time)
                offset = elapsed_frames / fps
                phase_duration = phase_frames / fps
                self.play(animation, UpdateFromAlphaFunc(captions,
                          lambda m, alpha, offset=offset, d=phase_duration: caption_at(m, (offset + alpha*d) / duration),
                          rate_func=linear), run_time=(phase_frames-1e-5)/fps, rate_func=linear)
                record['actions'].append({'name': name, 'start': phase_start, 'end': float(self.time)})
                elapsed_frames = end_frame
            self.beat_index += 1
            return
        visual = []
        if animations:
            if action_start > 0:
                visual.append(Wait(action_start))
            visual.append(AnimationGroup(*animations, run_time=action_end-action_start))
            visual.append(Wait(max(.001, duration-action_end)))
        else:
            visual.append(Wait(duration))
        self.play(Succession(*visual), UpdateFromAlphaFunc(captions, caption_at, rate_func=linear),
                  run_time=(frames - 1e-5) / fps, rate_func=linear)
        self.beat_index += 1

    def axes(self, center=(-3.1, .1, 0), width=5.7, height=3.5, span=None):
        # A symmetric, continuously varying vertical scale preserves every peak.
        # Only the high-degree experiments zoom; ordinary curves use +/-1.5.
        span_fn = span or (lambda: 1.5)
        ax = Axes(x_range=[0, 1, .25], y_range=[-1.5, 1.5, 1],
                  x_length=width, y_length=height, tips=False,
                  axis_config={'color': MUTED, 'stroke_width': 1.4},
                  y_axis_config={'include_ticks': False}).move_to(center)
        origin = ax.c2p(0, 0).copy()
        unit_x = ax.c2p(1, 0) - origin
        half_y = (ax.c2p(0, 1.5) - origin).copy()
        ax.c2p = lambda x, y=0: origin + x * unit_x + y * half_y / span_fn()
        ax.dynamic_span = span is not None
        grid = always_redraw(lambda: VGroup(*[
            Line(ax.c2p(0, y), ax.c2p(1, y), color=MUTED, stroke_opacity=.12)
            for y in [-3, -2, -1, 1, 2, 3] if abs(y) < span_fn()]))
        labels = VGroup()
        for x in [0, .5, 1]:
            labels.add(tex(str(x), 19, MUTED).move_to(ax.c2p(x, -span_fn()) + DOWN * .22))
        for y in [-3, -2, -1, 0, 1, 2, 3]:
            label = tex(str(y), 19, MUTED)
            label.add_updater(lambda m, y=y: m.next_to(ax.c2p(0, y), LEFT, buff=.12)
                              .set_opacity(1 if abs(y) < span_fn() else 0))
            labels.add(label)
        labels.add(tex('x', 22).next_to(ax.c2p(1, -span_fn()), RIGHT, buff=.18))
        labels.add(tex(r't,\ y', 22).next_to(ax.c2p(0, span_fn()), UP, buff=.17))
        self.add(raised_panel(center, width + .85, height + .60), grid, ax, labels)
        return ax

    def curve_span(self, weights, minimum=1.5):
        return max(minimum, 1.10 * float(np.max(np.abs(eval_poly(weights, np.linspace(0, 1, 241))))))

    def slider(self, tracker, low, high, position, width=4.8, label='M', ticks=None, color=MODEL_RED):
        base = NumberLine(x_range=[low, high, 1], length=width, include_ticks=False,
                          color=MUTED, stroke_width=2).move_to(position)
        groove = base.copy().set_stroke(BLACK, 7)
        group = VGroup(groove, base)
        for v in (ticks if ticks is not None else range(int(low), int(high) + 1)):
            pos = base.n2p(v)
            group.add(Line(pos + DOWN * .07, pos + UP * .07, color=MUTED))
            group.add(tex(str(v), 17, MUTED).next_to(pos, DOWN, buff=.13))
        group.add(tex(label, 26, color).next_to(base, LEFT, buff=.25))
        knob = DepthDot(base.n2p(tracker.get_value()), color=color, radius=.105)
        knob.add_updater(lambda m: m.move_to(base.n2p(tracker.get_value())))
        group.add(knob)
        return group

    def degree_label(self, tracker, pos):
        labels = VGroup(*[tex(f'M={i}', 28, MODEL_RED).move_to(pos) for i in range(10)])
        labels.add(jp('遷移中', 22, MUTED).move_to(pos))
        def update(m):
            value = tracker.get_value()
            key = round(value) if abs(value - round(value)) < 1e-5 else 10
            for i, item in enumerate(m):
                item.set_opacity(1 if i == key else 0)
        labels.add_updater(update)
        update(labels)
        return labels

    def question(self):
        ax = self.axes(center=(0, .1, 0), width=9, height=3.7)
        dots = data_dots(ax)
        self.beat(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=.15), moving=False, end_sentence=1)
        cursor = ValueTracker(.45)
        guide = always_redraw(lambda: DashedLine(ax.c2p(cursor.get_value(), -1.4), ax.c2p(cursor.get_value(), 1.4), color=MUTED))
        self.add(guide)
        self.beat(cursor.animate.set_value(.72))
        truth = graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.4)
        equation = tex(r't_n=\sin(2\pi x_n)+\epsilon_n', 32).move_to([0, -2.45, 0])
        self.add(equation)
        self.beat(Create(truth), Indicate(equation, scale_factor=1.015), moving=False)
        self.beat(Indicate(dots, color=BLUE_DATA, scale_factor=1.04))
        model = graph_curve(ax, WEIGHTS[3])
        self.beat(Create(model), FadeOut(guide))
        test = data_dots(ax, XT[::10], TT[::10], TEST_ORANGE, .05)
        self.beat(LaggedStart(*[FadeIn(d, shift=DOWN * .3) for d in test], lag_ratio=.1), moving=False)

    def knobs(self):
        ax = self.axes()
        self.add(raised_panel((3.4, .15, 0), 5.2, 4.10))
        self.add(data_dots(ax), graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.3))
        trackers = [ValueTracker(0) for _ in range(4)]
        weights = lambda: np.array([v.get_value() for v in trackers])
        model = always_redraw(lambda: graph_curve(ax, weights()))
        self.add(model)
        formula = MathTex('y=', 'w_0', '+', 'w_1 x', '+', 'w_2 x^2', '+', 'w_3 x^3', font_size=33).move_to([0, -2.45, 0])
        for j, c in enumerate(TERM_COLORS):
            formula[1 + 2 * j].set_color(c)
        self.add(formula)
        knobs = []
        for j, tracker in enumerate(trackers):
            x = 1.2 + j * 1.45
            rail = VGroup(Line([x, -1.25, 0], [x, 1.55, 0], color=BLACK, stroke_width=7),
                          Line([x, -1.25, 0], [x, 1.55, 0], color=MUTED, stroke_width=2))
            dot = DepthDot(color=TERM_COLORS[j], radius=.11)
            dot.add_updater(lambda m, tr=tracker, xx=x: m.move_to([xx, .15 + tr.get_value() * .4, 0]))
            number = readout('', tracker.get_value, [x, -1.68, 0], TERM_COLORS[j], 1, 22)
            group = VGroup(rail, dot, tex(f'w_{j}', 28, TERM_COLORS[j]).move_to([x, 2.02, 0]), number)
            knobs.append(group)
        self.add(knobs[0])
        self.beat(trackers[0].animate.set_value(.8), Indicate(formula[1]), start_sentence=1)
        self.add(knobs[1])
        self.beat(trackers[1].animate.set_value(-1.6), Indicate(formula[3]), start_sentence=1)
        self.add(knobs[2])
        self.beat(trackers[2].animate.set_value(1.8), Indicate(formula[5]))
        self.add(knobs[3])
        self.beat(trackers[3].animate.set_value(-1.6), Indicate(formula[7]))
        self.beat(trackers[0].animate.set_value(.1), trackers[1].animate.set_value(1.5),
                  trackers[2].animate.set_value(-2.5), trackers[3].animate.set_value(.7))
        compact = tex(r'y(x,\mathbf w)=\sum_{j=0}^{M}w_jx^j', 35).move_to(formula)
        count = tex(r'M=3\quad\Rightarrow\quad 4', 32, RESIDUAL_YELLOW).move_to([3.35, -2.3, 0])
        compact.move_to([-2.7, -2.45, 0])
        duration = self.beat_cues()[-1]['end']
        self.beat(Succession(AnimationGroup(TransformMatchingTex(formula, compact), FadeIn(count), run_time=1.2),
                             Indicate(compact, scale_factor=1.015, run_time=duration-1.2)), moving=False)
        self.beat(trackers[3].animate.set_value(0))
        self.beat(trackers[0].animate.set_value(.8), trackers[1].animate.set_value(-1.4), trackers[2].animate.set_value(0))

    def squares(self):
        ax = self.axes(width=5.4)
        offset = ValueTracker(.35)
        w = lambda: WEIGHTS[1] + np.r_[offset.get_value(), np.zeros(9)]
        model = always_redraw(lambda: graph_curve(ax, w()))
        lines = always_redraw(lambda: residuals(ax, w()))
        self.add(data_dots(ax), model)
        formula = MathTex(r'r_n=', r'y(x_n,\mathbf w)', '-', 't_n', font_size=32).move_to([0, -2.47, 0])
        formula[1].set_color(MODEL_RED); formula[3].set_color(BLUE_DATA)
        self.add(formula)
        self.beat(Create(lines), Indicate(formula, scale_factor=1.015), moving=False)
        signed = VGroup(jp('符号付きの和（例）', 25), tex(r'(+1)+(-1)=0', 34, BLUE_DATA),
                        jp('ずれていても、ゼロになる', 21, MUTED)).arrange(DOWN, buff=.25).move_to([3.1, 1.1, 0])
        signed_vectors = VGroup(Arrow([1.6, .15, 0], [3.1, .15, 0], buff=0, color=BLUE_DATA),
                                Arrow([3.1, -.05, 0], [1.6, -.05, 0], buff=0, color=TEST_ORANGE))
        self.beat(FadeIn(signed), Succession(GrowArrow(signed_vectors[0]), GrowArrow(signed_vectors[1])), moving=False)
        squared = VGroup(jp('二乗和', 25), tex(r'(+1)^2+(-1)^2=2', 33, RESIDUAL_YELLOW)).arrange(DOWN, buff=.2).move_to([3.1, -1, 0])
        self.beat(FadeIn(squared), Indicate(signed_vectors), moving=False)
        self.remove(signed, squared, signed_vectors)
        title = jp('各残差の二乗（共通縮尺の面積）', 22, RESIDUAL_YELLOW).move_to([3.1, 2.1, 0])
        self.add(title)
        scale = .62  # One common area scale; independent of curve-axis zoom.
        def tiles():
            result = VGroup()
            for i, r in enumerate(eval_poly(w(), X) - T):
                square = Square(side_length=max(.008, scale * abs(r)), color=RESIDUAL_YELLOW,
                                fill_color=RESIDUAL_YELLOW, fill_opacity=.3, stroke_width=1.5)
                square.move_to([.9 + 1.05 * (i % 5), 1.12 - 1.45 * (i // 5), 0])
                result.add(relief_faces(square, RESIDUAL_YELLOW, thickness=.035))
            return result
        tile_static = tiles()
        self.beat(LaggedStart(*[TransformFromCopy(lines[i], tile_static[i]) for i in range(10)], lag_ratio=.1))
        self.remove(tile_static, *tile_static)
        tile_dynamic = always_redraw(tiles)
        self.add(tile_dynamic)
        energy = lambda: .5 * np.sum((eval_poly(w(), X) - T) ** 2)
        bar_start = np.array([.8, -1.55, 0])
        bar = always_redraw(lambda: Rectangle(width=max(.005, energy() * .63), height=.18,
                            stroke_width=0, fill_color=RESIDUAL_YELLOW, fill_opacity=.85).move_to(bar_start, aligned_edge=LEFT))
        energy_num = readout('E=', energy, [3, -1.98, 0], RESIDUAL_YELLOW)
        objective = MathTex(r'E(\mathbf w)=', r'\frac12', r'\sum_{n=1}^N', 'r_n^2', font_size=33).move_to(formula)
        objective[-1].set_color(RESIDUAL_YELLOW)
        self.add(bar, energy_num)
        collected = VGroup(*[Dot(bar_start + RIGHT * energy() * .315, radius=.04, color=RESIDUAL_YELLOW) for _ in tile_static])
        self.beat(TransformMatchingTex(formula, objective, run_time=1.2),
                  LaggedStart(*[TransformFromCopy(s, target) for s, target in zip(tile_static, collected)],
                              lag_ratio=.1, run_time=self.beat_cues()[-1]['end']))
        self.remove(collected, *collected)
        self.beat(offset.animate.set_value(.7))
        self.beat(offset.animate.set_value(-.6))
        self.beat(Indicate(objective[2]), Indicate(objective[1]), moving=False)
        self.beat(offset.animate.set_value(0))

    def valley(self):
        ax = self.axes(width=5.3)
        alpha = ValueTracker(0)
        best = WEIGHTS[1][:2]
        initial = np.array([-.8, .5])
        weights = lambda: (1 - alpha.get_value()) * initial + alpha.get_value() * best
        self.add(data_dots(ax))
        line = always_redraw(lambda: graph_curve(ax, weights()))
        errors = always_redraw(lambda: residuals(ax, weights()))
        self.add(line, errors)
        landscape = LossLandscape()
        landscape.remove(landscape.contours)
        self.add(landscape)
        w0_label = tex('w_0', 25, BLUE_DATA)
        w1_label = tex('w_1', 25, REG_PURPLE)
        w0_label.add_updater(lambda m: m.next_to(landscape.axis_end(0), RIGHT, buff=.1))
        w1_label.add_updater(lambda m: m.next_to(landscape.axis_end(1), UP, buff=.1))
        dot = DepthDot(landscape.point(*weights()), radius=.105, color=RESIDUAL_YELLOW)
        dot.add_updater(lambda m: m.move_to(landscape.point(*weights())))
        bottom = DepthDot(landscape.point(*best), radius=.06, color=TRUE_GREEN)
        bottom.add_updater(lambda m: m.move_to(landscape.point(*best)))
        self.add(dot, w0_label, w1_label)
        self.beat(Indicate(dot), Indicate(line), moving=False)
        height_note = VGroup(jp('高さが誤差', 23, RESIDUAL_YELLOW), tex('E', 28, RESIDUAL_YELLOW))
        height_note.arrange(RIGHT, buff=.15).move_to([3.35, 2.35, 0])
        cues = self.beat_cues()
        reveal_start = cues[2]['start']
        reveal_duration = cues[-1]['end'] - reveal_start
        self.beat(phases=[
            ('same-error contours', reveal_start,
             lambda: AnimationGroup(Create(landscape.contours), FadeIn(bottom))),
            ('contour map rises into error surface', reveal_duration,
             lambda: AnimationGroup(landscape.lift.animate.set_value(1),
                                    landscape.elevation.animate.set_value(landscape.FINAL_ELEVATION),
                                    landscape.azimuth.animate.set_value(landscape.FINAL_AZIMUTH),
                                    landscape.surface_opacity.animate.set_value(1), FadeIn(height_note))),
        ])
        # Keep foreground markers above the inset, whose faces sort internally.
        self.bring_to_front(dot, bottom, w0_label, w1_label)
        projection = always_redraw(lambda: DashedLine(
            landscape.point(*weights(), energy=0), landscape.point(*weights()),
            color=RESIDUAL_YELLOW, stroke_width=1.7, dash_length=.07))
        footprint = DepthDot(radius=.05, color=RESIDUAL_YELLOW)
        footprint.add_updater(lambda m: m.move_to(landscape.point(*weights(), energy=0)))
        path = always_redraw(lambda: polyline([
            landscape.point(*((1-a)*initial + a*best))
            for a in np.linspace(0, max(alpha.get_value(), 1e-6), 49)],
            RESIDUAL_YELLOW, 2.6))
        self.add(projection, footprint, path)
        self.bring_to_front(dot, bottom)
        energy = readout('E=', lambda: float(linear_fit_error(*weights())), [3.3, -2.35, 0], RESIDUAL_YELLOW)
        formula = tex(r'y=w_0+w_1x', 32).move_to([-3, -2.35, 0])
        # Values and the fitted line share exactly the same coefficient tracker.
        w0 = readout('w_0=', lambda: weights()[0], [1.75, -1.90, 0], BLUE_DATA, 2, 21)
        w1 = readout('w_1=', lambda: weights()[1], [4.70, -1.90, 0], REG_PURPLE, 2, 21)
        self.add(energy, formula, w0, w1)
        self.beat(alpha.animate.set_value(.45))
        self.beat(alpha.animate.set_value(.85))
        arrival = self.sentence_duration(0)
        remainder = self.beat_cues()[-1]['end'] - arrival
        self.beat(phases=[('arrive', arrival, lambda: alpha.animate.set_value(1)),
                          ('explain fit', remainder, lambda: AnimationGroup(Indicate(line, scale_factor=1.015), Indicate(energy)))])
        truth = graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.55)
        self.beat(Create(truth), Indicate(line))

    def aid_card(self, heading):
        """Temporarily replace the body, retaining the title and caption clock."""
        retained = [*self.mobjects[:2], self.subtitle]
        body = [m for m in self.mobjects if m not in retained]
        self.remove(*body)
        frame = RoundedRectangle(width=10.5, height=4.55, corner_radius=.12,
                                 stroke_color=MUTED, stroke_width=1.2,
                                 fill_color=BG, fill_opacity=1).move_to([0, .05, 0])
        heading = jp(heading, 25).move_to([0, 1.92, 0])
        return body, VGroup(frame, heading)

    def restore_aid_body(self, body):
        retained = [*self.mobjects[:2], self.subtitle]
        self.remove(*[m for m in self.mobjects if m not in retained])
        self.add(*body)

    def rms_aid(self):
        body, card = self.aid_card('補足：RMSを作る3段階（2点の例）')
        stages = VGroup(*[jp(label, 25, AID_OPERATION).move_to([x, 1.25, 0])
                          for label, x in [('二乗', -3.2), ('平均', 0), ('平方根', 3.2)]])
        arrows = VGroup(Arrow([-3.85, -.3, 0], [-3.85, .6, 0], buff=0, color=AID_INPUT),
                        Arrow([-2.55, .6, 0], [-2.55, -.3, 0], buff=0, color=AID_INPUT))
        inputs = VGroup(tex('+2', 30, AID_INPUT).move_to([-3.85, .86, 0]),
                        tex('-2', 30, AID_INPUT).move_to([-2.55, .86, 0]))
        squares = VGroup(*[Square(side_length=.9, color=AID_OPERATION,
                                  fill_opacity=.25).move_to([x, -.05, 0])
                           for x in [-3.85, -2.55]])
        areas = VGroup(*[tex('4', 32, AID_OPERATION).move_to(sq) for sq in squares])
        total = Rectangle(width=1.8, height=.9, color=AID_OPERATION,
                          fill_opacity=.25).move_to([0, -.05, 0])
        mean = Square(side_length=.9, color=AID_OPERATION,
                      fill_opacity=.25).move_to(total)
        sum_formula = tex('4+4=8', 30, AID_OPERATION).move_to([0, -.96, 0])
        mean_formula = tex(r'\frac{4+4}{2}=4', 30, AID_OPERATION).move_to([0, -1.18, 0])
        result = Line([2.75, -.05, 0], [3.65, -.05, 0], color=AID_RESULT, stroke_width=6)
        result_formula = tex(r'\sqrt{4}=2', 34, AID_RESULT).move_to([3.2, -1.05, 0])
        unit = jp('元の値と同じ単位', 23, AID_RESULT).move_to([3.2, -1.75, 0])
        cues = self.beat_cues()
        a, b, c = [q['end'] - q['start'] for q in cues]
        self.beat(phases=[
            ('RMS example enter', .6, lambda: FadeIn(card)),
            ('signed residuals +2 -2', a * .70 - .6,
             lambda: AnimationGroup(FadeIn(stages), FadeIn(inputs), Create(arrows))),
            ('square residual lengths into areas 4 4', a * .30,
             lambda: AnimationGroup(ReplacementTransform(arrows, squares), FadeIn(areas))),
            ('sum areas to 8', b * .5,
             lambda: AnimationGroup(TransformFromCopy(squares, total), FadeIn(sum_formula))),
            ('divide by two to mean area 4', b * .5,
             lambda: AnimationGroup(ReplacementTransform(total, mean), ReplacementTransform(sum_formula, mean_formula))),
            ('square root to length 2', c * .48,
             lambda: AnimationGroup(TransformFromCopy(mean, result), FadeIn(result_formula))),
            ('original units', c * .52, lambda: FadeIn(unit)),
        ])
        self.restore_aid_body(body)

    def log_scale_aid(self):
        body, card = self.aid_card('補足：大きさを圧縮し、符号を残す（説明用の例）')
        definition = tex(r'\operatorname{sgn}(w)\,\log_{10}(1+|w|)', 30,
                         AID_OPERATION).move_to([0, 1.2, 0])
        axis = NumberLine(x_range=[-3, 3, 1], length=7.2,
                          include_numbers=True, font_size=24, color=MUTED).move_to([0, -.75, 0])
        inputs = VGroup(*[tex(str(v), 32, AID_INPUT).move_to(axis.n2p(i) + UP * 1.35)
                          for i, v in enumerate([9, 99, 999], 1)])
        targets = VGroup(*[Dot(axis.n2p(i), color=AID_OPERATION, radius=.075) for i in [1, 2, 3]])
        links = VGroup(*[Arrow(label.get_bottom(), target.get_top(), buff=.12,
                              color=AID_OPERATION, stroke_width=2) for label, target in zip(inputs, targets)])
        neg = tex('-99', 32, AID_INPUT).move_to(axis.n2p(-2) + UP * 1.35)
        neg_link = Arrow(neg.get_bottom(), axis.n2p(-2) + UP * .08,
                         buff=.12, color=AID_OPERATION, stroke_width=2)
        neg_dot = Dot(axis.n2p(-2), color=AID_OPERATION, radius=.075)
        # Equal lengths in opposite directions preserve the sign visibly.
        positive = Arrow([-4.25, .05, 0], [-4.25, .95, 0], buff=0, color=AID_RESULT)
        negative = Arrow([-4.25, .05, 0], [-4.25, -.85, 0], buff=0, color=AID_RESULT)
        pos_label = tex('+2', 26, AID_RESULT).next_to(positive, LEFT, buff=.1)
        neg_label = tex('-2', 26, AID_RESULT).next_to(negative, LEFT, buff=.1)
        note = jp('棒の長さは同じ・向きは逆', 25, AID_RESULT).move_to([0, -1.8, 0])
        a, b, c = [q['end'] - q['start'] for q in self.beat_cues()]
        self.beat(phases=[
            ('log example enter', .6, lambda: FadeIn(card)),
            ('coefficient inputs 9 99 999', a * .70 - .6,
             lambda: AnimationGroup(FadeIn(definition), FadeIn(axis), FadeIn(inputs))),
            ('map to equal ticks 1 2 3', a * .30,
             lambda: LaggedStart(*[AnimationGroup(GrowArrow(link), FadeIn(target))
                                  for link, target in zip(links, targets)], lag_ratio=.25)),
            ('compress magnitude retain direction', b,
             lambda: AnimationGroup(Indicate(definition, scale_factor=1.02), GrowArrow(positive), FadeIn(pos_label))),
            ('negative coefficient -99 to -2', c * .70,
             lambda: AnimationGroup(FadeIn(neg), GrowArrow(neg_link), FadeIn(neg_dot))),
            ('negative bar points down with length 2', c * .30,
             lambda: AnimationGroup(GrowArrow(negative), FadeIn(neg_label), FadeIn(note))),
        ])
        self.restore_aid_body(body)

    def degrees(self):
        tracker = ValueTracker(0)
        w = lambda: degree_weights(tracker.get_value())
        ax = self.axes(width=5.5, span=lambda: self.curve_span(w()))
        self.add(data_dots(ax), always_redraw(lambda: graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.35)))
        model = always_redraw(lambda: graph_curve(ax, w()))
        def live_residuals(xs, ts, color):
            group = VGroup()
            for a, b in zip(xs, ts):
                line = Line(ax.c2p(a, b), ax.c2p(a, float(eval_poly(w(), a))) + UP * 1e-8,
                            color=color, stroke_width=2)
                line.add_updater(lambda m, a=a, b=b: m.put_start_and_end_on(
                    ax.c2p(a, b), ax.c2p(a, float(eval_poly(w(), a))) + UP * 1e-8))
                group.add(line)
            return group
        lines = live_residuals(X, T, RESIDUAL_YELLOW)
        test_dots = data_dots(ax, XT[::10], TT[::10], TEST_ORANGE, .033)
        test_lines = live_residuals(XT[::10], TT[::10], TEST_ORANGE).set_opacity(.6)
        errax = Axes(x_range=[0, 9, 1], y_range=[0, 1.1, .25], x_length=4.65, y_length=3.0,
                     tips=False, axis_config={'color': MUTED, 'stroke_width': 1.5}).move_to([3.45, .1, 0])
        self.add(raised_panel((3.45, .1, 0), 5.40, 4.35),
                 errax, tex('M', 23).next_to(errax.c2p(9, 0), RIGHT, buff=.13),
                 jp('RMS', 23).move_to([1.25, 2.02, 0]))
        for v in [0, .5, 1.0]:
            self.add(tex(str(v), 18, MUTED).next_to(errax.c2p(0, v), LEFT, buff=.12))
        for v in [0, 3, 6, 9]:
            self.add(tex(str(v), 18, MUTED).next_to(errax.c2p(v, 0), DOWN, buff=.12))
        self.add(jp('訓練', 21, BLUE_DATA).move_to([3.4, 2.1, 0]), jp('テスト', 21, TEST_ORANGE).move_to([5, 2.1, 0]))
        self.add(self.slider(tracker, 0, 9, [-3.1, -2.35, 0], width=4.6), self.degree_label(tracker, [-3.1, 2.12, 0]))
        self.add(model, lines)
        preview_lines = residuals(ax, w(), XT[::10], TT[::10], TEST_ORANGE).set_opacity(.6)
        self.beat(FadeIn(test_dots), Create(preview_lines), moving=False)
        self.remove(preview_lines)
        self.add(test_lines)
        rms_formula = tex(r'E_{\rm RMS}=\sqrt{\frac1N\sum_n r_n^2}', 27).move_to([3.4, -2.45, 0])
        self.add(rms_formula)
        counter = ValueTracker(0)
        count_label = readout(r'n=', counter.get_value, [3.6, 1.75, 0], RESIDUAL_YELLOW, 0, 20)
        self.add(count_label)
        train_pts = [Dot(errax.c2p(m, TRAIN_RMS[m]), color=BLUE_DATA, radius=.05) for m in range(10)]
        test_pts = [Dot(errax.c2p(m, TEST_RMS[m]), color=TEST_ORANGE, radius=.05) for m in range(10)]
        # One growing path per set; no duplicate completed lines.
        train_path = VMobject(color=BLUE_DATA, stroke_width=2.5)
        test_path = VMobject(color=TEST_ORANGE, stroke_width=2.5)
        self.add(train_path, test_path)
        def count_residuals(seconds):
            counter.set_value(0)
            return AnimationGroup(
                Succession(*[ShowPassingFlash(lines[i].copy().clear_updaters().set_stroke(RESIDUAL_YELLOW, 4), time_width=.8) for i in range(10)], run_time=seconds),
                counter.animate(run_time=seconds, rate_func=linear).set_value(10))
        duration = self.beat_cues()[-1]['end']
        self.beat(phases=[
            ('count residuals', duration * .64, lambda: count_residuals(duration * .64)),
            ('RMS stamp', duration * .36, lambda: AnimationGroup(TransformFromCopy(lines.copy().clear_updaters(), train_pts[0]),
                                                              TransformFromCopy(test_lines.copy().clear_updaters(), test_pts[0])))])
        self.rms_aid()
        for m in range(1, 10):
            counter.set_value(0)
            # Use callbacks only at the fitted endpoint; RMS markers are never interpolated observations.
            cues = self.beat_cues()
            morph_seconds = cues[0]['end']
            count_seconds = cues[1]['end'] - cues[1]['start']
            def add_paths(m=m):
                train_path.set_points_as_corners([p.get_center() for p in train_pts[:m + 1]])
                test_path.set_points_as_corners([p.get_center() for p in test_pts[:m + 1]])
            # Each factory runs after the preceding phase, using the current fitted residuals.
            phases = [('M slider', morph_seconds, lambda m=m: tracker.animate.set_value(m)),
                      ('count residuals', count_seconds * .64, lambda: count_residuals(count_seconds * .64)),
                      ('RMS stamp', count_seconds * .36,
                       lambda m=m: AnimationGroup(TransformFromCopy(lines.copy().clear_updaters(), train_pts[m]),
                                                   TransformFromCopy(test_lines.copy().clear_updaters(), test_pts[m])))]
            remaining = cues[-1]['end'] - morph_seconds - count_seconds
            if remaining > .01:
                phases.append(('explain RMS', remaining, lambda: Indicate(rms_formula, scale_factor=1.015)))
            self.beat(phases=phases)
            add_paths()
        highlight = Circle(radius=.23, color=RESIDUAL_YELLOW).move_to(ax.c2p(.05, eval_poly(WEIGHTS[9], .05)))
        self.beat(Create(highlight), Indicate(test_pts[9]), moving=False)

    def coefficient_axes(self, center=(3.3, .1, 0), width=4.8, height=3.25):
        ax = Axes(x_range=[-.5, 9.5, 1], y_range=[-6.5, 6.5, 2], x_length=width, y_length=height,
                  tips=False, axis_config={'color': MUTED, 'stroke_width': 1.2}).move_to(center)
        self.add(raised_panel(np.array(center) + DOWN * .16, width + .85, height + .55), ax)
        for i in range(10):
            self.add(tex(str(i), 18, MUTED).move_to(ax.c2p(i, -6.5) + DOWN * .17))
        for y in [-6, -3, 3, 6]:
            self.add(tex(str(y), 18, MUTED).next_to(ax.c2p(-.5, y), LEFT, buff=.12))
        return ax

    def coefficient_bars(self, ax, weights, color=REG_PURPLE):
        bars = VGroup()
        for j, value in enumerate(weights):
            height = np.sign(value) * np.log10(1 + abs(value))
            a, b = ax.c2p(j, 0), ax.c2p(j, height)
            front = Rectangle(width=.22, height=max(.006, abs(b[1] - a[1])),
                              stroke_width=0, fill_color=color, fill_opacity=.95)
            bars.add(relief_faces(front.move_to((a + b) / 2), color, thickness=.03))
        return bars

    def coefficient_scale_label(self, position):
        # A single TeX expression with explicit delimiters avoids split-glyph artifacts.
        return tex(r'\operatorname{sgn}(w_j)\,\log_{10}\!\left(1+\lvert w_j\rvert\right)',
                   28, REG_PURPLE).move_to(position)

    def coefficients(self):
        tracker = ValueTracker(3)
        w = lambda: degree_weights(tracker.get_value())
        ax = self.axes(width=5.4, span=lambda: self.curve_span(w()))
        self.add(data_dots(ax), always_redraw(lambda: graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.3)))
        self.add(always_redraw(lambda: graph_curve(ax, w())))
        barsax = self.coefficient_axes()
        bars = always_redraw(lambda: self.coefficient_bars(barsax, w()))
        self.add(bars, self.degree_label(tracker, [-3, -2.35, 0]))
        scale_formula = self.coefficient_scale_label([3.3, 2.15, 0])
        maximum = readout(r'\log_{10}\!\left(1+\max_j\lvert w_j\rvert\right)=', lambda: np.log10(1 + np.max(abs(w()))), [3.3, -2.35, 0], REG_PURPLE, 2, 25)
        self.add(maximum)
        self.beat(Create(bars), moving=False)
        self.add(scale_formula)
        self.beat(Indicate(scale_formula, scale_factor=1.02), moving=False)
        self.log_scale_aid()
        self.beat(tracker.animate.set_value(6))
        self.beat(tracker.animate.set_value(9))
        self.beat(tracker.animate.set_value(3))
        self.beat(tracker.animate.set_value(9))

    def more_data(self):
        n = ValueTracker(10)
        ax = self.axes(center=(0, .1, 0), width=9, height=3.65,
                       span=lambda: self.curve_span(growing_weights(n.get_value()), minimum=1.7))
        curve = always_redraw(lambda: graph_curve(ax, growing_weights(n.get_value())))
        self.add(always_redraw(lambda: graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.4)), curve)
        dots = data_dots(ax, X_ALL, T_ALL, radius=.034)
        # Existing points stay fixed. Each new point falls as its data weight is introduced.
        for i, dot in enumerate(dots):
            dot.clear_updaters()
            def update(m, i=i):
                target = ax.c2p(X_ALL[i], T_ALL[i])
                a = np.clip(n.get_value() - i, 0, 1) if i >= 10 else 1
                m.move_to(target + UP * .45 * (1 - a)).set_opacity(a)
            dot.add_updater(update)
        self.add(dots)
        count = readout('N=', lambda: np.floor(n.get_value() + 1e-6), [-1.1, -2.45, 0], BLUE_DATA, 0, 32)
        self.add(count, tex('M=9', 32, MODEL_RED).move_to([1.5, -2.45, 0]))
        self.beat(Indicate(dots[:10], scale_factor=1.08), moving=False)
        self.beat(n.animate.set_value(15))
        self.beat(Indicate(dots[10:15], scale_factor=1.4))
        self.beat(n.animate.set_value(40))
        arrival = self.sentence_duration(0)
        remainder = self.beat_cues()[-1]['end'] - arrival
        self.beat(phases=[('N to 100', arrival, lambda: n.animate.set_value(100)),
                          ('explain fit', remainder, lambda: Indicate(curve, color=TRUE_GREEN, scale_factor=1.015))])
        self.beat(Indicate(curve, color=TRUE_GREEN, scale_factor=1.02))

    def regularization(self):
        loglam = ValueTracker(-32)
        w = lambda: ridge_weights(float(loglam.get_value()))
        ax = self.axes(center=(-3.15, .45, 0), width=5.45, height=3.1,
                       span=lambda: self.curve_span(w()))
        self.add(data_dots(ax), always_redraw(lambda: graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.3)))
        curve = always_redraw(lambda: graph_curve(ax, w()))
        lines = always_redraw(lambda: residuals(ax, w()).set_opacity(.55))
        self.add(curve, lines)
        barsax = self.coefficient_axes(center=(3.3, 1.25, 0), height=1.55, width=4.7)
        bars = always_redraw(lambda: self.coefficient_bars(barsax, w()))
        self.add(bars)
        self.add(self.coefficient_scale_label([3.3, 2.35, 0]))
        def springs():
            # One representative spring, beside w6: two coils leave the bar visible.
            j = 6
            value = w()[j]
            h = np.sign(value) * np.log10(1 + abs(value))
            a, b = barsax.c2p(j, 0) + RIGHT * .21, barsax.c2p(j, h) + RIGHT * .21
            pts = [a + (b - a) * u + RIGHT * .045 * np.sin(4 * PI * u)
                   for u in np.linspace(0, 1, 25)]
            return VGroup(polyline(pts, RESIDUAL_YELLOW, 1.4, .85),
                          Dot(a, radius=.022, color=RESIDUAL_YELLOW),
                          Dot(b, radius=.022, color=RESIDUAL_YELLOW))
        rubber = always_redraw(springs)
        self.beat(FadeIn(rubber), moving=False)
        formula = MathTex(r'\widetilde E=', r'\frac12\sum_n(y(x_n,\mathbf w)-t_n)^2', '+', r'\frac{\lambda}{2}\sum_{j=0}^{9}w_j^2', font_size=30).move_to([0, -2.55, 0])
        formula[1].set_color(RESIDUAL_YELLOW); formula[3].set_color(REG_PURPLE)
        self.add(formula)
        self.beat(Succession(
            AnimationGroup(Indicate(formula[1], scale_factor=1.015), Indicate(lines), run_time=self.sentence_duration(0)),
            AnimationGroup(Indicate(formula[3], scale_factor=1.015), Indicate(bars), run_time=self.sentence_duration(1)),
            Indicate(formula[0], scale_factor=1.015, run_time=self.sentence_duration(2))), moving=False)
        slider = self.slider(loglam, -32, 0, [-3, -1.9, 0], width=4.6, label=r'\ln\lambda', ticks=[-32, -24, -16, -8, 0], color=REG_PURPLE)
        self.add(slider, readout(r'\ln\lambda=', loglam.get_value, [-3.1, 2.25, 0], REG_PURPLE, 1, 25))
        err = Axes(x_range=[0, 32, 8], y_range=[0, 1.1, .5], x_length=4.65, y_length=1.32,
                   tips=False, axis_config={'color': MUTED, 'stroke_width': 1.2}).move_to([3.3, -1.25, 0])
        values = np.linspace(-32, 0, 161)
        training = [rms_error(ridge_weights(float(l)), X, T) for l in values]
        testing = [rms_error(ridge_weights(float(l)), XT, TT) for l in values]
        self.add(err, jp('RMS', 19).move_to([1.15, -.23, 0]),
                 jp('訓練', 19, BLUE_DATA).move_to([2.8, -.23, 0]),
                 jp('テスト', 19, TEST_ORANGE).move_to([4.4, -.23, 0]))
        for v in [-32, -16, 0]:
            self.add(tex(str(v), 16, MUTED).next_to(err.c2p(v + 32, 0), DOWN, buff=.08))
        for v in [0, 1]:
            self.add(tex(str(v), 16, MUTED).next_to(err.c2p(0, v), LEFT, buff=.08))
        self.add(tex(r'\ln\lambda', 20).next_to(err.c2p(32, 0), RIGHT, buff=.12))
        train_path = polyline([err.c2p(l + 32, e) for l, e in zip(values, training)], BLUE_DATA, 2)
        test_path = polyline([err.c2p(l + 32, e) for l, e in zip(values, testing)], TEST_ORANGE, 2)
        markers = always_redraw(lambda: VGroup(Dot(err.c2p(loglam.get_value() + 32, rms_error(w(), X, T)), radius=.055, color=BLUE_DATA),
                                              Dot(err.c2p(loglam.get_value() + 32, rms_error(w(), XT, TT)), radius=.055, color=TEST_ORANGE)))
        self.add(train_path, test_path, markers)
        self.beat(Indicate(formula[3]), Indicate(rubber), moving=False)
        self.beat(loglam.animate.set_value(-18))
        self.beat(loglam.animate.set_value(0))
        self.beat(loglam.animate.set_value(-32))
        self.beat(loglam.animate.set_value(-18))
        self.beat(loglam.animate.set_value(-8))
        self.beat(loglam.animate.set_value(-18))

    def uncertainty(self):
        ax = self.axes(center=(0, .1, 0), width=9, height=3.6)
        observations = data_dots(ax)
        self.add(observations, graph_curve(ax, ridge_weights(-18)))
        truth = graph_curve(ax, values=sine, color=TRUE_GREEN, opacity=.65)
        self.beat(Create(truth), moving=False)
        repeated = np.random.default_rng(44).normal(float(sine(.65)), NOISE_STD, 18)
        newdots = data_dots(ax, np.full(18, .65), repeated, TEST_ORANGE, .04)
        self.beat(LaggedStart(*[FadeIn(d, shift=DOWN * .15) for d in newdots], lag_ratio=.2))
        u = np.linspace(0, 1, 160)
        vertices = [ax.c2p(x, sine(x) + 2 * NOISE_STD) for x in u] + [ax.c2p(x, sine(x) - 2 * NOISE_STD) for x in u[::-1]]
        band = Polygon(*vertices, stroke_width=0, fill_color=TRUE_GREEN, fill_opacity=.15).set_z_index(-1)
        label = tex(r'\sigma=0.25\qquad \sin(2\pi x)\pm 2\sigma', 31, TRUE_GREEN).move_to([0, -2.45, 0])
        self.add(label)
        self.beat(FadeIn(band), Indicate(label, scale_factor=1.015), moving=False)
        self.beat(Indicate(newdots, scale_factor=1.12), Indicate(observations, scale_factor=1.03))
        question = jp('どんな値が、どれくらいありそうか？', 34).move_to([0, -2.45, 0])
        self.beat(ReplacementTransform(label, question), moving=False)


class PRML11DepthPreview(PRML11PolynomialCurveFitting):
    """Render just the narrated map-to-surface experiment for quick iteration."""
    scene_indices = [3]
    timeline_name = 'prml11_depth_timeline.json'
