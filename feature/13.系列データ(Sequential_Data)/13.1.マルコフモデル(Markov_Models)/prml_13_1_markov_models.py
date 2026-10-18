"""PRML 13.1: visual story of memory in observed sequential data."""
import json
from pathlib import Path

import numpy as np
from manim import *

from narration_content import SCENES
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST
from markov_model import filter_history, predict_rain, parameters

BG = '#111823'
BLUE = '#58C4DD'
YELLOW = '#FFFF00'
GREEN = '#83C167'
PURPLE = '#9A72AC'
RED = '#FC6255'
MUTED = '#A8B3C0'


def label(value, x, y, size=22, color=WHITE):
    return jp(value, size, color).move_to([x, y, 0])


def mathline(value, y=-2.40, size=27):
    mob = tex(value, size, YELLOW).move_to([0, y, 0])
    if mob.width > 11.8:
        mob.scale(11.8 / mob.width)
    return mob


def weather_chain(history, y, show_future=True):
    group = VGroup()
    group.add(label('昨日', -5.3, y+.65, 16, MUTED),
              label('今日', -3.7, y+.65, 16, MUTED))
    for i, wet in enumerate(history):
        x = -5.3 + 1.6*i
        color = BLUE if wet else YELLOW
        dot = Circle(radius=.28, stroke_color=color, stroke_width=2.1,
                     fill_color=color, fill_opacity=.18).move_to([x, y, 0])
        group.add(dot, label('雨' if wet else '晴', x, y, 21, color))
        if i:
            group.add(Arrow([x-1.23, y, 0], [x-.39, y, 0],
                            buff=0, stroke_width=2, color=MUTED, tip_length=.12))
    if show_future:
        group.add(Arrow([-3.30, y, 0], [-2.46, y, 0], buff=0,
                        stroke_width=2, color=MUTED, tip_length=.12),
                  Circle(radius=.28, stroke_color=MUTED, stroke_width=1.5).move_to([-2.05, y, 0]),
                  label('？', -2.05, y, 21, MUTED))
    return group


def forecast_bars(values, heading='明日の雨', accent=GREEN):
    group = VGroup(label(heading, 3.9, 2.10, 25))
    x0, length = 1.75, 3.85
    group.add(Line([x0, -1.70, 0], [x0+length, -1.70, 0], color=MUTED, stroke_width=1.5))
    for tick in (0, .5, 1):
        x = x0 + length*tick
        group.add(Line([x, -1.70, 0], [x, -1.56, 0], color=MUTED, stroke_width=1.3),
                  label(f'{int(tick*100)}', x, -1.97, 14, MUTED))
    group.add(label('確率（%）', 3.68, -2.35, 16, MUTED))
    focus = None
    for i, (name, value) in enumerate(zip(('晴→雨', '雨→雨'), values)):
        y = .70 - 1.12*i
        color = accent if i else BLUE
        group.add(label(name, 1.02, y, 17, color))
        bar = Line([x0, y, 0], [x0+length*value, y, 0],
                   stroke_width=20, color=color)
        group.add(bar, label(f'{value*100:.1f}%', min(6.18, x0+length*value+.55), y, 18, color))
        if i == 1:
            focus = bar
    return group, focus


def opening_panel(bi, ending=False):
    first = weather_chain([0, 1], .80)
    second = weather_chain([1, 1], -.75)
    group = VGroup(first, second)
    if ending or bi >= 3:
        values = [predict_rain([0, 1]), predict_rain([1, 1])]
    else:
        values = [.5, .5]
    bars, focus = forecast_bars(values)
    group.add(bars)
    if ending:
        group.add(label('同じ条件の固定モデル', -3.68, -2.31, 17, MUTED))
    elif bi < 3:
        group.add(label('予想してから確かめる', -3.68, -2.31, 17, YELLOW))
    else:
        group.add(label('説明用の定常モデル', -3.68, -2.31, 17, MUTED))
    if ending and bi in (1, 2):
        x = -5.3 if bi == 1 else -3.7
        rings = VGroup(*[Circle(radius=.40, color=YELLOW,
                                stroke_width=2).move_to([x, y, 0])
                         for y in (.80, -.75)])
        group.add(rings)
        return group, rings
    return group, VGroup(first, second) if bi < 3 else focus


def observation_graph(order=0):
    group = VGroup()
    xs = [-5.7, -4.15, -2.60, -1.05]
    for i, x in enumerate(xs):
        group.add(Circle(radius=.25, stroke_color=BLUE, stroke_width=2,
                         fill_color=BLUE, fill_opacity=.18).move_to([x, .30, 0]),
                  tex(f'x_{{{i+1}}}', 21, BLUE).move_to([x, -.31, 0]))
        if i and order >= 1:
            group.add(Arrow([xs[i-1]+.34, .30, 0], [x-.34, .30, 0],
                            buff=0, color=YELLOW, stroke_width=2, tip_length=.12))
        if i >= 2 and order >= 2:
            group.add(CurvedArrow([xs[i-2], .64, 0], [x-.10, .64, 0],
                                  angle=-.55, color=PURPLE, stroke_width=2,
                                  tip_length=.11))
    return group


def recap_condition():
    group = VGroup()
    heading = label('復習: 1.2 条件付き確率', -3.53, 2.12, 20)
    group.add(heading, SurroundingRectangle(heading, color=YELLOW,
                                             stroke_width=1.4, buff=.11))
    group.add(label('今日', -5.45, 1.16, 19, BLUE),
              label('明日', -2.44, 1.16, 19, GREEN),
              Arrow([-4.8, 1.15, 0], [-3.09, 1.15, 0], buff=0,
                    color=YELLOW, stroke_width=3, tip_length=.17),
              label('雨', -5.45, .44, 24, BLUE),
              label('雨', -2.44, .44, 24, GREEN),
              tex('A', 24, BLUE).move_to([1.53, 1.05, 0]),
              label('今日が雨', 3.15, 1.05, 20, BLUE),
              tex('B', 24, GREEN).move_to([1.53, .30, 0]),
              label('明日が雨', 3.15, .30, 20, GREEN),
              mathline(r'p(A,B)=p(A)p(B\mid A)', -1.17, 28))
    return group


def recap_mixture():
    """Recreate the two coloured component curves and green mixture of 9.2."""
    group = VGroup()
    heading = label('復習: 9.2 隠れた成分', -4.60, 2.18, 20)
    group.add(heading, SurroundingRectangle(heading, color=YELLOW,
                                             stroke_width=1.4, buff=.11))
    axes = Axes(x_range=[0, 6, 1], y_range=[0, .62, .2],
                x_length=8.2, y_length=2.75, tips=False,
                axis_config={'stroke_color': MUTED, 'stroke_width': 1.5,
                             'include_numbers': False}).move_to([0, -.28, 0])
    group.add(axes)
    for tick in (-2, 0, 2):
        point = axes.c2p(tick+3, 0)
        group.add(Line(point, point+DOWN*.13, color=MUTED,
                       stroke_width=1),
                  label(str(tick), point[0], point[1]-.31, 14, MUTED))
    group.add(label('観測値', 4.96, axes.c2p(0, 0)[1]-.02, 16, MUTED),
              label('確率密度', -5.20, 1.52, 16, MUTED))
    for height in (0, .3, .6):
        point = axes.c2p(0, height)
        group.add(label(f'{height:.1f}', point[0]-.40, point[1], 13, MUTED))
    def gaussian(u, mean):
        return .5*np.exp(-.5*((u-mean)/.72)**2)/(.72*np.sqrt(2*np.pi))
    red = axes.plot(lambda u: gaussian(u-3, -1.12), x_range=[0, 6],
                    color=RED, stroke_width=3)
    blue = axes.plot(lambda u: gaussian(u-3, 1.12), x_range=[0, 6],
                     color=BLUE, stroke_width=3)
    mixture = axes.plot(lambda u: gaussian(u-3, -1.12)+gaussian(u-3, 1.12),
                        x_range=[0, 6], color=GREEN, stroke_width=4)
    group.add(red, blue, mixture,
              label('成分1', -3.05, 1.10, 17, RED),
              label('成分2', 3.08, 1.10, 17, BLUE),
              label('重ねた分布', 0, -2.33, 18, GREEN))
    return group, mixture


def chain_panel(bi, order=1):
    group = VGroup(observation_graph(order))
    if order == 1:
        vals = [.644, .644]
        formula = r'p(x_1,\ldots,x_N)=p(x_1)\prod_{n=2}^{N}p(x_n\mid x_{n-1})'
        if bi == 4:
            formula = r'p(x_{n+1}\mid x_1,\ldots,x_n)=p(x_{n+1}\mid x_n)'
    else:
        vals = ([predict_rain([0, 1, 1]), predict_rain([1, 1, 1])]
                if bi == 4 else [predict_rain([0, 1]), predict_rain([1, 1])])
        formula = (r'p(x_1)p(x_2\mid x_1)\prod_{n=3}^{N}p(x_n\mid x_{n-1},x_{n-2})'
                   if bi == 2 else r'p(x_n\mid x_{n-1},x_{n-2})')
    bars, focus = forecast_bars(vals, '直前からの予測', GREEN)
    group.add(bars)
    if bi in (2, 3, 4):
        group.add(mathline(formula))
    else:
        group.add(label('一次: 直前の1日' if order == 1 else '二次: 直前の2日',
                        -3.42, -2.35, 19, MUTED))
    return group, focus


def parameter_panel(bi):
    m = [1, 1, 2, 3, 3, 5, 5][bi]
    count = parameters(2, m)
    group = VGroup(label('記憶長', -4.95, 2.20, 20, MUTED),
                   tex(f'M={m}', 30, YELLOW).move_to([-3.60, 2.20, 0]),
                   label('条件表の行', -3.90, 1.50, 20))
    group.add(Line([-2.55, 2.20, 0], [.55, 2.20, 0], color=MUTED,
                   stroke_width=2))
    for tick in range(1, 6):
        x = -2.55 + (tick-1)*3.10/4
        group.add(Line([x, 2.06, 0], [x, 2.34, 0], color=MUTED,
                       stroke_width=1.4),
                  label(str(tick), x, 1.83, 14, MUTED))
    group.add(Dot([-2.55+(m-1)*3.10/4, 2.20, 0], radius=.105,
                  color=YELLOW))
    shown = min(2**m, 8)
    focus = None
    for j in range(shown):
        x = -5.75 + (j % 4)*1.15
        y = .75 - (j // 4)*.78
        square = Square(side_length=.56, stroke_color=BLUE,
                        fill_color=BLUE, fill_opacity=.17,
                        stroke_width=1.6).move_to([x, y, 0])
        group.add(square, label(f'{j:0{min(m,3)}b}', x, y, 15, BLUE))
        if j == shown-1:
            focus = square
    if 2**m > shown:
        group.add(label(f'ほか {2**m-shown} 行', -3.95, -1.26, 17, MUTED))
    group.add(label('自由な数', 2.0, 1.45, 20),
              Line([1.20, .56, 0], [5.98, .56, 0], color=MUTED, stroke_width=1.5))
    for tick in (0, 8, 16, 24, 32):
        x = 1.20 + 4.78*tick/32
        group.add(Line([x, .56, 0], [x, .40, 0], color=MUTED, stroke_width=1),
                  label(str(tick), x, .18, 14, MUTED))
    bar = Line([1.20, 1.05, 0], [1.20+4.78*count/32, 1.05, 0],
               stroke_width=22, color=GREEN)
    group.add(bar, label(str(count), min(6.15, 1.20+4.78*count/32+.30), 1.05, 19, GREEN))
    if bi in (2, 3, 5):
        group.add(mathline(r'K^M(K-1)'))
    else:
        group.add(label('2状態: 1行に自由な値が1個', 0, -2.32, 19, YELLOW))
    return group, bar if bi >= 3 else focus


def state_graph(bi):
    group = VGroup()
    xs = [-5.25, -2.65, -.05, 2.55, 5.15]
    group.add(label('隠れた状態', -5.25, 2.24, 20, PURPLE),
              label('見える天気', -5.25, -1.60, 20, BLUE))
    focus = None
    for i, x in enumerate(xs[:min(5, bi+1)]):
        upper = Circle(radius=.33, stroke_color=PURPLE, stroke_width=2,
                       fill_color=PURPLE, fill_opacity=.15).move_to([x, 1.19, 0])
        lower = Circle(radius=.29, stroke_color=BLUE, stroke_width=2,
                       fill_color=BLUE, fill_opacity=.15).move_to([x, -.88, 0])
        group.add(upper, lower, tex(f'z_{{{i+1}}}', 29, WHITE).move_to(upper),
                  tex(f'x_{{{i+1}}}', 29, WHITE).move_to(lower))
        if i:
            group.add(Arrow([xs[i-1]+.42, 1.19, 0], [x-.42, 1.19, 0],
                            color=YELLOW, buff=0, stroke_width=2, tip_length=.14))
        group.add(Arrow([x, .75, 0], [x, -.48, 0],
                        color=GREEN, buff=0, stroke_width=2, tip_length=.14))
        if i == min(4, bi):
            focus = upper
    if bi in (2, 3):
        group.add(label('留まる 0.9　　乾き→雨 0.2　　湿り→雨 0.8',
                        0, -2.20, 18, GREEN))
    elif bi == 0:
        heading = label('復習: 9.2 隠れた成分', 0, -2.18, 19)
        group.add(heading, SurroundingRectangle(heading, color=YELLOW,
                                                stroke_width=1.4, buff=.10))
    elif bi == 5:
        group.add(mathline(r'z_{n+1}\perp\!\!\!\perp z_{n-1}\mid z_n', -2.20, 27))
    else:
        group.add(label('状態の鎖は1歩、観測の予測は過去の列から', 0, -2.18, 18, MUTED))
    return group, focus


def posterior_panel(bi):
    histories = ([0], [1]) if bi <= 2 else ([0, 1], [1, 1])
    group = VGroup()
    focus = None
    for j, history in enumerate(histories):
        x0 = -6.25 + j*6.36
        posterior = ([.5, .5] if bi < 0 else
                     filter_history(history)[-1]['next_prior' if bi == 2
                                                  else 'posterior'])
        group.add(label('観測前' if bi < 0 else '晴→雨' if history == [0, 1] else '雨→雨' if history == [1, 1]
                        else '昨日: 晴' if history == [0] else '昨日: 雨',
                        x0+2.5, 2.10, 20, BLUE if j == 0 else GREEN))
        for k, name in enumerate(('乾き', '湿り')):
            y = 1.13-k*.88
            group.add(label(name, x0+.45, y, 16, MUTED),
                      Line([x0+1.20, y, 0], [x0+5.25, y, 0], color=MUTED,
                           stroke_width=1.1))
            bar = Line([x0+1.20, y, 0], [x0+1.20+3.9*posterior[k], y, 0],
                       color=PURPLE if k == 0 else GREEN, stroke_width=15)
            group.add(bar, label(f'{posterior[k]*100:.1f}%', x0+5.80, y, 17,
                                 PURPLE if k == 0 else GREEN))
            if j == 1 and k == 1:
                focus = bar
        next_rain = .5 if bi < 0 else predict_rain(history)
        group.add(label(f'{"今日" if bi == 2 else "明日"}の雨 {next_rain*100:.1f}%',
                        x0+2.90, -1.28, 20, YELLOW))
    for tick in (0, 50, 100):
        group.add(label(str(tick), -5.05+3.9*tick/100, -1.87, 13, MUTED),
                  label(str(tick), 1.31+3.9*tick/100, -1.87, 13, MUTED))
    group.add(label('状態の確率（%）', 0, -2.28, 16, MUTED))
    if bi == 4:
        group.add(mathline(r'p(z_1)\prod_{n=2}^{N}p(z_n\mid z_{n-1})\prod_{n=1}^{N}p(x_n\mid z_n)', -2.60, 21))
    return group, focus


def panel(si, bi):
    if si == 0:
        return opening_panel(bi)
    if si == 1:
        if bi in (3, 4):
            card = recap_condition()
            return card, card[0]
        group = VGroup(observation_graph(0))
        bars, focus = forecast_bars([.5, .5], '独立モデルの予測', BLUE)
        group.add(bars, label('矢印がない → 順序を使わない', -3.39, -2.34, 18, MUTED))
        return group, focus
    if si == 2:
        return chain_panel(bi, 1)
    if si == 3:
        return chain_panel(bi, 2)
    if si == 4:
        return parameter_panel(bi)
    if si == 5:
        return recap_mixture() if bi == 0 else state_graph(bi)
    if si == 6:
        return posterior_panel(bi)
    group, focus = opening_panel(bi, True)
    if bi == 4:
        group.add(label('独立 50%　一次 64.4%　状態を追う 54.0% / 70.1%',
                        0, -2.57, 17, YELLOW))
    return group, focus


class PRML131MarkovModels(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.manifest = {e['id']: e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline = []
        for si, story in enumerate(SCENES):
            self.begin(si)
            old = None
            for bi, _ in enumerate(story['beats']):
                state, focus = panel(si, bi)
                if si == 6 and bi == 0:
                    initial, _ = posterior_panel(-1)
                    self.add(initial)
                    duration = self.boundaries[bi]
                    first = self.sentence_duration(0)
                    self.beat(phases=[('最初の重み', first, lambda: Wait()),
                                      ('観測で更新', duration-first,
                                       lambda a=initial, b=state:
                                       Succession(FadeOut(a, run_time=.05),
                                                  FadeIn(b, run_time=.05),
                                                  Wait(.90)))])
                    old = state
                    continue
                transition = (FadeIn(state) if old is None else
                              Succession(FadeOut(old, run_time=.20),
                                         FadeIn(state, run_time=.20), Wait(.60)))
                duration = self.boundaries[bi] - (self.boundaries[bi-1] if bi else 0)
                switch = min(1.2, duration*.20)
                self.beat(phases=[('表示切替', switch, lambda a=transition: a),
                                  ('注目箇所', duration-switch,
                                   lambda f=focus: Indicate(f, color=YELLOW,
                                                            scale_factor=1.015))])
                old = state
            self.timeline[-1]['end'] = float(self.time)
        (Path(config.media_dir)/'prml131_timeline.json').write_text(
            json.dumps(self.timeline, ensure_ascii=False, indent=2)+'\n')
