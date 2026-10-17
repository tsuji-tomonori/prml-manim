"""A geometric proof of the EM ascent step, on the 9.3 sixteen-point example."""
import json
from pathlib import Path

import numpy as np
from manim import *

from em_model import (POINTS, PROBE_INDEX, INITIAL, components, decomposition,
                      em_history, pi_slice)
from make_voicevox_narration import MANIFEST
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex

BG = '#10141F'
RED = ManimColor('#FC6255')
BLUE = ManimColor('#58C4DD')
YELLOW = ManimColor('#FFFF00')
GREEN = ManimColor('#83C167')
PURPLE = ManimColor('#9A72AC')
MUTED = ManimColor('#A8B2C5')
OLD, NEW = em_history(1)[:2]
Q = OLD['gamma']
ALT_Q = .9 * Q + .1 * Q[:, ::-1]
OLD_D = decomposition(Q, OLD['mu'], OLD['sigma'], OLD['pi'])
NEW_D = decomposition(Q, NEW['mu'], NEW['sigma'], NEW['pi'])
ALT_D = decomposition(ALT_Q, OLD['mu'], OLD['sigma'], OLD['pi'])


def J(s, size=20, color=WHITE):
    return jp(str(s), size, color)


def quick(t):
    return smooth(min(1, 12 * t))


def pt(x, density):
    return np.array([-5.24 + .90 * (float(x) + 3), -1.38 + 5.7 * float(density), 0])


def density_curves(state):
    xs = np.linspace(-3, 3, 151)
    data = components(xs, state['mu'], state['sigma'], state['pi'])
    return VGroup(*[VMobject().set_points_as_corners([pt(x, y) for x, y in zip(xs, ys)])
                    .set_stroke(color, 3) for ys, color in zip(
                        (data[:, 0], data[:, 1], data.sum(axis=1)), (RED, BLUE, GREEN))])


def density_plot(state=OLD):
    axis = VGroup(Line(pt(-3, 0), pt(3, 0), color=MUTED, stroke_width=2),
                  Line(pt(-3, 0), pt(-3, .52), color=MUTED, stroke_width=2),
                  J('密度', 17, MUTED).move_to([-5.35, 1.9, 0]),
                  J('16点', 17, MUTED).move_to([-4.75, -2.70, 0]))
    for x in (-2, 0, 2):
        axis.add(Line(pt(x, 0), pt(x, 0) + DOWN * .11, color=MUTED, stroke_width=1),
                 J(str(x), 15, MUTED).move_to([pt(x, 0)[0], -1.72, 0]))
    dots = VGroup(*[Dot([pt(x, 0)[0], -2.26, 0], radius=.067,
                        color=interpolate_color(BLUE, RED, float(state['gamma'][i, 0])))
                    for i, x in enumerate(POINTS)])
    ring = Circle(radius=.15, color=YELLOW, stroke_width=2).move_to(dots[PROBE_INDEX])
    ring.set_z_index(3)
    return axis, density_curves(state), dots, ring


def level(value):
    return -1.50 + .85 * (value + 30)


def height_diagram(lower, observed, with_labels=True):
    """Actual measured values mapped linearly to two heights, without a backing box."""
    group = VGroup()
    for x, y, color in ((2.72, lower, BLUE), (4.88, observed, RED)):
        group.add(Line([x - .75, level(y), 0], [x + .75, level(y), 0],
                       color=color, stroke_width=5))
        group.add(Line([x, -1.50, 0], [x, level(y), 0], color=color,
                       stroke_width=3, stroke_opacity=.38))
    group.add(Line([1.60, -1.50, 0], [5.99, -1.50, 0], color=MUTED, stroke_width=1.5))
    if with_labels:
        group.add(J('下界', 19, BLUE).move_to([2.72, -1.76, 0]),
                  J('観測', 19, RED).move_to([4.88, -1.76, 0]))
    group.figure_kind = 'height'
    return group


def probability_bars(q, center_x=3.75):
    result = VGroup()
    for i, (label, value, color) in enumerate(zip(('赤い札', '青い札'), (q, 1 - q), (RED, BLUE))):
        y = .45 - .70 * i
        result.add(J(label, 18, color).move_to([center_x - 1.45, y, 0]))
        result.add(Line([center_x - .50, y, 0], [center_x - .50 + 2.00 * value, y, 0],
                        color=color, stroke_width=12))
        result.add(J(f'{value:.3f}', 18).move_to([center_x + 1.45, y, 0]))
    result.figure_kind = 'probability'
    return result


def recap_kl_bars(assumed):
    """Recreate the four-bin P/Q bar operation from 1.6, without importing it."""
    actual = (.5, .25, .125, .125)
    group = VGroup(Rectangle(width=4.46, height=3.24, stroke_color=YELLOW,
                             stroke_width=1.1, fill_opacity=0).move_to([3.74, -.11, 0]),
                   J('復習: 1.6  KLダイバージェンス', 18, YELLOW).move_to([3.73, 1.28, 0]))
    for i, (p, q) in enumerate(zip(actual, assumed)):
        x = 2.27 + i * .99
        base = -1.05
        fill = Rectangle(width=.57, height=2.3 * p, stroke_width=0,
                         fill_color=ManimColor('#58B5ED'), fill_opacity=.9)
        fill.move_to([x, base + fill.height / 2, 0])
        outline = Rectangle(width=.61, height=2.3 * q, stroke_color=ManimColor('#FFB45B'),
                            stroke_width=2, fill_opacity=0)
        outline.move_to([x, base + outline.height / 2, 0])
        group.add(fill, outline)
    group.add(J('青: 基準  橙: 想定', 17, MUTED).move_to([3.74, -1.48, 0]))
    group.figure_kind = 'recap'
    return group


def recap_coin_prior():
    """Recreate the 1.2 three-head coin update, without importing that scene."""
    def xy(theta, density):
        return np.array([1.85 + 3.83 * theta, -1.20 + .82 * density, 0])

    def curve(values, color):
        xs = np.linspace(0, 1, 101)
        return VMobject().set_points_as_corners([xy(x, values(x)) for x in xs]).set_stroke(color, 3)

    blue = ManimColor('#58B5ED')
    orange = ManimColor('#FFB45B')
    group = VGroup(
        Rectangle(width=4.46, height=3.52, stroke_color=YELLOW,
                  stroke_width=1.1, fill_opacity=0).move_to([3.74, -.06, 0]),
        J('復習: 1.2 事前から事後', 18, YELLOW).move_to([3.74, 1.43, 0]),
        Line(xy(0, 0), xy(1, 0), color=MUTED, stroke_width=1.5),
        curve(lambda x: 6 * x * (1 - x), blue),
        curve(lambda x: x ** 3, orange),
        J('青: 事前   橙: 尤度   紫: 事後', 16, MUTED).move_to([3.74, -1.52, 0]),
        curve(lambda x: 30 * x ** 4 * (1 - x), PURPLE),
    )
    group[-1].set_stroke(opacity=0)
    group.figure_kind = 'prior_recap'
    return group


def pi_plane(q):
    """Fixed means and variances; red pi alone varies from .3 to .75."""
    def xy(pi, value):
        return np.array([1.78 + (pi - .3) * 8.80, -1.28 + (value + 30.5) * .54, 0])
    xs = np.linspace(.3, .75, 101)
    obs = np.array([pi_slice(float(x), q)['observed'] for x in xs])
    low = np.array([pi_slice(float(x), q)['lower'] for x in xs])
    axis = VGroup(Line(xy(.3, -30.5), xy(.75, -30.5), color=MUTED, stroke_width=1.7),
                  Line(xy(.3, -30.5), xy(.3, -26.5), color=MUTED, stroke_width=1.7))
    for x in (.3, .5, .7):
        axis.add(Line(xy(x, -30.5), xy(x, -30.5) + DOWN * .1, color=MUTED),
                 J(f'{x:.1f}', 15, MUTED).move_to(xy(x, -30.5) + DOWN * .29))
    axis.add(J('赤の混合係数', 17, MUTED).move_to([3.80, -2.08, 0]))
    red = VMobject().set_points_as_corners([xy(x, y) for x, y in zip(xs, obs)]).set_stroke(RED, 3)
    blue = VMobject().set_points_as_corners([xy(x, y) for x, y in zip(xs, low)]).set_stroke(BLUE, 3)
    return axis, red, blue, xy


class PRML94GeneralEM(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.manifest = {e['id']: e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline = []
        for i, method in enumerate((self.opening, self.decomposition, self.lower_bound,
                                    self.expectation, self.maximization, self.parameter_curve,
                                    self.extensions, self.return_to_example)):
            self.begin(i)
            self.note = self.formula = self.figure = None
            method()
            assert self.beat_index == len(SCENES[i]['beats'])
            self.timeline[-1]['end'] = float(self.time)
        (Path(config.media_dir) / 'prml94_timeline.json').write_text(
            json.dumps(self.timeline, ensure_ascii=False, indent=2) + '\n')

    def setup_plot(self, state=OLD):
        self.axis, self.curves, self.dots, self.ring = density_plot(state)
        self.add(self.axis, self.curves, self.dots, self.ring)

    def set_note(self, s, color=MUTED):
        mob = J(s, 21, color)
        if mob.width > 4.75:
            mob.scale_to_fit_width(4.75)
        mob.move_to([3.78, -2.47, 0])
        if self.note is not None:
            self.remove(self.note)
        self.note = mob
        return FadeIn(mob, rate_func=quick)

    def set_formula(self, s):
        mob = tex(s, 34)
        if mob.width > 4.9:
            mob.scale_to_fit_width(4.9)
        if mob.height > .83:
            mob.scale_to_fit_height(.83)
        mob.move_to([3.78, 2.30, 0])
        if self.formula is not None:
            self.remove(self.formula)
        self.formula = mob
        return FadeIn(mob, rate_func=quick)

    def set_figure(self, mob):
        if self.figure is None:
            self.figure = mob
            return FadeIn(mob, rate_func=quick)
        old = self.figure
        self.figure = mob
        if getattr(old, 'figure_kind', None) == getattr(mob, 'figure_kind', None):
            return ReplacementTransform(old, mob)
        return AnimationGroup(FadeOut(old, rate_func=quick),
                              FadeIn(mob, rate_func=quick), lag_ratio=.1)

    def say(self, note, *moves, formula=None, figure=None, color=MUTED):
        animations = [self.set_note(note, color)]
        if formula is not None:
            animations.append(self.set_formula(formula))
        if figure is not None:
            animations.append(self.set_figure(figure))
        animations.extend(moves)
        self.beat(*animations)

    def move_state(self, state):
        return AnimationGroup(Transform(self.curves, density_curves(state)),
                              *[dot.animate.set_color(interpolate_color(BLUE, RED, float(state['gamma'][i, 0])))
                                for i, dot in enumerate(self.dots)], lag_ratio=0)

    def opening(self):
        self.setup_plot()
        self.say('9.3 と同じ16点', Indicate(self.dots, color=YELLOW, scale_factor=1.04),
                 figure=height_diagram(OLD_D['lower'], OLD_D['observed']))
        self.say('上がるか、下がるか？', Indicate(self.ring, color=YELLOW, scale_factor=1.2),
                 formula=r'\ell(\theta)=\ln p(X\mid\theta)')
        self.say('1回で観測の採点が上昇', self.move_state(NEW),
                 figure=height_diagram(NEW_D['lower'], NEW_D['observed']),
                 formula=r'-27.737\longrightarrow-26.521', color=GREEN)
        self.say('同じ16点で約1.217上昇', Indicate(self.figure, color=YELLOW, scale_factor=1.02),
                 formula=r'\Delta\ell\simeq1.217')
        self.say('成分の札は見えない', Indicate(self.ring, color=YELLOW, scale_factor=1.2))
        self.say('見えない札でなぜ上がる？', self.move_state(OLD), color=YELLOW)
        self.say('観測 X と潜在変数 Z', formula=r'X\quad Z\quad\theta')

    def decomposition(self):
        self.setup_plot()
        self.say('札の候補を足して観測の確率', formula=r'p(X|\theta)=\sum_Zp(X,Z|\theta)',
                 figure=height_diagram(ALT_D['lower'], ALT_D['observed']))
        self.say('仮の札の重みを置く', figure=probability_bars(float(ALT_Q[PROBE_INDEX, 0])))
        self.say('観測の採点は下界と差の和',
                 figure=height_diagram(ALT_D['lower'], ALT_D['observed']),
                 formula=r'\ln p(X|\theta)=\mathcal L(q,\theta)+\mathrm{KL}(q\|p)')
        self.say('同時確率を事後と観測の積へ',
                 formula=r'p(X,Z|\theta)=p(Z|X,\theta)p(X|\theta)')
        self.say('下界は札付きの対数の平均', formula=r'\mathcal L=\sum_Zq(Z)\ln\frac{p(X,Z|\theta)}{q(Z)}')
        self.say('差は札の分布のずれ', formula=r'\mathrm{KL}=\sum_Zq(Z)\ln\frac{q(Z)}{p(Z|X,\theta)}')
        self.say('下界と観測の間の隙間', Indicate(self.figure[0], color=YELLOW, scale_factor=1.06))

    def lower_bound(self):
        self.setup_plot()
        self.say('復習: 1.6 KLダイバージェンス', figure=recap_kl_bars((.1, .2, .3, .4)), color=YELLOW)
        self.say('分布が重なると差は0', figure=recap_kl_bars((.5, .25, .125, .125)),
                 formula=r'\mathrm{KL}(q\|p)\geq0')
        self.say('下界は観測を越えない', figure=height_diagram(ALT_D['lower'], ALT_D['observed']),
                 formula=r'\mathcal L(q,\theta)\leq\ln p(X|\theta)')
        self.say('等号は札の分布が事後と一致するとき',
                 formula=r'q(Z)=p(Z|X,\theta)')
        self.say('同じ16点で実測', figure=height_diagram(OLD_D['lower'], OLD_D['observed']),
                 formula=r'\mathcal L=\ell=-27.737')
        self.say('隙間が0なのはなぜ？', Indicate(self.ring, color=YELLOW, scale_factor=1.18))
        self.say('期待値 = 確率を重みにした平均', figure=probability_bars(float(Q[PROBE_INDEX, 0])),
                 formula=r'E[z]=0.564\cdot1+0.436\cdot0')

    def expectation(self):
        self.setup_plot()
        self.say('仮の札の重みをずらす', figure=height_diagram(ALT_D['lower'], ALT_D['observed']))
        self.say('E: 今の山で札を推測', figure=probability_bars(float(Q[PROBE_INDEX, 0])),
                 formula=r'q(Z)\leftarrow p(Z|X,\theta^{old})')
        self.say('札の分布が一致して KL=0', figure=height_diagram(OLD_D['lower'], OLD_D['observed']),
                 formula=r'\mathrm{KL}(q\|p)=0')
        self.say('山の設定はまだ固定', Indicate(self.curves[2], color=YELLOW, scale_factor=1.01))
        self.say('0.15 の札を次へ渡す', Indicate(self.ring, color=YELLOW, scale_factor=1.18),
                 figure=probability_bars(float(Q[PROBE_INDEX, 0])))
        self.say('全16点の負担率を合わせる', Indicate(self.dots, color=YELLOW, scale_factor=1.03),
                 formula=r'q(Z)=p(Z|X,\theta^{old})')
        self.say('下界を上げたら観測は？', figure=height_diagram(OLD_D['lower'], OLD_D['observed']))

    def maximization(self):
        self.setup_plot()
        self.say('M: 札を固定して山を動かす', figure=probability_bars(float(Q[PROBE_INDEX, 0])),
                 formula=r'q=q^{old},\quad\theta\longrightarrow\theta^{new}')
        self.say('下界 = Q + エントロピー', figure=height_diagram(OLD_D['lower'], OLD_D['observed']),
                 formula=r'\mathcal L(q,\theta)=Q(\theta,\theta^{old})+H(q)')
        self.say('Qが上がると下界も上がる', self.move_state(NEW),
                 figure=height_diagram(NEW_D['lower'], NEW_D['observed']),
                 formula=r'Q:-32.706\longrightarrow-31.655')
        self.say('札のエントロピーは固定', Indicate(self.figure[0], color=YELLOW, scale_factor=1.06),
                 formula=r'H(q)=4.968')
        self.say('観測は新しい下界以上', Indicate(self.figure[2], color=YELLOW, scale_factor=1.06))
        self.say('新しい隙間は KL=0.166', formula=r'-26.521-(-26.687)=0.166')
        self.say('新しい観測は古い観測以上', formula=r'\ell(\theta^{new})\geq\mathcal L(q,\theta^{new})\geq\ell(\theta^{old})',
                 color=GREEN)

    def parameter_curve(self):
        self.setup_plot()
        axis, red, blue, xy = pi_plane(Q)
        self.say('中心と幅を固定した断面', figure=VGroup(axis, red, blue))
        self.say('全パラメータ更新とは別の実験',
                 Indicate(axis[0], color=YELLOW, scale_factor=1.01))
        old_p = Dot(xy(.65, pi_slice(.65)['observed']), color=YELLOW, radius=.085)
        self.say('赤の混合係数 0.650', FadeIn(old_p, rate_func=quick),
                 formula=r'\pi_1=0.650')
        self.say('Eで赤と青が接する', Indicate(old_p, color=YELLOW, scale_factor=1.18))
        opt = float(Q[:, 0].mean())
        new_p = Dot(xy(opt, pi_slice(opt)['lower']), color=YELLOW, radius=.085)
        self.say('青い曲線の山へ Mで移動', Transform(old_p, new_p),
                 formula=r'\pi_1=0.545')
        new_q = pi_slice(opt)['posterior']
        _, _, second_blue, _ = pi_plane(new_q)
        self.say('再Eで新しい位置に接する', Transform(blue, second_blue),
                 Transform(old_p, Dot(xy(opt, pi_slice(opt)['observed']), color=YELLOW, radius=.085)))
        self.say('独立な点なら札も点ごと', Indicate(self.dots, color=YELLOW, scale_factor=1.02),
                 formula=r'p(Z|X,\theta)=\prod_np(z_n|x_n,\theta)')

    def extensions(self):
        self.setup_plot()
        recap = recap_coin_prior()
        self.figure = recap
        self.add(recap)
        self.say('復習: 1.2 事前から事後', recap[-1].animate.set_stroke(opacity=1),
                 formula=r'p(\theta\mid D)\propto p(D\mid\theta)p(\theta)', color=YELLOW)
        self.say('山の設定に事前分布を足す', figure=height_diagram(OLD_D['lower'], OLD_D['observed']))
        self.say('MAP推定: 事前も加える', formula=r'\mathcal L(q,\theta)+\ln p(\theta)')
        self.say('Mで最大が難しいこともある', Indicate(self.curves[2], color=YELLOW, scale_factor=1.01))
        mid = dict(mu=.5*(OLD['mu']+NEW['mu']), sigma=.5*(OLD['sigma']+NEW['sigma']),
                   pi=.5*(OLD['pi']+NEW['pi']), gamma=Q)
        d = decomposition(Q, mid['mu'], mid['sigma'], mid['pi'])
        self.say('一般化EM: 下界を上げる一歩', self.move_state(mid),
                 figure=height_diagram(d['lower'], d['observed']),
                 formula=r'\Delta\mathcal L\geq0', color=GREEN)
        self.say('部分Eでは下界の変化を追う', Indicate(self.ring, color=YELLOW, scale_factor=1.2),
                 formula=r'q_n(z_n)\leftarrow p(z_n|x_n,\theta)')
        self.say('上昇しても局所的な山で止まりうる',
                 Indicate(self.curves[2], color=YELLOW, scale_factor=1.01),
                 formula=r'\ell(\theta^{new})\geq\ell(\theta^{old})')
        self.say('冒頭の16点へ戻る', self.move_state(OLD),
                 figure=height_diagram(OLD_D['lower'], OLD_D['observed']),
                 formula=r'\mathcal L_{old}=\ell_{old}')

    def return_to_example(self):
        self.setup_plot()
        self.say('同じ16点と x=0.15', Indicate(self.ring, color=YELLOW, scale_factor=1.2),
                 figure=probability_bars(float(Q[PROBE_INDEX, 0])))
        self.say('Eで下界と観測が一致', figure=height_diagram(OLD_D['lower'], OLD_D['observed']),
                 formula=r'\mathcal L_{old}=\ell_{old}=-27.737')
        self.say('Mで下界が上昇', self.move_state(NEW),
                 figure=height_diagram(NEW_D['lower'], NEW_D['observed']),
                 formula=r'\mathcal L_{new}=-26.687')
        self.say('新しい KL は 0.166', formula=r'\ell_{new}=\mathcal L_{new}+0.166=-26.521')
        self.say('増分の差も新しい KL', Indicate(self.figure[0], color=YELLOW, scale_factor=1.06),
                 formula=r'1.217-1.051=0.166')
        self.say('一般的なEMアルゴリズムの保証', Indicate(self.figure[2], color=YELLOW, scale_factor=1.06),
                 color=GREEN)
        self.say('局所的な山で止まることもある', Indicate(self.curves[2], color=YELLOW, scale_factor=1.01))
