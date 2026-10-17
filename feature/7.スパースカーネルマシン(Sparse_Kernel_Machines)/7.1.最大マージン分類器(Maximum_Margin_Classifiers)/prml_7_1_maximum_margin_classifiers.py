"""The same six points explain maximum margins, sparse prediction and slack."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from margin_model import X, T, EXTRA, soft_svm, nonlinear_shift

BG = '#10141F'
BLUE = ManimColor('#58C4DD')
YELLOW = ManimColor('#FFFF00')
GREEN = ManimColor('#83C167')
PURPLE = ManimColor('#9A72AC')
RED = ManimColor('#FC6255')
MUTED = ManimColor('#A8B2C5')
_fade_in = FadeIn


def FadeIn(mobject, **kwargs):
    # Labels and referenced points are readable near the start of their speech.
    kwargs.setdefault('rate_func', lambda t: smooth(min(1., 10*t)))
    return _fade_in(mobject, **kwargs)


def L(s, x=0, y=2.48, size=26, color=WHITE):
    return jp(s, size, color).move_to((x, y, 0))


def F(s, x=0, y=2.45, size=30, color=WHITE):
    return tex(s, size, color).move_to((x, y, 0))


def graph():
    ax = Axes(x_range=[-2.6, 2.6, 1], y_range=[-1.55, 1.55, 1],
              x_length=9., y_length=3.75, tips=False,
              axis_config=dict(color=MUTED, stroke_width=1.2, include_ticks=False))
    ax.shift(DOWN*.30)
    labels = VGroup()
    for x in [-2, -1, 0, 1, 2]:
        p = ax.c2p(x, -1.55)
        labels.add(Line(p, p+UP*.065, color=MUTED, stroke_width=1))
        labels.add(F(str(x), p[0], p[1]-.22, 22, MUTED))
    labels.add(F('x_1', ax.c2p(2.6, -1.55)[0]+.30, ax.c2p(2.6, -1.55)[1]-.02, 24))
    labels.add(F('x_2', ax.c2p(-2.6, 1.55)[0]-.20, ax.c2p(-2.6, 1.55)[1]+.12, 24))
    labels.add(F('t=-1', -3.1, 1.97, 23, BLUE), F('t=+1', 3.1, 1.97, 23, GREEN))
    return ax, labels


def points(ax, extra=False):
    result = VGroup(*[Dot(ax.c2p(*xy), radius=.085, color=BLUE if t < 0 else GREEN).set_z_index(3)
                      for xy, t in zip(X, T)])
    if extra:
        result.add(Dot(ax.c2p(*EXTRA), radius=.10, color=RED).set_z_index(3))
    return result


def vertical(ax, x, color=GREEN, width=3):
    return Line(ax.c2p(x, -1.52), ax.c2p(x, 1.52), color=color, stroke_width=width)


def margin_pair(ax, x, half=1):
    return VGroup(*[DashedLine(ax.c2p(x+d, -1.52), ax.c2p(x+d, 1.52),
                              color=YELLOW, stroke_width=1.6, dash_length=.12)
                    for d in [-half, half]])


def distance(ax, at_x, point_x, point_y=0, color=YELLOW):
    return Line(ax.c2p(at_x, point_y), ax.c2p(point_x, point_y), color=color, stroke_width=4)


def halo(dot):
    return Circle(radius=.16, color=YELLOW, stroke_width=2).move_to(dot).set_z_index(4)

def pulse(group):
    return AnimationGroup(*[Indicate(m, color=YELLOW, scale_factor=1.18) for m in group])

def trace(group):
    return ShowPassingFlash(group.copy().set_stroke(YELLOW, 4), time_width=.5)


class PRML71MaximumMarginClassifiers(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.manifest = {e['id']: e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline = []
        for i, method in enumerate([self.opening, self.measure, self.scale, self.support,
                                    self.kernel, self.overlap, self.soft, self.return_to_question]):
            self.begin(i)
            self.current_note = None
            method()
            assert self.beat_index == len(SCENES[i]['beats']), (i, self.beat_index)
            self.timeline[-1]['end'] = float(self.time)
        Path(config.media_dir, 'prml71_timeline.json').write_text(
            json.dumps(self.timeline, ensure_ascii=False, indent=2)+'\n')

    def note(self, m):
        if self.current_note is not None:
            self.remove(self.current_note)
        self.current_note = m
        return FadeIn(m)

    def point_moves(self, ax, dots):
        targets = [(-2.25, -.7), (-1.7, 1.25), (2.25, -1.25), (1.7, .7)]
        return [d.animate.move_to(ax.c2p(*p))
                for d, p in zip([dots[1], dots[2], dots[4], dots[5]], targets)]

    def opening(self):
        recap = VGroup(L('復習 6.1・6.2', -3, 1.5, 27),
                       Dot((-2.5, -.1, 0), color=BLUE), Dot((-1.3, -.1, 0), color=BLUE),
                       Arrow((-.8, -.1, 0), (1.1, -.1, 0), color=YELLOW, buff=.1),
                       F(r'k(x,z)=\phi(x)^T\phi(z)', 2.65, -.1, 32, GREEN))
        self.beat(FadeIn(recap))
        self.remove(recap)
        ax, labs = graph(); dots = points(ax); candidate = vertical(ax, .4, PURPLE)
        self.beat(FadeIn(ax), FadeIn(labs), FadeIn(dots), Create(candidate))
        self.beat(self.note(L('点が少し動いても大丈夫？', color=YELLOW)),
                  Indicate(dots[3], color=YELLOW, scale_factor=1.35))
        short = distance(ax, .4, 1)
        num = F('0.6', 1.2, .15, 27, YELLOW)
        self.beat(Create(short), FadeIn(num), Indicate(dots[3], color=YELLOW, scale_factor=1.3))
        central = vertical(ax, 0); long = distance(ax, 0, 1)
        self.beat(ReplacementTransform(candidate, central), ReplacementTransform(short, long),
                  Transform(num, F('1', .85, .15, 27, YELLOW)))
        rings = VGroup(halo(dots[0]), halo(dots[3]))
        self.beat(FadeIn(rings), self.note(L('どの点が線を決める？', color=YELLOW)))
        corner = VMobject().set_points_as_corners([ax.c2p(.12, 0), ax.c2p(.12, .17), ax.c2p(0, .17)]).set_stroke(YELLOW, 2)
        self.beat(Create(corner), Indicate(long, color=YELLOW, scale_factor=1.04))

    def measure(self):
        ax, labs = graph(); dots = points(ax); line = vertical(ax, -.4, PURPLE)
        self.beat(FadeIn(ax), FadeIn(labs), FadeIn(dots), Create(line),
                  self.note(VGroup(L('復習 4.1', -3.05, 2.48, 23, BLUE), F(r'y(x)=w^Tx+b', 1, 2.48, 31))))
        arm = distance(ax, -.4, -1)
        self.beat(Create(arm), Indicate(dots[0], color=YELLOW, scale_factor=1.25))
        normal = Arrow(ax.c2p(-.4, .65), ax.c2p(.5, .65), buff=0, color=YELLOW)
        self.beat(self.note(F(r'd_n={t_ny(x_n)\over\|w\|}', color=YELLOW)), GrowArrow(normal))
        labels = VGroup(*[F(f'{abs(x+.4):.1f}', ax.c2p(x, y)[0]+(.4 if t > 0 else -.4),
                            ax.c2p(x, y)[1]+.24, 19, MUTED) for (x, y), t in zip(X, T)])
        self.beat(FadeIn(labels), Indicate(arm, color=YELLOW, scale_factor=1.06))
        self.remove(labels, normal)
        toward = vertical(ax, .4, PURPLE); arm2 = distance(ax, .4, 1)
        self.beat(ReplacementTransform(line, toward), ReplacementTransform(arm, arm2),
                  self.note(L('片側に寄ると、逆側が狭い', color=YELLOW)))
        center = vertical(ax, 0); margins = margin_pair(ax, 0)
        self.beat(ReplacementTransform(toward, center), ReplacementTransform(arm2, distance(ax, 0, 1)),
                  FadeIn(margins), self.note(L('最短距離 1', color=GREEN)))
        self.beat(Indicate(center, color=YELLOW, scale_factor=1.01), self.note(L('同じ線の式は何通り？')))

    def scale(self):
        ax, labs = graph(); dots = points(ax); line = vertical(ax, 0); margins = margin_pair(ax, 0)
        self.beat(FadeIn(ax), FadeIn(labs), FadeIn(dots), Create(line), FadeIn(margins),
                  self.note(F(r'y=x_1\quad y=2x_1')))
        normal = Arrow(ax.c2p(0, .45), ax.c2p(1., .45), buff=0, color=YELLOW, stroke_width=4)
        self.beat(phases=[
            ('法線', self.sentence_duration(0), lambda: AnimationGroup(GrowArrow(normal), self.note(L('重みは境界に垂直', color=YELLOW)))),
            ('尺度を2倍', self.sentence_duration(1), lambda: Transform(normal, Arrow(ax.c2p(0, .45), ax.c2p(2., .45), buff=0, color=YELLOW, stroke_width=4)))])
        self.beat(Transform(normal, Arrow(ax.c2p(0, .45), ax.c2p(1., .45), buff=0, color=YELLOW)),
                  self.note(F(r't_ny(x_n)\geq1')), FadeIn(VGroup(halo(dots[0]), halo(dots[3]))))
        self.beat(self.note(VGroup(L('余白', -2, 2.48, 25, YELLOW), F(r'm=1/\|w\|', .8, 2.48, 31, YELLOW))),
                  trace(margins))
        self.beat(self.note(F(r'\min_{w,b}\;\|w\|^2/2\qquad t_ny_n\geq1', size=28, color=GREEN)),
                  Indicate(line, color=YELLOW, scale_factor=1.005))
        bad = vertical(ax, .65, PURPLE); bad_margins = margin_pair(ax, .65)
        self.beat(ReplacementTransform(line, bad), ReplacementTransform(margins, bad_margins),
                  Indicate(dots[3], color=RED, scale_factor=1.35), self.note(L('点が余白の内側へ入る', color=RED)))
        self.beat(ReplacementTransform(bad, vertical(ax, 0)), ReplacementTransform(bad_margins, margin_pair(ax, 0)),
                  self.note(L('接している点だけ残す', color=YELLOW)),
                  *[dots[i].animate.set_opacity(.2) for i in [1, 2, 4, 5]])

    def support(self):
        ax, labs = graph(); dots = points(ax); line = vertical(ax, 0); margins = margin_pair(ax, 0)
        rings = VGroup(halo(dots[0]), halo(dots[3]))
        self.beat(FadeIn(ax), FadeIn(labs), FadeIn(dots), Create(line), FadeIn(margins), FadeIn(rings),
                  self.note(L('サポートベクトル 2点', color=YELLOW)))
        self.beat(*self.point_moves(ax, dots), Indicate(line, color=YELLOW, scale_factor=1.005))
        self.beat(*[dots[i].animate.move_to(ax.c2p(*X[i])) for i in [1, 2, 4, 5]],
                  self.note(L('接点が境界を止める', color=GREEN)), pulse(rings))
        coeff = VGroup(*[F(r'a_1', dots[0].get_x()-.65, dots[0].get_y()+.45, 23, YELLOW),
                         F(r'a_4', dots[3].get_x()+.65, dots[3].get_y()+.45, 23, YELLOW)])
        zeros = VGroup(*[F('a=0', dots[i].get_x(), dots[i].get_y()+.3, 20, MUTED) for i in [1, 2, 4, 5]])
        self.beat(FadeIn(coeff), FadeIn(zeros), self.note(F(r'a_n(t_ny_n-1)=0')))
        arrows = VGroup(Arrow(ax.c2p(0, -.65), ax.c2p(.5, -.65), color=BLUE, buff=0),
                        Arrow(ax.c2p(.5, -.65), ax.c2p(1, -.65), color=GREEN, buff=0))
        values = VGroup(*[F('a=0.5', d.get_x()+side*.65, d.get_y()+.45, 23, YELLOW) for d,side in zip([dots[0], dots[3]],[-1,1])])
        self.beat(Transform(coeff, values), FadeIn(arrows), self.note(F(r'w=\sum_n a_nt_nx_n=(1,0)', size=28, color=GREEN)))
        self.remove(arrows)
        self.beat(self.note(F(r'\max_a\left[\sum_n a_n-\frac12\sum_{n,m}a_na_mt_nt_m k(x_n,x_m)\right]', size=26)),
                  Indicate(coeff, color=YELLOW, scale_factor=1.06))
        self.beat(*[dots[i].animate.set_opacity(.18) for i in [1, 2, 4, 5]], FadeOut(zeros),
                  self.note(L('予測に残るのは 2点', color=GREEN)))

    def kernel(self):
        ax, labs = graph(); dots = points(ax); line = vertical(ax, 0)
        query = Dot(ax.c2p(.2, .8), radius=.08, color=YELLOW)
        arcs = VGroup(*[Line(d.get_center(), query.get_center(), color=c, stroke_width=2)
                       for d, c in [(dots[0], BLUE), (dots[3], GREEN)]])
        self.beat(FadeIn(ax), FadeIn(labs), FadeIn(dots), Create(line), FadeIn(query), Create(arcs),
                  self.note(F(r'k(x,z)=x^Tz')))
        self.beat(self.note(F(r'k(x,x_n)=\phi(x)^T\phi(x_n)', size=28, color=YELLOW)),
                  trace(arcs))
        self.beat(self.note(F(r'y(x)=\sum_n a_nt_nk(x,x_n)+b', size=28, color=GREEN)),
                  pulse(VGroup(dots[0], dots[3])))
        self.beat(*[dots[i].animate.set_color(MUTED).set_opacity(.3) for i in [1, 2, 4, 5]],
                  self.note(L('係数 0 の点は式から消える', color=GREEN)))
        self.remove(arcs, query)
        wave = VMobject().set_points_smoothly([ax.c2p(nonlinear_shift(y), y) for y in np.linspace(-1.5, 1.5, 55)]).set_stroke(GREEN, 3)
        self.beat(ReplacementTransform(line, wave),
                  *[dots[i].animate.move_to(ax.c2p(x+nonlinear_shift(y), y)) for i, (x, y) in enumerate(X)],
                  self.note(L('説明用：特徴を曲げるカーネル', size=25, color=YELLOW)))
        self.beat(self.note(L('支える点は学習で決まる', color=GREEN)),
                  pulse(VGroup(dots[0], dots[3])))
        self.beat(ReplacementTransform(wave, vertical(ax, 0)),
                  *[dots[i].animate.move_to(ax.c2p(*X[i])).set_color(BLUE if T[i] < 0 else GREEN).set_opacity(1) for i in range(6)],
                  self.note(L('元の6点に、逆ラベルの点を追加？', size=24, color=RED)))

    def overlap(self):
        ax, labs = graph(); dots = points(ax, True); line = vertical(ax, 0); margins = margin_pair(ax, 0)
        self.beat(FadeIn(ax), FadeIn(labs), FadeIn(dots), Create(line), FadeIn(margins),
                  self.note(L('赤い点は青クラス', color=RED)))
        hull = Polygon(*[ax.c2p(*p) for p in X[3:]], color=GREEN, fill_color=GREEN, fill_opacity=.1, stroke_width=1)
        self.beat(FadeIn(hull), Indicate(dots[-1], color=RED, scale_factor=1.3),
                  self.note(L('緑の三角形の内側に青クラス', size=25, color=RED)))
        slack = distance(ax, -1, 1.4, -.5, RED)
        self.beat(Create(slack), self.note(F(r'\xi_n=\max(0,1-t_ny_n)', color=YELLOW)),
                  FadeIn(F(r'\xi_7=2.4', .3, -1.25, 24, RED)))
        original = dots[3].get_center().copy()
        self.beat(phases=[
            ('正しい側の侵入', self.sentence_duration(0), lambda: AnimationGroup(dots[3].animate.move_to(ax.c2p(.55, 0)), self.note(L('正しい側でも余白へ侵入する', size=25, color=YELLOW)))),
            ('誤分類した点', self.sentence_duration(1), lambda: Indicate(dots[-1], color=RED, scale_factor=1.3))])
        self.beat(dots[3].animate.move_to(original), self.note(F(r't_ny(x_n)\geq1-\xi_n')))
        self.beat(Indicate(slack, color=RED, scale_factor=1.04), self.note(L('ずれには費用が必要', color=RED)))
        self.beat(FadeOut(hull), self.note(L('余白とずれを比べる', color=YELLOW)),
                  trace(margins))

    def soft(self):
        ax, labs = graph(); dots = points(ax, True)
        s1, s2 = soft_svm(.2), soft_svm(.5)
        line = vertical(ax, s1['decision_x']); margins = margin_pair(ax, s1['decision_x'], 1/s1['w'][0])
        self.add(ax, labs, dots, line, margins)
        bars = VGroup(); bottom = -1.7
        for value in s1['slack'][[0, 3, 6]]:
            h = .60*float(value)
            bars.add(Rectangle(width=.48, height=h, fill_color=YELLOW, fill_opacity=.8,
                               stroke_color=BG, stroke_width=1).move_to((5.35, bottom+h/2, 0)))
            bottom += h
        total = F(r'2.648', 5.35, .25, 23, YELLOW)
        self.beat(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=.25), FadeIn(total),
                  self.note(F(r'\xi_1+\xi_4+\xi_7=\sum_n\xi_n', size=28, color=YELLOW)))
        self.beat(self.note(F(r'\min\;\|w\|^2/2+C\sum_n\xi_n', size=29, color=GREEN)),
                  trace(margins), Indicate(bars, color=YELLOW, scale_factor=1.03))
        slider_track = Line((-1.7, 2.45, 0), (.8, 2.45, 0), color=MUTED, stroke_width=2)
        knob = Dot((-1.7, 2.45, 0), radius=.09, color=YELLOW)
        counter = F('C=0.2', 2.3, 2.45, 29, YELLOW)
        slider = VGroup(slider_track, knob, counter)
        self.beat(self.note(slider), Indicate(dots[-1], color=RED, scale_factor=1.25))
        line2 = vertical(ax, s2['decision_x']); margins2 = margin_pair(ax, s2['decision_x'], 1/s2['w'][0])
        self.beat(ReplacementTransform(line, line2), ReplacementTransform(margins, margins2),
                  knob.animate.move_to((.8, 2.45, 0)), Transform(counter, F('C=0.5', 2.3, 2.45, 29, YELLOW), rate_func=lambda t: float(t>=.5)),
                  FadeOut(bars), FadeOut(total))
        # Child animations can make slider parts top-level scene objects.
        self.remove(slider, slider_track, knob, counter)
        hinge_ax = Axes(x_range=[-.5, 2, 1], y_range=[0, 1.6, 1], x_length=1.6, y_length=1.5,
                        tips=False, axis_config=dict(color=MUTED, include_ticks=False, stroke_width=1)).move_to((5.45, .35, 0))
        hinge = VGroup(hinge_ax, hinge_ax.plot(lambda z: max(0, 1-z), x_range=[-.5, 2], color=RED, use_smoothing=False),
                       F('1', *hinge_ax.c2p(1, 0)[:2]+np.array([0, -.20]), size=19),
                       L('得点', 5.65, -.85, 20, RED))
        self.beat(FadeIn(hinge), self.note(F(r'[1-t_ny(x_n)]_+', color=RED)))
        rings = VGroup(*[halo(dots[i]) for i in np.flatnonzero(s2['alpha'] > 1e-6)])
        self.beat(FadeIn(rings), self.note(F(r'0\leq a_n\leq C', color=GREEN)))
        self.beat(FadeOut(dots[-1]), FadeOut(hinge), FadeOut(rings),
                  ReplacementTransform(line2, vertical(ax, 0)), ReplacementTransform(margins2, margin_pair(ax, 0)),
                  self.note(L('最初の6点に戻る', color=YELLOW)))

    def return_to_question(self):
        ax, labs = graph(); dots = points(ax); candidate = vertical(ax, .4, PURPLE); center = vertical(ax, 0)
        self.beat(FadeIn(ax), FadeIn(labs), FadeIn(dots), Create(candidate), Create(center),
                  self.note(L('どちらも6点すべて正解')))
        d1 = distance(ax, .4, 1); d2 = distance(ax, 0, 1)
        self.beat(phases=[
            ('紫の余白', self.sentence_duration(0), lambda: AnimationGroup(Create(d1), self.note(F(r'x_1=0.4\quad m=0.6', color=PURPLE)))),
            ('中央の余白', self.sentence_duration(1), lambda: AnimationGroup(ReplacementTransform(d1, d2), FadeOut(candidate), self.note(F(r'x_1=0\quad m=1', color=GREEN))))])
        rings = VGroup(halo(dots[0]), halo(dots[3]))
        self.beat(FadeIn(rings), *[MoveAlongPath(dots[i], Line(dots[i].get_center(), ax.c2p(X[i, 0], X[i, 1]*.6)), rate_func=there_and_back) for i in [1, 2, 4, 5]],
                  self.note(L('境界を支えるのは 2点', color=YELLOW)))
        self.beat(self.note(L('全点正解から、余白で選ぶ', color=GREEN)), Indicate(d2, color=YELLOW, scale_factor=1.05))
        self.beat(*[dots[i].animate.set_color(MUTED).set_opacity(.25) for i in [1, 2, 4, 5]],
                  self.note(F(r'y(x)=\sum_{n\in S}a_nt_nk(x,x_n)+b', size=28, color=GREEN)))
        self.beat(self.note(L('重なる点には余白とずれの調整', size=24, color=RED)),
                  Indicate(center, color=YELLOW, scale_factor=1.005))
        self.beat(self.note(L('確率まで欲しいなら？ 7.2へ', color=YELLOW)),
                  pulse(rings))
