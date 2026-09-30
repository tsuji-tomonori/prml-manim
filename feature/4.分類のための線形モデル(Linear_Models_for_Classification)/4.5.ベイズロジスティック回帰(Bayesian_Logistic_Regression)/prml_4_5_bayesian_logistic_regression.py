"""PRML 4.5: continuous visual experiments, implemented in Manim CE."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from bayesian_model import (
    X, T, ALPHA, MAP, COV, SW, SV, SAMPLES, sigmoid, normal, cdf,
    design, scalar_density, scalar_energy, stats, kappa, predictive_integral, predict,
)

RED = ManimColor('#FF7884')
BLUE = ManimColor('#67B9F0')
GREEN = ManimColor('#7CDBAD')
PURPLE = ManimColor('#C5A0F4')
YELLOW = ManimColor('#FFE18B')
MUTED = ManimColor('#A6B1C4')


def curve(ax, x, y, color=GREEN, width=3, opacity=1):
    x, y = np.broadcast_arrays(x, y)
    o = ax.c2p(0, 0)
    pts = o + x[:, None]*(ax.c2p(1, 0)-o) + y[:, None]*(ax.c2p(0, 1)-o)
    return VMobject().set_points_as_corners(pts).set_stroke(color, width, opacity)


def contours(ax, center=MAP, cov=COV, color=GREEN):
    theta = np.linspace(0, TAU, 101)
    unit = np.array([np.cos(theta), np.sin(theta)])
    return VGroup(*[curve(ax, *(center[:, None] + r*np.linalg.cholesky(cov)@unit),
                          color, 2.5, .9) for r in [.65, 1.3, 1.95]])


def readout(label, getter, pos, color=WHITE, places=2):
    prefix = tex(label, 25, color)
    n = DecimalNumber(float(getter()), num_decimal_places=places, font_size=25, color=color)
    g = VGroup(prefix, n).arrange(RIGHT, buff=.13).move_to(pos)
    anchor = n.get_left().copy()
    n.add_updater(lambda m: m.set_value(float(getter())).move_to(anchor, aligned_edge=LEFT))
    return g


class PRML45BayesianLogisticRegression(NarratedScene):
    def construct(self):
        self.camera.background_color = '#10141F'
        self.timeline = []
        self.manifest = {s['id']: s for s in json.loads(MANIFEST.read_text())['scenes']}
        for i, method in enumerate([self.question, self.update_posterior, self.laplace,
                                    self.ellipse, self.projection, self.average,
                                    self.approximation, self.decision]):
            self.begin(i)
            method()
            assert self.beat_index == len(self.story['beats'])
            self.timeline[-1]['end'] = float(self.time)
        Path('media/prml45_timeline.json').write_text(
            json.dumps(self.timeline, ensure_ascii=False, indent=2)+'\n')

    def axes(self, xr=(-1.7, 3, 1), yr=(0, 1, .5), pos=(0, .15, 0),
             width=9, height=3.4, labels=('x', 'p')):
        ax = Axes(x_range=xr, y_range=yr, x_length=width, y_length=height,
                  tips=False, axis_config={'include_ticks': False,
                                           'color': MUTED, 'stroke_width': 1.2}).move_to(pos)
        marks = VGroup()
        for x in [xr[0], (xr[0]+xr[1])/2, xr[1]]:
            marks.add(tex(f'{x:g}', 17, MUTED).move_to(ax.c2p(x, yr[0])+DOWN*.23))
        for y in [yr[0], yr[1]]:
            marks.add(tex(f'{y:g}', 17, MUTED).next_to(ax.c2p(xr[0], y), LEFT, buff=.12))
        marks.add(tex(labels[0], 24).next_to(ax.c2p(xr[1], yr[0]), RIGHT, buff=.2),
                  tex(labels[1], 24).next_to(ax.c2p(xr[0], yr[1]), UP, buff=.12))
        ax.labels = marks
        self.add(ax, marks)
        return ax

    def equation(self, *parts, y=-2.45, size=29):
        f = MathTex(*parts, font_size=size).move_to([0, y, 0])
        if f.width > 12.4:
            f.scale_to_fit_width(12.4)
        self.add(f)
        return f

    def note(self, text, color=MUTED):
        m = jp(text, 21, color).move_to([0, 2.55, 0])
        self.add(m)
        return m

    def slider(self, tracker, lo, hi, pos, label, color=YELLOW, width=3):
        rail = NumberLine(x_range=[lo, hi, hi-lo], length=width,
                          include_ticks=False, color=MUTED).move_to(pos)
        dot = Dot(color=color, radius=.07).add_updater(
            lambda m: m.move_to(rail.n2p(tracker.get_value())))
        g = VGroup(rail, dot, readout(label+'=', tracker.get_value,
                                     np.array(pos)+UP*.34, color))
        self.add(g)
        return g

    def data(self, ax):
        return VGroup(*[Dot(ax.c2p(x, t), color=RED if t else BLUE, radius=.066)
                        .set_stroke(WHITE, 1) for x, t in zip(X, T)])

    def question(self):
        ax = self.axes()
        u = np.linspace(-1.7, 3, 181)
        dots = self.data(ax)
        note = self.note('観測：青 = 0、赤 = 1　　曲線：赤いクラスの確率')
        self.beat(FadeIn(dots))
        b, w = ValueTracker(-.5), ValueTracker(.65)
        line = always_redraw(lambda: curve(ax, u, sigmoid(b.get_value()+w.get_value()*u), RED))
        f = self.equation(r'p(C_1\mid x,w)=\sigma(', r'w_0', '+', r'w_1x', ')',
                          y=-2.65)
        f[1].set_color(PURPLE); f[3].set_color(YELLOW)
        self.remove(note)
        sigmoid_formula = self.equation(r'\sigma(a)=\frac{1}{1+e^{-a}}', y=2.5, size=28)
        self.beat(Create(line))
        self.remove(sigmoid_formula)
        self.add(note)
        s0 = self.slider(b, -1.2, .6, [-2.7, -2.1, 0], 'w_0', PURPLE)
        s1 = self.slider(w, .3, 2, [2.7, -2.1, 0], 'w_1', YELLOW)
        self.beat(b.animate.set_value(.4), end_sentence=1)
        self.beat(w.animate.set_value(1.8), end_sentence=1)
        self.regression_recap()
        self.remove(line, s0, s1, f)
        cloud = VGroup(*[curve(ax, u, sigmoid(design(u)@v), GREEN, 1.5, .48) for v in SAMPLES])
        self.equation(r'p(C_1\mid x,\mathbf t)\simeq\int\sigma(w^{\mathsf T}\phi)q(w)\,dw')
        self.beat(Create(cloud))
        probe = ValueTracker(-1.3)
        vertical = always_redraw(lambda: DashedLine(ax.c2p(probe.get_value(), 0),
                                                    ax.c2p(probe.get_value(), 1), color=YELLOW))
        marker = always_redraw(lambda: Dot(ax.c2p(probe.get_value(),
                                    float(predictive_integral(*stats([probe.get_value()]))[0])),
                                    color=YELLOW, radius=.08))
        self.add(vertical, marker)
        self.beat(probe.animate.set_value(2.7))

    def update_posterior(self):
        ax = self.axes((-2, 4, 1), (0, .85, .2), labels=('w', 'p(w)'))
        u = np.linspace(-2, 4, 221)
        n = ValueTracker(0)
        prior = curve(ax, u, normal(u, 0, 1/ALPHA), PURPLE)
        note = self.note('切片を 0 に固定した、傾きだけのモデル')
        f = self.equation(r'p(w)=\mathcal N(w\mid0,\alpha^{-1})', y=-2.45)
        f.set_color(PURPLE)
        self.beat(Create(prior))
        self.beat(ShowPassingFlash(prior.copy().set_stroke(PURPLE, 5), time_width=.35), Circumscribe(f, color=PURPLE))
        post = always_redraw(lambda: curve(ax, u, scalar_density(u, n.get_value()), GREEN))
        self.add(post)
        self.remove(f)
        f = self.equation(r'p(w\mid\mathbf t)\propto', r'p(w)', r'\prod_n y_n^{t_n}(1-y_n)^{1-t_n}')
        f[1].set_color(PURPLE); f[2].set_color(YELLOW)
        counter = readout('N=', n.get_value, [4.5, 1.4, 0], YELLOW, 0)
        self.add(counter)
        self.equation(r'y_n=\sigma(wx_n),\quad t_n\in\{0,1\}', y=2.0, size=24)
        self.beat(n.animate.set_value(1), start_sentence=0)
        # Data are introduced at integer n; in-between frames are tempered updates.
        self.beat(n.animate.set_value(len(X)))
        new = MathTex(r'\ln p(w\mid\mathbf t)=',
                      r'-\tfrac12\alpha w^2',
                      r'+\sum_n[t_n\ln y_n+(1-t_n)\ln(1-y_n)]+c',
                      font_size=27).move_to([0, -2.45, 0])
        new[1].set_color(PURPLE); new[2].set_color(YELLOW)
        self.beat(phases=[
            ('posterior product', self.sentence_duration(0), lambda: Circumscribe(f, color=PURPLE)),
            ('log transform', .65, lambda: ReplacementTransform(f, new)),
            ('log terms', self.sentence_duration(1)-.65, lambda: Circumscribe(new, color=YELLOW)),
        ])
        scan = ValueTracker(-1)
        marker = always_redraw(lambda: Dot(ax.c2p(scan.get_value(),
                         scalar_density(np.array([scan.get_value()]))[0]), color=YELLOW))
        self.add(marker)
        self.beat(scan.animate.set_value(3.3))

    def laplace(self):
        ax = self.axes((-.6, 2.8, 1), (0, 6, 1), labels=('w', r'E(w)-E_{\min}'))
        u = np.linspace(-.6, 2.8, 221)
        exact = curve(ax, u, scalar_energy(u), YELLOW)
        f = self.equation(r'E(w)=-\ln p(w\mid\mathbf t)+c')
        self.beat(Create(exact))
        p = ValueTracker(-.35)
        dot = always_redraw(lambda: Dot(ax.c2p(p.get_value(), scalar_energy(p.get_value())),
                                        color=RED, radius=.09))
        self.add(dot)
        self.beat(p.animate.set_value(SW), end_sentence=1)
        self.laplace_recap()
        quad = curve(ax, u, .5*(u-SW)**2/SV, GREEN)
        self.remove(f)
        f = self.equation(r'E(w)\simeq E(w_{\mathrm{MAP}})+',
                          r'\tfrac12 H(w-w_{\mathrm{MAP}})^2')
        f[1].set_color(GREEN)
        self.beat(Create(quad))
        radius = ValueTracker(.2)
        chord = always_redraw(lambda: Line(
            ax.c2p(SW-radius.get_value(), .5*radius.get_value()**2/SV),
            ax.c2p(SW+radius.get_value(), .5*radius.get_value()**2/SV), color=PURPLE, stroke_width=5))
        self.add(chord)
        self.beat(radius.animate.set_value(1.1))
        self.remove(ax, ax.labels, exact, quad, dot, chord, f)
        ax = self.axes((-.6, 2.8, 1), (0, .8, .2), labels=('w', 'p(w)'))
        density = curve(ax, u, scalar_density(u), YELLOW)
        gaussian = curve(ax, u, normal(u, SW, SV), GREEN)
        self.note('黄：数値正規化した事後分布　　緑：ラプラス近似')
        self.add(density)
        self.beat(Create(gaussian))
        f = self.equation(r'q(w)=\mathcal N(w\mid w_{\mathrm{MAP}},H^{-1})')
        values = VGroup(tex(f'w_{{\\mathrm{{MAP}}}}={SW:.3f}', 24, RED),
                        tex(f'H^{{-1}}={SV:.3f}', 24, GREEN)).arrange(RIGHT, buff=1).move_to([0, 2.05, 0])
        self.add(values)
        self.beat(Circumscribe(f, color=GREEN), ShowPassingFlash(gaussian.copy().set_stroke(GREEN, 5), time_width=.35))

    def ellipse(self):
        ax = self.axes((-2.1, 1.1, 1), (-.3, 2.6, 1), (-3.1, .1, 0),
                       4.6, 3.3, ('w_0', 'w_1'))
        data_ax = self.axes(pos=(3.1, .1, 0), width=4.6, height=3.3)
        u = np.linspace(-1.7, 3, 141)
        b, w = ValueTracker(float(MAP[0])), ValueTracker(float(MAP[1]))
        dot = always_redraw(lambda: Dot(ax.c2p(b.get_value(), w.get_value()), color=RED))
        line = always_redraw(lambda: curve(data_ax, u, sigmoid(b.get_value()+w.get_value()*u), RED))
        self.add(dot, line, self.data(data_ax))
        note = self.note('重みの空間　→　入力ごとの予測')
        f = self.equation(r'a=', r'w_0', '+', r'w_1x', r',\quad y=\sigma(a)')
        f[1].set_color(PURPLE); f[3].set_color(YELLOW)
        self.beat(Indicate(dot), ShowPassingFlash(line.copy().clear_updaters().set_stroke(RED, 5), time_width=.35))
        self.beat(b.animate.set_value(.3), w.animate.set_value(1.7), end_sentence=1)
        ell = contours(ax)
        self.beat(Create(ell), b.animate.set_value(MAP[0]), w.animate.set_value(MAP[1]))
        self.remove(note)
        self.equation(r'q(w)=\mathcal N(w\mid w_{\mathrm{MAP}},S_N)', y=2.5)
        theta = ValueTracker(0)
        self.remove(dot, line)
        pair = lambda: MAP + 1.6*np.linalg.cholesky(COV)@np.array([np.cos(theta.get_value()), np.sin(theta.get_value())])
        dot = always_redraw(lambda: Dot(ax.c2p(*pair()), color=RED))
        line = always_redraw(lambda: curve(data_ax, u, sigmoid(design(u)@pair()), RED))
        self.add(dot, line)
        self.beat(theta.animate.set_value(TAU))
        self.remove(f)
        f = self.equation(r'S_N^{-1}=', r'S_0^{-1}', r'+\sum_n', r'y_n(1-y_n)',
                          r'\phi_n\phi_n^{\mathsf T}', size=30)
        f[1].set_color(PURPLE); f[3].set_color(YELLOW); f[4].set_color(BLUE)
        self.beat(Circumscribe(f[1], color=PURPLE), Circumscribe(f[2:], color=YELLOW))
        self.precision_aid()
        self.beat(Indicate(f[3], color=YELLOW), theta.animate.set_value(2*TAU))

    def projection(self):
        ax = self.axes((-2.1, 1.1, 1), (-.3, 2.6, 1), (-3.25, .1, 0),
                       4.2, 3.2, ('w_0', 'w_1'))
        out = self.axes((-5, 7, 2), (0, .65, .2), (3.1, .1, 0),
                        4.7, 3.2, ('a', 'p(a)'))
        self.add(contours(ax))
        x = ValueTracker(.3)
        self.slider(x, -.8, 2, [0, 2.4, 0], 'x', YELLOW, width=2.7)
        phi = lambda: np.array([1, x.get_value()])
        origin = ax.c2p(*MAP)
        direction = always_redraw(lambda: Arrow(origin,
                       ax.c2p(*(MAP + .75*phi()/np.linalg.norm(phi()))),
                       buff=0, color=YELLOW, stroke_width=4))
        f = self.equation(r'a=w^{\mathsf T}\phi,\qquad\phi=(1,x)^{\mathsf T}')
        self.beat(GrowArrow(direction))
        # Fixed sample cloud and corresponding weighted sums, joined through animation.
        cloud = VGroup(*[Dot(ax.c2p(*v), color=BLUE, radius=.045) for v in SAMPLES])
        projected = always_redraw(lambda: VGroup(*[Dot(out.c2p(float(v@phi()), .02),
                                                  color=BLUE, radius=.045) for v in SAMPLES]))
        self.add(cloud)
        self.beat(TransformFromCopy(cloud, projected))
        u = np.linspace(-5, 7, 221)
        density = always_redraw(lambda: curve(out, u, normal(u, *[float(v[0]) for v in stats([x.get_value()])]), GREEN))
        recap = jp("復習: 2.3 ガウス分布の線形変換", 21).move_to([-3.0, 2.45, 0])
        border = SurroundingRectangle(recap, color="#FFFF00", buff=.12, stroke_width=1.2)
        self.add(recap, border)
        self.beat(Create(density))
        self.remove(recap, border)
        self.remove(f)
        f = self.equation(r'\mu_a=w_{\mathrm{MAP}}^{\mathsf T}\phi',
                          r',\qquad\sigma_a^2=\phi^{\mathsf T}S_N\phi')
        f[0].set_color(RED); f[1].set_color(GREEN)
        self.beat(x.animate.set_value(1.1), start_sentence=1)
        self.beat(x.animate.set_value(2), start_sentence=1)
        self.beat(x.animate.set_value(-.8))

    def average(self):
        ax = self.axes((-7, 11, 3), (0, 1.05, .5), width=8.8, pos=(-.65, .15, 0),
                       labels=('a', ''))
        u = np.linspace(-7, 11, 301)
        var = ValueTracker(.36)
        self.note('赤：シグモイドの確率　紫：ロジットの密度　黄：積')
        f = self.equation(r'\mu_a=2', y=-2.45)
        density = always_redraw(lambda: curve(ax, u, normal(u, 2, var.get_value()), PURPLE))
        slider = self.slider(var, .36, 9, [3.5, -2.3, 0], r'\sigma_a^2', PURPLE, 2.4)
        self.beat(Create(density))
        sig = curve(ax, u, sigmoid(u), RED)
        dot = Dot(ax.c2p(2, sigmoid(2)), color=RED, radius=.09)
        self.beat(Create(sig), FadeIn(dot))
        scan = ValueTracker(-7)
        def area():
            xx = np.linspace(-7, max(-6.999, scan.get_value()), 180)
            yy = sigmoid(xx)*normal(xx, 2, var.get_value())
            return Polygon(ax.c2p(xx[0], 0), *[ax.c2p(a,b) for a,b in zip(xx, yy)],
                           ax.c2p(xx[-1], 0), stroke_width=0, fill_color=YELLOW, fill_opacity=.65)
        filled = always_redraw(area)
        self.add(filled)
        self.beat(scan.animate.set_value(11))
        mean = lambda: float(predictive_integral(2, var.get_value()))
        bar = NumberLine(x_range=[0, 1, .5], length=3.4, rotation=PI/2,
                         include_numbers=False, color=MUTED).move_to([5.1, .15, 0])
        avgdot = Dot(color=GREEN, radius=.09).add_updater(lambda m: m.move_to(bar.n2p(mean())))
        mapdot = Dot(bar.n2p(sigmoid(2)), color=RED, radius=.075)
        bar_labels = VGroup(*[tex(str(v), 18, MUTED).next_to(bar.n2p(v), RIGHT, buff=.17)
                              for v in [0, .5, 1]])
        self.add(bar, bar_labels, avgdot, mapdot,
                 readout('P=', mean, [5.1, 2.2, 0], GREEN, 3))
        self.beat(var.animate.set_value(9))
        self.beat(var.animate.set_value(.36))
        self.remove(f)
        f = self.equation(r'p(C_1\mid\phi,\mathbf t)\simeq',
                          r'\int\sigma(a)\mathcal N(a\mid\mu_a,\sigma_a^2)\,da',
                          size=28)
        f[1].set_color(YELLOW)
        self.remove(slider)
        self.add(f)
        self.beat(var.animate.set_value(4), Circumscribe(f, color=YELLOW))

    def approximation(self):
        ax = self.axes((-5, 5, 2), (0, 1, .5), labels=('a', ''))
        u = np.linspace(-5, 5, 221)
        lam = ValueTracker(1)
        red = curve(ax, u, sigmoid(u), RED)
        blue = always_redraw(lambda: curve(ax, u, cdf(lam.get_value()*u), BLUE))
        note = self.note('赤：シグモイド　　青：標準正規 CDF')
        self.add(red)
        f = self.equation(r'\sigma(a)\simeq\Phi(\lambda a)')
        self.beat(Create(blue), lam.animate.set_value(np.sqrt(np.pi/8)), start_sentence=1)
        tangent = curve(ax, np.array([-.9,.9]), np.array([.275,.725]), YELLOW, 5)
        self.remove(f)
        f = self.equation(r"\sigma'(0)=\tfrac14=\lambda/\sqrt{2\pi}",
                          r'\quad\Rightarrow\quad\lambda^2=\pi/8')
        self.beat(Create(tangent), Circumscribe(f, color=YELLOW))
        self.remove(f)
        f = self.equation(r'\int\Phi(\lambda a)\mathcal N(a\mid\mu,v)\,da',
                          r'=\Phi\!\left(\frac{\mu}{\sqrt{\lambda^{-2}+v}}\right)', size=29)
        self.beat(Circumscribe(f, color=BLUE))
        self.remove(f, ax, ax.labels, red, blue, tangent, note)
        ax = self.axes((0, 9, 3), (.5, .9, .1), labels=('v', 'P'), height=3.1)
        vv = np.linspace(0, 9, 151)
        green = curve(ax, vv, sigmoid(2*kappa(vv)), GREEN)
        f = self.equation(r'P\simeq\sigma(', r'\kappa(v)', r'\mu),\qquad',
                          r'\kappa(v)=(1+\pi v/8)^{-1/2}', size=29)
        f[1].set_color(GREEN); f[3].set_color(GREEN)
        note = self.note('平均を 2 に固定：分散を増やすと、確率は 0.5 へ')
        self.beat(Create(green))
        var = ValueTracker(0)
        marker = always_redraw(lambda: Dot(ax.c2p(var.get_value(), sigmoid(2*kappa(var.get_value()))),
                                          color=GREEN, radius=.09))
        nums = readout(r'\kappa=', lambda: kappa(var.get_value()), [3, 1.4, 0], GREEN)
        self.add(marker, nums)
        self.beat(var.animate.set_value(9))
        samples = np.linspace(0, 9, 13)
        numerical = VGroup(*[Dot(ax.c2p(v, predictive_integral(2,v)), color=WHITE, radius=.045)
                             for v in samples])
        self.beat(FadeIn(numerical), var.animate.set_value(0))

    def decision(self):
        ax = self.axes()
        u = np.linspace(-1.7, 3, 221)
        mu, var = stats(u)
        scale = ValueTracker(0)
        base = curve(ax, u, sigmoid(mu), RED, 2.5, .55)
        line = always_redraw(lambda: curve(ax, u, predict(u, scale.get_value()), GREEN))
        self.add(self.data(ax), base, line)
        note = self.note('赤：MAP　　緑：ラプラス近似＋予測積分の近似')
        f = self.equation(r'p(C_1\mid\phi,\mathbf t)\simeq\sigma(\kappa(\sigma_a^2)\mu_a)')
        self.beat(scale.animate.set_value(1), start_sentence=1)
        boundary = -MAP[0]/MAP[1]
        horizontal = DashedLine(ax.c2p(-1.7,.5),ax.c2p(3,.5),color=MUTED)
        vertical = DashedLine(ax.c2p(boundary,0),ax.c2p(boundary,1),color=YELLOW)
        label = tex(r'\mu_a=0', 24, YELLOW).move_to(ax.c2p(boundary,1)+UP*.3)
        self.beat(Create(horizontal), Create(vertical), FadeIn(label))
        threshold = ValueTracker(.5)
        self.remove(horizontal)
        horizontal = always_redraw(lambda: DashedLine(ax.c2p(-1.7,threshold.get_value()),
                                                       ax.c2p(3,threshold.get_value()),color=PURPLE))
        def crossing(bayes):
            values = predict(u) if bayes else sigmoid(mu)
            return float(np.interp(threshold.get_value(), values, u))
        points = always_redraw(lambda: VGroup(*[Dot(ax.c2p(crossing(b),threshold.get_value()),
                                                   color=c,radius=.075) for b,c in [(False,RED),(True,GREEN)]]))
        threshold_label = DecimalNumber(.5, num_decimal_places=1, font_size=24, color=PURPLE)
        threshold_label.add_updater(lambda m: m.set_value(threshold.get_value()).next_to(
            ax.c2p(3, threshold.get_value()), RIGHT, buff=.15))
        self.add(horizontal, points, threshold_label)
        self.beat(threshold.animate.set_value(.8), end_sentence=1)
        self.remove(horizontal, points, threshold_label)
        self.beat(scale.animate.set_value(.05), Indicate(vertical, color=YELLOW))
        self.remove(f)
        f = self.equation(r'w_{\mathrm{MAP}},S_N', r'\longrightarrow',
                          r'\mu_a,\sigma_a^2', r'\longrightarrow',
                          r'\mathbb E[\sigma(a)]')
        f[0].set_color(PURPLE); f[2].set_color(YELLOW); f[4].set_color(GREEN)
        self.beat(scale.animate.set_value(1), Circumscribe(f, color=GREEN))
        cloud = VGroup(*[curve(ax, u, sigmoid(design(u)@v), GREEN, 1.2, .23) for v in SAMPLES])
        self.beat(Create(cloud), ShowPassingFlash(line.copy().clear_updaters().set_stroke(GREEN, 5), time_width=.35))


    def body_card(self, label):
        """Temporarily replace the body; keep the title and the PCM clock."""
        saved = [m for m in self.mobjects if m is not self.subtitle]
        self.clear()
        self.add(*[m for m in saved if m.get_center()[1] > 3])
        frame = RoundedRectangle(width=10.4, height=4.45, corner_radius=.12,
                                 color='#FFFF00', stroke_width=1.2).move_to([0, .1, 0])
        self.add(frame, jp(label, 23).move_to([-4.85, 2.02, 0], aligned_edge=LEFT))
        return saved

    def restore_body(self, saved):
        self.clear()
        self.add(*saved)
        self.subtitle = None

    def regression_recap(self):
        saved = self.body_card('復習: 3.3 候補ごとの予測を平均')
        # 3.3 curves()/candidates(): preserve candidate colors and red mean.
        colors = ['#FF6B77', '#FFE079', '#C29AFF', '#FFB45B', '#58B5ED', '#77D49A']
        left = self.axes((-1, 1, 1), (-3, 3, 1), (-2.8, .0, 0), 3.5, 2.35, ('x', 'y'))
        right = self.axes((-1, 1, 1), (0, 1, .5), (2.6, .0, 0), 3.5, 2.35, ('x', 'p'))
        self.add(jp('説明用の候補：同じ重みの組を使う', 19, MUTED).move_to([0, 1.45, 0]))
        xx = np.linspace(-1, 1, 101)
        values = np.array([b+w*xx for b, w in [(-.8,1.4),(-.3,1.1),(.4,1.8),(.6,.9),(-.5,1.9),(.1,1.3)]])
        candidates = VGroup(*[curve(left, xx, v, c, 1.5, .6) for v,c in zip(values,colors)])
        mean = curve(left, xx, values.mean(axis=0), colors[0], 4)
        probabilities = VGroup(*[curve(right, xx, sigmoid(v), c, 1.5, .6) for v,c in zip(values,colors)])
        average = curve(right, xx, sigmoid(values).mean(axis=0), colors[0], 4)
        arrow = Arrow([-.8,.25,0],[.5,.25,0],color='#FFE079',buff=.08)
        label = tex(r'\sigma',28,'#FFE079').next_to(arrow,UP,buff=.1)
        legend = jp('細線：分布からの候補　太い赤線：平均',20).move_to([0,-1.85,0])
        self.add(candidates, legend)
        first, second = [self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('3.3 prediction mean', first, lambda: Create(mean)),
            ('sigmoid each candidate then average', second*.7,
             lambda: AnimationGroup(GrowArrow(arrow),FadeIn(label),TransformFromCopy(candidates,probabilities))),
            ('mean of class probabilities', second*.3, lambda: Create(average)),
        ])
        self.restore_body(saved)

    def laplace_recap(self):
        saved = self.body_card('復習: 4.4 ラプラス近似')
        # 4.4 curvature(): red quadratic/Gaussian and gold width.
        red, gold = '#FF6B77', '#FFE079'
        left = self.axes((-1.4,1.4,1),(0,4,1),(-2.8,0,0),3.5,2.25,('z','h(z)'))
        right = self.axes((-1.4,1.4,1),(0,.9,.3),(2.65,0,0),3.5,2.25,('z','q(z)'))
        xx = np.linspace(-1.4,1.4,121); a = ValueTracker(1)
        bowl = always_redraw(lambda:curve(left,xx,.5*a.get_value()*xx**2,red))
        bell = always_redraw(lambda:curve(right,xx,normal(xx,0,1/a.get_value()),red))
        width = always_redraw(lambda:Line(right.c2p(-1/np.sqrt(a.get_value()),.12),
                                         right.c2p(1/np.sqrt(a.get_value()),.12),color=gold,stroke_width=4))
        self.add(jp('説明用の一変数：谷底の近く、曲率は正',19,MUTED).move_to([0,1.45,0]),
                 tex(r'h(z)\simeq\tfrac12Az^2',26,red).move_to([-2.7,-1.65,0]),
                 tex(r'\mathrm{Var}[z]=A^{-1}',26,red).move_to([2.6,-1.65,0]))
        mapping = tex(r'A>0\quad\longrightarrow\quad H>0',25,gold).move_to([0,2.6,0])
        first, second = [self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('local quadratic', first, lambda:Create(bowl)),
            ('Gaussian with inverse curvature', second*.45, lambda:AnimationGroup(Create(bell),Create(width))),
            ('curvature and width', second*.3, lambda:a.animate.set_value(4)),
            ('A becomes H', second*.25, lambda:FadeIn(mapping)),
        ])
        self.restore_body(saved)

    def precision_aid(self):
        saved = self.body_card('補足: 観測一つが加える精度')
        blue, yellow, green, purple = '#58C4DD', '#FFFF00', '#83C167', '#9A72AC'
        self.add(jp('説明用の例：特徴 (1, 2)、予測確率 0.5',20,MUTED).move_to([0,1.45,0]))
        column = Matrix([['1'],['2']],v_buff=.65).scale(.65).set_color(blue)
        row = Matrix([['1','2']],h_buff=.85).scale(.65).set_color(purple)
        product = Matrix([['1','2'],['2','4']],h_buff=.85,v_buff=.65).scale(.65).set_color(yellow)
        eq = VGroup(column,tex(r'\times',28),row,tex('=',28),product).arrange(RIGHT,buff=.35).move_to([0,.35,0])
        entries = product.get_entries()
        self.add(column,row,eq[1],eq[3],product.get_brackets())
        factors = [(column.get_entries()[i],row.get_entries()[j]) for i in range(2) for j in range(2)]
        coefficient = tex(r'y(1-y)=0.25',29,yellow).move_to([-2.6,-.95,0])
        weighted = Matrix([['0.25','0.5'],['0.5','1']],h_buff=1.05,v_buff=.65).scale(.65).set_color(green).move_to(product)
        caption = jp('観測の精度',20,green).next_to(weighted,UP,buff=.13)
        result = MathTex(r'S_0^{-1}', '+', r'0.25\phi\phi^{\mathsf T}', '=H',
                         r'\quad\longrightarrow\quad', r'S_N=H^{-1}',font_size=29).move_to([0,-1.8,0])
        result[0].set_color(purple);result[2].set_color(green);result[5].set_color(green)
        first,second,third = [self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('four outer product cells', first,
             lambda:LaggedStart(*[TransformFromCopy(VGroup(*pair),cell) for pair,cell in zip(factors,entries)],lag_ratio=.3)),
            ('coefficient on all cells', second,
             lambda:AnimationGroup(FadeIn(coefficient),Transform(product,weighted),FadeIn(caption))),
            ('add precision then invert', third, lambda:Write(result)),
        ])
        self.restore_body(saved)
