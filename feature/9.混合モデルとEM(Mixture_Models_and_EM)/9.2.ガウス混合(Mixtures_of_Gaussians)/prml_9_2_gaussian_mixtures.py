"""A single one-dimensional experiment motivates responsibilities and EM."""
import json
from pathlib import Path
import numpy as np
from manim import *

from mixture_model import POINTS, PROBE_INDEX, INITIAL, em_history, components, collapsed_example
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex

BG = '#10141F'
RED = ManimColor('#FC6255')
BLUE = ManimColor('#58C4DD')
YELLOW = ManimColor('#FFFF00')
GREEN = ManimColor('#83C167')
PURPLE = ManimColor('#9A72AC')
MUTED = ManimColor('#A8B2C5')
STATES = em_history(20)


def J(message, size=22, color=WHITE):
    return jp(str(message), size, color)


def xcoord(x):
    return -5.15 + (float(x)+3) * 1.03


def ycoord(y):
    return -1.55 + float(y) * 7.0


def point(x, y):
    return np.array([xcoord(x), ycoord(y), 0.])


def density_curve(mu, sigma, pi, which):
    values = np.linspace(-3, 3, 161)
    weighted = components(values, np.asarray(mu), np.asarray(sigma), np.asarray(pi))
    heights = weighted.sum(axis=1) if which == 2 else weighted[:, which]
    color = (RED, BLUE, GREEN)[which]
    return VMobject().set_points_as_corners([point(x, y) for x, y in zip(values, heights)]).set_stroke(color, 3.1)


def curves_for(state):
    return VGroup(*[density_curve(state['mu'], state['sigma'], state['pi'], i) for i in range(3)])


def ink_color(red_share):
    return interpolate_color(BLUE, RED, float(red_share))


def base_plot(state, colored=False):
    xaxis = Line(point(-3, 0), point(3, 0), color=MUTED, stroke_width=2.1)
    yaxis = Line(point(-3, 0), point(-3, .49), color=MUTED, stroke_width=2.1)
    labels = VGroup(J('密度', 17, MUTED).move_to([-5.1, 2.18, 0]),
                    J('観測16点', 17, MUTED).move_to([-4.55, -2.82, 0]))
    for x in (-2, 0, 2):
        xx = xcoord(x)
        labels.add(Line([xx, -1.55, 0], [xx, -1.65, 0], color=MUTED, stroke_width=1.4))
        labels.add(J(str(x), 15, MUTED).move_to([xx, -1.9, 0]))
    dots = VGroup(*[Dot([xcoord(x), -2.45, 0], radius=.065,
                        color=ink_color(state['gamma'][i, 0]) if colored else WHITE)
                    for i, x in enumerate(POINTS)])
    dots[PROBE_INDEX].set_z_index(3)
    probe = Circle(radius=.155, color=YELLOW, stroke_width=2.1).move_to(dots[PROBE_INDEX]).set_z_index(2)
    return VGroup(xaxis, yaxis, labels), dots, probe


def bars(red_share, raw=False):
    if raw:
        q = components([POINTS[PROBE_INDEX]], INITIAL['mu'], INITIAL['sigma'], INITIAL['pi'])[0]
        amounts = q
        scale = 16
        texts = [f'{q[0]:.3f}', f'{q[1]:.3f}']
    else:
        amounts = np.array([red_share, 1-red_share])
        scale = 3.1
        texts = [f'{amounts[0]:.3f}', f'{amounts[1]:.3f}']
    out = VGroup()
    for i, (amount, color) in enumerate(zip(amounts, (RED, BLUE))):
        y = .35 - .61*i
        label = J('赤' if i == 0 else '青', 22, color).move_to([1.75, y, 0])
        bar = Rectangle(width=float(amount)*scale, height=.22, stroke_width=0,
                        fill_color=color, fill_opacity=.85).move_to([2.12+float(amount)*scale/2, y, 0])
        value = J(texts[i], 18, WHITE).next_to(bar, RIGHT, buff=.12)
        out.add(label, bar, value)
    return out


def recap_23_plot():
    """Only the two weighted hills and their sum from 2.3, redrawn locally."""
    parameters = dict(mu=np.array([-1.78034723, 1.57241301]),
                      sigma=np.array([.48746931, .72204252]),
                      pi=np.array([.44937729, .55062271]))
    xs = np.linspace(-3.6, 3.6, 101)
    q = components(xs, parameters['mu'], parameters['sigma'], parameters['pi'])
    base = Line([2.04, 1.04, 0], [5.76, 1.04, 0], color=MUTED, stroke_width=1.3)
    colors = (ManimColor('#58B5ED'), ManimColor('#FFE079'), ManimColor('#C29AFF'))
    curves = VGroup()
    for i in range(3):
        yy = q.sum(axis=1) if i == 2 else q[:, i]
        coords = [np.array([2.04+(x+3.6)*3.72/7.2, 1.04+y*2.75, 0]) for x, y in zip(xs, yy)]
        curves.add(VMobject().set_points_as_corners(coords).set_stroke(colors[i], 2.2))
    return VGroup(base, curves)


def parameter_counters(trackers):
    """Numeric glyphs are redrawn at each value, avoiding distorted digit morphs."""
    rows = VGroup()
    for i, (key, symbol) in enumerate((('mu', r'\mu='), ('sigma', r'\sigma='), ('pi', r'\pi='))):
        y = .53-.52*i
        first, second = trackers[key]
        rows.add(VGroup(
            tex(symbol, 25, WHITE).move_to([2.38, y, 0]),
            always_redraw(lambda v=first, yy=y: DecimalNumber(v.get_value(), num_decimal_places=3,
                          font_size=25, color=RED).move_to([3.45, yy, 0])),
            J(' / ', 20, MUTED).move_to([4.25, y, 0]),
            always_redraw(lambda v=second, yy=y: DecimalNumber(v.get_value(), num_decimal_places=3,
                          font_size=25, color=BLUE).move_to([5.05, yy, 0])),
        ))
    return rows


class PRML92GaussianMixtures(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.manifest = {e['id']: e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline = []
        for i, method in enumerate((self.opening, self.two_hills, self.latent,
                                    self.likelihood, self.e_step, self.m_step,
                                    self.iteration, self.return_to_probe)):
            self.begin(i)
            self.note = None
            self.formula = None
            self.value = None
            self.bars = None
            method()
            assert self.beat_index == len(SCENES[i]['beats'])
            self.timeline[-1]['end'] = float(self.time)
        (Path(config.media_dir)/'prml92_timeline.json').write_text(
            json.dumps(self.timeline, ensure_ascii=False, indent=2)+'\n')

    def setup_plot(self, state=STATES[0], colored=False, show_curves=True):
        axes, self.dots, self.probe = base_plot(state, colored)
        self.curves = curves_for(state)
        if not show_curves:
            self.curves.set_stroke(opacity=0)
        self.add(axes, self.curves, self.dots)
        self.current_state = state
        return axes

    def note_anim(self, message, color=MUTED):
        mob = J(message, 21, color)
        if mob.width > 4.45:
            mob.scale_to_fit_width(4.45)
        mob.move_to([3.78, -2.42, 0])
        if self.note is not None:
            self.remove(self.note)
        self.note = mob
        return FadeIn(mob, rate_func=lambda t: smooth(min(1, 5*t)))

    def show_formula(self, formula, color=WHITE):
        mob = tex(formula, 29, color)
        if mob.width > 4.5:
            mob.scale_to_fit_width(4.5)
        if mob.height > 1.05:
            mob.scale_to_fit_height(1.05)
        mob.move_to([3.78, 1.72, 0])
        if self.formula is not None:
            self.remove(self.formula)
        self.formula = mob
        return FadeIn(mob)

    def show_value(self, label, color=GREEN, math=False):
        mob = tex(label, 27, color) if math else J(label, 21, color)
        if mob.width > 4.45:
            mob.scale_to_fit_width(4.45)
        mob.move_to([3.78, .80, 0])
        if self.value is not None:
            self.remove(self.value)
        self.value = mob
        return FadeIn(mob)

    def show_bars(self, red_share, raw=False):
        target = bars(red_share, raw)
        if self.bars is None:
            self.bars = target
            return FadeIn(target)
        return Transform(self.bars, target)

    def say(self, message, *animations, color=MUTED):
        self.beat(self.note_anim(message, color), *animations)

    def animate_params(self, state, key):
        row = {'mu': 0, 'sigma': 1, 'pi': 2}[key]
        return AnimationGroup(
            self.param_trackers[key][0].animate.set_value(float(state[key][0])),
            self.param_trackers[key][1].animate.set_value(float(state[key][1])),
            lag_ratio=0,
        )

    def update_curves(self, state, recolor=True):
        self.current_state = state
        animations = [Transform(self.curves, curves_for(state))]
        if recolor:
            animations += [dot.animate.set_color(ink_color(state['gamma'][i, 0]))
                           for i, dot in enumerate(self.dots)]
        return AnimationGroup(*animations, lag_ratio=0)

    def opening(self):
        self.setup_plot(show_curves=False)
        boundary = DashedLine([xcoord(0), -1.52, 0], [xcoord(0), 1.9, 0],
                              color=PURPLE, dash_length=.12, stroke_width=2)
        mean_dots = VGroup(Dot(point(-1.1, 0), color=RED, radius=.095),
                           Dot(point(1.1, 0), color=BLUE, radius=.095))
        self.say('9.1 の硬い境界', FadeIn(boundary), FadeIn(mean_dots))
        self.say('新しい1次元の16点', Indicate(self.dots, color=YELLOW, scale_factor=1.02))
        self.say('黄色い輪の点に注目', FadeIn(self.probe))
        distances = VGroup(Line([xcoord(-1.1), -2.18, 0], [xcoord(.15), -2.18, 0], color=RED, stroke_width=2),
                           Line([xcoord(.15), -2.05, 0], [xcoord(1.1), -2.05, 0], color=BLUE, stroke_width=2))
        self.say('中心まで 赤1.25・青0.95', FadeIn(distances))
        self.say('青が勝つ？', Indicate(self.probe, color=YELLOW, scale_factor=1.2))
        self.curves.set_stroke(opacity=1)
        self.say('赤0.564・青0.436', FadeIn(self.curves), self.show_bars(STATES[0]['gamma'][PROBE_INDEX, 0]))
        self.say('距離だけでは決まらない', self.show_formula(r'\gamma_1=0.564>\gamma_2=0.436', YELLOW))
        self.say('最後に同じ点へ戻る', Indicate(self.probe, color=YELLOW, scale_factor=1.2))

    def two_hills(self):
        self.setup_plot()
        recap = J('復習: 2.3 ガウス混合', 20, WHITE).move_to([3.77, 2.55, 0])
        recap_box = SurroundingRectangle(recap, color=YELLOW, buff=.10, stroke_width=1.8)
        recap_hills = recap_23_plot()
        self.say('2.3 で見た2つの山', FadeIn(recap), FadeIn(recap_box), FadeIn(recap_hills),
                 Indicate(self.curves[0], color=YELLOW, scale_factor=1))
        self.say('赤い山の中心と幅', FadeOut(recap_hills), self.show_formula(r'\mu_1=-1.1,\quad\sigma_1=0.9', RED),
                 Indicate(self.curves[0], color=RED, scale_factor=1))
        self.say('青い山の中心と幅', self.show_formula(r'\mu_2=1.1,\quad\sigma_2=1.2', BLUE),
                 Indicate(self.curves[1], color=BLUE, scale_factor=1))
        self.say('混合係数は合計1', self.show_formula(r'\pi_1=0.65,\quad\pi_2=0.35', YELLOW),
                 FadeOut(recap), FadeOut(recap_box))
        self.say('同じ位置で高さを足す', Indicate(self.curves[2], color=GREEN, scale_factor=1))
        self.say('緑がガウス混合の密度', self.show_formula(r'p(x)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)', GREEN))
        self.say('0.15 の2本の高さ', FadeIn(self.probe), self.show_bars(STATES[0]['gamma'][PROBE_INDEX, 0], raw=True))
        self.say('見えない成分番号へ', Indicate(self.probe, color=YELLOW, scale_factor=1.2))

    def latent(self):
        self.setup_plot()
        self.add(self.probe)
        self.say('点の裏に隠れた番号', self.show_formula(r'z\in\{(1,0),(0,1)\}', YELLOW))
        self.say('まず山を選ぶ', self.show_formula(r'p(z_k=1)=\pi_k', PURPLE))
        self.say('次に点を出す', self.show_formula(r'p(x\mid z_k=1)=\mathcal N(x\mid\mu_k,\Sigma_k)', BLUE))
        self.say('観測点から逆向きに推測', Indicate(self.probe, color=YELLOW, scale_factor=1.2))
        self.say('2本の重み付き密度', self.show_bars(STATES[0]['gamma'][PROBE_INDEX, 0], raw=True))
        self.say('合計を1に正規化', self.show_bars(STATES[0]['gamma'][PROBE_INDEX, 0]),
                 self.show_formula(r'\gamma_{nk}=\frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_n\mid\mu_j,\Sigma_j)}', GREEN))
        self.say('事後確率＝負担率', self.show_value('赤56.4%・青43.6%', YELLOW))
        self.say('今度は全16点へ', Indicate(self.dots, color=YELLOW, scale_factor=1.02))

    def likelihood(self):
        self.setup_plot()
        self.say('全16点の緑の密度', Indicate(self.curves[2], color=GREEN, scale_factor=1))
        examples = VGroup()
        weighted = components(POINTS, INITIAL['mu'], INITIAL['sigma'], INITIAL['pi']).sum(axis=1)
        for i in (0, PROBE_INDEX):
            x = POINTS[i]
            examples.add(Line(point(x, 0), point(x, weighted[i]), color=YELLOW, stroke_width=2.5))
        self.say('2点の例：積を対数で和へ', FadeIn(examples),
                 self.show_formula(r'\ln[p(x_1)p(x_9)]=\ln p(x_1)+\ln p(x_9)', YELLOW))
        self.say('内側で山を足す', FadeOut(examples), self.show_formula(r'\ell=\sum_n\ln p(x_n)', GREEN))
        self.say('初期の対数尤度', self.show_value(r'\ell=-27.737', GREEN, math=True))
        separate = dict(mu=np.array([.15, 0.]), sigma=np.array([.03, 1.55]),
                        pi=np.array([.1, .9]))
        broad_blue = density_curve(separate['mu'], separate['sigma'], separate['pi'], 1)
        self.say('別の極限操作', Succession(FadeOut(self.curves), FadeIn(broad_blue)),
                 self.show_formula(r'\sigma_1\rightarrow 0', RED),
                 self.show_value('別の設定：赤を0.15に固定', RED), FadeIn(self.probe))
        spike = Arrow([xcoord(.15), -1.4, 0], [xcoord(.15), 1.95, 0], color=RED,
                      stroke_width=4, buff=0, max_tip_length_to_length_ratio=.10)
        self.say('1点で密度が上昇', FadeIn(spike),
                 self.show_value('幅0.005：密度8.209', RED))
        self.say('潰れた解に注意', self.show_formula(r'\sigma\rightarrow0:\quad\ell\rightarrow+\infty', YELLOW),
                 Indicate(spike, color=YELLOW, scale_factor=1))
        self.say('通常の山へ戻る', Succession(AnimationGroup(FadeOut(spike), FadeOut(broad_blue)),
                                                  FadeIn(self.curves)),
                 self.show_formula(r'\sigma_1=0.9,\quad\sigma_2=1.2', BLUE))

    def e_step(self):
        self.setup_plot()
        self.say('山を固定して色を計算', self.show_formula(r'\gamma_{nk}=p(z_{nk}=1\mid x_n)', GREEN))
        self.say('左端はほぼ赤', self.dots[0].animate.set_color(ink_color(STATES[0]['gamma'][0, 0])),
                 self.dots[1].animate.set_color(ink_color(STATES[0]['gamma'][1, 0])))
        mid = range(6, 11)
        self.say('真ん中は混色', *[self.dots[i].animate.set_color(ink_color(STATES[0]['gamma'][i, 0])) for i in mid])
        self.say('右端はほぼ青', *[self.dots[i].animate.set_color(ink_color(STATES[0]['gamma'][i, 0])) for i in (14, 15)])
        rest = [i for i in range(16) if i not in (0, 1, 14, 15, *mid)]
        self.say('全点の負担率を表示', *[self.dots[i].animate.set_color(ink_color(STATES[0]['gamma'][i, 0])) for i in rest])
        hard = DashedLine([xcoord(0), -1.5, 0], [xcoord(0), 1.8, 0], color=PURPLE,
                          stroke_width=2, dash_length=.12)
        self.say('硬い境界との違い', FadeIn(hard))
        self.say('Eでは山は固定', self.show_value('点の色だけが変わる', GREEN))
        self.say('濃いインクの重心は？', FadeOut(hard), FadeIn(self.probe),
                 Indicate(self.curves[0], color=RED, scale_factor=1))

    def m_step(self):
        self.setup_plot(colored=True)
        self.param_trackers = {key: [ValueTracker(float(value)) for value in STATES[0][key]]
                               for key in ('mu', 'sigma', 'pi')}
        self.params = parameter_counters(self.param_trackers)
        self.add(self.params)
        start = STATES[0]
        end = STATES[1]
        stage_mu = dict(mu=end['mu'], sigma=start['sigma'], pi=start['pi'], gamma=start['gamma'])
        stage_sigma = dict(mu=end['mu'], sigma=end['sigma'], pi=start['pi'], gamma=start['gamma'])
        self.say('負担率を重りにする', self.show_formula(r'N_k=\sum_n\gamma_{nk}', YELLOW))
        self.say('濃い点ほど強く引く', self.show_formula(r'\mu_k=\frac{\sum_n\gamma_{nk}x_n}{N_k}', GREEN))
        self.say('中心を重心へ動かす', self.update_curves(stage_mu, recolor=False),
                 self.animate_params(stage_mu, 'mu'))
        self.say('新しい中心からのずれ', Indicate(self.curves[0], color=RED, scale_factor=1))
        self.say('広がりも更新', self.update_curves(stage_sigma, recolor=False),
                 self.animate_params(stage_sigma, 'sigma'), self.show_formula(r'\sigma_k^2=\frac{\sum_n\gamma_{nk}(x_n-\mu_k)^2}{N_k}', GREEN))
        self.say('負担率の合計が混合比', self.update_curves(end, recolor=False),
                 self.animate_params(end, 'pi'), self.show_formula(r'\pi_k=\frac{N_k}{N}', YELLOW))
        self.say('3つのつまみを更新', Indicate(self.params, color=YELLOW, scale_factor=1.02))
        self.say('新しい山でもう1度E', self.update_curves(end, recolor=True),
                 FadeIn(self.probe))

    def trace_mob(self, upto):
        indexes = [0, 1, 2, 5, 20]
        xx = [2.05+float(i)*.91 for i in range(5)]
        values = [STATES[i]['log_likelihood'] for i in indexes]
        yy = [-.95 + (v-values[0])*.60 for v in values]
        n = indexes.index(upto)+1
        path = VGroup(*[Line([xx[j], yy[j], 0], [xx[j+1], yy[j+1], 0], color=GREEN, stroke_width=2.5)
                        for j in range(n-1)])
        dots = VGroup(*[Dot([xx[j], yy[j], 0], radius=.07, color=YELLOW) for j in range(n)])
        labels = VGroup(*[J(str(indexes[j]), 16, MUTED).move_to([xx[j], -1.33, 0]) for j in range(5)])
        title = J('抜粋した往復 →', 17, MUTED).move_to([4.15, -1.62, 0])
        return VGroup(path, dots, labels, title)

    def iteration(self):
        self.setup_plot(colored=True)
        self.trace = self.trace_mob(0)
        self.add(self.trace)
        self.say('1往復目', self.update_curves(STATES[1]), self.show_value(r'\ell=-26.521', GREEN, math=True),
                 Transform(self.trace, self.trace_mob(1)))
        self.say('2往復目', self.update_curves(STATES[2]), self.show_value(r'\ell=-26.293', GREEN, math=True),
                 Transform(self.trace, self.trace_mob(2)))
        self.say('5往復目', self.update_curves(STATES[5]), self.show_value(r'\ell=-26.145', GREEN, math=True),
                 Transform(self.trace, self.trace_mob(5)))
        self.say('20往復目', self.update_curves(STATES[20]), self.show_value(r'\ell=-26.141', GREEN, math=True),
                 Transform(self.trace, self.trace_mob(20)))
        self.say('色を求めて山を更新', self.show_formula(r'E:\gamma\quad\longrightarrow\quad M:\mu,\Sigma,\pi', YELLOW))
        self.say('対数尤度は下がらない', Indicate(self.trace, color=YELLOW, scale_factor=1))
        self.say('初期値と特異点には注意', self.show_value('局所解・成分の潰れ', RED))
        self.say('最初の0.15へ戻る', FadeIn(self.probe), Indicate(self.probe, color=YELLOW, scale_factor=1.2))

    def return_to_probe(self):
        self.setup_plot(colored=True)
        self.add(self.probe)
        self.say('同じ16点・同じ初期の山', self.show_formula(r'\mu=(-1.1,1.1)', WHITE))
        self.say('距離は青の方が短い', self.show_value('赤1.25・青0.95', BLUE))
        self.say('重みと幅は違う', self.show_formula(r'\pi=(0.65,0.35),\quad\sigma=(0.9,1.2)', YELLOW))
        self.say('この位置の山の高さ', self.show_bars(STATES[0]['gamma'][PROBE_INDEX, 0], raw=True))
        self.say('赤が0.564で優勢', self.show_bars(STATES[0]['gamma'][PROBE_INDEX, 0]),
                 self.show_formula(r'\gamma_1=0.564,\quad\gamma_2=0.436', GREEN))
        self.say('負担率で山を作り直す', Indicate(self.curves, color=YELLOW, scale_factor=1))
        self.say('局所解と特異点に注意', self.show_value('対数尤度と幅を確認', RED))
        self.say('9.3でEMの一般形へ', Indicate(self.probe, color=YELLOW, scale_factor=1.2))
