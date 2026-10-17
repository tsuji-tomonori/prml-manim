"""One visual investigation of a narrow factorized posterior."""
import json
from pathlib import Path
import numpy as np
from manim import *

from variational_model import (COVARIANCE, INITIAL_MEAN, FACTOR_VARIANCE,
                               contour, coordinate_steps, kl_qp, lower_bound,
                               q_covariance)
from make_voicevox_narration import MANIFEST
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex

BG = '#10141F'
BLUE = ManimColor('#58C4DD')
GREEN = ManimColor('#83C167')
YELLOW = ManimColor('#FFFF00')
PURPLE = ManimColor('#9A72AC')
RED = ManimColor('#FC6255')
MUTED = ManimColor('#A8B2C5')
ROWS = coordinate_steps(6)


def J(string, size=20, color=WHITE):
    return jp(str(string), size, color)


def quick(alpha):
    return smooth(min(1, 11 * alpha))


def xy(x, y):
    return np.array([-2.55 + .72 * x, -.05 + .69 * y, 0])


def curve(points, color, width=2.7, opacity=1):
    return VMobject().set_points_as_corners(points).set_stroke(color, width, opacity=opacity)


def contours(mean, covariance, color):
    return VGroup(*[curve([xy(*point) for point in contour(mean, covariance, r)], color,
                          2.4 if r == 1.95 else 1.8, .95 if r == 1.95 else .67)
                    for r in (.75, 1.35, 1.95)])


def ellipse_axes():
    group = VGroup(Line(xy(-4.2, 0), xy(4.2, 0), color=MUTED, stroke_width=1.3),
                   Line(xy(0, -3.15), xy(0, 3.15), color=MUTED, stroke_width=1.3))
    for value in (-2, 0, 2):
        group.add(Line(xy(value, 0) + DOWN * .07, xy(value, 0) + UP * .07,
                       color=MUTED, stroke_width=1.0))
        if value:
            group.add(J(str(value), 14, MUTED).move_to(xy(value, 0) + DOWN * .23))
    group.add(J('0', 14, MUTED).move_to(xy(0, 0) + DOWN * .23 + LEFT * .14),
              tex('z_1', 24, MUTED).move_to(xy(4.2, 0) + RIGHT * .17),
              tex('z_2', 24, MUTED).move_to(xy(0, 3.15) + UP * .17),
              J('緑: 事後分布', 18, GREEN).move_to([-4.35, 2.55, 0]),
              J('青: 因子化した近似', 18, BLUE).move_to([-1.35, 2.55, 0]))
    return group


def bound_figure(mean, covariance=None):
    lower = lower_bound(mean, covariance)
    evidence = np.log(.2)
    def height(value):
        return -1.35 + 1.20 * (value + 3.12)
    group = VGroup(Line([1.75, -1.50, 0], [6.08, -1.50, 0],
                        color=MUTED, stroke_width=1.4))
    for x, value, color, name in ((2.83, lower, BLUE, '下界'),
                                  (5.03, evidence, GREEN, '証拠')):
        h = height(value)
        group.add(Line([x, -1.50, 0], [x, h, 0], color=color,
                       stroke_width=10, stroke_opacity=.45),
                  Line([x-.67, h, 0], [x+.67, h, 0], color=color, stroke_width=4),
                  J(name, 18, color).move_to([x, -1.84, 0]),
                  J(f'{value:.3f}', 18, color).move_to([x, h+.26, 0]))
    group.figure_kind = 'bound'
    return group


def recap_bars(aligned=False):
    source = (.5, .25, .125, .125)
    other = source if aligned else (.1, .2, .3, .4)
    group = VGroup(Rectangle(width=4.58, height=3.37, stroke_color=YELLOW,
                             stroke_width=1.1, fill_opacity=0).move_to([3.9, -.03, 0]),
                   J('復習: 1.6  確率のずれ', 18, YELLOW).move_to([3.9, 1.39, 0]),
                   J('青: 基準  橙: 比較', 17, MUTED).move_to([3.9, -1.45, 0]))
    for i, (a, b) in enumerate(zip(source, other)):
        x = 2.42 + i * .98
        base = -1.05
        first = Rectangle(width=.53, height=2.15*a, stroke_width=0,
                          fill_color=BLUE, fill_opacity=.82).move_to([x, base+2.15*a/2, 0])
        second = Rectangle(width=.57, height=2.15*b, stroke_color=ManimColor('#FFB45B'),
                           stroke_width=2, fill_opacity=0).move_to([x, base+2.15*b/2, 0])
        group.add(first, second)
    group.figure_kind = 'recap'
    return group


def recap_card(label, figure):
    card = VGroup(Rectangle(width=4.70, height=4.26, stroke_color=YELLOW,
                            stroke_width=1.1, fill_opacity=0).move_to([3.9, .08, 0]),
                  J(label, 18, WHITE).move_to([3.9, 1.88, 0]), figure)
    card.figure_kind = 'recap'
    return card


def factor_figure(expectation=False):
    group = VGroup()
    if expectation:
        group.add(J('説明用: 正の重み', 19, YELLOW).move_to([3.9, 1.35, 0]))
        for x, label, color in ((2.95, '0', BLUE), (4.95, '1', PURPLE)):
            group.add(Dot([x, .1, 0], radius=.12, color=color),
                      J(label, 20, color).move_to([x, -.25, 0]),
                      J('0.5', 17, YELLOW).move_to([x, .46, 0]))
        group.add(Line([2.95, -.83, 0], [4.95, -.83, 0], color=MUTED, stroke_width=2),
                  Dot([3.95, -.83, 0], radius=.1, color=GREEN),
                  J('平均 0.5', 18, GREEN).move_to([3.95, -1.28, 0]))
    else:
        for y, color, label in ((.55, BLUE, 'z₁ の平均'), (-.45, PURPLE, 'z₂ の平均')):
            group.add(J(label, 18, color).move_to([2.45, y, 0]),
                      Line([3.22, y, 0], [5.65, y, 0], color=color, stroke_width=11),
                      Dot([4.45, y, 0], radius=.11, color=YELLOW))
        group.add(J('片方を固定して、もう片方を更新', 18, MUTED).move_to([3.95, -1.30, 0]))
    group.figure_kind = 'factor'
    return group


def history_figure(upto):
    group = VGroup(Line([2.05, -1.42, 0], [6.38, -1.42, 0], color=MUTED,
                        stroke_width=1.4),
                   J('更新', 17, MUTED).move_to([5.85, -1.68, 0]))
    def coord(i):
        return np.array([2.22+.50*i, -1.25+1.27*(ROWS[i]['lower']+3.1), 0])
    if upto > 0:
        group.add(curve([coord(i) for i in range(upto+1)], BLUE, 3))
    for i in range(upto+1):
        group.add(Dot(coord(i), radius=.085, color=YELLOW if i==upto else BLUE))
    group.add(J(f'下界 {ROWS[upto]["lower"]:.3f}', 19, BLUE).move_to([4.0, 1.25, 0]))
    group.figure_kind = 'history'
    return group


def width_figure(reverse=False, expanded=False):
    group = VGroup()
    xleft, ybase = 1.85, -1.22
    def pt(x, value):
        return np.array([3.92 + .95*x, ybase + 3.0*value, 0])
    xs = np.linspace(-2.15, 2.15, 101)
    def normal(xs, variance):
        return np.exp(-xs*xs/(2*variance))/np.sqrt(2*np.pi*variance)
    group.add(Line(pt(-2.15, 0), pt(2.15, 0), color=MUTED, stroke_width=1.4),
              J('片方の値', 17, MUTED).move_to([3.92, -1.63, 0]))
    for x in (-2, 0, 2):
        group.add(Line(pt(x, 0)+DOWN*.06,pt(x,0)+UP*.06,color=MUTED,stroke_width=1),
                  J(str(x), 15, MUTED).move_to(pt(x, 0)+DOWN*.24))
    group.add(curve([pt(x, y) for x, y in zip(xs, normal(xs, COVARIANCE[0,0]))], GREEN, 3),
              curve([pt(x, y) for x, y in zip(xs, normal(xs, COVARIANCE[0,0] if reverse or expanded else FACTOR_VARIANCE))],
                    PURPLE if reverse else BLUE, 3),
              J('緑: 周辺分布', 17, GREEN).move_to([2.78, 1.36, 0]),
              J('紫: 逆向き' if reverse else ('青: 広い分布族' if expanded else '青: 変分因子'),
                17, PURPLE if reverse else BLUE).move_to([5.05, 1.36, 0]))
    group.figure_kind = 'width'
    return group


def bimodal_figure(mode='wide'):
    def pt(x, value):
        return np.array([1.52*x, -1.1+4.6*value, 0])
    xs=np.linspace(-3.55,3.55,151)
    def normal(x, mu, sigma):
        return np.exp(-.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    true=.5*normal(xs,-1.35,.49)+.5*normal(xs,1.35,.49)
    if mode=='left':
        approx=normal(xs,-1.35,.49)
    elif mode=='right':
        approx=normal(xs,1.35,.49)
    else:
        approx=normal(xs,0,1.55)
    group=VGroup(Line(pt(-3.55,0),pt(3.55,0),color=MUTED,stroke_width=1.5),
                 curve([pt(x,y) for x,y in zip(xs,true)],GREEN,3),
                 curve([pt(x,y) for x,y in zip(xs,approx)],BLUE,3),
                 J('模式図: 2つの峰',19,YELLOW).move_to([-3.95,2.40,0]),
                 J('緑: 対象  青: 近似',18,MUTED).move_to([3.9,-1.72,0]))
    group.figure_kind='bimodal'
    return group


class PRML101VariationalInference(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        methods=(self.opening,self.bound,self.factorized,self.iterate,
                 self.width,self.direction,self.family,self.return_to_start)
        for i,method in enumerate(methods):
            self.begin(i)
            self.note=self.formula=self.figure=None
            self.setup_plot()
            method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml101_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def setup_plot(self):
        self.axis=ellipse_axes()
        self.true=contours([0,0],COVARIANCE,GREEN)
        self.approx=contours([0,0],q_covariance(),BLUE)
        self.center=Dot(xy(0,0),radius=.09,color=YELLOW)
        self.add(self.axis,self.true,self.approx,self.center)

    def setup_bimodal(self):
        self.axis=self.true=self.approx=self.center=None
        self.left=bimodal_figure('wide')
        self.add(self.left)

    def set_bimodal(self,mode):
        if not hasattr(self,'left') or self.left is None:
            self.left=bimodal_figure(mode)
            self.remove(self.axis,self.true,self.approx,self.center)
            return FadeIn(self.left,rate_func=quick)
        return Transform(self.left,bimodal_figure(mode))

    def clear_figure(self):
        old=self.figure
        self.figure=None
        return FadeOut(old,rate_func=quick)

    def back_to_ellipse(self):
        previous=self.left
        self.axis=ellipse_axes()
        self.true=contours([0,0],COVARIANCE,GREEN)
        self.approx=contours([0,0],q_covariance(),BLUE)
        self.center=Dot(xy(0,0),radius=.09,color=YELLOW)
        self.left=None
        self.remove(previous)
        return FadeIn(VGroup(self.axis,self.true,self.approx,self.center),rate_func=quick)

    def set_note(self,string,color=MUTED):
        mob=J(string,20,color)
        if mob.width>4.85:mob.scale_to_fit_width(4.85)
        mob.move_to([3.9,-2.43,0])
        if self.note is not None:self.remove(self.note)
        self.note=mob
        return FadeIn(mob,rate_func=quick)

    def set_formula(self,string):
        mob=tex(string,32)
        if mob.width>4.85:mob.scale_to_fit_width(4.85)
        if mob.height>.78:mob.scale_to_fit_height(.78)
        mob.move_to([3.90,2.38,0])
        if self.formula is not None:self.remove(self.formula)
        self.formula=mob
        return FadeIn(mob,rate_func=quick)

    def set_figure(self,mob):
        if self.figure is None:
            self.figure=mob
            return FadeIn(mob,rate_func=quick)
        old=self.figure
        self.remove(old)
        self.figure=mob
        return FadeIn(mob,rate_func=quick)

    def place_mean(self,mean):
        self.approx.become(contours(mean,q_covariance(),BLUE))
        self.center.move_to(xy(*mean))

    def set_mean(self,mean,covariance=None):
        mean=np.asarray(mean)
        covariance=q_covariance() if covariance is None else covariance
        return AnimationGroup(Transform(self.approx,contours(mean,covariance,BLUE)),
                              self.center.animate.move_to(xy(*mean)))

    def say(self,note,*moves,formula=None,figure=None,color=MUTED):
        animations=[self.set_note(note,color)]
        if formula is not None:animations.append(self.set_formula(formula))
        if figure is not None:animations.append(self.set_figure(figure))
        animations.extend(moves)
        self.beat(*animations)

    def opening(self):
        # Keep the proposed approximation unseen until after the viewer's guess.
        self.remove(self.approx)
        self.say('9.4 の問いを受け継ぐ',
                 figure=recap_card('復習: 9.4  下界',bound_figure([0,0])),color=YELLOW)
        self.say('2つの未知量が連動する',Indicate(self.true,color=YELLOW,scale_factor=1.01),
                 self.clear_figure())
        self.say('独立な近似の幅を予想',formula=r'q(z_1,z_2)=q_1(z_1)q_2(z_2)')
        self.say('中心は同じ、分散は細い',FadeIn(self.approx),figure=width_figure(),
                 formula=r'1.389\longrightarrow0.500',color=RED)
        self.say('観測を増やしていない',Indicate(self.approx,color=YELLOW,scale_factor=1.01))
        self.say('なぜ細い確信になる？',figure=width_figure(),color=YELLOW)
        self.say('緑が事後、青が近似',formula=r'p(Z|X)\quad q(Z)')

    def bound(self):
        self.place_mean(INITIAL_MEAN)
        self.say('説明用の証拠 0.2',figure=bound_figure(INITIAL_MEAN),formula=r'p(X)=0.2')
        self.say('下界と KL の隙間',formula=r'\ln p(X)=\mathcal L(q)+\mathrm{KL}(q\|p)')
        self.remove(self.figure)
        self.figure=recap_bars(True)
        self.add(self.figure)
        self.say('復習: 1.6 の確率棒',
                 Succession(Wait(.42),Transform(self.figure,recap_bars(),run_time=.58)),
                 formula=r'\mathrm{KL}(q\|p)\geq0',color=YELLOW)
        self.say('下界の中身を平均で読む',figure=bound_figure(INITIAL_MEAN),
                 formula=r'\mathcal L=E_q[\ln p(X,Z)]-E_q[\ln q(Z)]')
        self.say('初期の下界は −3.090',figure=bound_figure(INITIAL_MEAN),
                 formula=r'-3.090+1.481=-1.609')
        self.say('証拠は固定、下界が動く',self.set_mean([.64,.8]),
                 figure=bound_figure([.64,.8]))
        self.say('緑を選べないから制約を置く',self.set_mean(INITIAL_MEAN),
                 figure=bound_figure(INITIAL_MEAN),
                 formula=r'\max_{q\in\mathcal Q}\mathcal L(q)')

    def factorized(self):
        self.place_mean(INITIAL_MEAN)
        self.say('因子化した分布を使う',figure=factor_figure(),
                 formula=r'q(z_1,z_2)=q_1(z_1)q_2(z_2)')
        self.say('斜めの相関を表せない',Indicate(self.true,color=YELLOW,scale_factor=1.01))
        self.say('相手を固定して片方を更新',Indicate(self.center,color=YELLOW,scale_factor=1.2))
        self.say('期待値 = 重み付き平均',figure=factor_figure(True),
                 formula=r'0\cdot0.5+1\cdot0.5=0.5')
        self.say('対数の同時分布を相手で平均',figure=factor_figure(),
                 formula=r'\ln q_1^*=E_{q_2}[\ln p(X,Z)]+C')
        self.say('この例は相手の平均の 0.8 倍',self.set_mean([.64,.8]),
                 formula=r'm_1\leftarrow0.8m_2,\quad V_1=0.5')
        self.say('次は受け渡しを繰り返す',self.set_mean(INITIAL_MEAN),
                 figure=history_figure(0))

    def iterate(self):
        self.place_mean(INITIAL_MEAN)
        self.say('初期平均は 1.5 と 0.8',figure=history_figure(0),
                 formula=r'(m_1,m_2)=(1.5,0.8)')
        self.say('まず 1番を 0.64 へ',self.set_mean(ROWS[1]['mean']),
                 figure=history_figure(1),formula=r'm_1\leftarrow0.8m_2=0.64')
        self.say('次に 2番を 0.512 へ',self.set_mean(ROWS[2]['mean']),
                 figure=history_figure(2),formula=r'm_2\leftarrow0.8m_1=0.512')
        self.say('下界が 1歩ずつ上がる',self.set_mean(ROWS[4]['mean']),
                 figure=history_figure(4),formula=r'\mathcal L(q)\nearrow')
        self.say('中心は 0 に近づく',self.set_mean(ROWS[8]['mean']),figure=history_figure(8),
                 formula=r'(m_1,m_2)\longrightarrow(0,0)')
        self.place_mean([0,0])
        self.say('最適でも KL は 0.511',figure=bound_figure([0,0]),
                 formula=r'\mathrm{KL}(q\|p)=0.511')
        self.say('残るのは形の違い',Indicate(self.true,color=YELLOW,scale_factor=1.01),
                 figure=width_figure())

    def width(self):
        self.say('斜めの広がりを横へ写す',Indicate(self.axis[0],color=YELLOW,scale_factor=1.01),
                 figure=recap_card('復習: 2.3  相関と周辺分布',width_figure()),color=YELLOW)
        self.say('緑を横軸へ投影',Indicate(self.true,color=YELLOW,scale_factor=1.01),
                 figure=width_figure(),formula=r'\mathrm{Var}_{p}(z_1)=1.389')
        self.say('青の分散は 0.500',Indicate(self.approx,color=YELLOW,scale_factor=1.01),
                 formula=r'\mathrm{Var}_{q}(z_1)=0.500')
        self.say('曲がり方の逆数が分散',formula=r'\Lambda_{11}=2\quad\Rightarrow\quad V_1=1/2')
        self.say('斜めの連動を失った',Indicate(self.true,color=YELLOW,scale_factor=1.01))
        self.say('分散比は 0.36',formula=r'0.500/1.389\simeq0.36')
        self.say('この対称なガウス例の結果',figure=width_figure(),color=YELLOW)

    def direction(self):
        self.say('変分推論は q から p への KL',figure=width_figure(),
                 formula=r'\mathrm{KL}(q\|p)')
        self.say('逆向きは真の周辺分布へ',figure=width_figure(True),
                 formula=r'\mathrm{KL}(p\|q)')
        self.remove(self.figure)
        self.figure=None
        self.say('離れた 2峰の模式図',self.set_bimodal('wide'),
                 formula=r'p(z)=\tfrac12p_1(z)+\tfrac12p_2(z)')
        self.say('谷に重みを置くと差が大きい',color=YELLOW)
        self.say('q から p なら片方へ',self.set_bimodal('left'),
                 formula=r'\mathrm{KL}(q\|p)')
        self.say('逆向きなら両方を覆う',self.set_bimodal('wide'),
                 formula=r'\mathrm{KL}(p\|q)')
        self.say('KL の向きで形が変わる',self.back_to_ellipse(),
                 formula=r'\mathrm{KL}(q\|p)',
                 color=YELLOW)

    def family(self):
        self.say('計算可能な分布族を選ぶ',figure=width_figure(),
                 formula=r'q\in\mathcal Q')
        self.say('独立な家族では傾けられない',Indicate(self.approx,color=YELLOW,scale_factor=1.01))
        self.say('斜めも許したらどうなる？',self.set_mean([0,0],COVARIANCE),
                 figure=width_figure(expanded=True),formula=r'\mathcal Q_2\supset\mathcal Q_1')
        self.say('この例は緑へ到達できる',figure=bound_figure([0,0],COVARIANCE),
                 formula=r'q=p\quad\Rightarrow\quad\mathrm{KL}=0')
        self.say('現実には計算量との相談',self.set_mean([0,0]),
                 figure=bound_figure([0,0]),formula=r'q\in\mathcal Q_1')
        self.say('候補を増やすほど近づける',self.set_mean([0,0],COVARIANCE),
                 figure=bound_figure([0,0],COVARIANCE),
                 formula=r'\mathcal Q_2\supset\mathcal Q_1',color=GREEN)
        self.say('今回は因子化の代償を見る',self.set_mean([0,0]),
                 figure=width_figure(),formula=r'q\in\mathcal Q_1')

    def return_to_start(self):
        self.say('冒頭と同じ楕円',figure=width_figure())
        self.say('分散 1.389 と 0.500',formula=r'\mathrm{Var}_{p}=1.389,\quad\mathrm{Var}_{q}=0.500')
        self.say('交互更新で中心は合う',
                 self.set_mean(ROWS[8]['mean']),figure=history_figure(8))
        self.say('独立の制約で幅は細い',self.set_mean([0,0]),
                 figure=width_figure())
        self.say('最適下界は −2.120',figure=bound_figure([0,0]),
                 formula=r'-2.120+0.511=-1.609')
        self.say('連動を捨てた代償が残る',figure=width_figure(),color=GREEN)
        self.say('10.2 では混合モデルへ',formula=r'10.2',
                 color=YELLOW)
