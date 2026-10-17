"""PRML 10.3: one extrapolation, two alternating distributions."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import gamma
from manim import *
from regression_model import X, T, PROBE, BETA, A0, B0, fit, step, prediction, phi
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
HISTORY = fit()
WEAK, STRONG = step(.01), step(100.)
NOTES = [
    ['10.2の交互更新を回帰へ','同じ10点と未知の位置','右端の高さを予想','弱い事前精度','強い事前精度','強さは誰が決める？','条件を固定して追う'],
    ['復習: 3.3 ベイズ線形回帰','4つの基底関数','係数の候補曲線','既知の観測雑音','精度が大きいほど縮む','強さは固定だった','強さにも候補を'],
    ['正の精度を分布にする','復習: 2.3 ガンマ分布','観測→係数→精度','2つの因子に分ける','分布で平均する','交互に更新する','係数の大きさを見る'],
    ['4つの係数を二乗','期待値は重み付き平均','平均の二乗と広がり','ガンマ分布を更新','広がりは率へ加わる','初回の平均精度','曲線へ返す'],
    ['平均精度を渡す','係数の分布を更新','事前と観測の精度','観測から平均を求める','2つの分布を往復','平均精度は約1.88','下界で確認'],
    ['変分下界を測る','1巡目から2巡目','更新が落ち着く','復習: 3.5 エビデンス近似','幅広い事前の極限','初期値への注意','予測幅は？'],
    ['同じ未知の位置','予測の中心','観測雑音の分散','係数の分散','2つの分散を足す','標準偏差は約0.77','固定した基底の条件'],
    ['同じ10点、同じ位置','係数と精度を往復','平均と標準偏差','生成時の値','問いへの答え','次の近似へ'],
]
FORMULAS = [
    ['',r'N=10',r'x=1.35','','',r'\alpha\ ?',''],
    [r'3.3',r'y(x,w)=w^T\phi(x)',r'w=(w_0,w_1,w_2,w_3)',r'\beta^{-1}=0.09',r'p(w\mid\alpha)=\mathcal N(0,\alpha^{-1}I)','', ''],
    [r'\alpha>0',r'p(\alpha)=\mathrm{Gam}(\alpha\mid a_0,b_0)',r'p(t,w,\alpha)=p(t\mid w)p(w\mid\alpha)p(\alpha)',r'q(w,\alpha)=q(w)q(\alpha)',r'E[\cdot]','',''],
    [r'w^Tw=\sum_{j=0}^3w_j^2',r'E[w^Tw]',r'E[w^Tw]=m_N^Tm_N+\mathrm{Tr}(S_N)',r'a_N=a_0+2,\quad b_N=b_0+\tfrac12 E[w^Tw]','',r'E[\alpha]\simeq1.73',''],
    [r'E[\alpha]=a_N/b_N',r'q(w)=\mathcal N(m_N,S_N)',r'S_N^{-1}=E[\alpha]I+\beta\Phi^T\Phi',r'm_N=\beta S_N\Phi^Tt','',r'E[\alpha]\simeq1.88',r'\mathcal L(q)'],
    [r'\mathcal L(q)',r'-10.31\rightarrow-10.28',r'\mathcal L\simeq-10.27',r'3.5',r'a_0,b_0\rightarrow0','',''],
    [r'x=1.35',r'E[t\mid x]\simeq2.35',r'\beta^{-1}=0.09',r'\phi^TS_N\phi\simeq0.503',r'\sigma^2=\beta^{-1}+\phi^TS_N\phi',r'\sigma\simeq0.77',''],
    [r'x=1.35',r'q(w)\leftrightarrow q(\alpha)',r'E[t]\simeq2.35,\quad\sigma\simeq0.77',r't_{\rm true}\simeq2.45',r'0.09+0.503',''],
]
ITERATIONS = [
    [0,0,0,-1,-2,-2,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,1,1,2,4,10,24],
    [0,1,2,3,5,10,24],
    [24,24,24,24,24,24,24],
    [-2,24,24,24,24,24],
]


def state_for(index):
    return WEAK if index == -1 else STRONG if index == -2 else HISTORY[index]


def J(value, size=19, color=WHITE):
    return jp(str(value), size, color)


def pt(x, y):
    return np.array([-5.48 + (x + 1.16) * 2.46, -2.12 + (y + 2.0) * .83, 0])


def graph_axes():
    left, right, bottom, top = -1.16, 1.44, -2.0, 3.42
    group = VGroup(Line(pt(left,bottom),pt(right,bottom),color=MUTED,stroke_width=1.25),
                   Line(pt(left,bottom),pt(left,top),color=MUTED,stroke_width=1.25))
    for x in (-1,0,1):
        group.add(Line(pt(x,bottom)+DOWN*.07,pt(x,bottom)+UP*.07,color=MUTED,stroke_width=1),
                  J(str(x),14,MUTED).move_to(pt(x,bottom)+DOWN*.23))
    for y in (-1,0,1,2,3):
        group.add(Line(pt(left,y)+LEFT*.06,pt(left,y)+RIGHT*.06,color=MUTED,stroke_width=1),
                  J(str(y),13,MUTED).move_to(pt(left,y)+LEFT*.21))
    group.add(tex('x',22,MUTED).move_to(pt(right,bottom)+RIGHT*.22),
              tex('t',22,MUTED).move_to(pt(left,top)+UP*.2))
    return group


def observations():
    return VGroup(*[Dot(pt(float(x),float(y)),radius=.057,color=BLUE)
                    for x,y in zip(X,T)])


def curve(state, show_band=False, show_truth=False, candidates=False, visible=True):
    xs=np.linspace(-1.15,1.43,111)
    design=phi(xs)
    means=design@state['mean']
    variances=np.einsum('ij,jk,ik->i',design,state['covariance'],design)
    group=VGroup()
    sd=np.sqrt(variances + (1/BETA if show_band==2 else 0))
    top=np.minimum(means+sd,3.4)
    bottom=np.maximum(means-sd,-2.0)
    polygon=Polygon(*[pt(x,y) for x,y in zip(xs,top)],
                    *[pt(x,y) for x,y in zip(xs[::-1],bottom[::-1])],
                    stroke_width=0,fill_color=GREEN if show_band==2 else PURPLE,
                    fill_opacity=.15 if show_band else 0)
    group.add(polygon)
    weights=np.random.default_rng(33).multivariate_normal(state['mean'],state['covariance'],size=4)
    for weight in weights:
        yy=design@weight
        visible_points=[pt(x,y) for x,y in zip(xs,yy) if -1.98<=y<=3.39]
        line=VMobject().set_points_as_corners(visible_points if len(visible_points)>1 else [pt(-1.15,-1.97),pt(-1.14,-1.97)])
        group.add(line.set_stroke(BLUE,width=1.7,opacity=.48 if candidates else 0))
    visible_mean=[pt(x,y) for x,y in zip(xs,means) if -1.98<=y<=3.4]
    line=VMobject().set_points_as_corners(visible_mean)
    group.add(line.set_stroke(GREEN,width=3.5,opacity=1 if visible else 0))
    p=prediction(state)
    group.add(Dot(pt(PROBE,p['mean']),radius=.065,color=YELLOW,fill_opacity=1 if visible else 0))
    group.add(Dot(pt(PROBE,p['true_mean']),radius=.065,color=RED,fill_opacity=1 if show_truth else 0),
              J('生成時',17,RED).move_to(pt(1.05,3.25)).set_opacity(1 if show_truth else 0))
    return group


def side_axes():
    return VGroup(Line([2.55,-1.62,0],[6.18,-1.62,0],color=MUTED,stroke_width=1.15),
                  Line([2.55,-1.62,0],[2.55,.82,0],color=MUTED,stroke_width=1.15))


def alpha_side(alpha):
    group=VGroup(side_axes(), J('事前精度を動かす',18,MUTED).move_to([4.3,.95,0]))
    # An explicit slider, with the numerical value kept next to the symbol.
    prefix=tex(r'\alpha=',25,YELLOW).move_to([3.75,-1.03,0])
    number=DecimalNumber(alpha,num_decimal_places=2,font_size=25,color=YELLOW,edge_to_fix=LEFT)
    number.next_to(prefix,RIGHT,buff=.08)
    group.add(Line([2.95,-.35,0],[5.8,-.35,0],color=PURPLE,stroke_width=4),
              Dot([2.95+2.85*(np.log10(alpha)+2)/4,-.35,0],radius=.1,color=YELLOW),
              prefix,number)
    return group


def weight_side(state):
    group=VGroup(side_axes(),J('係数の平均と広がり',18,MUTED).move_to([4.3,.95,0]))
    for j,(m,v) in enumerate(zip(state['mean'],np.diag(state['covariance']))):
        x=2.95+j*.84
        group.add(Line([x,-.95,0],[x,-.95+.8*float(m),0],color=GREEN,stroke_width=14),
                  Line([x-.1,-.95+.8*(m-np.sqrt(v)),0],[x+.1,-.95+.8*(m-np.sqrt(v)),0],color=PURPLE,stroke_width=2),
                  J(str(j),15,MUTED).move_to([x,-1.86,0]))
    return group


def gamma_side(state, prior=False):
    a,b=(A0,B0) if prior else (state['a'],state['b'])
    mean=a/b
    group=VGroup(side_axes(),J('精度の事前分布' if prior else '精度の更新後分布',18,MUTED).move_to([4.3,.95,0]))
    xs=np.linspace(.05,6.8,90)
    ys=gamma.pdf(xs,a,scale=1/b)
    peak=max(ys.max(),.01)
    line=VMobject().set_points_as_corners([[2.64+3.43*x/6.8,-1.55+2.18*y/peak,0] for x,y in zip(xs,ys)])
    group.add(line.set_stroke(PURPLE,width=3),
              Dot([2.64+3.43*mean/6.8,-1.55+2.18*gamma.pdf(mean,a,scale=1/b)/peak,0],radius=.07,color=YELLOW),
              J(f'E[α] = {mean:.2f}',18,YELLOW).move_to([4.3,-2.14,0]))
    return group


def squared_side(state,stage=2):
    m=state['mean'];diagonal=np.diag(state['covariance'])
    group=VGroup(side_axes(),J('係数の二乗の平均',18,MUTED).move_to([4.3,.95,0]))
    for j,(mm,vv) in enumerate(zip(m,diagonal)):
        x=2.95+j*.84
        group.add(Line([x,-1.45,0],[x,-1.45+1.35*mm*mm,0],color=BLUE,stroke_width=18),
                  Line([x,-1.45+1.35*mm*mm,0],[x,-1.45+1.35*(mm*mm+vv),0],color=YELLOW,
                       stroke_width=18,stroke_opacity=1 if stage>=1 else 0),
                  J(str(j),15,MUTED).move_to([x,-1.87,0]))
    group.add(J(f'合計 {sum(m*m)+sum(diagonal):.2f}',18,GREEN).move_to([4.3,-2.2,0])
              .set_opacity(1 if stage>=2 else 0))
    return group


def bound_side(index):
    group=VGroup(side_axes(),J('変分下界',18,MUTED).move_to([4.3,.95,0]))
    vals=np.array([state['bound'] for state in HISTORY[:11]])
    upto=max(0,min(index,10))
    coordinates=[[2.66+3.3*j/10,-1.4+2.0*(v-vals[0])/(vals[-1]-vals[0]),0] for j,v in enumerate(vals[:upto+1])]
    if len(coordinates)==1: coordinates.append(np.array(coordinates[0])+RIGHT*.001)
    line=VMobject().set_points_as_corners(coordinates)
    group.add(line.set_stroke(GREEN,width=3),Dot(coordinates[-1],radius=.07,color=YELLOW),
              DecimalNumber(vals[upto],num_decimal_places=2,font_size=18,color=GREEN,
                            edge_to_fix=LEFT).move_to([4.3,-2.16,0]))
    return group


def evidence_side():
    """A compact 3.5 reminder: current data, fixed-alpha log evidence."""
    group=VGroup(side_axes(),J('3.5 精度を1点で選ぶ',18,MUTED).move_to([4.3,.95,0]))
    design=phi(X)
    alphas=np.logspace(-2,2.5,91)
    values=[]
    for alpha in alphas:
        covariance=np.eye(len(T))/BETA+design@design.T/alpha
        sign,det=np.linalg.slogdet(covariance)
        values.append(-.5*(len(T)*np.log(2*np.pi)+det+T@np.linalg.solve(covariance,T)))
    values=np.array(values)
    yy=(values-values.min())/(values.max()-values.min())
    coordinates=[[2.68+3.34*j/(len(alphas)-1),-1.44+2.1*y,0] for j,y in enumerate(yy)]
    line=VMobject().set_points_as_corners(coordinates)
    peak=int(np.argmax(values))
    group.add(line.set_stroke(PURPLE,width=3),Dot(coordinates[peak],radius=.08,color=YELLOW),
              J('1つの精度を選ぶ',17,YELLOW).move_to([4.3,-2.12,0]))
    return group


def variance_side(state, stage=2):
    p=prediction(state)
    noise=p['noise_variance'];weight=p['weight_variance']
    group=VGroup(side_axes(),J('右端の予測分散',18,MUTED).move_to([4.3,.95,0]))
    group.add(Line([3.42,-1.47,0],[3.42,-1.47+2.9*noise,0],color=BLUE,stroke_width=33),
              J('雑音 0.09',17,BLUE).move_to([3.36,-1.98,0]))
    group.add(Line([5.18,-1.47,0],[5.18,-1.47+2.9*weight,0],color=PURPLE,
                   stroke_width=33,stroke_opacity=1 if stage>=1 else 0),
              J(f'係数 {weight:.3f}',17,PURPLE).move_to([5.2,-1.98,0])
              .set_opacity(1 if stage>=1 else 0),
              J(f'合計 {noise+weight:.3f}',19,GREEN).move_to([4.3,.35,0])
              .set_opacity(1 if stage>=2 else 0))
    return group


class PRML103VariationalRegression(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            self.add(graph_axes(),observations(),
                     Dot(pt(PROBE,-2.0),radius=.065,color=YELLOW),
                     tex('x=1.35',20,YELLOW).move_to(pt(PROBE,-2.0)+DOWN*.44))
            state=WEAK if si==7 else state_for(ITERATIONS[si][0])
            self.current=curve(state,visible=si!=0)
            self.side=alpha_side(.01) if si==7 else side_visual(si,0,state)
            self.side_mode=side_kind(si,0)
            self.note=None;self.formula=None;self.recap_border=None
            self.add(self.current,self.side)
            for bi in range(len(story['beats'])):
                self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml103_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def run_beat(self,si,bi):
        note=J(NOTES[si][bi],19,MUTED).move_to([4.35,2.48,0])
        if note.width>4.1: note.scale(4.1/note.width)
        if self.note is not None: self.remove(self.note)
        self.note=note;self.add(note)
        if self.recap_border is not None:self.remove(self.recap_border)
        self.recap_border=None
        if (si,bi) in ((1,0),(2,1),(5,3)):
            self.recap_border=Rectangle(width=4.3,height=5.7,color=YELLOW,
                                        stroke_width=1.0).move_to([4.35,0,0])
            self.add(self.recap_border)
        formula=FORMULAS[si][bi]
        if self.formula is not None:self.remove(self.formula)
        self.formula=None
        if formula:
            mob=tex(formula,23,YELLOW).move_to([4.32,1.77,0])
            if mob.width>4.05:mob.scale(4.05/mob.width)
            self.formula=mob;self.add(mob)
        state=state_for(ITERATIONS[si][bi])
        band=(2 if bi>=1 else False) if si==7 else (2 if bi>=4 else 1 if bi>=3 else False) if si==6 else False
        target=curve(state,show_band=band,show_truth=(si==7 and bi>=3),
                     candidates=(si==1 and bi in (0,1,2,3)),visible=(si!=0 or bi>=3))
        side=side_visual(si,bi,state)
        mode=side_kind(si,bi)
        if mode==self.side_mode:
            if mode=='alpha' and abs(self.side[5].get_value()-side[5].get_value())>1e-9:
                cues=self.beat_cues()
                span=cues[-1]['end']-cues[0]['start']
                side_animation=AnimationGroup(Transform(self.side[3],side[3]),
                                             ChangeDecimalToValue(self.side[5],side[5].get_value()),
                                             run_time=span)
            elif mode=='bound':
                cues=self.beat_cues()
                span=cues[-1]['end']-cues[0]['start']
                side_animation=AnimationGroup(Transform(self.side[2],side[2]),
                                             Transform(self.side[3],side[3]),
                                             ChangeDecimalToValue(self.side[4],side[4].get_value()),
                                             run_time=span)
            else:
                side_animation=Transform(self.side,side)
        else:
            cues=self.beat_cues()
            span=cues[-1]['end']-cues[0]['start']
            fade=min(.9,span/3)
            side_animation=Succession(FadeOut(self.side,run_time=fade/2),
                                      FadeIn(side,run_time=fade/2),
                                      Wait(max(.001,span-fade)))
        self.beat(Transform(self.current,target),side_animation)
        if mode!=self.side_mode:self.side=side
        self.side_mode=mode


def side_visual(si,bi,state):
    if si==0:
        return alpha_side(.01 if bi==3 else 100 if bi in (4,5) else 1.)
    if si==1:return weight_side(state)
    if si==2:return gamma_side(state,prior=bi<5)
    if si==3:
        return squared_side(state,stage=min(bi,2)) if bi<3 else gamma_side(state,prior=bi==3)
    if si==4:return weight_side(state) if bi>=1 else gamma_side(state)
    if si==5:return evidence_side() if bi==3 else bound_side(ITERATIONS[si][bi])
    if si==6:return variance_side(state,0 if bi<=2 else 1 if bi==3 else 2)
    return alpha_side(100.) if bi==0 else variance_side(state,2)


def side_kind(si,bi):
    if si==2:return 'prior_gamma' if bi<5 else 'updated_gamma'
    if si==3:return 'squared' if bi<3 else 'prior_gamma' if bi==3 else 'updated_gamma'
    if si==4:return 'updated_gamma' if bi==0 else 'weight'
    if si==5:return 'evidence' if bi==3 else 'bound'
    if si==7:return 'alpha' if bi==0 else 'variance'
    return ('alpha','weight',None,None,None,None,'variance')[si]
