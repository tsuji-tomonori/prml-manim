"""PRML 9.3: from unseen component labels to weighted complete-data EM."""
import json
from pathlib import Path
import numpy as np
from manim import *
from mixture_model import (POINTS, PROBE_INDEX, INITIAL, em_history, components,
                           common_variance_responsibility, BINARY_PATTERNS,
                           BINARY_INITIAL, bernoulli_step)
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


def J(message, size=21, color=WHITE):
    return jp(str(message), size, color)


def X(x):
    return -5.25 + (float(x)+3)*.96


def Y(y):
    return -1.52 + float(y)*6.7


def P(x,y):
    return np.array([X(x),Y(y),0.])


def curve(mu,sigma,pi,part):
    xs=np.linspace(-3,3,151)
    data=components(xs,np.asarray(mu),np.asarray(sigma),np.asarray(pi))
    yy=data.sum(axis=1) if part==2 else data[:,part]
    return VMobject().set_points_as_corners([P(x,y) for x,y in zip(xs,yy)]).set_stroke((RED,BLUE,GREEN)[part],3)


def curves(state):
    return VGroup(*[curve(state['mu'],state['sigma'],state['pi'],i) for i in range(3)])


def common_curves(variance):
    """The same density scale for every fixed-variance K-means comparison."""
    xs=np.linspace(-3,3,181)
    data=components(xs,INITIAL['mu'],np.sqrt(variance)*np.ones(2),INITIAL['pi'])
    all_y=(data[:,0],data[:,1],data.sum(axis=1))
    return VGroup(*[
        VMobject().set_points_as_corners([
            np.array([X(x),-1.52+2.9*float(y),0.]) for x,y in zip(xs,yy)
        ]).set_stroke(color,3)
        for yy,color in zip(all_y,(RED,BLUE,GREEN))
    ])


def ink(gamma):
    return interpolate_color(BLUE,RED,float(gamma))


def bars(red, labels=('赤','青')):
    out=VGroup()
    for i,(share,color,label) in enumerate(zip((red,1-red),(RED,BLUE),labels)):
        y=.24-.63*i
        name=J(label,21,color).move_to([1.68,y,0])
        length=max(.03,float(share)*2.25)
        bar=Rectangle(width=length,height=.20,stroke_width=0,fill_color=color,fill_opacity=.9)
        bar.move_to([2.2+length/2,y,0])
        value=J(f'{share:.3f}',18,WHITE).next_to(bar,RIGHT,buff=.16)
        out.add(name,bar,value)
    return out


def plot(state, colored=True):
    axis=VGroup(Line(P(-3,0),P(3,0),color=MUTED,stroke_width=2),
                Line(P(-3,0),P(-3,.51),color=MUTED,stroke_width=2),
                J('密度',17,MUTED).move_to([-5.17,2.20,0]),
                J('観測16点',17,MUTED).move_to([-4.45,-2.80,0]))
    for x in (-2,0,2):
        axis.add(Line([X(x),Y(0),0],[X(x),Y(0)-.12,0],color=MUTED,stroke_width=1.3),
                 J(str(x),15,MUTED).move_to([X(x),-1.91,0]))
    dots=VGroup(*[Dot([X(x),-2.40,0],radius=.067,
                      color=ink(state['gamma'][i,0]) if colored else WHITE)
                  for i,x in enumerate(POINTS)])
    ring=Circle(radius=.152,color=YELLOW,stroke_width=2).move_to(dots[PROBE_INDEX])
    ring.set_z_index(3)
    return axis,dots,ring


def pixel_grid(values, center, unit=.37, probabilistic=False, color=BLUE):
    group=VGroup()
    cx,cy=center
    for i,value in enumerate(values):
        row,col=divmod(i,2)
        sq=Square(side_length=unit,stroke_color=MUTED,stroke_width=1.5,
                  fill_color=WHITE if not probabilistic else color,
                  fill_opacity=float(value) if probabilistic else (.9 if value else .08))
        sq.move_to([cx+(col-.5)*unit,cy+(.5-row)*unit,0])
        group.add(sq)
    return group


def width_marks(variance):
    """Actual one-standard-deviation intervals, shown below the density plot."""
    spread=float(np.sqrt(variance))
    return VGroup(
        Line([X(-1.1-spread),-.87,0],[X(-1.1+spread),-.87,0],color=RED,stroke_width=7),
        Line([X(1.1-spread),-1.07,0],[X(1.1+spread),-1.07,0],color=BLUE,stroke_width=7),
    )


class PRML93AlternativeEM(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate((self.opening,self.complete_data,self.expectation,
                                   self.maximization,self.cycles,self.kmeans,
                                   self.bernoulli,self.return_to_probe)):
            self.begin(i)
            self.note=self.formula=self.value=self.bar_mob=None
            method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml93_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def setup_plot(self,state=STATES[0],colored=True):
        axis,self.dots,self.ring=plot(state,colored)
        self.curves=curves(state)
        self.add(axis,self.curves,self.dots,self.ring)
        self.state=state

    def replace_note(self,message,color=MUTED):
        mob=J(message,21,color)
        if mob.width>4.45:mob.scale_to_fit_width(4.45)
        mob.move_to([3.75,-2.38,0])
        if self.note is not None:self.remove(self.note)
        self.note=mob
        return FadeIn(mob,rate_func=lambda t:smooth(min(1,6*t)))

    def replace_formula(self,formula,color=WHITE):
        mob=tex(formula,36,color)
        if mob.width>4.55:mob.scale_to_fit_width(4.55)
        if mob.height>1.03:mob.scale_to_fit_height(1.03)
        mob.move_to([3.77,1.68,0])
        if self.formula is not None:self.remove(self.formula)
        self.formula=mob
        return FadeIn(mob,rate_func=lambda t:smooth(min(1,6*t)))

    def replace_value(self,message,color=GREEN):
        mob=J(message,21,color)
        if mob.width>4.5:mob.scale_to_fit_width(4.5)
        mob.move_to([3.78,.88,0])
        if self.value is not None:self.remove(self.value)
        self.value=mob
        return FadeIn(mob,rate_func=lambda t:smooth(min(1,6*t)))

    def replace_bars(self,red):
        target=bars(red)
        if self.bar_mob is None:
            self.bar_mob=target
            return FadeIn(target,rate_func=lambda t:smooth(min(1,6*t)))
        old=self.bar_mob
        self.remove(old)
        self.bar_mob=target
        return FadeIn(target,rate_func=lambda t:smooth(min(1,6*t)))

    def say(self,message,*anims,formula=None,value=None,red=None,color=MUTED):
        visual=[self.replace_note(message,color)]
        if formula is not None:visual.append(self.replace_formula(formula))
        if value is not None:visual.append(self.replace_value(value))
        if red is not None:visual.append(self.replace_bars(red))
        visual.extend(anims)
        self.beat(*visual)

    def update_state(self,state,recolor=True):
        self.state=state
        actions=[Transform(self.curves,curves(state))]
        if recolor:
            actions.extend(dot.animate.set_color(ink(state['gamma'][i,0])) for i,dot in enumerate(self.dots))
        return AnimationGroup(*actions,lag_ratio=0)

    def opening(self):
        self.setup_plot()
        self.say('9.2と同じ16点',Indicate(self.dots,color=YELLOW,scale_factor=1.02))
        blue_center=Dot(P(1.1,0),color=BLUE,radius=.10)
        red_center=Dot(P(-1.1,0),color=RED,radius=.10)
        self.say('0.15は青い中心に近い',FadeIn(blue_center),FadeIn(red_center),
                 formula=r'|0.15-1.1|=0.95')
        self.say('赤い負担率のほうが高い',Indicate(self.ring,color=YELLOW,scale_factor=1.16),
                 red=STATES[0]['gamma'][PROBE_INDEX,0])
        self.say('本当の札は見えない',value='赤0.564  /  青0.436')
        self.say('札なしでどう更新した？',formula=r'z_{n1}\in\{0,1\}',
                 color=YELLOW)
        self.say('仮に札が見えたら？',Indicate(self.dots[:8],color=RED,scale_factor=1.03))
        self.say('同じ点から2つの計算',formula=r'X\quad\longrightarrow\quad (X,Z)')

    def complete_data(self):
        self.setup_plot(colored=False)
        self.say('観測できるのは点だけ',formula=r'X=\{x_1,\ldots,x_{16}\}')
        self.say('対数の内側に成分の和',formula=r'\ln p(X|\theta)=\sum_n\ln\sum_k p(x_n,z_k=1|\theta)',
                 value='観測データの対数尤度')
        colors=[RED if x<0 else BLUE for x in POINTS]
        self.say('もし札が分かれば',*[dot.animate.set_color(c) for dot,c in zip(self.dots,colors)],
                 formula=r'z_{nk}\in\{0,1\}')
        self.say('点と札が完全データ',value='完全データ  (X, Z)')
        self.say('成分ごとの対数へ',formula=r'\ln p(X,Z|\theta)=\sum_{n,k}z_{nk}\ln p(x_n,z_k=1|\theta)')
        self.say('和と対数の順番が変わる',formula=r'\ln\sum_k p_k\quad\longrightarrow\quad\sum_k z_k\ln p_k',
                 value='札があれば成分ごとに計算')
        self.say('赤い点だけの重心',Indicate(self.dots[:8],color=RED,scale_factor=1.09),
                 formula=r'\mu_1=\frac{\sum_n z_{n1}x_n}{\sum_n z_{n1}}')
        self.say('札の代わりに候補を重ねる',red=STATES[0]['gamma'][PROBE_INDEX,0],
                 value='赤い札  /  青い札')

    def expectation(self):
        self.setup_plot()
        # A small explicit average of two indicator outcomes, with no decorative backing.
        ticket1=J('赤の札 1',23,RED).move_to([2.55,.45,0])
        ticket0=J('赤の札 0',23,BLUE).move_to([4.9,.45,0])
        weight1=J('0.564',19,YELLOW).next_to(ticket1,DOWN,buff=.18)
        weight0=J('0.436',19,YELLOW).next_to(ticket0,DOWN,buff=.18)
        self.say('確率で重み付けした平均',FadeIn(ticket1),FadeIn(ticket0),FadeIn(weight1),FadeIn(weight0),
                 formula=r'E[z]=\sum_z p(z)z')
        self.say('赤い札の期待値は0.564',formula=r'E[z_{n1}]=0.564\cdot1+0.436\cdot0=0.564',
                 value='期待値 = 赤い負担率')
        self.remove(ticket1,ticket0,weight1,weight0)
        self.say('今の山で札を推測',red=STATES[0]['gamma'][PROBE_INDEX,0],
                 formula=r'p(Z|X,\theta^{old})')
        self.say('札付きの対数を平均',formula=r'Q(\theta,\theta^{old})=E_{Z|X,\theta^{old}}[\ln p(X,Z|\theta)]',
                 value='Eステップ')
        self.say('札の0/1が負担率になる',formula=r'Q=\sum_{n,k}\gamma_{nk}(\ln\pi_k+\ln\mathcal N_k(x_n))',
                 value='赤い点にも青い点にも重み')
        self.say('16点で同じ平均',value='有効な点数  赤8.724  青7.276',
                 formula=r'N_k=\sum_n\gamma_{nk}')
        self.say('山を動かすと重みも変わる？',Indicate(self.curves[0],color=RED,scale_factor=1),
                 red=STATES[0]['gamma'][PROBE_INDEX,0])
        self.say('M中の負担率は固定',value='赤0.564 を固定',color=YELLOW)

    def maximization(self):
        self.setup_plot()
        old_gamma=STATES[0]['gamma'][PROBE_INDEX,0]
        self.say('固定した重みでQを最大化',formula=r'\theta^{new}=\arg\max_\theta Q(\theta,\theta^{old})',
                 red=old_gamma)
        self.say('各点の赤い重り',Indicate(self.curves[0],color=RED,scale_factor=1),
                 formula=r'\gamma_{n1}=E[z_{n1}]')
        self.say('赤い中心は-0.944へ',self.update_state(STATES[1],recolor=False),
                 formula=r'\mu_k^{new}=\frac{\sum_n\gamma_{nk}x_n}{N_k}',
                 value='赤の中心  -1.100 → -0.944')
        self.say('有効な点数は整数とは限らない',value='赤8.724点分  /  青7.276点分',
                 formula=r'N_k=\sum_n\gamma_{nk}')
        self.say('新しい中心から幅を測る',formula=r'(\sigma_k^2)^{new}=\frac{\sum_n\gamma_{nk}(x_n-\mu_k^{new})^2}{N_k}',
                 value='標準偏差  赤0.868  青0.891')
        self.say('混合係数は有効な点数',formula=r'\pi_k^{new}=N_k/N,\quad N_k=\sum_n\gamma_{nk}',
                 value='赤0.545  /  青0.455')
        self.say('9.2と同じ1回目',formula=r'Q:\;-32.706\ \longrightarrow\ -31.655',
                 red=old_gamma)
        self.say('次のEでは重みを再計算',self.update_state(STATES[1]),red=STATES[1]['gamma'][PROBE_INDEX,0],
                 value='赤0.564 → 0.503')

    def cycles(self):
        self.setup_plot(STATES[1])
        self.say('新しい山でE: 札を再推測',red=STATES[1]['gamma'][PROBE_INDEX,0],
                 value='赤い負担率 0.503')
        self.say('次のM: 重みを固定して更新',self.update_state(STATES[2],recolor=False),
                 value='赤0.503を固定')
        self.say('観測の点数を測る',formula=r'\ln p(X|\theta):\;-27.737\to-26.521',
                 value='観測の対数尤度')
        self.say('2回目も少し上がる',self.update_state(STATES[2]),
                 red=STATES[2]['gamma'][PROBE_INDEX,0],
                 formula=r'\ln p(X|\theta):\;-26.521\to-26.293')
        self.say('Qと観測の採点は別',formula=r'Q\ne\ln p(X|\theta)',
                 value='固定した重みの目的  /  観測の採点')
        self.say('今回は下がらない',self.update_state(STATES[5]),
                 formula=r'\ln p(X|\theta_5)\simeq-26.145')
        self.say('硬い札へ近づく？',value='同じ分散で幅を絞る',
                 formula=r'\Sigma_1=\Sigma_2=\epsilon I')

    def kmeans(self):
        self.setup_plot(STATES[0])
        self.curves.become(common_curves(1.))
        marks=width_marks(1.)
        self.say('共通の分散を固定',FadeIn(marks),formula=r'\Sigma_1=\Sigma_2=\epsilon I',
                 value='別の説明用実験')
        self.say('幅1では柔らかい札',red=common_variance_responsibility(.15,1.)[0],
                 value='分散 ε = 1')
        self.say('分散を少し縮める',Transform(marks,width_marks(.5)),Transform(self.curves,common_curves(.5)),
                 red=common_variance_responsibility(.15,.5)[0],
                 value='分散 ε = 0.5')
        self.say('遠い成分の重みが速く消える',
                 formula=r'\gamma_{nk}\propto\pi_k e^{-\|x_n-\mu_k\|^2/(2\epsilon)}',
                 value='中心からの距離で比較')
        self.say('近い青が優勢に',Transform(marks,width_marks(.2)),Transform(self.curves,common_curves(.2)),
                 red=common_variance_responsibility(.15,.2)[0],
                 value='分散 ε = 0.2')
        self.say('0か1の札へ近づく',Transform(marks,width_marks(.05)),Transform(self.curves,common_curves(.05)),
                 red=common_variance_responsibility(.15,.05)[0],
                 value='分散 ε = 0.05',formula=r'\gamma_{nk}\longrightarrow r_{nk}')
        hard=VGroup(*[Dot([X(x),-2.40,0],radius=.067,color=RED if abs(x+1.1)<abs(x-1.1) else BLUE)
                      for x in POINTS])
        self.say('9.1のK平均の割当て',Transform(self.dots,hard),
                 value='近い中心の点だけを平均')
        self.say('等距離では別扱い',formula=r'\epsilon\to0\quad (\pi_k>0)',
                 value='中心と等距離の点は除く')

    def bernoulli(self):
        # Replace Gaussian plots with binary pixel grids; square edges encode pixels.
        recap_label=J('復習: 2.1 ベルヌーイ分布',21,WHITE).move_to([-3.7,2.17,0])
        recap_box=Rectangle(width=4.0,height=2.3,color=YELLOW,stroke_width=1.5).move_to([-3.7,1.3,0])
        zero=J('0',26,BLUE).move_to([-4.8,.9,0])
        one=J('1',26,GREEN).move_to([-3.6,.9,0])
        pbar=Line([-4.8,.4,0],[-3.6,.4,0],color=YELLOW,stroke_width=6)
        self.say('2.1の0/1を復習',FadeIn(recap_label),FadeIn(recap_box),FadeIn(zero),FadeIn(one),FadeIn(pbar),
                 formula=r'p(x=1)=\mu')
        self.remove(recap_label,recap_box,zero,one,pbar)
        grids=VGroup(*[pixel_grid(row,(-5.35+i*.76,.1),.34) for i,row in enumerate(BINARY_PATTERNS)])
        self.say('説明用の4画素模様',FadeIn(grids),value='観測8個・各4画素')
        red_grid=pixel_grid(BINARY_INITIAL['mu'][0],(-3.55,1.85),.48,True,RED)
        blue_grid=pixel_grid(BINARY_INITIAL['mu'][1],(-1.95,1.85),.48,True,BLUE)
        red_label=J('赤',20,RED).move_to([-3.55,2.48,0])
        blue_label=J('青',20,BLUE).move_to([-1.95,2.48,0])
        self.say('赤と青の模様の確率',FadeIn(red_grid),FadeIn(blue_grid),FadeIn(red_label),FadeIn(blue_label),
                 value='成分内で画素は独立')
        self.say('札を隠すとまた和',formula=r'p(x)=\sum_k\pi_k p(x|\mu_k)',
                 value='ベルヌーイ混合')
        gamma,new_mu,new_pi=bernoulli_step()
        self.say('Eで模様から札を推測',formula=r'\gamma_{nk}=\frac{\pi_kp(x_n|\mu_k)}{\sum_j\pi_jp(x_n|\mu_j)}',
                 red=gamma[0,0])
        updated_red=pixel_grid(new_mu[0],(-3.55,1.85),.48,True,RED)
        updated_blue=pixel_grid(new_mu[1],(-1.95,1.85),.48,True,BLUE)
        self.say('Mで画素ごとに平均',Transform(red_grid,updated_red),Transform(blue_grid,updated_blue),
                 formula=r'\mu_{ki}^{new}=\frac{\sum_n\gamma_{nk}x_{ni}}{N_k}')
        self.say('同じEMの骨組み',value='見えない札 → 重み → 更新',
                 formula=r'z_{nk}\quad\longrightarrow\quad E[z_{nk}]=\gamma_{nk}')

    def return_to_probe(self):
        self.setup_plot(STATES[0])
        self.say('同じ16点と初期の山',formula=r'\mu=(-1.1,1.1),\quad\sigma=(0.9,1.2)',
                 value='9.2と同じ条件')
        self.say('青に近いのに赤が0.564',red=STATES[0]['gamma'][PROBE_INDEX,0],
                 value='注目点 x = 0.15')
        self.say('赤い札の期待値',formula=r'E[z_{n1}]=0.564\cdot1+0.436\cdot0=0.564',
                 red=STATES[0]['gamma'][PROBE_INDEX,0])
        self.say('青い札と合わせて1',formula=r'E[z_{n2}]=0.436',
                 red=STATES[0]['gamma'][PROBE_INDEX,0])
        self.say('全点でQを作る',formula=r'Q=E_{Z|X,\theta^{old}}[\ln p(X,Z|\theta)]',
                 value='Eで重み、Mで山')
        self.say('札なしでも更新できる',self.update_state(STATES[1]),
                 formula=r'\mu_1:\;-1.100\to-0.944',
                 value='負担率で完全データを平均')
        self.say('次は上昇の理由へ',formula=r'\ln p(X|\theta^{new})\geq\ln p(X|\theta^{old})',
                 value='9.4で保証を確かめる')
