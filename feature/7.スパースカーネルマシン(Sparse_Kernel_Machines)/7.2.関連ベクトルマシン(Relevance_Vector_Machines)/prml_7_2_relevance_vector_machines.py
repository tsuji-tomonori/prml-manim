"""Two relevance vectors explain eleven observations and a predictive interval."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene,jp,tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from rvm_model import X,T,WIDTH,BETA,MODEL,predict

BG='#10141F'
BLUE=ManimColor('#58C4DD'); YELLOW=ManimColor('#FFFF00')
GREEN=ManimColor('#83C167'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
_fade_in=FadeIn
def FadeIn(mobject,**kwargs):
    kwargs.setdefault('rate_func',lambda t:smooth(min(1.,10*t)))
    return _fade_in(mobject,**kwargs)

def L(s,x=0,y=2.46,size=25,color=WHITE):return jp(s,size,color).move_to((x,y,0))
def F(s,x=0,y=2.46,size=28,color=WHITE):return tex(s,size,color).move_to((x,y,0))

def graph():
    ax=Axes(x_range=[0,1,.2],y_range=[-1.05,1.22,.5],x_length=9.3,y_length=3.6,tips=False,
            axis_config=dict(color=MUTED,stroke_width=1.2,include_ticks=False)).shift(DOWN*.48)
    labels=VGroup()
    for x in [0,.2,.4,.6,.8,1]:
        point=ax.c2p(x,-1.05)
        labels.add(Line(point,point+UP*.06,color=MUTED,stroke_width=1))
        labels.add(F(f'{x:g}',point[0],point[1]-.25,18,MUTED))
    labels.add(F('x',ax.c2p(1,-1.05)[0]+.38,ax.c2p(1,-1.05)[1]-.05,22))
    labels.add(F('t',ax.c2p(0,1.22)[0]-.28,ax.c2p(0,1.22)[1]+.15,22))
    return ax,labels

def points(ax):return VGroup(*[Dot(ax.c2p(float(x),float(t)),radius=.073,color=BLUE).set_z_index(4) for x,t in zip(X,T)])
def curve(ax,model=MODEL,color=GREEN,width=3):return ax.plot(lambda z:predict(z,model)[0],x_range=[0,1,.01],color=color,stroke_width=width)
def kernel(ax,index,coefficient=1,color=PURPLE):
    center=float(X[index])
    return ax.plot(lambda z:coefficient*np.exp(-.5*((z-center)/WIDTH)**2),x_range=[0,1,.01],color=color,stroke_width=2.2)
def circles(dots):return VGroup(*[Circle(radius=.14,color=YELLOW,stroke_width=2.3).move_to(dots[i]).set_z_index(5) for i in [3,7]])
def band(ax,variance_scale=1.,color=PURPLE,opacity=.16):
    grid=np.linspace(0,1,100)
    mu,var,wvar=predict(grid)
    sigma=np.sqrt(wvar if variance_scale==0 else var)
    pts=[ax.c2p(float(z),float(v)) for z,v in zip(grid,mu+2*sigma)]
    pts += [ax.c2p(float(z),float(v)) for z,v in zip(grid[::-1],(mu-2*sigma)[::-1])]
    return Polygon(*pts,stroke_width=0,fill_color=color,fill_opacity=opacity).set_z_index(0)
def query(ax):
    mu=predict(.45)[0]
    return VGroup(DashedLine(ax.c2p(.45,-1.02),ax.c2p(.45,mu),color=YELLOW,stroke_width=1.5,dash_length=.11),
                  Dot(ax.c2p(.45,mu),radius=.09,color=YELLOW).set_z_index(6))

class PRML72RelevanceVectorMachines(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.opening,self.materials,self.priors,self.evidence,self.geometry,self.selection,self.uncertainty,self.classification,self.return_to_question]):
            self.begin(i);self.current_note=None;method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml72_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
    def note(self,m):
        if self.current_note is not None:self.remove(self.current_note)
        self.current_note=m
        return FadeIn(m)
    def base(self,with_curve=False):
        ax,labels=graph();dots=points(ax)
        self.add(ax,labels,dots)
        if with_curve:self.add(curve(ax))
        return ax,dots
    def opening(self):
        small=VGroup(L('復習 7.1',-3.5,2.28,23,BLUE),
                     Dot((-2.2,.1,0),color=BLUE),Dot((2.2,.1,0),color=GREEN),
                     Line((0,-1.1,0),(0,1.2,0),color=GREEN),
                     Circle(radius=.15,color=YELLOW).move_to((-2.2,.1,0)),
                     Circle(radius=.15,color=YELLOW).move_to((2.2,.1,0)))
        self.beat(FadeIn(small))
        self.remove(small)
        ax,dots=self.base()
        self.beat(FadeIn(ax),FadeIn(dots),self.note(L('11個の観測点',color=BLUE)))
        candidates=VGroup(*[Circle(radius=.13,color=MUTED,stroke_width=1).move_to(d) for d in dots])
        self.beat(FadeIn(candidates),self.note(L('全部の点を使う？',color=YELLOW)))
        fit=curve(ax);rings=circles(dots)
        self.beat(FadeOut(candidates),Create(fit),FadeIn(rings),self.note(L('残った中心は2個',color=YELLOW)))
        mark=query(ax)
        self.beat(FadeIn(mark),self.note(F(r'x=0.45',color=YELLOW)))
        self.beat(Indicate(rings,color=YELLOW,scale_factor=1.08),self.note(L('2個で値も幅も？',color=YELLOW)))
        hill=kernel(ax,3,.7,PURPLE)
        self.beat(Create(hill),self.note(L('各点の山から調べる',color=PURPLE)))
    def materials(self):
        ax,dots=self.base()
        one=kernel(ax,3,1,PURPLE)
        self.beat(Create(one),Indicate(dots[3],color=YELLOW,scale_factor=1.25),self.note(L('点を中心にした山',color=PURPLE)))
        allhills=VGroup(*[kernel(ax,i,.3,PURPLE).set_opacity(.32) for i in range(11) if i!=3])
        self.beat(LaggedStart(*[Create(h) for h in allhills],lag_ratio=.09),self.note(L('11個の山が候補',color=PURPLE)))
        down=kernel(ax,3,-.8,RED)
        self.beat(ReplacementTransform(one,down),self.note(F(r'w_i<0',color=RED)))
        fit=curve(ax)
        self.beat(FadeOut(allhills),FadeOut(down),Create(fit),self.note(L('足し合わせた緑の曲線',color=GREEN)))
        self.beat(self.note(F(r'y(x)=\sum_n w_n k(x,x_n)+b',color=GREEN)),Indicate(fit,color=YELLOW,scale_factor=1.005))
        residuals=VGroup(*[Line(ax.c2p(float(x),float(predict(x)[0])),ax.c2p(float(x),float(t)),color=RED,stroke_width=1.5) for x,t in zip(X,T)])
        self.beat(Create(residuals),self.note(L('点と曲線のずれ＝観測雑音',color=RED)))
        self.beat(FadeOut(residuals),*[(dots[i].animate.set_opacity(.25)) for i in range(11) if i not in [3,7]],self.note(L('どの山を縮める？',color=YELLOW)))
    def priors(self):
        ax,dots=self.base(True)
        self.beat(self.note(L('復習 3.3：係数も分布',-1.5,2.45,24,BLUE)),Indicate(dots[3],color=YELLOW,scale_factor=1.25))
        # The inset shows one scalar coefficient and its Gaussian prior.
        baseline=Line((3.85,1.1,0),(6.2,1.1,0),color=MUTED,stroke_width=1.3)
        wide=FunctionGraph(lambda z:1.1+.75*np.exp(-.5*((z-5.03)/.48)**2),x_range=[3.85,6.2,.02],color=PURPLE)
        self.beat(Create(baseline),Create(wide),self.note(L('係数0を中心とする分布',color=PURPLE)))
        narrow=FunctionGraph(lambda z:1.1+.95*np.exp(-.5*((z-5.03)/.22)**2),x_range=[3.85,6.2,.02],color=PURPLE)
        hill=kernel(ax,3,MODEL['mean'][0],YELLOW)
        self.beat(ReplacementTransform(wide,narrow),Create(hill),self.note(F(r'\alpha_i\uparrow',color=YELLOW)))
        other_hill=kernel(ax,7,MODEL['mean'][1],RED)
        self.beat(hill.animate.set_opacity(.12),Create(other_hill),self.note(L('山ごとに別の精度',color=YELLOW)))
        posterior=FunctionGraph(lambda z:1.1+.87*np.exp(-.5*((z-5.31)/.24)**2),x_range=[3.85,6.2,.02],color=GREEN)
        self.beat(ReplacementTransform(narrow,posterior),self.note(L('データで係数の分布を更新',color=GREEN)),Indicate(dots[7],color=YELLOW,scale_factor=1.2))
        self.beat(FadeOut(hill),self.note(F(r'\alpha_i\to\infty\Rightarrow w_i\to0',color=YELLOW)),Indicate(other_hill,color=YELLOW,scale_factor=1.02))
        self.beat(FadeOut(other_hill),self.note(L('次は設定の選び方',color=YELLOW)),Indicate(posterior,color=YELLOW,scale_factor=1.02))
    def evidence(self):
        badge=L('復習 3.4・3.5',-3.6,2.44,23,BLUE)
        ax=Axes(x_range=[-3,3,1],y_range=[0,1.15,.25],x_length=9,y_length=3.5,tips=False,axis_config=dict(color=MUTED,stroke_width=1.2,include_ticks=False)).shift(DOWN*.65)
        prior=ax.plot(lambda z:np.exp(-z*z/2),x_range=[-3,3,.03],color=PURPLE)
        like=ax.plot(lambda z:np.exp(-.5*((z-.95)/.62)**2),x_range=[-3,3,.03],color=BLUE)
        prod=ax.plot(lambda z:np.exp(-z*z/2-.5*((z-.95)/.62)**2),x_range=[-3,3,.03],color=YELLOW)
        self.beat(FadeIn(ax),FadeIn(badge),Create(prior),Create(like))
        self.remove(badge)
        narrower=ax.plot(lambda z:np.exp(-.5*(z/.35)**2),x_range=[-3,3,.03],color=PURPLE)
        self.beat(ReplacementTransform(prior,narrower),self.note(L('広すぎても狭すぎても困る',color=PURPLE)))
        self.beat(ReplacementTransform(narrower,prior),Create(prod),self.note(L('重なりを全部足す',color=YELLOW)))
        self.beat(self.note(F(r'p(t\mid\alpha,\beta)=\int p(t\mid w,\beta)p(w\mid\alpha)\,dw',size=31,color=WHITE)),Indicate(prod,color=YELLOW,scale_factor=1.02))
        faded=prod.copy().set_opacity(.16)
        self.beat(ReplacementTransform(prod,faded),self.note(L('1個の山を外して比較',color=YELLOW)))
        self.beat(ReplacementTransform(faded,prod),self.note(L('証拠が高い設定を選ぶ',color=GREEN)))
        self.beat(Indicate(prod,color=YELLOW,scale_factor=1.02),self.note(L('観測2個で向きを見る',color=YELLOW)))
    def geometry(self):
        ax=Axes(x_range=[-2,2,1],y_range=[-2,2,1],x_length=5,y_length=4,tips=False,axis_config=dict(color=MUTED,stroke_width=1.2,include_ticks=False)).shift(DOWN*.42)
        target=Arrow(ax.c2p(0,0),ax.c2p(1.3,1.25),color=BLUE,buff=0)
        basis=Arrow(ax.c2p(0,0),ax.c2p(-.5,1.35),color=PURPLE,buff=0)
        self.beat(FadeIn(ax),GrowArrow(target),GrowArrow(basis),self.note(L('説明用：観測値2個の矢印',color=BLUE)))
        circle=Circle(radius=1.05,color=GREEN,stroke_width=2).move_to(ax.c2p(0,0))
        ellipse=Ellipse(width=1.25,height=3.1,color=RED,stroke_width=2).rotate(.4).move_to(ax.c2p(0,0))
        self.beat(Create(ellipse),Indicate(basis,color=YELLOW,scale_factor=1.08),self.note(L('ずれた向きへ伸ばすと損',color=RED)))
        self.beat(Create(circle),Indicate(ellipse,color=RED,scale_factor=1.02),self.note(L('丸い雑音と山の伸び',color=GREEN)))
        aligned=Arrow(ax.c2p(0,0),ax.c2p(1.15,1.05),color=PURPLE,buff=0)
        rotated=Ellipse(width=3.1,height=1.25,color=GREEN,stroke_width=2).rotate(PI/4).move_to(ax.c2p(0,0))
        self.beat(FadeOut(circle),ReplacementTransform(basis,aligned),ReplacementTransform(ellipse,rotated),self.note(L('向きが合えば観測を説明',color=GREEN)))
        self.beat(self.note(F(r'q_i^2>s_i\Rightarrow\alpha_i<\infty',size=32,color=YELLOW)),Indicate(aligned,color=YELLOW,scale_factor=1.05))
        self.beat(self.note(L('合う向きと重なりを比べる',color=PURPLE)),Indicate(target,color=YELLOW,scale_factor=1.04))
        self.beat(self.note(L('11候補を順に判定',color=YELLOW)),Indicate(rotated,color=YELLOW,scale_factor=1.02))
    def selection(self):
        ax,dots=self.base()
        rings=VGroup(*[Circle(radius=.13,color=MUTED,stroke_width=1).move_to(d) for d in dots])
        self.beat(FadeIn(rings),self.note(L('11個の点中心が候補',color=PURPLE)))
        knob=Dot((-3.3,2.1,0),radius=.08,color=YELLOW)
        track=Line((-3.3,2.1,0),(2.6,2.1,0),color=MUTED,stroke_width=2)
        self.beat(FadeIn(track),FadeIn(knob),self.note(L('1個ずつ精度を更新',color=YELLOW)))
        ev=F('3.648',3.95,2.1,23,YELLOW)
        fit=curve(ax)
        self.beat(knob.animate.move_to((2.6,2.1,0)),Create(fit),FadeIn(ev),self.note(L('周辺尤度が上がる',color=GREEN)))
        selected=circles(dots)
        self.beat(FadeOut(rings),FadeIn(selected),self.note(F(r'x=0.3,\;0.7',color=YELLOW)))
        left=kernel(ax,3,MODEL['mean'][0],PURPLE)
        right=kernel(ax,7,MODEL['mean'][1],RED)
        self.beat(Create(left),Create(right),self.note(L('上向きと下向きの山',color=YELLOW)))
        self.beat(FadeOut(left),FadeOut(right),Indicate(fit,color=YELLOW,scale_factor=1.01),self.note(L('2つで緑の波を作る',color=GREEN)))
        self.beat(Indicate(selected,color=YELLOW,scale_factor=1.08),self.note(L('曲線の幅はどう出す？',color=YELLOW)))
    def uncertainty(self):
        ax,dots=self.base(True)
        self.beat(self.note(L('復習 3.3：重みの分布を予測へ',color=BLUE)),Indicate(dots[3],color=YELLOW,scale_factor=1.25))
        samples=VGroup()
        for shift in [-.16,-.08,.08,.16]:
            samples.add(ax.plot(lambda z,s=shift:predict(z)[0]+s*np.exp(-.5*((z-.5)/.25)**2),x_range=[0,1,.02],color=PURPLE,stroke_width=1.1).set_opacity(.5))
        self.beat(FadeIn(samples),self.note(L('係数の候補から曲線も複数',color=PURPLE)))
        weight_band=band(ax,0,PURPLE,.22)
        self.beat(FadeOut(samples),FadeIn(weight_band),self.note(L('紫は係数から来る幅',color=PURPLE)))
        total_band=band(ax,1,YELLOW,.15)
        self.beat(FadeIn(total_band),self.note(F(r'\sigma^2(x)=\beta^{-1}+\phi(x)^T\Sigma\phi(x)',size=27,color=YELLOW)))
        mark=query(ax)
        self.beat(FadeIn(mark),self.note(F(r'x=0.45\quad\mu=0.288',color=GREEN)))
        self.beat(Indicate(total_band,color=YELLOW,scale_factor=1.01),self.note(F(r'[-0.053,\;0.628]',color=YELLOW)))
        edge=ax.c2p(1,-.8)
        outward=Arrow(edge,(6.0,edge[1],0),buff=0,color=RED,stroke_width=3)
        self.beat(GrowArrow(outward),self.note(L('外側の細い幅には注意',color=RED)))
    def classification(self):
        ax,dots=self.base(True)
        self.beat(self.note(L('復習 4.2・4.4',color=BLUE)),Indicate(dots[3],color=YELLOW,scale_factor=1.15))
        self.beat(self.note(L('説明用：回帰曲線を得点にする',color=YELLOW)),Indicate(curve(ax),color=YELLOW,scale_factor=1.01))
        sigmoid=ax.plot(lambda z:1/(1+np.exp(-predict(z)[0])),x_range=[0,1,.01],color=YELLOW,stroke_width=3)
        colors=VGroup()
        for i in range(50):
            z=(i+.5)/50
            probability=1/(1+np.exp(-predict(z)[0]))
            colors.add(Rectangle(width=9.3/50,height=.13,stroke_width=0,
                                 fill_color=interpolate_color(BLUE,GREEN,probability),fill_opacity=.95)
                       .move_to((ax.c2p(z,1.22)[0],1.67,0)))
        self.beat(Create(sigmoid),FadeIn(colors),FadeIn(L('黄色：確率',5.5,.52,19,YELLOW)),self.note(F(r'p(t=1\mid x,w)=\sigma(w^T\phi(x))',color=YELLOW)))
        rings=circles(dots)
        self.beat(FadeIn(rings),self.note(L('係数ごとの事前は共通',color=PURPLE)))
        bell=FunctionGraph(lambda z:.95+.75*np.exp(-.5*((z-5.5)/.21)**2),x_range=[4.85,6.15,.02],color=PURPLE)
        self.beat(Create(bell),self.note(L('分類の事後は局所的に近似',color=PURPLE)))
        self.beat(Indicate(bell,color=YELLOW,scale_factor=1.02),self.note(L('分類の学習には近似が必要',color=YELLOW)))
        self.beat(Indicate(rings,color=YELLOW,scale_factor=1.06),self.note(L('予測には少数の山を使う',color=GREEN)))
    def return_to_question(self):
        ax,dots=self.base()
        self.beat(self.note(L('冒頭と同じ11点',color=BLUE)),Indicate(dots,color=YELLOW,scale_factor=1.06))
        fit=curve(ax);rings=circles(dots)
        self.beat(Create(fit),FadeIn(rings),self.note(F(r'x=0.3,\;0.7',color=YELLOW)))
        mark=query(ax)
        self.beat(FadeIn(mark),self.note(F(r'x=0.45\quad\mu=0.288',color=GREEN)))
        total_band=band(ax,1,YELLOW,.16)
        self.beat(FadeIn(total_band),self.note(F(r'\sigma^2=0.02896\quad[-0.053,0.628]',size=27,color=YELLOW)))
        self.beat(Indicate(rings,color=YELLOW,scale_factor=1.08),self.note(L('証拠が不要な山を外した',color=GREEN)))
        self.beat(Indicate(total_band,color=YELLOW,scale_factor=1.01),self.note(L('幅は係数と雑音から',color=PURPLE)))
        self.beat(self.note(L('外の場所でも信じてよい？',color=YELLOW)),Indicate(mark,color=YELLOW,scale_factor=1.03))
