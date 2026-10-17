"""PRML 6.4 — a prediction becomes a distribution over possible outcomes."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from gp_model import X, T, PROBE, ELL, NOISE_SD, MEAN_PROBE, SD_PROBE, kernel, predict, prior_samples, log_evidence

BG = '#10141F'
BLUE = ManimColor('#58C4DD')
YELLOW = ManimColor('#FFFF00')
GREEN = ManimColor('#83C167')
PURPLE = ManimColor('#9A72AC')
RED = ManimColor('#FC6255')
MUTED = ManimColor('#A8B2C5')

_fade_in = FadeIn
def FadeIn(mobject, **kwargs):
    kwargs.setdefault('rate_func', lambda t: smooth(min(1., 35*t)))
    return _fade_in(mobject, **kwargs)


def L(s, x=0, y=0, size=25, color=WHITE):
    return jp(s, size, color).move_to((x, y, 0))


def F(s, x=0, y=0, size=33, color=WHITE):
    m = tex(s, size, color).move_to((x, y, 0))
    if m.width > 11.5:
        m.scale_to_fit_width(11.5)
    return m


def axes():
    ax = Axes(x_range=[-1.5, 3., .5], y_range=[-2.7, 2.7, 1],
              x_length=9.5, y_length=4.55, tips=False,
              axis_config=dict(color=MUTED, stroke_width=1.3, include_ticks=False))
    ax.move_to((0, -.05, 0))
    labels = VGroup()
    for x in [-1, 0, 1, 2.5]:
        a = ax.c2p(x, -2.7)
        labels.add(Line(a, a+UP*.07, color=MUTED, stroke_width=1))
        labels.add(F(str(x), a[0], a[1]-.24, 17, MUTED))
    labels.add(F('x', ax.c2p(3,-2.7)[0]+.2, ax.c2p(3,-2.7)[1], 20))
    for y in [-2, 0, 2]:
        a=ax.c2p(-1.5,y)
        labels.add(F(str(y),a[0]-.27,a[1],16,MUTED))
    labels.add(F('t',ax.c2p(-1.5,2.7)[0]-.22,ax.c2p(-1.5,2.7)[1]+.15,20))
    return ax, labels


def path(ax, xs, ys, color=GREEN, width=3):
    return VMobject().set_points_smoothly([ax.c2p(float(x),float(y)) for x,y in zip(xs,ys)]).set_stroke(color,width)


def gp_band(ax, ell=ELL, noise=NOISE_SD, factor=2):
    xs=np.linspace(-1.5,3.,91)
    mean, var=predict(xs,ell,noise)
    sd=np.sqrt(var)
    upper=np.clip(mean+factor*sd,-2.65,2.65)
    lower=np.clip(mean-factor*sd,-2.65,2.65)
    band=Polygon(*[ax.c2p(float(x),float(y)) for x,y in zip(np.r_[xs,xs[::-1]],np.r_[lower,upper[::-1]])],
                 color=GREEN, stroke_opacity=0, fill_color=GREEN, fill_opacity=.17)
    return band,path(ax,xs,mean,GREEN,3)


def dots(ax, color=BLUE):
    return VGroup(*[Dot(ax.c2p(x,t),radius=.07,color=color) for x,t in zip(X,T)])


def marker(ax, x, color=YELLOW):
    mean,var=predict([x])
    return Dot(ax.c2p(x,mean[0]),radius=.09,color=color)


def section_card(section, name, left, right):
    content=VGroup(L(f'復習: {section} {name}',-2.2,2.05,28,WHITE),
                   L(left,-2.5,.25,27,BLUE),Arrow((-1.1,.25,0),(1.1,.25,0),buff=.08,color=YELLOW),
                   L(right,2.55,.25,27,GREEN))
    if section == '3.3':
        xs=np.linspace(-1.7,1.7,51)
        for phase,color in [(0,PURPLE),(.7,BLUE),(1.4,GREEN)]:
            ys=-1.25+.22*np.sin(2*xs+phase)
            content.add(VMobject().set_points_smoothly([(float(x),float(y),0) for x,y in zip(xs,ys)]).set_stroke(color,2.5))
    else:
        content.add(Ellipse(width=2.7,height=.95,color=PURPLE,stroke_width=2).rotate(PI/5).move_to((0,-1.2,0)))
        content.add(Line((.45,-1.95,0),(.45,-.45,0),color=YELLOW,stroke_width=2))
    border=SurroundingRectangle(content,buff=.28,color=YELLOW,stroke_width=1.1)
    return VGroup(border,content)


class PRML64GaussianProcesses(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        methods=[self.opening,self.functions,self.correlation,self.noise,
                 self.conditioning,self.distance,self.hyperparameters,self.return_to_question]
        for i, method in enumerate(methods):
            self.begin(i)
            method()
            assert self.beat_index==len(SCENES[i]['beats']),(i,self.beat_index)
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml64_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def opening(self):
        intro=VGroup(L('6.3  近い点からの予測',-2.4,.9,30,BLUE),F(r'\hat y_{\mathrm{NW}}(2.5)=0.983401',1,-.45,38,GREEN))
        self.beat(FadeIn(intro))
        self.remove(intro)
        ax,labs=axes();points=dots(ax)
        self.beat(Create(ax),FadeIn(labs),FadeIn(points))
        guess=VGroup(Dot(ax.c2p(PROBE,1),radius=.09,color=YELLOW),L('ここでも高い？',3.75,1.48,25,YELLOW))
        self.beat(FadeIn(guess),Indicate(points[-1],color=YELLOW,scale_factor=1.3))
        band,line=gp_band(ax)
        probe=marker(ax,PROBE)
        self.beat(FadeOut(guess),FadeIn(band),Create(line),FadeIn(probe))
        result=VGroup(F(r'm(2.5)=0.123',3.05,2.15,29,RED),L('幅は広い',3.45,1.65,23,YELLOW))
        self.beat(FadeIn(result),Indicate(probe,color=YELLOW,scale_factor=1.3))
        self.remove(result)
        setting=VGroup(F(r'\ell=0.7',-3.2,2.0,27,YELLOW),F(r'\sigma_{\epsilon}=0.2',3.15,2.,27,RED))
        self.beat(FadeIn(setting),Indicate(band,color=YELLOW,scale_factor=1.005))
        self.remove(ax,labs,points,band,line,probe,setting)
        question=VGroup(L('どうして平均が戻る？',0,.75,32,GREEN),L('なぜ幅が広がる？',0,-.4,32,YELLOW))
        self.beat(FadeIn(question))

    def functions(self):
        recap=section_card('3.3','重みと関数','重みの候補','曲線の候補')
        self.beat(FadeIn(recap))
        self.remove(recap)
        ax,labs=axes()
        grid=np.linspace(-1.5,3.,91)
        samples=prior_samples(grid,4)
        curves=VGroup(*[path(ax,grid,sample,c,2) for sample,c in zip(samples,[PURPLE,BLUE,GREEN,RED])])
        self.beat(Create(ax),FadeIn(labs),*[Create(curve) for curve in curves])
        choice=VGroup(*[Dot(ax.c2p(x,float(np.interp(x,grid,samples[0]))),radius=.08,color=YELLOW) for x in X])
        self.beat(Indicate(curves,color=YELLOW,scale_factor=1.005),FadeIn(choice))
        heights=VGroup(*[F(fr'y_{i+1}',ax.c2p(x,0)[0],2.1,25,YELLOW) for i,x in enumerate(X)])
        self.beat(FadeIn(heights),Indicate(choice,color=YELLOW,scale_factor=1.25))
        note=VGroup(L('平均',-2.3,1.52,26,BLUE),L('共分散',2.2,1.52,26,GREEN),F(r'\mathbb E[y]=0',0,-1.6,31,YELLOW))
        self.beat(FadeIn(note))
        self.remove(ax,labs,curves,choice,heights,note)
        formula=VGroup(F(r'y=\Phi w',0,.75,43,GREEN),L('重みを引く → 曲線が変わる',0,-.7,27,BLUE))
        self.beat(FadeIn(formula))
        self.remove(formula)
        pair=VGroup(Dot(LEFT*2,color=BLUE),Dot(RIGHT*2,color=GREEN),Line(LEFT*2,RIGHT*2,color=YELLOW,stroke_width=4),L('2点の高さは連動する？',0,-1.1,29))
        self.beat(FadeIn(pair))

    def correlation(self):
        ax,labs=axes()
        x2=ValueTracker(-.6)
        ell=ValueTracker(.7)
        left=Dot(ax.c2p(-1,1),radius=.1,color=BLUE)
        right=Dot(ax.c2p(x2.get_value(),1),radius=.1,color=GREEN)
        right.add_updater(lambda m:m.move_to(ax.c2p(x2.get_value(),1)))
        def connection():
            strength=float(kernel([-1],[x2.get_value()],ell.get_value())[0,0])
            return Line(left.get_center(),right.get_center(),color=YELLOW).set_stroke(
                width=1+7*strength,opacity=.08+.92*strength)
        link=always_redraw(connection)
        self.beat(Create(ax),FadeIn(labs),FadeIn(left),FadeIn(right),Create(link))
        helper=VGroup(L('同じ向きに揺れる',0,1.85,27,YELLOW),Arrow((-1.4,.7,0),(-1.4,1.2,0),color=BLUE),Arrow((1,.7,0),(1,1.2,0),color=GREEN))
        self.beat(FadeIn(helper))
        self.remove(helper)
        formula=F(r'k(x,z)=e^{-(x-z)^2/(2\ell^2)}',0,-1.65,40,YELLOW)
        self.beat(FadeIn(formula),Indicate(link,color=YELLOW,scale_factor=1.02))
        self.beat(x2.animate.set_value(2.5))
        x2.set_value(0.)
        value=always_redraw(lambda:F(r'\ell='+f'{ell.get_value():.1f}',3.3,1.88,27,YELLOW))
        self.add(value)
        self.beat(ell.animate.set_value(.35),Indicate(link,color=YELLOW,scale_factor=1.01))
        self.remove(ax,labs,left,right,link,formula,value)
        cells=VGroup()
        K=kernel(X,X)
        for i in range(3):
            for j in range(3):
                square=Square(.77,stroke_width=1.5,color=MUTED,fill_color=GREEN,fill_opacity=float(K[i,j])*.7)
                square.move_to(((j-1)*.83,(1-i)*.83,0))
                cells.add(square)
        gram=VGroup(L('近い点ほど明るい',0,2.05,26,YELLOW),cells,F(r'K_{nm}=k(x_n,x_m)',0,-1.68,38,WHITE))
        self.beat(FadeIn(gram))
        self.remove(gram)
        bridge=VGroup(L('曲線の高さ',-2.3,.6,29,PURPLE),F(r'+\ \epsilon_n',.8,.6,35,RED),L('観測した目標',0,-.75,29,BLUE))
        self.beat(FadeIn(bridge))

    def noise(self):
        ax,labs=axes();grid=np.linspace(-1.5,3.,91)
        sample=prior_samples(grid,1,seed=6430)[0]
        curve=path(ax,grid,np.clip(sample,-2.5,2.5),PURPLE,3)
        schematic=L('仕組みの模式図',0,2.1,24,YELLOW)
        self.beat(Create(ax),FadeIn(labs),Create(curve),FadeIn(schematic))
        latent=np.interp(X,grid,sample)
        ypoints=VGroup(*[Dot(ax.c2p(x,y),radius=.075,color=BLUE) for x,y in zip(X,latent)])
        self.beat(FadeIn(ypoints))
        observed=latent+np.array([.22,-.15,.18])
        red=VGroup(*[Dot(ax.c2p(x,y),radius=.08,color=RED) for x,y in zip(X,observed)])
        arrows=VGroup(*[Arrow(ax.c2p(x,a),ax.c2p(x,b),buff=.05,color=YELLOW,stroke_width=3) for x,a,b in zip(X,latent,observed)])
        self.beat(FadeIn(red),Create(arrows))
        label=L('各点で独立なノイズ',2.9,1.95,23,YELLOW)
        self.beat(FadeIn(label),Indicate(arrows,color=YELLOW,scale_factor=1.05))
        self.remove(ax,labs,curve,ypoints,red,arrows,label,schematic)
        identity=VGroup(F(r'C=K+0.04I',0,.75,43,GREEN),L('曲線の連動',-2.45,-.45,26,PURPLE),L('観測の揺れ',2.45,-.45,26,RED))
        self.beat(FadeIn(identity))
        self.remove(identity)
        ax,labs=axes();band,line=gp_band(ax)
        self.beat(Create(ax),FadeIn(labs),FadeIn(band),Create(line),FadeIn(dots(ax)))
        self.remove(ax,labs,band,line)
        next_line=VGroup(F(r'\begin{pmatrix}t_1\\t_2\\t_3\\t_*\end{pmatrix}',0,.1,40,YELLOW),L('見えた3点で新点を絞る',0,-1.7,26,GREEN))
        self.beat(FadeIn(next_line))

    def conditioning(self):
        recap=section_card('2.3','条件付きガウス','2点の分布','切り口の分布')
        self.beat(FadeIn(recap))
        self.remove(recap)
        ax=Axes(x_range=[-2.4,2.4,1],y_range=[-2.3,2.3,1],x_length=6.,y_length=4.1,tips=False,axis_config=dict(color=MUTED,include_ticks=False)).move_to((0,-.05,0))
        ellipses=VGroup(*[Ellipse(width=w,height=h,color=PURPLE,stroke_width=2).rotate(PI/4).move_to(ax.c2p(0,0)) for w,h in [(4.2,1.7),(3.1,1.25),(1.8,.75)]])
        cut=Line(ax.c2p(.85,-2),ax.c2p(.85,2),color=YELLOW,stroke_width=3)
        mark=Dot(ax.c2p(.85,.72),radius=.1,color=GREEN)
        self.beat(Create(ax),Create(ellipses),Create(cut),FadeIn(mark))
        self.remove(ax,ellipses,cut,mark)
        blocks=VGroup(F(r'C_{N+1}=\begin{pmatrix}C_N&k\\k^T&c\end{pmatrix}',0,.2,40,GREEN),L('既知点',-2.65,-1.65,23,BLUE),L('新点',2.55,-1.65,23,YELLOW))
        self.beat(FadeIn(blocks))
        self.remove(blocks)
        ax,labs=axes();points=dots(ax);band,line=gp_band(ax)
        self.add(ax,labs,points,band,line)
        mean=F(r'm(x_*)=k^T C_N^{-1}t',0,2.1,38,WHITE)
        self.beat(FadeIn(mean),Indicate(line,color=YELLOW,scale_factor=1.01))
        self.remove(mean)
        variance=F(r's_t^2(x_*)=c-k^T C_N^{-1}k',0,2.1,38,YELLOW)
        self.beat(FadeIn(variance),Indicate(band,color=YELLOW,scale_factor=1.01))
        self.remove(variance)
        local=VGroup(Dot(ax.c2p(1,.95),radius=.1,color=YELLOW),L('点の近くは細い',3.2,1.95,24,YELLOW))
        self.beat(FadeIn(local),Indicate(points[-1],color=YELLOW,scale_factor=1.3))
        self.remove(local)
        move=ValueTracker(1.)
        moving=Dot(radius=.1,color=YELLOW).add_updater(lambda m:m.move_to(ax.c2p(move.get_value(),float(predict([move.get_value()])[0][0]))))
        self.add(moving)
        self.beat(move.animate.set_value(PROBE))

    def distance(self):
        ax,labs=axes();points=dots(ax);band,line=gp_band(ax)
        self.beat(Create(ax),FadeIn(labs),FadeIn(points),FadeIn(band),Create(line))
        x=ValueTracker(1.)
        mark=Dot(radius=.095,color=YELLOW).add_updater(lambda m:m.move_to(ax.c2p(x.get_value(),float(predict([x.get_value()])[0][0]))))
        guide=always_redraw(lambda:DashedLine(ax.c2p(x.get_value(),-2.45),ax.c2p(x.get_value(),2.35),color=YELLOW,dash_length=.08,stroke_width=1.5))
        self.add(mark,guide)
        self.beat(x.animate.set_value(PROBE))
        kval=F(r'k(1,2.5)\approx 0.1007',2.6,2.05,27,YELLOW)
        self.beat(FadeIn(kval),Indicate(mark,color=YELLOW,scale_factor=1.3))
        self.remove(kval)
        zero=VGroup(F(r'\mathbb E[y(x)]=0',-2.7,2.05,27,PURPLE),DashedLine(ax.c2p(-1.5,0),ax.c2p(3,0),color=PURPLE,stroke_width=1))
        self.beat(FadeIn(zero),Indicate(line,color=YELLOW,scale_factor=1.01))
        self.remove(zero)
        prior=F(r's_{\mathrm{prior},t}^2=1.04',2.9,2.05,28,PURPLE)
        self.beat(FadeIn(prior),Indicate(band,color=YELLOW,scale_factor=1.005))
        self.remove(prior)
        numbers=VGroup(F(r'm(2.5)=0.123',-2.5,2.08,27,GREEN),F(r's_t(2.5)=1.014',2.5,2.08,27,YELLOW))
        self.beat(FadeIn(numbers),Indicate(mark,color=YELLOW,scale_factor=1.3))
        self.remove(numbers)
        equation=F(r'k(x_*,X)\downarrow\quad\Rightarrow\quad s_t^2(x_*)\uparrow',0,2.08,28,YELLOW)
        self.beat(FadeIn(equation),Indicate(band,color=YELLOW,scale_factor=1.01))

    def hyperparameters(self):
        ax,labs=axes();points=dots(ax)
        ell=ValueTracker(.7);noise=ValueTracker(.2)
        band=always_redraw(lambda:gp_band(ax,ell.get_value(),noise.get_value())[0])
        line=always_redraw(lambda:gp_band(ax,ell.get_value(),noise.get_value())[1])
        knob=always_redraw(lambda:F(r'\ell='+f'{ell.get_value():.1f}',-2.9,2.05,28,YELLOW))
        self.add(ax,labs,points,band,line,knob)
        self.beat(ell.animate.set_value(.35))
        self.beat(ell.animate.set_value(1.25))
        noise_label=always_redraw(lambda:F(r'\sigma_{\epsilon}='+f'{noise.get_value():.1f}',2.7,2.05,27,RED))
        self.add(noise_label)
        self.beat(noise.animate.set_value(.5))
        self.remove(ax,labs,points,band,line,knob,noise_label)
        candidates=VGroup(*[F(fr'\ell={v:.1f}\quad\ln p(t|\ell)={log_evidence(v):.2f}',0,1.05-i*.8,31,c) for i,(v,c) in enumerate([(.35,BLUE),(.7,GREEN),(1.25,WHITE)])])
        self.beat(FadeIn(candidates))
        label=L('観測3点の自然さを比べる',0,-1.65,26,YELLOW)
        self.beat(FadeIn(label),Indicate(candidates,color=YELLOW,scale_factor=1.02))
        self.remove(candidates,label)
        evidence=VGroup(F(r'\ln p(t|\theta)=-\frac12\ln|C|-\frac12t^TC^{-1}t-\frac N2\ln(2\pi)',0,.35,38,WHITE),L('設定で、観測の分布が変わる',0,-1.45,25,YELLOW))
        self.beat(FadeIn(evidence))
        self.remove(evidence)
        fixed=VGroup(F(r'\ell=0.7',-2.4,.3,34,YELLOW),F(r'\sigma_{\epsilon}=0.2',2.4,.3,34,RED),L('主例の固定値へ',0,-1.15,25,GREEN))
        self.beat(FadeIn(fixed))

    def return_to_question(self):
        ax,labs=axes();points=dots(ax);band,line=gp_band(ax)
        conditions=VGroup(F(r'X=(-1,0,1)',-2.5,2.1,26,BLUE),F(r'x_*=2.5',2.5,2.1,26,YELLOW))
        self.beat(Create(ax),FadeIn(labs),FadeIn(points),FadeIn(conditions))
        mark=marker(ax,PROBE)
        mean_label=F(r'm=0.123285',2.75,1.7,28,GREEN)
        self.beat(Create(line),FadeIn(mark),FadeIn(mean_label))
        self.remove(conditions,mean_label)
        interval=VGroup(F(r'\sigma_t=1.014353',-2.5,2.1,27,YELLOW),F(r'[-1.905,\ 2.152]',2.65,2.1,27,GREEN))
        self.beat(FadeIn(band),FadeIn(interval),Indicate(mark,color=YELLOW,scale_factor=1.3))
        self.remove(interval)
        compared=VGroup(F(r'6.3:\ 0.983401',-2.5,2.1,27,BLUE),F(r'6.4:\ 0.123285',2.5,2.1,27,GREEN))
        self.beat(FadeIn(compared))
        self.remove(compared)
        answer=L('遠方ほど観測の支えが薄い',0,2.1,28,YELLOW)
        self.beat(FadeIn(answer),Indicate(band,color=YELLOW,scale_factor=1.005))
        self.remove(answer)
        limit=L('固定したモデルの下での予測帯',0,2.1,27,PURPLE)
        self.beat(FadeIn(limit))
        self.remove(ax,labs,points,band,line,mark,limit)
        next_question=VGroup(L('入力が増えたら、どの方向が重要？',0,.7,28,YELLOW),L('関連度の選択・分類へ',0,-.6,27,GREEN))
        self.beat(FadeIn(next_question))
