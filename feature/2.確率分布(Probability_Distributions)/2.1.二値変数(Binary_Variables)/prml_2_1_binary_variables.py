"""PRML 2.1: linked visual experiments in Manim Community."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
from manim import *
from binary_model import *
from caption_layout import jp, tex
from narrated_scene import NarratedScene
from make_voicevox_narration import MANIFEST
from narration_content import SCENES

BG='#10141F'
HEAD=ManimColor('#77D49A')
TAIL=ManimColor('#58B5ED')
YELLOW=ManimColor('#FFE079')
PRIOR_COLOR=ManimColor('#C29AFF')
POST=ManimColor('#FFB45B')
MUTED=ManimColor('#A8B2C5')


def readout(label,getter,pos,color=WHITE,places=3,size=26):
    number=DecimalNumber(getter(),num_decimal_places=places,font_size=size,color=color)
    group=VGroup(tex(label,size,color),number).arrange(RIGHT,buff=.12).move_to(pos)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group


def curve(ax,fn,color=POST,lo=0,hi=1):
    xs=np.linspace(lo,hi,301)
    return VMobject().set_points_as_corners([ax.c2p(x,float(fn(x))) for x in xs]).set_stroke(color,3)


def area(ax,fn,lo=0,hi=1,color=POST):
    xs=np.linspace(lo,max(lo+1e-5,hi),151)
    return Polygon(ax.c2p(lo,0),*[ax.c2p(x,float(fn(x))) for x in xs],ax.c2p(hi,0),
                   stroke_width=0,fill_color=color,fill_opacity=.25)


def coin(value,r=.27):
    color=HEAD if value else TAIL
    return VGroup(Circle(radius=r,color=color,fill_color=color,fill_opacity=.17),tex(str(value),29,WHITE))


def coins(values,y=1.85,r=.23):
    return VGroup(*[coin(x,r) for x in values]).arrange(RIGHT,buff=.22).move_to([0,y,0])


def pulse(m,color=YELLOW):
    return Circumscribe(m,color=color,buff=.10)


class PRML21BinaryVariables(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.timeline=[]
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        for i,method in enumerate([self.question,self.moments,self.fit,self.counts,self.prior,
                                  self.update,self.sequence,self.predict,self.uncertainty]):
            self.begin(i)
            if not self.audio_entry: raise RuntimeError('Generate matching narration before rendering')
            method()
            assert self.beat_index==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path('media/prml21_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def f(self,*parts,y=-2.45,size=32,colors=None):
        m=MathTex(*parts,font_size=size).move_to([0,y,0])
        if colors:
            for p,c in zip(m,colors):p.set_color(c)
        if m.width>12.6: raise ValueError('Formula exceeds safe width')
        return m

    def ax(self,xr=(0,1,.25),yr=(0,4,1),width=8.6,height=3.1,center=(0,0,0),xlabel=r'\mu',ylabel='確率密度'):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
            axis_config=dict(color=MUTED,stroke_width=1.4,include_numbers=True,font_size=19)).move_to(center)
        labels=VGroup(tex(xlabel,25).next_to(ax.x_axis,RIGHT,buff=.15),
            jp(ylabel,20,MUTED).move_to([-width/2+.4,center[1]+height/2+.35,0]))
        self.add(ax,labels)
        ax.labels=labels
        return ax

    def slider(self,t,lo=0,hi=1,label=r'\mu=',pos=(0,2.15,0),width=4):
        line=Line(LEFT*width/2,RIGHT*width/2,color=MUTED).move_to(pos)
        dot=Dot(color=YELLOW,radius=.07).add_updater(lambda m:m.move_to(line.point_from_proportion((t.get_value()-lo)/(hi-lo))))
        return VGroup(line,dot,readout(label,t.get_value,[pos[0]+width/2+1.0,pos[1],0],YELLOW,2))

    def discard(self,*mobjects):
        # Individual FadeIn/Transform animations may add children as scene roots.
        # Remove the complete family, including those roots, at scene transitions.
        return self.remove(*[m for root in mobjects for m in root.get_family()])

    def replace_formula(self,old,new):
        self.discard(old);self.add(new);return new

    def question(self):
        cs=VGroup(*[VGroup(Circle(radius=.53,color=HEAD),jp('表',34,HEAD)) for _ in range(3)]).arrange(RIGHT,buff=.5).move_to([-.8,.8,0])
        q=jp('？',55,YELLOW).move_to([3,.8,0])
        self.beat(phases=[('three-heads',self.sentence_duration(0),lambda:LaggedStart(*[FadeIn(c,shift=UP*.3) for c in cs],lag_ratio=.3)),
            ('next-question',self.sentence_duration(1),lambda:FadeIn(q)),
            ('question-emphasis',self.sentence_duration(2),lambda:pulse(q))])
        zeros=VGroup(coin(0,.5),jp('裏',26,TAIL)).arrange(DOWN,buff=.15).move_to([3,.7,0])
        self.beat(*[Transform(c,coin(1,.53).move_to(c),rate_func=lambda t:smooth(min(1,5*t))) for c in cs],Transform(q,zeros,rate_func=lambda t:smooth(min(1,5*t))))
        self.discard(cs,q)
        mu=ValueTracker(.5)
        ax=self.ax(xr=(-.5,1.5,.5),yr=(0,1,.25),width=6.5,xlabel='x',ylabel='確率')
        origin=ax.c2p(0,0).copy(); ux=ax.c2p(1,0)-origin; uy=ax.c2p(0,1)-origin
        ax.y_axis.shift(-.5*ux)
        ax.c2p=lambda x,y=0:origin+x*ux+y*uy
        ax.x_axis.ticks.set_opacity(0)
        chart=always_redraw(lambda:VGroup(*[Rectangle(width=1,height=max(.002,3.1*p),fill_color=c,fill_opacity=.8,stroke_width=0)
            .move_to(ax.c2p(i,0)+UP*max(.002,3.1*p)/2) for i,p,c in zip([0,1],bernoulli(mu.get_value()),[TAIL,HEAD])]))
        ax.x_axis.numbers.set_opacity(0)
        self.add(tex('0',22,TAIL).next_to(ax.c2p(0,0),DOWN,buff=.15),tex('1',22,HEAD).next_to(ax.c2p(1,0),DOWN,buff=.15))
        labels=VGroup(tex(r'1-\mu',27,TAIL).move_to([-4.9,.3,0]),tex(r'\mu',29,HEAD).move_to([4.5,.3,0]))
        sl=self.slider(mu);self.add(chart,labels,sl)
        f=self.f(r'p(x=1\mid\mu)=',r'\mu',r',\quad p(x=0\mid\mu)=',r'1-\mu',colors=[WHITE,HEAD,WHITE,TAIL])
        self.add(f)
        self.beat(mu.animate.set_value(.8),start_sentence=1)
        self.beat(mu.animate.set_value(.2))
        f=self.replace_formula(f,self.f(r'\operatorname{Bern}(x\mid\mu)=',r'\mu^x',r'(1-\mu)^{1-x}',colors=[WHITE,HEAD,TAIL]))
        sub=self.f(r'x=1:\quad \mu^1(1-\mu)^0=\mu',y=-1.98,size=27)
        self.add(sub)
        self.beat(pulse(sub),pulse(f[1]),start_sentence=1)
        self.discard(sub)
        sub=self.f(r'x=0:\quad\mu^0(1-\mu)^1=1-\mu',y=-1.98,size=27)
        self.add(sub)
        self.beat(pulse(sub),pulse(f[2]))

    def moments(self):
        mu=ValueTracker(.2)
        line=NumberLine(x_range=[0,1,.25],length=8,include_numbers=True,font_size=22).move_to([0,.1,0])
        weights=always_redraw(lambda:VGroup(*[Rectangle(width=.75,height=max(.002,2*p),fill_color=c,fill_opacity=.8,stroke_width=0)
            .move_to(line.n2p(x)+UP*max(.002,2*p)/2) for x,p,c in zip([0,1],bernoulli(mu.get_value()),[TAIL,HEAD])]))
        pivot=always_redraw(lambda:Triangle(color=YELLOW,fill_opacity=1).scale(.13).next_to(line.n2p(mu.get_value()),DOWN,buff=.1))
        mean=readout(r'\mathbb E[x]=',mu.get_value,[0,-1,0],YELLOW)
        self.add(line,weights,pivot,mean,self.slider(mu))
        self.beat(pulse(pivot))
        self.beat(mu.animate.set_value(.8))
        f=self.f(r'\mathbb E[x]=',r'0(1-\mu)',r'+',r'1\mu',r'=\mu',colors=[YELLOW,TAIL,WHITE,HEAD,YELLOW])
        self.beat(FadeIn(f,rate_func=lambda t:smooth(min(1,5*t))),LaggedStart(pulse(weights[0],TAIL),pulse(weights[1],HEAD),lag_ratio=.5))
        distances=always_redraw(lambda:VGroup(Line(line.n2p(0)+DOWN*.5,line.n2p(mu.get_value())+DOWN*.5,color=TAIL),
            Line(line.n2p(mu.get_value())+DOWN*.65,line.n2p(1)+DOWN*.65,color=HEAD)))
        self.add(distances)
        f=self.replace_formula(f,self.f(r'\operatorname{var}[x]=',r'(1-\mu)\mu^2',r'+',r'\mu(1-\mu)^2',r'=\mu(1-\mu)',size=28,colors=[WHITE,TAIL,WHITE,HEAD,YELLOW]))
        self.beat(mu.animate.set_value(.35))
        self.discard(line,weights,pivot,mean,distances)
        ax=self.ax(yr=(0,.27,.05),ylabel='分散',height=3)
        graph=curve(ax,lambda u:u*(1-u),YELLOW)
        dot=always_redraw(lambda:Dot(ax.c2p(mu.get_value(),mu.get_value()*(1-mu.get_value())),color=HEAD))
        self.add(graph,dot)
        self.beat(mu.animate.set_value(.5),end_sentence=1)
        self.beat(mu.animate.set_value(.95),end_sentence=1)

    def fit(self):
        cs=coins(OBS,.7,r=.34)
        count=self.f(r'N=8,\quad m=\sum_n x_n=5,\quad l=N-m=3',y=-.7)
        self.beat(LaggedStart(*[FadeIn(c) for c in cs],lag_ratio=.15),FadeIn(count),end_sentence=2)
        factors=VGroup(*[tex(r'\mu' if x else r'(1-\mu)',26,HEAD if x else TAIL) for x in OBS]).arrange(RIGHT,buff=.26).move_to([0,.7,0])
        f=self.f(r'p(\mathcal D\mid\mu)=\prod_{n=1}^N p(x_n\mid\mu)=',r'\mu^5',r'(1-\mu)^3',colors=[WHITE,HEAD,TAIL],size=29)
        self.beat(Transform(cs,factors,rate_func=lambda t:smooth(min(1,4*t))),FadeIn(f,rate_func=lambda t:smooth(min(1,5*t))))
        self.discard(cs,count)
        mu=ValueTracker(.15)
        ax=self.ax(yr=(0,.0055,.001),ylabel='尤度',height=2.55,center=(0,.2,0))
        graph=curve(ax,likelihood,YELLOW)
        point=always_redraw(lambda:Dot(ax.c2p(mu.get_value(),likelihood(mu.get_value())),color=POST))
        guide=always_redraw(lambda:DashedLine(ax.c2p(mu.get_value(),0),point.get_center(),color=POST))
        self.add(graph,point,guide,self.slider(mu))
        self.beat(mu.animate.set_value(.625))
        self.beat(mu.animate.set_value(.9))
        f=self.replace_formula(f,self.f(r'\ln p(\mathcal D\mid\mu)=m\ln\mu+(N-m)\ln(1-\mu)',size=29))
        deriv=self.f(r'\frac{m}{\mu}-\frac{N-m}{1-\mu}=0\quad\Longrightarrow\quad\mu_{\rm ML}=\frac mN',y=-1.88,size=25)
        self.add(deriv)
        self.beat(mu.animate.set_value(.625),pulse(deriv),end_sentence=2)
        top=coins(OBS,2.15,r=.18)
        # Remove slider to leave room for the permuted observations.
        for m in list(self.mobjects):
            if isinstance(m,VGroup) and len(m)==3 and any(isinstance(x,Line) for x in m): self.discard(m)
        self.add(top)
        self.beat(Transform(top,coins(OBS[::-1],2.15,r=.18),rate_func=lambda t:smooth(min(1,4*t))),pulse(deriv))

    def counts(self):
        rows=[]
        for chosen in itertools.combinations(range(4),2):
            rows.append(coins([int(i in chosen) for i in range(4)],0,r=.22))
        grid=VGroup(*rows).arrange_in_grid(rows=2,cols=3,buff=(.8,.7)).move_to([0,.5,0])
        self.beat(LaggedStart(*[FadeIn(row) for row in grid],lag_ratio=.15))
        f=self.f(r'p(m=2\mid N=4,\mu)=',r'6',r'\mu^2(1-\mu)^2',colors=[WHITE,YELLOW,HEAD])
        self.beat(FadeIn(f,rate_func=lambda t:smooth(min(1,5*t))),LaggedStart(*[pulse(r) for r in grid],lag_ratio=.12))
        f=self.replace_formula(f,self.f(r'\operatorname{Bin}(m\mid N,\mu)=',r'\binom Nm',r'\mu^m(1-\mu)^{N-m}',colors=[WHITE,YELLOW,HEAD]))
        choose=self.f(r'\binom Nm=\frac{N!}{m!(N-m)!}',y=-1.6,size=29)
        self.beat(FadeIn(choose,rate_func=lambda t:smooth(min(1,5*t))),pulse(grid[0]))
        self.discard(grid,choose)
        mu=ValueTracker(.25)
        ax=self.ax(xr=(-.6,8.6,1),yr=(0,.38,.1),ylabel='確率',xlabel='m',height=2.9,center=(0,-.1,0))
        chart=always_redraw(lambda:VGroup(*[Polygon(ax.c2p(i-.35,0),ax.c2p(i-.35,p),ax.c2p(i+.35,p),ax.c2p(i+.35,0),
            stroke_width=0,fill_color=HEAD,fill_opacity=.8) for i,p in enumerate(binomial(8,mu.get_value()))]))
        self.add(chart,self.slider(mu))
        self.beat(mu.animate.set_value(.7))
        f=self.replace_formula(f,self.f(r'\mathbb E[m]=N\mu,\qquad\operatorname{var}[m]=N\mu(1-\mu)',size=30))
        meanline=always_redraw(lambda:DashedLine(ax.c2p(8*mu.get_value(),0),ax.c2p(8*mu.get_value(),.35),color=YELLOW))
        self.add(meanline)
        self.beat(mu.animate.set_value(.5))
        self.beat(pulse(ax.x_axis),mu.animate.set_value(.625))

    def prior(self):
        ax=self.ax(yr=(0,4,1))
        g=curve(ax,lambda u:u**3,YELLOW)
        note=jp('３回とも表：最尤推定は１',25,YELLOW).move_to([0,2.15,0])
        tip=Dot(ax.c2p(1,1),color=YELLOW)
        self.add(note)
        self.beat(Create(g),GrowFromCenter(tip))
        self.discard(g,note,tip)
        # Fixed axes compare densities at their true height; every density has area 1.
        a,b=ValueTracker(1),ValueTracker(1)
        g=always_redraw(lambda:curve(ax,lambda u:beta_pdf(u,a.get_value(),b.get_value()),PRIOR_COLOR,
            lo=.005 if min(a.get_value(),b.get_value())<1 else 0,hi=.995 if min(a.get_value(),b.get_value())<1 else 1))
        fill=always_redraw(lambda:area(ax,lambda u:beta_pdf(u,a.get_value(),b.get_value()),color=PRIOR_COLOR))
        counters=VGroup(readout('a=',a.get_value,[-1.8,2.15,0],HEAD,1),readout('b=',b.get_value,[1.8,2.15,0],TAIL,1))
        self.add(fill,g,counters)
        self.beat(pulse(counters))
        self.beat(a.animate.set_value(3),b.animate.set_value(2))
        mean=always_redraw(lambda:DashedLine(ax.c2p(beta_mean(a.get_value(),b.get_value()),0),ax.c2p(beta_mean(a.get_value(),b.get_value()),3.5),color=YELLOW))
        self.add(mean)
        self.beat(a.animate.set_value(9),b.animate.set_value(6))
        f=self.f(r'\operatorname{Beta}(\mu\mid a,b)=',r'\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}',r'\mu^{a-1}(1-\mu)^{b-1}',size=29,colors=[WHITE,PRIOR_COLOR,WHITE])
        sweep=ValueTracker(.1)
        strip=always_redraw(lambda:area(ax,lambda u:beta_pdf(u,a.get_value(),b.get_value()),.1,sweep.get_value(),YELLOW))
        self.add(f,strip)
        self.beat(sweep.animate.set_value(.9),pulse(f[1]))
        self.discard(strip,f,mean,fill)
        f=self.f(r'\mathbb E[\mu]=\frac a{a+b},\qquad\operatorname{var}[\mu]=\frac{ab}{(a+b)^2(a+b+1)}',size=30)
        self.add(f)
        tailnote=jp('両端で密度は発散（端のごく近くは省略）',20,MUTED).move_to([0,.6,0])
        arrows=VGroup(*[Arrow(ax.c2p(x,2.9),ax.c2p(x,3.9),color=PRIOR_COLOR,buff=0) for x in [.005,.995]])
        def show_u():
            return AnimationGroup(a.animate.set_value(.7),b.animate.set_value(.7),rate_func=lambda t:smooth(min(1,3*t)))
        def label_u():
            self.add(tailnote,arrows)
            return pulse(g)
        def restore_prior():
            self.discard(tailnote,arrows)
            return AnimationGroup(a.animate.set_value(3),b.animate.set_value(2))
        self.beat(phases=[('moments',self.sentence_duration(0),lambda:pulse(f)),
            ('u-transition',1.,show_u),
            ('u-shaped',self.sentence_duration(1)-1.,label_u),
            ('chosen-prior',self.sentence_duration(2),restore_prior)])

    def update(self):
        ax=self.ax(yr=(0,2.6,.5),height=3)
        mult=ValueTracker(0);norm=ValueTracker(1)
        fn=lambda u:beta_pdf(u,3,2)*((1-mult.get_value())+mult.get_value()*u)*norm.get_value()
        g=always_redraw(lambda:curve(ax,fn,POST))
        base=curve(ax,lambda u:beta_pdf(u,3,2),PRIOR_COLOR).set_stroke(opacity=.35)
        xs=[.2,.4,.6,.8]
        stems=always_redraw(lambda:VGroup(*[Line(ax.c2p(u,0),ax.c2p(u,fn(u)),color=POST) for u in xs]))
        note=jp('事前分布 → 表を１回観測',24).move_to([0,2.15,0])
        self.add(g,base,stems,note)
        self.beat(LaggedStart(*[pulse(stem) for stem in stems],lag_ratio=.2))
        weights=VGroup(*[tex(str(u),23,HEAD).move_to(ax.c2p(u,beta_pdf(u,3,2)) + UP*.35) for u in xs])
        f=self.f(r'p(\mu\mid x=1)\propto',r'p(\mu)',r'\mu',colors=[WHITE,PRIOR_COLOR,HEAD])
        self.add(f,weights)
        self.beat(mult.animate.set_value(1))
        self.discard(weights)
        integral=readout(r'\int q(\mu)\,d\mu=',lambda:.6*norm.get_value(),[2.8,2.15,0],YELLOW,2)
        self.discard(note);self.add(integral,jp('面積を１へ戻す',24).move_to([-2.5,2.15,0]))
        self.beat(norm.animate.set_value(1/.6),start_sentence=1)
        self.discard(f)
        f=self.f(r'\mu^{a-1}(1-\mu)^{b-1}\times',r'\mu',r'=\mu^{(a+1)-1}(1-\mu)^{b-1}',size=29,colors=[PRIOR_COLOR,HEAD,POST])
        self.add(f)
        self.beat(pulse(f[1]),pulse(f[2]))
        self.discard(g,stems,integral)
        b=ValueTracker(2)
        g=always_redraw(lambda:curve(ax,lambda u:beta_pdf(u,4,b.get_value()),POST))
        self.add(g)
        self.discard(f)
        f=self.f(r'x=0:\quad\operatorname{Beta}(\mu\mid4,2)\ \longrightarrow\ \operatorname{Beta}(\mu\mid4,3)',size=29)
        self.add(f)
        self.beat(b.animate.set_value(3))
        self.discard(f)
        f=self.f(r'\operatorname{Beta}(\mu\mid a,b)\ \longrightarrow\ ',r'\operatorname{Beta}(\mu\mid a+m,b+l)',size=30,colors=[PRIOR_COLOR,POST])
        counts=VGroup(readout('a+m=',lambda:4,[-2,-1.95,0],HEAD,0),readout('b+l=',lambda:3,[2,-1.95,0],TAIL,0))
        self.add(f,counts)
        self.beat(pulse(counts),pulse(g))

    def sequence(self):
        ax=self.ax(yr=(0,3.4,1),height=2.7,center=(0,-.25,0))
        a,b=ValueTracker(3),ValueTracker(2)
        g=always_redraw(lambda:curve(ax,lambda u:beta_pdf(u,a.get_value(),b.get_value()),POST))
        cs=coins(OBS,2.1,r=.19).set_opacity(.2)
        counters=VGroup(readout('a=',a.get_value,[-2,-1.95,0],HEAD,1),readout('b=',b.get_value,[2,-1.95,0],TAIL,1))
        f=self.f(r'x=1:\ a\leftarrow a+1\qquad x=0:\ b\leftarrow b+1',size=29)
        self.add(g,cs,counters,f)
        self.beat(pulse(counters))
        def step(i, aa, bb):
            return lambda:AnimationGroup(a.animate.set_value(aa),b.animate.set_value(bb),cs[i].animate.set_opacity(1),lag_ratio=0)
        aa,bb=3,2
        for indices in [range(3),range(3,8)]:
            phases=[];duration=sum(c['end']-c['start'] for c in self.beat_cues())
            for i in indices:
                aa+=int(OBS[i]);bb+=1-int(OBS[i]);phases.append((f'observe-{i+1}',duration/len(indices),step(i,aa,bb)))
            self.beat(phases=phases)
        self.discard(f)
        f=self.f(r'(3,2)+(5,3)=(8,5)',colors=[POST],size=34)
        batch=curve(ax,lambda u:beta_pdf(u,8,5),PRIOR_COLOR)
        self.add(f)
        self.beat(Create(batch),pulse(counters))
        self.discard(batch)
        self.beat(Transform(cs,coins(OBS[::-1],2.1,r=.19),rate_func=lambda t:smooth(min(1,4*t))),pulse(g))
        self.beat(cs.animate.scale(.2).move_to([0,-1.95,0]).set_opacity(0),pulse(counters))

    def predict(self):
        ax=self.ax(yr=(0,3.2,1),height=3)
        fn=lambda u:beta_pdf(u,8,5)
        g=curve(ax,fn,POST)
        u=ValueTracker(.15)
        stem=always_redraw(lambda:Line(ax.c2p(u.get_value(),0),ax.c2p(u.get_value(),fn(u.get_value())),color=YELLOW))
        self.add(g,stem)
        self.beat(u.animate.set_value(.85))
        self.discard(stem)
        stop=ValueTracker(.002)
        weighted=curve(ax,lambda u:u*fn(u),YELLOW)
        fill=always_redraw(lambda:area(ax,lambda u:u*fn(u),0,stop.get_value(),YELLOW))
        label=jp('オレンジ：事後密度　黄色：候補の確率 × 密度',22).move_to([0,2.15,0])
        self.add(weighted,fill,label)
        self.beat(stop.animate.set_value(1))
        f=self.f(r'p(x=1\mid\mathcal D)=\int_0^1\mu p(\mu\mid\mathcal D)\,d\mu=\frac{a+m}{a+b+N}',size=29)
        answer=tex(r'\frac{8}{13}\approx0.615',36,YELLOW).move_to([-2,1.1,0])
        self.add(f)
        self.beat(FadeIn(answer,rate_func=lambda t:smooth(min(1,5*t))),pulse(fill))
        self.discard(ax,ax.labels,g,fill,weighted,label,answer,f)
        line=NumberLine(x_range=[.595,.63,.005],length=8,include_numbers=True,font_size=18,decimal_number_config={'num_decimal_places':3}).move_to([0,.4,0])
        p=ValueTracker(.6)
        marker=always_redraw(lambda:Dot(line.n2p(p.get_value()),color=POST))
        labels=VGroup(jp('事前の平均 0.600',23,PRIOR_COLOR).move_to([-3,-.4,0]),jp('観測の割合 0.625',23,HEAD).move_to([3,-.4,0]))
        f=self.f(r'\frac{a+b}{a+b+N}\frac{a}{a+b}+\frac{N}{a+b+N}\frac{m}{N}',size=32)
        self.add(line,marker,labels,f)
        self.beat(p.animate.set_value(8/13))
        self.discard(line,marker,labels,f)
        cs=coins([1,1,1],.8,.45)
        f=self.f(r'a=b=1:\quad p(x=1\mid1,1,1)=\frac{1+3}{1+1+3}=\frac45',size=33)
        self.add(cs,f)
        self.beat(FadeIn(f,rate_func=lambda t:smooth(min(1,5*t))),LaggedStart(*[pulse(c,HEAD) for c in cs],lag_ratio=.2))
        self.discard(cs,f)
        n=ValueTracker(8)
        line=NumberLine(x_range=[.60,.626,.005],length=8,include_numbers=True,font_size=18,decimal_number_config={'num_decimal_places':3}).move_to([0,.4,0])
        pred=lambda:(3+5*n.get_value()/8)/(5+n.get_value())
        marker=always_redraw(lambda:Dot(line.n2p(pred()),color=POST))
        f=self.f(r'\frac{3+5N/8}{5+N}\ \longrightarrow\ \frac58\quad(N\to\infty)',size=33)
        self.add(line,marker,f,readout('N=',n.get_value,[0,1.6,0],WHITE,0))
        self.beat(n.animate.set_value(80))

    def uncertainty(self):
        ax=self.ax(yr=(0,10,2),height=3)
        b=ValueTracker(1)
        g=always_redraw(lambda:curve(ax,lambda u:beta_pdf(u,9,b.get_value()),POST))
        old=curve(ax,lambda u:beta_pdf(u,9,1),PRIOR_COLOR).set_stroke(opacity=.35)
        v=readout(r'\operatorname{var}[\mu]=',lambda:beta_var(9,b.get_value()),[1.8,2.15,0],POST,5)
        self.add(g,old,v)
        self.beat(b.animate.set_value(2),start_sentence=1)
        h=curve(ax,lambda u:beta_pdf(u,10,1),HEAD)
        label=jp('緑：表の後　オレンジ：裏の後　紫：観測前',22).move_to([0,-2.05,0])
        self.add(label)
        self.beat(Create(h))
        self.discard(ax,ax.labels,g,old,h,v,label)
        r=variance_decomposition(9,1)
        # Heights share the same linear variance scale.
        def bar(x,val,color):
            return Rectangle(width=1.3,height=val*240,fill_color=color,fill_opacity=.8,stroke_width=0).move_to([x,-1+val*120,0])
        bars=VGroup(bar(-3,r['variances'][1],HEAD),bar(0,r['variances'][0],TAIL),bar(3,r['prior'],PRIOR_COLOR))
        labels=VGroup(jp('表の後（確率0.9）',20,HEAD).move_to([-3,-1.4,0]),jp('裏の後（確率0.1）',20,TAIL).move_to([0,-1.4,0]),jp('観測前',22,PRIOR_COLOR).move_to([3,-1.4,0]))
        nums=VGroup(*[tex(f'{val:.5f}',25,c).next_to(m,UP,buff=.15) for m,val,c in zip(bars,[r['variances'][1],r['variances'][0],r['prior']],[HEAD,TAIL,PRIOR_COLOR])])
        self.add(bars,labels,nums)
        f=self.f(r'0.9\times0.00689+0.1\times0.01240\approx0.00744<0.00818',size=27)
        self.beat(FadeIn(f,rate_func=lambda t:smooth(min(1,5*t))),LaggedStart(pulse(bars[0]),pulse(bars[1]),lag_ratio=.5))
        self.discard(nums,f,labels,bars)
        whole=bar(-2,r['prior'],PRIOR_COLOR)
        remaining=bar(2,r['remaining'],POST)
        moved=bar(2,r['moved'],YELLOW).next_to(remaining,UP,buff=0)
        labels=VGroup(jp('観測前の分散',24,PRIOR_COLOR).move_to([-2,-1.4,0]),jp('残る分散の平均 ＋ 平均の移動の分散',23).move_to([1.5,1.7,0]))
        f=self.f(r'\operatorname{var}[\mu]=',r'\mathbb E_{\mathcal D}[\operatorname{var}[\mu\mid\mathcal D]]',r'+\operatorname{var}_{\mathcal D}[\mathbb E[\mu\mid\mathcal D]]',size=27,colors=[PRIOR_COLOR,POST,YELLOW])
        self.add(whole,labels,f)
        self.beat(TransformFromCopy(whole,remaining),FadeIn(moved))
        self.discard(whole,remaining,moved,labels,f)
        line=NumberLine(x_range=[.8,.92,.02],length=8,include_numbers=True,font_size=20,decimal_number_config={'num_decimal_places':2}).move_to([0,0,0])
        dots=VGroup(Dot(line.n2p(9/11),radius=.08,color=TAIL),Dot(line.n2p(10/11),radius=.18,color=HEAD))
        f=self.f(r'\mathbb E_{\mathcal D}[\mathbb E[\mu\mid\mathcal D]]=\mathbb E[\mu]=0.9',size=32)
        self.add(line,dots,f)
        self.beat(*[d.animate.move_to(line.n2p(.9)) for d in dots])
        self.discard(line,dots,f)
        chain=VGroup(*[VGroup(jp(title,25,col),tex(formula,32,col)).arrange(DOWN,buff=.4) for title,formula,col in [
            ('観測',r'0,1,1,\ldots',HEAD),('尤度',r'\mu^m(1-\mu)^l',YELLOW),('事後',r'\operatorname{Beta}(a+m,b+l)',POST),('予測',r'\frac{a+m}{a+b+N}',TAIL)]]).arrange(RIGHT,buff=.6).move_to([0,.2,0])
        if chain.width>12.4: chain.scale_to_fit_width(12.4)
        self.beat(LaggedStart(*[FadeIn(c,shift=RIGHT*.15) for c in chain],lag_ratio=.3))
