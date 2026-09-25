"""PRML 1.2: linked area, distribution and prediction experiments in Manim CE."""
from pathlib import Path
import json
import math
import numpy as np
from manim import *
from video_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry
from narration_content import SCENES
from probability_model import *

BG='#10141F'
RED=ManimColor('#FF6B77')
BLUE=ManimColor('#58B5ED')
GREEN=ManimColor('#77D49A')
ORANGE=ManimColor('#FFB45B')
YELLOW=ManimColor('#FFE079')
PURPLE=ManimColor('#C29AFF')
MUTED=ManimColor('#A8B2C5')


def line_graph(ax,x,y,color=BLUE,opacity=1,width=3):
    x,y=np.asarray(x),np.asarray(y)
    o=ax.c2p(0,0)
    points=o+x[:,None]*(ax.c2p(1,0)-o)+y[:,None]*(ax.c2p(0,1)-o)
    return VMobject().set_points_as_corners(points).set_stroke(color,width,opacity)


def area_graph(ax,x,y,color=BLUE,opacity=.3):
    p=line_graph(ax,np.r_[x[0],x,x[-1],x[0]],np.r_[0,y,0,0],color)
    return p.set_stroke(width=0).set_fill(color,opacity)


def rect(w,h,center,color,opacity=.55):
    return Rectangle(width=max(w,.001),height=max(h,.001),stroke_color=color,stroke_width=2,fill_color=color,fill_opacity=opacity).move_to(center)


def number(label,getter,pos,color=WHITE,places=3):
    prefix=tex(label,26,color)
    value=DecimalNumber(getter(),num_decimal_places=places,font_size=26,color=color)
    g=VGroup(prefix,value).arrange(RIGHT,buff=.12).move_to(pos)
    anchor=value.get_left().copy()
    value.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return g


class PRML12ProbabilityTheory(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.boxes,self.rules,self.bayes,self.density,self.moments,self.likelihood,self.bias,self.parameters,self.regression,self.prediction]):
            self.begin(i)
            method()
            assert self.beat_index==len(self.story['beats']),self.story['id']
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml12_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def begin(self,index):
        self.clear()
        self.story=SCENES[index]
        self.beat_index=0
        self.subtitle=None
        self.equation=None
        self.legend_group=None
        self.add(jp(self.story['title'],32).move_to([0,3.35,0]))
        entry=self.manifest[self.story['id']]
        if not valid_entry(self.story,entry):
            raise RuntimeError('Generate matching narration before rendering: '+self.story['id'])
        self.audio_entry=entry
        self.durations=entry['beat_durations']
        self.boundaries=np.ceil(np.cumsum(self.durations)*config.frame_rate-1e-6)/config.frame_rate
        self.scene_start=float(self.time)
        self.add_sound(str(OUTPUT_DIR/f"{self.story['id']}.wav"))
        self.timeline.append(dict(id=self.story['id'],title=self.story['title'],start=self.scene_start,reference=self.story['reference'],audio=True,beats=[]))

    def legend(self,*items):
        if self.legend_group is not None:self.remove(self.legend_group)
        self.legend_group=VGroup(*[VGroup(Dot(radius=.045,color=c),jp(t,18,c)).arrange(RIGHT,buff=.1) for t,c in items]).arrange(RIGHT,buff=.5).move_to([0,2.77,0])
        self.add(self.legend_group)

    def formula(self,*parts,colors=None,size=31):
        new=MathTex(*parts,font_size=size).move_to([0,-2.45,0])
        if new.width>12.7:raise ValueError('Equation too wide: '+''.join(parts))
        if colors:
            for i,c in colors.items():new[i].set_color(c)
        old=self.equation
        self.equation=new
        return FadeIn(new) if old is None else ReplacementTransform(old,new)

    def axes(self,xr,yr,width=9,height=3.4,center=(0,.3,0),xlabel='x',ylabel='p(x)'):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,axis_config={'color':MUTED,'stroke_width':1.4,'include_ticks':True})
        ax.move_to(center)
        labels=VGroup()
        # Labels are placed on the outer edges so negative-y plots stay legible.
        for v in np.arange(xr[0],xr[1]+.001,xr[2]):
            labels.add(tex(f'{v:g}',17,MUTED).move_to(ax.c2p(v,yr[0])+DOWN*.22))
        for v in np.arange(yr[0],yr[1]+.001,yr[2]):
            labels.add(tex(f'{v:g}',17,MUTED).next_to(ax.c2p(xr[0],v),LEFT,buff=.13))
        labels.add(tex(xlabel,22).next_to(ax.c2p(xr[1],yr[0]),RIGHT,buff=.17))
        labels.add(tex(ylabel,22).next_to(ax.c2p(xr[0],yr[1]),UP,buff=.13))
        self.add(ax,labels)
        ax.labels=labels
        return ax

    def slider(self,tracker,lo,hi,pos,label,color=YELLOW,width=3):
        base=NumberLine(x_range=[lo,hi],length=width,include_ticks=False,color=MUTED,stroke_width=2).move_to(pos)
        knob=Dot(base.n2p(tracker.get_value()),color=color,radius=.075)
        knob.add_updater(lambda m:m.move_to(base.n2p(tracker.get_value())))
        g=VGroup(base,knob,tex(label,24,color).next_to(base,LEFT,buff=.2))
        g.add(number('',tracker.get_value,np.array(pos)+RIGHT*(width/2+.5),color,2))
        self.add(g)
        return g

    def boxes(self):
        self.legend(('リンゴ',GREEN),('オレンジ',ORANGE))
        fruit=VGroup(Circle(.23,color=ORANGE,fill_opacity=1),jp('?',42)).arrange(RIGHT,buff=.5).move_to([0,.1,0])
        self.beat(FadeIn(fruit,shift=UP*.4))
        boxes=VGroup(); fruits=VGroup()
        for x,c,n in [(-2.6,RED,1),(2.6,BLUE,4)]:
            box=RoundedRectangle(width=3.1,height=1.8,corner_radius=.13,color=c).move_to([x,.5,0]); boxes.add(box)
            dots=VGroup(*[Circle(.14,color=GREEN if k<n else ORANGE,fill_opacity=1).move_to([x-.9+k*.45,.5,0]) for k in range(4 if n==1 else 5)])
            fruits.add(dots)
        self.beat(FadeOut(fruit),Create(boxes),LaggedStart(*[FadeIn(g) for g in fruits],lag_ratio=.2))
        q=ValueTracker(.5)
        s=self.slider(q,.1,.9,[0,-1.3,0],'p(B=r)',RED,4)
        self.beat(Indicate(fruits,scale_factor=1.08),self.formula(r'p(B=r)+p(B=b)=1'))
        self.beat(q.animate.set_value(.3))
        labels=VGroup(tex('B=r',28,RED).move_to([-2.6,1.9,0]),tex('B=b',28,BLUE).move_to([2.6,1.9,0]))
        self.beat(FadeIn(labels),self.formula(r'B\in\{r,b\},\qquad F\in\{a,o\}'))
        oranges=VGroup(*[d for g in fruits for d in g if d.get_color()==ORANGE])
        self.beat(Indicate(oranges,scale_factor=1.15),self.formula(r'p(B=r\mid F=o)=\ ?'))

    def mosaic(self,q=.3,ar=.75,ab=.2):
        w,h=7,2.65; left=-3.5; bottom=-1
        cells=VGroup()
        for x,width,a,c in [(left,w*q,ar,RED),(left+w*q,w*(1-q),ab,BLUE)]:
            cells.add(rect(width,h*a,[x+width/2,bottom+h*a/2,0],c,.65))
            cells.add(rect(width,h*(1-a),[x+width/2,bottom+h*a+h*(1-a)/2,0],GREEN,.2))
        return cells

    def rules(self):
        self.legend(('赤い箱のオレンジ',RED),('青い箱のオレンジ',BLUE),('リンゴ',GREEN))
        boxes=VGroup(rect(2.1,2.65,[-2.45,.325,0],RED,.22),rect(4.9,2.65,[1.05,.325,0],BLUE,.22))
        self.beat(FadeIn(boxes))
        cells=self.mosaic()
        self.beat(ReplacementTransform(boxes,cells))
        labels=VGroup(tex('0.3',25,RED).move_to([-2.45,-1.4,0]),tex('0.7',25,BLUE).move_to([1.05,-1.4,0]),tex('0.75',24,RED).move_to([-4.1,0,0]),tex('0.2',24,BLUE).move_to([4.05,-.73,0]))
        self.beat(FadeIn(labels),self.formula(r'p(B,F)=',r'p(F\mid B)',r'p(B)',colors={1:ORANGE,2:RED}))
        j=box_joint()
        values=VGroup(tex(f'{j[0,0]:.3f}',28).move_to(cells[0]),tex(f'{j[1,0]:.3f}',28).move_to(cells[2]))
        self.beat(FadeIn(values),Indicate(cells[0],scale_factor=1.02))
        # Common height h=2.65: area is conserved while the orange widths add.
        targets=VGroup(rect(7*j[0,0],2.65,[-3.5+7*j[0,0]/2,.325,0],RED,.65),rect(7*j[1,0],2.65,[-3.5+7*j[0,0]+7*j[1,0]/2,.325,0],BLUE,.65))
        self.beat(FadeOut(values),FadeOut(labels),FadeOut(cells[1]),FadeOut(cells[3]),Transform(cells[0],targets[0]),Transform(cells[2],targets[1]),self.formula(r'p(F=o)=0.225+0.140=0.365'))
        self.beat(self.formula(r'p(F)=\sum_B p(B,F)'),Indicate(VGroup(cells[0],cells[2]),scale_factor=1.025))
        outline=SurroundingRectangle(VGroup(cells[0],cells[2]),color=YELLOW,buff=.08)
        self.beat(Create(outline),self.formula(r'p(B\mid F)=\frac{p(B,F)}{p(F)}\quad(p(F)>0)'))

    def bayes(self):
        self.legend(('赤い箱由来',RED),('青い箱由来',BLUE))
        cells=self.mosaic();self.add(cells)
        self.beat(cells[1].animate.set_opacity(.07),cells[3].animate.set_opacity(.07))
        self.remove(cells[1],cells[3])
        j=box_joint()[:,0]; total=j.sum()
        strip=VGroup(*[rect(7*j[k],2.65,[-3.5+7*j[:k].sum()+3.5*j[k],.325,0],[RED,BLUE][k],.65) for k in range(2)])
        normalized=VGroup(*[rect(7*j[k]/total,2.65,[-3.5+7*j[:k].sum()/total+3.5*j[k]/total,.325,0],[RED,BLUE][k],.65) for k in range(2)])
        self.beat(phases=[('equal-height area',self.sentence_duration(0),lambda:AnimationGroup(Transform(cells[0],strip[0]),Transform(cells[2],strip[1]))),('normalize',self.sentence_duration(1),lambda:AnimationGroup(Transform(cells[0],normalized[0]),Transform(cells[2],normalized[1])))])
        val=tex(f'{box_posterior()[0]*100:.1f}\\%',40,WHITE).move_to([-1,0.4,0])
        self.beat(FadeIn(val),self.formula(r'\frac{0.225}{0.225+0.140}=0.6164\ldots'))
        self.beat(self.formula(r'p(B\mid F)=',r'\frac{p(F\mid B)p(B)}{p(F)}',colors={0:RED,1:ORANGE}),Indicate(val))
        self.remove(cells, cells[0], cells[2], val)
        q=ValueTracker(.3); a=ValueTracker(.2)
        def bars():
            z=box_posterior(q.get_value(),.75,a.get_value())[0]
            return VGroup(rect(7*z,2.65,[-3.5+3.5*z,.325,0],RED,.65),rect(7*(1-z),2.65,[3.5*z,.325,0],BLUE,.65))
        dynamic=always_redraw(bars);self.add(dynamic)
        self.slider(q,.05,.5,[-2.3,-1.6,0],'p(r)',RED,2)
        self.slider(a,.2,.75,[2.8,-1.6,0],'p(o|b)',BLUE,2)
        value=number('p(r|o)=',lambda:box_posterior(q.get_value(),.75,a.get_value())[0],[0,2.1,0],RED)
        self.add(value)
        self.beat(q.animate.set_value(.1))
        self.beat(a.animate.set_value(.75))
        self.beat(self.formula(r'p(B,F)=p(B)p(F)\quad\Longrightarrow\quad p(B\mid F)=p(B)'),Circumscribe(value))

    def density(self):
        self.legend(('密度',GREEN),('区間の確率',BLUE))
        ax=self.axes([-3,3,1],[0,1.5,.5],height=3.3)
        sig=ValueTracker(.8); left=ValueTracker(-.2); right=ValueTracker(.2)
        x=np.linspace(-3,3,241)
        curve=always_redraw(lambda:line_graph(ax,x,gaussian(x,0,sig.get_value()),GREEN))
        def fill():
            u=np.linspace(left.get_value(),right.get_value(),100)
            return area_graph(ax,u,gaussian(u,0,sig.get_value()),BLUE)
        area=always_redraw(fill)
        self.add(area,curve)
        prob=lambda:.5*(math.erf(right.get_value()/sig.get_value()/np.sqrt(2))-math.erf(left.get_value()/sig.get_value()/np.sqrt(2)))
        read=number('P=',prob,[3.3,1.95,0],BLUE);self.add(read)
        self.beat(left.animate.set_value(-1.2),right.animate.set_value(1.2))
        self.beat(left.animate.set_value(.35),right.animate.set_value(.50),self.formula(r'P(x<X<x+\Delta x)\approx p(x)\Delta x'))
        self.beat(left.animate.set_value(-3),right.animate.set_value(3),self.formula(r'P(a<X<b)=\int_a^b p(x)\,dx,\qquad \int_{-\infty}^{\infty}p(x)\,dx=1',size=28))
        self.beat(sig.animate.set_value(.28))
        sig.set_value(.8);right.set_value(-2.8)
        self.beat(right.animate.set_value(3),self.formula(r'P(z)=\int_{-\infty}^{z}p(x)\,dx,\qquad P\prime(z)=p(z)'))
        self.remove(area,curve,read,ax,ax.labels)
        ax=self.axes([0,2.5,.5],[0,1.2,.4],xlabel=r'x\ \to\ y',ylabel='p',height=3.3)
        scale=ValueTracker(1)
        box=always_redraw(lambda:Polygon(ax.c2p(0,0),ax.c2p(scale.get_value(),0),ax.c2p(scale.get_value(),1/scale.get_value()),ax.c2p(0,1/scale.get_value()),color=BLUE,fill_opacity=.4))
        self.add(box)
        self.beat(scale.animate.set_value(2),self.formula(r'y=2x,\qquad p_y(y)=\tfrac12 p_x(y/2)'))
        self.beat(self.formula(r'p_y(y)=p_x(g(y))|g\prime(y)|,\quad p(x)=\int p(x,y)\,dy',size=29),Circumscribe(box))

    def moments(self):
        self.legend(('確率',BLUE),('平均',YELLOW),('二乗の寄与',PURPLE))
        ax=self.axes([0,4,1],[0,.8,.2],height=3.2,ylabel='p(x)')
        base=np.array([.05,.2,.5,.2,.05]);shift=np.array([.01,.04,.15,.39,.41]);spread=np.array([.35,.10,.10,.10,.35])
        q=ValueTracker(0);wide=ValueTracker(0); xx=np.arange(5)
        probs=lambda:(1-wide.get_value())*((1-q.get_value())*base+q.get_value()*shift)+wide.get_value()*spread
        mean=lambda:float(xx@probs())
        bars=always_redraw(lambda:VGroup(*[Polygon(ax.c2p(k-.13,0),ax.c2p(k+.13,0),ax.c2p(k+.13,p),ax.c2p(k-.13,p),color=BLUE,fill_opacity=.6) for k,p in enumerate(probs())]))
        mark=always_redraw(lambda:Line(ax.c2p(mean(),0),ax.c2p(mean(),.65),color=YELLOW))
        self.add(bars,mark)
        read=number(r'\mathbb E[x]=',mean,[2.5,2.1,0],YELLOW,2);self.add(read)
        self.beat(self.formula(r'\mathbb E[x]=\sum_x xp(x)'),Circumscribe(read))
        self.beat(q.animate.set_value(1),self.formula(r'\mathbb E[f]=\sum_x p(x)f(x)'))
        q.set_value(0)
        self.beat(wide.animate.set_value(1))
        squares=VGroup()
        for k,p in enumerate(spread):
            side=.3*abs(k-2)*np.sqrt(p)
            if side: squares.add(Square(side_length=side,color=PURPLE,fill_opacity=.7).move_to(ax.c2p(k,.65)))
        self.beat(FadeIn(squares),self.formula(r'\operatorname{var}[x]=\mathbb E[(x-\mathbb E[x])^2]=\mathbb E[x^2]-\mathbb E[x]^2',size=28))
        self.remove(ax,ax.labels,bars,mark,read,squares)
        self.legend(('観測の組',BLUE),('共分散',PURPLE))
        ax=self.axes([-2,2,1],[-2,2,1],height=3.3,ylabel='y')
        xx=np.linspace(-1.7,1.7,31);noise=.25*np.sin(np.arange(31)*2.1);noise-=noise.mean()
        slope=ValueTracker(.8);bend=ValueTracker(0)
        yy=lambda:(1-bend.get_value())*(slope.get_value()*xx+noise)+bend.get_value()*(xx**2-xx.mean()**2-1)
        cloud=always_redraw(lambda:VGroup(*[Dot(ax.c2p(a,b),radius=.05,color=BLUE) for a,b in zip(xx,yy())]))
        cov=lambda:float(np.mean((xx-xx.mean())*(yy()-yy().mean())))
        cv=number(r'\operatorname{cov}[x,y]=',cov,[2.8,2.1,0],PURPLE);self.add(cloud,cv)
        self.beat(slope.animate.set_value(-.8),self.formula(r'\operatorname{cov}[x,y]=\mathbb E[(x-\mathbb E[x])(y-\mathbb E[y])] ',size=29))
        self.beat(bend.animate.set_value(1),self.formula(r'y=x^2-1,\qquad \operatorname{cov}[x,y]=0'),end_sentence=1)
        self.beat(self.formula(r'\mathbb E[f]\approx\frac1N\sum_n f(x_n),\quad \Sigma_{ij}=\operatorname{cov}[x_i,x_j]',size=29),Indicate(cloud,scale_factor=1.03))

    def likelihood(self):
        self.legend(('ガウス密度',RED),('観測とその密度',BLUE))
        ax=self.axes([-3,3,1],[0,.8,.2],height=3.1,center=(0,.25,0))
        mu=ValueTracker(-.7);sig=ValueTracker(.6);xx=np.linspace(-3,3,241)
        self.slider(mu,-1,1,[-2.3,2.25,0],r'\mu',YELLOW,2)
        self.slider(sig,.5,1.2,[2,2.25,0],r'\sigma',RED,2)
        curve=always_redraw(lambda:line_graph(ax,xx,gaussian(xx,mu.get_value(),sig.get_value()),RED));self.add(curve)
        meanline=always_redraw(lambda:Line(ax.c2p(mu.get_value(),0),ax.c2p(mu.get_value(),.72),color=YELLOW));self.add(meanline)
        self.beat(self.formula(r'\mathcal N(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]',size=31))
        self.beat(mu.animate.set_value(.7))
        self.beat(sig.animate.set_value(1.15),self.formula(r'\mathbb E[x]=\mu,\quad \operatorname{var}[x]=\sigma^2,\quad \beta=1/\sigma^2',colors={0:YELLOW}))
        dots=VGroup(*[Dot(ax.c2p(x,0),color=BLUE,radius=.065) for x in OBS])
        heights=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,0),ax.c2p(x,float(gaussian(x,mu.get_value(),sig.get_value()))),color=BLUE,stroke_width=2) for x in OBS]))
        self.add(heights)
        self.beat(FadeIn(dots),self.formula(r'L(\mu,\sigma^2)=\prod_{n=1}^N\mathcal N(x_n|\mu,\sigma^2)'))
        ll=lambda:float(np.log(gaussian(OBS,mu.get_value(),sig.get_value())).sum())
        read=number(r'\ln L=',ll,[2.9,1.9,0],BLUE,2);self.add(read)
        self.beat(mu.animate.set_value(-.8))
        self.beat(mu.animate.set_value(float(OBS.mean())),sig.animate.set_value(float(OBS.std())),self.formula(r'\ln L=-\frac{1}{2\sigma^2}\sum_n(x_n-\mu)^2-\frac N2\ln(2\pi\sigma^2)',size=30))
        self.beat(self.formula(r'\mu_{\rm ML}=\frac1N\sum_nx_n,\quad\sigma^2_{\rm ML}=\frac1N\sum_n(x_n-\mu_{\rm ML})^2',size=29),Indicate(dots,scale_factor=1.08))

    def bias(self):
        self.legend(('真の分布',GREEN),('2点からの推定',RED))
        ax=self.axes([-3,3,1],[0,1.2,.4],height=3.2)
        xx=np.linspace(-3,3,241);truth=line_graph(ax,xx,gaussian(xx),GREEN);self.add(truth)
        # Three real draws with visible spread; no fabricated normal-fit values.
        pairs=PAIRS[[7,3,4]]
        dotgroup=lambda pair:VGroup(*[Dot(ax.c2p(x,0),color=BLUE,radius=.07) for x in pair])
        fitted=lambda pair:line_graph(ax,xx,gaussian(xx,pair.mean(),pair.std()),RED)
        dots=dotgroup(pairs[0]);curve=fitted(pairs[0])
        self.beat(FadeIn(dots),Create(curve))
        self.beat(phases=[('draw 2',self.sentence_duration(0),lambda:AnimationGroup(Transform(dots,dotgroup(pairs[1])),Transform(curve,fitted(pairs[1])))),('draw 3',self.sentence_duration(1),lambda:AnimationGroup(Transform(dots,dotgroup(pairs[2])),Transform(curve,fitted(pairs[2]))))])
        self.remove(ax,ax.labels,truth,dots,curve)
        ax=self.axes([0,4000,1000],[0,1.2,.4],xlabel='K',ylabel=r'\overline{\sigma^2_{ML}}',height=3.2)
        theory=DashedLine(ax.c2p(0,.5),ax.c2p(4000,.5),color=YELLOW)
        truevar=DashedLine(ax.c2p(0,1),ax.c2p(4000,1),color=GREEN)
        self.legend(('真の分散 = 1',GREEN),('推定分散の平均',RED),('理論値 = 0.5',YELLOW))
        self.add(theory,truevar)
        # Display all partial averages, revealing actual computed samples over time.
        running=line_graph(ax,np.arange(1,4001),VAR_RUNNING,RED)
        self.beat(Create(running),self.formula(r'N=2,\quad \sigma^2=1,\quad \mathbb E[\sigma^2_{\rm ML}]=0.5'))
        self.remove(ax,ax.labels,theory,truevar,running)
        self.legend(('推定分散の期待値 / 真の分散',YELLOW))
        ax=self.axes([2,30,7],[0,1,.25],xlabel='N',ylabel=r'(N-1)/N',height=3.2)
        ns=np.arange(2,31);curve=line_graph(ax,ns,(ns-1)/ns,YELLOW)
        self.beat(Create(curve),self.formula(r'\mathbb E[\sigma^2_{\rm ML}]=\frac{N-1}{N}\sigma^2'))
        self.beat(self.formula(r'\widetilde\sigma^2=\frac{1}{N-1}\sum_n(x_n-\mu_{\rm ML})^2\quad(N>1)'),Indicate(curve,scale_factor=1.02))

    def parameters(self):
        self.legend(('尤度',ORANGE),('事前密度',BLUE),('事後密度',PURPLE))
        ax=self.axes([0,1,.2],[0,2.6,.65],xlabel=r'\theta',ylabel=r'L,\ p',height=3.2)
        xx=np.linspace(0,1,241);like=line_graph(ax,xx,xx**3,ORANGE)
        self.add(jp('D：観測済みデータ（表が3回）',18,MUTED).move_to([2.3,2.28,0]))
        self.beat(Create(like),self.formula(r'\theta=P(\mathrm{heads}),\quad L(\theta)=\theta^3'))
        dot=Dot(ax.c2p(.5,.125),color=YELLOW)
        self.add(dot)
        self.beat(MoveAlongPath(dot,line_graph(ax,np.linspace(.5,1,100),np.linspace(.5,1,100)**3)),self.formula(r'\theta_{\rm ML}=1'))
        prior=line_graph(ax,xx,coin_prior(xx),BLUE)
        self.beat(Create(prior),self.formula(r'p(\theta)=6\theta(1-\theta)'))
        product=line_graph(ax,xx,coin_prior(xx)*xx**3,PURPLE)
        post=line_graph(ax,xx,coin_posterior(xx),PURPLE)
        self.beat(phases=[('multiply',self.sentence_duration(0)*.5,lambda:Transform(prior,product)),('normalize',self.sentence_duration(0)*.5+self.sentence_duration(1),lambda:Transform(prior,post))])
        mean=5/7
        mark=Line(ax.c2p(mean,0),ax.c2p(mean,2.5),color=YELLOW)
        self.beat(Create(mark),self.formula(r'P(\mathrm{heads\ next}|D)=\int_0^1\theta\,p(\theta|D)\,d\theta=\frac57',size=29))
        self.beat(self.formula(r'p(\mathbf w|D)=\frac{p(D|\mathbf w)p(\mathbf w)}{\int p(D|\mathbf w)p(\mathbf w)\,d\mathbf w}',size=31),Indicate(prior,scale_factor=1.03))

    def regression(self):
        self.legend(('観測',BLUE),('平均曲線',RED),('ノイズ密度',YELLOW))
        ax=self.axes([0,1,.25],[-1.4,1.4,.7],height=3.1,ylabel='t')
        xx=np.linspace(0,1,241);logalpha=ValueTracker(np.log(ALPHA));offset=ValueTracker(.3);cursor=ValueTracker(.1)
        weights=lambda:posterior(np.exp(logalpha.get_value()))[0]
        values=lambda u:phi(u)@weights()+offset.get_value()
        curve=always_redraw(lambda:line_graph(ax,xx,values(xx),RED));self.add(curve)
        def bell():
            x=cursor.get_value();m=float(values(x));t=np.linspace(m-.45,m+.45,100)
            return line_graph(ax,x+.045*gaussian(t,m,.15),t,YELLOW)
        density=always_redraw(bell);self.add(density)
        self.beat(cursor.animate.set_value(.8))
        self.beat(cursor.animate.set_value(.25),self.formula(r'p(t|x,\mathbf w,\beta)=\mathcal N(t|y(x,\mathbf w),\beta^{-1})',size=31))
        dots=VGroup(*[Dot(ax.c2p(x,t),color=BLUE,radius=.06) for x,t in zip(X,T)])
        residuals=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,float(values(x))),color=YELLOW,stroke_width=2) for x,t in zip(X,T)]))
        self.remove(density);self.add(dots,residuals)
        self.beat(offset.animate.set_value(0),self.formula(r'-\ln L=',r'\frac\beta2\sum_n\{y(x_n,\mathbf w)-t_n\}^2',r'+\mathrm{const.}',colors={1:YELLOW},size=29))
        self.beat(Circumscribe(residuals),Indicate(self.equation[1],scale_factor=1.05))
        self.beat(self.formula(r'p(\mathbf w|\alpha)\propto\exp(-\tfrac\alpha2\mathbf w^T\mathbf w)',colors={0:PURPLE}),Circumscribe(curve))
        self.slider(logalpha,np.log(ALPHA),np.log(8),[1.7,2.05,0],r'\ln\alpha',PURPLE,3)
        self.beat(logalpha.animate.set_value(np.log(8)),self.formula(r'\frac\beta2\sum_n(y_n-t_n)^2',r'+\frac\alpha2\mathbf w^T\mathbf w',colors={0:YELLOW,1:PURPLE}))
        self.beat(logalpha.animate.set_value(np.log(ALPHA)),self.formula(r'\lambda=\alpha/\beta,\qquad \mathbf w_{\rm MAP}=\arg\max_{\mathbf w}p(\mathbf w|D)',size=29))

    def prediction(self):
        self.legend(('観測',BLUE),('予測の平均',RED),('ノイズ',YELLOW),('係数の不確かさ',PURPLE))
        ax=self.axes([0,1,.25],[-1.4,1.4,.7],height=3.1,ylabel='t')
        xx=np.linspace(0,1,241);m,v=predictive(xx)
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.055,color=BLUE) for x,t in zip(X,T)]);self.add(dots)
        samples=VGroup(*[line_graph(ax,xx,phi(xx)@w,PURPLE,.45,2) for w in SAMPLE_W])
        self.beat(LaggedStart(*[Create(c) for c in samples],lag_ratio=.1))
        cursor=ValueTracker(.5)
        def bells():
            x=cursor.get_value(); means=phi(x)@SAMPLE_W.T
            return VGroup(*[line_graph(ax,x+.03*gaussian(t:=np.linspace(mu-.45,mu+.45,80),mu,.15),t,PURPLE,.45,1.5) for mu in means])
        densities=always_redraw(bells);self.add(densities)
        self.beat(cursor.animate.set_value(.8))
        self.beat(cursor.animate.set_value(.5),self.formula(r'p(t|x,D)=\int p(t|x,\mathbf w)p(\mathbf w|D)\,d\mathbf w',size=30))
        self.remove(densities)
        def band(sd,color,opacity):
            return Polygon(*[ax.c2p(x,y) for x,y in zip(xx,m+sd)],*[ax.c2p(x,y) for x,y in zip(xx[::-1],(m-sd)[::-1])],stroke_width=0,fill_color=color,fill_opacity=opacity)
        totalband=band(np.sqrt(v),PURPLE,.3)
        meanline=line_graph(ax,xx,m,RED)
        self.beat(FadeOut(samples),FadeIn(totalband),FadeIn(meanline),self.formula(r'p(t|x,D)=\mathcal N(t|m(x),s^2(x))'))
        noiseband=band(np.full_like(xx,.15),YELLOW,.35)
        self.beat(FadeIn(noiseband),self.formula(r's^2(x)=',r'\beta^{-1}',r'+\phi(x)^T S\phi(x)',colors={1:YELLOW,2:PURPLE}))
        # A separate stacked bar is in variance units, not standard deviations.
        def uncertainty_bar():
            _,var=predictive(cursor.get_value());noise=1/BETA;extra=float(var-noise)
            return VGroup(rect(1.5,12*noise,[5.55,.4+6*noise,0],YELLOW,.8),rect(1.5,12*extra,[5.55,.4+12*noise+6*extra,0],PURPLE,.8))
        stack=always_redraw(uncertainty_bar)
        guide=always_redraw(lambda:DashedLine(ax.c2p(cursor.get_value(),-1.3),ax.c2p(cursor.get_value(),1.3),color=MUTED))
        self.add(stack,guide,jp('分散の内訳',18).move_to([5.5,1.95,0]))
        self.beat(cursor.animate.set_value(1),self.formula(r'\phi(x)=(1,x,x^2,x^3)^T,\quad S^{-1}=\alpha I+\beta\sum_n\phi(x_n)\phi(x_n)^T',size=25))
        self.beat(cursor.animate.set_value(.1),self.formula(r'm(x)=\beta\phi(x)^TS\sum_n\phi(x_n)t_n,\quad s^2(x)=\beta^{-1}+\phi(x)^TS\phi(x)',size=26))
