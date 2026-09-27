"""PRML 2.2 — one observation, counts, and a moving map of probabilities."""
from pathlib import Path
import json
import numpy as np
from manim import *
from caption_layout import jp, tex
from narrated_scene import NarratedScene
from make_voicevox_narration import MANIFEST
from narration_content import SCENES
from multinomial_model import (SEQUENCE, COUNTS, PERMUTATIONS, coefficient,
                                relative_likelihood, mle_slice, density_raster, predictive)

COLORS=[ManimColor('#FF727C'),ManimColor('#64B5F6'),ManimColor('#FFE079')]
MUTED=ManimColor('#A8B2C5')
GREEN=ManimColor('#77D49A')
PURPLE=ManimColor('#C29AFF')
BG='#10141F'
VERTICES=np.array([[-5.4,-1.35,0],[-.8,-1.35,0],[-3.1,2.15,0]])


def equation(s, pos=(0,-2.35,0), size=33):
    m=MathTex(s,font_size=size,tex_to_color_map={r"\mu_"+str(i+1):c for i,c in enumerate(COLORS)})
    for i,c in enumerate(COLORS,1):
        m.set_color_by_tex(r'\mu_'+str(i),c,substring=False)
    m.move_to(pos)
    if m.width>12.5:
        raise ValueError(f'Equation too wide: {s}')
    return m


def tokens(values, y=1.2, spacing=.7, radius=.20):
    return VGroup(*[Dot([(i-(len(values)-1)/2)*spacing,y,0],radius=radius,color=COLORS[k])
                     for i,k in enumerate(values)])


def readout(label, getter, pos, color=WHITE, places=2):
    prefix=tex(label,27,color)
    number=DecimalNumber(getter(),num_decimal_places=places,font_size=27,color=color)
    group=VGroup(prefix,number).arrange(RIGHT,buff=.12).move_to(pos)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group


def bars(getter, xs=(-3,0,3), bottom=-1.1, height=2.8, width=.8, labels=True):
    result=VGroup()
    for k,x in enumerate(xs):
        rect=Rectangle(width=width,height=.01,fill_color=COLORS[k],fill_opacity=.85,stroke_width=0)
        rect.add_updater(lambda m,k=k,x=x:m.stretch_to_fit_height(max(.008,height*getter()[k])).move_to([x,bottom,0],aligned_edge=DOWN))
        result.add(rect)
        num=DecimalNumber(getter()[k],num_decimal_places=2,font_size=25,color=COLORS[k])
        num.add_updater(lambda m,k=k,x=x:m.set_value(getter()[k]).move_to([x,bottom+height*getter()[k]+.25,0]))
        result.add(num)
        if labels:
            result.add(tex(r'\mu_'+str(k+1),27,COLORS[k]).move_to([x,bottom-.30,0]))
    result.update(0)
    return result


def simplex():
    outline=Polygon(*VERTICES,color=MUTED,stroke_width=2)
    labels=VGroup(*[tex(r'\mu_'+str(k+1)+'=1',23,COLORS[k]).move_to(v+np.array([0,-.35 if k<2 else .3,0])) for k,v in enumerate(VERTICES)])
    return VGroup(outline,labels)


def heatmap(getter):
    mob=ImageMobject(density_raster(getter())).set_resampling_algorithm(RESAMPLING_ALGORITHMS['bilinear'])
    mob.stretch_to_fit_width(4.6).stretch_to_fit_height(3.5).move_to([-3.1,.4,0])
    previous=[None]
    def update(m):
        a=np.asarray(getter())
        if previous[0] is None or not np.allclose(a,previous[0],atol=1e-8):
            m.pixel_array=density_raster(a)
            previous[0]=a.copy()
    mob.add_updater(update)
    return mob


def density_key():
    shades=VGroup(*[Rectangle(width=.17,height=.18,stroke_width=0,fill_opacity=1,
                    fill_color=interpolate_color(ManimColor('#18234A'),ManimColor('#FDE788'),i/23)) for i in range(24)])
    shades.arrange(RIGHT,buff=0).move_to([-3.1,-2.12,0])
    labels=VGroup(*[jp(label,16,MUTED).move_to([-5.14+4.08*np.log1p(v)/np.log(21),-2.40,0])
                     for v,label in [(0,'0'),(2,'2'),(6,'6'),(20,'20以上')]])
    return VGroup(shades,labels,jp('密度',16,MUTED).move_to([-5.8,-2.12,0]))



class PRML22MultinomialVariables(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.observation,self.selector,self.counts,self.mle,
                                   self.arrangements,self.probability_map,self.prior,self.update,self.predict]):
            self.begin(i)
            method()
            assert self.beat_index==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        p=Path(config.media_dir)/'prml22_timeline.json'
        p.write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def formula_to(self,s,pos=(0,-2.35,0),size=33):
        new=equation(s,pos,size)
        old=self.formula
        self.formula=new
        return FadeIn(new) if old is None else ReplacementTransform(old,new)

    def observation(self):
        seq=tokens(SEQUENCE)
        note=jp('自作の観測例：毎回、玉を戻す',23,MUTED).move_to([0,2.25,0])
        self.add(note)
        self.beat(LaggedStart(*[FadeIn(d,shift=DOWN*.4) for d in seq],lag_ratio=.15))
        targets=[]; used=[0,0,0]
        for k in SEQUENCE:
            targets.append(np.array([(-3+3*k),-.8+used[k]*.45,0]));used[k]+=1
        counts=VGroup(*[tex(str(c),31,COLORS[k]).move_to([-3+3*k,-1.3,0]) for k,c in enumerate(COUNTS)])
        self.beat(*[d.animate.move_to(p) for d,p in zip(seq,targets)],FadeIn(counts))
        cards=VGroup(*[VGroup(RoundedRectangle(width=1.6,height=.8,corner_radius=.12,color=c),jp(name,29,c)).move_to([-3+3*k,.6,0]) for k,(name,c) in enumerate(zip(['赤','青','黄色'],COLORS))])
        self.beat(FadeOut(seq),FadeOut(counts),FadeIn(cards),self.formula_to(r'x_k\in\{0,1\},\quad\sum_{k=1}^{K}x_k=1'))
        highlight=SurroundingRectangle(cards[1],color=COLORS[1],buff=.12)
        vec=equation(r'x=(0,1,0)^{\mathsf T}',(0,-.7,0),40)
        self.beat(Create(highlight),FadeIn(vec))
        self.beat(highlight.animate.become(SurroundingRectangle(cards[0],color=COLORS[0],buff=.12)),Transform(vec,equation(r'x=(1,0,0)^{\mathsf T}',(0,-.7,0),40)))
        self.beat(FadeOut(cards),FadeOut(highlight),Transform(vec,equation(r'x=(0,0,1,0,0,0)^{\mathsf T}',(0,.35,0),43)),self.formula_to(r'K=6\quad\text{(2.25)}'))

    def selector(self):
        red=ValueTracker(.5)
        get=lambda:np.array([red.get_value(),.6*(1-red.get_value()),.4*(1-red.get_value())])
        chart=bars(get,xs=(-4.6,-3.0,-1.4))
        self.add(chart)
        self.beat(self.formula_to(r'\mu_k\ge0,\qquad\sum_{k=1}^{K}\mu_k=1'))
        slider=NumberLine(x_range=[0,1,.5],length=3.5,include_numbers=True,font_size=20).move_to([3,.9,0])
        knob=Dot(color=COLORS[0]).add_updater(lambda m:m.move_to(slider.n2p(red.get_value())))
        self.add(slider,knob,tex(r'\mu_1',28,COLORS[0]).move_to([3,1.6,0]))
        self.beat(red.animate.set_value(.7))
        expr=equation(r'{{\mu_1^0}}\,{{\mu_2^1}}\,{{\mu_3^0}}={{\mu_2}}',(3,-.6,0),35)
        for term,c in zip(expr[:3],COLORS): term.set_color(c)
        self.beat(FadeIn(expr),Indicate(chart[3].copy(),color=COLORS[1]))
        self.beat(self.formula_to(r'p(x\mid\mu)=\prod_{k=1}^{K}\mu_k^{x_k}\quad\text{(2.26)}'))
        self.beat(Transform(expr,equation(r'\mu_1^0\,\mu_2^0\,\mu_3^1=\mu_3',(3,-.6,0),35)),Indicate(chart[6].copy(),color=COLORS[2]))
        self.beat(FadeOut(expr),FadeOut(slider),FadeOut(knob),self.formula_to(r'E[x\mid\mu]=\sum_xp(x\mid\mu)x=\mu\quad\text{(2.28)}'),red.animate.set_value(.5))

    def counts(self):
        seq=tokens(SEQUENCE,y=1.35)
        self.add(seq)
        self.beat(self.formula_to(r'p(D\mid\mu)=\prod_{n=1}^{N}\prod_{k=1}^{K}\mu_k^{x_{nk}}'))
        order=np.argsort(SEQUENCE,kind='stable'); slots=[np.array([(i-4.5)*.7,.45,0]) for i in range(10)]
        self.beat(*[seq[j].animate.move_to(slots[i]) for i,j in enumerate(order)],self.formula_to(r'p(D\mid\mu)={{\mu_1^6}}\,{{\mu_2^3}}\,{{\mu_3}}'))
        count=equation(r'm=(6,3,1),\qquad N=10',(0,1.9,0),34)
        self.beat(FadeIn(count),self.formula_to(r'p(D\mid\mu)=\prod_{k=1}^{K}\mu_k^{m_k}\quad\text{(2.29)}'))
        self.beat(*[d.animate.move_to([-d.get_x(),d.get_y(),0]) for d in seq])
        matrix=equation(r'\begin{pmatrix}1\\0\\0\end{pmatrix}+\begin{pmatrix}0\\1\\0\end{pmatrix}+\cdots=\begin{pmatrix}6\\3\\1\end{pmatrix}',(0,.25,0),37)
        self.beat(FadeOut(seq),FadeIn(matrix),self.formula_to(r'm_k=\sum_{n=1}^{N}x_{nk}\quad\text{(2.30)}'))
        condition=jp('独立な観測 × 共通の確率',32,GREEN).move_to([0,-1.3,0])
        self.beat(FadeIn(condition),Indicate(count))

    def mle(self):
        t=ValueTracker(.2)
        ax=Axes(x_range=[0,1,.2],y_range=[0,1,.5],x_length=8,y_length=2.7,tips=False,
                axis_config={'include_numbers':True,'font_size':20}).move_to([0,.1,0])
        curve=ax.plot(lambda v:float(relative_likelihood(v)),x_range=[.005,.995,.01],color=GREEN)
        point=Dot(color=COLORS[0]).add_updater(lambda m:m.move_to(ax.c2p(t.get_value(),relative_likelihood(t.get_value()))))
        line=always_redraw(lambda:Line(ax.c2p(t.get_value(),0),point.get_center(),color=COLORS[0]))
        label=jp('尤度 / 最大尤度',23,GREEN).move_to([-3.6,2.15,0])
        mu=readout(r'\mu_1=',t.get_value,(3.4,2.15,0),COLORS[0])
        self.add(ax,curve,line,point,label,mu)
        self.beat(self.formula_to(r'\mu=(t,\;3(1-t)/4,\;(1-t)/4)'))
        self.beat(t.animate.set_value(.6))
        self.beat(t.animate.set_value(.85))
        self.beat(t.animate.set_value(.6),self.formula_to(r'\mu_k^{\rm ML}=\frac{m_k}{N},\quad\mu^{\rm ML}=(0.6,0.3,0.1)\quad\text{(2.33)}',size=31))
        group=VGroup(ax,curve,line,point,label,mu)
        group.clear_updaters(recursive=True)
        deriv=equation(r'\sum_km_k\ln\mu_k+\lambda\left(\sum_k\mu_k-1\right)',(0,.8,0),42)
        self.beat(FadeOut(group),FadeIn(deriv),self.formula_to(r'\frac{m_k}{\mu_k}+\lambda=0\quad\Longrightarrow\quad\mu_k=-\frac{m_k}{\lambda}\quad\text{(2.32)}'))
        self.beat(Transform(deriv,equation(r'1=\sum_k\mu_k=-\frac{N}{\lambda}\quad\Longrightarrow\quad\lambda=-N',(0,.8,0),40)),self.formula_to(r'\mu_k^{\rm ML}=m_k/N\qquad(m_k=0\Rightarrow\mu_k^{\rm ML}=0)'))

    def arrangements(self):
        single=tokens([0,0,1,2],y=.6)
        label=equation(r'm=(2,1,1),\qquad N=4',(0,2.1,0),34)
        self.add(label)
        self.beat(FadeIn(single),self.formula_to(r'p(\text{one sequence}\mid\mu)=\mu_1^2\mu_2\mu_3'))
        rows=VGroup(*[tokens(p,y=0,spacing=.35,radius=.12).move_to([-4+4*(i%3),1.25-.62*(i//3),0]) for i,p in enumerate(PERMUTATIONS)])
        self.beat(FadeOut(single),LaggedStart(*[FadeIn(r) for r in rows],lag_ratio=.13))
        self.beat(self.formula_to(r'\frac{4!}{2!\,1!\,1!}=\frac{24}{2}=12'),Indicate(rows[0]),Indicate(rows[1]))
        self.beat(self.formula_to(r'12\times(0.5^2\times0.3\times0.2)=12\times0.015=0.18',size=32),LaggedStart(*[Indicate(r,color=GREEN) for r in rows],lag_ratio=.1))
        self.beat(FadeOut(rows),self.formula_to(r'\mathrm{Mult}(m\mid\mu,N)=\frac{N!}{\prod_km_k!}\prod_k\mu_k^{m_k}\quad\text{(2.34)}',pos=(0,.35,0),size=39),FadeIn(equation(r'm_k\in\{0,1,\ldots\},\quad\sum_km_k=N',(0,-1.5,0),34)))
        brace=jp('並び方の数は、固定した回数だけで決まる',27,GREEN).move_to([0,1.35,0])
        self.beat(FadeIn(brace),Indicate(self.formula))

    def probability_map(self):
        weights=[ValueTracker(1),ValueTracker(0),ValueTracker(0)]
        get=lambda:np.array([t.get_value() for t in weights])
        tri=simplex()
        dot=Dot(color=WHITE,radius=.10).add_updater(lambda m:m.move_to(get()@VERTICES))
        chart=bars(get,xs=(1.8,3.3,4.8),bottom=-.7,height=2.6)
        self.add(tri,dot,chart)
        self.beat(self.formula_to(r'\mu=(1,0,0)'))
        self.beat(weights[0].animate.set_value(.5),weights[1].animate.set_value(.5),self.formula_to(r'\mu=(1/2,1/2,0)'))
        self.beat(*[t.animate.set_value(1/3) for t in weights],self.formula_to(r'P=\mu_1V_1+\mu_2V_2+\mu_3V_3'))
        self.beat(weights[0].animate.set_value(.1),weights[1].animate.set_value(.1),weights[2].animate.set_value(.8))
        self.beat(weights[0].animate.set_value(.5),weights[1].animate.set_value(.3),weights[2].animate.set_value(.2),self.formula_to(r'\mu_3=1-\mu_1-\mu_2\qquad\text{dimension}=K-1'))
        samples=np.random.default_rng(2206).dirichlet([1,1,1],100)@VERTICES
        cloud=VGroup(*[Dot(p,radius=.035,color=PURPLE) for p in samples])
        self.beat(LaggedStart(*[FadeIn(p) for p in cloud],lag_ratio=.015),self.formula_to(r'p(x\mid\mu)\quad\longrightarrow\quad p(\mu)'))

    def prior(self):
        a=[ValueTracker(1) for _ in range(3)]
        get=lambda:np.array([t.get_value() for t in a])
        raster=heatmap(get);tri=simplex()
        self.add(raster,tri,density_key())
        nums=VGroup(*[readout(r'\alpha_'+str(i+1)+'=',t.get_value,(3,1.9-i*.65,0),c,1) for i,(t,c) in enumerate(zip(a,COLORS))])
        self.add(nums)
        mean=Dot(radius=.07,color=WHITE).add_updater(lambda m:m.move_to(get()/get().sum()@VERTICES))
        self.add(mean,jp('白点：平均',19,MUTED).move_to([3,-.4,0]))
        self.beat(FadeIn(jp('ディリクレ分布',30,PURPLE).move_to([3,-1.05,0])))
        self.beat(*[t.animate.set_value(.3) for t in a])
        self.beat(*[t.animate.set_value(1) for t in a])
        self.beat(*[t.animate.set_value(8) for t in a])
        self.beat(self.formula_to(r'\mathrm{Dir}(\mu\mid\alpha)=\frac{\Gamma(\alpha_0)}{\prod_k\Gamma(\alpha_k)}\prod_k\mu_k^{\alpha_k-1}',pos=(0,-2.70,0),size=28),FadeIn(equation(r'\alpha_0=\sum_k\alpha_k,\quad\alpha_k>0',(3,-1.8,0),25)))
        self.beat(a[0].animate.set_value(20))

    def update(self):
        a=[ValueTracker(2) for _ in range(3)]
        get=lambda:np.array([t.get_value() for t in a])
        self.add(heatmap(get),simplex(),density_key())
        nums=VGroup(*[readout(r'\alpha_'+str(i+1)+'+m_'+str(i+1)+'=',t.get_value,(3,1.7-i*.65,0),c,1) for i,(t,c) in enumerate(zip(a,COLORS))])
        self.add(nums)
        chips=tokens(SEQUENCE,spacing=.42,radius=.13).move_to([3,-1.0,0])
        self.beat(FadeIn(chips),FadeIn(jp('事前 → 観測 → 事後',25,PURPLE).move_to([3,2.35,0])))
        self.beat(a[0].animate.set_value(3),Indicate(chips[0]),self.formula_to(r'p(\mu\mid x_1)\propto\mu_1\,p(\mu)',pos=(3,-2.05,0),size=29))
        self.beat(self.formula_to(r'\mu_k^{\alpha_k-1}\mu_k^{m_k}=\mu_k^{(\alpha_k+m_k)-1}',pos=(3,-2.05,0),size=28))
        self.beat(a[0].animate.set_value(8),a[1].animate.set_value(5),a[2].animate.set_value(3),LaggedStart(*[Indicate(c) for c in chips[1:]],lag_ratio=.16))
        self.beat(self.formula_to(r'p(\mu\mid D,\alpha)=\mathrm{Dir}(\mu\mid\alpha+m)',pos=(0,-2.70,0),size=30),FadeIn(jp('共役事前分布',28,GREEN).move_to([3,-1.8,0])))
        blue=Dot([3,-.35,0],radius=.16,color=COLORS[1])
        self.beat(a[1].animate.set_value(6),GrowFromCenter(blue))

    def predict(self):
        amount=ValueTracker(1); mix=ValueTracker(0)
        counts=lambda:np.array([3,1,0])*amount.get_value()
        get=lambda:(1-mix.get_value())*np.array([.75,.25,0])+mix.get_value()*predictive([1,1,1],counts())
        chart=bars(get,xs=(-3,0,3),bottom=-.8,height=2.6)
        counter=readout('N=',lambda:counts().sum(),(4.7,2.2,0),WHITE,0)
        self.add(chart,counter)
        self.beat(self.formula_to(r'm=(3,1,0)\qquad\mu^{\rm ML}=(0.75,0.25,0)'))
        self.beat(mix.animate.set_value(1),self.formula_to(r'\alpha=(1,1,1)\qquad p(x_{\rm next})=(4/7,2/7,1/7)'))
        self.beat(self.formula_to(r'p(x_{{\rm next},k}=1\mid D)=\int\mu_kp(\mu\mid D)\,d\mu=\frac{\alpha_k+m_k}{\alpha_0+N}',size=31))
        self.beat(self.formula_to(r'\frac{\alpha_0}{\alpha_0+N}\frac{\alpha_k}{\alpha_0}+\frac{N}{\alpha_0+N}\frac{m_k}{N}',size=37),Indicate(chart[0].copy()),Indicate(chart[6].copy()))
        self.beat(amount.animate.set_value(10),self.formula_to(r'p(x_{{\rm next},3}=1\mid D)=\frac{1}{N+3}'))
        chart.clear_updaters(recursive=True);counter.clear_updaters(recursive=True)
        chain=VGroup(jp('一回の色',28,COLORS[0]),tex(r'\longrightarrow',36),jp('回数の組',28,COLORS[1]),tex(r'\longrightarrow',36),jp('確率の地図',28,COLORS[2])).arrange(RIGHT,buff=.35).move_to([0,1,0])
        line=Line([-3,-.8,0],[3,-.8,0],color=PURPLE)
        self.beat(FadeOut(chart),FadeOut(counter),FadeIn(chain),Create(line),self.formula_to(r'K=2:\quad\mathrm{Dir}(\mu_1,1-\mu_1\mid\alpha_1,\alpha_2)=\mathrm{Beta}(\mu_1\mid\alpha_1,\alpha_2)',size=29))
