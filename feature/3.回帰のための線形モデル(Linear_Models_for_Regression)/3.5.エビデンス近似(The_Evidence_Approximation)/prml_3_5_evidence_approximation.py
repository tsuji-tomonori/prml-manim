"""PRML 3.5: linked visual experiments in Manim Community Edition."""
from manim import *
import numpy as np
from caption_layout import jp,tex
from scene_support import NarratedScene
from evidence_model import *

DATA=ManimColor('#58B5ED'); MODEL=ManimColor('#FF6B77'); PRIOR=ManimColor('#C29AFF')
EV_COLOR=ManimColor('#FFE079'); POST=ManimColor('#77D49A'); MUTED=ManimColor('#A8B2C5')
COLORS=[DATA,MODEL,PRIOR,POST,ORANGE,TEAL,PINK,YELLOW,BLUE_D,GREEN_D]

def path(ax,x,y,color=MODEL,width=3):
    o=ax.c2p(0,0); ux=ax.c2p(1,0)-o;uy=ax.c2p(0,1)-o
    pts=o+np.asarray(x)[:,None]*ux+np.asarray(y)[:,None]*uy
    return VMobject().set_points_as_corners(pts).set_stroke(color,width)

def number(label,getter,pos,color=WHITE,places=2):
    prefix=tex(label,25,color); num=DecimalNumber(getter(),num_decimal_places=places,font_size=25,color=color)
    group=VGroup(prefix,num).arrange(RIGHT,buff=.12).move_to(pos); anchor=num.get_left().copy()
    num.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group

class PRML35EvidenceApproximation(NarratedScene):
    def scenes(self):
        return [self.question,self.integrate_or_choose,self.area,self.gaussian,self.degree,
                self.directions,self.effective,self.crossing,self.iterate,self.prediction]

    def axes(self,xr,yr,center=(0,.1,0),width=9,height=3.4,xlabel='x',ylabel='',xticks=None,yticks=None):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.4,'include_ticks':False}).move_to(center)
        labels=VGroup()
        for v in (xticks if xticks is not None else [xr[0],xr[1]]):
            labels.add(tex(f'{v:g}',18,MUTED).move_to(ax.c2p(v,yr[0])+DOWN*.22))
        for v in (yticks if yticks is not None else [yr[0],yr[1]]):
            labels.add(tex(f'{v:g}',18,MUTED).next_to(ax.c2p(xr[0],v),LEFT,buff=.12))
        if xlabel:labels.add(tex(xlabel,23).next_to(ax.c2p(xr[1],yr[0]),RIGHT,buff=.2))
        if ylabel:labels.add(tex(ylabel,23).next_to(ax.c2p(xr[0],yr[1]),UP,buff=.16))
        self.add(ax,labels);ax.labels=labels
        return ax

    def regression(self,center=(0,.25,0),width=9,height=3.3,span=1.6):
        ax=self.axes([0,1,.5],[-span,span,1],center,width,height,'x',r't,\ y',xticks=[0,.5,1],yticks=[-1,0,1])
        dots=VGroup(*[Dot(ax.c2p(x,t),color=DATA,radius=.055) for x,t in zip(X,T)])
        self.add(dots);return ax,dots

    def slider(self,tr,lo=-6,hi=6,pos=(0,-2.45,0),width=6,label=r'\ln\alpha',color=PRIOR):
        rail=NumberLine(x_range=[lo,hi,1],length=width,include_ticks=False,color=MUTED).move_to(pos)
        knob=Dot(color=color,radius=.08).add_updater(lambda m:m.move_to(rail.n2p(tr.get_value())))
        labels=VGroup(*[tex(f'{v:g}',18,MUTED).next_to(rail.n2p(v),DOWN,buff=.12) for v in [lo,(lo+hi)/2,hi]])
        label=tex(label,26,color).next_to(rail,LEFT,buff=.3)
        group=VGroup(rail,knob,labels,label);self.add(group);return group

    def formula(self,s,pos=(0,2.45,0),size=30,color=WHITE):
        m=tex(s,size,color).move_to(pos)
        if m.width>12.6:m.scale_to_fit_width(12.6)
        self.add(m);return m

    def legend(self,items,y=2.7):
        g=VGroup(*[VGroup(Line(ORIGIN,RIGHT*.32,color=c),jp(t,19,c)).arrange(RIGHT,buff=.1) for t,c in items]).arrange(RIGHT,buff=.5).move_to([0,y,0]);self.add(g);return g

    def emphasis(self,m,color=EV_COLOR):
        # Static overlay avoids suspending live geometry updaters.
        return ShowPassingFlash(m.copy().clear_updaters().set_stroke(color,5),time_width=.6)

    def explain_change(self, old, new):
        duration=self.beat_cues()[-1]['end']
        transition=min(1.5,duration*.22)
        self.beat(phases=[('equation transform',transition,lambda:TransformMatchingTex(old,new)),
                          ('explain complete equation',duration-transition,lambda:Indicate(new,scale_factor=1.015))])

    def review_card(self, label):
        """Replace only the body; narration remains on the sentence PCM clock."""
        saved=[m for m in self.mobjects if m is not self.subtitle]
        header=[m for m in saved if m.get_center()[1]>3.0]
        self.clear()
        label=jp(label,23).move_to([-4.85,2.02,0],aligned_edge=LEFT)
        self.add(*header,label)
        return saved

    def restore_review(self, saved):
        for mob in self.mobjects:
            if mob not in saved:
                mob.clear_updaters(recursive=True)
        self.clear()
        self.add(*saved)

    def gaussian_recap(self):
        saved=self.review_card('復習: 2.3 ガウス分布')
        # 2.3 geometry(): blue points, green rings, green/purple principal axes.
        ax=self.axes([-2.4,2.4,1],[-1.9,1.9,1],center=(-2.6,-.1,0),
                     width=3.7,height=3.7*3.8/4.8,xlabel='',xticks=[],yticks=[])
        s1=ValueTracker(1);s2=ValueTracker(1);angle=ValueTracker(0)
        def mat():
            a=angle.get_value();r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            return r@np.diag([s1.get_value(),s2.get_value()])
        base=np.random.default_rng(2303).normal(size=(100,2))
        base=base[np.linalg.norm(base,axis=1)<1.65]
        points=always_redraw(lambda:VGroup(*[Dot(ax.c2p(*v),radius=.026,color=DATA) for v in base@mat().T]))
        theta=np.linspace(0,TAU,181)
        def rings():
            result=VGroup()
            for radius in [1,1.5]:
                xy=radius*np.c_[np.cos(theta),np.sin(theta)]@mat().T
                result.add(path(ax,xy[:,0],xy[:,1],POST,2))
            return result
        ring=always_redraw(rings)
        axes=always_redraw(lambda:VGroup(*[Arrow(ax.c2p(0,0),ax.c2p(*mat()[:,i]),buff=0,
                          color=c,stroke_width=3) for i,c in enumerate([POST,PRIOR])]))
        covariance=VGroup(jp('共分散：方向ごとの分散',23,POST),
                          tex(r'\Sigma\mathbf u_i=\lambda_i^\Sigma\mathbf u_i',28,POST),
                          tex(r'\sigma_i=\sqrt{\lambda_i^\Sigma}',29,POST)).arrange(DOWN,buff=.18).move_to([2.1,.85,0])
        precision=VGroup(jp('精度：方向ごとの曲率',23,PRIOR),
                         tex(r'A=\Sigma^{-1},\quad a_i=1/\lambda_i^\Sigma',28,PRIOR),
                         tex(r'\sigma_i=1/\sqrt{a_i}',29,EV_COLOR)).arrange(DOWN,buff=.18).move_to([2.1,-.73,0])
        mapping=VGroup(jp('今回の共分散は',19,MUTED),tex(r'S_N',23,MUTED),jp('／ 正の固有値',19,MUTED)).arrange(RIGHT,buff=.1).move_to([0,-1.76,0])
        self.add(points,ring,axes,covariance)
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('R2.3 circle to ellipse',a,lambda:AnimationGroup(s1.animate.set_value(1.6),s2.animate.set_value(.65),angle.animate.set_value(.6))),
            ('R2.3 covariance to precision',b*.55,lambda:AnimationGroup(FadeIn(precision),FadeIn(mapping))),
            ('R2.3 higher curvature narrower widths',b*.45,lambda:AnimationGroup(s1.animate.set_value(.8),s2.animate.set_value(.45),Indicate(precision[-1],scale_factor=1.04))),
        ])
        self.restore_review(saved)

    def square_aid(self):
        saved=self.review_card('補足：平方完成で中心を読む')
        blue='#58C4DD';yellow='#FFFF00';purple='#9A72AC'
        self.add(jp('説明用の一変数の例',19,MUTED).move_to([3.1,2.02,0]))
        ax=self.axes([-1,3,1],[0,12,3],center=(-2.6,-.1,0),width=3.5,height=2.6,
                     xlabel='w',xticks=[0,1,2],yticks=[0,3])
        x=np.linspace(-1,3,201);curve=path(ax,x,2*x*x-4*x+5,blue)
        expanded=tex(r'2w^2-4w+5',34,blue).move_to([2.0,.95,0])
        completed=MathTex(r'2(w-1)^2',r'+3',font_size=34,color=blue).move_to(expanded)
        completed[1].set_color(purple)
        bottom=Dot(ax.c2p(1,3),color=yellow)
        level=Line(ax.c2p(0,0),ax.c2p(0,3),color=purple,stroke_width=5)
        guide=DashedLine(ax.c2p(1,0),ax.c2p(1,3),color=yellow)
        center=tex(r'w=1',32,yellow).move_to([2,-.15,0])
        constant=VGroup(jp('一定の高さ',23,purple),tex('3',30,purple)).arrange(RIGHT,buff=.2).move_to([2,-.95,0])
        self.add(curve,expanded)
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('V13a same quadratic',a*.60,lambda:Create(curve)),
            ('V13a clear expanded form',.4,lambda:FadeOut(expanded)),
            ('V13a show completed square',a*.40-.4,lambda:FadeIn(completed)),
            ('V13a minimum at w=1',b*.65,lambda:AnimationGroup(FadeIn(bottom),Create(guide),FadeIn(center))),
            ('V13a constant height 3',b*.35,lambda:AnimationGroup(Create(level),FadeIn(constant))),
        ])
        self.restore_review(saved)

    def volume_aid(self):
        saved=self.review_card('補足：幅の積と行列式')
        blue='#58C4DD';yellow='#FFFF00';green='#83C167'
        self.add(jp('説明用の例：正定値の精度 A',19,MUTED).move_to([2.9,2.02,0]))
        left=np.array([-3.9,-.65,0]);scale=2.0
        square=Square(side_length=scale,color=blue,fill_color=blue,fill_opacity=.25).move_to(left,aligned_edge=DL)
        rectangle=Rectangle(width=scale/2,height=scale,color=blue,fill_color=blue,fill_opacity=.25).move_to(left,aligned_edge=DL)
        h1=tex('1',28,blue).move_to(left+[1,-.36,0]);half=tex(r'\tfrac12',28,blue).move_to(left+[.5,-.36,0])
        vertical=tex('1',28,blue).move_to(left+[-.28,1,0])
        precision=tex(r'A=\begin{pmatrix}4&0\\0&1\end{pmatrix}',31).move_to([1.9,1.1,0])
        widths=tex(r'\sigma_1=\frac1{\sqrt4}=\frac12,\quad\sigma_2=1',28,blue).move_to([1.9,.1,0])
        product=tex(r'\sigma_1\sigma_2=\frac12=\frac1{\sqrt{\det A}}',31,yellow).move_to([1.9,-.9,0])
        caption=jp('幅の因子',22,yellow).move_to([-3.1,-1.55,0])
        factors=MathTex(r'e^{-E(\mathbf m_N)}',r'(2\pi)^{M/2}',r'|A|^{-1/2}',font_size=27,color=green).move_to([1.8,-1.73,0])
        factors[2].set_color(yellow)
        self.add(square,h1,vertical,precision)
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('V10 clear unit width label',.4,lambda:FadeOut(h1)),
            ('V10 curvature to standard deviations',a-.9,lambda:AnimationGroup(Transform(square,rectangle),FadeIn(widths))),
            ('V10 show half width',.5,lambda:FadeIn(half)),
            ('V10 product of widths',b,lambda:AnimationGroup(FadeIn(product),FadeIn(caption),Indicate(square,color=yellow,scale_factor=1.02))),
            ('V10 common Gaussian factor and peak',c,lambda:AnimationGroup(FadeIn(factors),Indicate(product,scale_factor=1.02))),
        ])
        self.restore_review(saved)

    def question(self):
        ax,dots=self.regression();self.remove(dots)
        tr=ValueTracker(1.2)
        curve=always_redraw(lambda:path(ax,U,PU@at(tr.get_value())['mean']))
        guide=DashedLine(ax.c2p(HIDDEN_X,-1.5),ax.c2p(HIDDEN_X,1.45),color=EV_COLOR,stroke_opacity=.55)
        unknown=tex('?',32,EV_COLOR).move_to(ax.c2p(HIDDEN_X,1.5)+UP*.2)
        mark=always_redraw(lambda:Dot(ax.c2p(HIDDEN_X,hidden_prediction(tr.get_value())),color=EV_COLOR,radius=.09))
        sl=self.slider(tr)
        legend=self.legend([('観測',DATA),('予測',MODEL)])
        self.beat(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=.12),Create(guide),FadeIn(unknown))
        prediction=number(r'\hat y(0.75)=',lambda:hidden_prediction(tr.get_value()),[-3.5,2.05,0],EV_COLOR,2)
        self.add(curve,mark,prediction)
        first=self.sentence_duration(0);second=self.sentence_duration(1)
        self.beat(phases=[('predict before moving',.85,lambda:Wait()),
                          ('weaken prior',first-.85,lambda:tr.animate.set_value(-6)),
                          ('show first prediction',second,lambda:Indicate(mark,scale_factor=1.15))])
        first=self.sentence_duration(0);second=self.sentence_duration(1)
        self.beat(phases=[('strengthen prior',first,lambda:tr.animate.set_value(6)),
                          ('show changed prediction',second,lambda:Indicate(mark,scale_factor=1.15))])
        first=self.sentence_duration(0);second=self.sentence_duration(1)
        self.beat(phases=[('return to middle',first,lambda:tr.animate.set_value(1.2)),
                          ('ask how to choose',second,lambda:Indicate(prediction,scale_factor=1.04))])
        self.remove(mark,prediction,unknown,guide)
        basis=VGroup(*[path(ax,U,PU[:,j],COLORS[j],1.4).set_stroke(opacity=.55) for j in range(10)])
        self.beat(LaggedStart(*[Create(m) for m in basis],lag_ratio=.15))
        self.remove(*basis.get_family())
        f=self.formula(r'p(\mathbf w|\alpha)=\mathcal N(\mathbf0,\alpha^{-1}I),\quad \sigma^2=\beta^{-1},\quad \lambda_{\rm reg}=\alpha/\beta',pos=(0,2.1,0),size=26)
        self.beat(Indicate(f,scale_factor=1.03),tr.animate.set_value(1.2))

    def integrate_or_choose(self):
        ax,dots=self.regression(height=2.8,center=(0,.55,0));s=at(OPT)
        samples=np.random.default_rng(7).multivariate_normal(s['mean'],s['cov'],8)
        curves=VGroup(*[path(ax,U,PU@w,POST,1.5).set_stroke(opacity=.45) for w in samples])
        self.beat(LaggedStart(*[Create(c) for c in curves],lag_ratio=.15))
        mean=path(ax,U,PU@s['mean'],MODEL,4);self.beat(Create(mean),self.emphasis(curves))
        full=self.formula(r'p(t_*|\mathbf t)=\iiint p(t_*|\mathbf w,\beta)p(\mathbf w|\mathbf t,\alpha,\beta)p(\alpha,\beta|\mathbf t)\,d\mathbf w\,d\alpha\,d\beta',pos=(0,-1.9,0),size=27)
        self.beat(Indicate(full,scale_factor=1.015))
        self.remove(ax,ax.labels,mean,*dots.get_family(),*curves.get_family())
        hax=self.axes([-3,3,1],[0,2.2,1],center=(0,.6,0),width=7,height=2.6,xlabel=r'\ln\alpha\quad(\beta\ {\rm fixed})',ylabel='',xticks=[],yticks=[])
        note=jp('精度の事後分布：集中の模式図',22,MUTED).move_to([0,2.45,0]);self.add(note)
        width=ValueTracker(.9)
        bell=always_redraw(lambda:path(hax,np.linspace(-3,3,241),normal(np.linspace(-3,3,241),0,width.get_value()**2),PRIOR))
        self.add(bell);self.beat(width.animate.set_value(.2))
        approx=tex(r'p(t_*|\mathbf t)\approx\int p(t_*|\mathbf w,\hat\beta)p(\mathbf w|\mathbf t,\hat\alpha,\hat\beta)\,d\mathbf w',28).move_to(full)
        self.explain_change(full,approx)
        bayes=self.formula(r'p(\alpha,\beta|\mathbf t)\propto p(\mathbf t|\alpha,\beta)\,p(\alpha,\beta)',pos=(0,-2.65,0),size=29,color=EV_COLOR)
        self.beat(Indicate(bayes,scale_factor=1.015))

    def area(self):
        ax=self.axes([-5,5,1],[0,1.5,.5],center=(0,.1,0),width=9,height=3.3,xlabel='w',ylabel='',xticks=[-5,0,5],yticks=[0,.5,1,1.5])
        self.legend([('尤度',MODEL),('事前',PRIOR),('積の面積',EV_COLOR)])
        xs=np.linspace(-5,5,401);tr=ValueTracker(0);sweep=ValueTracker(-5)
        like=normal(1.2,xs,.35**2)
        likelihood=path(ax,xs,like,MODEL)
        prior=always_redraw(lambda:path(ax,xs,normal(xs,0,np.exp(-tr.get_value())),PRIOR))
        def area():
            z=np.linspace(-5,sweep.get_value()+1e-6,241)
            y=normal(1.2,z,.35**2)*normal(z,0,np.exp(-tr.get_value()))
            return Polygon(ax.c2p(z[0],0),*[ax.c2p(x,v) for x,v in zip(z,y)],ax.c2p(z[-1],0),stroke_width=0,fill_color=EV_COLOR,fill_opacity=.45)
        product=always_redraw(lambda:path(ax,xs,like*normal(xs,0,np.exp(-tr.get_value())),EV_COLOR))
        recap=jp('復習: 3.4 証拠の積分',22).move_to([-3.7,2.24,0])
        self.add(recap)
        self.beat(Create(likelihood))
        self.remove(recap)
        self.add(prior);self.beat(Create(prior))
        self.add(product,always_redraw(area))
        f=self.formula(r'p(t|\alpha,\beta)=\int p(t|w,\beta)\,p(w|\alpha)\,dw',pos=(0,-2.2,0),size=30,color=EV_COLOR)
        self.add(number(r'p(t|\alpha,\beta)=',lambda:float(normal(1.2,0,np.exp(-tr.get_value())+.35**2)),[0,-2.8,0],EV_COLOR,3))
        self.beat(sweep.animate.set_value(5))
        self.beat(tr.animate.set_value(2.5))
        self.beat(tr.animate.set_value(-2))
        self.beat(tr.animate.set_value(-np.log(1.2**2-.35**2)))

    def gaussian(self):
        self.gaussian_recap()
        f=MathTex(r'E(\mathbf w)=',r'\frac\beta2\|\mathbf t-\Phi\mathbf w\|^2','+',r'\frac\alpha2\mathbf w^T\mathbf w',font_size=34).move_to([0,2.25,0]);f[1].set_color(EV_COLOR);f[3].set_color(PRIOR)
        phi_label=jp('Φ：材料を各観測点で計算した表',22,MUTED).move_to([0,1.55,0]);self.add(f,phi_label)
        ax=self.axes([-3,3,1],[0,1.1,.5],center=(0,-.15,0),width=7,height=2.3,xlabel=r'w-m',ylabel='',xticks=[-3,0,3],yticks=[0,1])
        tr=ValueTracker(1);xs=np.linspace(-3,3,241)
        bell=always_redraw(lambda:path(ax,xs,np.exp(-.5*tr.get_value()*xs**2),POST))
        self.add(bell);self.beat(Indicate(f[1]),Indicate(f[3]))
        sq=tex(r'E(\mathbf w)=E(\mathbf m_N)+\tfrac12(\mathbf w-\mathbf m_N)^TA(\mathbf w-\mathbf m_N)',30).move_to(f)
        self.explain_change(f,sq)
        self.square_aid()
        af=self.formula(r'A=\alpha I+\beta\Phi^T\Phi=S_N^{-1},\qquad \mathbf m_N=\beta A^{-1}\Phi^T\mathbf t',pos=(0,-2.2,0),size=28,color=PRIOR)
        self.beat(tr.animate.set_value(5),Indicate(af,scale_factor=1.015))
        integral=self.formula(r'\int e^{-E(\mathbf w)}d\mathbf w=e^{-E(\mathbf m_N)}(2\pi)^{M/2}|A|^{-1/2}',pos=(0,-2.8,0),size=29,color=POST)
        self.beat(tr.animate.set_value(.45),Indicate(integral,scale_factor=1.015))
        self.volume_aid()
        self.remove(ax,ax.labels,bell,af,integral,sq,phi_label)
        # Full expression, split across two readable rows.
        log1=MathTex(r'\ln p(\mathbf t|\alpha,\beta)=',r'\frac M2\ln\alpha',r'+\frac N2\ln\beta',font_size=38).move_to([0,.65,0]);log1[1].set_color(PRIOR);log1[2].set_color(DATA)
        log2=MathTex(r'-E(\mathbf m_N)',r'-\frac12\ln|A|',r'-\frac N2\ln(2\pi)',font_size=38).move_to([.5,-.35,0]);log2[0].set_color(EV_COLOR);log2[1].set_color(POST)
        self.add(jp('山の高さ × 幅 → 対数で足し算',27).move_to([0,2.2,0]))
        self.add(log1,log2)
        self.beat(Indicate(log1,scale_factor=1.015),Indicate(log2,scale_factor=1.015))
        self.beat(Indicate(log1[1],color=EV_COLOR),Indicate(log2[1],color=EV_COLOR))

    def degree(self):
        ax,dots=self.regression(center=(0,.9,0),width=8,height=2.5)
        evs=np.array([s['logev'] for s in POLY]); e=self.axes([0,9,1],[-90,-10,20],center=(0,-1.65,0),width=8,height=1.15,xlabel='d',ylabel='',xticks=[0,3,6,9],yticks=[-80,-20])
        self.add(jp('対数\nエビデンス',18,EV_COLOR).move_to([-5.5,-1.6,0]))
        tr=ValueTracker(0)
        def current():
            v=tr.get_value();lo=int(np.floor(v));hi=min(lo+1,9);return (1-v+lo)*POLY_CURVES[lo]+(v-lo)*POLY_CURVES[hi]
        curve=always_redraw(lambda:path(ax,U,current()))
        pts=[Dot(e.c2p(i,v),color=EV_COLOR,radius=.07) for i,v in enumerate(evs)]
        labels=VGroup(*[tex(f'd={i}',26,MODEL).move_to([3.8,2.6,0]) for i in range(10)],jp('形の遷移',22,MUTED).move_to([3.8,2.6,0]))
        def update_label(m):
            v=tr.get_value();key=round(v) if abs(v-round(v))<1e-5 else 10
            for i,c in enumerate(m):c.set_opacity(int(i==key))
        labels.add_updater(update_label);self.add(curve,labels)
        self.formula(r'\alpha=0.005,\quad\beta=16,\quad M=d+1',pos=(-1,2.6,0),size=26)
        self.beat(FadeIn(pts[0]),Indicate(curve))
        for i in [1,2,3]:
            d=self.beat_cues()[-1]['end']
            self.beat(phases=[('change degree',d*.75,lambda i=i:tr.animate.set_value(i)),('compute evidence',d*.25,lambda i=i:FadeIn(pts[i]))])
        d=self.beat_cues()[-1]['end']/6
        phases=[]
        for i in range(4,10):
            phases.extend([('change degree',d*.75,lambda i=i:tr.animate.set_value(i)),('compute evidence',d*.25,lambda i=i:FadeIn(pts[i]))])
        self.beat(phases=phases)
        best=int(np.argmax(evs));duration=self.beat_cues()[-1]['end'];arrival=self.sentence_duration(0)
        self.beat(phases=[('select optimum',arrival,lambda:tr.animate.set_value(best)),('explain selected model',duration-arrival,lambda:Circumscribe(pts[best],color=POST))])

    def directions(self):
        ax=self.axes([-1,4,1],[-1,3,1],center=(-1.1,.15,0),width=7,height=3.6,xlabel=r'\widetilde w_1',ylabel=r'\widetilde w_2',xticks=[0,2,4],yticks=[0,1,2,3])
        lam=np.array([.7,12.]);ml=np.array([2.,1.5]);theta=np.linspace(0,2*np.pi,181)
        contours=VGroup(*[path(ax,ml[0]+np.sqrt(2*v/lam[0])*np.cos(theta),ml[1]+np.sqrt(2*v/lam[1])*np.sin(theta),MODEL,1.8) for v in [.15,.5,1.]])
        self.legend([('尤度の等高線',MODEL),('事前の等高線',PRIOR),('事後平均',POST)])
        probe=Dot(ax.c2p(*ml),color=EV_COLOR)
        self.beat(Create(contours),FadeIn(probe))
        self.beat(probe.animate.move_to(ax.c2p(3.3,1.5)))
        probe.move_to(ax.c2p(*ml));self.beat(probe.animate.move_to(ax.c2p(2,2.4)))
        self.remove(probe);tr=ValueTracker(.01)
        prior=VGroup(*[path(ax,r*np.cos(theta),r*np.sin(theta),PRIOR,1.5) for r in [.5,1.]])
        post=always_redraw(lambda:Dot(ax.c2p(*(lam/(lam+tr.get_value())*ml)),color=POST,radius=.09))
        self.add(post,Dot(ax.c2p(*ml),color=MODEL,radius=.065),tex(r'\mathbf w_{ML}',24,MODEL).move_to(ax.c2p(2.1,2.2)))
        self.beat(FadeIn(prior),tr.animate.set_value(3))
        self.formula(r'\beta\Phi^T\Phi\mathbf u_i=\lambda_i\mathbf u_i',pos=(0,-2.2,0),size=29)
        f=self.formula(r'\widetilde m_i=\frac{\lambda_i}{\alpha+\lambda_i}\widetilde w_{ML,i}',pos=(0,-2.8,0),size=29,color=POST)
        self.add(number(r'\alpha=',tr.get_value,[4.4,1,0],PRIOR),number('q_1=',lambda:lam[0]/(lam[0]+tr.get_value()),[4.4,.2,0],EV_COLOR),number('q_2=',lambda:lam[1]/(lam[1]+tr.get_value()),[4.4,-.6,0],EV_COLOR))
        self.beat(Indicate(f),self.emphasis(contours))
        self.beat(tr.animate.set_value(.01),FadeOut(prior))

    def effective(self):
        tr=ValueTracker(-6)
        ax=self.axes([.5,10.5,1],[0,1,1],center=(0,.65,0),width=9,height=2.4,xlabel='i',ylabel='q_i',xticks=list(range(1,11)),yticks=[0,.5,1])
        def bars():
            return VGroup(*[Rectangle(width=.48,height=max(.006,q*2.4),stroke_width=0,fill_color=COLORS[i],fill_opacity=.85).move_to(ax.c2p(i+1,q/2)) for i,q in enumerate(at(tr.get_value())['fractions'])])
        bs=always_redraw(bars);sl=self.slider(tr,pos=(0,-2.6,0))
        self.formula(r'q_i=\frac{\lambda_i}{\alpha+\lambda_i},\quad\gamma=\sum_iq_i,\quad0\leq\gamma\leq M',pos=(0,2.45,0),size=30,color=EV_COLOR)
        count=number(r'\gamma=',lambda:at(tr.get_value())['gamma'],[0,-1.55,0],EV_COLOR);self.add(bs,count)
        self.beat(Create(bs));self.beat(tr.animate.set_value(6))
        self.beat(tr.animate.set_value(OPT),self.emphasis(bs))
        self.remove(ax,ax.labels,bs,count,sl)
        weights=np.array([at(float(z))['mean'] for z in GRID]);gammas=np.array([at(float(z))['gamma'] for z in GRID])
        span=float(np.ceil(np.max(abs(weights))))
        wax=self.axes([0,10,2],[-span,span,1],center=(0,.15,0),width=8.6,height=3.0,xlabel=r'\gamma',ylabel='w_j',xticks=[0,5,10],yticks=[-span,0,span])
        trails=VGroup(*[path(wax,gammas,weights[:,i],COLORS[i],1.5) for i in range(10)])
        markers=always_redraw(lambda:VGroup(*[Dot(wax.c2p(at(tr.get_value())['gamma'],v),color=COLORS[i],radius=.06) for i,v in enumerate(at(tr.get_value())['mean'])]))
        tr.set_value(6);self.add(trails,markers)
        self.beat(tr.animate.set_value(-6));self.beat(tr.animate.set_value(6))
        f=self.formula(r'\alpha\mathbf m_N^T\mathbf m_N=\gamma\quad\Longrightarrow\quad\alpha_{\rm new}=\frac\gamma{\mathbf m_N^T\mathbf m_N}',pos=(0,-2.6,0),size=30,color=EV_COLOR)
        self.beat(tr.animate.set_value(OPT),Indicate(f,scale_factor=1.015))

    def crossing(self):
        tr=ValueTracker(-6)
        g=np.array([at(float(z))['gamma'] for z in GRID]);r=np.array([at(float(z))['alpha']*at(float(z))['norm'] for z in GRID]);top=float(np.ceil(max(g.max(),r.max())))
        ax=self.axes([-6,6,3],[0,top,5],center=(0,1.05,0),width=8,height=1.65,xlabel=r'\ln\alpha',xticks=[-6,0,6],yticks=[0,top])
        legend=self.legend([('有効な個数',EV_COLOR),('精度 × 重みの二乗',PRIOR)],y=2.7)
        curves=VGroup(path(ax,GRID,g,EV_COLOR),path(ax,GRID,r,PRIOR));self.add(curves)
        lo=float(np.floor(EV.min()/10)*10);hi=float(np.ceil(EV.max()/10)*10)
        ex=self.axes([-6,6,3],[lo,hi,10],center=(0,-1.2,0),width=8,height=1.6,xlabel=r'\ln\alpha',xticks=[-6,0,6],yticks=[lo,hi])
        self.add(jp('対数\nエビデンス',18,POST).move_to([-5.5,-1.2,0]),path(ex,GRID,EV,POST))
        dots=always_redraw(lambda:VGroup(Dot(ax.c2p(tr.get_value(),at(tr.get_value())['gamma']),color=EV_COLOR,radius=.065),Dot(ax.c2p(tr.get_value(),at(tr.get_value())['alpha']*at(tr.get_value())['norm']),color=PRIOR,radius=.065),Dot(ex.c2p(tr.get_value(),at(tr.get_value())['logev']),color=POST,radius=.075)))
        guide=always_redraw(lambda:DashedLine(ax.c2p(tr.get_value(),top),ex.c2p(tr.get_value(),lo),color=MUTED,stroke_opacity=.4))
        self.add(dots,guide)
        self.formula(r'\frac{\partial\ln p}{\partial\ln\alpha}=\tfrac12(\gamma-\alpha\mathbf m_N^T\mathbf m_N)',pos=(0,-2.65,0),size=29)
        self.beat(Create(curves));self.beat(tr.animate.set_value(-1));self.beat(tr.animate.set_value(OPT));self.beat(tr.animate.set_value(5))
        self.remove(ax,ax.labels,curves,dots,guide,legend)
        testax=self.axes([-6,6,3],[.2,.8,.2],center=(0,1.05,0),width=8,height=1.65,xlabel=r'\ln\alpha',xticks=[-6,0,6],yticks=[.2,.5,.8])
        self.add(jp('テスト RMS',19,DATA).move_to([-5.4,1.1,0]),jp('上：答え合わせ　／　下：選択に使う量',23,MUTED).move_to([0,2.65,0]))
        testcurve=path(testax,GRID,TEST,DATA)
        marks=always_redraw(lambda:VGroup(Dot(ex.c2p(tr.get_value(),at(tr.get_value())['logev']),color=POST),Dot(testax.c2p(tr.get_value(),float(np.interp(tr.get_value(),GRID,TEST))),color=DATA)))
        self.add(marks);self.beat(Create(testcurve),tr.animate.set_value(-3));self.beat(tr.animate.set_value(OPT))

    def iterate(self):
        ax,dots=self.regression(center=(-2.5,.4,0),width=6,height=3.1)
        tr=ValueTracker(0)
        def state():
            # Intermediate log-precision interpolation is visual only; integer steps are exact updates.
            v=tr.get_value();i=min(int(v),len(ITER)-2);f=v-i
            return stats(np.exp((1-f)*np.log(ITER[i]['alpha'])+f*np.log(ITER[i+1]['alpha'])),np.exp((1-f)*np.log(ITER[i]['beta'])+f*np.log(ITER[i+1]['beta'])))
        curve=always_redraw(lambda:path(ax,U,PU@state()['mean']))
        residual=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,y),color=EV_COLOR,stroke_width=2) for x,t,y in zip(X,T,PHI@state()['mean'])]))
        self.add(curve,residual)
        f=self.formula(r'\sigma_{ML}^2=\frac1N\sum_n(t_n-y_n)^2',pos=(0,-2.35,0),size=30,color=EV_COLOR)
        self.add(number('RSS=',lambda:state()['rss'],[4,1.6,0],EV_COLOR),number(r'\gamma=',lambda:state()['gamma'],[4,.9,0],PRIOR))
        self.beat(LaggedStart(*[self.emphasis(m) for m in residual],lag_ratio=.12))
        new=tex(r'\beta_{\rm new}^{-1}=\frac{\sum_n(t_n-\mathbf m_N^T\phi(x_n))^2}{N-\gamma}',31,EV_COLOR).move_to(f)
        self.explain_change(f,new)
        free=number(r'N-\gamma=',lambda:len(T)-state()['gamma'],[4,.2,0],EV_COLOR)
        units=VGroup(*[Square(side_length=.17,fill_color=DATA,fill_opacity=1,stroke_width=0) for _ in range(18)]).arrange_in_grid(rows=3,buff=.09).move_to([4,-.8,0]);self.add(free,units)
        self.beat(LaggedStart(*[m.animate.set_opacity(1-.85*np.clip(state()['gamma']-i,0,1)) for i,m in enumerate(units)],lag_ratio=.08))
        self.remove(*units.get_family());self.add(number(r'\alpha=',lambda:state()['alpha'],[4,-.55,0],PRIOR),number(r'\beta=',lambda:state()['beta'],[4,-1.25,0],DATA))
        update=self.formula(r'\alpha_{\rm new}=\gamma/(\mathbf m_N^T\mathbf m_N)',pos=(0,2.6,0),size=29,color=PRIOR)
        self.beat(tr.animate.set_value(1));self.beat(tr.animate.set_value(15))
        self.remove(update,new)
        limit=self.formula(r'\gamma\to M,\quad\alpha\approx\frac{M}{2E_W},\quad\beta\approx\frac{N}{2E_D}\quad(N\gg M)',pos=(0,-2.45,0),size=30,color=POST)
        self.beat(tr.animate.set_value(30),Indicate(limit))

    def prediction(self):
        ax,dots=self.regression(height=3.1,center=(0,.25,0),span=2)
        # Return to the exact fixed-beta experiment used in the opening.
        s=at(OPT);mean=PU@s['mean'];sd=np.sqrt(1/s['beta']+np.einsum('ij,jk,ik->i',PU,s['cov'],PU))
        samples=np.random.default_rng(19).multivariate_normal(s['mean'],s['cov'],8)
        curves=VGroup(*[path(ax,U,PU@w,POST,1.3).set_stroke(opacity=.5) for w in samples]);self.add(path(ax,U,mean,MODEL))
        self.beat(LaggedStart(*[Create(c) for c in curves],lag_ratio=.12))
        band=Polygon(*[ax.c2p(x,y) for x,y in zip(U,mean+2*sd)],*[ax.c2p(x,y) for x,y in zip(U[::-1],(mean-2*sd)[::-1])],stroke_width=0,fill_color=POST,fill_opacity=.18).set_z_index(-1)
        self.formula(r'\sigma_*^2=\hat\beta^{-1}+\phi(x_*)^TS_N\phi(x_*),\qquad \mu_*\pm2\sigma_*',pos=(0,-2.45,0),size=29,color=POST)
        self.beat(FadeIn(band),FadeOut(curves))
        tr=ValueTracker(.1)
        def slice_():
            x=tr.get_value();p=design([x])[0];m=p@s['mean'];v=np.sqrt(1/s['beta']+p@s['cov']@p)
            return VGroup(Line(ax.c2p(x,m-2*v),ax.c2p(x,m+2*v),color=EV_COLOR,stroke_width=4),Dot(ax.c2p(x,m),color=EV_COLOR))
        marker=always_redraw(slice_);self.add(marker);self.beat(tr.animate.set_value(HIDDEN_X))
        truth=Circle(radius=.17,color=ORANGE,stroke_width=3).move_to(ax.c2p(HIDDEN_X,HIDDEN_TRUE))
        label=tex(r'\hat y(0.75)=-0.94,\qquad \sin(2\pi\cdot0.75)=-1.00',30,EV_COLOR).move_to([0,2.52,0])
        fixed=tex(r'\alpha=\hat\alpha,\qquad\beta=16',30,PRIOR).move_to([0,2.52,0])
        first=self.sentence_duration(0);second=self.sentence_duration(1)
        reveal=min(1.1,first*.45);clear=.35;show=.75
        self.beat(phases=[('reveal generated wave',reveal,lambda:AnimationGroup(FadeIn(truth),FadeIn(label))),
                          ('read prediction and truth',first-reveal,lambda:Wait()),
                          ('clear result formula',clear,lambda:FadeOut(label)),
                          ('show fixed precisions',show,lambda:FadeIn(fixed)),
                          ('explain omitted precision uncertainty',second-clear-show,lambda:Wait())])
        self.remove(label,fixed)
        comparison=self.formula(r'-0.71,\ -0.23\quad\longrightarrow\quad -0.94',pos=(0,2.52,0),size=31,color=EV_COLOR)
        self.beat(Indicate(comparison,scale_factor=1.015),self.emphasis(path(ax,U,mean,MODEL)))
        self.beat(Indicate(truth,color=ORANGE),Indicate(marker,scale_factor=1.04))
