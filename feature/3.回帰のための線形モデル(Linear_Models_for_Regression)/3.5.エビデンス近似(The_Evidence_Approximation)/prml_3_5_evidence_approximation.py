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

    def question(self):
        ax,dots=self.regression();self.remove(dots)
        tr=ValueTracker(1.2);curve=always_redraw(lambda:path(ax,U,PU@at(tr.get_value())['mean']))
        sl=self.slider(tr);self.add(curve)
        self.legend([('観測',DATA),('事後平均',MODEL)])
        self.beat(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=.12))
        self.beat(tr.animate.set_value(-6))
        self.beat(tr.animate.set_value(6))
        self.beat(tr.animate.set_value(OPT))
        basis=VGroup(*[path(ax,U,PU[:,j],COLORS[j],1.4).set_stroke(opacity=.55) for j in range(10)])
        self.beat(LaggedStart(*[Create(m) for m in basis],lag_ratio=.15))
        self.remove(*basis.get_family())
        f=self.formula(r'p(\mathbf w|\alpha)=\mathcal N(\mathbf0,\alpha^{-1}I),\quad \sigma^2=\beta^{-1},\quad \lambda_{\rm reg}=\alpha/\beta',pos=(0,2.1,0),size=26)
        self.beat(Indicate(f,scale_factor=1.03),tr.animate.set_value(OPT+.5))

    def integrate_or_choose(self):
        ax,dots=self.regression(height=2.8,center=(0,.55,0));s=at(OPT)
        samples=np.random.default_rng(7).multivariate_normal(s['mean'],s['cov'],8)
        curves=VGroup(*[path(ax,U,PU@w,POST,1.5).set_stroke(opacity=.45) for w in samples])
        self.beat(LaggedStart(*[Create(c) for c in curves],lag_ratio=.15))
        mean=path(ax,U,PU@s['mean'],MODEL,4);self.beat(Create(mean),self.emphasis(curves))
        full=self.formula(r'p(t_*|\mathbf t)=\iiint p(t_*|\mathbf w,\beta)p(\mathbf w|\mathbf t,\alpha,\beta)p(\alpha,\beta|\mathbf t)\,d\mathbf w\,d\alpha\,d\beta',pos=(0,-1.9,0),size=27)
        self.beat(Write(full))
        self.remove(ax,ax.labels,mean,*dots.get_family(),*curves.get_family())
        hax=self.axes([-3,3,1],[0,2.2,1],center=(0,.6,0),width=7,height=2.6,xlabel=r'\ln\alpha\quad(\beta\ {\rm fixed})',ylabel='',xticks=[],yticks=[])
        note=jp('精度の事後分布：集中の模式図',22,MUTED).move_to([0,2.45,0]);self.add(note)
        width=ValueTracker(.9)
        bell=always_redraw(lambda:path(hax,np.linspace(-3,3,241),normal(np.linspace(-3,3,241),0,width.get_value()**2),PRIOR))
        self.add(bell);self.beat(width.animate.set_value(.2))
        approx=tex(r'p(t_*|\mathbf t)\approx\int p(t_*|\mathbf w,\hat\beta)p(\mathbf w|\mathbf t,\hat\alpha,\hat\beta)\,d\mathbf w',28).move_to(full)
        self.beat(TransformMatchingTex(full,approx),Indicate(bell))
        bayes=self.formula(r'p(\alpha,\beta|\mathbf t)\propto p(\mathbf t|\alpha,\beta)\,p(\alpha,\beta)',pos=(0,-2.65,0),size=29,color=EV_COLOR)
        self.beat(Write(bayes))

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
        self.beat(Create(likelihood))
        self.add(prior);self.beat(Create(prior))
        self.add(product,always_redraw(area))
        f=self.formula(r'p(t|\alpha,\beta)=\int p(t|w,\beta)\,p(w|\alpha)\,dw',pos=(0,-2.2,0),size=30,color=EV_COLOR)
        self.add(number(r'p(t|\alpha,\beta)=',lambda:float(normal(1.2,0,np.exp(-tr.get_value())+.35**2)),[0,-2.8,0],EV_COLOR,3))
        self.beat(sweep.animate.set_value(5))
        self.beat(tr.animate.set_value(2.5))
        self.beat(tr.animate.set_value(-2))
        self.beat(tr.animate.set_value(-np.log(1.2**2-.35**2)))

    def gaussian(self):
        f=MathTex(r'E(\mathbf w)=',r'\frac\beta2\|\mathbf t-\Phi\mathbf w\|^2','+',r'\frac\alpha2\mathbf w^T\mathbf w',font_size=34).move_to([0,2.25,0]);f[1].set_color(EV_COLOR);f[3].set_color(PRIOR)
        phi_label=jp('Φ：材料を各観測点で計算した表',22,MUTED).move_to([0,1.55,0]);self.add(f,phi_label)
        ax=self.axes([-3,3,1],[0,1.1,.5],center=(0,-.15,0),width=7,height=2.3,xlabel=r'w-m',ylabel='',xticks=[-3,0,3],yticks=[0,1])
        tr=ValueTracker(1);xs=np.linspace(-3,3,241)
        bell=always_redraw(lambda:path(ax,xs,np.exp(-.5*tr.get_value()*xs**2),POST))
        self.add(bell);self.beat(Indicate(f[1]),Indicate(f[3]))
        sq=tex(r'E(\mathbf w)=E(\mathbf m_N)+\tfrac12(\mathbf w-\mathbf m_N)^TA(\mathbf w-\mathbf m_N)',30).move_to(f)
        self.beat(TransformMatchingTex(f,sq))
        af=self.formula(r'A=\alpha I+\beta\Phi^T\Phi=S_N^{-1},\qquad \mathbf m_N=\beta A^{-1}\Phi^T\mathbf t',pos=(0,-2.2,0),size=28,color=PRIOR)
        self.beat(tr.animate.set_value(5),Indicate(af,scale_factor=1.015))
        integral=self.formula(r'\int e^{-E(\mathbf w)}d\mathbf w=e^{-E(\mathbf m_N)}(2\pi)^{M/2}|A|^{-1/2}',pos=(0,-2.8,0),size=29,color=POST)
        self.beat(tr.animate.set_value(.45),Write(integral))
        self.remove(ax,ax.labels,bell,af,integral,sq,phi_label)
        # Full expression, split across two readable rows.
        log1=MathTex(r'\ln p(\mathbf t|\alpha,\beta)=',r'\frac M2\ln\alpha',r'+\frac N2\ln\beta',font_size=38).move_to([0,.65,0]);log1[1].set_color(PRIOR);log1[2].set_color(DATA)
        log2=MathTex(r'-E(\mathbf m_N)',r'-\frac12\ln|A|',r'-\frac N2\ln(2\pi)',font_size=38).move_to([.5,-.35,0]);log2[0].set_color(EV_COLOR);log2[1].set_color(POST)
        self.add(jp('山の高さ × 幅 → 対数で足し算',27).move_to([0,2.2,0]))
        self.beat(Write(log1),Write(log2))
        brace=SurroundingRectangle(VGroup(log1,log2),color=EV_COLOR,buff=.25)
        self.beat(Create(brace),Indicate(log1[1]),Indicate(log2[1]))

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
        best=int(np.argmax(evs));self.beat(tr.animate.set_value(best),Circumscribe(pts[best],color=POST))

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
        self.beat(tr.animate.set_value(OPT),Write(f))

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
        self.beat(TransformMatchingTex(f,new))
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
        s=ITER[-1];mean=PU@s['mean'];sd=np.sqrt(1/s['beta']+np.einsum('ij,jk,ik->i',PU,s['cov'],PU))
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
        marker=always_redraw(slice_);self.add(marker);self.beat(tr.animate.set_value(.9))
        note=self.formula(r'\alpha=\hat\alpha,\quad\beta=\hat\beta',pos=(0,2.5,0),size=31,color=PRIOR)
        self.beat(tr.animate.set_value(.45),Indicate(note))
        self.remove(note)
        flow=VGroup(jp('重みを積分',25,POST),tex(r'\longrightarrow',28),jp('精度を選ぶ',25,PRIOR),tex(r'\longrightarrow',28),jp('予測を平均',25,EV_COLOR)).arrange(RIGHT,buff=.25).move_to([0,2.55,0])
        self.beat(LaggedStart(*[FadeIn(m) for m in flow],lag_ratio=.15),tr.animate.set_value(.75))
        self.beat(tr.animate.set_value(.25),self.emphasis(path(ax,U,mean,MODEL)))
