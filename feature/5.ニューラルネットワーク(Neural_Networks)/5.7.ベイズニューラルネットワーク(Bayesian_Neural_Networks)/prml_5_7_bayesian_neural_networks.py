"""PRML 5.7 — linked experiments in weights, predictions and probability (CE)."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scipy.special import expit
from contour_lines import contours as contour_segments
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST
from narration_content import SCENES
from bayesian_model import X,T,GRID,ALPHA,BETA,output,jacobian,variance,gaussian,kappa,integrated_sigmoid,evidence_example,objective

DATA=ManimColor('#58B5ED'); POST=ManimColor('#70DBA5'); PRIOR=ManimColor('#C29AFF')
NOISE=ManimColor('#FFB45B'); MAP=ManimColor('#FF6B77'); GOLD=ManimColor('#FFE079'); MUTED=ManimColor('#A8B2C5')
D=np.load(Path(__file__).with_name('model_data.npz'))
W=D['w']; COV=D['cov']; A=D['A']; EIG,U=np.linalg.eigh(A)
MEAN=output(W,GRID); VAR=variance(W,COV,GRID)

def line(ax,x,y,color=POST,width=3,opacity=1):
    origin=ax.c2p(0,0)
    pts=origin+np.asarray(x)[:,None]*(ax.c2p(1,0)-origin)+np.asarray(y)[:,None]*(ax.c2p(0,1)-origin)
    return VMobject().set_points_as_corners(pts).set_stroke(color,width,opacity)

def band(ax,mean,sd,color=POST):
    pts=[ax.c2p(x,y) for x,y in zip(GRID,mean+2*sd)]
    pts += [ax.c2p(x,y) for x,y in zip(GRID[::-1],(mean-2*sd)[::-1])]
    return Polygon(*pts,stroke_width=0,fill_color=color,fill_opacity=.23)

def readout(label,getter,pos,color=WHITE):
    lab=tex(label,25,color);num=DecimalNumber(getter(),num_decimal_places=3,font_size=25,color=color)
    group=VGroup(lab,num).arrange(RIGHT,buff=.12).move_to(pos)
    anchor=num.get_left().copy()
    num.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group

class PRML57BayesianNeuralNetworks(NarratedScene):
    def construct(self):
        self.camera.background_color='#10141F'
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.question,self.update,self.laplace,self.project,self.two_variances,self.evidence,self.hyper,self.classification,self.marginal]):
            self.begin(i);self.formula=None;self.note=None
            method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml57_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def axes(self,x=(-2.6,2.6,1),y=(-2,2,1),center=(0,.25,0),width=9,height=3.4,xlabel='x',ylabel=r't,\ y'):
        ax=Axes(x_range=x,y_range=y,x_length=width,y_length=height,tips=False,
                axis_config=dict(color=MUTED,stroke_width=1.3,include_ticks=False)).move_to(center)
        labs=VGroup()
        for v in np.arange(np.ceil(x[0]/x[2])*x[2],x[1]+.001,x[2]):
            labs.add(tex(f'{v:g}',18,MUTED).move_to(ax.c2p(v,y[0])+DOWN*.22))
        for v in np.arange(np.ceil(y[0]/y[2])*y[2],y[1]+.001,y[2]):
            labs.add(tex(f'{v:g}',18,MUTED).next_to(ax.c2p(x[0],v),LEFT,buff=.12))
        labs.add(tex(xlabel,23).next_to(ax.c2p(x[1],y[0]),RIGHT,buff=.22))
        labs.add(tex(ylabel,23).next_to(ax.c2p(x[0],y[1]),UP,buff=.15))
        self.add(ax,labs);return ax

    def dots(self,ax):return VGroup(*[Dot(ax.c2p(x,t),radius=.05,color=DATA) for x,t in zip(X[:,0],T)])
    def legend(self,items):
        mob=VGroup(*[VGroup(Line(ORIGIN,RIGHT*.3,color=c),jp(s,18,c)).arrange(RIGHT,buff=.1) for s,c in items]).arrange(RIGHT,buff=.45).move_to([0,2.65,0])
        self.add(mob);return mob
    def eq(self,*parts,colors=None):
        m=MathTex(*parts,font_size=29).move_to([0,-2.35,0])
        if m.width>12.4:m.scale_to_fit_width(12.4)
        colors=colors or [WHITE,POST,NOISE,PRIOR,GOLD]
        if len(parts)>1:
            for i,p in enumerate(m):p.set_color(colors[i%len(colors)])
        old=self.formula;self.formula=m
        if old is not None:self.remove(old)
        return FadeIn(m,rate_func=lambda t:smooth(min(1.,6*t)))
    def note_anim(self,text,color=MUTED):
        m=jp(text,21,color).move_to([0,2.12,0]);old=self.note;self.note=m
        if old is not None:self.remove(old)
        return FadeIn(m,rate_func=lambda t:smooth(min(1.,6*t)))
    def slider(self,tracker,lo,hi,label,pos=(0,-1.9,0),width=5,color=GOLD):
        base=NumberLine(x_range=[lo,hi],length=width,include_ticks=False,color=MUTED).move_to(pos)
        dot=Dot(color=color,radius=.075).add_updater(lambda m:m.move_to(base.n2p(tracker.get_value())))
        labelmob=tex(label,24,color).next_to(base,LEFT,buff=.2)
        self.add(base,dot,labelmob);return VGroup(base,dot,labelmob)
    def scan(self,ax,tracker,cov_scale=lambda:1.,noise_scale=lambda:1.):
        def make():
            x=tracker.get_value();mu=output(W,[x])[0];s=np.sqrt(cov_scale()*variance(W,COV,[x])[0]+noise_scale()/BETA)
            return VGroup(DashedLine(ax.c2p(x,ax.y_range[0]),ax.c2p(x,ax.y_range[1]),color=GOLD,dash_length=.08,stroke_width=1),
                          Line(ax.c2p(x,mu-2*s),ax.c2p(x,mu+2*s),color=GOLD,stroke_width=5),Dot(ax.c2p(x,mu),color=MAP,radius=.065))
        return always_redraw(make)
    def network(self,wgetter,pos=(4.6,.3,0)):
        pts=[np.array([-1,0,0]),np.array([0,.6,0]),np.array([0,-.6,0]),np.array([1,0,0])]
        pts=[p+np.array(pos) for p in pts]
        nodes=VGroup(*[Circle(radius=.13,stroke_color=WHITE,fill_color='#10141F',fill_opacity=1).move_to(p) for p in pts])
        def edges():
            w=wgetter();return VGroup(*[Line(nodes[i].get_center(),nodes[j].get_center(),buff=.14,color=POST if v>=0 else MAP,stroke_width=1+2*abs(v)) for i,j,v in [(0,1,w[0]),(0,2,w[1]),(1,3,w[4]),(2,3,w[5])]])
        return VGroup(always_redraw(edges),nodes)

    def question(self):
        ax=self.axes(center=(-1.2,.2,0),width=8,height=3.5)
        dots=self.dots(ax);curve=line(ax,GRID,MEAN,MAP)
        self.legend([('観測',DATA),('代表の予測',MAP),('重みの候補',POST)])
        self.beat(FadeIn(dots),Create(curve))
        x=ValueTracker(0);scan=self.scan(ax,x);self.add(scan)
        self.beat(x.animate.set_value(2.35))
        tweak=ValueTracker(0);getter=lambda:W+np.eye(7)[0]*tweak.get_value()
        curve.add_updater(lambda m:m.become(line(ax,GRID,output(getter(),GRID),MAP)))
        net=self.network(getter);self.add(net);self.slider(tweak,-.14,.14,r'\Delta w_1',pos=(4.2,-.9,0),width=1.8)
        self.beat(tweak.animate.set_value(.14).set_rate_func(there_and_back))
        samples=VGroup(*[line(ax,GRID,output(w,GRID),POST,1.5,.48) for w in D['samples']])
        self.beat(LaggedStart(*[Create(m) for m in samples],lag_ratio=.12))
        self.beat(x.animate.set_value(-2.35))
        self.curves_recap()
        self.beat(x.animate.set_value(.5),self.eq(r'p(t\mid x,D)=',r'\int p(t\mid x,w)',r'p(w\mid D)\,dw'))

    def update(self):
        self.add(jp('復習: 1.2 ベイズ更新',18).move_to([0,2.96,0]))
        ax=self.axes(x=(-3,3,1),y=(0,1.5,.5),height=3.1,center=(0,.35,0),xlabel=r'w_\parallel',ylabel=r'\mathrm{density}')
        self.legend([('事前',PRIOR),('尤度（最大値を1へ換算）',NOISE),('条件付き事後',POST)])
        grid=np.linspace(-3,3,501);direction=U[:,0];coordinate=float(W@direction);mu=0.
        precision=ValueTracker(.8)
        prior=always_redraw(lambda:line(ax,grid,gaussian(grid,mu,1/precision.get_value()),PRIOR))
        self.add(prior)
        self.beat(precision.animate.set_value(3),self.eq(r'p(w\mid\alpha)=',r'\mathcal N(w\mid0,\alpha^{-1}I)',colors=[WHITE,PRIOR]))
        self.beat(precision.animate.set_value(ALPHA),self.eq(r'p(t\mid x,w,\beta)=',r'\mathcal N(t\mid y(x,w),',r'\beta^{-1})'))
        energies=np.array([BETA*np.sum((output(W+(s-coordinate)*direction,X)-T)**2)/2 for s in grid]);like=np.exp(-energies+energies.min())
        lc=line(ax,grid,like,NOISE)
        self.beat(Create(lc),self.eq(r'p(D\mid w,\beta)=\prod_n\mathcal N(t_n\mid y(x_n,w),\beta^{-1})'),self.note_anim('他の座標を固定した、重み空間の一本の軸'))
        power=ValueTracker(0)
        full=np.linspace(-12,12,2401)
        efull=np.array([BETA*np.sum((output(W+(s-coordinate)*direction,X)-T)**2)/2 for s in full])
        def density():
            log=-ALPHA*full**2/2-power.get_value()*efull
            v=np.exp(log-log.max());v/=np.trapezoid(v,full)
            return np.interp(grid,full,v)
        post=always_redraw(lambda:line(ax,grid,density(),POST));self.add(post)
        self.beat(power.animate.set_value(1),self.eq(r'p(w\mid D)\propto',r'p(D\mid w)',r'p(w)',colors=[WHITE,NOISE,PRIOR]))
        self.beat(Indicate(post,color=POST,scale_factor=1),self.eq(r'E(w)=',r'\frac{\beta}{2}\sum_n[y(x_n,w)-t_n]^2',r'+\frac{\alpha}{2}w^Tw',colors=[WHITE,NOISE,PRIOR]))
        mode=Dot(ax.c2p(coordinate,float(np.interp(coordinate,grid,density()))),color=MAP)
        self.beat(FadeIn(mode),Indicate(post,scale_factor=1),self.eq(r'w_{\mathrm{MAP}}=\operatorname*{arg\,min}_w E(w)'))

    def laplace(self):
        ax=self.axes(x=(-.32,.32,.2),y=(0,3.5,1),height=3.2,center=(-2,.3,0),width=6.5,xlabel='s',ylabel=r'E(w)-E(w_{\rm MAP})')
        self.legend([('実際の断面',DATA),('局所二次近似',GOLD)])
        angle=ValueTracker(0);grid=np.linspace(-.32,.32,181)
        def direction():return np.cos(angle.get_value())*U[:,2]+np.sin(angle.get_value())*U[:,4]
        e0=objective(W,X,T,ALPHA,BETA)[0]
        def actual():return np.array([objective(W+s*direction(),X,T,ALPHA,BETA)[0]-e0 for s in grid])
        actual_curve=always_redraw(lambda:line(ax,grid,actual(),DATA))
        self.add(actual_curve)
        self.beat(Create(Dot(ax.c2p(0,0),color=MAP)),self.note_anim('谷底を通る断面。横軸の s は重みの変位'))
        self.laplace_recap()
        quad=always_redraw(lambda:line(ax,grid,.5*(direction()@A@direction())*grid**2,GOLD))
        self.add(quad)
        self.beat(self.eq(r'E(w)\simeq E(w_{\rm MAP})+',r'\frac12\Delta w^TA\Delta w'))
        # Keep the true nonquadratic curve inside the chart while rotating.
        bell_ax=self.axes(x=(-1.2,1.2,1),y=(0,3,1),height=2.8,center=(4.6,.3,0),width=2.5,xlabel='s',ylabel='q(s)')
        sg=np.linspace(-1.2,1.2,161)
        bell=always_redraw(lambda:line(bell_ax,sg,gaussian(sg,0,1/(direction()@A@direction())),POST))
        self.add(bell)
        self.beat(angle.animate.set_value(.55),self.note_anim('曲がり方が大きい方向 → 幅が狭い',GOLD))
        self.beat(angle.animate.set_value(0),self.note_anim('曲がり方が小さい方向 → 幅が広い',POST))
        self.beat(Indicate(quad,scale_factor=1),self.eq(r'A=',r'\alpha I',r'+\beta H',r',\quad H=\nabla\nabla\frac12\sum_n(y_n-t_n)^2'))
        self.beat(self.eq(r'q(w\mid D)=',r'\mathcal N(w\mid w_{\rm MAP},A^{-1})'),self.note_anim('正定値の A が必要。一つの山の近似',PRIOR),angle.animate.set_value(.25))

    def project(self):
        ax=self.axes(y=(-1.4,1.4,1),center=(-1.15,.2,0),width=8,height=3.35)
        self.legend([('元のネットワーク',DATA),('重みについて線形化',POST)])
        self.add(self.dots(ax),line(ax,GRID,MEAN,MAP))
        theta=ValueTracker(0)
        delta=lambda:.16*(np.cos(theta.get_value())*U[:,0]/np.sqrt(EIG[0])+np.sin(theta.get_value())*U[:,1]/np.sqrt(EIG[1]))
        exact=always_redraw(lambda:line(ax,GRID,output(W+delta(),GRID),DATA))
        approx=always_redraw(lambda:line(ax,GRID,MEAN+jacobian(W,GRID)@delta(),POST,2))
        circle=Ellipse(width=1.6,height=1.6*np.sqrt(EIG[0]/EIG[1]),color=PRIOR).move_to([4.75,.4,0]);dot=Dot(color=GOLD).add_updater(lambda m:m.move_to(circle.point_at_angle(theta.get_value())))
        lab=jp('重みの小さな変化',19,PRIOR).next_to(circle,DOWN,buff=.3)
        self.add(circle,dot,lab,exact)
        self.beat(theta.animate.set_value(PI),self.eq(r'p(t\mid x,D)\simeq\int p(t\mid x,w)q(w\mid D)\,dw'))
        self.add(approx)
        self.beat(theta.animate.set_value(2*PI),self.eq(r'y(x,w)\simeq',r'y(x,w_{\rm MAP})',r'+g^T(w-w_{\rm MAP})'))
        self.beat(theta.animate.set_value(3*PI),self.eq(r'g=\left.\nabla_w y(x,w)\right|_{w_{\rm MAP}}'))
        x=ValueTracker(-2.3);scan=self.scan(ax,x,noise_scale=lambda:0);self.add(scan)
        self.beat(x.animate.set_value(2.3))
        self.beat(x.animate.set_value(0),self.eq(r'\operatorname{Var}_q[y_{\rm lin}]=',r'g^TA^{-1}g'))
        self.projection_aid()
        self.beat(theta.animate.set_value(4*PI),self.note_anim('① 重みの分布をガウス化　② 出力を重みについて線形化'))

    def two_variances(self):
        self.add(jp('復習: 3.3 予測分散',18).move_to([0,2.96,0]))
        ax=self.axes(y=(-1.4,1.4,1),center=(-1.3,.3,0),width=7.6,height=3.25)
        self.legend([('重み ±2標準偏差',POST),('観測ノイズ',NOISE),('予測全体',DATA)])
        scale=ValueTracker(1);noise=ValueTracker(0)
        b=always_redraw(lambda:band(ax,MEAN,np.sqrt(scale.get_value()*VAR+noise.get_value()/BETA),DATA if noise.get_value()>.1 else POST))
        self.add(b,self.dots(ax),line(ax,GRID,MEAN,MAP))
        self.beat(self.eq(r'y(x,w_{\rm MAP})\ \pm\ 2\sqrt{',r'g^TA^{-1}g',r'}'))
        x=ValueTracker(0)
        self.add(self.scan(ax,x,cov_scale=scale.get_value,noise_scale=noise.get_value))
        def bars():
            v=float(variance(W,COV,[x.get_value()])[0])*scale.get_value();a=noise.get_value()/BETA
            start=np.array([3.5,-.6,0]);unit=18
            first=Rectangle(width=.65,height=max(.001,a*unit),fill_color=NOISE,fill_opacity=.85,stroke_width=0).move_to(start+UP*a*unit/2)
            second=Rectangle(width=.65,height=max(.001,v*unit),fill_color=POST,fill_opacity=.85,stroke_width=0).move_to(start+UP*(a+v/2)*unit)
            return VGroup(first,second)
        bar=always_redraw(bars);self.add(bar,jp('分散の和',21).move_to([3.5,1.5,0]),readout(r'\sigma^2=',lambda:noise.get_value()/BETA+scale.get_value()*variance(W,COV,[x.get_value()])[0],(4.35,-1.25,0)))
        self.beat(noise.animate.set_value(1),self.eq(r'p(t\mid x,D)\simeq\mathcal N(t\mid y(x,w_{\rm MAP}),\sigma^2(x))'))
        self.beat(self.eq(r'\sigma^2(x)=',r'g^TA^{-1}g',r'+\beta^{-1}'),Indicate(bar,scale_factor=1))
        self.slider(x,-2.4,2.4,'x',pos=(-1.3,-1.85,0),width=4.5)
        self.beat(x.animate.set_value(2.4))
        self.beat(scale.animate.set_value(.03),self.note_anim('思考実験：重みの共分散だけを縮める'))
        self.beat(x.animate.set_value(-2.4),self.note_anim('±2標準偏差は、近似ガウス予測の約95%の範囲'))

    def evidence(self):
        self.add(jp('復習: 3.5 エビデンス',18).move_to([0,2.96,0]))
        ax=self.axes(x=(-3,4,1),y=(0,1.7,.5),height=3.1,center=(0,.3,0),xlabel='w',ylabel=r'\mathrm{density}')
        self.legend([('事前',PRIOR),('尤度',NOISE),('積と面積',POST)])
        sd=ValueTracker(1.1);end=ValueTracker(-3);grid=np.linspace(-3,4,401)
        like=gaussian(1.2,grid,.35**2)
        prior=always_redraw(lambda:line(ax,grid,gaussian(grid,0,sd.get_value()**2),PRIOR))
        lc=line(ax,grid,like,NOISE);self.add(prior,lc)
        self.beat(self.eq(r'p(D\mid\alpha,\beta)=',r'\int p(D\mid w,\beta)p(w\mid\alpha)\,dw'),self.note_anim('積分を見るための一変数の説明例'))
        prod=lambda:like*gaussian(grid,0,sd.get_value()**2)
        product=always_redraw(lambda:line(ax,grid,prod(),POST))
        def area():
            xx=grid[grid<=end.get_value()];yy=prod()[:len(xx)]
            if len(xx)<2:return VMobject()
            return Polygon(ax.c2p(xx[0],0),*[ax.c2p(x,y) for x,y in zip(xx,yy)],ax.c2p(xx[-1],0),stroke_width=0,fill_color=POST,fill_opacity=.4)
        ar=always_redraw(area);self.add(product,ar)
        value=readout(r'p(D)=',lambda:evidence_example(sd.get_value()),(4.35,1.7,0),POST);self.add(value)
        self.beat(end.animate.set_value(4))
        self.beat(sd.animate.set_value(2.8))
        self.beat(phases=[('狭すぎる事前',self.sentence_duration(0),lambda:sd.animate.set_value(.25)),
                          ('中間の幅へ',self.sentence_duration(1),lambda:sd.animate.set_value(1.1))])
        self.remove(prior,lc,product,ar,value,ax)
        # Preserve graph labels only until this transition; full clearing avoids stray ticks.
        for m in list(self.mobjects):
            if m not in [self.subtitle,self.formula]: self.remove(m)
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]))
        formula=MathTex(r'\log p(D\mid\alpha,\beta)\simeq',r'-E(w_{\rm MAP})',r'-\frac12\log|A|',r'+\frac W2\log\alpha+\frac N2\log\beta-\frac N2\log(2\pi)',font_size=29).arrange(DOWN,buff=.3).move_to([0,.6,0])
        formula[1].set_color(MAP);formula[2].set_color(POST);formula[3].set_color(PRIOR)
        self.beat(FadeIn(formula,rate_func=lambda t:smooth(min(1.,6*t))),self.eq(r'W:\ \mathrm{weights},\qquad N:\ \mathrm{observations}'))
        self.beat(Indicate(formula[2]),self.eq(r'\log|A|=\sum_i\log a_i',r'\qquad(a_i>0)'))

    def hyper(self):
        self.effective_recap()
        alpha=ValueTracker(1);lam=np.array([.1,1,10,100]);base=-.8
        def bars():
            vals=lam/(alpha.get_value()+lam)
            return VGroup(*[Rectangle(width=.85,height=2*v,fill_color=POST,fill_opacity=.8,stroke_width=0).move_to([-3+2*i,base+v,0]) for i,v in enumerate(vals)])
        bar=always_redraw(bars);self.add(bar)
        for i,l in enumerate(lam):self.add(tex(r'\lambda='+f'{l:g}',21).move_to([-3+2*i,-1.15,0]))
        self.legend([('データで決まる寄与',POST),('4方向の説明例',MUTED)])
        self.beat(self.eq(r'\beta Hu_i=\lambda_i u_i'),self.note_anim('ここでは非負の固有値を使う説明例'))
        self.slider(alpha,.1,10,r'\alpha',pos=(0,-1.75,0))
        val=readout(r'\gamma=',lambda:np.sum(lam/(alpha.get_value()+lam)),(4.9,1.6,0),POST);self.add(val)
        self.beat(alpha.animate.set_value(10),self.eq(r'\gamma=\sum_{i=1}^W',r'\frac{\lambda_i}{\alpha+\lambda_i}'))
        self.beat(alpha.animate.set_value(.3),self.eq(r'\alpha_{\rm new}=\frac{\gamma}{w_{\rm MAP}^Tw_{\rm MAP}}'))
        for m in list(self.mobjects):
            if m not in [self.subtitle,self.formula]:self.remove(m)
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]))
        # Actual alternating updates, displayed one row per computed iterate.
        rows=VGroup(tex(r'k\qquad\alpha\qquad\beta\qquad\gamma',27))
        for i,(a,b,g) in enumerate(zip(D['history_alpha'],D['history_beta'],D['history_gamma'])):
            rows.add(tex(f'{i}\\qquad {a:.3f}\\qquad {b:.2f}\\qquad {g:.3f}',25,POST if i else MUTED))
        rows.arrange(DOWN,buff=.21).move_to([0,.5,0])
        self.beat(LaggedStart(*[FadeIn(r) for r in rows],lag_ratio=.3),self.eq(r'\beta_{\rm new}^{-1}=\frac{\sum_n(y_n-t_n)^2}{N-\gamma}'))
        self.beat(Indicate(rows[-1],scale_factor=1),self.note_anim('非線形モデルでは、固有値の変化を無視した再推定'))
        self.remove(*rows,self.note);self.note=None
        net=self.network(lambda:W,pos=(-3,.5,0))
        net[1][1].set_color(DATA);net[1][2].set_color(NOISE)
        lab1=tex('h_1',27,DATA).next_to(net[1][1],UP);lab2=tex('h_2',27,NOISE).next_to(net[1][2],DOWN)
        self.add(net,lab1,lab2)
        ax=self.axes(center=(3.25,.2,0),width=4.5,height=2.8)
        self.add(line(ax,GRID,MEAN,POST),jp("関数は同じ",22,POST).move_to([3.25,1.95,0]))
        self.beat(Swap(VGroup(net[1][1],lab1),VGroup(net[1][2],lab2)),self.eq(r'\text{tanh}:\quad M!\,2^M\ \text{equivalent modes}'))

    def class_axes(self):
        ax=self.axes(x=(-2.2,2.2,1),y=(-2.2,2.2,1),height=3.55,width=6.2,center=(-2,.2,0),xlabel='x_1',ylabel='x_2')
        dots=VGroup(*[Dot(ax.c2p(*p),radius=.045,color=NOISE if t else DATA) for p,t in zip(D['cx'],D['ct'])]);self.add(dots)
        return ax
    def contours(self,ax,w,scale=0,levels=(.1,.3,.5,.7,.9)):
        xy=np.linspace(-2.2,2.2,81);xx,yy=np.meshgrid(xy,xy);points=np.c_[xx.ravel(),yy.ravel()]
        a=output(w,points);base_activation=a.reshape(xx.shape)
        if scale: a=a*kappa(scale*variance(w,D['ccov'],points))
        p=expit(a).reshape(xx.shape)
        
        group=VGroup()
        for level in levels:
            curve=VMobject().set_stroke(POST if level==.5 else GOLD,3 if level==.5 else 1.5)
            for segment in contour_segments(xy,xy,base_activation if level==.5 else p,0 if level==.5 else level):
                curve.start_new_path(ax.c2p(*segment[0]));curve.add_line_to(ax.c2p(*segment[1]))
            group.add(curve)
        return group

    def classification(self):
        ax=self.class_axes();self.legend([('クラス0',DATA),('クラス1',NOISE),('確率0.5',POST)])
        weak=self.contours(ax,D['weak'],levels=(.5,));self.add(weak)
        tag=jp('弱い正則化',23,MAP).move_to([3.4,1.4,0]);self.add(tag)
        self.beat(Create(weak),self.eq(r'y_n=\sigma(a(x_n,w)),\quad t_n\in\{0,1\}'))
        chosen=self.contours(ax,D['cw'],levels=(.5,))
        self.beat(ReplacementTransform(weak,chosen),Transform(tag,jp('エビデンスで選択',23,POST).move_to(tag)))
        self.beat(self.eq(r'\log p(D\mid w)=\sum_n[t_n\log y_n+(1-t_n)\log(1-y_n)]'),self.note_anim('ベルヌーイ尤度：回帰の β は使わない'))
        self.beat(self.eq(r'E(w)=-\log p(D\mid w)+\frac\alpha2 w^Tw'),Indicate(chosen,scale_factor=1))
        evidence=VGroup(*[tex(r'\alpha='+f'{a:g}'+r'\ :\ '+f'{v:.2f}',23,POST if a==D['calpha'] else MUTED) for a,v in zip(D['class_alphas'],D['class_evidence'])]).arrange(DOWN,buff=.18).move_to([3.45,-.25,0])
        self.add(tex(r'\log p(D\mid\alpha)',23).move_to([3.45,.95,0]))
        self.beat(FadeIn(evidence),self.eq(r'\log p(D\mid\alpha)\simeq-E(w_{\rm MAP})-\frac12\log|A|+\frac W2\log\alpha'))
        extra=self.contours(ax,D['cw'],levels=(.1,.3,.7,.9))
        self.beat(Create(extra),self.note_anim('緑：0.5　黄：0.1、0.3、0.7、0.9（MAP予測）'))

    def marginal(self):
        ax=self.axes(x=(-7,9,2),y=(0,1,.5),width=9,height=3.15,center=(0,.35,0),xlabel='a',ylabel=r'\sigma(a)')
        grid=np.linspace(-7,9,321);sig=line(ax,grid,expit(grid),DATA);self.add(sig)
        self.legend([('シグモイド',DATA),('活性の密度',PRIOR),('平均した確率',POST)])
        self.beat(self.eq(r'a(x,w)\simeq a_{\rm MAP}+b^T(w-w_{\rm MAP}),\quad b=\nabla_w a'))
        self.sigmoid_recap()
        var=ValueTracker(.2)
        density=always_redraw(lambda:line(ax,grid,gaussian(grid,2,var.get_value()),PRIOR))
        self.add(density)
        meanline=always_redraw(lambda:line(ax,[-7,9],[integrated_sigmoid(2,var.get_value())]*2,POST,2))
        self.add(meanline)
        self.beat(self.eq(r'p(t=1\mid x,D)=\int\sigma(a)\mathcal N(a\mid a_{\rm MAP},\sigma_a^2)\,da'))
        self.slider(var,.2,9,r'\sigma_a^2',pos=(0,-1.85,0))
        self.add(readout(r'p=',lambda:integrated_sigmoid(2,var.get_value()),(4.5,1.55,0),POST))
        self.beat(var.animate.set_value(9))
        self.beat(self.eq(r'p(t=1\mid x,D)\simeq\sigma(\kappa(\sigma_a^2)a_{\rm MAP}),\quad',r'\kappa(v)=(1+\pi v/8)^{-1/2}'),self.note_anim('中心を固定：代表値 0.881 → 分布で平均 約0.717'))
        for m in list(self.mobjects):
            if m not in [self.subtitle,self.formula]:self.remove(m)
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]));self.note=None
        ax=self.class_axes();scale=ValueTracker(0)
        contours=always_redraw(lambda:self.contours(ax,D['cw'],scale.get_value()));self.add(contours)
        self.add(jp('緑：0.5の境界',23,POST).move_to([3.8,1.2,0]),jp('黄：周囲の確率',23,GOLD).move_to([3.8,.65,0]))
        self.beat(scale.animate.set_value(1),self.note_anim('同じ MAP を固定し、共分散を0倍から1倍へ'))
        self.beat(Indicate(contours,scale_factor=1),self.eq(r'D\ \longrightarrow\ p(w\mid D)\ \longrightarrow\ p(t\mid x,D)'),self.note_anim('代表値とともに、不確かさを予測へ渡す',POST))

    def body_card(self, label):
        """Suspend the body without changing its trackers or formula references."""
        saved=[m for m in self.mobjects if m is not self.subtitle]
        self.clear();self.subtitle=None
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]),
                 RoundedRectangle(width=10.6,height=4.8,corner_radius=.12,
                                  color='#FFFF00',stroke_width=1.2),
                 jp(label,23).move_to([-4.95,2.08,0],aligned_edge=LEFT))
        return saved

    def restore_body(self, saved):
        self.clear();self.add(*saved);self.subtitle=None

    def card_phases(self, *actions):
        """Lazy sentence actions share the PCM clock, including sentence boundaries."""
        assert len(actions)==len(self.beat_cues())
        self.beat(phases=[(name,self.sentence_duration(i),factory)
                          for i,(name,factory) in enumerate(actions)])

    def curves_recap(self):
        saved=self.body_card('復習: 3.3 重みの分布から予測へ')
        # Reproduce the six RBF posterior samples in 3.3 curves(), N=25.
        colors=[MAP,GOLD,PRIOR,NOISE,DATA,ManimColor('#77D49A')]
        xr=np.r_[.35,.75,.15,.9,np.random.default_rng(331).uniform(0,1,21)]
        tr=np.sin(2*PI*xr)+np.random.default_rng(332).normal(0,.2,25)
        phi=lambda x:np.exp(-.5*((np.atleast_1d(x)[:,None]-np.linspace(0,1,9))/.14)**2)
        f=phi(xr);cov=np.linalg.inv(2*np.eye(9)+25*f.T@f);mu=cov@(25*f.T@tr)
        weights=mu+np.random.default_rng(339).normal(size=(6,9))@np.linalg.cholesky(cov).T
        ax=self.axes(x=(0,1,.5),y=(-1.5,1.5,1),width=6.1,height=2.3,center=(-1.6,.1,0),ylabel='y')
        x=np.linspace(0,1,161);ys=phi(x)@weights.T;x0=.6;values=phi(x0)@weights.T
        curves=VGroup(*[line(ax,x,ys[:,i],colors[i],2,.75) for i in range(6)])
        guide=DashedLine(ax.c2p(x0,-1.4),ax.c2p(x0,1.4),color=MUTED)
        dots=VGroup(*[Dot(ax.c2p(x0,v),radius=.06,color=colors[i]) for i,v in enumerate(values[0])])
        axis=NumberLine(x_range=[-1.5,1.5,1],length=2.3,rotation=PI/2,include_ticks=False,color=MUTED).move_to([3.5,.1,0])
        targets=VGroup(*[Dot(axis.n2p(v),color=colors[i],radius=.06) for i,v in enumerate(values[0])])
        average=Dot(axis.n2p(values.mean()),color=MAP,radius=.1)
        mapping=jp('重みの候補 → ネットワークの予測の幅',22,POST).move_to([0,-2.02,0])
        self.add(curves,guide,axis,jp('同じ入力',18).move_to([-.6,1.53,0]),
                 jp('予測の高さ',20).move_to([3.5,1.53,0]),
                 jp('事後分布からの6標本',19,MUTED).move_to([-1.6,-1.63,0]))
        self.card_phases(
            ('候補の高さを集めて標本平均へ',lambda:Succession(FadeIn(dots),Transform(dots,targets),
              Transform(dots,VGroup(*[Dot(average.get_center(),color=c,radius=.055) for c in colors])),FadeIn(average))),
            ('今回のネットワークへ対応',lambda:AnimationGroup(FadeIn(mapping),
              FadeIn(jp('標本の平均',20,MAP).move_to([3.5,-1.63,0])))))
        self.restore_body(saved)

    def laplace_recap(self):
        saved=self.body_card('復習: 4.4 ラプラス近似')
        a=ValueTracker(1.)
        left=self.axes(x=(-1.5,1.5,1),y=(0,5,2),width=3.4,height=2.05,center=(-2.7,.12,0),xlabel='s',ylabel='h(s)')
        right=self.axes(x=(-2.5,2.5,2),y=(0,1,.5),width=3.4,height=2.05,center=(2.7,.12,0),xlabel='s',ylabel='q(s)')
        x=np.linspace(-1.5,1.5,161);z=np.linspace(-2.5,2.5,161)
        bowl=always_redraw(lambda:line(left,x,.5*a.get_value()*x*x,MAP))
        bell=always_redraw(lambda:line(right,z,gaussian(z,0,1/a.get_value()),MAP))
        span=always_redraw(lambda:Line(right.c2p(-1/np.sqrt(a.get_value()),.12),right.c2p(1/np.sqrt(a.get_value()),.12),color=GOLD,stroke_width=5))
        transform=tex(r'e^{-h}\ \longrightarrow',24,GOLD).move_to([0,.45,0])
        condition=jp('条件：負の対数事後の曲率が全方向で正（正定値）',19).move_to([0,-2.04,0])
        mapping=VGroup(jp('本編の近似',18),Dot(color=MAP),tex(r'\to',22),Dot(color=GOLD),
                       jp('ガウス',18),Dot(color=MAP),tex(r'\to',22),Dot(color=POST)).arrange(RIGHT,buff=.15).move_to([0,-1.62,0])
        self.add(bowl,bell,span,transform,
                 tex(r'h(s)=\tfrac12 A s^2',25,MAP).move_to([-2.7,1.52,0]),
                 tex(r'\sigma=1/\sqrt A',25,GOLD).move_to([2.7,1.52,0]),condition)
        self.card_phases(
            ('曲率を増やすとガウスの幅が狭まる',lambda:a.animate.set_value(4)),
            ('正定値の条件と本編の配色へ',lambda:AnimationGroup(Indicate(condition,scale_factor=1),FadeIn(mapping))))
        self.restore_body(saved)

    def projection_aid(self):
        saved=self.body_card('補足: 感度と予測の広がり')
        blue,yellow,green='#58C4DD','#FFFF00','#83C167'
        scale=ValueTracker(1)
        # Same covariance determines ellipse, projection extent and variance.
        cov=np.diag([4.,1.]);radius=np.sqrt(np.diag(cov));center=np.array([-3.,.2,0])
        ellipse=Ellipse(width=2*radius[0]*.65,height=2*radius[1]*.65,color=blue).move_to(center)
        shadow=Line(center+[-radius[0]*.65,-1.,0],center+[radius[0]*.65,-1.,0],color=yellow,stroke_width=5)
        guides=VGroup(*[DashedLine(center+[sign*radius[0]*.65,0,0],center+[sign*radius[0]*.65,-1,0],color=yellow) for sign in [-1,1]])
        ax=NumberLine(x_range=[-4,4,2],length=4.1,include_numbers=True,font_size=19,color=MUTED).move_to([2.55,-.3,0])
        sd=lambda:np.sqrt(np.array([scale.get_value(),0])@cov@np.array([scale.get_value(),0]))
        width=always_redraw(lambda:Line(ax.n2p(-sd()),ax.n2p(sd()),color=green,stroke_width=6))
        nums=readout(r'\mathrm{SD}=',sd,(2.5,.6,0),green)
        var=tex(r'\mathrm{Var}:\ 4\ \longrightarrow\ 16\quad(\times4)',29,green).move_to([1.4,-1.63,0])
        self.add(ellipse,shadow,guides,ax,width,nums,
                 tex(r'C=A^{-1}=\mathrm{diag}(4,1)',26,blue).move_to([-2.6,1.48,0]),
                 tex(r'g=(1,0)\ \longrightarrow\ (2,0)',26,yellow).move_to([2.4,1.48,0]),
                 Arrow([-1.2,.2,0],[.1,.2,0],color=yellow),
                 jp('単位方向への影',19,yellow).move_to([-3,-1.18,0]),
                 jp('説明用の2重み。雲は固定。幅は標準偏差。',19,MUTED).move_to([0,-2.04,0]))
        self.card_phases(
            ('感度1から2で標準偏差2から4へ',lambda:scale.animate.set_value(2)),
            ('標準偏差を二乗して分散4から16へ',lambda:Write(var)))
        self.restore_body(saved)

    def effective_recap(self):
        saved=self.body_card('復習: 3.5 エビデンスと有効パラメータ数')
        colors=[DATA,MAP,PRIOR,ManimColor('#77D49A')]
        a=ValueTracker(.1);lam=np.array([.1,1,10,100])
        ax=self.axes(x=(.5,4.5,1),y=(0,1,.5),width=6,height=1.8,center=(-1,.05,0),xlabel='i',ylabel='q_i')
        def bars():
            q=lam/(a.get_value()+lam)
            return VGroup(*[Rectangle(width=.55,height=max(.005,1.8*v),fill_color=colors[i],fill_opacity=.85,stroke_width=0).move_to(ax.c2p(i+1,v/2)) for i,v in enumerate(q)])
        bs=always_redraw(bars)
        count=readout(r'\gamma=',lambda:sum(lam/(a.get_value()+lam)),(3.55,.35,0),GOLD)
        mapping=VGroup(jp('本編の寄与の棒',19),*[Dot(color=c,radius=.05) for c in colors],
                       tex(r'\to',23),Dot(color=POST)).arrange(RIGHT,buff=.14).move_to([0,-1.67,0])
        self.add(bs,count,tex(r'q_i=\frac{\lambda_i}{\alpha+\lambda_i},\qquad\gamma=\sum_i q_i',28,GOLD).move_to([0,1.48,0]),
                 jp('説明用の4方向：事前が強いほど0、データが強いほど1',19,MUTED).move_to([0,-1.27,0]),
                 jp('今回の更新式は、固有値の変化を無視した近似',20).move_to([0,-2.07,0]))
        self.card_phases(
            ('事前の強さを増やして寄与を縮める',lambda:a.animate.set_value(10)),
            ('四方向の寄与を合計する',lambda:a.animate.set_value(1)),
            ('今回の近似へつなぐ',lambda:FadeIn(mapping)))
        self.restore_body(saved)

    def sigmoid_recap(self):
        saved=self.body_card('復習: 4.5 シグモイドに通してから平均')
        red,purple,yellow,green='#FF7884','#C5A0F4','#FFE18B','#7CDBAD'
        ax=self.axes(x=(-7,11,4),y=(0,1,.5),width=6.2,height=2.1,center=(-1.5,.1,0),xlabel='a',ylabel='')
        x=np.linspace(-7,11,301);var=9.;end=ValueTracker(-7)
        sig=line(ax,x,expit(x),red);pdf=line(ax,x,gaussian(x,2,var),purple)
        def area():
            xx=np.linspace(-7,max(-6.999,end.get_value()),181)
            return Polygon(ax.c2p(xx[0],0),*[ax.c2p(v,p) for v,p in zip(xx,expit(xx)*gaussian(xx,2,var))],ax.c2p(xx[-1],0),fill_color=yellow,fill_opacity=.65,stroke_width=0)
        fill=always_redraw(area)
        p=integrated_sigmoid(2,var)
        summary=VGroup(tex(r'\sigma(2)=0.881',25,red),tex(r'\mathbb E[\sigma(a)]=0.717',25,green)).arrange(DOWN,buff=.4).move_to([3.05,.25,0])
        mapping=VGroup(jp('本編のシグモイド',18),Dot(color=red),tex(r'\to',22),Dot(color=DATA),
                       jp('平均',18,green),tex(r'\to',22),Dot(color=POST)).arrange(RIGHT,buff=.14).move_to([0,-2.02,0])
        self.add(sig,pdf,fill,Dot(ax.c2p(2,expit(2)),color=red),
                 jp('赤：確率　紫：密度　黄：積の面積',20).move_to([0,1.54,0]),
                 tex(r'a\sim\mathcal N(2,9),\qquad p=\int\sigma(a)q(a)\,da',26).move_to([0,-1.57,0]))
        self.card_phases(
            ('積の面積を足して確率を平均',lambda:Succession(end.animate.set_value(11),FadeIn(summary))),
            ('活性の線形化から同じ平均へ',lambda:FadeIn(mapping)))
        self.restore_body(saved)
