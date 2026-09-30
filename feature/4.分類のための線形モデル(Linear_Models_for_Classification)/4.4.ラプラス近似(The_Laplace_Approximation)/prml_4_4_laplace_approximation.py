"""PRML 4.4 — continuous local geometry, implemented with Manim CE."""
from manim import *
import numpy as np
from scene_support import NarratedScene, jp, tex, polyline, BLUE, RED, GOLD, GREEN, PURPLE, MUTED
from laplace_model import *


AID_INPUT="#58C4DD"
AID_OPERATION="#FFFF00"
AID_RESULT="#83C167"
AID_COMPARE="#9A72AC"

class PRML44LaplaceApproximation(NarratedScene):
    def construct(self):
        for i,method in enumerate([self.question,self.mode_and_log,self.curvature,self.normalization,
                                   self.multivariate,self.integral,self.model_evidence,self.bic,self.limits]):
            self.begin(i);method();self.finish()
        self.save_timeline()

    def question(self):
        self.legend(('元の分布',BLUE),('ガウス近似',RED))
        ax=self.axes(y=(0,.9,.3),ylabel='p(z)')
        curve=self.curve(ax,pdf)
        self.beat(Create(curve))
        t=ValueTracker(-1.2)
        dot=always_redraw(lambda:Dot(ax.c2p(t.get_value(),pdf(t.get_value())),color=GOLD))
        self.add(dot)
        self.beat(t.animate.set_value(2.1))
        self.beat(self.equation(r'p(z\mid D)',r'\propto',r'p(D\mid z)',r'p(z)',colors={0:BLUE,2:GOLD,3:PURPLE}),t.animate.set_value(Z0))
        self.beat(self.equation(r'f(z)=',r'e^{-z^2/(2\cdot1.1^2)}',r'\frac{1}{1+e^{-(4z+0.8)}}',colors={1:PURPLE,2:GOLD}),t.animate.set_value(1.6))
        scan=ValueTracker(-1.99)
        area=always_redraw(lambda:self.area(ax,pdf,hi=scan.get_value()))
        self.add(area)
        self.beat(scan.animate.set_value(3),self.equation(r'p(z)=\frac{f(z)}{Z}',r',\quad Z=\int f(z)\,dz',colors={0:BLUE,1:GOLD}))
        focus=Circle(radius=.5,color=GOLD).move_to(ax.c2p(Z0,pdf(Z0)))
        self.beat(Create(focus),t.animate.set_value(Z0))
        self.gaussian_recap()
        q=self.curve(ax,gaussian,RED)
        self.beat(Create(q),FadeOut(area),FadeOut(focus))
        self.beat(t.animate.set_value(Z0+.45),self.equation(r'q(z)=\mathcal N(z\mid z_0,A^{-1})',colors={0:RED}))

    def mode_and_log(self):
        self.legend(('関数と対数',BLUE),('頂上・接線',GOLD),('二次近似',RED))
        x=ValueTracker(-.6);scale=ValueTracker(1);morph=ValueTracker(0)
        ax=self.changing_axes((-1,2.5,.5),lambda:(-4*morph.get_value(),1),(-4,1,1))
        fn=lambda z:(1-morph.get_value())*scale.get_value()*f(z)+morph.get_value()*logf(z)
        c=always_redraw(lambda:self.curve(ax,fn))
        dot=always_redraw(lambda:Dot(ax.c2p(x.get_value(),fn(x.get_value())),color=GOLD))
        self.add(c,dot)
        self.beat(x.animate.set_value(Z0+.3),self.equation(r'z_0=\arg\max_z f(z)',colors={0:GOLD}))
        def tangent():
            z=x.get_value();v=fn(z);s=(1-morph.get_value())*scale.get_value()*f(z)*grad(z)+morph.get_value()*grad(z)
            return Line(ax.c2p(z-.3,v-.3*s),ax.c2p(z+.3,v+.3*s),color=GOLD)
        line=always_redraw(tangent);self.add(line)
        self.beat(x.animate.set_value(Z0),self.equation(r'f\prime(z_0)=0',colors={0:GOLD}))
        self.slider(scale,.4,1.2,'c');self.beat(scale.animate.set_value(.45),self.equation(r'\arg\max_z c f(z)=z_0\quad(c>0)'))
        self.beat(morph.animate.set_value(1),self.equation(r'g(z)=\ln f(z)',colors={0:BLUE}))
        marker=ValueTracker(Z0-.28)
        near=always_redraw(lambda:Dot(ax.c2p(marker.get_value(),logf(marker.get_value())),color=RED,radius=.065))
        self.add(near)
        self.beat(marker.animate.set_value(Z0+.28))
        quadratic=lambda z:logf(Z0)-.5*A*(z-Z0)**2
        quad=self.curve(ax,quadratic,RED)
        self.beat(Create(quad),self.equation(r'g(z)\simeq',r'g(z_0)',r'+g\prime(z_0)(z-z_0)',r'+\frac12g^{\prime\prime}(z_0)(z-z_0)^2',colors={1:BLUE,2:GOLD,3:RED},size=29))
        self.beat(self.equation(r'\ln f(z)\simeq',r'\ln f(z_0)',r'-\frac A2(z-z_0)^2',colors={1:BLUE,2:RED}),marker.animate.set_value(Z0))
        self.beat(self.equation(r'A=-\left.\frac{d^2\ln f}{dz^2}\right|_{z_0}',colors={0:RED}),marker.animate.set_value(Z0+.2))

    def curvature(self):
        legend=self.legend(('元の負の対数（谷底を0に）',BLUE),('二次の形',RED))
        m=ValueTracker(0)
        ax=self.changing_axes((-1.5,2.5,1),lambda:(0,5-4*m.get_value()),(0,5,1),height=3.2)
        original=self.curve(ax,bowl);self.beat(Create(original),self.equation(r'h(z)=\ln f(z_0)-\ln f(z)'))
        a=ValueTracker(A)
        fn=lambda z:(1-m.get_value())*.5*a.get_value()*(z-Z0)**2+m.get_value()*np.exp(-.5*a.get_value()*(z-Z0)**2)
        curve=always_redraw(lambda:self.curve(ax,fn,RED));self.add(curve)
        self.beat(self.equation(r'h(z)\simeq\frac A2(z-z_0)^2',colors={0:RED}),Create(Dot(ax.c2p(Z0,0),color=GOLD)))
        self.slider(a,.7,6,'A',RED)
        self.number('A=',a.get_value,[-2.7,2.25,0],RED)
        span=always_redraw(lambda:Line(ax.c2p(Z0-1/np.sqrt(a.get_value()),.5),ax.c2p(Z0+1/np.sqrt(a.get_value()),.5),color=GOLD,stroke_width=4))
        self.add(span)
        self.beat(a.animate.set_value(6),FadeOut(original))
        self.beat(a.animate.set_value(.7))
        # Reuse x coordinates; rescale vertical view to show the whole bell.
        self.remove(span)
        self.beat(m.animate.set_value(1),self.equation(r'e^{-h(z)}\simeq e^{-A(z-z_0)^2/2}',colors={0:RED}))
        # Enlarge the normalized bell vertically with a labelled new density axis.
        self.remove(curve,ax,legend)
        self.legend(('正規化したガウス密度',RED),('標準偏差の幅',GOLD))
        # Remove old axis labels only; title, slider and captions are retained below.
        for obj in list(self.mobjects):
            if isinstance(obj,VGroup) and obj is not self.formula and obj is not self.caption and len(obj)>5:self.remove(obj)
        ax=self.axes(x=(-1.5,2.5,1),y=(0,1.1,.5),height=3.2,center=(0,.5,0),ylabel='q(z)')
        curve=always_redraw(lambda:self.curve(ax,lambda z:gaussian(z,Z0,a.get_value()),RED));self.add(curve)
        self.beat(a.animate.set_value(6),self.equation(r'q(z)=\sqrt{\frac{A}{2\pi}}\,e^{-A(z-z_0)^2/2}',colors={0:RED}))
        width=always_redraw(lambda:Line(ax.c2p(Z0-1/np.sqrt(a.get_value()),.12),ax.c2p(Z0+1/np.sqrt(a.get_value()),.12),color=GOLD,stroke_width=4));self.add(width)
        self.beat(a.animate.set_value(1),self.equation(r'\mathrm{Var}[z]=A^{-1}',r',\qquad \sigma=A^{-1/2}',colors={0:RED,1:GOLD}))
        self.beat(a.animate.set_value(2.5),self.equation(r'A>0',colors={0:GOLD}))

    def normalization(self):
        self.legend(('元の関数・密度',BLUE),('局所近似・ガウス',RED))
        ax=self.axes(y=(0,1,.25))
        n=ValueTracker(0)
        pf=lambda z:f(z)/((1-n.get_value())+n.get_value()*Z)
        qf=lambda z:local(z)/((1-n.get_value())+n.get_value()*ZL)
        pcurve=always_redraw(lambda:self.curve(ax,pf));qcurve=always_redraw(lambda:self.curve(ax,qf,RED))
        self.add(pcurve)
        self.beat(Create(qcurve),self.equation(r'f(z)\simeq',r'f(z_0)e^{-A(z-z_0)^2/2}',colors={1:RED}))
        areas=always_redraw(lambda:VGroup(self.area(ax,pf),self.area(ax,qf,RED)))
        self.add(areas)
        self.number('Z=',lambda:Z,[-3,2.2,0],BLUE);self.number(r'Z_{\rm L}=',lambda:ZL,[3,2.2,0],RED)
        scan=ValueTracker(-1.5)
        dot=always_redraw(lambda:Dot(ax.c2p(scan.get_value(),pf(scan.get_value())),color=GOLD));self.add(dot)
        self.beat(scan.animate.set_value(2.5))
        self.beat(n.animate.set_value(1),self.equation(r'p=f/Z',r',\qquad q=f_{\rm local}/Z_{\rm L}',colors={0:BLUE,1:RED}))
        difference=Line(ax.c2p(Z0,pdf(Z0)),ax.c2p(Z0,gaussian(Z0)),color=GOLD,stroke_width=5)
        self.beat(Create(difference),scan.animate.set_value(Z0))
        self.beat(self.equation(r'q(z)=\sqrt{\frac A{2\pi}}e^{-A(z-z_0)^2/2}',r'=\mathcal N(z\mid z_0,A^{-1})',colors={0:RED,1:RED},size=29),FadeOut(areas))
        self.beat(scan.animate.set_value(2))
        meanline=DashedLine(ax.c2p(MEAN,0),ax.c2p(MEAN,.6),color=PURPLE)
        modeline=DashedLine(ax.c2p(Z0,0),ax.c2p(Z0,.8),color=GOLD)
        self.beat(Create(meanline),Create(modeline),self.equation(r'z_0='+f'{Z0:.3f}',r',\quad \mathbb E_p[z]='+f'{MEAN:.3f}',colors={0:GOLD,1:PURPLE}))
        self.beat(scan.animate.set_value(Z0),self.equation(r'z_0\quad\longrightarrow\quad A\quad\longrightarrow\quad q(z)',colors={0:RED}))

    def multivariate(self):
        self.legend(('同じ高さの等高線',RED),('曲率の大きい方向',GOLD),('小さい方向',BLUE))
        ax=self.axes(x=(-2.5,2.5,1),y=(-2.5,2.5,1),width=4.0,height=4.0,center=(-2.1,.4,0),xlabel='z_1',ylabel='z_2')
        a=ValueTracker(1);theta=ValueTracker(0)
        def contours():
            r=rotation(theta.get_value());out=VGroup();t=np.linspace(0,2*np.pi,161)
            for radius in [.7,1.4,2.1]:
                xy=r@(np.array([np.cos(t)/np.sqrt(a.get_value()),np.sin(t)])*radius)
                out.add(polyline([ax.c2p(*z) for z in xy.T],RED,2))
            return out
        ell=always_redraw(contours);self.add(ell,Dot(ax.c2p(0,0),color=GOLD))
        self.beat(self.equation(r'(z-z_0)^T A(z-z_0)=c',colors={0:RED}))
        self.beat(a.animate.set_value(5))
        self.beat(theta.animate.set_value(.65))
        mat=tex(r'A=-\nabla\nabla\ln f(z_0)',30,GOLD).move_to([3,1.5,0])
        self.beat(Write(mat),self.equation(r'\ln f(z)\simeq\ln f(z_0)-\frac12(z-z_0)^TA(z-z_0)',size=28))
        self.hessian_aid()
        def vectors():
            r=rotation(theta.get_value())
            return VGroup(Arrow(ax.c2p(0,0),ax.c2p(*(r[:,0]/np.sqrt(a.get_value())*1.5)),buff=0,color=GOLD),Arrow(ax.c2p(0,0),ax.c2p(*(r[:,1]*1.5)),buff=0,color=BLUE))
        vec=always_redraw(vectors);self.add(vec)
        self.number(r'\lambda_1=',a.get_value,[3,.6,0],GOLD)
        self.beat(theta.animate.set_value(1),self.equation(r'\sigma_i=1/\sqrt{\lambda_i}',colors={0:GOLD}))
        self.beat(a.animate.set_value(2),self.equation(r'q(z)=\frac{|A|^{1/2}}{(2\pi)^{M/2}}e^{-\frac12(z-z_0)^TA(z-z_0)}',r',\quad\Sigma=A^{-1}',colors={0:RED,1:GOLD},size=27))
        # Show a flat direction explicitly, not an ill-defined Gaussian ellipse.
        self.remove(ell,vec)
        a.set_value(0)
        flat=VGroup(*[Line(ax.c2p(-2.3,y),ax.c2p(2.3,y),color=RED) for y in [-1.4,-.7,.7,1.4]])
        self.beat(Create(flat),self.equation(r'A=\begin{pmatrix}0&0\\0&1\end{pmatrix}\quad\Rightarrow\quad |A|=0',colors={0:GOLD}))
        self.remove(flat);a.set_value(2);self.add(ell,vec)
        self.beat(a.animate.set_value(5),self.equation(r'A\succ0\quad\Longleftrightarrow\quad\lambda_i>0\ (\forall i)',colors={0:GOLD}))

    def integral(self):
        self.legend(('未正規化のガウス形',RED),('積分＝面積',PURPLE))
        ax=self.axes(x=(-3,3,1),y=(0,2.4,.6),height=3.35,center=(0,.5,0))
        h=ValueTracker(2);sigma=ValueTracker(.75)
        fn=lambda z:h.get_value()*np.exp(-.5*(z/sigma.get_value())**2)
        curve=always_redraw(lambda:self.curve(ax,fn,RED));self.add(curve)
        self.beat(self.equation(r'f_{\rm local}(z)=f(z_0)e^{-A(z-z_0)^2/2}',colors={0:RED}))
        def rectangles():
            group=VGroup();dx=.15
            for z in np.arange(-3,3,dx):
                v=fn(z+dx/2)
                group.add(Polygon(ax.c2p(z,0),ax.c2p(z,v),ax.c2p(z+dx,v),ax.c2p(z+dx,0),fill_color=PURPLE,fill_opacity=.35,stroke_color=PURPLE,stroke_width=.4))
            return group
        bars=always_redraw(rectangles);self.add(bars)
        self.beat(self.equation(r'Z=\int f(z)\,dz',colors={0:PURPLE}))
        self.number(r'\mathrm{area}=',lambda:h.get_value()*sigma.get_value()*np.sqrt(2*np.pi),[2.8,2.2,0],PURPLE)
        self.slider(sigma,.2,1,r'\sigma',GOLD)
        self.beat(sigma.animate.set_value(.25))
        self.beat(h.animate.set_value(1.4),sigma.animate.set_value(.85))
        self.beat(self.equation(r'Z_{\rm L}=',r'f(z_0)',r'\sqrt{2\pi/A}',colors={0:PURPLE,1:RED,2:GOLD}),sigma.animate.set_value(.7))
        self.beat(sigma.animate.set_value(.9),self.equation(r'|A|=\prod_{i=1}^{M}\lambda_i',r',\quad \prod_i\lambda_i^{-1/2}=|A|^{-1/2}',colors={1:GOLD},size=30))
        self.determinant_aid()
        self.beat(self.equation(r'Z\simeq',r'f(z_0)',r'\frac{(2\pi)^{M/2}}{|A|^{1/2}}',colors={0:PURPLE,1:RED,2:GOLD}),sigma.animate.set_value(.6))
        self.beat(sigma.animate.set_value(.85),self.equation(r'Z\simeq Z_{\rm L}',colors={0:PURPLE}))

    def model_evidence(self):
        self.evidence_recap()
        self.legend(('尤度',GOLD),('事前',BLUE),('積・証拠',PURPLE))
        ax=self.axes(x=(-4.5,4.5,1.5),y=(0,1.1,.5),height=3.25,center=(0,.5,0),xlabel=r'\theta')
        width=ValueTracker(.8);prior=ValueTracker(5)
        likelihood=lambda z:np.exp(-.5*(z/width.get_value())**2)
        lc=always_redraw(lambda:self.curve(ax,likelihood,GOLD));self.add(lc)
        self.beat(self.equation(r'L(\theta)=p(D\mid\theta)',colors={0:GOLD}))
        pr=always_redraw(lambda:VGroup(Line(ax.c2p(-prior.get_value()/2,0),ax.c2p(-prior.get_value()/2,1/prior.get_value()),color=BLUE),Line(ax.c2p(-prior.get_value()/2,1/prior.get_value()),ax.c2p(prior.get_value()/2,1/prior.get_value()),color=BLUE),Line(ax.c2p(prior.get_value()/2,0),ax.c2p(prior.get_value()/2,1/prior.get_value()),color=BLUE)))
        product=always_redraw(lambda:self.area(ax,lambda z:likelihood(z)/prior.get_value(),PURPLE,lo=-prior.get_value()/2,hi=prior.get_value()/2))
        self.add(pr,product)
        self.beat(self.equation(r'p(D)=\int',r'p(D\mid\theta)',r'p(\theta)',r'\,d\theta',colors={1:GOLD,2:BLUE}))
        self.number('p(D)=',lambda:evidence(width.get_value(),prior.get_value()),[3,2.15,0],PURPLE)
        mapdot=always_redraw(lambda:Dot(ax.c2p(0,1/prior.get_value()),color=PURPLE))
        self.add(mapdot)
        self.beat(self.equation(r'\theta_{\rm MAP}=\arg\max_\theta p(D\mid\theta)p(\theta)',colors={0:PURPLE}))
        self.beat(width.animate.set_value(.3))
        self.beat(prior.animate.set_value(8))
        self.beat(self.equation(r'\ln p(D)\simeq',r'\ln p(D\mid\theta_{\rm MAP})',r'+\ln p(\theta_{\rm MAP})',r'+\frac M2\ln(2\pi)',r'-\frac12\ln|A|',colors={1:GOLD,2:BLUE,3:PURPLE,4:PURPLE},size=26),width.animate.set_value(.6))
        bracket=Brace(VGroup(*self.formula[2:]),UP,color=PURPLE,buff=.12)
        label=jp('オッカム因子',20,PURPLE).next_to(bracket,UP,buff=.1)
        self.beat(Create(bracket),FadeIn(label),width.animate.set_value(.4))
        self.remove(bracket,label)
        self.beat(self.equation(r'A=-\nabla\nabla\ln[p(D\mid\theta)p(\theta)]\big|_{\theta_{\rm MAP}}',colors={0:PURPLE},size=30),width.animate.set_value(.55))

    def bic(self):
        self.legend(('パラメータ2個',BLUE),('パラメータ5個',RED))
        n=ValueTracker(20)
        conditions=jp('広い事前・十分なデータ・全方向が決まる',23,GOLD).move_to([0,2.16,0])
        h_label=jp('H：1件あたりの平均的な曲率の行列',21,MUTED).move_to([0,1.65,0])
        self.add(conditions,h_label)
        concentration=always_redraw(lambda:Ellipse(width=4*np.sqrt(20/n.get_value()),height=2*np.sqrt(20/n.get_value()),color=PURPLE).move_to([0,.3,0]))
        self.add(concentration)
        self.beat(n.animate.set_value(60),self.equation(r'A\simeq N H',colors={0:GOLD}))
        self.beat(n.animate.set_value(20),self.equation(r'|A|\simeq N^M|H|',colors={0:GOLD}))
        self.remove(concentration,h_label)
        ax=self.axes(x=(0,3,1),y=(0,15,5),width=6.4,height=2.5,center=(0,.4,0),xlabel='M')
        # Category labels replace the numeric x-axis labels below.
        self.remove(self.mobjects[-1])
        self.add(tex('2',23,BLUE).move_to(ax.c2p(1,0)+DOWN*.22),tex('5',23,RED).move_to(ax.c2p(2,0)+DOWN*.22),tex(r'\frac M2\ln N',25).move_to([-4.7,.7,0]))
        def bars():
            out=VGroup()
            for x,m,c in [(1,2,BLUE),(2,5,RED)]:
                v=m/2*np.log(n.get_value());out.add(Polygon(ax.c2p(x-.22,0),ax.c2p(x-.22,v),ax.c2p(x+.22,v),ax.c2p(x+.22,0),fill_color=c,fill_opacity=.8,stroke_width=0))
            return out
        bar=always_redraw(bars);self.add(bar)
        self.slider(n,20,200,'N',GOLD,y=-1.75)
        self.number('N=',n.get_value,[4.5,1.2,0])
        self.number('M=2:',lambda:np.log(n.get_value()),[4.5,.4,0],BLUE)
        self.number('M=5:',lambda:2.5*np.log(n.get_value()),[4.5,-.4,0],RED)
        self.beat(self.equation(r'\frac12\ln|A|\simeq\frac M2\ln N+\frac12\ln|H|',size=30))
        self.beat(n.animate.set_value(200))
        self.beat(n.animate.set_value(100))
        self.beat(self.equation(r'S=\ln p(D\mid\theta_{\rm MAP})-\frac M2\ln N',colors={0:GREEN},size=30),n.animate.set_value(200))
        self.beat(self.equation(r'\mathrm{BIC}=-2\ln p(D\mid\hat\theta_{\rm ML})+M\ln N',colors={0:GREEN},size=29),n.animate.set_value(100))
        self.beat(ReplacementTransform(conditions,jp('平らな方向があると、この簡略化は使えない',23,GOLD).move_to(conditions)),self.equation(r'|H|=0\quad\Rightarrow\quad\ln|H|\ \mathrm{undefined}',colors={0:GOLD}))

    def limits(self):
        self.legend(('元の密度',BLUE),('選んだモードの近似',RED))
        ax=self.axes(x=(-3,3,1),y=(0,1.2,.4),height=3.4,center=(0,.45,0))
        weight=ValueTracker(.001)
        pc=always_redraw(lambda:self.curve(ax,lambda z:mixture(z,weight.get_value())))
        mu=ValueTracker(-1.5);prec=ValueTracker(5)
        qc=always_redraw(lambda:self.curve(ax,lambda z:gaussian(z,mu.get_value(),prec.get_value()),RED))
        self.add(pc,qc)
        self.beat(self.equation(r'q(z)=\mathcal N(z\mid z_0,A^{-1})',colors={0:RED}))
        self.beat(weight.animate.set_value(.6),self.equation(r'p(z)=(1-r)\mathcal N(z\mid-1.5,1/5)+r\mathcal N(z\mid1.6,1/7)',colors={0:BLUE},size=26))
        m,a=mixture_laplace(1)
        self.beat(mu.animate.set_value(m),prec.animate.set_value(a),self.equation(r'z_0:\ -1.5\ \longrightarrow\ 1.6',colors={0:RED}))
        self.remove(pc,qc)
        skew=self.curve(ax,pdf);q=self.curve(ax,gaussian,RED)
        self.beat(Create(skew),Create(q),self.equation(r'p(z)\ne q(z)',colors={0:GOLD}))
        # Reference label runs inside the existing beat; no extra narration.
        recap=VGroup(jp("復習: 1.2 密度の変換",22))
        recap.add(SurroundingRectangle(recap[0],color="#FFFF00",buff=.12,stroke_width=1.5))
        recap.move_to([-2.8,2.2,0]);self.add(recap)
        # Positive-variable mapping is shown explicitly with its Jacobian.
        self.beat(self.equation(r'u=\ln\tau',r',\quad p_\tau(\tau)=\frac{p_u(\ln\tau)}{\tau}\quad(\tau>0)',colors={0:GOLD,1:PURPLE},size=29),FadeOut(skew),FadeOut(q))
        self.remove(recap)
        power=ValueTracker(1)
        # A normalized powered target illustrates concentration; no claim of all datasets.
        def concentrated(z):
            v=power.get_value();norm=np.trapezoid(np.exp(v*(logf(GRID)-logf(Z0))),GRID)
            return np.exp(v*(logf(z)-logf(Z0)))/norm
        # Horizontal standardized coordinates keep the increasingly sharp peak visible.
        self.remove(ax)
        for obj in list(self.mobjects):
            if isinstance(obj,VGroup) and obj is not self.formula and obj is not self.caption and len(obj)>5:self.remove(obj)
        ax=self.axes(x=(-3,3,1),y=(0,.7,.2),height=3.4,center=(0,.45,0),xlabel=r'\sqrt{nA}(z-z_0)')
        def standardized(x):
            v=power.get_value();s=np.sqrt(v*A)
            return concentrated(Z0+x/s)/s
        pc=always_redraw(lambda:self.curve(ax,standardized));qc=self.curve(ax,lambda z:gaussian(z,0,1),RED)
        self.add(pc,qc)
        self.number('n=',power.get_value,[3.2,2.15,0],GOLD)
        self.add(jp('幅をそろえて形を比較',20,GOLD).move_to([-1.8,2.15,0]))
        self.beat(power.animate.set_value(12),self.equation(r'p_n(z)\propto f(z)^n',r',\quad\mathrm{standardized}',colors={0:BLUE},size=30))
        self.beat(power.animate.set_value(20),self.equation(r'z_0\ \longrightarrow\ A\ \longrightarrow\ \mathcal N(z\mid z_0,A^{-1})',colors={0:RED}))
        self.beat(power.animate.set_value(30),self.equation(r'p(w\mid D)\approx\mathcal N(w\mid w_{\rm MAP},S_N)',colors={0:PURPLE}))

    def gaussian_recap(self):
        saved=self.body_card('復習: 2.3 ガウス分布')
        # 2.3 shape(): red normalized bell, yellow centre, red width control.
        ax=Axes(x_range=(-5,5,2),y_range=(0,.7,.2),x_length=5.4,y_length=2.15,
                tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=18)).move_to([-2.3,.1,0])
        mu=ValueTracker(0);sd=ValueTracker(.75)
        bell=always_redraw(lambda:self.curve(ax,lambda x:gaussian(x,mu.get_value(),sd.get_value()**-2),RED))
        centre=always_redraw(lambda:DashedLine(ax.c2p(mu.get_value(),0),ax.c2p(mu.get_value(),gaussian(mu.get_value(),mu.get_value(),sd.get_value()**-2)),color=GOLD))
        def knob(tr,lo,hi,x,label,color):
            rail=Line([x-1,-1.55,0],[x+1,-1.55,0],color=MUTED)
            dot=always_redraw(lambda:Dot(rail.point_from_proportion((tr.get_value()-lo)/(hi-lo)),color=color,radius=.065))
            return VGroup(rail,dot,tex(label,25,color).move_to([x,-1.95,0]))
        self.add(ax,bell,centre,tex('z',23).next_to(ax.x_axis,RIGHT),
                 knob(mu,-1.5,1.5,-3.8,r'\mu',GOLD),knob(sd,.55,1.4,-.8,r'\sigma',RED),
                 jp('密度の面積 = 1',21,MUTED).move_to([-2.3,1.55,0]))
        source=VGroup(VGroup(tex(r'\mu',30,GOLD),jp('中心',23,GOLD)).arrange(RIGHT,buff=.3),
                      VGroup(tex(r'\sigma^2',30,RED),jp('分散',23,RED)).arrange(RIGHT,buff=.3)).arrange(DOWN,buff=.6).move_to([3,.4,0])
        self.add(source)
        self.beat(Succession(mu.animate.set_value(1.2),sd.animate.set_value(1.3)))
        target=VGroup(tex(r'\mu=z_0',32,GOLD),tex(r'\sigma^2=A^{-1}',32,RED)).arrange(DOWN,buff=.6).move_to(source)
        labels=VGroup(jp('今回：元の山のモード',21,GOLD).move_to([3,1.55,0]),
                      jp('負の対数の曲率 A > 0',21,RED).move_to([3,-1.05,0]))
        self.beat(self.brief(Succession(FadeOut(source),AnimationGroup(FadeIn(target),FadeIn(labels)))),
                  mu.animate.set_value(Z0),sd.animate.set_value(A**-.5))
        self.restore_body(saved)

    def hessian_aid(self):
        saved=self.body_card('補足: 縦へ動くと、横の傾きは？')
        self.add(jp('負の対数の説明用の例',20,MUTED).move_to([2.65,1.45,0]))
        ax=Axes(x_range=(-1.5,1.5,1),y_range=(-.2,2.8,1),x_length=4.7,y_length=2.35,
                tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=18)).move_to([-2.65,.1,0])
        y=ValueTracker(0);x=ValueTracker(-.6)
        h=lambda u:.5*(u*u+u*y.get_value()+y.get_value()**2)
        graph=always_redraw(lambda:self.curve(ax,h,AID_INPUT))
        def tangent():
            u=x.get_value();slope=u+.5*y.get_value()
            return Line(ax.c2p(u-.6,h(u)-.6*slope),ax.c2p(u+.6,h(u)+.6*slope),color=AID_OPERATION,stroke_width=4)
        line=always_redraw(tangent)
        dot=always_redraw(lambda:Dot(ax.c2p(x.get_value(),h(x.get_value())),color=AID_OPERATION))
        self.add(ax,graph,line,dot,tex('x',24).next_to(ax.x_axis,RIGHT),
                 jp('y を固定した断面：横軸 x、縦軸 h',18,MUTED).move_to([-2.65,1.55,0]),
                 tex(r'h(x,y)=\frac{x^2+xy+y^2}{2}',29,AID_INPUT).move_to([2.7,.85,0]))
        self.number('y=',y.get_value,[-4,-1.55,0],AID_COMPARE)
        self.number(r'\partial h/\partial x=',lambda:x.get_value()+.5*y.get_value(),[-1,-1.55,0],AID_OPERATION)
        self.beat(x.animate.set_value(0))
        atzero=tex(r'x=0:\quad h_x=\tfrac12y',29,AID_OPERATION).move_to([2.7,-.05,0])
        self.add(atzero)
        self.beat(y.animate.set_value(1))
        mat=Matrix([['1','0.5'],['0.5','1']],h_buff=1.0,v_buff=.65).scale(.65).move_to([2.95,-1.1,0])
        entries=mat.get_entries();entries[1].set_color(AID_COMPARE);entries[2].set_color(AID_COMPARE)
        label=tex('H=',28).next_to(mat,LEFT,buff=.15)
        cross=tex(r'\frac{\Delta h_x}{\Delta y}=\frac{0.5}{1}=0.5',28,AID_COMPARE).move_to([2.7,-.05,0])
        self.beat(self.brief(Succession(FadeOut(atzero),AnimationGroup(FadeIn(cross),FadeIn(mat),FadeIn(label)))),
                  AnimationGroup(Circumscribe(entries[1],color=AID_COMPARE,buff=.08),
                                 Circumscribe(entries[2],color=AID_COMPARE,buff=.08)),
                  self.equation(r'H=\nabla\nabla h=A\succ0',colors={0:AID_RESULT},size=29))
        self.restore_body(saved)

    def determinant_aid(self):
        saved=self.body_card('補足: 行列式から、幅の積へ')
        self.add(jp('説明用の二方向：主軸に沿って測る',21,MUTED).move_to([0,1.4,0]))
        a=ValueTracker(1)
        square=Square(side_length=2,color=MUTED,stroke_width=1.5).move_to([-2.7,-.1,0])
        rect=always_redraw(lambda:Rectangle(width=2/np.sqrt(a.get_value()),height=2,
                    stroke_color=AID_INPUT,fill_color=AID_INPUT,fill_opacity=.35).move_to(square.get_left(),aligned_edge=LEFT))
        self.add(square,rect,tex(r'\sigma_2=1',28,AID_INPUT).move_to([-4.55,-.1,0]))
        width=always_redraw(lambda:Brace(rect,DOWN,color=AID_INPUT,buff=.08))
        self.add(width)
        self.number(r'\sigma_1=',lambda:a.get_value()**-.5,[-2.7,-1.85,0],AID_INPUT)
        self.number(r'\lambda_1=',a.get_value,[2.6,.75,0],AID_OPERATION)
        self.add(tex(r'\lambda_2=1',28,AID_OPERATION).move_to([2.6,.05,0]))
        self.beat(a.animate.set_value(4))
        determinant=tex(r'|A|=4\times1=4',30,AID_OPERATION).move_to([2.6,-.65,0])
        result=tex(r'\sigma_1\sigma_2=|A|^{-1/2}=\frac12',30,AID_COMPARE).move_to([2.6,-1.5,0])
        self.beat(self.brief(AnimationGroup(FadeIn(determinant),FadeIn(result))),
                  Circumscribe(rect,color=AID_OPERATION),
                  self.equation(r'Z_{\rm L}=f(z_0)(2\pi)^{M/2}',r'|A|^{-1/2}',colors={1:AID_RESULT},size=29))
        self.restore_body(saved)

    def evidence_recap(self):
        saved=self.body_card('復習: 3.4 モデル証拠')
        # Reproduce 3.4 integral(): observation 1, noise .35, uniform prior 1/4.
        ax=Axes(x_range=(-2,2,1),y_range=(0,1.3,.5),x_length=5.2,y_length=2.65,
                tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=18)).move_to([-2.45,-.1,0])
        like=lambda w:gaussian(w,1,.35**-2)
        product=lambda w:like(w)/4
        lc=self.curve(ax,like,BLUE)
        prior=polyline([ax.c2p(-2,0),ax.c2p(-2,.25),ax.c2p(2,.25),ax.c2p(2,0)],PURPLE)
        pc=self.curve(ax,product,GOLD)
        self.add(ax,lc,prior,tex('w',24).next_to(ax.x_axis,RIGHT),
                 jp('尤度',22,BLUE).move_to([2.5,1.25,0]),jp('事前',22,PURPLE).move_to([2.5,.65,0]),
                 jp('積の面積 = 証拠',22,GOLD).move_to([2.5,.05,0]))
        scan=ValueTracker(-1.95)
        def rectangles():
            out=VGroup()
            for w in np.arange(-1.95,2,.1):
                if w>scan.get_value():break
                v=product(w)
                out.add(Polygon(ax.c2p(w-.05,0),ax.c2p(w-.05,v),ax.c2p(w+.05,v),ax.c2p(w+.05,0),
                                fill_color=GOLD,fill_opacity=.5,stroke_color=GOLD,stroke_width=.4))
            return out
        bars=always_redraw(rectangles)
        self.add(bars)
        self.beat(Create(pc),scan.animate.set_value(2))
        maximum=Dot(ax.c2p(1,product(1)),color=RED,radius=.07)
        label=jp('頂上の高さ',20,RED).move_to([2.5,-.6,0])
        bridge=VGroup(jp('今回の図の色へ',20).move_to([2.5,-1.15,0]),
                      tex(r'L:\ ',23,BLUE),tex(r'\longrightarrow',23),tex('L',23,GOLD),
                      tex(r'p:\ ',23,PURPLE),tex(r'\longrightarrow',23),tex('p',23,BLUE))
        VGroup(*bridge[1:]).arrange(RIGHT,buff=.14).move_to([2.5,-1.6,0])
        product_bridge=VGroup(jp('積',19,GOLD),tex(r'\longrightarrow',21),jp('積・証拠',19,PURPLE)).arrange(RIGHT,buff=.12).move_to([2.5,-1.97,0])
        bridge.add(product_bridge)
        self.beat(self.brief(AnimationGroup(FadeIn(maximum),FadeIn(label),FadeIn(bridge))),
                  self.equation(r'p(D)=\int',r'L(w)',r'p(w)',r'\,dw',colors={1:BLUE,2:PURPLE}),
                  Circumscribe(bars,color=GOLD))
        self.restore_body(saved)
