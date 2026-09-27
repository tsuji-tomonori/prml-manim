"""PRML 4.4 — continuous local geometry, implemented with Manim CE."""
from manim import *
import numpy as np
from scene_support import NarratedScene, jp, tex, polyline, BLUE, RED, GOLD, GREEN, PURPLE, MUTED
from laplace_model import *


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
        q=self.curve(ax,gaussian,RED)
        self.beat(Create(q),FadeOut(area),FadeOut(focus))
        self.beat(t.animate.set_value(Z0+.45),self.equation(r'q(z)=\mathcal N(z\mid z_0,A^{-1})',colors={0:RED}))

    def mode_and_log(self):
        self.legend(('関数と対数',BLUE),('頂上・接線',GOLD),('二次近似',RED))
        ax=self.axes(x=(-1,2.5,.5),y=(-4,1,1),height=3.35,center=(0,.5,0))
        x=ValueTracker(-.6);scale=ValueTracker(1);morph=ValueTracker(0)
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
        ax=self.axes(x=(-1.5,2.5,1),y=(0,5,1),height=3.2,center=(0,.5,0))
        original=self.curve(ax,bowl);self.beat(Create(original),self.equation(r'h(z)=\ln f(z_0)-\ln f(z)'))
        a=ValueTracker(A);m=ValueTracker(0)
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
        ax=self.axes(x=(-2.5,2.5,1),y=(-2.5,2.5,1),width=4.4,height=4.4,center=(-2.1,.25,0),xlabel='z_1',ylabel='z_2')
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
        def vectors():
            r=rotation(theta.get_value())
            return VGroup(Arrow(ax.c2p(0,0),ax.c2p(*(r[:,0]/np.sqrt(a.get_value())*1.5)),buff=0,color=GOLD),Arrow(ax.c2p(0,0),ax.c2p(*(r[:,1]*1.5)),buff=0,color=BLUE))
        vec=always_redraw(vectors);self.add(vec)
        self.number(r'\lambda_1=',a.get_value,[3,.6,0],GOLD)
        self.beat(theta.animate.set_value(1),self.equation(r'\sigma_i=1/\sqrt{\lambda_i}',colors={0:GOLD}))
        self.beat(a.animate.set_value(2),self.equation(r'q(z)=\frac{|A|^{1/2}}{(2\pi)^{M/2}}e^{-\frac12(z-z_0)^TA(z-z_0)}',r',\quad\Sigma=A^{-1}',colors={0:RED,1:GOLD},size=27))
        # Show a flat direction explicitly, not an ill-defined Gaussian ellipse.
        self.remove(ell,vec)
        flat=VGroup(*[Line(ax.c2p(-2.3,y),ax.c2p(2.3,y),color=RED) for y in [-1.4,-.7,.7,1.4]])
        self.beat(Create(flat),self.equation(r'A=\begin{pmatrix}0&0\\0&1\end{pmatrix}\quad\Rightarrow\quad |A|=0',colors={0:GOLD}))
        self.remove(flat);self.add(ell,vec)
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
        self.beat(self.equation(r'Z\simeq',r'f(z_0)',r'\frac{(2\pi)^{M/2}}{|A|^{1/2}}',colors={0:PURPLE,1:RED,2:GOLD}),sigma.animate.set_value(.6))
        self.beat(sigma.animate.set_value(.85),self.equation(r'Z\simeq Z_{\rm L}',colors={0:PURPLE}))

    def model_evidence(self):
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
        self.beat(Create(Dot(ax.c2p(0,1/prior.get_value()),color=PURPLE)),self.equation(r'\theta_{\rm MAP}=\arg\max_\theta p(D\mid\theta)p(\theta)',colors={0:PURPLE}))
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
        self.add(conditions)
        concentration=always_redraw(lambda:Ellipse(width=4*np.sqrt(20/n.get_value()),height=2*np.sqrt(20/n.get_value()),color=PURPLE).move_to([0,.3,0]))
        self.add(concentration)
        self.beat(n.animate.set_value(60),self.equation(r'A\simeq N H',colors={0:GOLD}))
        self.beat(n.animate.set_value(20),self.equation(r'|A|\simeq N^M|H|',colors={0:GOLD}))
        self.remove(concentration)
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
        # Positive-variable mapping is shown explicitly with its Jacobian.
        self.beat(self.equation(r'u=\ln\tau',r',\quad p_\tau(\tau)=\frac{p_u(\ln\tau)}{\tau}\quad(\tau>0)',colors={0:GOLD,1:PURPLE},size=29),FadeOut(skew),FadeOut(q))
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
