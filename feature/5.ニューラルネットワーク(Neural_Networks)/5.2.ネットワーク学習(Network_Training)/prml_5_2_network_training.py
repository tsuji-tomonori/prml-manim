"""PRML 5.2 — linked numerical experiments in Manim Community."""
import json
from pathlib import Path
import numpy as np
from manim import *
from training_model import *
from scene_support import *

RED=RED_CLASS; BLUE=BLUE_CLASS; GREEN=GREEN_CLASS; YELLOW=YELLOW_ACC; PURPLE=PURPLE_ACC

class PRML52NetworkTraining(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.entries={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,fn in enumerate([self.regression,self.gaussian,self.binary,self.multiclass,self.landscape,
                              self.quadratic,self.eigen,self.information,self.steps,self.online]):
            self.begin(i);fn()
            assert self.bi==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml52_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def regression(self):
        ax=self.plot_axes(x=(-1.8,1.8,.6),y=(-1.2,1.2,.6),width=8,height=3.35,center=(-1,-.15,0),labels=('x',r't,\ y'))
        for label in ax.x_axis.numbers:
            label.set_y(ax.c2p(0,-1.2)[1]-.2)
        w=ValueTracker(.2)
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.065,color=BLUE) for x,t in zip(X,T)])
        g=always_redraw(lambda:curve(ax,lambda x:prediction(x,w.get_value()),color=RED))
        self.add(dots,g,knob('w=',w,.1,1.4,at=(-1,-2.5,0)))
        self.equation(r'y(x,w)=',r'w',r'\tanh(1.8x)-0.35\tanh(3(x-0.5))',size=28)[1].set_color(RED)
        self.beat(w.animate.set_value(.35))
        self.beat(w.animate.set_value(1.15))
        residual=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,prediction(x,w.get_value())),color=YELLOW,stroke_width=2.5) for x,t in zip(X,T)]))
        self.add(residual, *[jp(t,19,co).move_to([5,1.7-.45*i,0]) for i,(t,co) in enumerate([('青：観測',BLUE),('赤：予測',RED),('黄：ずれ',YELLOW)])])
        self.beat(w.animate.set_value(1.4))
        f=self.equation(r'E(w)=\frac12\sum_{n=1}^N',r'(y(x_n,w)-t_n)^2');f[1].set_color(YELLOW)
        self.add(number('E=',lambda:regression_error(w.get_value()),[5,.05,0],YELLOW))
        self.beat(w.animate.set_value(1.1))
        self.beat(w.animate.set_value(.2))
        self.beat(w.animate.set_value(REG_W))

    def gaussian(self):
        self.gaussian_recap()
        ax=self.plot_axes(x=(-2.5,2.5,1),y=(0,1.5,.5),width=8,height=3.15,center=(-1,-.15,0),labels=('t',r'p(t\mid x,w)'))
        mu=ValueTracker(-.6);beta=ValueTracker(2)
        density=lambda t:np.sqrt(beta.get_value()/(2*np.pi))*np.exp(-.5*beta.get_value()*(t-mu.get_value())**2)
        g=always_redraw(lambda:curve(ax,density,color=GREEN))
        center=always_redraw(lambda:DashedLine(ax.c2p(mu.get_value(),0),ax.c2p(mu.get_value(),density(mu.get_value())),color=RED))
        height=always_redraw(lambda:Line(ax.c2p(.8,0),ax.c2p(.8,density(.8)),color=BLUE,stroke_width=5))
        self.add(g,center,height,Dot(ax.c2p(.8,0),color=BLUE),number(r'p(t_n)=',lambda:density(.8),[5,.6,0],BLUE),knob(r'y=',mu,-1,1,at=(-1,-2.5,0),color=RED))
        self.equation(r'p(t\mid x,w)=\mathcal N(t\mid y(x,w),\beta^{-1})',size=31)
        self.beat(mu.animate.set_value(.8))
        self.beat(mu.animate.set_value(-.5))
        self.add(number(r'\beta=',beta.get_value,[5,-.3,0],PURPLE))
        self.beat(beta.animate.set_value(8))
        self.equation(r'-\ln p(\mathbf t\mid X,w,\beta)=',r'\frac\beta2\sum_n(y_n-t_n)^2',r'-\frac N2\ln\beta+\frac N2\ln(2\pi)',size=25)[1].set_color(YELLOW)
        self.beat(mu.animate.set_value(.1))
        self.equation(r'\beta_{\rm ML}^{-1}=\frac1N\sum_n(y(x_n,w_{\rm ML})-t_n)^2',size=31)
        self.beat(mu.animate.set_value(.8))
        self.equation(r'p(\mathbf t\mid x,w)=\mathcal N(\mathbf t\mid\mathbf y,\beta^{-1}I)',r'\quad\beta_{\rm ML}^{-1}=\frac1{NK}\sum_n\|\mathbf y_n-\mathbf t_n\|^2',size=25)
        self.beat(beta.animate.set_value(3))

    def binary(self):
        self.loss_recap()
        ax=self.plot_axes(x=(-4,4,2),y=(0,1,.5),width=8,height=3.3,center=(-1,-.1,0),labels=('a','y'))
        a=ValueTracker(-3);target=ValueTracker(1)
        g=curve(ax,sigmoid,color=RED)
        point=always_redraw(lambda:Dot(ax.c2p(a.get_value(),sigmoid(a.get_value())),color=YELLOW,radius=.09))
        self.add(g,point,knob('a=',a,-4,4,at=(-1,-2.5,0)))
        self.add(number('y=',lambda:sigmoid(a.get_value()),[5,.9,0],RED),number('E_n=',lambda:binary_loss(a.get_value(),target.get_value()),[5,-.1,0],YELLOW))
        self.equation(r'y=\sigma(a)=\frac1{1+e^{-a}}',r'\qquad t\in\{0,1\}')
        self.beat(a.animate.set_value(2.5))
        self.equation(r't=1:\quad E_n=-\ln y',size=35)
        self.beat(a.animate.set_value(-1.5))
        self.equation(r'p(t\mid x,w)=y^t(1-y)^{1-t}')
        self.beat(a.animate.set_value(-4))
        target.set_value(0)
        self.equation(r'E=-\sum_n[t_n\ln y_n+(1-t_n)\ln(1-y_n)]',size=29)
        self.add(note('いまの正解：0',at=(4.8,1.75,0),color=BLUE))
        self.beat(a.animate.set_value(2.5))
        self.equation(r'\frac{\partial E_n}{\partial a}=',r'y-t',size=38)[1].set_color(YELLOW)
        self.add(number('y-t=',lambda:sigmoid(a.get_value())-target.get_value(),[5,-1.05,0],YELLOW))
        self.beat(a.animate.set_value(.5))
        self.beat(a.animate.set_value(-4))

    def multiclass(self):
        a=ValueTracker(-1);c=ValueTracker(0);mode=ValueTracker(0)
        scores=lambda:np.array([a.get_value(),.5,-.2])+c.get_value()
        probs=lambda:softmax(scores()) if mode.get_value()<.5 else sigmoid(scores())
        colors=[RED,BLUE,GREEN]
        bars=always_redraw(lambda:VGroup(*[Rectangle(width=1.15,height=max(.015,2.15*float(p)),stroke_width=0,fill_color=co,fill_opacity=.85).move_to([x,-1.6+1.075*p,0]) for x,p,co in zip([-3,0,3],probs(),colors)]))
        self.add(bars,Line([-4,-1.6,0],[4,-1.6,0],color=MUTED))
        for k,x in enumerate([-3,0,3]):
            self.add(tex(f'C_{k+1}',29,colors[k]).move_to([x,-1.95,0]),number('y=',lambda k=k:probs()[k],[x,.95,0],colors[k]),number('a=',lambda k=k:scores()[k],[x,1.65,0],colors[k]))
        slider=knob('a_1=',a,-1,3,at=(-1,-2.6,0))
        self.add(slider,number(r'\sum y_k=',lambda:probs().sum(),[5,.1,0],YELLOW))
        self.equation(r'y_k=\frac{e^{a_k}}{\sum_j e^{a_j}}',size=34)
        self.beat(a.animate.set_value(0))
        self.beat(a.animate.set_value(2))
        self.equation(r'E=-\sum_n\sum_k t_{nk}\ln y_k(x_n,w)',size=31)
        self.beat(a.animate.set_value(3))
        self.equation(r'\operatorname{softmax}(a+c\mathbf1)=\operatorname{softmax}(a)',size=31)
        self.remove(*slider.get_family())
        shift_slider=knob('c=',c,0,1.5,at=(-1,-2.6,0))
        self.add(shift_slider)
        self.beat(c.animate.set_value(1.5))
        self.remove(*shift_slider.get_family());self.add(slider)
        mode.set_value(1);c.set_value(0)
        self.equation(r'y_k=\sigma(a_k),\quad p(\mathbf t\mid x,w)=\prod_k y_k^{t_k}(1-y_k)^{1-t_k}',size=28)
        self.beat(a.animate.set_value(.5))
        mode.set_value(0)
        self.equation(r'\sum_k y_k=1',r'\qquad \frac{\partial E_n}{\partial a_k}=y_k-t_k',size=34)
        self.beat(a.animate.set_value(2.5))

    def landscape(self):
        ax=self.plot_axes(x=(-2.3,2.3,1),y=(0,2.7,1),width=9,height=3.3,center=(-.5,-.1,0),labels=('w','E'))
        g=curve(ax,landscape,color=BLUE); q=ValueTracker(-2.1)
        dot=always_redraw(lambda:Dot(ax.c2p(q.get_value(),landscape(q.get_value())),color=YELLOW,radius=.085))
        tangent=always_redraw(lambda:curve(ax,lambda x:landscape(q.get_value())+landscape_grad(q.get_value())*(x-q.get_value()),lo=q.get_value()-.25,hi=q.get_value()+.25,color=GREEN))
        self.add(g,dot,note('誤差の形を調べる説明用の断面'))
        self.equation(r'E(w)=0.18(w^2-2.2)^2+0.16w+0.45',size=29)
        self.beat(q.animate.set_value(-1.9))
        self.add(tangent)
        self.beat(q.animate.set_value(-1.7))
        r=ValueTracker(2.1);rd=always_redraw(lambda:Dot(ax.c2p(r.get_value(),landscape(r.get_value())),color=RED,radius=.085))
        self.add(rd)
        self.beat(q.animate.set_value(STATIONARY[0]),r.animate.set_value(STATIONARY[2]))
        labels=VGroup(jp('大域最小',23,GREEN).move_to([-3,-2.1,0]),jp('局所最小',23,RED).move_to([2.5,-2.1,0]))
        self.add(labels);self.beat(Indicate(labels,scale_factor=1.07))
        self.equation(r'\nabla E(w)=0\quad\text{(stationary)}')
        mid=Dot(ax.c2p(STATIONARY[1],landscape(STATIONARY[1])),color=PURPLE,radius=.1)
        self.add(mid);self.beat(Indicate(mid,scale_factor=2))
        self.equation(r'w^{(\tau+1)}=w^{(\tau)}+\Delta w^{(\tau)}')
        q.set_value(-2.1);r.set_value(2.1)
        self.beat(q.animate.set_value(STATIONARY[0]),r.animate.set_value(STATIONARY[2]))

    def quadratic(self):
        self.taylor_recap()
        ax=self.plot_axes(x=(-2.3,2.3,1),y=(-.3,2.7,1),width=9,height=3.3,center=(-.5,-.1,0),labels=('w','E'))
        q=ValueTracker(-1.2); span=ValueTracker(.35)
        g=curve(ax,landscape,color=BLUE)
        linear=lambda x:landscape(q.get_value())+landscape_grad(q.get_value())*(x-q.get_value())
        quad=lambda x:linear(x)+.5*landscape_hess(q.get_value())*(x-q.get_value())**2
        tangent=always_redraw(lambda:curve(ax,linear,lo=q.get_value()-span.get_value(),hi=q.get_value()+span.get_value(),color=GREEN))
        parabola=always_redraw(lambda:curve(ax,quad,lo=q.get_value()-span.get_value(),hi=q.get_value()+span.get_value(),color=YELLOW))
        dot=always_redraw(lambda:Dot(ax.c2p(q.get_value(),landscape(q.get_value())),color=WHITE))
        self.add(g,dot,tangent,note('青：元の関数　緑：接線　黄：二次近似'))
        self.equation(r'\delta=w-\widehat w',r'\qquad E(w)\approx E(\widehat w)+b\delta')
        self.beat(q.animate.set_value(-1.35))
        self.beat(span.animate.set_value(.65))
        self.add(parabola);self.equation(r'E(w)\approx E(\widehat w)+b\delta+',r'\frac12H\delta^2')[1].set_color(YELLOW)
        self.beat(span.animate.set_value(.4))
        self.beat(q.animate.set_value(-1.05))
        self.equation(r'b=\nabla E(\widehat w)',r'\quad H_{ij}=\frac{\partial^2 E}{\partial w_i\partial w_j}',size=31)
        self.beat(q.animate.set_value(-1.4))
        self.gradient_aid()
        self.beat(Indicate(self.formula[1],color=YELLOW,scale_factor=1.04))
        self.equation(r'E(w)\approx E(w^*)+\frac12(w-w^*)^T H(w-w^*)',size=30)
        self.beat(q.animate.set_value(STATIONARY[0]))

    def contour(self,ax,lam=9,theta=.4):
        r=rotation(theta); group=VGroup()
        for radius in [.45,.9,1.4,1.9]:
            angle=np.linspace(0,TAU,161)
            p=(r@np.array([radius*np.cos(angle),radius/np.sqrt(lam)*np.sin(angle)])).T
            group.add(VMobject().set_points_as_corners([ax.c2p(*v) for v in p]).set_stroke(BLUE,2,.8))
        return group

    def eigen(self):
        ax=self.plot_axes(x=(-2.3,2.3,1),y=(-2.3,2.3,1),width=4.0,height=4.0,center=(-1.8,-.2,0),labels=('w_1','w_2'))
        lam=ValueTracker(1);theta=ValueTracker(0)
        lines=always_redraw(lambda:self.contour(ax,lam.get_value(),theta.get_value()))
        self.add(lines,Dot(ax.c2p(0,0),color=WHITE),number(r'\lambda_1=',lambda:1,[4,.8,0],GREEN),number(r'\lambda_2=',lam.get_value,[4,0,0],PURPLE))
        self.equation(r'E-E^*=\frac12(w-w^*)^TH(w-w^*)',size=31)
        self.beat(Create(lines))
        self.beat(lam.animate.set_value(9))
        vectors=always_redraw(lambda:VGroup(*[Arrow(ax.c2p(0,0),ax.c2p(*(rotation(theta.get_value())[:,k]*1.8)),buff=0,color=co,stroke_width=3) for k,co in enumerate([GREEN,PURPLE])]))
        self.add(vectors)
        for k,co in enumerate([GREEN,PURPLE]):
            label=tex(f'u_{k+1}',25,co)
            label.add_updater(lambda m,k=k:m.move_to(ax.c2p(*(rotation(theta.get_value())[:,k]*2.1))+(LEFT*.6 if k==1 else RIGHT*.15)))
            self.add(label)
        self.equation(r'Hu_i=\lambda_i u_i',r'\qquad u_i^T u_j=\delta_{ij}')
        self.beat(theta.animate.set_value(.6))
        self.equation(r'w-w^*=\sum_i\alpha_i u_i',r'\quad E-E^*\approx\frac12\sum_i\lambda_i\alpha_i^2',size=27)
        self.add(tex(r'r_i\propto\lambda_i^{-1/2}',30,YELLOW).move_to([4,-1,0]))
        self.beat(lam.animate.set_value(4))
        self.equation(r'v^THv>0\ (v\ne0)\quad\Longleftrightarrow\quad\lambda_i>0\ \forall i',size=29)
        self.beat(theta.animate.set_value(.25))
        self.equation(r'\lambda_i=0:\quad\text{higher-order terms matter}',size=29)
        def narrow_valley():
            self.equation(r'0<\lambda_1\ll\lambda_2',size=37)
            return lam.animate.set_value(9)
        self.beat(actions=[lambda:Indicate(self.formula,scale_factor=1.02),narrow_valley])

    def information(self):
        ax=self.plot_axes(x=(-2.3,2.3,1),y=(-2.3,2.3,1),width=3.9,height=3.9,center=(-2,-.15,0),labels=('w_1','w_2'))
        q=ValueTracker(0)
        pos=lambda:np.array([1.2-.7*q.get_value(),.65-.8*q.get_value()])
        dot=always_redraw(lambda:Dot(ax.c2p(*pos()),color=YELLOW,radius=.09))
        self.add(self.contour(ax),dot,number('E=',lambda:.5*pos()@H@pos(),[4,.7,0],YELLOW))
        self.equation(r'w\longmapsto E(w)\quad\text{(one value)}')
        self.beat(q.animate.set_value(.2))
        arrow=always_redraw(lambda:Arrow(ax.c2p(*pos()),ax.c2p(*(pos()+.45*H@pos())),buff=0,color=GREEN))
        self.add(arrow,number(r'g_1=',lambda:(H@pos())[0],[4,0,0],GREEN),number(r'g_2=',lambda:(H@pos())[1],[4,-.7,0],PURPLE))
        self.equation(r'\nabla E=\begin{pmatrix}\partial E/\partial w_1\\\partial E/\partial w_2\end{pmatrix}',size=31)
        self.beat(q.animate.set_value(.6))
        self.equation(r'b,H:\quad W+\frac{W(W+1)}2=\frac{W(W+3)}2',size=32)
        self.beat(q.animate.set_value(.1))
        self.equation(r'\text{values: }O(W^3)\qquad\text{gradients: }O(W^2)',size=30)
        self.add(note('二次近似での情報量の見積もり／収束保証ではない',color=YELLOW))
        self.beat(q.animate.set_value(.5))
        self.equation(r'\text{backpropagation: }O(W)\quad\text{per example}',size=31)
        self.beat(q.animate.set_value(.8))
        self.equation(r'\nabla E\quad\longrightarrow\quad\Delta w',size=42)
        self.beat(q.animate.set_value(.9))

    def steps(self):
        ax=self.plot_axes(x=(-2.4,2.4,1),y=(-2.4,2.4,1),width=3.9,height=3.9,center=(-1.7,-.15,0),labels=('w_1','w_2'))
        self.add(self.contour(ax))
        eta=ValueTracker(.04);q=ValueTracker(0);path=descent(.04)
        point=lambda:sample_path(path,q.get_value())
        dot=always_redraw(lambda:Dot(ax.c2p(*point()),color=YELLOW,radius=.09))
        trace=always_redraw(lambda:VMobject().set_points_as_corners([ax.c2p(*v) for v in [*path[:int(q.get_value())+1],point()+1e-10]]).set_stroke(YELLOW,2.5))
        arrow=always_redraw(lambda:Arrow(ax.c2p(*point()),ax.c2p(*(point()-.15*H@point())),buff=0,color=GREEN))
        self.add(dot,trace,arrow,number(r'\eta=',eta.get_value,[4,.8,0],PURPLE),number('E=',lambda:.5*point()@H@point(),[4,0,0],YELLOW))
        self.equation(r'w^{(\tau+1)}=w^{(\tau)}-',r'\eta',r'\nabla E(w^{(\tau)})',size=31)[1].set_color(PURPLE)
        self.beat(q.animate.set_value(1))
        self.beat(q.animate.set_value(20))
        path=descent(.20);q.set_value(0);eta.set_value(.20)
        self.beat(q.animate.set_value(20))
        path=descent(.24,7);q.set_value(0);eta.set_value(.24)
        self.beat(q.animate.set_value(7))
        path=descent(.2);q.set_value(0);eta.set_value(.2)
        self.equation(r'0<\eta<\frac2{\lambda_{\max}}=\frac29',size=37)
        self.add(note('この正定値二次関数での安定条件',color=YELLOW))
        self.beat(q.animate.set_value(15))
        self.equation(r'\text{conjugate gradients}\quad /\quad\text{quasi-Newton}',size=29)
        self.beat(q.animate.set_value(20))

    def online(self):
        self.add(jp('復習: 3.1 一例ずつの勾配で更新',20).move_to([0,2.94,0]))
        ax=self.plot_axes(x=(-1.6,1.8,.8),y=(0,10,2),width=8.4,height=3.2,center=(-.7,-.1,0),labels=('w','E'))
        total=curve(ax,scalar_error,color=BLUE)
        pieces=VGroup(*[curve(ax,lambda w,t=t:.5*(w-t)**2,color=co).set_stroke(width=1.6) for t,co in zip(TARGETS,[RED,GREEN,PURPLE,YELLOW])])
        self.add(total,pieces,note('青：全体の和　細い曲線：一例の誤差'))
        self.equation(r'y(x_n,w)=w',r'\quad E_n=\frac12(w-t_n)^2',r'\quad E=\sum_nE_n',size=30)
        self.beat(LaggedStart(*[Indicate(p,scale_factor=1.02) for p in pieces],lag_ratio=.2))
        q=ValueTracker(0);path=batch_online()
        w=lambda:float(sample_path(path,q.get_value()))
        dot=always_redraw(lambda:Dot(ax.c2p(w(),scalar_error(w())),color=WHITE,radius=.09))
        self.add(dot,number('w=',w,[5,.8,0],WHITE),number('E=',lambda:scalar_error(w()),[5,-.1,0],BLUE))
        self.equation(r'w\leftarrow w-\eta\sum_n(w-t_n)',size=35)
        self.beat(q.animate.set_value(12))
        path=batch_online(True);q.set_value(0)
        active=always_redraw(lambda:Dot(ax.c2p(TARGETS[min(int(q.get_value()),23)%4],0),color=YELLOW,radius=.1))
        self.add(active);self.equation(r'w\leftarrow w-\eta(w-t_n)',size=35)
        self.beat(q.animate.set_value(16))
        self.equation(r'\nabla E(w^*)=0\quad\not\Rightarrow\quad\nabla E_n(w^*)=0',size=32)
        self.beat(q.animate.set_value(24))
        self.equation(r'\text{batch}\quad\longleftrightarrow\quad\text{mini-batch}\quad\longleftrightarrow\quad\text{one example}',size=26)
        q.set_value(0);self.beat(q.animate.set_value(16))
        self.equation(r'p(t\mid x,w)\ \longrightarrow\ E(w)\ \longrightarrow\ \nabla E\ \longrightarrow\ w_{\rm new}',size=32)
        self.beat(q.animate.set_value(24))

    def body_card(self,label):
        """Temporarily replace the body; keep trackers and the original PCM clock."""
        saved=[m for m in self.mobjects if m is not self.subtitle]
        self.clear()
        self.add(*[m for m in saved if m.get_center()[1]>3.1])
        self.add(RoundedRectangle(width=10.4,height=4.6,corner_radius=.12,
                                  color='#FFFF00',stroke_width=1.2).move_to([0,.1,0]),
                 jp(label,23).move_to([-4.85,2.06,0],aligned_edge=LEFT))
        return saved

    def restore_body(self,saved):
        self.clear();self.add(*saved);self.subtitle=None

    def gaussian_recap(self):
        saved=self.body_card('復習: 3.1 ガウスのノイズと二乗誤差')
        # 3.1 likelihood(): horizontal green density, red prediction,
        # blue observation and gold residual / square, at one fixed input.
        data,model,basis,gold='#58B5ED','#FF6B77','#77D49A','#FFE079'
        ax=self.plot_axes(x=(0,1.2,.6),y=(-1.5,1.5,1),width=2.3,height=2.45,
                          center=(-3.15,-.05,0),labels=('p','t'))
        obs=ValueTracker(1.1);ts=np.linspace(-1.5,1.5,161)
        profile=VMobject().set_points_as_corners([ax.c2p(np.exp(-t*t)/np.sqrt(np.pi),t) for t in ts]).set_stroke(basis,3)
        residual=always_redraw(lambda:Line(ax.c2p(.08,0),ax.c2p(.08,obs.get_value()),color=gold,stroke_width=4))
        dot=always_redraw(lambda:Dot(ax.c2p(0,obs.get_value()),color=data,radius=.075))
        square=always_redraw(lambda:Square(side_length=obs.get_value(),color=gold,fill_opacity=.4).move_to([-.55,-.45,0]))
        formula=tex(r'-\ln p=\frac{\beta}{2}\sum_n(t_n-y_n)^2+\mathrm{const}',28,gold).move_to([1.05,1.23,0])
        mapping=VGroup(tex(r'y=\mathbf w^T\boldsymbol\phi(x)',29,model),
                       Arrow(UP*.4,DOWN*.4,buff=.05,color=gold),
                       tex(r'y=y(x,\mathbf w)',29,model)).arrange(DOWN,buff=.2).move_to([2.45,-.35,0])
        self.add(profile,residual,dot,square,Dot(ax.c2p(0,0),color=model),
                 tex(r"(t-y)^2",23,gold).move_to([-.55,.4,0]),
                 jp('赤：予測の中心',19,model).move_to([-3,1.5,0]),
                 jp('独立・共通の精度を固定',20).move_to([.6,-1.77,0]))
        self.beat(actions=[lambda:obs.animate.set_value(.45),
                           lambda:Write(formula),lambda:FadeIn(mapping)])
        self.restore_body(saved)

    def loss_recap(self):
        saved=self.body_card('復習: 4.3 正解クラスの確率と損失')
        ax=self.plot_axes(x=(0,1,.2),y=(0,3,1),width=5.4,height=2.45,
                          center=(-1.5,-.15,0),labels=('y','E_n'))
        p=ValueTracker(.9)
        graph=curve(ax,lambda y:-np.log(y),lo=.05,hi=1,color=RED)
        dot=always_redraw(lambda:Dot(ax.c2p(p.get_value(),-np.log(p.get_value())),color=YELLOW,radius=.09))
        self.add(dot,tex(r't=1:\quad E_n=-\ln y',30,RED).move_to([0,1.4,0]),
                 number('y=',p.get_value,[3,.5,0],RED),
                 number('E_n=',lambda:-np.log(p.get_value()),[3,-.25,0],YELLOW),
                 tex(r'\text{network}\ \longrightarrow\ y=\sigma(a)',27).move_to([0,-1.86,0]))
        self.beat(actions=[lambda:Create(graph),lambda:p.animate.set_value(.1)])
        self.restore_body(saved)

    def taylor_recap(self):
        saved=self.body_card('復習: 4.4 頂上の近くを二次式で近似')
        blue,red,gold='#58B5ED','#FF6B77','#FFE079'
        # Copy just the reviewed 4.4 example, without importing another section.
        logf=lambda z:-.5*(np.asarray(z)/1.1)**2-np.logaddexp(0,-(4*np.asarray(z)+.8))
        grad=lambda z:-z/1.1**2+4*sigmoid(-(4*z+.8))
        prec=lambda z:1/1.1**2+16*sigmoid(4*z+.8)*(1-sigmoid(4*z+.8))
        z0=0.
        for _ in range(30): z0+=grad(z0)/prec(z0)
        ax=self.plot_axes(x=(-.8,1.8,1),y=(-2,2,1),width=5.2,height=2.5,
                          center=(-1.5,-.1,0),labels=('z','g'))
        axis_names=self.mobjects[-1]
        q=ValueTracker(z0);sign=ValueTracker(1)
        f=lambda z:sign.get_value()*logf(z)
        local=lambda z:sign.get_value()*(logf(q.get_value())+grad(q.get_value())*(z-q.get_value())-.5*prec(q.get_value())*(z-q.get_value())**2)
        graph=always_redraw(lambda:curve(ax,f,color=blue))
        approx=always_redraw(lambda:curve(ax,local,lo=q.get_value()-.5,hi=q.get_value()+.5,color=red))
        point=always_redraw(lambda:Dot(ax.c2p(q.get_value(),f(q.get_value())),color=gold))
        tangent=always_redraw(lambda:curve(ax,lambda z:f(q.get_value())+sign.get_value()*grad(q.get_value())*(z-q.get_value()),lo=q.get_value()-.35,hi=q.get_value()+.35,color=gold))
        bracket=always_redraw(lambda:BraceBetweenPoints(ax.c2p(q.get_value()-.5,-1.7),ax.c2p(q.get_value()+.5,-1.7),direction=DOWN,color=red))
        terms=VGroup(tex(r'E(\widehat w)',28,blue),tex(r'+b\delta',30,gold),tex(r'+\frac12H\delta^2',29,red)).arrange(DOWN,buff=.26).move_to([3.15,.05,0])
        label=tex(r'g(z)=\ln f(z)',28,blue).move_to([0,1.47,0])
        self.add(graph,point,label,bracket,jp('基準点の近くだけ',20,red).move_to([-1.5,-1.92,0]))
        def to_error():
            sign.set_value(-1)
            self.remove(label)
            axis_names[0].become(tex('w',25).move_to(axis_names[0]))
            axis_names[1].become(tex('E',25).move_to(axis_names[1]))
            self.add(tex(r'E=-\ln f,\quad \delta=w-\widehat w',27,blue).move_to([0,1.47,0]),tangent,terms)
            return q.animate.set_value(.9)
        self.beat(actions=[lambda:Create(approx),to_error])
        self.restore_body(saved)

    def gradient_aid(self):
        saved=self.body_card('補足: ほかを固定して測る傾き → 勾配')
        blue,yellow,green,purple='#58C4DD','#FFFF00','#83C167','#9A72AC'
        self.add(tex(r'E=w_1^2+\frac12w_2^2,\quad (w_1,w_2)=(1,-1)',25,blue).move_to([0,1.47,0]),
                 jp('説明用の例',18,MUTED).move_to([3.95,1.47,0]))
        ax=self.plot_axes(x=(-2,2.5,1),y=(-2.4,2.4,1),width=2.45,height=2.45,
                          center=(-3.35,-.05,0),labels=('w_1','w_2'))
        contours=VGroup()
        for r in [.5,1,1.5]:
            angles=np.linspace(0,TAU,121)
            contours.add(VMobject().set_points_as_corners([ax.c2p(r*np.cos(t),np.sqrt(2)*r*np.sin(t)) for t in angles]).set_stroke(blue,1.4))
        a=ValueTracker(.8);b=ValueTracker(-1)
        dot=always_redraw(lambda:Dot(ax.c2p(a.get_value(),b.get_value()),color=yellow,radius=.07))
        horizontal=DashedLine(ax.c2p(-.4,-1),ax.c2p(2,-1),color=yellow)
        vertical=DashedLine(ax.c2p(1,-1.8),ax.c2p(1,.8),color=yellow)
        # Sparse axis labels keep the two slices readable at 480p.
        cut1=Axes(x_range=(.6,1.4,.4),y_range=(.6,2.6,1),x_length=2.1,y_length=.9,
                  tips=False,axis_config=dict(color=MUTED,stroke_width=1.2)).move_to([.05,.35,0])
        cut2=Axes(x_range=(-1.4,-.6,.4),y_range=(1,2.2,.6),x_length=2.1,y_length=.9,
                  tips=False,axis_config=dict(color=MUTED,stroke_width=1.2)).move_to([.05,-1.1,0])
        self.add(cut1,cut2,tex(r'E(w_1,-1)',22,blue).move_to([.05,1.05,0]),
                 tex(r'E(1,w_2)',22,blue).move_to([.05,-.4,0]),
                 tex('w_1',22).move_to([1.38,-.15,0]),tex('w_2',22).move_to([1.38,-1.6,0]))
        c1=curve(cut1,lambda w:w*w+.5,color=blue)
        c2=curve(cut2,lambda w:1+.5*w*w,color=blue)
        t1=always_redraw(lambda:curve(cut1,lambda w:a.get_value()**2+.5+2*a.get_value()*(w-a.get_value()),color=yellow))
        t2=always_redraw(lambda:curve(cut2,lambda w:1+.5*b.get_value()**2+b.get_value()*(w-b.get_value()),color=yellow))
        p1=always_redraw(lambda:Dot(cut1.c2p(a.get_value(),a.get_value()**2+.5),color=yellow,radius=.06))
        p2=always_redraw(lambda:Dot(cut2.c2p(b.get_value(),1+.5*b.get_value()**2),color=yellow,radius=.06))
        v1=number(r'\frac{\partial E}{\partial w_1}=',lambda:2*a.get_value(),[2.85,.6,0],yellow,places=1,size=25)
        v2=number(r'\frac{\partial E}{\partial w_2}=',b.get_value,[2.85,-.55,0],yellow,places=1,size=25)
        vector=tex(r'\nabla E=\begin{pmatrix}2\\-1\end{pmatrix}',31,yellow).move_to([3.05,0,0])
        arrow=Arrow(ax.c2p(1,-1),ax.c2p(1.8,-1.4),buff=0,color=yellow,stroke_width=3)
        self.add(contours,dot,horizontal,c1,c2,p1,p2)
        def first():
            self.add(t1,v1)
            return a.animate.set_value(1)
        def second():
            a.set_value(1);b.set_value(-1.2);self.remove(horizontal);self.add(vertical,t2,v2)
            return b.animate.set_value(-1)
        def collect():
            b.set_value(-1);self.remove(vertical)
            return AnimationGroup(ReplacementTransform(VGroup(v1,v2),vector),GrowArrow(arrow))
        self.beat(actions=[first,second,collect])
        self.restore_body(saved)
