"""PRML 4.3: linked visual experiments in Manim Community."""
import json
from pathlib import Path
import numpy as np
from manim import *
from discriminative_model import *
from scene_support import *

AID_INPUT='#58C4DD'
AID_OPERATION='#FFFF00'
AID_RESULT='#83C167'
AID_COMPARE='#9A72AC'

class PRML43ProbabilisticDiscriminativeModels(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.entries={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.probability,self.features,self.cross_entropy,self.separation,
                                   self.irls,self.multiclass,self.probit,self.outliers,self.canonical]):
            self.begin(i); method()
            assert self.bi==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        p=Path(config.media_dir)/'prml43_timeline.json'
        p.write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def probability(self):
        ax=self.plot_axes(labels=('x','p(C_1|x)'))
        k=ValueTracker(1.2); b=ValueTracker(0); q=ValueTracker(-2.4)
        pred=lambda x:sigmoid(k.get_value()*x+b.get_value())
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.065,color=RED_CLASS if t else BLUE_CLASS) for x,t in zip(X,T)])
        probe=always_redraw(lambda:Dot(ax.c2p(q.get_value(),.5),radius=.1,color=WHITE))
        self.add(dots,probe,note('赤：クラス1　　青：クラス2　　白：未知の入力'))
        self.beat(q.animate.set_value(.1))
        self.remove(probe)
        graph=always_redraw(lambda:curve(ax,pred))
        probe=always_redraw(lambda:Dot(ax.c2p(q.get_value(),pred(q.get_value())),radius=.1,color=WHITE))
        guide=always_redraw(lambda:DashedLine(ax.c2p(q.get_value(),0),probe.get_center(),color=MUTED))
        self.add(graph,probe,guide,number('p=',lambda:pred(q.get_value()),[5.35,.6,0],YELLOW_ACC))
        self.beat(q.animate.set_value(2.4))
        self.equation(r'a=w_0+w_1x',r'\qquad \sigma(a)=\frac{1}{1+e^{-a}}')
        self.beat(q.animate.set_value(-1.8))
        sliders=VGroup(knob('w_1=',k,.5,4,at=(-2,-2.5,0)),knob('w_0=',b,-2,2,at=(-2,-2.5,0)))
        # Place sliders below the graph, using only one at a time.
        self.remove(*[m for m in self.mobjects if isinstance(m,Text) and m.get_center()[1]<-2])
        self.add(sliders[0])
        def bias_action():
            self.remove(*sliders[0].get_family());self.add(sliders[1])
            return b.animate.set_value(1.7)
        self.beat(actions=[lambda:k.animate.set_value(3.5),bias_action])
        route=VGroup(tex(r'p(x|C_k),\ p(C_k)\ \longrightarrow\ p(C_k|x)',25),
                     tex(r'w\ \longrightarrow\ p(C_k|x)',29,YELLOW_ACC)).arrange(DOWN,buff=.12).move_to([0,-2.4,0])
        self.remove(*sliders[1].get_family())
        self.add(route)
        self.beat(Indicate(route,color=YELLOW_ACC,scale_factor=1.03))
        f=self.equation(r'p(C_1|\phi)=',r'\sigma(w^T\phi)',r'\qquad p(C_2|\phi)=1-y')
        f[1].set_color(YELLOW_ACC)
        self.beat(q.animate.set_value(.6))

    def features(self):
        ax=Axes(x_range=(-1.5,1.5,.75),y_range=(-1.5,1.5,.75),x_length=4,y_length=4,tips=False,
                axis_config=dict(color=MUTED,include_numbers=False,font_size=19)).move_to([-1.6,-.25,0])
        fa=Axes(x_range=(0,1.8,.6),y_range=(0,1.8,.6),x_length=4,y_length=4,tips=False,
                axis_config=dict(color=MUTED,include_numbers=False,font_size=19)).move_to([-1.6,-.25,0])
        names=VGroup(tex('x_1',25).next_to(ax.x_axis,RIGHT),tex('x_2',25).next_to(ax.y_axis,UP))
        fnames=VGroup(tex(r'\phi_1',25).next_to(fa.x_axis,RIGHT),tex(r'\phi_2',25).next_to(fa.y_axis,UP))
        stage=ax.copy();alpha=ValueTracker(0);angle=ValueTracker(0)
        def pos(p):
            return (1-alpha.get_value())*ax.c2p(*p)+alpha.get_value()*fa.c2p(*(p**2))
        points=VGroup(*[Dot(radius=.065,color=RED_CLASS if t else BLUE_CLASS) for t in RING_T])
        points.add_updater(lambda m:[d.move_to(pos(p)) for d,p in zip(m,RING)])
        line=always_redraw(lambda:Line(ax.c2p(-1.4*np.cos(angle.get_value()),-1.4*np.sin(angle.get_value())),ax.c2p(1.4*np.cos(angle.get_value()),1.4*np.sin(angle.get_value())),color=WHITE))
        self.add(stage,names,points,line,note('自作データ：各18点',at=(3.5,-.65,0)))
        self.beat(angle.animate.set_value(PI*.8))
        self.remove(line)
        self.equation(r'(x_1,x_2)\ \longrightarrow\ (\phi_1,\phi_2)=(x_1^2,x_2^2)')
        self.add(note('点の色と対応を保つ',at=(3.5,.15,0),color=YELLOW_ACC))
        self.remove(names)
        self.beat(alpha.animate.set_value(1),Transform(stage,fa))
        self.add(fnames)
        boundary=Line(fa.c2p(0,1),fa.c2p(1,0),color=YELLOW_ACC,stroke_width=4)
        self.equation(r'a=1-\phi_1-\phi_2',r'\qquad a=0')
        self.beat(Create(boundary))
        self.remove(boundary)
        theta=np.linspace(0,2*np.pi,161);circle=np.column_stack([np.cos(theta),np.sin(theta)])
        edge=always_redraw(lambda:VMobject().set_points_as_corners([pos(p) for p in circle]).set_stroke(YELLOW_ACC,4))
        self.add(edge);self.equation(r'\phi_1+\phi_2=1',r'\quad\Longleftrightarrow\quad x_1^2+x_2^2=1')
        oldnames=VGroup(tex('x_1',25).next_to(ax.x_axis,RIGHT),tex('x_2',25).next_to(ax.y_axis,UP))
        self.remove(fnames)
        recap=self.recap_label('復習: 3.1 固定基底関数')
        self.add(recap)
        self.beat(alpha.animate.set_value(0),Transform(stage,ax))
        self.remove(recap)
        self.add(oldnames)
        coincident=VGroup(Dot(ax.c2p(.25,.2),radius=.13,color=RED_CLASS),Dot(ax.c2p(.25,.2),radius=.075,color=BLUE_CLASS))
        self.add(coincident)
        self.beat(Indicate(coincident,color=WHITE,scale_factor=1.7))
        self.equation(r'\phi=(1,x_1^2,x_2^2)^T',r'\qquad w=(1,-1,-1)^T')
        self.beat(Indicate(edge.copy(),color=YELLOW_ACC,scale_factor=1.05))

    def cross_entropy(self):
        ax=self.plot_axes(x=(.02,.98,.2),y=(0,4,1),labels=('y','E_n'),center=(-.5,-.2,0))
        p=ValueTracker(.9); target=ValueTracker(1)
        cost=lambda x:-target.get_value()*np.log(x)-(1-target.get_value())*np.log(1-x)
        graph=always_redraw(lambda:curve(ax,cost,color=RED_CLASS if target.get_value()>.5 else BLUE_CLASS))
        dot=always_redraw(lambda:Dot(ax.c2p(p.get_value(),cost(p.get_value())),color=YELLOW_ACC,radius=.095))
        self.add(graph,dot,number('E_n=',lambda:cost(p.get_value()),[5.3,.5,0],YELLOW_ACC),knob('y=',p,.02,.98))
        self.equation(r't=1:\quad E_n=-\ln y',size=36)
        self.beat(p.animate.set_value(.1))
        self.cost_recap()
        self.beat(p.animate.set_value(.025))
        self.equation(r't=0:\quad E_n=-\ln(1-y)',size=36)
        target.set_value(0)
        self.beat(p.animate.set_value(.85))
        self.equation(r'p(t|w)=\prod_n y_n^{t_n}(1-y_n)^{1-t_n}',size=29)
        def show_cross_entropy():
            self.equation(r'E=-\sum_n[t_n\ln y_n+(1-t_n)\ln(1-y_n)]',size=31)
            return Indicate(self.formula,color=YELLOW_ACC,scale_factor=1.02)
        self.beat(actions=[lambda:Indicate(self.formula,color=YELLOW_ACC),show_cross_entropy])
        self.remove(graph,dot)
        # Replace the loss graph by prediction-to-target arrows on the same stage.
        self.remove(ax,*[m for m in self.mobjects if m is not self.formula and m is not self.subtitle and m.get_center()[1]<2.2])
        ax=self.plot_axes(labels=('x','y'),height=3,center=(-.5,-.1,0))
        shift=ValueTracker(-1.2)
        g=always_redraw(lambda:curve(ax,lambda x:sigmoid(x+shift.get_value())))
        residual=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,sigmoid(x+shift.get_value())),ax.c2p(x,t),color=RED_CLASS if t else BLUE_CLASS,stroke_width=2) for x,t in zip(X,T)]))
        self.add(g,residual)
        f=self.equation(r'\nabla E=',r'\sum_n',r'(y_n-t_n)',r'\phi_n'); f[2].set_color(YELLOW_ACC); f[3].set_color(GREEN_CLASS)
        self.beat(shift.animate.set_value(.35))
        chain=tex(r'\frac{y-t}{y(1-y)}\;\cdot\; y(1-y)\;\cdot\;\phi=(y-t)\phi',31).move_to([0,-2.35,0])
        self.add(chain)
        self.beat(Indicate(chain,color=YELLOW_ACC,scale_factor=1.04))

        self.chain_aid()

    def separation(self):
        ax=self.plot_axes(labels=('x','y'))
        k=ValueTracker(1); boundary=ValueTracker(0)
        g=always_redraw(lambda:curve(ax,lambda x:sigmoid(k.get_value()*(x-boundary.get_value()))))
        slider=knob('w_1=',k,1,20)
        self.add(g,VGroup(*[Dot(ax.c2p(x,t),radius=.075,color=RED_CLASS if t else BLUE_CLASS) for x,t in zip(X,SEP_T)]),slider)
        loss_num=number('E=',lambda:loss(np.array([-k.get_value()*boundary.get_value(),k.get_value()]),SEP_T),[5.1,.6,0],YELLOW_ACC)
        self.add(loss_num);self.equation(r'y=\sigma(w_1x)',r'\qquad w_1\uparrow')
        self.beat(k.animate.set_value(5))
        self.beat(k.animate.set_value(20))
        self.equation(r'\|w\|\to\infty',r'\qquad E\to0',r'\quad\text{(separable)}')
        self.beat(boundary.animate.set_value(.12))
        self.remove(loss_num,slider)
        lam=ValueTracker(.015)
        # Numerical optimization at each value, not a hand-drawn flattening.
        g.clear_updaters(); self.remove(g)
        rg=always_redraw(lambda:curve(ax,lambda x:sigmoid(fit(SEP_T,lam.get_value(),steps=12)[-1][0]+fit(SEP_T,lam.get_value(),steps=12)[-1][1]*x)))
        self.add(rg,number(r'\lambda=',lam.get_value,[5.1,.6,0],PURPLE_ACC))
        f=self.equation(r'E_{\rm reg}=E+',r'\frac{\lambda}{2}\|w\|^2');f[1].set_color(PURPLE_ACC)
        recap=self.recap_label('復習: 3.3 ガウス事前分布').shift(UP*.45)
        self.add(recap)
        self.beat(lam.animate.set_value(1.5))
        self.remove(recap)
        self.add(note('訓練ラベルの正解率 ≠ 未知の点での確率の正しさ'))
        self.beat(lam.animate.set_value(.3))

    def irls(self):
        ax=self.plot_axes(x=(-4,4,2),labels=('a','y'),height=3.2)
        a=ValueTracker(-1); frac=ValueTracker(0)
        y=lambda:float(sigmoid(a.get_value()))
        z=lambda:a.get_value()+(1-y())/(y()*(1-y()))
        graph=curve(ax,sigmoid)
        tangent=always_redraw(lambda:Line(ax.c2p(a.get_value(),y()),ax.c2p(a.get_value()+frac.get_value()*(z()-a.get_value()),y()+frac.get_value()*(1-y())),color=PURPLE_ACC,stroke_width=4))
        contact=always_redraw(lambda:Dot(ax.c2p(a.get_value(),y()),color=WHITE))
        tangent_note=note('接線の先にある、目標の高さ y=1')
        self.add(graph,tangent,contact,tangent_note)
        self.equation(r'y=\sigma(a)',r'\qquad t=1')
        self.beat(frac.animate.set_value(1))
        self.equation(r'z_n=a_n+',r'\frac{t_n-y_n}{y_n(1-y_n)}');self.formula[1].set_color(PURPLE_ACC)
        zlabel=number('z=',z,[5.2,.6,0],PURPLE_ACC)
        self.add(zlabel)
        self.beat(a.animate.set_value(.3))
        self.irls_recap()
        self.remove(graph,tangent,contact,tangent_note,zlabel)
        self.add(note('分散は y=0.5 のとき最大：0.25'))
        rgraph=curve(ax,lambda x:sigmoid(x)*(1-sigmoid(x)),color=GREEN_CLASS)
        self.equation(r'R_{nn}=',r'y_n(1-y_n)',r'=\mathrm{var}[t_n]');self.formula[1].set_color(GREEN_CLASS)
        self.add(rgraph)
        self.beat(Create(rgraph))
        self.remove(*[m for m in self.mobjects if m is not self.formula and m is not self.subtitle and m.get_center()[1]<2.2])
        ax=self.plot_axes(labels=('x','y'),height=3.2)
        step=ValueTracker(0)
        w=lambda:interpolated_weight(step.get_value())
        g=always_redraw(lambda:curve(ax,lambda x:sigmoid(w()[0]+w()[1]*x)))
        dots=VGroup(*[Dot(radius=.08,color=RED_CLASS if t else BLUE_CLASS) for t in T])
        def update_dots(m):
            prob=sigmoid(PHI@w());r=prob*(1-prob)
            for dot,x,t,rv in zip(m,X,T,r):
                dot.set_width(.38*np.sqrt(rv/.25)).move_to(ax.c2p(x,t))
        dots.add_updater(update_dots)
        self.add(g,dots,number('E=',lambda:loss(w()),[5.15,.5,0],YELLOW_ACC),note('円の面積：分散 R　／　整数位置が実際の反復'))
        self.equation(r'w_{\rm new}=(\Phi^TR\Phi)^{-1}\Phi^TRz',size=34)
        self.beat(step.animate.set_value(4))
        self.equation(r'w_{\rm new}=w-H^{-1}\nabla E',r'\qquad H=\Phi^TR\Phi',size=31)
        self.beat(step.animate.set_value(7))
        self.newton_aid()
        self.equation(r'u^THu=\sum_n R_{nn}(\phi_n^Tu)^2\geq0',size=34)
        self.remove(*[m for m in self.mobjects if isinstance(m,Text) and m.get_center()[1]<-2])
        self.add(note('凸：局所的な谷底が、全体でも谷底になる形'))
        self.beat(Indicate(dots.copy(),color=GREEN_CLASS,scale_factor=1.04))

    def multiclass(self):
        a=ValueTracker(-1); common=ValueTracker(0)
        logits=lambda:np.array([a.get_value(),.5,0])+common.get_value()
        p=lambda:softmax(logits())
        colors=[RED_CLASS,BLUE_CLASS,GREEN_CLASS]
        baseline=-1.5
        bars=VGroup(*[Rectangle(width=1.3,height=.1,fill_color=c,fill_opacity=.85,stroke_width=0) for c in colors])
        for i,bar in enumerate(bars):
            bar.add_updater(lambda m,i=i:m.stretch_to_fit_height(3*p()[i]).move_to([-3+i*3,baseline,0],aligned_edge=DOWN))
        self.add(bars)
        for i,c in enumerate(colors):
            self.add(tex(f'C_{i+1}',29,c).move_to([-3+i*3,-1.88,0]),number(f'a_{i+1}=',lambda i=i:logits()[i],[-3+i*3,1.85,0],c))
            value=number('p=',lambda i=i:p()[i],[-3+i*3,0,0],WHITE)
            value.add_updater(lambda m,i=i:m.next_to(bars[i],UP,buff=.16))
            self.add(value)
        self.equation(r'a_k=w_k^T\phi',size=37)
        control=knob('a_1=',a,-1,3)
        self.add(control)
        self.beat(a.animate.set_value(1))
        self.equation(r'y_k=\frac{e^{a_k}}{\sum_j e^{a_j}}',r'\qquad \sum_k y_k=1',size=36)
        self.beat(a.animate.set_value(3))
        self.equation(r'\operatorname{softmax}(a+c\mathbf1)=\operatorname{softmax}(a)',size=31)
        self.remove(*control.get_family())
        self.beat(common.animate.set_value(5))
        common.set_value(0)
        self.add(control)
        self.equation(r't=(0,1,0)',r'\qquad E_n=-\ln y_2',size=36)
        self.beat(a.animate.set_value(-1))
        self.equation(r'\nabla_{w_j}E=\sum_n',r'(y_{nj}-t_{nj})',r'\phi_n');self.formula[1].set_color(YELLOW_ACC)
        self.beat(a.animate.set_value(.8))

    def probit(self):
        ax=self.plot_axes(x=(-3.5,3.5,1),y=(0,1,.5),labels=(r'a,\theta',r'p(\theta),\ F(a)'))
        a=ValueTracker(0);theta=ValueTracker(-1.5)
        pdf=curve(ax,normal_pdf,color=BLUE_CLASS)
        threshold=always_redraw(lambda:Dot(ax.c2p(theta.get_value(),0),radius=.11,color=RED_CLASS if theta.get_value()<=a.get_value() else BLUE_CLASS))
        limit=always_redraw(lambda:DashedLine(ax.c2p(a.get_value(),0),ax.c2p(a.get_value(),1),color=YELLOW_ACC))
        self.equation(r't=1\quad\Longleftrightarrow\quad \theta\leq a')
        self.add(pdf,threshold,limit,note('青：しきい値の密度　　黄：入力スコア'))
        self.beat(theta.animate.set_value(1.5))
        def area():
            u=np.linspace(-3.5,a.get_value(),100)
            return Polygon(ax.c2p(-3.5,0),*[ax.c2p(x,normal_pdf(x)) for x in u],ax.c2p(a.get_value(),0),stroke_width=0,fill_color=BLUE_CLASS,fill_opacity=.45)
        region=always_redraw(area);self.add(region)
        self.equation(r'P(t=1|a)=',r'\int_{-\infty}^{a}p(\theta)\,d\theta',size=35);self.formula[1].set_color(BLUE_CLASS)
        self.beat(a.animate.set_value(-1.5))
        cdf=curve(ax,normal_cdf,color=RED_CLASS)
        moving=always_redraw(lambda:Dot(ax.c2p(a.get_value(),normal_cdf(a.get_value())),color=RED_CLASS,radius=.09))
        self.add(cdf,moving,number('F(a)=',lambda:normal_cdf(a.get_value()),[5.15,.6,0],RED_CLASS))
        self.beat(a.animate.set_value(1.6))
        self.equation(r'\Phi(a)=\int_{-\infty}^a\mathcal N(\theta|0,1)\,d\theta',size=34)
        self.beat(a.animate.set_value(0))
        scale=ValueTracker(1)
        logistic=always_redraw(lambda:curve(ax,lambda x:sigmoid(scale.get_value()*x),color=YELLOW_ACC))
        self.add(logistic);self.equation(r'\Phi(a)\quad\text{vs.}\quad\sigma(1.7a)',size=34)
        self.beat(scale.animate.set_value(1.7))

    def outliers(self):
        ax=self.plot_axes(x=(0,5,1),y=(0,16,4),labels=('-a','E_n'),height=3.5)
        distance=ValueTracker(.2)
        logloss=lambda x:np.logaddexp(0,1.7*x)
        probitloss=lambda x:-np.log(normal_cdf(-x))
        dot=always_redraw(lambda:Dot(ax.c2p(distance.get_value(),logloss(distance.get_value())),color=RED_CLASS,radius=.12))
        self.add(curve(ax,logloss,color=YELLOW_ACC),dot,note('記録ラベル t=1　／　点数 a<0：赤に不利'))
        self.equation(r'E_n=-\ln P(t=1|a)',size=36)
        self.beat(distance.animate.set_value(3))
        self.add(curve(ax,probitloss,color=PURPLE_ACC),note('黄：logistic（1.7倍）　紫：probit',at=(0,1.7,0)))
        self.beat(distance.animate.set_value(5))
        self.remove(*[m for m in self.mobjects if m is not self.formula and m is not self.subtitle and m.get_center()[1]<2.2])
        ax=self.plot_axes(x=(-6,6,3),labels=('a','P(t_{obs}=1|a)'))
        eps=ValueTracker(0)
        g=always_redraw(lambda:curve(ax,lambda x:noisy_probability(x,eps.get_value())))
        self.add(g,knob(r'\epsilon=',eps,0,.3,color=PURPLE_ACC))
        self.equation(r'P(t_{\rm obs}=1|a)=(1-\epsilon)\sigma(a)+\epsilon[1-\sigma(a)]',size=30)
        self.beat(eps.animate.set_value(.04))
        self.equation(r'P(t_{\rm obs}=1|a)=',r'\epsilon+(1-2\epsilon)\sigma(a)',size=34);self.formula[1].set_color(PURPLE_ACC)
        bounds=always_redraw(lambda:VGroup(*[DashedLine(ax.c2p(-6,y),ax.c2p(6,y),color=PURPLE_ACC) for y in [eps.get_value(),1-eps.get_value()]]))
        self.add(bounds)
        self.beat(actions=[lambda:eps.animate.set_value(.1),lambda:Indicate(bounds.copy(),color=PURPLE_ACC,scale_factor=1.01)])
        self.beat(eps.animate.set_value(.22))

    def canonical(self):
        forms=[r'\nabla E_{\rm Gaussian}=\frac{1}{s}\sum_n(y_n-t_n)\phi_n',r'\nabla E_{\rm Bernoulli}=\sum_n(y_n-t_n)\phi_n',r'\nabla_{w_j}E_{\rm softmax}=\sum_n(y_{nj}-t_{nj})\phi_n']
        equations=VGroup(*[tex(f,32,c) for f,c in zip(forms,[BLUE_CLASS,YELLOW_ACC,GREEN_CLASS])]).arrange(DOWN,buff=.55).move_to([0,.2,0])
        self.add(equations)
        self.beat(LaggedStart(*[Indicate(m,scale_factor=1.05) for m in equations],lag_ratio=.3))
        self.link_recap()
        self.remove(*equations.get_family())
        self.equation(r'p(t|\eta,s)=\frac1s h(t/s)g(\eta)e^{\eta t/s}',size=34)
        mean=tex(r'y=\mathbb E[t|\eta]',38,RED_CLASS).move_to([-3,.3,0]);natural=tex(r'\eta=\psi(y)',38,GREEN_CLASS).move_to([3,.3,0])
        arrows=VGroup(Arrow([-1.2,.55,0],[1.2,.55,0],color=GREEN_CLASS),Arrow([1.2,-.15,0],[-1.2,-.15,0],color=RED_CLASS))
        self.add(mean,natural,arrows,note('平均と自然パラメータの対応'))
        self.beat(Indicate(arrows,color=YELLOW_ACC))
        self.equation(r'y=\sigma(a)',r'\quad\Longleftrightarrow\quad',r'a=\ln\frac{y}{1-y}',size=37)
        self.beat(actions=[lambda:Indicate(arrows[1],color=RED_CLASS),lambda:Indicate(arrows[0],color=GREEN_CLASS)])
        self.equation(r'\eta=a=w^T\phi',r'\qquad f^{-1}=\psi',size=37)
        cancel=tex(r"\psi'(y)\,f'(a)=1",40,YELLOW_ACC).move_to([0,-1.4,0]);self.add(cancel)
        self.beat(Indicate(cancel,scale_factor=1.08))
        self.equation(r'\nabla E=',r'\frac1s',r'\sum_n(y_n-t_n)\phi_n',size=38);self.formula[1].set_color(PURPLE_ACC)
        self.beat(Indicate(self.formula[1],scale_factor=1.15,color=PURPLE_ACC))
        self.remove(mean,natural,arrows,cancel,*[m for m in self.mobjects if isinstance(m,Text) and m.get_center()[1]<0])
        ax=self.plot_axes(labels=('x','y'),height=3.1)
        shift=ValueTracker(-1)
        g=always_redraw(lambda:curve(ax,lambda x:sigmoid(1.5*x+shift.get_value())))
        self.add(g,note('特徴 → スコア → 確率 → ラベルへの損失'))
        self.beat(shift.animate.set_value(.5))

    def cost_recap(self):
        saved,frame,heading=self.body_card('復習: 1.6 負の対数のコスト')
        # 1.6 learning(): orange model cost, yellow position, blue observation 1.
        orange='#FFB45B'; yellow='#FFE079'; blue='#58B5ED'
        ax=Axes(x_range=(0,1,.2),y_range=(0,3,1),x_length=5.2,y_length=2.5,
                tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=20)).move_to([-1.9,.1,0])
        theta=ValueTracker(.9)
        graph=curve(ax,lambda x:-np.log(x),.06,.98,orange)
        dot=always_redraw(lambda:Dot(ax.c2p(theta.get_value(),-np.log(theta.get_value())),color=yellow,radius=.08))
        symbol=tex(r'\theta',27,yellow).next_to(ax.x_axis,RIGHT)
        formula=tex(r'\bar L=-\ln\theta',32,orange).move_to([-1.8,-1.65,0])
        self.add(ax,graph,dot,symbol,formula,
                 jp('説明用の例：観測は 1 が一つ',21).move_to([0,1.45,0]),
                 number('p=',theta.get_value,[3.45,.6,0],yellow),
                 number(r'-\ln p=',lambda:-np.log(theta.get_value()),[3.45,-.1,0],orange))
        bridge=VGroup(Dot([-0.95,0,0],color=blue),Arrow([-.6,0,0],[.6,0,0],buff=0,color=WHITE),Dot([.95,0,0],color=RED_CLASS)).move_to([3.3,-.95,0])
        bridge.add(jp('観測 1 → 赤の正解',20).move_to([3.3,-1.55,0]))
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('R1.6 lower correct probability',a,lambda:theta.animate.set_value(.1)),
            ('R1.6 transfer cost to classification',b,lambda:AnimationGroup(
                Transform(symbol,tex('y',27,yellow).move_to(symbol)),
                Transform(formula,tex(r'E_n=-\ln y',32,orange).move_to(formula)),FadeIn(bridge))),
        ])
        self.restore_body(saved)

    def chain_aid(self):
        saved,frame,heading=self.body_card('補足: 小さな変化を、倍率でつなぐ')
        self.add(jp('説明用の例：t = 1',20,MUTED).move_to([3.8,1.5,0]))
        boxes=VGroup(*[VGroup(RoundedRectangle(width=1.9,height=.85,corner_radius=.08,color=c),
                     tex(label,34,c)).move_to([x,.4,0]) for x,label,c in
                     [(-3.7,'a=0',AID_INPUT),(0,'y=0.5',AID_OPERATION),(3.7,r'E=-\ln y',AID_RESULT)]])
        arrows=VGroup(Arrow([-2.7,.4,0],[-1,.4,0],buff=.07,color=AID_OPERATION),
                      Arrow([1,.4,0],[2.7,.4,0],buff=.07,color=AID_OPERATION))
        factors=VGroup(tex(r'\frac{dy}{da}=\frac14',30,AID_OPERATION).move_to([-1.85,1.15,0]),
                       tex(r'\frac{dE}{dy}=-2',30,AID_OPERATION).move_to([1.85,1.15,0]))
        deltas=VGroup(*[tex(label,29,c).move_to([x,-.5,0]) for x,label,c in
                      [(-3.7,r'\Delta a=0.04',AID_INPUT),(0,r'\Delta y\approx0.01',AID_OPERATION),
                       (3.7,r'\Delta E\approx-0.02',AID_RESULT)]])
        product=tex(r'\frac{dE}{da}=\frac14\times(-2)=-\frac12',35,AID_RESULT).move_to([0,-1.55,0])
        identity=tex(r'\frac{dE}{da}=-\frac12=0.5-1=y-t',35,AID_RESULT).move_to([0,-1.55,0])
        def show_difference():
            self.remove(product)
            self.add(identity)
            return Create(Line([-2.8,-2.,0],[2.8,-2.,0],color=AID_OPERATION,stroke_width=2))
        self.add(boxes,arrows,factors)
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('V14a small input change',a*.25,lambda:FadeIn(deltas[0])),
            ('V14a propagate to probability',a*.35,lambda:FadeIn(deltas[1],shift=RIGHT*.2)),
            ('V14a propagate to error',a*.4,lambda:FadeIn(deltas[2],shift=RIGHT*.2)),
            ('V14a multiply local derivatives',b,lambda:FadeIn(product,rate_func=lambda t:min(1,4*t))),
            ('V14a identify prediction minus target',c,show_difference),
        ])
        self.restore_body(saved)

    def irls_recap(self):
        saved,frame,heading=self.body_card('復習: 2.1 ベルヌーイの分散')
        mu=ValueTracker(.2)
        line=NumberLine(x_range=[0,1,.5],length=6,include_numbers=True,font_size=24).move_to([0,-.25,0])
        bars=always_redraw(lambda:VGroup(*[Rectangle(width=.65,height=2*p,color=c,fill_opacity=.75,stroke_width=0)
            .move_to(line.n2p(x),aligned_edge=DOWN) for x,p,c in [(0,1-mu.get_value(),'#58B5ED'),(1,mu.get_value(),'#77D49A')]]))
        pivot=always_redraw(lambda:Triangle(color='#FFE079',fill_opacity=1).scale(.13).next_to(line.n2p(mu.get_value()),DOWN,buff=.1))
        variance=number(r'\mu(1-\mu)=',lambda:mu.get_value()*(1-mu.get_value()),[0,-1.35,0],'#FFE079')
        old=VGroup(line,bars,pivot,variance,tex(r'\mu=y',28,'#FFE079').move_to([3.9,.9,0]))
        self.add(old)
        ax=Axes(x_range=(0,3,1),y_range=(0,2,1),x_length=4,y_length=2.4,tips=False,
                axis_config=dict(color=MUTED)).move_to([-2.65,.05,0])
        xs=np.array([.5,1.5,2.5]); targets=np.array([1.5,.3,1.7]); pred=np.array([.7,1.,1.3]);res=targets-pred
        model=Line(ax.c2p(0,.55),ax.c2p(3,1.45),color='#FF6B77')
        dots=VGroup(*[Dot(ax.c2p(x,t),color='#58B5ED',radius=.07) for x,t in zip(xs,targets)])
        residuals=VGroup(*[Line(ax.c2p(x,p),ax.c2p(x,t),color='#FFE079',stroke_width=4) for x,t,p in zip(xs,targets,pred)])
        squares=VGroup(*[Square(side_length=abs(r)*1.25,color='#FFE079',fill_opacity=.55).move_to([1.2+i*1.5,.4,0]) for i,r in enumerate(res)])
        eq=tex(r'\sum_n(t_n-y_n)^2',32,'#FFE079').move_to([0,-1.65,0])
        explanation=jp('説明用の3点・面積がずれの二乗',20,MUTED).move_to([0,1.45,0])
        labels=VGroup(tex('t_n',25,'#58B5ED').move_to([-5.1,1.,0]),tex('y_n',25,'#FF6B77').move_to([-5.1,-.5,0]))
        weights=np.array([.09,.25,.16])
        weighted=VGroup(*[sq.copy().scale(np.sqrt(r)).set_color(GREEN_CLASS) for sq,r in zip(squares,weights)])
        weightlabels=VGroup(*[tex(f'R_{{{i+1},{i+1}}}={r:.2f}',23,GREEN_CLASS).move_to([1.2+i*1.5,-.55,0]) for i,r in enumerate(weights)])
        def least_squares():
            self.remove(*old.get_family())
            heading.become(jp('復習: 3.1 残差の二乗から最小二乗へ',25).move_to([-5.35,2.05,0],aligned_edge=LEFT))
            self.add(ax,model,dots,residuals,explanation,labels)
            return AnimationGroup(TransformFromCopy(residuals,squares),FadeIn(eq))
        def weighted_squares():
            explanation.become(jp('今回：有効目標へのずれに、分散の重み',21,MUTED).move_to([0,1.45,0]))
            labels.become(VGroup(tex('z_n',25,'#58B5ED').move_to([-5.1,1.,0]),
                tex(r'\phi_n^Tw',23,'#FF6B77').move_to([-5.1,-.5,0])))
            eq.become(tex(r'\sum_n R_{nn}(z_n-\phi_n^Tw)^2',32,GREEN_CLASS).move_to([0,-1.65,0]))
            return AnimationGroup(Transform(squares,weighted),FadeIn(weightlabels,rate_func=lambda t:min(1,4*t)))

        a,b,c=[self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('R2.1 Bernoulli variance',a,lambda:mu.animate.set_value(.5)),
            ('R3.1 residual squared areas',b,least_squares),
            ('R3.1 effective targets and variance weights',c*.7,weighted_squares),
            ('R3.1 recompute on next iteration',c*.3,lambda:FadeIn(jp('更新のたびに R と z を計算し直す',20).move_to([0,-2.15,0]))),
        ])
        self.restore_body(saved)

    def newton_aid(self):
        saved,frame,heading=self.body_card('補足: 近くを放物線で近似する')
        # At w=0: E=1, g=2, H=4. This illustrative quartic is convex.
        ax=Axes(x_range=(-1.2,.4,.4),y_range=(0,2.3,1),x_length=5.3,y_length=2.6,
                tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=20)).move_to([-1.8,-.05,0])
        error=lambda w:1+2*w+2*w*w+.5*w**4
        quadratic=lambda w:1+2*w+2*w*w
        actual=curve(ax,error,-1.15,.38,AID_INPUT)
        local=curve(ax,quadratic,-.9,.35,AID_OPERATION)
        point=Dot(ax.c2p(0,1),color=AID_INPUT,radius=.085)
        tangent=Line(ax.c2p(-.18,.64),ax.c2p(.18,1.36),color=AID_COMPARE)
        arrow=Arrow(ax.c2p(0,.16),ax.c2p(-.5,.16),buff=0,color=AID_COMPARE)
        step=tex(r'\Delta w=-\frac{g}{H}=-\frac24=-0.5',30,AID_RESULT).move_to([2.75,-.8,0])
        self.add(ax,actual,point,tex('w',24).next_to(ax.x_axis,RIGHT),
                 jp('説明用の一変数・曲率 H > 0',20,MUTED).move_to([0,1.45,0]),
                 jp('青：元の誤差',21,AID_INPUT).move_to([3.15,.85,0]),
                 jp('黄：局所近似',21,AID_OPERATION).move_to([3.15,.35,0]))
        vals=tex(r'g=2,\quad H=4',30,AID_OPERATION).move_to([2.8,-.2,0])
        note=jp('近似した谷底へ一歩 → 計算し直す',22).move_to([0,-1.9,0])
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('V11b local quadratic approximation',a*.65,lambda:Create(local)),
            ('V11b show approximating minimum',a*.35,lambda:AnimationGroup(FadeIn(vals),FadeIn(tangent),FadeIn(note))),
            ('V11b gradient divided by curvature',b,lambda:AnimationGroup(GrowArrow(arrow),FadeIn(step),point.animate.move_to(ax.c2p(-.5,quadratic(-.5))))),
        ])
        self.restore_body(saved)

    def link_recap(self):
        saved,frame,heading=self.body_card('復習: 2.4 自然パラメータと平均')
        # 2.4 likelihood(): green sigmoid, yellow tracking point and eta slider.
        eta=ValueTracker(-1.5)
        ax=Axes(x_range=(-3,3,1),y_range=(0,1,.5),x_length=5,y_length=2.4,
                tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=20)).move_to([-2,-.2,0])
        graph=curve(ax,sigmoid,color='#77D49A')
        dot=always_redraw(lambda:Dot(ax.c2p(eta.get_value(),sigmoid(eta.get_value())),color='#FFE079',radius=.08))
        line=NumberLine(x_range=[-3,3,1],length=3.8,include_numbers=False,color=MUTED).move_to([-2.6,-1.85,0])
        slider=VGroup(line,always_redraw(lambda:Dot(line.n2p(eta.get_value()),radius=.07,color='#FFE079')))
        self.add(ax,graph,dot,slider,tex(r'\eta',25,'#FFE079').next_to(line,LEFT),
                 tex(r'\eta',25).next_to(ax.x_axis,RIGHT),tex(r'\mu',25).next_to(ax.y_axis,UP),
                 number(r'\mu=',lambda:sigmoid(eta.get_value()),[3.4,.95,0],'#FFE079'),
                 jp('ベルヌーイの例',20,MUTED).move_to([2.95,1.5,0]))
        forward=VGroup(tex(r'\eta\ \longrightarrow\ \mu',31,'#77D49A'),jp('活性化関数',22)).arrange(DOWN,buff=.12).move_to([3.2,.2,0])
        backward=VGroup(tex(r'\eta\ \longleftarrow\ \mu',31,'#77D49A'),jp('リンク関数',22)).arrange(DOWN,buff=.12).move_to([3.2,-.9,0])
        bridge=tex(r'\eta=a=w^T\phi,\quad\mu=y',28,YELLOW_ACC).move_to([0,-2.2,0])
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('R2.4 natural parameter controls mean',a,lambda:eta.animate.set_value(1.5)),
            ('R2.4 forward and inverse mapping',b*.65,lambda:LaggedStart(FadeIn(forward),FadeIn(backward),lag_ratio=.45)),
            ('R2.4 link to current variables',b*.35,lambda:FadeIn(bridge)),
        ])
        self.restore_body(saved)
