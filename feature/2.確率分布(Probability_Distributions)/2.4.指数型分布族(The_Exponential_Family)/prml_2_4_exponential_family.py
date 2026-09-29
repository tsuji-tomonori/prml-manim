"""PRML 2.4 — linked visual experiments, implemented in Manim Community."""
from manim import *
import numpy as np
from scene_support import *
from exponential_model import COINS, POINTS, sigmoid, softmax, normal, gaussian_natural, beta_pdf, log_likelihood, transformed_density


class PRML24ExponentialFamily(NarratedScene):
    def construct(self):
        self.run_scenes([self.weights,self.family,self.categories,self.gaussian,
                         self.statistics,self.likelihood,self.conjugacy,
                         self.coordinates,self.invariance,self.summary])

    def bars(self, getter, colors=(BLUE,RED), ymax=1., labels=('0 / 裏','1 / 表')):
        n=len(colors)
        ax=Axes(x_range=[-.7,n-.3,1],y_range=[0,ymax,ymax/2],x_length=7,y_length=3,
                tips=False,axis_config=dict(color=MUTED),
                y_axis_config=dict(include_numbers=True,font_size=22))
        ax.move_to([0,.05,0])
        display_axes=VGroup(ax.x_axis,ax.y_axis.copy().shift(ax.c2p(-.7,0)-ax.c2p(0,0)))
        group=VGroup()
        for k,color in enumerate(colors):
            bar=always_redraw(lambda k=k,color=color: Rectangle(width=.85,height=max(.001,3*getter()[k]/ymax),
                    color=color,fill_opacity=.7).move_to(ax.c2p(k,0),aligned_edge=DOWN))
            value=DecimalNumber(getter()[k],num_decimal_places=2,font_size=25,color=color)
            value.add_updater(lambda m,k=k:m.set_value(getter()[k]).move_to(ax.c2p(k,getter()[k])+UP*.25))
            label=jp(labels[k],23,color).move_to(ax.c2p(k,0)+DOWN*.36)
            group.add(bar,value,label)
        return VGroup(display_axes,group)

    def weights(self):
        eta=ValueTracker(0.)
        norm=ValueTracker(1.)
        def vals():
            w=np.array([1.,np.exp(eta.get_value())])
            return w/(1+norm.get_value()*(w.sum()-1))
        # Fixed 0..1 probability chart, replaced explicitly by a 0..3 weights chart.
        bars=self.bars(vals)
        label=jp('確率：合計 1',25,GREEN).move_to([0,2.4,0])
        control=slider(eta,-2,2)
        self.add(bars,label,control)
        self.beat(self.highlight(bars),eta.animate.set_value(.4))
        self.beat(lambda:eta.animate.set_value(1.6),lambda:eta.animate.set_value(0.))
        self.remove(bars,label)
        norm.set_value(0.)
        bars=self.bars(vals,ymax=3.)
        label=jp('重み：まだ合計 1 ではない',25,YELLOW).move_to([0,2.4,0])
        self.add(bars,label)
        self.beat(eta.animate.set_value(np.log(2.5)),self.highlight(bars))
        eq=formula(r'w_0=1,\quad w_1=e^\eta',(0,2.4,0))
        self.remove(label); self.add(eq)
        self.beat(self.highlight(eq),eta.animate.set_value(-1.5))
        self.beat(norm.animate.set_value(1.),eta.animate.set_value(np.log(2.)))
        final=formula(r'p(0)=\frac{1}{1+e^\eta},\quad p(1)=\frac{e^\eta}{1+e^\eta}',(0,2.4,0))
        self.beat(self.change(eq,final),eta.animate.set_value(0.))

    def family(self):
        eta=ValueTracker(-2.)
        ax,labels=axes([-3,3,1],[0,1,.5],width=7,height=2.8,center=(0,-.1,0),xlabel=r'\eta',ylabel=r'\mu')
        graph=curve(ax,sigmoid,-3,3,GREEN)
        dot=always_redraw(lambda:Dot(ax.c2p(eta.get_value(),sigmoid(eta.get_value())),color=YELLOW,radius=.08))
        eq=formula(r'p(x|\mu)=\mu^x(1-\mu)^{1-x}')
        control=slider(eta,-3,3)
        self.add(eq,ax,labels,graph,dot,control)
        self.beat(self.highlight(eq),self.highlight(graph))
        logit=formula(r'\eta=\ln\frac{\mu}{1-\mu},\qquad \mu=\frac{1}{1+e^{-\eta}}')
        self.beat(self.match(eq,logit),eta.animate.set_value(2.))
        self.beat(lambda:eta.animate.set_value(-1.),lambda:eta.animate.set_value(0.))
        self.remove(ax,labels,graph,dot,control)
        full=MathTex(r'p(x|\eta)=',r'h(x)',r'g(\eta)',r'\exp\{',r'\eta^{\mathrm T}',r'u(x)',r'\}',font_size=43).move_to([0,.65,0])
        for i,c in [(1,MUTED),(2,PURPLE),(4,YELLOW),(5,BLUE)]: full[i].set_color(c)
        self.beat(self.match(logit,full),self.highlight(VGroup(full[4],full[5])))
        self.inner_product_aid()
        meanings=VGroup(jp('特徴量：データから作る材料',24,BLUE),jp('自然パラメータ：材料の重み',24,YELLOW),jp('土台 × 正規化係数',24,PURPLE)).arrange(DOWN,buff=.22).move_to([0,-.8,0])
        self.add(jp('連続：積分　離散：和',18,MUTED).move_to([0,-1.8,0]))
        normalization=formula(r'g(\eta)\int h(x)\exp\{\eta^{\mathrm T}u(x)\}\,dx=1',(0,-2.3,0),30)
        self.beat(FadeIn(meanings[:2]),AnimationGroup(FadeIn(meanings[2]),FadeIn(normalization)))
        sub=formula(r'u(x)=x,\quad h(x)=1,\quad g(\eta)=\frac{1}{1+e^\eta}',(0,2.35,0),31)
        self.beat(FadeIn(sub),self.highlight(full))


    def review_body(self, label):
        body=[m for m in self.mobjects if m is not self.header and m is not self.subtitle]
        self.remove(*body)
        frame=RoundedRectangle(width=11.6,height=5.1,corner_radius=.12,
                               color='#FFFF00',stroke_width=1.3)
        heading=jp(label,24).move_to([-5.4,2.17,0],aligned_edge=LEFT)
        self.add(frame,heading)
        return body

    def restore_body(self, body):
        temporary=[m for m in self.mobjects if m is not self.header and m is not self.subtitle]
        for m in temporary:
            m.clear_updaters(recursive=True)
        self.remove(*temporary)
        self.add(*body)

    def inner_product_aid(self):
        body=self.review_body('補足：縦から横へ、掛けて足す')
        blue='#58C4DD'; purple='#9A72AC'; yellow='#FFFF00'; green='#83C167'
        self.add(jp('説明用の例：2つの成分',20,MUTED).move_to([2.8,2.16,0]))
        col=Matrix([[2],[-1]],element_to_mobject=lambda x:tex(str(x),32,purple),
                   v_buff=.6).move_to([-4, .55,0])
        name=tex(r'\eta=',30,purple).next_to(col,LEFT)
        row=Matrix([[2,-1]],element_to_mobject=lambda x:tex(str(x),32,purple),
                   h_buff=.85).move_to([-1,.55,0])
        rowname=tex(r'\eta^T=',30,purple).next_to(row,UP,buff=.25)
        u=Matrix([[1],[3]],element_to_mobject=lambda x:tex(str(x),32,blue),
                 v_buff=.6).move_to([1.25,.55,0])
        uname=tex('u=',30,blue).next_to(u,UP,buff=.25)
        arrow=Arrow([-3,.55,0],[-2.2,.55,0],color=yellow,buff=.05)
        self.add(col,name)
        products=VGroup(tex(r'2\times1=+2',32,yellow),tex(r'(-1)\times3=-3',32,yellow)).arrange(DOWN,buff=.35).move_to([3.65,.55,0])
        total=tex(r'\eta^T u=2-3=-1',38,green).move_to([0,-1.55,0])
        # The example retains signs; projection geometry is unnecessary here.
        self.beat(
            AnimationGroup(TransformFromCopy(col,row),FadeIn(rowname),GrowArrow(arrow)),
            AnimationGroup(FadeIn(u),FadeIn(uname),
                LaggedStart(*[TransformFromCopy(VGroup(row.get_entries()[i],u.get_entries()[i]),products[i])
                              for i in range(2)],lag_ratio=.4)),
            AnimationGroup(TransformFromCopy(products,total),
                           ShowPassingFlash(SurroundingRectangle(products,color=yellow),time_width=.7)),
        )
        self.restore_body(body)

    def mean_recap(self):
        body=self.review_body('復習: 1.2 期待値')
        # Preserve 1.2 moments(): blue probability bars, yellow mean marker.
        mu=ValueTracker(.3)
        ax=Axes(x_range=[0,1,.5],y_range=[0,1,.5],x_length=5.4,y_length=2,
                tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=20))
        ax.move_to([-1.6,-.05,0])
        zero=tex('0',22).move_to(ax.c2p(0,0)+DOWN*.3)
        bars=always_redraw(lambda:VGroup(*[
            Rectangle(width=.42,height=2*p,color=BLUE,fill_opacity=.6)
            .move_to(ax.c2p(k,0),aligned_edge=DOWN)
            for k,p in [(0,1-mu.get_value()),(1,mu.get_value())]]))
        mean=always_redraw(lambda:Line(ax.c2p(mu.get_value(),0),ax.c2p(mu.get_value(),1),color=YELLOW))
        fulcrum=always_redraw(lambda:Triangle(color=YELLOW,fill_opacity=1).scale(.1)
                             .move_to(ax.c2p(mu.get_value(),0)+DOWN*.13))
        labels=VGroup(jp('確率',22,BLUE).move_to([-4.8,1.33,0]),
                      jp('平均',22,YELLOW).move_to([.7,1.33,0]),
                      tex('x',24).next_to(ax.x_axis,RIGHT,buff=.3))
        value=readout(r'\mathbb E[x]=\mu=',mu.get_value,(3.2,.7,0),YELLOW)
        eq=tex(r'0(1-\mu)+1\mu=\mu',32,YELLOW).move_to([-1.6,-1.78,0])
        link=tex(r'\mathbb E[x]\ \longrightarrow\ A^{\prime}(\eta)',30,YELLOW).move_to([3.0,-1.5,0])
        self.add(ax,bars,mean,fulcrum,labels,value,zero)
        self.beat(
            self.highlight(bars,BLUE),
            AnimationGroup(mu.animate.set_value(.7),FadeIn(eq)),
            AnimationGroup(FadeIn(link),self.highlight(value)()),
        )
        self.restore_body(body)

    def curvature_aid(self):
        body=self.review_body('補足：二階微分は、傾きの変わりやすさ')
        blue='#58C4DD'; yellow='#FFFF00'; green='#83C167'
        eta=ValueTracker(0.)
        mu=lambda:float(sigmoid(eta.get_value()))
        variance=lambda:mu()*(1-mu())
        self.add(jp('ベルヌーイの例',20,MUTED).move_to([3.7,2.17,0]),
                 jp('微分する変数：',20).move_to([2.55,1.55,0]),
                 tex(r'\eta',27,yellow).move_to([4.05,1.55,0]))
        def small_axes(ymax,center):
            a=Axes(x_range=[-3,3,1],y_range=[0,ymax,ymax],x_length=4.6,y_length=1.2,
                   tips=False,axis_config=dict(color=MUTED,include_numbers=True,font_size=17)).move_to(center)
            return a
        upper=small_axes(3.2,(-2.6,.85,0));lower=small_axes(1.,(-2.6,-1.13,0))
        A=lambda x:np.logaddexp(0,x)
        def tangent(ax,f,df):
            e=eta.get_value()
            return Line(ax.c2p(e-.55,f(e)-.55*df()),ax.c2p(e+.55,f(e)+.55*df()),
                        color=yellow,stroke_width=4)
        top_tan=always_redraw(lambda:tangent(upper,A,mu))
        low_tan=always_redraw(lambda:tangent(lower,sigmoid,variance))
        top=curve(upper,A,-3,3,blue);bottom=curve(lower,sigmoid,-3,3,blue)
        dot=always_redraw(lambda:Dot(lower.c2p(eta.get_value(),mu()),color=yellow,radius=.065))
        tag1=tex('A',24,blue).move_to([-5.25,1.42,0])
        tag2=tex(r'A^{\prime}=\mu',24,blue).move_to([-4.8,-.32,0])
        xs=VGroup(*[tex(r'\eta',22).next_to(a.x_axis,RIGHT,buff=.12) for a in [upper,lower]])
        slope=readout(r'A^{\prime}=\mu=',mu,(2.9,.87,0),yellow)
        second=readout(r'A^{\prime\prime}=',variance,(2.9,.1,0),yellow)
        equals=tex(r'A^{\prime\prime}=\mu(1-\mu)',30,green).move_to([2.9,-.6,0])
        # Equal-height bars come from the same Bernoulli distribution.
        def bar(x,color):
            return always_redraw(lambda:Rectangle(width=.55,height=3*variance(),color=color,fill_opacity=.65)
                                 .move_to([x,-1.75,0],aligned_edge=DOWN))
        bars=VGroup(bar(2.1,yellow),bar(3.7,green))
        labels=VGroup(jp('変化率',19,yellow).move_to([2.1,-2.0,0]),
                      jp('分散',19,green).move_to([3.7,-2.,0]))
        comparison=tex(r'\mu:0.50\to0.10\quad A^{\prime\prime}:0.25\to0.09',24).move_to([-.3,-2.3,0])
        self.add(upper,top,top_tan,tag1,slope)
        self.beat(
            AnimationGroup(FadeIn(lower),FadeIn(bottom),TransformFromCopy(slope[0],tag2),FadeIn(xs),
                           FadeIn(low_tan),FadeIn(dot)),
            AnimationGroup(FadeIn(second),FadeIn(equals),FadeIn(bars),FadeIn(labels)),
            AnimationGroup(eta.animate(rate_func=lambda a:smooth(np.clip((a-.45)/.4,0,1)))
                           .set_value(np.log(1/9)),FadeIn(comparison,rate_func=lambda a:min(1,8*a))),
        )
        self.restore_body(body)

    def categories(self):
        a,b=ValueTracker(0),ValueTracker(0)
        values=lambda:softmax([a.get_value(),b.get_value(),0])
        bars=self.bars(values,colors=(RED,BLUE,GREEN),labels=('赤','青','緑'))
        eq=formula(r'x=(1,0,0),\ (0,1,0),\ (0,0,1)',(0,2.45,0))
        controls=VGroup(slider(a,-2,2,(-3.2,-2.35,0),r'\eta_1',width=3.0,color=RED),
                        slider(b,-2,2,(2.8,-2.35,0),r'\eta_2',width=3.0,color=BLUE))
        self.add(bars,eq,controls)
        self.beat(self.highlight(bars),self.highlight(eq))
        self.beat(self.change(eq,formula(r'w=(e^{\eta_1},e^{\eta_2},1)',(0,2.45,0))),a.animate.set_value(1.7))
        self.beat(b.animate.set_value(1.3),a.animate.set_value(-.7))
        soft=formula(r'\mu_k=\frac{e^{\eta_k}}{1+e^{\eta_1}+e^{\eta_2}},\quad\mu_3=\frac{1}{1+e^{\eta_1}+e^{\eta_2}}',(0,2.45,0),29)
        self.beat(self.change(eq,soft),self.highlight(eq))
        self.beat(AnimationGroup(a.animate.set_value(.8),b.animate.set_value(-1.2)),self.highlight(bars))
        reduced=formula(r'p(x|\eta)=\frac{\exp(\eta_1 x_1+\eta_2 x_2)}{1+e^{\eta_1}+e^{\eta_2}}',(0,2.45,0),31)
        self.beat(self.change(eq,reduced),AnimationGroup(a.animate.set_value(0),b.animate.set_value(0)))

    def gaussian(self):
        mu,sigma=ValueTracker(-.8),ValueTracker(1.)
        ax,labels=axes([-4,4,2],[0,.8,.4],width=8.4,height=2.8,center=(0,-.1,0),ylabel=r'p(x)')
        graph=always_redraw(lambda:curve(ax,lambda x:normal(x,mu.get_value(),sigma.get_value()),-4,4,GREEN,True))
        vertical=always_redraw(lambda:DashedLine(ax.c2p(mu.get_value(),0),ax.c2p(mu.get_value(),normal(mu.get_value(),mu.get_value(),sigma.get_value())),color=YELLOW))
        eq=formula(r'p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]',(0,2.4,0),31)
        controls=VGroup(slider(mu,-1,1,(-3.4,-2.35,0),r'\mu',width=2.7),slider(sigma,.5,1.5,(2.6,-2.35,0),r'\sigma',width=2.7,color=PURPLE))
        self.add(ax,labels,graph,vertical,eq,controls)
        self.beat(self.highlight(graph),mu.animate.set_value(.8))
        self.beat(lambda:sigma.animate.set_value(.55),lambda:sigma.animate.set_value(1.))
        expansion=MathTex(r'-\frac{(x-\mu)^2}{2\sigma^2}=',r'\frac{\mu}{\sigma^2}x',r'-\frac{1}{2\sigma^2}x^2',r'-\frac{\mu^2}{2\sigma^2}',font_size=31).move_to([0,2.4,0])
        expansion[1].set_color(YELLOW); expansion[2].set_color(PURPLE)
        self.beat(self.match(eq,expansion),self.highlight(expansion[1:3]))
        natural=formula(r'\eta_1=\mu/\sigma^2,\quad\eta_2=-1/(2\sigma^2),\quad u(x)=(x,x^2)^{\mathrm T}',(0,2.5,0),28)
        self.remove(expansion);self.add(natural)
        nums=VGroup(readout(r'\eta_1=',lambda:gaussian_natural(mu.get_value(),sigma.get_value())[0],(-3,1.7,0)),
                    readout(r'\eta_2=',lambda:gaussian_natural(mu.get_value(),sigma.get_value())[1],(3,1.7,0),PURPLE))
        self.add(nums)
        self.beat(self.highlight(natural),AnimationGroup(mu.animate.set_value(-.5),sigma.animate.set_value(.7)))
        self.beat(sigma.animate.set_value(1.1),self.highlight(nums[1]))
        self.remove(nums)
        g=formula(r'h(x)=(2\pi)^{-1/2},\quad g(\eta)=\sqrt{-2\eta_2}\exp\!\left(\frac{\eta_1^2}{4\eta_2}\right)',(0,2.45,0),29)
        self.beat(self.change(natural,g),AnimationGroup(mu.animate.set_value(0),sigma.animate.set_value(.8)))

    def statistics(self):
        coins=VGroup(*[VGroup(Circle(radius=.23,color=RED if x else BLUE,fill_opacity=.5),tex(str(x),25)).move_to([-4.5+i,1.2,0]) for i,x in enumerate(COINS)])
        eq=formula(r'N=10,\quad S=\sum_n x_n=7',(0,-.15,0),38)
        self.add(coins)
        self.beat(LaggedStart(*[FadeIn(c) for c in coins],lag_ratio=.15),FadeIn(eq))
        order=np.argsort(COINS)
        destinations={int(i):[-4.5+j,1.2,0] for j,i in enumerate(order)}
        self.beat(AnimationGroup(*[c.animate.move_to(destinations[i]) for i,c in enumerate(coins)]),self.highlight(eq))
        self.beat(AnimationGroup(*[c.animate.scale(.1).move_to(eq.get_center()) for c in coins]),self.highlight(eq))
        self.remove(coins,*coins)
        likelihood=MathTex(r'p(X|\eta)=',r'\prod_n h(x_n)',r'g(\eta)^N',r'\exp\!\left\{\eta^{\mathrm T}',r'\sum_n u(x_n)',r'\right\}',font_size=32).move_to([0,1.8,0])
        likelihood[4].set_color(BLUE); likelihood[2].set_color(PURPLE)
        self.beat(FadeIn(likelihood),self.highlight(likelihood[4]))
        self.remove(eq,likelihood)
        line=NumberLine(x_range=[-1.5,2,.5],length=9,include_numbers=True,font_size=22).move_to([0,1.0,0])
        dots=VGroup(*[Dot(line.n2p(x),color=BLUE,radius=.07) for x in POINTS])
        count=ValueTracker(0)
        n=lambda:min(6,int(count.get_value()+1e-6))
        stats=VGroup(readout('N=',n,(-4,-.25,0),YELLOW,0),
                     readout(r'\sum x_n=',lambda:POINTS[:n()].sum(),(0,-.25,0),BLUE),
                     readout(r'\sum x_n^2=',lambda:(POINTS[:n()]**2).sum(),(4,-.25,0),PURPLE))
        for i,d in enumerate(dots):
            d.add_updater(lambda m,i=i:m.set_color(BLUE if i<n() else MUTED).set_opacity(1 if i<n() else .35))
        result=formula(r'\mu_{\rm ML}='+f'{POINTS.mean():.2f}'+r',\quad\sigma^2_{\rm ML}=\frac{\sum x_n^2}{N}-\mu_{\rm ML}^2='+f'{POINTS.var():.3f}',(0,-1.5,0),33)
        self.add(line,dots,stats)
        self.beat(count.animate.set_value(6),FadeIn(result))
        self.beat(LaggedStart(*[Indicate(d,color=YELLOW) for d in dots],lag_ratio=.2),self.highlight(VGroup(stats,result)))

    def likelihood(self):
        eta=ValueTracker(-1.5)
        ax,labels=axes([-3,3,1],[0,1,.5],width=8,height=2.8,center=(0,-.1,0),xlabel=r'\eta',ylabel=r'\mathbb E[x]')
        graph=curve(ax,sigmoid,-3,3,GREEN)
        target=DashedLine(ax.c2p(-3,.7),ax.c2p(3,.7),color=BLUE)
        dot=always_redraw(lambda:Dot(ax.c2p(eta.get_value(),sigmoid(eta.get_value())),color=YELLOW,radius=.09))
        read=readout(r'\mu=',lambda:sigmoid(eta.get_value()),(4.8,.5,0))
        eq=formula(r'\ell(\eta)=7\eta-10\ln(1+e^\eta)',(0,2.35,0),34)
        self.add(ax,labels,graph,target,dot,read,eq,slider(eta,-3,3))
        self.beat(self.highlight(eq),self.highlight(target,BLUE))
        self.beat(eta.animate.set_value(np.log(7/3)),self.highlight(dot))
        self.mean_recap()
        self.remove(graph,target,dot,labels,read)
        # Replace the coordinate system explicitly: the new ordinate is A, not E[x].
        self.remove(ax)
        ax,labels=axes([-3,3,1],[0,3.2,1],width=8,height=2.8,center=(0,-.1,0),xlabel=r'\eta',ylabel='A')
        graph=curve(ax,lambda x:np.logaddexp(0,x),-3,3,GREEN)
        tangent=always_redraw(lambda:Line(ax.c2p(eta.get_value()-.55,np.logaddexp(0,eta.get_value())-.55*sigmoid(eta.get_value())),ax.c2p(eta.get_value()+.55,np.logaddexp(0,eta.get_value())+.55*sigmoid(eta.get_value())),color=YELLOW,stroke_width=5))
        read=readout(r"A'(\eta)=",lambda:sigmoid(eta.get_value()),(4.8,.6,0))
        self.add(ax,labels,graph,tangent,read)
        self.beat(self.change(eq,formula(r'A(\eta)=-\ln g(\eta),\quad\nabla A=\mathbb E[u(x)]')),eta.animate.set_value(-1.5))
        self.beat(eta.animate.set_value(1.5),self.change(eq,formula(r'\nabla^2 A=\operatorname{Cov}[u(x)]')))
        self.curvature_aid()
        optimum=formula(r'-\nabla\ln g(\eta_{\rm ML})=\frac1N\sum_n u(x_n)',(0,2.35,0),34)
        self.beat(AnimationGroup(self.change(eq,optimum),eta.animate.set_value(np.log(7/3))),self.highlight(eq))
        self.beat(self.highlight(eq),AnimationGroup(eta.animate.set_value(3),self.change(eq,formula(r'S=N:\quad \mu_{\rm ML}=1,\quad\eta\to+\infty'))))

    def conjugacy(self):
        t=ValueTracker(0.)
        ax,labels=axes([0,1,.2],[0,3.5,1],width=8,height=2.8,center=(0,-.1,0),xlabel=r'\mu',ylabel=r'p(\mu)')
        graph=always_redraw(lambda:curve(ax,lambda x:beta_pdf(x,2+7*t.get_value(),2+3*t.get_value()),0,1,PURPLE,True))
        mean=always_redraw(lambda:DashedLine(ax.c2p((2+7*t.get_value())/(4+10*t.get_value()),0),ax.c2p((2+7*t.get_value())/(4+10*t.get_value()),2.6),color=YELLOW))
        eq=formula(r'p(\mu)=\operatorname{Beta}(\mu|2,2)')
        counts=VGroup(readout('a=',lambda:2+7*t.get_value(),(-2,-2.3,0),RED),readout('b=',lambda:2+3*t.get_value(),(2,-2.3,0),BLUE))
        self.add(ax,labels,graph,mean,eq,counts)
        self.beat(self.highlight(graph),self.highlight(eq))
        self.beat(self.change(eq,formula(r'p(\mu)=\operatorname{Beta}(\mu|a,b)')),t.animate.set_value(1))
        self.beat(self.highlight(counts),self.change(eq,formula(r'\operatorname{Beta}(2,2)\ \longrightarrow\ \operatorname{Beta}(9,5)')))
        self.remove(ax,labels,graph,mean,counts,eq)
        prior=formula(r'p(\eta|\chi,\nu)=f(\chi,\nu)g(\eta)^\nu\exp\{\nu\eta^{\mathrm T}\chi\}',(0,1.8,0),33)
        post=formula(r'p(\eta|X,\chi,\nu)\propto g(\eta)^{\nu+N}\exp\{\eta^{\mathrm T}(\nu\chi+S)\}',(0,.55,0),33)
        self.beat(FadeIn(prior),FadeIn(post))
        update=formula(r'\nu\to\nu+N,\quad\chi\to\frac{\nu\chi+S}{\nu+N},\qquad S=\sum_nu(x_n)',(0,-.9,0),32)
        self.beat(self.highlight(prior),FadeIn(update))
        jacobian=formula(r'p_\eta(\eta)=p_\mu(\mu)\left|\frac{d\mu}{d\eta}\right|',(0,-2.25,0),29)
        self.beat(FadeIn(jacobian),self.highlight(jacobian))

    def coordinates(self):
        q=ValueTracker(1.)
        ax,labels=axes([0,1,.25],[0,2,.5],width=8,height=3,center=(0,-.05,0),xlabel=r'\lambda\ \to\ \eta',ylabel='p')
        colors=[BLUE,GREEN,YELLOW,PURPLE]
        def piece(k):
            p=q.get_value(); lo=(k/4)**(1/p); hi=((k+1)/4)**(1/p)
            return curve(ax,lambda x:transformed_density(x,p),lo,hi,colors[k],True)
        areas=VGroup(*[always_redraw(lambda k=k:piece(k)) for k in range(4)])
        eq=formula(r'p_\lambda(\lambda)=1,\quad 0\leq\lambda\leq1')
        recap=jp('復習: 1.2 密度と座標変換',21).move_to([-5.7,2.87,0],aligned_edge=LEFT)
        self.add(ax,labels,areas,eq,recap)
        self.beat(self.highlight(eq),self.highlight(areas))
        self.beat(self.highlight(areas),self.highlight(ax.x_axis))
        transform=formula(r'\lambda=\eta^q,\quad q:1\to2,\quad 0\leq\eta\leq1')
        self.beat(self.change(eq,transform),q.animate.set_value(1.45))
        self.beat(q.animate.set_value(2),self.change(eq,formula(r'p_\eta(\eta)=p_\lambda(\eta^2)\left|2\eta\right|')))
        mass=formula(r'\int_{\sqrt{k/4}}^{\sqrt{(k+1)/4}}2\eta\,d\eta=\frac14',(0,-2.15,0),29)
        self.beat(AnimationGroup(self.change(eq,formula(r'p_\eta(\eta)=2\eta')),FadeIn(mass)),self.highlight(areas))
        self.beat(self.highlight(eq),self.highlight(mass))

    def invariance(self):
        shift=ValueTracker(-1.5)
        ax,labels=axes([-4,4,2],[0,1.3,.5],width=9,height=2.6,center=(0,.1,0),xlabel=r'\mu',ylabel=r'p(\mu)')
        flat=Line(ax.c2p(-4,1),ax.c2p(4,1),color=PURPLE)
        patch=always_redraw(lambda:Polygon(ax.c2p(shift.get_value(),0),ax.c2p(shift.get_value()+1.5,0),ax.c2p(shift.get_value()+1.5,1),ax.c2p(shift.get_value(),1),color=GREEN,fill_opacity=.4))
        eq=formula(r'p(x|\mu)=f(x-\mu),\quad p(\mu)=\mathrm{const}')
        self.add(ax,labels,flat,patch,eq)
        self.beat(lambda:shift.animate.set_value(.5),lambda:shift.animate.set_value(-2.5))
        divergent=formula(r'\int_{-\infty}^{\infty}c\,d\mu=\infty\quad(c>0)',(0,-2.25,0),33)
        self.beat(self.highlight(flat),FadeIn(divergent))
        self.remove(ax,labels,flat,patch,divergent)
        logax,loglabels=axes([0,3,1],[0,1.3,.5],width=9,height=2.6,center=(0,.1,0),xlabel=r'\log_{10}\sigma',ylabel=r'p_{\log_{10}\sigma}')
        tiles=VGroup(*[Polygon(logax.c2p(i,0),logax.c2p(i+1,0),logax.c2p(i+1,1),logax.c2p(i,1),color=c,fill_opacity=.35) for i,c in enumerate([BLUE,GREEN,PURPLE])])
        numbers=VGroup(*[tex(s,25).move_to(logax.c2p(i+.5,.5)) for i,s in enumerate([r'1\to10',r'10\to100',r'100\to1000'])])
        self.add(logax,loglabels,tiles,numbers)
        scale=formula(r'p(x|\sigma)=\frac1\sigma f\!\left(\frac{x}{\sigma}\right)')
        self.beat(self.change(eq,scale),self.highlight(tiles))
        self.beat(self.change(eq,formula(r'p(\sigma)\propto\frac1\sigma\quad\Longleftrightarrow\quad p(\ln\sigma)=\mathrm{const}')),LaggedStart(*[Indicate(n) for n in numbers],lag_ratio=.25))
        precision=formula(r'\lambda=\sigma^{-2},\quad p(\lambda)\propto\lambda^{-1}',(0,-2.25,0),33)
        self.beat(FadeIn(precision),self.highlight(eq))
        warning=jp('事後分布の積分が有限か、確認する',27,YELLOW).move_to([0,2.4,0])
        self.beat(self.replace(eq,warning),self.highlight(warning))

    def summary(self):
        ax,labels=axes([-3,3,1],[0,.8,.4],width=8,height=2.8,center=(0,-.2,0),ylabel='p(x)')
        mu=ValueTracker(-.7)
        graph=always_redraw(lambda:curve(ax,lambda x:normal(x,mu.get_value(),.7),-3,3,GREEN,True))
        self.add(ax,labels,graph)
        eq=formula(r'p(x|\eta)=h(x)g(\eta)\exp\{\eta^{\mathrm T}u(x)\}')
        self.beat(mu.animate.set_value(.7),FadeIn(eq))
        self.beat(mu.animate.set_value(-.7),self.highlight(eq))
        self.remove(graph)
        model=curve(ax,lambda x:normal(x,POINTS.mean(),np.sqrt(POINTS.var())),-3,3,GREEN,True)
        dots=VGroup(*[Dot(ax.c2p(x,.02),color=BLUE,radius=.07) for x in POINTS])
        self.add(dots)
        stats=formula(r'S=\sum_nu(x_n),\qquad\mathbb E_{\eta_{\rm ML}}[u(x)]=S/N')
        self.beat(self.change(eq,stats),FadeIn(model))
        post=formula(r'(\nu\chi,\nu)\longrightarrow(\nu\chi+S,\nu+N)')
        self.beat(self.change(eq,post),self.highlight(eq))
        mixed=curve(ax,lambda x:.5*normal(x,-1.2,.45)+.5*normal(x,1.1,.5),-3,3,PURPLE,True)
        note=jp('一般のガウス混合は、この単純な形の外へ',26,PURPLE).move_to([0,2.4,0])
        self.beat(self.highlight(model),AnimationGroup(Transform(model,mixed),self.replace(eq,note)))
        next_note=jp('次へ：データから、分布の形を作る',28,BLUE).move_to([0,-2.35,0])
        self.beat(self.highlight(model),FadeIn(next_note))
