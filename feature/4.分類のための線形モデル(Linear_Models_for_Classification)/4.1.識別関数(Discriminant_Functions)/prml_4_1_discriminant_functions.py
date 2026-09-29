"""PRML 4.1 — linked geometric experiments, implemented in Manim Community."""
import numpy as np
from manim import *
from caption_layout import jp, tex
from scene_support import NarratedScene
from discriminant_model import *

BLUE_CLS=ManimColor('#58B5ED'); ORANGE_CLS=ManimColor('#FFB45B')
GREEN_CLS=ManimColor('#77D49A'); YELLOW_W=ManimColor('#FFE079')
PURPLE_B=ManimColor('#C29AFF'); RED_LINE=ManimColor('#FF6B77')
MUTED=ManimColor('#A8B2C5'); COLORS=[BLUE_CLS,ORANGE_CLS,GREEN_CLS]
AID_INPUT=ManimColor('#58C4DD'); AID_OPERATION=ManimColor('#FFFF00')
AID_RESULT=ManimColor('#83C167'); AID_COMPARE=ManimColor('#9A72AC')

def readout(label,getter,pos,color=WHITE,places=2,size=27):
    number=DecimalNumber(getter(),num_decimal_places=places,font_size=size,color=color)
    group=VGroup(tex(label,size,color),number).arrange(RIGHT,buff=.15).move_to(pos)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group

def slider(tracker,lo,hi,pos,label,color=YELLOW_W,width=2.6):
    rail=Line(LEFT*width/2,RIGHT*width/2,color=MUTED,stroke_width=2).move_to(pos)
    dot=Dot(radius=.065,color=color)
    dot.add_updater(lambda m:m.move_to(rail.get_start()+(tracker.get_value()-lo)/(hi-lo)*rail.get_vector()))
    return VGroup(rail,dot,tex(label,25,color).next_to(rail,LEFT,buff=.2))

def boundary(ax,c,color=RED_LINE,width=3):
    b,w1,w2=c; xmin,xmax=ax.x_range[:2];ymin,ymax=ax.y_range[:2]
    pts=[]
    if abs(w2)>1e-10:
        for x in [xmin,xmax]:
            y=-(b+w1*x)/w2
            if ymin-1e-8<=y<=ymax+1e-8:pts.append([x,y])
    if abs(w1)>1e-10:
        for y in [ymin,ymax]:
            x=-(b+w2*y)/w1
            if xmin-1e-8<=x<=xmax+1e-8:pts.append([x,y])
    if len(pts)<2:return Line(ax.c2p(0,0),ax.c2p(0,0)+RIGHT*1e-6,stroke_opacity=0)
    # Pick the farthest pair to handle a line through a rectangle corner.
    a,c=max(((a,c) for a in pts for c in pts),key=lambda p:np.linalg.norm(np.array(p[0])-p[1]))
    return Line(ax.c2p(*a),ax.c2p(*c),color=color,stroke_width=width)

def cloud(ax,groups,radius=.05):
    return VGroup(*[VGroup(*[Dot(ax.c2p(*p),radius=radius,color=COLORS[k]) for p in g]) for k,g in enumerate(groups)])

def regions(ax,W,opacity=.15):
    bounds=(*ax.x_range[:2],*ax.y_range[:2]);out=VGroup()
    for k in range(W.shape[1]):
        p=region(W,k,bounds)
        if len(p)>2:out.add(Polygon(*[ax.c2p(*v) for v in p],fill_color=COLORS[k],fill_opacity=opacity,stroke_width=1.3,stroke_color=COLORS[k]))
    return out

class PRML41DiscriminantFunctions(NarratedScene):
    def scenes(self):
        return [self.geometry,self.distance,self.multiclass,self.least_squares_scene,
                self.fisher,self.relation,self.multi_fisher,self.perceptron_scene,
                self.limits,self.summary]

    def hint(self,text,color=MUTED):
        mob=jp(text,20,color).move_to([0,2.75,0]);self.add(mob);return mob

    def axes(self,xr=(-3.1,3.1,1),yr=(-2.1,2.1,1),center=(-2,.1,0),width=6.2,height=4.2,labels=('x_1','x_2')):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.2,'include_ticks':False}).move_to(center)
        self.add(ax)
        self.add(tex(labels[0],23,MUTED).next_to(ax.c2p(xr[1],0),RIGHT,buff=.1),
                 tex(labels[1],23,MUTED).move_to(ax.c2p(0,yr[1])+UP*.18+LEFT*.18))
        return ax

    def bottom(self,formula,color=WHITE,size=31):
        m=tex(formula,size,color).move_to([0,-2.53,0]);return m

    def formula(self,parts,pos=(3.7,1.5,0),size=32):
        m=MathTex(*parts,font_size=size).move_to(pos);return m

    def geometry(self):
        self.hint('青：クラス1　橙：クラス2　白：調べる入力')
        ax=self.axes();dots=cloud(ax,BASE)
        theta=ValueTracker(.45);bias=ValueTracker(0);px=ValueTracker(.6)
        w=lambda:direction(theta.get_value());coef=lambda:np.r_[bias.get_value(),w()]
        probe=always_redraw(lambda:Dot(ax.c2p(px.get_value(),.1),color=WHITE,radius=.085))
        self.add(probe);self.beat(FadeIn(dots))
        line=always_redraw(lambda:boundary(ax,coef()))
        self.add(line);self.beat(px.animate.set_value(-.8))
        formula=self.formula(['y(x)=',r'w^Tx','+',r'w_0'])
        formula[1].set_color(YELLOW_W);formula[3].set_color(PURPLE_B)
        expanded=tex(r'=w_1x_1+w_2x_2+w_0',27).next_to(formula,DOWN,buff=.24)
        self.beat(Write(formula),Write(expanded),px.animate.set_value(.8))
        arrow=always_redraw(lambda:Arrow(ax.c2p(*(-bias.get_value()*w())),ax.c2p(*((1.25-bias.get_value())*w())),buff=0,color=YELLOW_W,stroke_width=5))
        knob=slider(theta,-.8,1.3,[3.7,.05,0],r'\theta')
        self.add(arrow,knob);self.beat(theta.animate.set_value(1.1))
        bk=slider(bias,-1,1,[3.7,-.7,0],'w_0',PURPLE_B)
        self.add(bk);self.beat(bias.animate.set_value(.9))
        val=readout('y(x)=',lambda:w()@np.array([px.get_value(),.1])+bias.get_value(),[3.7,-1.5,0])
        rule=self.bottom(r'y(x)\geq0\Rightarrow C_1\qquad y(x)<0\Rightarrow C_2')
        probe.add_updater(lambda m:m.set_color(BLUE_CLS if w()@np.array([px.get_value(),.1])+bias.get_value()>=0 else ORANGE_CLS))
        self.add(val);self.beat(Write(rule),px.animate.set_value(-2.8),bias.animate.set_value(-.3))

    def distance(self):
        self.hint('同じ境界でも、スコアの倍率は変えられる')
        ax=self.axes();theta=.65;u=direction(theta);b=-.4
        r=ValueTracker(1.25);scale=ValueTracker(1)
        foot=-b*u+.3*np.array([-u[1],u[0]])
        point=lambda:foot+r.get_value()*u
        line=boundary(ax,np.r_[b,u]);self.add(line)
        dot=always_redraw(lambda:Dot(ax.c2p(*point()),radius=.08,color=WHITE))
        footdot=Dot(ax.c2p(*foot),color=MUTED)
        perp=always_redraw(lambda:DashedLine(ax.c2p(*foot),ax.c2p(*point())+UP*1e-6,color=YELLOW_W,stroke_width=3))
        self.add(dot,footdot);self.beat(Create(perp))
        arrow=always_redraw(lambda:Arrow(ax.c2p(0,0),ax.c2p(*(u*scale.get_value())),buff=0,color=YELLOW_W))
        self.add(arrow,readout('y(x)=',lambda:scale.get_value()*r.get_value(),[3.7,1.5,0],YELLOW_W),
                 readout(r'\|w\|=',scale.get_value,[3.7,.65,0],PURPLE_B))
        self.beat(phases=[('positive distance',self.sentence_duration(0),lambda:r.animate.set_value(1.8)),
                          ('cross boundary',self.sentence_duration(1),lambda:r.animate.set_value(-1.1))])
        sk=slider(scale,1,2,[3.7,-.25,0],'c',PURPLE_B);self.add(sk)
        self.beat(scale.animate.set_value(2),Write(self.bottom(r'(w,w_0)\mapsto(2w,2w_0)')))
        # The new equation replaces the previous footer, without a second layer.
        for m in list(self.mobjects):
            if isinstance(m,MathTex) and m.get_center()[1]<-2:self.remove(m)
        eq=self.bottom(r'r=\frac{y(x)}{\|w\|}\qquad x=x_{\perp}+r\frac{w}{\|w\|}')
        self.add(readout('r=',r.get_value,[3.7,-1.2,0],GREEN_CLS))
        self.beat(Write(eq),scale.animate.set_value(1))
        new=self.bottom(r'\widetilde{x}=(1,x^T)^T\qquad y=\widetilde{w}^{T}\widetilde{x}')
        self.beat(ReplacementTransform(eq,new),r.animate.set_value(.9))

    def multiclass(self):
        self.hint('一つの座標上で、二値の判定から最大スコアへ')
        ax=self.axes();q=Dot(ax.c2p(.5,.6),color=WHITE,radius=.085)
        ambiguous=Polygon(*[ax.c2p(*p) for p in [[0,0],[3.1,0],[3.1,2.1],[0,2.1]]],fill_color=YELLOW_W,fill_opacity=.2,stroke_width=0)
        note=VGroup(jp('青：右側なら正',24,BLUE_CLS),jp('橙：上側なら正',24,ORANGE_CLS),tex(r'C_1>C_2>C_3>C_1',28,YELLOW_W),jp('票が 1：1：1',23)).arrange(DOWN,buff=.32).move_to([3.7,.7,0])
        self.add(ambiguous,q);self.beat(FadeIn(note),Circumscribe(q,color=YELLOW_W))
        self.remove(ambiguous,note,q)
        bias=ValueTracker(0);W=lambda:MULTI_W+np.array([[0,0,bias.get_value()],[0,0,0],[0,0,0]])
        regs=always_redraw(lambda:regions(ax,W()));self.add(regs)
        px=ValueTracker(-1.8);py=ValueTracker(.8)
        probe=lambda:np.array([px.get_value(),py.get_value()])
        dot=always_redraw(lambda:Dot(ax.c2p(*probe()),color=WHITE,radius=.085))
        bars=VGroup(*[readout(f'y_{k+1}=',lambda k=k:scores(probe(),W())[0,k],[3.7,1.35-.65*k,0],COLORS[k]) for k in range(3)])
        self.add(dot,bars);f=self.bottom(r'y_k=w_k^Tx+w_{k0}\qquad k^*=\arg\max_k y_k')
        self.beat(Write(f),px.animate.set_value(1.8))
        ring=always_redraw(lambda:Circle(radius=.13,color=COLORS[int(np.argmax(scores(probe(),W())))]).move_to(dot))
        self.add(ring);self.beat(px.animate.set_value(0),py.animate.set_value(-1.4))
        f2=self.bottom(r'(w_k-w_j)^Tx+(w_{k0}-w_{j0})=0')
        self.beat(ReplacementTransform(f,f2),py.animate.set_value(1.8))
        a=np.array([.6,.7]);b=np.array([2.4,1.4])
        px.set_value(a[0]);py.set_value(a[1])
        path=Line(ax.c2p(*a),ax.c2p(*b),color=YELLOW_W)
        self.add(path);self.beat(px.animate.set_value(b[0]),py.animate.set_value(b[1]),Write(tex(r'\widehat{x}=\lambda x_A+(1-\lambda)x_B',25).move_to([3.7,-1.4,0])))
        self.add(slider(bias,0,2,[3.7,-.8,0],'w_{30}',GREEN_CLS));self.beat(bias.animate.set_value(2))

    def least_squares_scene(self):
        self.least_squares_recap()
        self.hint('独自データを毎フレーム再学習する')
        ax=self.axes(xr=(-2.6,2.4,1),yr=(-4.2,1.8,1),width=3.5,height=4.2,center=(-2.6,.1,0))
        amount=ValueTracker(0);W=lambda:least_squares(ls_data(amount.get_value()))
        dots=always_redraw(lambda:cloud(ax,ls_data(amount.get_value())))
        line=always_redraw(lambda:boundary(ax,W()[:,0]-W()[:,1]))
        self.add(dots,line)
        f=self.bottom(r'E_D=\frac12\sum_n\|W^T\widetilde{x}_n-t_n\|^2')
        labels=VGroup(tex(r'C_1:\ (1,0)',30,BLUE_CLS),tex(r'C_2:\ (0,1)',30,ORANGE_CLS)).arrange(DOWN,buff=.3).move_to([3.5,1.5,0])
        self.beat(Write(f),FadeIn(labels),ShowPassingFlash(line.copy().clear_updaters(),time_width=.5))
        fit=tex(r'W=\widetilde{X}^{\dagger}T',36,YELLOW_W).move_to([3.5,.35,0])
        self.beat(Write(fit),Circumscribe(labels,color=YELLOW_W))
        self.add(slider(amount,0,1,[3.5,-.5,0],'s',PURPLE_B));self.beat(amount.animate.set_value(1))
        lossax=Axes(x_range=[-.5,2.2,1],y_range=[0,2.6,1],x_length=2.8,y_length=1.4,tips=False,axis_config={'include_ticks':False,'color':MUTED}).move_to([3.5,-1.45,0])
        z=ValueTracker(1);curve=lossax.plot(lambda x:(x-1)**2,color=ORANGE_CLS)
        marker=always_redraw(lambda:Dot(lossax.c2p(z.get_value(),(z.get_value()-1)**2),color=YELLOW_W))
        self.add(lossax,curve,marker,tex('(y-1)^2',21).next_to(lossax,RIGHT,buff=.08))
        self.beat(z.animate.set_value(2.1))
        self.remove(lossax,curve,marker)
        for m in list(self.mobjects):
            if isinstance(m,MathTex) and m.tex_string=='(y-1)^2':self.remove(m)
        px=ValueTracker(-2.2);sample=lambda:np.array([px.get_value(),-2.5])
        probe=always_redraw(lambda:Dot(ax.c2p(*sample()),color=WHITE,radius=.085))
        outputs=VGroup(*[readout(f'y_{k+1}=',lambda k=k:scores(sample(),W())[0,k],[3+.95*k,-1.35-.45*k,0],COLORS[k],2,25) for k in range(2)])
        self.add(probe,outputs);self.beat(px.animate.set_value(2.2))
        # Same screen, new dataset: the middle band loses everywhere.
        self.remove(*[m for m in self.mobjects if m is not self.subtitle and m.get_center()[1]<2.5])
        ax=self.axes(xr=(-2.3,2.3,1),yr=(-1.6,1.6,1),width=5.8,height=3.9,center=(-2,.1,0))
        points=cloud(ax,BANDS);reg=regions(ax,BAND_W);self.add(reg,points)
        note=VGroup(jp('中央の橙が選ばれない',26,ORANGE_CLS),tex('y_2=0.20',31,ORANGE_CLS),tex(r'\max(y_1,y_3)\geq0.40',28)).arrange(DOWN,buff=.4).move_to([3.6,.7,0])
        self.beat(FadeIn(note),Indicate(points[1],color=ORANGE_CLS,scale_factor=1.18))
        good=VGroup(*[Line(ax.c2p(x,-1.6),ax.c2p(x,1.6),color=GREEN_CLS,stroke_width=3) for x in [-.9,.9]])
        self.beat(Create(good),FadeOut(reg),Write(jp('目標の数への近さ ≠ 分類のよさ',27).move_to([0,-2.53,0])))

    def fisher_display(self):
        ax=self.axes(xr=(-3.4,3.4,1),yr=(-2,3.2,1),width=5.3,height=4.05,center=(-2.75,.2,0))
        dots=cloud(ax,FISH,.037);self.add(dots)
        theta=ValueTracker(1.1);w=lambda:direction(theta.get_value())
        axis=always_redraw(lambda:Line(ax.c2p(*(-2.5*w())),ax.c2p(*(2.5*w())),color=YELLOW_W,stroke_width=3))
        projected=always_redraw(lambda:VGroup(*[VGroup(*[Dot(ax.c2p(*(w()*(p@w()))),radius=.035,color=COLORS[k]) for p in g]) for k,g in enumerate(FISH)]))
        # Show a few perpendicular guide lines, then all projected dots.
        guides=always_redraw(lambda:VGroup(*[Line(ax.c2p(*p),ax.c2p(*(w()*(p@w()))),color=COLORS[k],stroke_width=1,stroke_opacity=.45) for k,g in enumerate(FISH) for p in g[::4]]))
        strip=NumberLine(x_range=[-4,4,2],length=4.3,include_numbers=False,include_ticks=True,color=MUTED).move_to([3.55,-.45,0])
        shadow=VGroup(*[Dot(radius=.032,color=COLORS[k]).add_updater(lambda m,p=p,k=k:m.move_to(strip.n2p(p@w())+UP*(.12 if k==0 else -.12))) for k,g in enumerate(FISH) for p in g])
        means=always_redraw(lambda:VGroup(*[Line(strip.n2p(m@w())+DOWN*.35,strip.n2p(m@w())+UP*.35,color=COLORS[k],stroke_width=3) for k,m in enumerate(MEANS)]))
        self.add(strip,shadow,means,tex('y=w^Tx',30).move_to([3.55,.5,0]),
                 slider(theta,-1.1,1.2,[3.55,-1.35,0],r'\theta'))
        return ax,dots,theta,axis,projected,guides,strip,shadow

    def fisher(self):
        self.gaussian_recap()
        self.hint('青・橙：元の点　淡い線：垂線　右の点列：同じ射影値')
        ax,dots,theta,axis,projected,guides,strip,shadow=self.fisher_display()
        self.beat(Create(axis),FadeIn(guides),FadeIn(projected))
        self.beat(theta.animate.set_value(THETA_MEAN))
        self.beat(theta.animate.set_value(.75))
        f=self.bottom(r'J(w)=\frac{(m_2-m_1)^2}{s_1^2+s_2^2}',size=35)
        val=readout('J=',lambda:criterion(direction(theta.get_value())),[3.55,1.5,0],YELLOW_W,3)
        self.add(val);self.beat(Write(f),theta.animate.set_value(THETA_MEAN))
        self.beat(theta.animate.set_value(THETA_FISHER))
        f2=self.bottom(r'J=\frac{w^TS_Bw}{w^TS_Ww}\qquad w\propto S_W^{-1}(m_2-m_1)',size=30)
        f2.set_color(YELLOW_W)
        self.beat(ReplacementTransform(f,f2),Circumscribe(val,color=YELLOW_W))
        self.scatter_aid()
        self.beat(Circumscribe(f2,color=YELLOW_W))
        threshold=ValueTracker(-.6)
        cut=always_redraw(lambda:Line(strip.n2p(threshold.get_value())+UP*.55,strip.n2p(threshold.get_value())+DOWN*.55,color=RED_LINE,stroke_width=3))
        self.add(cut);self.beat(threshold.animate.set_value(.6))

    def body_card(self,title):
        saved=[m for m in self.mobjects if m is not self.subtitle]
        header=[m for m in saved if m.get_center()[1]>3]
        self.clear()
        frame=RoundedRectangle(width=10.4,height=4.45,corner_radius=.12,
                               color=AID_OPERATION,stroke_width=1.2).move_to([0,.1,0])
        label=jp(title,23).move_to([-4.85,2.02,0],aligned_edge=LEFT)
        self.add(*header,frame,label)
        return saved

    def restore_body(self,saved):
        self.clear()
        self.add(*saved)

    def least_squares_recap(self):
        saved=self.body_card('復習: 3.1 最小二乗')
        # Same three inputs, Gaussian bases and column colors as 3.1 matrix().
        xx=np.array([.2,.5,.8])
        phi=np.column_stack([np.ones(3),np.exp(-.5*((xx[:,None]-[.25,.75])/.18)**2)])
        weights=np.array([.2,.8,-.6])
        def table(values,x,color=WHITE):
            return DecimalMatrix(np.asarray(values),h_buff=.72,v_buff=.5,
                element_to_mobject_config={'num_decimal_places':2,'font_size':23}).set_color(color).move_to([x,.15,0])
        inputs=table(phi,-3.25);inputs.set_column_colors(BLUE_CLS,YELLOW_W,PURPLE_B)
        wm=table(weights[:,None],-.9)
        ym=table((phi@weights)[:,None],1.05,RED_LINE)
        target=table(np.array([1.,.3,-.4])[:,None],3.6,BLUE_CLS)
        labels=VGroup(*[jp(t,20).move_to([x,1.35,0]) for t,x in
            [('基底の値',-3.25),('重み',-.9),('予測',1.05),('目標の数',3.6)]])
        equal=tex('=',25).move_to([.1,.15,0])
        arrow=Arrow([1.85,.15,0],[2.65,.15,0],buff=0,color=RED_LINE)
        formula=tex(r'\mathbf y=\Phi\mathbf w',30,RED_LINE).move_to([0,-1.35,0])
        initial=VGroup(inputs,wm,ym,target,labels,equal,arrow,formula)
        # Classification uses raw coordinates with a leading 1, as in this scene.
        X=np.array([[1,-1,-.5],[1,1,.5],[1,.5,1]])
        T=np.array([[1,0],[0,1],[0,1]])
        W=np.linalg.solve(X,T)
        xi=table(X,-3.6);xi.set_column_colors(BLUE_CLS,YELLOW_W,PURPLE_B)
        wi=table(W,-1.05);yi=table(X@W,1.45,RED_LINE)
        ti=Matrix(T,h_buff=.55,v_buff=.5,element_to_mobject_config={'font_size':26}).move_to([3.8,.15,0])
        ti.get_rows()[0].set_color(BLUE_CLS)
        for row in ti.get_rows()[1:]:row.set_color(ORANGE_CLS)
        lab2=VGroup(*[jp(t,20).move_to([x,1.35,0]) for t,x in
            [('拡張入力',-3.6),('重み',-1.05),('スコア',1.45),('クラスの目標',3.8)]])
        destination=VGroup(xi,wi,yi,ti,lab2,tex('=',25).move_to([.25,.15,0]),
            tex(r'Y=\widetilde XW\qquad C_1:(1,0),\ C_2:(0,1)',29).move_to([0,-1.35,0]))
        self.add(jp('説明用の3点',18,MUTED).move_to([3.75,2.02,0]))
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('R3.1 rows weights predictions',a*.3,lambda:FadeIn(initial)),
            ('R3.1 prediction toward target',a*.7,lambda:AnimationGroup(
                Indicate(inputs.get_rows()[0],color=YELLOW_W),Indicate(arrow,color=RED_LINE))),
            ('R3.1 replace regression example',b*.12,lambda:FadeOut(initial)),
            ('R3.1 class target columns',b*.28,lambda:FadeIn(destination)),
            ('R3.1 class coding',b*.6,lambda:LaggedStart(
                *[Circumscribe(row,color=YELLOW_W) for row in ti.get_rows()],lag_ratio=.4)),
        ])
        self.restore_body(saved)

    def gaussian_recap(self):
        saved=self.body_card('復習: 2.3 方向ごとの広がり')
        ax=Axes(x_range=[-4.6,4.6,2],y_range=[-3.2,3.2,2],x_length=4.14,y_length=2.88,
                tips=False,axis_config={'color':MUTED,'include_ticks':False}).move_to([-2.45,-.15,0])
        self.add(ax,tex('x_1',23,MUTED).next_to(ax.x_axis,RIGHT,buff=.1),
                 tex('x_2',23,MUTED).next_to(ax.y_axis,UP,buff=.08))
        # Reproduce the cloud, green contours and green/purple principal axes.
        points=np.random.default_rng(2303).normal(size=(100,2))
        stretch=ValueTracker(0)
        rotation=np.array([[np.cos(.6),-np.sin(.6)],[np.sin(.6),np.cos(.6)]])
        mat=lambda:rotation@np.diag([1+.6*stretch.get_value(),1-.35*stretch.get_value()])
        dots=always_redraw(lambda:VGroup(*[Dot(ax.c2p(*p),radius=.022,color=BLUE_CLS) for p in points@mat().T]))
        angles=np.linspace(0,TAU,121)
        circle=np.column_stack([np.cos(angles),np.sin(angles)])
        rings=always_redraw(lambda:VGroup(*[
            VMobject(color=GREEN_CLS,stroke_width=2).set_points_as_corners(
                [ax.c2p(*p) for p in r*circle@mat().T]) for r in [1,2]]))
        principal=always_redraw(lambda:VGroup(*[Arrow(ax.c2p(0,0),ax.c2p(*mat()[:,i]),
            buff=0,color=c,stroke_width=4) for i,c in enumerate([GREEN_CLS,PURPLE_B])]))
        label=VGroup(jp('同じ点群でも、方向で幅が変わる',21),
            tex(r'\sqrt{\lambda_1}=1.6',28,GREEN_CLS),
            tex(r'\sqrt{\lambda_2}=0.65',28,PURPLE_B)).arrange(DOWN,buff=.3).move_to([2.3,.6,0])
        strip=NumberLine(x_range=[-3,3,1],length=3.8,include_ticks=False,color=MUTED).move_to([2.3,-1.15,0])
        # Two class clouds are a new schematic example, with original class colors.
        group_points=[np.array([[-.8,-.25],[-.6,.2],[-1.1,0]]),
                      np.array([[.6,-.15],[.9,.3],[1.2,.05]])]
        sources=VGroup(*[Dot([2.3+.65*p[0],-.15+p[1],0],radius=.06,color=COLORS[k])
                         for k,g in enumerate(group_points) for p in g])
        shadows=VGroup(*[Dot(strip.n2p(p[0])+UP*(.08 if k==0 else -.08),radius=.05,color=COLORS[k])
                         for k,g in enumerate(group_points) for p in g])
        axislabel=tex(r'y=w^Tx',26,YELLOW_W).move_to([2.3,-1.7,0])
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.add(dots,rings,principal)
        self.beat(phases=[
            ('R2.3 directional widths',a*.72,lambda:stretch.animate.set_value(1)),
            ('R2.3 principal axes',a*.28,lambda:FadeIn(label)),
            ('R2.3 bridge to two classes',b*.22,lambda:FadeOut(label)),
            ('R2.3 two class clouds',b*.22,lambda:AnimationGroup(FadeIn(sources),FadeIn(strip),FadeIn(axislabel))),
            ('R2.3 projection to one axis',b*.56,lambda:TransformFromCopy(sources,shadows)),
        ])
        self.restore_body(saved)

    def scatter_aid(self):
        saved=self.body_card('補足: ずれの外積と散布行列')
        self.add(jp('説明用の例',18,MUTED).move_to([3.85,2.02,0]))
        col=Matrix([[2],[1]],v_buff=.6,element_to_mobject_config={'font_size':32}).set_color(AID_INPUT).move_to([-3.4,.55,0])
        row=Matrix([[2,1]],h_buff=.65,element_to_mobject_config={'font_size':32}).set_color(AID_COMPARE).move_to([-1.65,.55,0])
        prod=Matrix([[4,2],[2,1]],v_buff=.6,h_buff=.65,element_to_mobject_config={'font_size':32}).set_color(AID_OPERATION).move_to([.85,.55,0])
        labels=VGroup(tex('r',27,AID_INPUT).next_to(col,UP,buff=.18),
                      tex('r^T',27,AID_COMPARE).next_to(row,UP,buff=.18),
                      tex('rr^T',27,AID_OPERATION).next_to(prod,UP,buff=.18))
        eq=tex('=',30).move_to([-.25,.55,0])
        self.add(col,row,eq,labels,prod.get_brackets())
        entries=prod.get_entries()
        # Four centered residuals: two points in each of two classes.
        total=Matrix([[10,2],[2,4]],h_buff=.7,v_buff=.6,
                     element_to_mobject_config={'font_size':32}).set_color(AID_RESULT).move_to([3.6,.55,0])
        total_label=tex(r'S_W=\sum_n r_nr_n^T',27,AID_RESULT).move_to([3.45,1.5,0])
        example=tex(r'C_1:\ \pm(2,1)\qquad C_2:\ \pm(1,-1)',27).move_to([0,-.95,0])
        residual_note=jp('各クラスの平均からのずれ',20,MUTED).move_to([0,-1.55,0])
        link=Arrow([1.65,.55,0],[2.55,.55,0],buff=0,color=AID_OPERATION)
        take=tex(r'w=(1,0)^T:\quad w^TS_Ww=2^2+(-2)^2+1^2+(-1)^2=10',
                 29,AID_RESULT).move_to([0,-.95,0])
        no_division=jp('点の数で割らない二乗偏差和',23).move_to([0,-1.65,0])
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('V08d multiply four cells',a*.58,lambda:LaggedStart(*[
                AnimationGroup(Indicate(col.get_entries()[i//2],color=AID_OPERATION),
                               Indicate(row.get_entries()[i%2],color=AID_OPERATION),FadeIn(entries[i]))
                for i in range(4)],lag_ratio=.6)),
            ('V08d sum centered residuals',a*.42,lambda:AnimationGroup(
                FadeIn(total),FadeIn(total_label),GrowArrow(link),FadeIn(example),FadeIn(residual_note))),
            ('V08d select horizontal scatter',b*.2,lambda:AnimationGroup(FadeOut(example),FadeOut(residual_note))),
            ('V08d horizontal sum of squares',b*.8,lambda:AnimationGroup(
                FadeIn(take),Circumscribe(total.get_entries()[0],color=AID_OPERATION))),
            ('V08d no division by count',c,lambda:FadeIn(no_division)),
        ])
        self.restore_body(saved)

    def relation(self):
        self.hint('同じ自作データ：青30点、橙22点、合計52点')
        ax=self.axes(xr=(-3.4,3.4,1),yr=(-2,3.2,1),width=5.3,height=4.05,center=(-2.75,.2,0))
        dots=cloud(ax,FISH,.04);self.add(dots)
        label=tex(r'N_1=30,\ N_2=22,\ N=52',29).move_to([3.3,1.6,0])
        self.beat(Write(label),*[Indicate(g,color=COLORS[k],scale_factor=1.03) for k,g in enumerate(dots)])
        f=self.bottom(r't_{C_1}=N/N_1\qquad t_{C_2}=-N/N_2')
        vals=VGroup(tex(f't_1={CODE[0]:.3f}',32,BLUE_CLS),tex(f't_2={CODE[-1]:.3f}',32,ORANGE_CLS)).arrange(DOWN,buff=.35).move_to([3.3,.45,0])
        self.beat(Write(f),FadeIn(vals))
        mean=Dot(ax.c2p(*FISH_MEAN),radius=.085,color=WHITE)
        bline=boundary(ax,CODE_W)
        f2=self.bottom(r'\sum_n t_n=0\qquad w_0=-w^Tm')
        self.beat(ReplacementTransform(f,f2),Create(bline),FadeIn(mean))
        angle=ValueTracker(.65)
        arrow=always_redraw(lambda:Arrow(ax.c2p(*FISH_MEAN),ax.c2p(*(FISH_MEAN+1.5*direction(angle.get_value()))),buff=0,color=YELLOW_W))
        # Opposite orientations, same line of projection.
        lsarrow=Arrow(ax.c2p(*FISH_MEAN),ax.c2p(*(FISH_MEAN+unit(CODE_W[1:])*1.2)),buff=0,color=PURPLE_B)
        self.add(arrow,lsarrow)
        self.beat(angle.animate.set_value(THETA_FISHER),Write(tex(r'|\cos\angle(w_{LS},w_F)|=1',27).move_to([3.3,-.9,0])))
        f3=self.bottom(r'w_{LS}\propto S_W^{-1}(m_1-m_2)\qquad y=w^T(x-m)',size=29)
        self.beat(ReplacementTransform(f2,f3),Circumscribe(mean,color=GREEN_CLS),ShowPassingFlash(bline.copy(),time_width=.4))

    def multi_fisher(self):
        self.hint('4次元の自作データ → クラス平均の張る平面を模式表示')
        ax=self.axes(xr=(-2,2,1),yr=(-1.5,1.5,1),width=5.4,height=4.05,center=(-2.5,.2,0),labels=('u','v'))
        pts=CENTERS[:,:2];m=pts.mean(0);dots=VGroup(*[Dot(ax.c2p(*p),color=COLORS[k],radius=.13) for k,p in enumerate(pts)])
        triangle=Polygon(*[ax.c2p(*p) for p in pts],color=YELLOW_W,fill_opacity=.1)
        self.add(dots);self.beat(Create(triangle))
        arrows=VGroup(*[Arrow(ax.c2p(*m),ax.c2p(*p),buff=.1,color=COLORS[k]) for k,p in enumerate(pts)])
        f=self.bottom(r'\sum_kN_k(m_k-m)=0')
        self.beat(Create(arrows),Write(f))
        f2=self.bottom(r'S_T=S_W+S_B\qquad S_B=\sum_kN_k(m_k-m)(m_k-m)^T',size=28)
        pieces=VGroup(jp('集団内の広がり',24,PURPLE_B),jp('集団間の隔たり',24,YELLOW_W),tex('y=W^Tx',32)).arrange(DOWN,buff=.4).move_to([3.6,1,0])
        self.beat(ReplacementTransform(f,f2),FadeIn(pieces),Indicate(triangle,scale_factor=1.05))
        criterion_eq=tex(r'J(W)=\mathrm{Tr}[(W^TS_WW)^{-1}(W^TS_BW)]',29).move_to([0,-2.52,0])
        self.remove(f2)
        eigax=Axes(x_range=[0,5,1],y_range=[0,45,10],x_length=3.7,y_length=1.5,tips=False,axis_config={'include_ticks':False,'color':MUTED}).move_to([3.5,-1.15,0])
        bars=VGroup(*[Rectangle(width=.45,height=max(.015,float(e)/45*1.5),fill_color=GREEN_CLS,fill_opacity=.8,stroke_width=0).move_to(eigax.c2p(i+1,0),aligned_edge=DOWN) for i,e in enumerate(EIG)])
        eiglabels=VGroup(*[tex(f'{max(0,e):.1f}',19).next_to(bar,UP,buff=.1) for e,bar in zip(EIG,bars)])
        self.add(eigax);self.beat(Write(criterion_eq),GrowFromEdge(bars,DOWN),FadeIn(eiglabels))
        count=tex(r'\operatorname{rank}(S_B)\leq K-1=2',31,YELLOW_W).move_to([0,-2.52,0])
        self.beat(ReplacementTransform(criterion_eq,count),Circumscribe(VGroup(bars[0],bars[1]),color=YELLOW_W))

    def perceptron_scene(self):
        self.hint('青：+1　橙：−1　黄色の輪：今回更新する点')
        ax=self.axes(xr=(-1.65,1.65,.5),yr=(-1.45,1.45,.5),width=4.78,height=4.2,center=(-2.7,.15,0))
        groups=[P_X[P_T==1],P_X[P_T==-1]];dots=cloud(ax,groups,.07)
        step=ValueTracker(0);w=lambda:interpolate_history(step.get_value())
        line=always_redraw(lambda:boundary(ax,w()));self.add(dots,line)
        error=readout(r'N_{\rm error}=',lambda:np.sum(np.where(augment(P_X)@w()>=0,1,-1)!=P_T),[3.5,1.7,0],RED_LINE,0)
        self.add(error);self.beat(ShowPassingFlash(line.copy().clear_updaters(),time_width=.5))
        f=self.bottom(r'a=w^T\phi(x),\quad f(a)=\begin{cases}+1&a\geq0\\-1&a<0\end{cases}',size=30)
        phi=tex(r'\phi(x)=(1,x_1,x_2)^T',31).move_to([3.5,.8,0])
        self.beat(Write(f),Write(phi),Indicate(dots[0],color=BLUE_CLS,scale_factor=1.08))
        ring=Circle(radius=.18,color=YELLOW_W).move_to(ax.c2p(*P_X[P_HISTORY[0]['index']]))
        self.add(ring)
        update=tex(r'w\leftarrow w+\eta\phi_nt_n',31,YELLOW_W).move_to([3.5,-.2,0])
        self.add(update,tex(r'\eta=1',25,MUTED).move_to([3.5,-.85,0]))
        self.beat(step.animate.set_value(1),start_sentence=1)
        ring.move_to(ax.c2p(*P_X[P_HISTORY[1]['index']]))
        self.beat(step.animate.set_value(2),start_sentence=1)
        ep=self.bottom(r'E_P(w)=-\sum_{n\in\mathcal{M}}w^T\phi_nt_n',size=33)
        self.beat(ReplacementTransform(f,ep),Circumscribe(error,color=RED_LINE))
        gain=tex(r'\Delta(t_na_n)=\eta\|\phi_n\|^2>0',30,GREEN_CLS).move_to([3.5,-1.55,0])
        self.beat(Write(gain),Indicate(ring,scale_factor=1.15))
        ring.move_to(ax.c2p(*P_X[P_HISTORY[2]['index']]))
        self.beat(step.animate.set_value(len(P_HISTORY)))

    def limits(self):
        self.hint('収束の条件は「使う特徴空間で分離できること」')
        ax=self.axes(xr=(-1.65,1.65,.5),yr=(-1.45,1.45,.5),width=4.78,height=4.2,center=(-2.7,.15,0))
        dots=cloud(ax,[P_X[P_T==1],P_X[P_T==-1]],.07);self.add(dots)
        offset=ValueTracker(P_FINAL[0]);line=always_redraw(lambda:boundary(ax,np.r_[offset.get_value(),P_FINAL[1:]]));self.add(line)
        f=jp('線形分離できる → 有限回の更新で解へ',27).move_to([0,-2.53,0])
        self.beat(Write(f),Circumscribe(dots,color=GREEN_CLS))
        self.beat(offset.animate.set_value(-.05))
        self.remove(dots,line,f)
        X=np.array([[-1,-1],[1,1],[-1,1],[1,-1]])
        xordots=cloud(ax,[X[:2],X[2:]],.1);self.add(xordots)
        angle=ValueTracker(.15);bad=always_redraw(lambda:boundary(ax,np.r_[0,direction(angle.get_value())]));self.add(bad)
        self.beat(angle.animate.set_value(2.8),Write(tex('XOR',36,YELLOW_W).move_to([3.5,1.6,0])))
        self.remove(bad)
        arrow=Arrow([.25,.1,0],[1.4,.1,0],color=MUTED)
        strip=NumberLine(x_range=[-1.5,1.5,1],length=3.8,include_numbers=False,color=MUTED).move_to([3.7,.1,0])
        target=VGroup(*[Dot(strip.n2p(p[0]*p[1])+UP*(.12 if i<2 else -.12),color=COLORS[0 if i<2 else 1],radius=.09) for i,p in enumerate(X)])
        source=VGroup(*[Dot(ax.c2p(*p),color=COLORS[0 if i<2 else 1],radius=.09) for i,p in enumerate(X)])
        self.add(strip,arrow,source,tex(r'z=x_1x_2',32).move_to([3.7,.9,0]),tex('-1',24).next_to(strip.n2p(-1),DOWN,buff=.4),tex('+1',24).next_to(strip.n2p(1),DOWN,buff=.4))
        self.beat(ReplacementTransform(source,target))
        cut=Line(strip.n2p(0)+UP*.65,strip.n2p(0)+DOWN*.65,color=RED_LINE)
        f2=self.bottom(r'\phi(x)=(1,x_1,x_2,x_1x_2)^T\qquad a=x_1x_2',size=30)
        self.beat(Create(cut),Write(f2),Circumscribe(target,color=GREEN_CLS))

    def summary(self):
        self.hint('同じ境界でも、学習で動かす理由が違う')
        ax=self.axes();amount=ValueTracker(0)
        dots=always_redraw(lambda:cloud(ax,ls_data(.35*amount.get_value())));self.add(dots)
        W=lambda:least_squares(ls_data(.35*amount.get_value()))
        line=always_redraw(lambda:boundary(ax,W()[:,0]-W()[:,1]));self.add(line)
        tag=jp('最小二乗：目標の数へ近づける',27,RED_LINE).move_to([2.8,1.6,0])
        self.add(tag);self.beat(amount.animate.set_value(1))
        self.remove(line,tag)
        summary_groups=ls_data(.35)
        theta=ValueTracker(.7);u=lambda:direction(theta.get_value())
        arrow=always_redraw(lambda:Arrow(ax.c2p(*(-1.5*u())),ax.c2p(*(1.5*u())),buff=0,color=YELLOW_W))
        proj=always_redraw(lambda:VGroup(*[Dot(ax.c2p(*(u()*(p@u()))),color=COLORS[k],radius=.035) for k,g in enumerate(summary_groups) for p in g]))
        tag=jp('Fisher：影の分離をよくする',27,YELLOW_W).move_to([3.1,1.6,0]);self.add(arrow,proj,tag)
        sw,sb=scatter(summary_groups);v=unit(np.linalg.solve(sw,summary_groups[0].mean(0)-summary_groups[1].mean(0)))
        self.beat(theta.animate.set_value(np.arctan2(v[1],v[0])))
        self.remove(arrow,proj,tag,dots)
        pdots=cloud(ax,[P_X[P_T==1],P_X[P_T==-1]]);step=ValueTracker(0)
        line=always_redraw(lambda:boundary(ax,interpolate_history(step.get_value())))
        self.add(pdots,line);tag=jp('パーセプトロン：誤りで更新する',25,GREEN_CLS).move_to([3,1.6,0]);self.add(tag)
        self.beat(step.animate.set_value(3))
        question=jp('どれくらい確かな判定？',28,YELLOW_W).move_to([3.7,.4,0])
        nextf=self.bottom(r'p(x\mid C_k)\quad\longrightarrow\quad p(C_k\mid x)',size=35)
        self.beat(FadeIn(question),Write(nextf),ShowPassingFlash(line.copy().clear_updaters(),time_width=.4))
