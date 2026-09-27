"""PRML 3.1 — linked visual experiments, rendered with Manim Community."""
from pathlib import Path
import json
import numpy as np
from manim import *
from narrated_scene import NarratedScene
from visual_support import jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from basis_model import *

DATA=ManimColor('#58B5ED')
MODEL=ManimColor('#FF6B77')
BASIS=ManimColor('#77D49A')
GOLD=ManimColor('#FFE079')
PURPLE=ManimColor('#C29AFF')
ORANGE=ManimColor('#FFB45B')
MUTED=ManimColor('#A8B2C5')
COLORS=[DATA,GOLD,PURPLE,ORANGE,BASIS]
U=np.linspace(0,1,241)
GRID=design(U)


def path(points,color=MODEL,width=3):
    return VMobject().set_points_as_corners(points).set_stroke(color,width)


def curve(ax, values, color=MODEL):
    # Vectorize the coordinate conversion; all curves retain their actual values.
    values=np.asarray(values)
    origin=ax.c2p(0,0)
    pts=origin+U[:,None]*(ax.c2p(1,0)-origin)+values[:,None]*(ax.c2p(0,1)-origin)
    return path(pts,color)


def dots(ax,x=X,t=T,color=DATA):
    return VGroup(*[Dot(ax.c2p(a,b),radius=.052,color=color) for a,b in zip(x,t)])


def readout(label,getter,pos,color=WHITE,places=2):
    prefix=tex(label,25,color)
    number=DecimalNumber(getter(),num_decimal_places=places,font_size=25,color=color)
    group=VGroup(prefix,number).arrange(RIGHT,buff=.13).move_to(pos)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group


class PRML31LinearBasisFunctionModels(NarratedScene):
    def construct(self):
        self.camera.background_color='#10141F'
        self.timeline=[]
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        for i,method in enumerate([self.knobs,self.bases,self.likelihood,self.matrix,
                                  self.projection,self.sequential,self.ridge,self.lasso,self.outputs]):
            self.begin(i)
            method()
            assert self.beat_index==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        out=Path(config.media_dir)/'prml31_timeline.json'
        out.write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def axes(self,center=(-1.7,.05,0),width=7.1,height=3.3,span=1.65):
        ax=Axes(x_range=[0,1,.25],y_range=[-span,span,1],x_length=width,y_length=height,
                tips=False,axis_config={'color':MUTED,'stroke_width':1.3,'include_ticks':False}).move_to(center)
        labels=VGroup()
        for x in [0,.5,1]:labels.add(tex(str(x),18,MUTED).next_to(ax.c2p(x,-span),DOWN,buff=.12))
        for y in [-1,0,1]:labels.add(tex(str(y),18,MUTED).next_to(ax.c2p(0,y),LEFT,buff=.12))
        labels.add(tex('x',22,MUTED).next_to(ax.c2p(1,-span),RIGHT,buff=.13))
        self.add(ax,labels)
        return ax

    def equation(self,formula,color=WHITE,size=31,y=-2.55):
        new=tex(formula,size,color).move_to([0,y,0])
        if new.width>12.4: raise ValueError('Formula too wide: '+formula)
        old=self.formula
        self.formula=new
        if old is None:return FadeIn(new)
        return ReplacementTransform(old,new)

    def note(self,text,pos=(3.6,1.55,0),color=MUTED,size=22):
        return jp(text,size,color).move_to(pos)

    def slider(self,tr,lo,hi,pos,label,color=GOLD,width=3):
        line=NumberLine(x_range=[lo,hi,1],length=width,include_ticks=False,color=MUTED).move_to(pos)
        knob=Dot(line.n2p(tr.get_value()),radius=.07,color=color)
        knob.add_updater(lambda m:m.move_to(line.n2p(tr.get_value())))
        label=readout(label,tr.get_value,np.array(pos)+UP*.43,color)
        return VGroup(line,knob,label)

    def bars(self,getter,pos=(3.8,-.5,0),scale=.25):
        base=np.array(pos)
        n=len(getter())
        return always_redraw(lambda:VGroup(*[
            Rectangle(width=.22,height=max(.006,abs(w)*scale),stroke_width=0,
                      fill_color=PURPLE if w>=0 else ORANGE,fill_opacity=.9)
            .move_to(base+RIGHT*((i-(n-1)/2)*.29)+UP*w*scale/2)
            for i,w in enumerate(getter())]))

    def knobs(self):
        ax=self.axes()
        d=dots(ax)
        self.beat(LaggedStart(*[FadeIn(o) for o in d],lag_ratio=.1))
        line=curve(ax,np.polyval(np.polyfit(X,T,1),U),MUTED)
        self.beat(Create(line),self.equation(r'y=w_0+w_1x'))
        self.remove(line)
        w0,w1,w2=[ValueTracker(0) for _ in range(3)]
        g=gaussian(U,[.28,.73],.15)
        pieces=VGroup(*[always_redraw(lambda j=j:curve(ax,[w1,w2][j].get_value()*g[:,j],COLORS[j+1])) for j in range(2)])
        total=always_redraw(lambda:curve(ax,w0.get_value()+w1.get_value()*g[:,0]+w2.get_value()*g[:,1]))
        self.add(pieces,total,self.slider(w1,-1.3,1.3,(3.8,1.1,0),'w_1=',GOLD),
                 self.slider(w2,-1.3,1.3,(3.8,-.05,0),'w_2=',PURPLE),
                 self.slider(w0,-.5,.5,(3.8,-1.2,0),'w_0=',DATA))
        self.beat(w1.animate.set_value(1.2),self.equation(r'y={{w_1\phi_1(x)}}',GOLD),start_sentence=1)
        self.beat(w1.animate.set_value(-.8))
        self.beat(w1.animate.set_value(.95),w2.animate.set_value(-1.05),self.equation(r'y={{w_1\phi_1(x)}}+{{w_2\phi_2(x)}}'))
        self.formula.set_color_by_tex('w_1',GOLD).set_color_by_tex('w_2',PURPLE)
        self.beat(w0.animate.set_value(.3),self.equation(r'y=w_0+w_1\phi_1(x)+w_2\phi_2(x)'))
        self.beat(w0.animate.set_value(0),w1.animate.set_value(1.1),w2.animate.set_value(-.9),
                  self.equation(r'y(\mathbf{x},\mathbf{w})=\sum_{j=0}^{M-1}w_j\phi_j(\mathbf{x})=\mathbf{w}^T\boldsymbol\phi(\mathbf{x})'))

    def bases(self):
        ax=self.axes(center=(-1.7,.15,0),span=1.2)
        amp=ValueTracker(.4)
        polys=VGroup(curve(ax,U,DATA),curve(ax,U**2,PURPLE),curve(ax,U**3,ORANGE))
        active=always_redraw(lambda:curve(ax,amp.get_value()*U**2,GOLD))
        self.add(polys,active)
        self.beat(amp.animate.set_value(1.1),self.equation(r'\phi_j(x)=x^j',GOLD))
        self.remove(polys,active)
        mu=ValueTracker(.3);s=ValueTracker(.12)
        gauss=always_redraw(lambda:curve(ax,gaussian(U,[mu.get_value()],s.get_value())[:,0],GOLD))
        self.add(self.slider(mu,.15,.85,(3.8,.9,0),r'\mu=',GOLD),self.slider(s,.07,.3,(3.8,-.45,0),'s=',PURPLE))
        self.beat(Create(gauss),self.equation(r'\phi_j(x)=\exp\!\left[-\frac{(x-\mu_j)^2}{2s^2}\right]',GOLD))
        self.beat(mu.animate.set_value(.7))
        self.beat(s.animate.set_value(.27))
        height=DashedLine(ax.c2p(0,1),ax.c2p(1,1),color=MUTED)
        label=self.note('基底：頂点の高さ 1',(3.8,1.9,0),GOLD,20)
        self.beat(Create(height),FadeIn(label),s.animate.set_value(.13))
        self.remove(gauss,height,label)
        sig=always_redraw(lambda:curve(ax,1/(1+np.exp(-(U-mu.get_value())/s.get_value())),BASIS))
        self.add(sig)
        self.beat(mu.animate.set_value(.3),self.equation(r'\phi_j(x)=\sigma((x-\mu_j)/s),\quad\sigma(a)=\frac{1}{1+e^{-a}}',BASIS))
        self.beat(mu.animate.set_value(.6),self.equation(r'\mathbf{x}\longrightarrow\boldsymbol\phi(\mathbf{x})\longrightarrow\mathbf{w}^T\boldsymbol\phi(\mathbf{x})'))

    def likelihood(self):
        ax=self.axes(span=1.85)
        a=ValueTracker(0)
        weights=lambda:(1-a.get_value())*fit(.3)+a.get_value()*W_ML
        self.add(dots(ax),always_redraw(lambda:curve(ax,GRID@weights())))
        beta=ValueTracker(4)
        # Horizontal density profiles, t runs vertically; density is scaled horizontally.
        x0=.4;mean=(design([x0])@fit(.3)).item();ts=np.linspace(-1.6,1.6,201)
        density=lambda:np.sqrt(beta.get_value()/(2*np.pi))*np.exp(-.5*beta.get_value()*(ts-mean)**2)
        profile=always_redraw(lambda:path([ax.c2p(x0+.10*p,t) for p,t in zip(density(),ts)],BASIS))
        self.beat(Create(profile),self.equation(r't=y(x,\mathbf w)+\epsilon,\quad p(t\mid x,\mathbf w,\beta)=\mathcal N(t\mid y,\beta^{-1})',size=29))
        self.add(self.slider(beta,4,25,(3.8,1,0),r'\beta=',BASIS))
        self.beat(beta.animate.set_value(20))
        dot_t=ValueTracker(mean+.15)
        measure=always_redraw(lambda:Line(ax.c2p(x0,dot_t.get_value()),ax.c2p(x0+.10*np.sqrt(beta.get_value()/(2*np.pi))*np.exp(-.5*beta.get_value()*(dot_t.get_value()-mean)**2),dot_t.get_value()),color=GOLD,stroke_width=4))
        self.add(measure)
        self.beat(dot_t.animate.set_value(mean+.6))
        self.beat(self.equation(r'p(\mathbf t\mid\mathbf w,\beta)=\prod_{n=1}^N\mathcal N(t_n\mid\mathbf w^T\boldsymbol\phi_n,\beta^{-1})'),Indicate(dots(ax),color=GOLD))
        self.remove(profile,measure)
        self.beat(self.equation(r'\ln p=\frac N2\ln\beta-\frac N2\ln(2\pi)-\beta E_D,\quad E_D=\frac12\sum_n(t_n-y_n)^2',size=28))
        # Offset the optimum: the total squared residual decreases along this path.
        shift=ValueTracker(.65)
        self.remove(*[m for m in list(self.mobjects) if len(m.updaters)>0])
        self.add(always_redraw(lambda:curve(ax,GRID@W_ML+shift.get_value())))
        res=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,y+shift.get_value()),color=GOLD,stroke_width=2) for x,t,y in zip(X,T,PHI@W_ML)]))
        squares=always_redraw(lambda:VGroup(*[Square(side_length=max(.005,abs(t-y-shift.get_value())*.4),fill_opacity=.55,fill_color=GOLD,stroke_width=1,color=GOLD).move_to([2.8+(i%4)*.58,-.4-(i//4)*.6,0]) for i,(t,y) in enumerate(zip(T,PHI@W_ML))]))
        self.add(res,squares,self.note('残差²：共通縮尺',(3.8,.15,0),GOLD,19))
        self.beat(shift.animate.set_value(0),self.equation(r'E_D(\mathbf w)=\frac12\sum_n(t_n-\mathbf w^T\boldsymbol\phi_n)^2',GOLD))
        variance=2*error(W_ML)/len(X)
        self.beat(self.equation(r'\beta_{\rm ML}^{-1}=\frac1N\sum_n(t_n-y_n)^2='+f'{variance:.4f}'),Indicate(squares.copy().clear_updaters(),scale_factor=1.08))

    def matrix(self):
        ax=self.axes(center=(-4,.1,0),width=4,height=2.7,span=1.2)
        xx=np.array([.2,.5,.8]);centers=np.array([.25,.75]);small=design(xx,centers,.18)
        for j,c in enumerate([GOLD,PURPLE]):self.add(curve(ax,gaussian(U,centers,.18)[:,j],c))
        cursor=ValueTracker(.2)
        guide=always_redraw(lambda:DashedLine(ax.c2p(cursor.get_value(),-1),ax.c2p(cursor.get_value(),1),color=DATA))
        self.add(guide)
        matrix=DecimalMatrix(small,element_to_mobject_config={'num_decimal_places':2,'font_size':30},h_buff=1.15,v_buff=.7).move_to([1.3,.65,0])
        matrix.set_column_colors(DATA,GOLD,PURPLE)
        rows=matrix.get_rows();cols=matrix.get_columns()
        self.add(matrix.get_brackets(),rows[0])
        self.beat(Indicate(rows[0]),self.equation(r'\phi(x_n)=[1,\phi_1(x_n),\phi_2(x_n)]^T'))
        self.beat(cursor.animate.set_value(.8),LaggedStart(FadeIn(rows[1]),FadeIn(rows[2]),lag_ratio=.4),self.equation(r'\Phi_{nj}=\phi_j(x_n),\qquad\Phi\in\mathbb R^{N\times M}'))
        w=np.array([.2,.8,-.6]);y=small@w
        wm=DecimalMatrix(w[:,None],element_to_mobject_config={'num_decimal_places':2,'font_size':27},v_buff=.7).move_to([3.65,.65,0])
        ym=DecimalMatrix(y[:,None],element_to_mobject_config={'num_decimal_places':2,'font_size':27},v_buff=.7).move_to([5.65,.65,0])
        self.add(tex('=',28).move_to([4.6,.65,0]))
        self.beat(FadeIn(wm),FadeIn(ym),self.equation(r'\mathbf y=\Phi\mathbf w',MODEL))
        self.beat(self.equation(r'\Phi^T(\Phi\mathbf w-\mathbf t)=0\quad\Longrightarrow\quad\Phi^T\Phi\mathbf w=\Phi^T\mathbf t'),Indicate(cols[1],color=GOLD))
        self.beat(self.equation(r'\mathbf w_{\rm ML}=(\Phi^T\Phi)^{-1}\Phi^T\mathbf t=\Phi^\dagger\mathbf t'),FadeIn(self.note('逆行列の式：列が独立',(1.5,1.95,0),MUTED,22)))
        # Explicitly label the temporary degeneracy experiment, without changing the stored data.
        duplicate=cols[1].copy().move_to(cols[2])
        self.beat(Transform(cols[2],duplicate),FadeIn(self.note('重なる列 → SVD で扱う',(1.5,-1.2,0),GOLD,22)))
        self.beat(self.equation(r'w_0=\bar t-\sum_{j=1}^{M-1}w_j\bar\phi_j'),Indicate(cols[0],color=DATA))

    def projection(self):
        ax=Axes(x_range=[0,3.5,1],y_range=[0,3.5,1],x_length=4.1,y_length=4.1,tips=False,
                axis_config={'color':MUTED,'include_numbers':True,'font_size':20}).move_to([-2.6,.1,0])
        self.add(ax,tex(r't_1,\ y_1',23).next_to(ax,DOWN,buff=.15),tex(r't_2,\ y_2',23).next_to(ax,LEFT,buff=.1))
        target=Arrow(ax.c2p(0,0),ax.c2p(1,3),buff=0,color=DATA)
        self.beat(GrowArrow(target),self.equation(r'\mathbf t=(1,3)^T',DATA))
        w=ValueTracker(.65)
        pred=always_redraw(lambda:Arrow(ax.c2p(0,0),ax.c2p(w.get_value(),w.get_value()),buff=0,color=MODEL))
        span=Line(ax.c2p(0,0),ax.c2p(3.4,3.4),color=BASIS)
        self.add(pred)
        self.beat(Create(span),w.animate.set_value(3),self.equation(r'\Phi=(1,1)^T,\quad\mathbf y=w_0(1,1)^T'))
        residual=always_redraw(lambda:Line(ax.c2p(w.get_value(),w.get_value()),ax.c2p(1,3),color=GOLD,stroke_width=4))
        self.add(residual,readout('E_D=',lambda:((1-w.get_value())**2+(3-w.get_value())**2)/2,(3.2,.9,0),GOLD))
        self.beat(w.animate.set_value(.9),self.equation(r'E_D=\frac12\|\mathbf t-\mathbf y\|^2',GOLD))
        self.beat(w.animate.set_value(2),end_sentence=1)
        corner=RightAngle(Line(ax.c2p(2,2),ax.c2p(3,3)),Line(ax.c2p(2,2),ax.c2p(1,3)),length=.22,color=WHITE)
        self.beat(Create(corner),self.equation(r'\mathbf t-\mathbf y=(-1,1)^T,\quad(1,1)\cdot(-1,1)=0'))
        self.beat(self.equation(r'\mathbf y=\Phi\Phi^\dagger\mathbf t,\qquad\Phi^T(\mathbf t-\mathbf y)=0'),Indicate(span,color=BASIS))
        # A small conceptual plane lives away from the numeric 2D axes.
        plane=Polygon([1.5,-.8,0],[4.6,-.8,0],[5.4,.4,0],[2.3,.4,0],color=BASIS,fill_opacity=.15)
        self.beat(FadeIn(plane),FadeIn(self.note('列の張る空間',(3.5,-1.2,0),BASIS,22)),self.equation(r'\dim S=\operatorname{rank}\Phi\le M'))

    def sequential(self):
        ax=self.axes()
        step=ValueTracker(0)
        weights=lambda:interpolate_steps(step.get_value())
        live=always_redraw(lambda:curve(ax,GRID@weights()))
        bars=self.bars(weights,scale=.7)
        first=ORDER[0];second=ORDER[1]
        self.add(live,bars,self.note('各基底の重み',(3.8,1.3,0),PURPLE))
        pt=dots(ax,X[[first]],T[[first]])
        residual=lambda i:always_redraw(lambda:Line(ax.c2p(X[i],T[i]),ax.c2p(X[i],float(PHI[i]@weights())),color=GOLD,stroke_width=3))
        r=residual(first)
        self.beat(FadeIn(pt),Create(r),self.equation(r'r_n=t_n-\mathbf w^T\boldsymbol\phi_n',GOLD))
        self.beat(step.animate.set_value(1))
        self.remove(r);r=residual(second);self.add(r,dots(ax,X[[second]],T[[second]]))
        self.beat(step.animate.set_value(2))
        response=curve(ax,gaussian(U,[CENTERS[1]],SCALE)[:,0],BASIS)
        self.beat(Create(response),self.equation(r'\mathbf w\leftarrow\mathbf w+\eta\,{{(t_n-\mathbf w^T\boldsymbol\phi_n)}}\,{{\boldsymbol\phi_n}}'))
        self.formula.set_color_by_tex('(t_n',GOLD).set_color_by_tex('{{\boldsymbol',BASIS)
        # One-step learning-rate experiment, always recomputed from the same old weights.
        eta=ValueTracker(.22);old=LMS[2];i=ORDER[2]
        self.remove(live,r,bars,response)
        experiment=always_redraw(lambda:curve(ax,GRID@lms_step(old,i,eta.get_value())))
        expbars=self.bars(lambda:lms_step(old,i,eta.get_value()),scale=.7)
        self.add(experiment,expbars,dots(ax,X[[i]],T[[i]]),self.slider(eta,.05,.45,(3.8,-1.8,0),r'\eta=',GOLD))
        self.beat(eta.animate.set_value(.45),self.equation(r'\Delta\mathbf w=\eta\,r_n\boldsymbol\phi_n'))
        self.remove(experiment,expbars)
        self.add(live,bars)
        arriving=always_redraw(lambda:dots(ax,X[ORDER[:max(3,int(step.get_value()))]],T[ORDER[:max(3,int(step.get_value()))]]))
        self.add(arriving)
        self.beat(step.animate.set_value(12),self.equation(r'\mathbf w^{(\tau+1)}=\mathbf w^{(\tau)}-\eta\nabla E_n'))
        self.beat(Indicate(bars.copy().clear_updaters(),color=PURPLE),self.equation(r'E_n=\frac12(t_n-\mathbf w^T\boldsymbol\phi_n)^2,\qquad\mathrm{LMS}'))

    def ridge(self):
        ax=self.axes(span=1.8)
        loglam=ValueTracker(-6)
        cache={'v':None,'w':None}
        def weights():
            v=loglam.get_value()
            if cache['v']!=v:cache.update(v=v,w=fit(10**v))
            return cache['w']
        self.add(dots(ax),always_redraw(lambda:curve(ax,GRID@weights())))
        self.add(self.slider(loglam,-6,2,(0,2.2,0),r'\log_{10}\lambda=',PURPLE,width=5))
        self.beat(loglam.animate.set_value(-5),self.equation(r'y(x)=\mathbf w^T\boldsymbol\phi(x)'))
        bars=self.bars(weights,scale=.12)
        self.add(self.note('重み：共通の線形縮尺',(3.8,1.55,0),PURPLE,19),readout(r'\|\mathbf w\|=',lambda:np.linalg.norm(weights()),(3.7,.95,0),PURPLE))
        self.beat(FadeIn(bars))
        self.beat(self.equation(r'{{E_D(\mathbf w)}}+{{\frac\lambda2\mathbf w^T\mathbf w}}'))
        self.formula.set_color_by_tex('E_D',GOLD).set_color_by_tex('lambda',PURPLE)
        self.beat(loglam.animate.set_value(-1))
        self.beat(loglam.animate.set_value(2))
        self.beat(loglam.animate.set_value(-5))
        self.beat(loglam.animate.set_value(-1),self.equation(r'\mathbf w=(\Phi^T\Phi+\lambda I)^{-1}\Phi^T\mathbf t',size=30))

    def lasso(self):
        ax=Axes(x_range=[-1.3,3,1],y_range=[-1.5,1.8,1],x_length=5.16,y_length=3.96,
                tips=False,axis_config={'color':MUTED,'include_numbers':True,'font_size':18}).move_to([-1.5,.05,0])
        self.add(ax,tex('w_1',23).next_to(ax.x_axis,RIGHT),tex('w_2',23).next_to(ax.y_axis,UP))
        center=CONSTRAINT_TARGET
        radius=ValueTracker(.12)
        circle=lambda r:ax.plot_parametric_curve(lambda a:np.r_[center+r*np.array([np.cos(a),np.sin(a)]),0],t_range=[0,TAU,.05],color=DATA)
        level=always_redraw(lambda:circle(radius.get_value()))
        point=Dot(ax.c2p(*center),color=DATA)
        self.beat(FadeIn(point),Create(level),self.equation(r'E_D=\frac12\|\mathbf w-(1.8,0.5)^T\|^2',DATA))
        q=ValueTracker(2)
        boundary=always_redraw(lambda:path([ax.c2p(*v) for v in q_boundary(q.get_value())],PURPLE,4))
        self.add(boundary,self.slider(q,.5,4,(4.3,.8,0),'q=',PURPLE,width=2.3))
        self.beat(radius.animate.set_value(np.linalg.norm(center)-1),self.equation(r'w_1^2+w_2^2\le1',PURPLE))
        opt=Dot(ax.c2p(*L2_POINT),radius=.085,color=MODEL)
        self.beat(FadeIn(opt),self.equation(r'\mathbf w^*='+r'('+f'{L2_POINT[0]:.3f},{L2_POINT[1]:.3f}'+r')^T',MODEL))
        self.beat(q.animate.set_value(1),radius.animate.set_value(np.linalg.norm(center-L1_POINT)),opt.animate.move_to(ax.c2p(*L1_POINT)),self.equation(r'|w_1|+|w_2|\le1',PURPLE))
        self.beat(Indicate(opt,color=GOLD),self.equation(r'\mathbf w^*=(1,0)^T\quad\Longrightarrow\quad w_2\phi_2(x)=0',MODEL))
        self.remove(level,opt)
        self.beat(phases=[('q2',self.sentence_duration(0)/2,lambda:q.animate.set_value(2)),
                          ('q1',self.sentence_duration(0)/2,lambda:q.animate.set_value(1)),
                          ('q4',self.sentence_duration(1)/2,lambda:q.animate.set_value(4)),
                          ('q05',self.sentence_duration(1)/2,lambda:q.animate.set_value(.5)),
                          ('nonconvex',self.sentence_duration(2),lambda:Create(Line(ax.c2p(1,0),ax.c2p(0,1),color=GOLD)))])
        self.beat(q.animate.set_value(1),self.equation(r'E_D+\frac\lambda2\sum_j|w_j|^q\quad\longleftrightarrow\quad\sum_j|w_j|^q\le c',size=29))

    def outputs(self):
        ax=self.axes()
        p=ValueTracker(0);change=ValueTracker(0)
        tmat=np.column_stack([T,T2]);wm=fit(t=tmat)
        first=lambda:GRID@wm[:,0]+change.get_value()
        self.beat(FadeIn(dots(ax)),FadeIn(dots(ax,t=T2,color=ORANGE)),self.equation(r'\mathbf t_n=(t_{n1},t_{n2})^T'))
        basis_curves=VGroup(*[curve(ax,GRID[:,j],COLORS[j%5]).set_opacity(.5) for j in range(1,9)])
        self.beat(Create(basis_curves),self.equation(r'\mathbf x\longrightarrow\boldsymbol\phi(\mathbf x)',BASIS))
        self.remove(basis_curves)
        c1=always_redraw(lambda:curve(ax,p.get_value()*first(),DATA));c2=always_redraw(lambda:curve(ax,p.get_value()*(GRID@wm[:,1]),ORANGE))
        self.add(c1,c2)
        label=VGroup(tex(r'\mathbf W=[\mathbf w_1\ \mathbf w_2]',29),jp('同じ基底・別の重み',21,MUTED)).arrange(DOWN,buff=.3).move_to([3.8,.65,0])
        self.beat(p.animate.set_value(1),FadeIn(label),self.equation(r'\mathbf y(\mathbf x)=\mathbf W^T\boldsymbol\phi(\mathbf x)'))
        self.beat(self.equation(r'p(\mathbf t\mid\mathbf x,\mathbf W,\beta)=\mathcal N(\mathbf t\mid\mathbf W^T\boldsymbol\phi,\beta^{-1}I)',size=29),Indicate(label,color=GOLD))
        self.beat(self.equation(r'\mathbf W_{\rm ML}=\Phi^\dagger\mathbf T,\qquad\mathbf w_k=\Phi^\dagger\mathbf t_k'),Indicate(label,color=BASIS))
        # Shift the first targets and refit: the constant basis makes the solution
        # exactly a shift of the first prediction; the second column is unchanged.
        live_dots=always_redraw(lambda:dots(ax,t=T+change.get_value()))
        for m in list(self.mobjects):
            if isinstance(m,VGroup) and len(m)==len(X) and all(isinstance(z,Dot) for z in m) and m[0].get_color()==DATA:self.remove(m)
        self.add(live_dots)
        self.beat(change.animate.set_value(.3))
        self.beat(change.animate.set_value(0),self.equation(r'\mathbf x\ \longrightarrow\ \boldsymbol\phi(\mathbf x)\ \longrightarrow\ \mathbf w^T\boldsymbol\phi(\mathbf x)',BASIS))
