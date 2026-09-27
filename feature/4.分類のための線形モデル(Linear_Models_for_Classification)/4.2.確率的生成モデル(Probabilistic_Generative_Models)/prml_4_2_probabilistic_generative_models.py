"""PRML 4.2: one evolving experiment per idea, implemented in Manim CE."""
import numpy as np
from manim import *
from scene_support import NarratedScene
from caption_layout import jp, tex
from generative_model import (MEANS,COV,OTHER_COV,RED_POINTS,BLUE_POINTS,BINARY_MU,
    sigmoid,normal1,posterior,linear_params,covariances,fit,contaminated,
    binary_scores,softmax,decision_paths)

RED_CLS=ManimColor('#FF6B77'); BLUE_CLS=ManimColor('#58B5ED'); GREEN_CLS=ManimColor('#77D49A')
GOLD=ManimColor('#FFE079'); ORANGE_CLS=ManimColor('#FFB45B'); PURPLE_CLS=ManimColor('#C29AFF'); MUTED=ManimColor('#A8B2C5')
COLORS=[RED_CLS,BLUE_CLS,GREEN_CLS]


def path(ax, points, color=GOLD, width=3):
    points=np.asarray(points); origin=ax.c2p(0,0)
    p=origin+points[:,0,None]*(ax.c2p(1,0)-origin)+points[:,1,None]*(ax.c2p(0,1)-origin)
    return VMobject().set_points_as_corners(p).set_stroke(color,width)


def contour(ax,mean,cov,color,levels=(.8,1.6)):
    theta=np.linspace(0,TAU,81); unit=np.array([np.cos(theta),np.sin(theta)])
    return VGroup(*[path(ax,(mean[:,None]+r*np.linalg.cholesky(cov)@unit).T,color,2) for r in levels])


def readout(label,get,pos,color=WHITE,places=2,size=26):
    number=DecimalNumber(float(get()),num_decimal_places=places,font_size=size,color=color)
    group=VGroup(tex(label,size,color),number).arrange(RIGHT,buff=.14).move_to(pos)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(float(get())).move_to(anchor,aligned_edge=LEFT))
    return group


class PRML42ProbabilisticGenerativeModels(NarratedScene):
    def scenes(self):
        return [self.bayes,self.logit,self.shared,self.priors,self.multiclass,self.learning,self.outlier,self.discrete,self.family]

    def axes(self, xr=(-3.5,3.5,1),yr=(-2.5,2.5,1),pos=(-1.15,.15,0),width=8.2,height=3.55,labels=('x_1','x_2')):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
            axis_config={'color':MUTED,'stroke_width':1.1,'include_ticks':False}).move_to(pos)
        marks=VGroup()
        for v in [xr[0],0,xr[1]]:
            marks.add(tex(f'{v:g}',17,MUTED).move_to(ax.c2p(v,yr[0])+DOWN*.23))
        for v in [yr[0],yr[1]]:
            marks.add(tex(f'{v:g}',17,MUTED).next_to(ax.c2p(xr[0],v),LEFT,buff=.12))
        marks.add(tex(labels[0],24).next_to(ax.c2p(xr[1],yr[0]),RIGHT,buff=.15),tex(labels[1],24).next_to(ax.c2p(xr[0],yr[1]),UP,buff=.12))
        self.add(ax,marks);ax.marks=marks;return ax

    def note(self,text,color=MUTED):
        m=jp(text,21,color).move_to([0,2.65,0]);self.add(m);return m

    def formula(self,*parts,size=30,y=-2.45):
        m=MathTex(*parts,font_size=size).move_to([0,y,0])
        if m.width>12.5: m.scale_to_fit_width(12.5)
        self.add(m);return m

    def equation(self,old,*parts,size=32,y=-2.45,colors=None):
        m=MathTex(*parts,font_size=size).move_to([0,y,0])
        if m.width>12.5:m.scale_to_fit_width(12.5)
        for i,c in (colors or {}).items():m[i].set_color(c)
        duration=self.beat_cues()[-1]['end']
        self.beat(phases=[('equation transition',.7,lambda:ReplacementTransform(old,m)),
            ('explain equation',duration-.7,lambda:Circumscribe(m,color=MUTED,buff=.12))])
        return m

    def slider(self,tr,lo,hi,pos,label,color=ORANGE_CLS,width=2.7):
        rail=NumberLine(x_range=[lo,hi,hi-lo],length=width,include_ticks=False,color=MUTED).move_to(pos)
        dot=Dot(radius=.075,color=color).add_updater(lambda m:m.move_to(rail.n2p(tr.get_value())))
        num=readout(label+'=',tr.get_value,np.array(pos)+UP*.4,color)
        ends=VGroup(tex(f'{lo:g}',17,MUTED).next_to(rail,LEFT,buff=.12),tex(f'{hi:g}',17,MUTED).next_to(rail,RIGHT,buff=.12))
        g=VGroup(rail,dot,num,ends);self.add(g);return g

    def dots(self,ax,red=RED_POINTS,blue=BLUE_POINTS):
        return VGroup(*[Dot(ax.c2p(*p),radius=.044,color=c) for a,c in [(red,RED_CLS),(blue,BLUE_CLS)] for p in a])

    def ellipses(self,ax,means=MEANS[:2],covs=None):
        return VGroup(*[contour(ax,m,c,col) for m,c,col in zip(means,covs or [COV]*len(means),COLORS)])

    def field(self,ax,means=lambda:MEANS[:2],covs=lambda:[COV,COV],priors=lambda:[.5,.5]):
        xs=np.linspace(ax.x_range[0],ax.x_range[1],160);ys=np.linspace(ax.y_range[1],ax.y_range[0],100)
        xx,yy=np.meshgrid(xs,ys);grid=np.stack([xx,yy],axis=-1)
        rgb=np.array([c.to_rgb() for c in COLORS])
        def create():
            p=posterior(grid,means(),covs(),priors())
            pixel=np.uint8(np.clip(p@rgb[:p.shape[-1]]*255,0,255))
            img=ImageMobject(pixel).set_resampling_algorithm(RESAMPLING_ALGORITHMS['bilinear'])
            img.stretch_to_fit_width(ax.x_length).stretch_to_fit_height(ax.y_length)
            img.move_to((ax.c2p(ax.x_range[0],ax.y_range[0])+ax.c2p(ax.x_range[1],ax.y_range[1]))/2)
            return img.set_opacity(.38).set_z_index(-4)
        img=create();img.add_updater(lambda m:m.become(create()));return img

    def boundary(self,ax,means=lambda:MEANS[:2],covs=lambda:[COV,COV],priors=lambda:[.5,.5]):
        return always_redraw(lambda:VGroup(*[path(ax,p,GOLD,2.8) for p in decision_paths(means(),covs(),priors(),ax.x_range[:2],ax.y_range[:2],75)]))

    def probability_bars(self,get,k=2,pos=(4.7,.2,0),title='事後確率'):
        left=pos[0]-(k-1)*.43;bottom=pos[1]-.85
        g=VGroup(jp(title,22).move_to([pos[0],pos[1]+1.55,0]))
        for i in range(k):
            x=left+i*.86
            bar=Rectangle(width=.48,height=1,stroke_width=0,fill_color=COLORS[i],fill_opacity=.9)
            bar.add_updater(lambda m,i=i,x=x:m.stretch_to_fit_height(max(.006,1.7*get()[i])).move_to([x,bottom,0],aligned_edge=DOWN))
            g.add(bar,tex(f'C_{i+1}',22,COLORS[i]).move_to([x,bottom-.25,0]),readout('',lambda i=i:get()[i],[x,bottom-.65,0],COLORS[i],2,22))
        self.add(g);return g

    def bayes(self):
        ax=self.axes(yr=(0,.48,.1),height=3.2,labels=('x',r'p(x\mid C_k)'))
        x=ValueTracker(0);scale=ValueTracker(1)
        grid=np.linspace(-3.5,3.5,180)
        curves=always_redraw(lambda:VGroup(*[path(ax,np.c_[grid,scale.get_value()*normal1(grid,m)],c) for m,c in [(-1.1,RED_CLS),(1.1,BLUE_CLS)]]))
        guide=always_redraw(lambda:Line(ax.c2p(x.get_value(),0),ax.c2p(x.get_value(),.46),color=WHITE,stroke_width=1.5))
        self.add(guide);note=self.note('赤：クラス1　青：クラス2　／　自作の密度')
        self.beat(Create(curves))
        f=self.formula(r'p(x\mid C_k)');self.beat(x.animate.set_value(-1.1))
        self.remove(f);f=self.formula(r'q_k=',r'p(x\mid C_k)',r'p(C_k)');f[2].set_color(ORANGE_CLS)
        self.beat(scale.animate.set_value(.5),Transform(ax.marks[-1],tex('q_k',24).move_to(ax.marks[-1])))
        prob=lambda:np.array([normal1(x.get_value(),-1.1),normal1(x.get_value(),1.1)])/sum([normal1(x.get_value(),-1.1),normal1(x.get_value(),1.1)])
        heights=always_redraw(lambda:VGroup(*[Dot(ax.c2p(x.get_value(),.5*normal1(x.get_value(),m)),radius=.07,color=c) for m,c in [(-1.1,RED_CLS),(1.1,BLUE_CLS)]]))
        self.add(heights);bars=self.probability_bars(prob)
        self.beat(Indicate(heights[0],color=RED_CLS,scale_factor=1),Indicate(heights[1],color=BLUE_CLS,scale_factor=1),Circumscribe(bars,color=GOLD))
        self.beat(x.animate.set_value(1.1))
        f=self.equation(f,r'p(C_1\mid x)=',r'\frac{q_1}{q_1+q_2}',r'=\frac{p(x\mid C_1)p(C_1)}{\sum_jp(x\mid C_j)p(C_j)}',colors={1:GOLD})
        self.beat(x.animate.set_value(0),Circumscribe(f,color=MUTED,buff=.1))

    def logit(self):
        ax=self.axes((-5,5,1),(0,1,.5),labels=('a',r'p(C_1\mid x)'),height=3.3)
        a=ValueTracker(0);grid=np.linspace(-5,5,180)
        curve=path(ax,np.c_[grid,sigmoid(grid)],GOLD)
        dot=always_redraw(lambda:Dot(ax.c2p(a.get_value(),sigmoid(a.get_value())),radius=.075,color=WHITE))
        guide=always_redraw(lambda:DashedLine(ax.c2p(a.get_value(),0),ax.c2p(a.get_value(),sigmoid(a.get_value())),color=MUTED))
        self.add(dot,guide);self.probability_bars(lambda:np.array([sigmoid(a.get_value()),sigmoid(-a.get_value())]))
        f=self.formula(r'\frac{q_1}{q_2}=1\quad\Rightarrow\quad p(C_1\mid x)=\frac12')
        self.beat(Indicate(dot))
        self.remove(f);f=self.formula(r'a=\ln\frac{q_1}{q_2}',r'\qquad q_1/q_2=3\Rightarrow p=3/4');f[0].set_color(GOLD)
        self.beat(a.animate.set_value(np.log(3)))
        self.remove(f);f=self.formula(r'a=\ln\frac{q_1}{q_2}');f.set_color(GOLD)
        self.beat(Create(curve),a.animate.set_value(0))
        num=readout('a=',a.get_value,[4.55,2.55,0],GOLD);self.add(num)
        self.beat(a.animate.set_value(4))
        self.beat(a.animate.set_value(-4))
        self.remove(f);f=self.formula(r'\sigma(a)=\frac{1}{1+e^{-a}}',r'\qquad\sigma(-a)=1-\sigma(a)');f[0].set_color(GOLD)
        self.beat(a.animate.set_value(4),Circumscribe(f,color=MUTED,buff=.1))
        self.equation(f,r'a=\ln\frac{p}{1-p}',r'\qquad p=\sigma(a)',colors={0:GOLD})

    def shared(self):
        ax=self.axes();dots=self.dots(ax);ell=self.ellipses(ax)
        note=self.note('同じ共分散：楕円の形は共通、中心は別')
        self.beat(FadeIn(dots),Create(ell))
        f=self.formula(r'p(x\mid C_k)=\frac{\exp[-\frac12(x-\mu_k)^T\Sigma^{-1}(x-\mu_k)]}{(2\pi)^{D/2}|\Sigma|^{1/2}}',size=28)
        ghost=ell[0].copy();self.add(ghost)
        self.beat(ghost.animate.move_to(ell[1]),Circumscribe(f,color=MUTED,buff=.1));self.remove(ghost)
        field=self.field(ax);boundary=self.boundary(ax);self.add(field)
        self.beat(Create(boundary))
        # Make room for the algebra; restore this same coordinate system afterwards.
        self.remove(ax,ax.marks,dots,ell,field,boundary,f,note)
        eq=MathTex(r'a=',r'[-\tfrac12x^T\Sigma^{-1}x',r'+\mu_1^T\Sigma^{-1}x+c_1]',r'-',r'[-\tfrac12x^T\Sigma^{-1}x',r'+\mu_2^T\Sigma^{-1}x+c_2]',font_size=30).move_to([0,.9,0])
        eq[1].set_color(PURPLE_CLS);eq[4].set_color(PURPLE_CLS);eq[2].set_color(RED_CLS);eq[5].set_color(BLUE_CLS)
        self.add(jp('対数を引くと、同じ二次項が現れる',27).move_to([0,2.2,0]))
        self.beat(phases=[('show expanded logs',.8,lambda:FadeIn(eq)),('match quadratic terms',self.beat_cues()[-1]['end']-.8,lambda:AnimationGroup(Circumscribe(eq[1],color=PURPLE_CLS),Circumscribe(eq[4],color=PURPLE_CLS)))])
        reduced=MathTex(r'a=',r'(\mu_1-\mu_2)^T\Sigma^{-1}x',r'+c_1-c_2',font_size=38).move_to([0,-.6,0]);reduced[1].set_color(GOLD)
        cross=VGroup(Cross(eq[1],stroke_color=GOLD),Cross(eq[4],stroke_color=GOLD))
        self.add(jp('c₁、c₂ は、入力に依存しない定数',23,MUTED).move_to([0,-1.6,0]))
        self.beat(phases=[('cancel common terms',self.sentence_duration(0),lambda:Create(cross)),('collect linear terms',.8,lambda:TransformFromCopy(eq,reduced)),('explain remaining terms',self.sentence_duration(1)-.8,lambda:Circumscribe(reduced,color=GOLD))])
        self.remove(*[m for m in self.mobjects if m is not self.subtitle and m.get_center()[1]<3])
        self.add(ax,ax.marks,dots,ell,field,boundary)
        f=self.formula(r'p(C_1\mid x)=\sigma(w^Tx+w_0),\qquad',r'w=\Sigma^{-1}(\mu_1-\mu_2)',size=29);f[1].set_color(GOLD)
        w,b=linear_params();center=-b*w/(w@w);normal=w/np.linalg.norm(w);tangent=np.array([-normal[1],normal[0]])
        arrow=Arrow(ax.c2p(*center),ax.c2p(*(center+normal)),buff=0,color=GOLD)
        self.beat(GrowArrow(arrow))
        point=Dot(ax.c2p(*(center-tangent)),radius=.075,color=WHITE);self.add(point)
        self.beat(phases=[('along boundary',self.sentence_duration(0)*.5,lambda:point.animate.move_to(ax.c2p(*(center+tangent)))),('cross boundary',self.sentence_duration(0)*.5,lambda:point.animate.move_to(ax.c2p(*(center+normal)))),('explain dimensions',self.sentence_duration(1),lambda:Indicate(arrow))])

    def priors(self):
        ax=self.axes();prior=ValueTracker(.5);get=lambda:[prior.get_value(),1-prior.get_value()]
        self.add(self.field(ax,priors=get),self.ellipses(ax),self.dots(ax))
        boundary=self.boundary(ax,priors=get);self.add(boundary)
        point=np.array([0,-.45]);self.add(Dot(ax.c2p(*point),radius=.08,color=WHITE))
        self.probability_bars(lambda:posterior(point,priors=get()))
        slider=self.slider(prior,.2,.8,[0,2.55,0],r'p(C_1)',width=3.3)
        f=self.formula(r'w_0=-\tfrac12\mu_1^T\Sigma^{-1}\mu_1+\tfrac12\mu_2^T\Sigma^{-1}\mu_2',r'+\ln\frac{p(C_1)}{p(C_2)}',size=29);f[1].set_color(ORANGE_CLS)
        self.beat(Indicate(boundary,scale_factor=1,color=GOLD))
        self.beat(prior.animate.set_value(.8))
        self.beat(Circumscribe(f[1],color=ORANGE_CLS),Indicate(boundary,scale_factor=1,color=GOLD))
        self.beat(prior.animate.set_value(.2))
        ghosts=VGroup(*[path(ax,p,MUTED,1.2).set_opacity(.5) for pi in [.2,.5,.8] for p in decision_paths(priors=[pi,1-pi])])
        self.beat(Create(ghosts));self.beat(prior.animate.set_value(.5))

    def multiclass(self):
        ax=self.axes();t=ValueTracker(0);x=ValueTracker(-1);y=ValueTracker(-.5)
        covs=lambda:covariances(t.get_value());means=lambda:MEANS;get=lambda:posterior([x.get_value(),y.get_value()],MEANS,covs())
        field=self.field(ax,means,covs,lambda:[1/3]*3);self.add(field)
        ell=always_redraw(lambda:self.ellipses(ax,MEANS,covs()));self.add(ell)
        point=always_redraw(lambda:Dot(ax.c2p(x.get_value(),y.get_value()),color=WHITE,radius=.075));self.add(point)
        self.probability_bars(get,3)
        note=self.note('色の混合比＝三つの事後確率')
        self.beat(x.animate.set_value(0),y.animate.set_value(.5))
        self.beat(x.animate.set_value(1.2),y.animate.set_value(-.4))
        f=self.formula(r'a_k=\ln[p(x\mid C_k)p(C_k)],\qquad',r'p(C_k\mid x)=\frac{e^{a_k}}{\sum_j e^{a_j}}',size=29);f[1].set_color(GOLD)
        self.beat(x.animate.set_value(0),y.animate.set_value(.9),Circumscribe(f,color=MUTED,buff=.1))
        self.remove(f);f=self.formula(r'a_k\equiv w_k^Tx+w_{k0},\quad w_k=\Sigma^{-1}\mu_k,\quad w_{k0}=-\tfrac12\mu_k^T\Sigma^{-1}\mu_k+\ln p(C_k)',size=25)
        boundary=self.boundary(ax,means,covs,lambda:[1/3]*3);self.beat(Create(boundary))
        self.remove(note);self.note('緑の共分散：共有の形から、別の形へ').move_to([-1.5,2.55,0])
        slider=self.slider(t,0,1,[4.65,2.35,0],r'\tau',GREEN_CLS,width=2.2)
        self.remove(f);f=self.formula(r'a_i-a_j=-\tfrac12x^T',r'(\Sigma_i^{-1}-\Sigma_j^{-1})',r'x+b_{ij}^Tx+c_{ij}',size=30);f[1].set_color(GREEN_CLS)
        self.beat(t.animate.set_value(1))
        self.beat(Circumscribe(f[1],color=GREEN_CLS),x.animate.set_value(-.5))
        self.beat(t.animate.set_value(0))

    def learning(self):
        ax=self.axes();dots=self.dots(ax);means,cov,prior=fit()
        self.note('ラベル付きの自作データ：赤30個、青20個')
        self.beat(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=.015))
        f=self.formula(r'L=\prod_{n=1}^{N}[\pi\mathcal N(x_n\mid\mu_1,\Sigma)]^{t_n}[(1-\pi)\mathcal N(x_n\mid\mu_2,\Sigma)]^{1-t_n}',size=26)
        label=tex(r't_n=1:C_1\quad t_n=0:C_2',24).move_to([4.45,1.35,0]);self.add(label)
        self.beat(Circumscribe(f,color=MUTED,buff=.1),Indicate(dots[:5]))
        self.remove(label,f);f=self.formula(r'\hat\pi=\frac{N_1}{N_1+N_2}=\frac{30}{50}=0.60');f.set_color(ORANGE_CLS)
        self.probability_bars(lambda:np.array([prior,1-prior]),title='事前確率')
        self.beat(Circumscribe(f,color=ORANGE_CLS),Indicate(dots[:30],color=RED_CLS,scale_factor=1.02))
        centers=VGroup(*[Dot(ax.c2p(*m),radius=.1,color=c) for m,c in zip(means,COLORS)])
        links=VGroup(*[Line(ax.c2p(*p),ax.c2p(*m),color=c,stroke_width=1,stroke_opacity=.55) for a,m,c in zip([RED_POINTS,BLUE_POINTS],means,COLORS) for p in a])
        self.remove(f);f=self.formula(r'\hat\mu_k=\frac1{N_k}\sum_{n\in C_k}x_n');f.set_color(GOLD)
        self.beat(Create(links),FadeIn(centers))
        self.remove(f);f=self.formula(r'r_n=x_n-\hat\mu_{c_n}')
        self.beat(Indicate(links,color=GOLD))
        centered=self.dots(ax,RED_POINTS-means[0],BLUE_POINTS-means[1]);common=contour(ax,np.zeros(2),cov,GOLD)
        self.remove(links,centers)
        self.remove(f);f=self.formula(r'\hat\Sigma=\frac1N\sum_n r_nr_n^T');f.set_color(GOLD)
        self.beat(Transform(dots,centered),Create(common))
        self.remove(f);f=self.formula(r'\hat\Sigma=\frac{N_1}{N}S_1+\frac{N_2}{N}S_2,\qquad S_k=\frac1{N_k}\sum_{n\in C_k}r_nr_n^T',size=28)
        ell=self.ellipses(ax,means,[cov,cov]);self.beat(Transform(dots,self.dots(ax)),ReplacementTransform(common,ell))
        field=self.field(ax,lambda:means,lambda:[cov,cov],lambda:[prior,1-prior]);self.add(field)
        boundary=self.boundary(ax,lambda:means,lambda:[cov,cov],lambda:[prior,1-prior]);self.beat(Create(boundary))

    def outlier(self):
        ax=self.axes((-3.5,4.8,1),(-2.5,3.7,1));t=ValueTracker(0)
        calc=lambda:fit(contaminated(t.get_value()),BLUE_POINTS)
        means=lambda:calc()[0];covs=lambda:[calc()[1]]*2;priors=lambda:[calc()[2],1-calc()[2]]
        dots=always_redraw(lambda:self.dots(ax,contaminated(t.get_value()),BLUE_POINTS))
        ell=always_redraw(lambda:self.ellipses(ax,means(),covs()))
        boundary=self.boundary(ax,means,covs,priors);self.add(dots,ell,boundary)
        focus=always_redraw(lambda:Circle(radius=.13,color=GOLD).move_to(ax.c2p(*contaminated(t.get_value())[0])))
        self.add(focus);self.note('赤い一点だけを移動し、最尤推定を毎回やり直す')
        r2=lambda:np.sum((contaminated(t.get_value())[0]-means()[0])**2)
        num=readout(r'\|r\|^2=',r2,[4.6,.8,0],GOLD);self.add(num)
        f=self.formula(r'\hat\mu_1=\mathrm{mean}(C_1),\qquad\hat\Sigma=\frac1N\sum_n r_nr_n^T',size=30)
        self.beat(Indicate(focus))
        self.beat(t.animate.set_value(1))
        residual=always_redraw(lambda:Line(ax.c2p(*means()[0]),ax.c2p(*contaminated(t.get_value())[0]),color=GOLD))
        self.beat(Create(residual),Circumscribe(num,color=GOLD))
        self.beat(t.animate.set_value(0))
        self.beat(Circumscribe(ell,color=GOLD),Circumscribe(f,color=MUTED,buff=.1))

    def discrete(self):
        combos=VGroup(*[VGroup(*[Square(.34,fill_color=GOLD if b=='1' else MUTED,fill_opacity=.7 if b=='1' else .12,stroke_width=1) for b in f'{n:03b}']).arrange(RIGHT,buff=.1) for n in range(8)]).arrange_in_grid(rows=2,cols=4,buff=(.5,.6)).move_to([0,.8,0])
        note=self.note('例：三つの単語が、ある＝1／ない＝0')
        self.beat(FadeIn(combos))
        f=self.formula(r'D=3:\ 2^3-1=7\qquad\longrightarrow\qquad D=10:\ 2^{10}-1=1023')
        self.beat(Circumscribe(f,color=GOLD),Indicate(combos))
        self.remove(combos,f)
        root=tex('C_k',34,PURPLE_CLS).move_to([0,1.9,0]);nodes=VGroup(*[VGroup(Square(.7,color=MUTED),tex(f'x_{i+1}',28)).move_to([(i-1)*2.4,.6,0]) for i in range(3)])
        arrows=VGroup(*[Arrow(root.get_bottom(),n.get_top(),buff=.15,color=MUTED) for n in nodes]);self.add(root)
        self.add(nodes)
        self.beat(phases=[('show conditional dependence',.8,lambda:Create(arrows)),('condition on class',self.beat_cues()[-1]['end']-.8,lambda:Circumscribe(root,color=PURPLE_CLS))])
        self.remove(root,arrows,nodes)
        bits=np.array([0,0,0]);bit_mobs=VGroup();params=VGroup()
        for i in range(3):
            x=-3+i*2.3
            bit_mobs.add(VGroup(Square(.68,color=MUTED),tex('0',34)).move_to([x,1.25,0]))
            params.add(tex(fr'\mu_{{1,{i+1}}}={BINARY_MU[0,i]:.2f}',24,RED_CLS).move_to([x,.35,0]),tex(fr'\mu_{{2,{i+1}}}={BINARY_MU[1,i]:.2f}',24,BLUE_CLS).move_to([x,-.25,0]))
        self.add(bit_mobs,params)
        bars=self.probability_bars(lambda:softmax(binary_scores(bits)),2,pos=(4.75,.2,0))
        f=self.formula(r'p(x\mid C_k)=\prod_i\mu_{ki}^{x_i}(1-\mu_{ki})^{1-x_i}',size=32)
        self.beat(Circumscribe(params,color=MUTED,buff=.15),Circumscribe(f,color=MUTED,buff=.1))
        self.remove(f);f=self.formula(r'a_k=\sum_i[x_i\ln\mu_{ki}+(1-x_i)\ln(1-\mu_{ki})]+\ln p(C_k)',size=29)
        rail=NumberLine(x_range=[-2,3,1],length=7,include_numbers=True,font_size=17,color=MUTED).move_to([-1,-1.25,0])
        odds=lambda:float(np.diff(binary_scores(bits)[::-1])[0])
        marker=Dot(radius=.075,color=GOLD).add_updater(lambda m:m.move_to(rail.n2p(odds())))
        value=readout(r'a_1-a_2=',odds,[-1,-.75,0],GOLD,2,23)
        self.add(rail,marker,value)
        # Binary inputs change discretely; only the visual emphasis is interpolated.
        def switch(i,value):
            old_odds=odds()
            bits[i]=value
            arrow=Arrow(rail.n2p(old_odds),rail.n2p(odds()),buff=0,color=GOLD,stroke_width=4)
            self.add(arrow)
            return Transform(bit_mobs[i][1],tex(str(value),34,GOLD).move_to(bit_mobs[i][1]))
        self.beat(phases=[('add word 1',.7,lambda:switch(0,1)),('explain contribution',self.beat_cues()[-1]['end']-.7,lambda:Circumscribe(params[:2],color=GOLD))])
        self.equation(f,r'a_k=\sum_i x_i',r'\ln\frac{\mu_{ki}}{1-\mu_{ki}}',r'+\sum_i\ln(1-\mu_{ki})+\ln p(C_k)',colors={1:GOLD},size=29)
        self.beat(phases=[('add word 3',.7,lambda:switch(2,1)),('explain posterior',self.beat_cues()[-1]['end']-.7,lambda:Circumscribe(bars,color=GOLD))])

    def family(self):
        self.note('指数型分布族：共通する「書き方」から考える')
        f=self.formula(r'\mathcal N(x\mid\mu,\Sigma)\qquad\mathrm{Bernoulli}(x\mid\mu)',size=36,y=1.1)
        self.beat(Circumscribe(f,color=GOLD))
        f=self.equation(f,r'p(x\mid\lambda_k)=',r'h(x)',r'g(\lambda_k)',r'\exp[\lambda_k^Tu(x)]',y=1.1,size=36,colors={1:PURPLE_CLS,2:ORANGE_CLS,3:GOLD})
        restriction=tex(r'u(x)=x,\qquad s\ \mathrm{shared}',36,GREEN_CLS).move_to([0,-.4,0])
        self.beat(phases=[('state restrictions',.6,lambda:FadeIn(restriction)),('explain restrictions',self.beat_cues()[-1]['end']-.6,lambda:Circumscribe(restriction,color=GREEN_CLS))])
        self.remove(restriction)
        restriction=tex(r'u(x)=x,\qquad s=1',30,GREEN_CLS).move_to([0,2.05,0]);self.add(restriction)
        ratio=MathTex(r'\frac{p(x\mid\lambda_1)}{p(x\mid\lambda_2)}=',r'\frac{h(x)}{h(x)}',r'\frac{g(\lambda_1)}{g(\lambda_2)}',r'e^{(\lambda_1-\lambda_2)^Tx}',font_size=35).move_to([0,-.5,0]);ratio[1].set_color(PURPLE_CLS);ratio[3].set_color(GOLD)
        self.beat(phases=[('form density ratio',.8,lambda:TransformFromCopy(f,ratio)),('cancel shared factor',self.beat_cues()[-1]['end']-.8,lambda:Create(Cross(ratio[1],stroke_color=GOLD)))])
        self.remove(f,ratio,*[m for m in self.mobjects if isinstance(m,Cross)])
        f=self.formula(r'a(x)=',r'(\lambda_1-\lambda_2)^Tx',r'+\ln\frac{g(\lambda_1)p(C_1)}{g(\lambda_2)p(C_2)}',size=34,y=.75);f[1].set_color(GOLD)
        k=self.formula(r'a_k\equiv\lambda_k^Tx+\ln g(\lambda_k)+\ln p(C_k)',size=30,y=-.6)
        self.beat(Circumscribe(f,color=GOLD),Circumscribe(k,color=MUTED))
        self.remove(*[m for m in self.mobjects if m is not self.subtitle and m.get_center()[1]<3])
        ax=self.axes();t=ValueTracker(0);means=lambda:MEANS;covs=lambda:covariances(t.get_value())
        self.add(self.field(ax,means,covs,lambda:[1/3]*3),always_redraw(lambda:self.ellipses(ax,MEANS,covs())),self.boundary(ax,means,covs,lambda:[1/3]*3))
        f=self.formula(r'p(x\mid C_k),p(C_k)\quad\xrightarrow{\mathrm{Bayes}}\quad p(C_k\mid x)',size=34)
        self.beat(Circumscribe(f,color=GOLD))
        self.beat(phases=[('different covariance',self.sentence_duration(0),lambda:t.animate.set_value(1)),('next section',self.sentence_duration(1),lambda:Circumscribe(f[-1],color=MUTED))])
