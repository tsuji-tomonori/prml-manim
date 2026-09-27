"""PRML 2.5: linked visual experiments in Manim Community."""
from pathlib import Path
import json
import math
import re
import numpy as np
from manim import *
from caption_layout import jp, tex
from narrated_scene import NarratedScene
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from nonparametric_model import *

COLOR_TEMPLATE=TexTemplate()
COLOR_TEMPLATE.add_to_preamble(r'\usepackage{xcolor}')
BG='#10141F'
DATA=ManimColor('#58B5ED'); MODEL=ManimColor('#FF6B77')
SMOOTH_GRID=np.linspace(0,1,401)
TRUE=ManimColor('#77D49A'); COUNT=ManimColor('#FFE079')
VOLUME=ManimColor('#C29AFF'); MUTED=ManimColor('#A8B2C5')


def curve(ax, x, y, color=MODEL, width=3, opacity=1):
    x,y=np.asarray(x),np.asarray(y)
    origin=ax.c2p(0,0)
    pts=origin+x[:,None]*(ax.c2p(1,0)-origin)+y[:,None]*(ax.c2p(0,1)-origin)
    return VMobject().set_points_as_corners(pts).set_stroke(color,width,opacity)


def readout(label, getter, pos, color=WHITE, places=3):
    prefix=tex(label,25,color)
    val=DecimalNumber(getter(),num_decimal_places=places,font_size=25,color=color)
    g=VGroup(prefix,val).arrange(RIGHT,buff=.12).move_to(pos)
    anchor=val.get_left().copy()
    val.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return g


class PRML25NonparametricMethods(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.timeline=[]
        self.manifest={s['id']:s for s in json.loads(MANIFEST.read_text())['scenes']}
        for i, method in enumerate([self.question,self.histograms,self.local,
                                    self.boxes,self.gaussians,self.nearest,
                                    self.classification,self.boundaries,self.cost]):
            self.begin(i)
            method()
            assert self.beat_index==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml25_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def formula(self, formula, y=-2.55, size=32, colors=None):
        # Match TeX commands atomically: isolating bare h would split \widehat.
        def paint(match):
            token=match.group(0)
            color=(colors or {}).get(token)
            return r'\textcolor[HTML]{'+color.to_hex().lstrip('#')+'}{'+token+'}' if color else token
        colored=re.sub(r"\\[A-Za-z]+(?:_[A-Za-z0-9])?|[A-Za-z](?:_[A-Za-z0-9])?",paint,formula)
        f=MathTex(colored,font_size=size,tex_template=COLOR_TEMPLATE)
        f.move_to([0,y,0])
        assert f.width<13 and f.height<1.2,(formula,f.width,f.height)
        self.add(f)
        return f

    def density_axes(self, top=6, dynamic=None):
        ax=Axes(x_range=[0,1,.2],y_range=[0,top,top/3],x_length=10,y_length=3.15,
                tips=False,axis_config=dict(color=MUTED,stroke_width=1.3),
                y_axis_config=dict(include_ticks=False)).move_to([0,-.15,0])
        if dynamic:
            origin=ax.c2p(0,0).copy();ux=ax.c2p(1,0)-origin;uy=ax.c2p(0,top)-origin
            ax.c2p=lambda x,y=0:origin+x*ux+y/dynamic()*uy
        labels=VGroup(*[tex(f'{x:g}',18,MUTED).next_to(ax.c2p(x,0),DOWN,buff=.2) for x in [0,.2,.4,.6,.8,1]])
        labels.add(tex('x',23).next_to(ax.c2p(1,0),RIGHT,buff=.2))
        labels.add(jp('密度',19,MUTED).move_to([-5.5,1.7,0]))
        if dynamic:
            val=DecimalNumber(dynamic(),num_decimal_places=1,font_size=18,color=MUTED)
            val.add_updater(lambda m:m.set_value(dynamic()).move_to([-5.4,1.38,0]))
            labels.add(val)
        else:
            labels.add(*[tex(f'{v:g}',18,MUTED).next_to(ax.c2p(0,v),LEFT,buff=.15) for v in [top/2,top]])
        self.add(ax,labels)
        self.plot_objects=VGroup(ax,labels)
        return ax

    def rug(self,ax,inside=None):
        g=VGroup(*[Line(ax.c2p(x,0)+DOWN*.07,ax.c2p(x,0)+UP*.13,stroke_width=2,color=DATA) for x in SAMPLES])
        if inside:
            for line,x in zip(g,SAMPLES):
                line.add_updater(lambda m,x=x:m.set_color(COUNT if inside(x) else DATA))
        return g

    def slider(self,tracker,lo,hi,label,pos=(0,2.18,0),color=VOLUME,integer=False):
        base=NumberLine(x_range=[lo,hi],length=4.0,include_ticks=False,color=MUTED).move_to(pos)
        g=VGroup(base,tex(label,27,color).next_to(base,LEFT,buff=.3))
        knob=Dot(radius=.065,color=color)
        knob.add_updater(lambda m:m.move_to(base.n2p(round(tracker.get_value()) if integer else tracker.get_value())))
        g.add(knob)
        g.add(readout('=',lambda:round(tracker.get_value()) if integer else tracker.get_value(),np.array(pos)+RIGHT*3.0,color,0 if integer else 3))
        self.add(g)
        return g

    def window(self,ax,x,width,height=5):
        return always_redraw(lambda:Rectangle(width=max(.0001,width()*10),height=abs(ax.c2p(0,height)[1]-ax.c2p(0,0)[1]),
                       stroke_color=VOLUME,stroke_width=2,fill_color=VOLUME,fill_opacity=.1)
                       .move_to(ax.c2p(x.get_value(),height/2)))

    def marker(self,ax,x,height):
        return always_redraw(lambda:VGroup(DashedLine(ax.c2p(x.get_value(),0),ax.c2p(x.get_value(),height()),color=COUNT,stroke_width=2),
                                               Dot(ax.c2p(x.get_value(),height()),radius=.065,color=COUNT)))

    def hist(self,ax,width,offset=0):
        vals,edges,_=histogram(width,offset)
        return VGroup(*[Rectangle(width=10*(b-a),height=max(.001,ax.c2p(0,v)[1]-ax.c2p(0,0)[1]),
                     fill_color=DATA,fill_opacity=.45,stroke_color=DATA,stroke_width=1)
                     .move_to(ax.c2p((a+b)/2,v/2)) for a,b,v in zip(edges[:-1],edges[1:],vals)])

    def question(self):
        ax=self.density_axes(5)
        dots=self.rug(ax)
        self.beat(LaggedStart(*[FadeIn(d,shift=UP*.2) for d in dots],lag_ratio=.02))
        x=ValueTracker(.15)
        indicator=always_redraw(lambda:Arrow(ax.c2p(x.get_value(),1.5),ax.c2p(x.get_value(),.15),buff=0,color=COUNT))
        self.add(indicator)
        self.beat(x.animate.set_value(.85))
        fit=curve(ax,GRID,fitted_gaussian(GRID));self.remove(indicator)
        label=jp('赤：単一ガウスの推定',22,MODEL).move_to([0,2.1,0]);self.add(label)
        self.beat(Create(fit))
        true=curve(ax,GRID,truth(GRID),TRUE,2)
        self.remove(label);label=jp('緑：実験の生成分布　　赤：推定',22).move_to([0,2.1,0]);self.add(label)
        self.beat(Create(true))
        right=ValueTracker(.201)
        def area():
            z=np.linspace(.2,right.get_value(),100)
            return Polygon(ax.c2p(z[0],0),*[ax.c2p(a,b) for a,b in zip(z,truth(z))],ax.c2p(z[-1],0),
                           stroke_width=0,fill_color=TRUE,fill_opacity=.5)
        shade=always_redraw(area);self.add(shade)
        self.formula(r'P(a<x<b)=\int_a^b p(x)\,dx')
        self.beat(right.animate.set_value(.38))
        self.beat(Transform(fit,curve(ax,GRID,kde(GRID,.06))),FadeOut(shade))

    def histograms(self):
        w=ValueTracker(.1);offset=ValueTracker(0)
        span=lambda:max(5.0,1.15*float(histogram(w.get_value(),offset.get_value())[0].max()))
        ax=self.density_axes(dynamic=span)
        bars=always_redraw(lambda:self.hist(ax,w.get_value(),offset.get_value()))
        self.add(self.rug(ax),bars)
        self.slider(w,.025,.5,r'\Delta')
        self.beat(w.animate.set_value(.08))
        select=SurroundingRectangle(bars[3],color=COUNT,buff=.02)
        f=self.formula(r'\text{area}=n_i/N',colors={'n_i':COUNT,'N':DATA})
        self.beat(Create(select))
        self.remove(select,f)
        f=self.formula(r'p_i={n_i\over N\Delta_i},\qquad p_i\Delta_i={n_i\over N}',colors={'n_i':COUNT,r'\Delta_i':VOLUME})
        self.beat(Indicate(bars[3],color=COUNT,scale_factor=1.03))
        self.beat(w.animate.set_value(.025))
        self.beat(w.animate.set_value(.5))
        self.beat(phases=[('restore_width',self.sentence_duration(0),lambda:w.animate.set_value(.08)),
                          ('shift_edges',self.sentence_duration(1),lambda:offset.animate.set_value(.04))])
        self.remove(f)
        self.formula(r'\sum_i p_i\Delta_i={1\over N}\sum_i n_i=1')
        self.beat(offset.animate.set_value(0))

    def local(self):
        ax=self.density_axes(6);x=ValueTracker(.28);w=ValueTracker(.16)
        count=lambda:local_count(x.get_value(),w.get_value())
        win=self.window(ax,x,w.get_value,5.2)
        self.add(self.rug(ax,lambda p:abs(p-x.get_value())<=w.get_value()/2),win)
        vals=VGroup(readout('K=',count,[-3.5,2.15,0],COUNT,0),readout('V=',w.get_value,[0,2.15,0],VOLUME),tex('N=60',25,DATA).move_to([3.5,2.15,0]))
        self.add(vals)
        self.beat(w.animate.set_value(.12))
        self.beat(x.animate.set_value(.5))
        f=self.formula(r'P=\int_R p(u)\,du\simeq K/N',colors={'K':COUNT})
        self.beat(x.animate.set_value(.68))
        self.remove(f);f=self.formula(r'P\simeq p(x)V',colors={'V':VOLUME})
        approx=always_redraw(lambda:Rectangle(width=w.get_value()*10,height=max(.001,ax.c2p(0,truth(x.get_value()))[1]-ax.c2p(0,0)[1]),
                                             stroke_color=TRUE,fill_color=TRUE,fill_opacity=.22)
                                             .move_to(ax.c2p(x.get_value(),truth(x.get_value())/2)))
        self.add(approx)
        self.beat(w.animate.set_value(.065))
        self.remove(f,approx);f=self.formula(r'\widehat p(x)={K\over NV}',colors={'K':COUNT,'V':VOLUME,'N':DATA})
        mark=self.marker(ax,x,lambda:count()/(N*w.get_value()));self.add(mark)
        self.beat(x.animate.set_value(.3))
        self.beat(phases=[('shrink',self.sentence_duration(0),lambda:w.animate.set_value(.03)),
                          ('expand',self.sentence_duration(1),lambda:w.animate.set_value(.25))])
        self.remove(f,win,mark,vals)
        # Two equations occupy the former readout and footer zones.
        self.formula(r'P(K)={N\choose K}P^K(1-P)^{N-K}',y=2.1,size=30)
        self.formula(r'\mathbb E[K/N]=P,\quad\mathrm{var}[K/N]={P(1-P)\over N}',size=30)
        self.clear()
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]))
        self.formula(r'P(K)={N\choose K}P^K(1-P)^{N-K}',y=2.45,size=30)
        self.formula(r'\mathbb E[K/N]=P,\quad\mathrm{var}[K/N]={P(1-P)\over N}',size=30)
        tr=ValueTracker(20)
        bx=Axes(x_range=[0,1,.2],y_range=[0,.2,.1],x_length=9,y_length=2.7,tips=False).move_to([0,-.2,0])
        self.add(bx,tex('K/N',25).next_to(bx.x_axis,RIGHT),tex('P=0.35',25).move_to([2.5,1.7,0]),
                 readout('N=',lambda:round(tr.get_value()),[-2.5,1.7,0],DATA,0))
        def binomial_bars():
            n=round(tr.get_value())
            probabilities=np.array([math.comb(n,k)*.35**k*.65**(n-k) for k in range(n+1)])
            return VGroup(*[Line(bx.c2p(k/n,0),bx.c2p(k/n,float(v)),color=COUNT,stroke_width=2)
                            for k,v in enumerate(probabilities) if v>1e-6])
        bars=always_redraw(binomial_bars);self.add(bars)
        self.beat(tr.animate.set_value(200))

    def boxes(self):
        ax=self.density_axes(6);x=ValueTracker(.12);h=.12
        win=self.window(ax,x,lambda:h,5)
        line=curve(ax,GRID,box_kde(GRID,h),MODEL)
        self.add(self.rug(ax,lambda p:abs(p-x.get_value())<=h/2),win,line)
        mark=self.marker(ax,x,lambda:box_kde(x.get_value(),h));self.add(mark)
        f=self.formula(r'K=\sum_{n=1}^N k\!\left({x-x_n\over h}\right)',colors={'K':COUNT,'h':VOLUME})
        self.beat(x.animate.set_value(.38))
        point=float(SAMPLES[15]);self.remove(win,mark,line)
        selected=Dot(ax.c2p(point,0),radius=.08,color=COUNT);self.add(selected)
        interval=Line(ax.c2p(point-h/2,1),ax.c2p(point+h/2,1),stroke_width=9,color=VOLUME)
        self.beat(Create(interval))
        box=Rectangle(width=h*10,height=3.15/(6*N*h),fill_color=COUNT,fill_opacity=.7,stroke_color=COUNT).move_to(ax.c2p(point,1/(2*N*h)))
        self.beat(ReplacementTransform(interval,box))
        boxes=VGroup(*[Rectangle(width=h*10,height=3.15/(6*N*h),fill_color=DATA,fill_opacity=.12,stroke_color=DATA,stroke_width=.6)
                       .move_to(ax.c2p(p,1/(2*N*h))) for p in SAMPLES])
        self.add(boxes);self.beat(Create(line),FadeOut(box))
        self.remove(f);f=self.formula(r'\widehat p(x)={1\over N}\sum_{n=1}^N{1\over h^D}k\!\left({x-x_n\over h}\right)',size=33,colors={'h':VOLUME})
        top=tex(r'V=h^D\qquad k(u)=\mathbf1\{|u_i|\leq 1/2\}',28).move_to([0,2.1,0]);self.add(top)
        self.beat(x.animate.set_value(.7),Indicate(boxes,scale_factor=1.01))
        self.add(mark)
        self.beat(x.animate.set_value(.82))

    def gaussians(self):
        h=ValueTracker(.06);number=ValueTracker(1)
        span=lambda:max(5,1.1*float(kde(SMOOTH_GRID,h.get_value()).max()))
        ax=self.density_axes(dynamic=span)
        # Fractional final contribution visualizes accumulation; denominator stays N.
        def cumulative():
            n=number.get_value();weights=np.clip(n-np.arange(N),0,1)
            return gaussian(SMOOTH_GRID[:,None],SAMPLES,h.get_value())@weights/N
        total=always_redraw(lambda:curve(ax,SMOOTH_GRID,cumulative()))
        kernels=always_redraw(lambda:VGroup(*[curve(ax,SMOOTH_GRID,gaussian(SMOOTH_GRID,p,h.get_value())/N,DATA,1,.3) for p in SAMPLES[::5]]))
        self.add(self.rug(ax),total,kernels)
        self.slider(h,.012,.25,'h')
        f=self.formula(r'\widehat p(x)={1\over N}\sum_n\mathcal N(x\mid x_n,h^2)',colors={'h':VOLUME})
        self.beat(number.animate.set_value(12))
        self.beat(number.animate.set_value(N))
        self.remove(f);f=self.formula(r'\widehat p(x)={1\over N}\sum_n{e^{-\|x-x_n\|^2/(2h^2)}\over(2\pi h^2)^{D/2}}',size=30,colors={'h':VOLUME})
        self.beat(h.animate.set_value(.075))
        self.beat(h.animate.set_value(.012))
        self.beat(h.animate.set_value(.25))
        self.remove(f);f=self.formula(r'k(u)\geq0,\quad\int k(u)\,du=1\quad\Longrightarrow\quad\int\widehat p(x)\,dx=1',size=29)
        self.beat(h.animate.set_value(.06))
        true=always_redraw(lambda:curve(ax,SMOOTH_GRID,truth(SMOOTH_GRID),TRUE,2,.6));self.add(true)
        self.beat(h.animate(rate_func=there_and_back).set_value(.1))

    def nearest(self):
        k=ValueTracker(7);x=ValueTracker(.42);growth=ValueTracker(.005)
        kval=lambda:int(round(k.get_value()))
        radius=lambda:float(knn_radius(x.get_value(),kval()))
        span=lambda:max(5,1.08*float(knn_density(GRID,kval()).max()))
        ax=self.density_axes(dynamic=span)
        self.slider(k,3,25,'K',color=COUNT,integer=True)
        win=self.window(ax,x,lambda:2*radius()*growth.get_value(),4)
        self.add(self.rug(ax,lambda p:abs(p-x.get_value())<=radius()*growth.get_value()+1e-12),win)
        f=self.formula(r'\widehat p(x)={K\over N\,2r_K(x)},\qquad V(x)=2r_K(x)',colors={'K':COUNT,'V':VOLUME})
        self.beat(growth.animate.set_value(1),start_sentence=1)
        density=always_redraw(lambda:curve(ax,GRID,knn_density(GRID,kval()),VOLUME))
        marker=self.marker(ax,x,lambda:knn_density(x.get_value(),kval()))
        self.add(density,marker)
        self.beat(x.animate.set_value(.29))
        self.beat(x.animate.set_value(.5))
        self.beat(x.animate.set_value(.72))
        self.beat(k.animate.set_value(3))
        self.beat(k.animate.set_value(25))
        self.remove(f)
        self.formula(r'\widehat p(x)\sim {K\over 2N|x|}\quad (|x|\to\infty),\qquad\int_{\mathbb R}\widehat p(x)\,dx=\infty',size=28)
        self.beat(x.animate.set_value(.8))

    def plane(self):
        ax=Axes(x_range=[-2,2,1],y_range=[-1.2,1.2,.6],x_length=6,y_length=3.6,tips=False,
                axis_config=dict(color=MUTED,stroke_width=1)).move_to([-2,.05,0])
        dots=VGroup(*[Dot(ax.c2p(*p),radius=.065,color=MODEL if lab==0 else DATA).set_stroke(BG,1) for p,lab in zip(POINTS,LABELS)])
        self.add(ax,dots,tex('x_1',23).next_to(ax.x_axis,RIGHT,buff=.2),tex('x_2',23).next_to(ax.y_axis,UP,buff=.12))
        return ax,dots

    def classification(self):
        ax,dots=self.plane();qx=ValueTracker(QUERY[0]);qy=ValueTracker(QUERY[1]);grow=ValueTracker(.01)
        q=lambda:np.array([qx.get_value(),qy.get_value()])
        index=lambda:neighbours(q(),5)[0]
        dot=always_redraw(lambda:Dot(ax.c2p(*q()),radius=.075,color=WHITE))
        circle=always_redraw(lambda:Circle(radius=neighbours(q(),5)[1]*1.5*grow.get_value(),color=VOLUME,stroke_width=2).move_to(ax.c2p(*q())))
        self.add(dot)
        self.beat(LaggedStart(*[Indicate(d,scale_factor=1.3) for d in dots],lag_ratio=.04))
        self.add(circle);self.beat(grow.animate.set_value(1))
        links=always_redraw(lambda:VGroup(*[Line(ax.c2p(*q()),ax.c2p(*POINTS[i]),color=COUNT,stroke_width=1.6) for i in index()]))
        self.add(links)
        self.add(readout(r'K_{\rm red}=',lambda:int((LABELS[index()]==0).sum()),[3.6,1.8,0],MODEL,0),
                 readout(r'K_{\rm blue}=',lambda:int((LABELS[index()]==1).sum()),[3.6,1.1,0],DATA,0))
        f=tex(r'K=5',32,COUNT).move_to([3.6,.4,0]);self.add(f)
        self.beat(Indicate(circle,scale_factor=1.02))
        a=self.formula(r'p(x\mid C_k)={K_k\over N_kV},\quad p(C_k)={N_k\over N},\quad p(x)={K\over NV}',size=28)
        self.beat(LaggedStart(*[Indicate(dots[i],color=COUNT) for i in index()],lag_ratio=.15))
        self.remove(a)
        a=self.formula(r'p(C_k\mid x)={p(x\mid C_k)p(C_k)\over p(x)}={K_k\over N_kV}{N_k\over N}{NV\over K}',size=29)
        self.beat(Indicate(a,scale_factor=1.01))
        b=MathTex(r'p(C_k\mid x)={K_k\over K}',font_size=37).move_to([0,-2.55,0])
        self.beat(TransformMatchingTex(a,b))
        self.remove(f);self.add(readout(r'p(C_{\rm red}\mid x)=',lambda:posterior(q(),5)[0],[3.6,-.4,0],MODEL,1),
                                readout(r'p(C_{\rm blue}\mid x)=',lambda:posterior(q(),5)[1],[3.6,-1.1,0],DATA,1))
        self.beat(qx.animate.set_value(.65),qy.animate.set_value(-.4))

    def boundaries(self):
        ax,dots=self.plane()
        def field(k):
            classes=classify_grid(k)
            colors=np.array([[83,37,47,255],[27,58,85,255]],dtype=np.uint8)
            return ImageMobject(colors[classes]).set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest']).stretch_to_fit_width(6).stretch_to_fit_height(3.6).move_to(ax.c2p(0,0)).set_z_index(-2)
        bg=field(1);self.add(bg)
        label=tex('K=1',36,COUNT).move_to([3.5,1.8,0]);self.add(label)
        cursor=Dot(ax.c2p(-1.2,.8),radius=.08,color=WHITE);self.add(cursor)
        self.beat(cursor.animate.move_to(ax.c2p(.9,-.5)))
        # Two-point construction in a separate explanatory inset, labelled as such.
        p1=np.array([2.5,.1,0]);p2=np.array([4.5,.1,0]);mid=(p1+p2)/2
        inset=VGroup(Dot(p1,color=MODEL),Dot(p2,color=DATA),Line(p1,p2,color=MUTED),
                     DashedLine(mid+UP*.8,mid+DOWN*.8,color=COUNT),jp('二点だけの場合',20).move_to([3.5,-1.2,0]))
        self.beat(Create(inset))
        nxt=field(5)
        self.beat(FadeOut(bg),FadeIn(nxt),Transform(label,tex('K=5',36,COUNT).move_to(label)))
        bg=nxt;nxt=field(11)
        self.beat(FadeOut(bg),FadeIn(nxt),Transform(label,tex('K=11',36,COUNT).move_to(label)))
        bg=nxt;nxt=field(5)
        self.beat(FadeOut(bg),FadeIn(nxt),Transform(label,tex('K=5',36,COUNT).move_to(label)),cursor.animate.move_to(ax.c2p(-.5,.3)))
        self.remove(inset)
        self.add(jp('極限での保証',25).move_to([3.5,.5,0]),tex(r'R^*\leq R_{1\mathrm{NN}}\leq2R^*',29).move_to([3.5,-.2,0]),
                 jp('最適な誤り率：',19).move_to([3.1,-1,0]),tex('R^*',24).move_to([4.5,-1,0]))
        self.formula(r'N\to\infty\quad\text{(i.i.d.)}',size=30)
        self.beat(cursor.animate.move_to(ax.c2p(.6,.5)))

    def cost(self):
        ax=self.density_axes(5);x=ValueTracker(.28);mode=ValueTracker(0)
        width=lambda:(1-mode.get_value())*.12+mode.get_value()*2*float(knn_radius(x.get_value(),7))
        win=self.window(ax,x,width,4.4);rug=self.rug(ax,lambda p:abs(p-x.get_value())<=width()/2)
        self.add(rug,win,curve(ax,GRID,kde(GRID,.06),MODEL))
        f=self.formula(r'V\ \mathrm{fixed}\quad\longleftrightarrow\quad K\ \mathrm{fixed}',colors={'V':VOLUME,'K':COUNT})
        self.beat(mode.animate.set_value(1),x.animate.set_value(.5))
        links=VGroup(*[Line(ax.c2p(.5,4.5),ax.c2p(p,0),stroke_width=.6,color=COUNT,stroke_opacity=.35) for p in SAMPLES])
        self.remove(f);f=self.formula(r'\mathrm{KDE}:\quad N\ \mathrm{terms}\ \Longrightarrow\ O(N)\ \mathrm{per\ query}',size=31)
        self.beat(LaggedStart(*[Create(l) for l in links],lag_ratio=.015))
        bars=self.hist(ax,.1)
        self.remove(f,links,win)
        self.beat(FadeOut(rug),FadeIn(bars))
        self.clear();self.add(jp(self.story['title'],34).move_to([0,3.35,0]))
        grid=VGroup(*[Square(side_length=.24,color=DATA,fill_opacity=.25).move_to([-1.35+i*.28,-1.25+j*.28,0]) for i in range(10) for j in range(10)])
        count=ValueTracker(1)
        self.add(grid,readout('D=',lambda:round(count.get_value()),[-3.8,1.8,0],COUNT,0),
                 readout('10^D=',lambda:10**round(count.get_value()),[1.8,1.8,0],VOLUME,0))
        self.formula(r'M^D\quad (M=10)',size=36)
        self.beat(count.animate.set_value(6))
        self.remove(grid)
        nodes=[np.array([0,1.0,0]),np.array([-2,0,0]),np.array([2,0,0]),np.array([-3,-1,0]),np.array([-1,-1,0]),np.array([1,-1,0]),np.array([3,-1,0])]
        tree=VGroup(*[Line(nodes[(i-1)//2],nodes[i],color=MUTED) for i in range(1,7)],*[Dot(p,color=DATA) for p in nodes])
        self.add(tree);path=VGroup(Line(nodes[0],nodes[1],color=COUNT,stroke_width=5),Line(nodes[1],nodes[4],color=COUNT,stroke_width=5))
        self.beat(Create(path))
        self.clear();self.add(jp('近くの点を数え、広さで割る',34).move_to([0,3.35,0]))
        ax=self.density_axes(5);self.add(self.rug(ax))
        self.formula(r'\widehat p(x)={K\over NV}',colors={'K':COUNT,'V':VOLUME})
        self.beat(Create(curve(ax,GRID,kde(GRID,.06))),Create(curve(ax,GRID,truth(GRID),TRUE,2,.5)))
