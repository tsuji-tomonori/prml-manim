"""PRML 3.3: linked visual experiments in Manim Community."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from bayesian_model import X,T,XR,TR,CENTERS,FORECAST_X,HELD_OUT_T,phi,posterior,predict,samples,kernel

DATA=ManimColor('#58B5ED'); MEAN=ManimColor('#FF6B77')
PRIOR=ManimColor('#C29AFF'); POST=ManimColor('#77D49A')
NOISE=ManimColor('#FFB45B'); YELLOW=ManimColor('#FFE079'); MUTED=ManimColor('#A8B2C5')
COLORS=[MEAN,YELLOW,PRIOR,NOISE,DATA,POST]

def path(ax,x,y,color=MEAN,width=3,opacity=1):
    o=ax.c2p(0,0); points=o+np.array(x)[:,None]*(ax.c2p(1,0)-o)+np.array(y)[:,None]*(ax.c2p(0,1)-o)
    return VMobject().set_points_as_corners(points).set_stroke(color,width,opacity)

def band(ax,x,y,var,color=POST,opacity=.22):
    sd=np.sqrt(np.maximum(var,0))
    return Polygon(*[ax.c2p(a,b) for a,b in zip(x,y+sd)],*[ax.c2p(a,b) for a,b in zip(x[::-1],(y-sd)[::-1])],stroke_width=0,fill_color=color,fill_opacity=opacity)

def contours(ax,m,c,color=POST):
    theta=np.linspace(0,TAU,100); unit=np.array([np.cos(theta),np.sin(theta)])
    return VGroup(*[path(ax,*(m[:,None]+r*np.linalg.cholesky(c)@unit),color,2.5,.8) for r in [.6,1.2,1.8]])

def number(label,get,pos,color=WHITE,places=2):
    prefix=tex(label,25,color); num=DecimalNumber(get(),num_decimal_places=places,font_size=25,color=color)
    g=VGroup(prefix,num).arrange(RIGHT,buff=.13).move_to(pos); anchor=num.get_left().copy()
    num.add_updater(lambda m:m.set_value(get()).move_to(anchor,aligned_edge=LEFT));return g

class PRML33BayesianLinearRegression(NarratedScene):
    def construct(self):
        self.camera.background_color='#10141F';self.timeline=[]
        self.manifest={s['id']:s for s in json.loads(MANIFEST.read_text())['scenes']}
        for i,method in enumerate([self.question,self.map,self.update,self.sequence,self.regularization,self.prediction,self.curves,self.smoother,self.limits]):
            self.begin(i);method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path('media/prml33_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def axes(self,pos=(0,.25,0),width=9,height=3.7,xr=(-1,1,.5),yr=(-1.6,1.6,1),labels=('x','t')):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,axis_config={'color':MUTED,'stroke_width':1.2,'include_ticks':False}).move_to(pos)
        marks=VGroup()
        for v in [xr[0],(xr[0]+xr[1])/2,xr[1]]:
            marks.add(tex(f'{v:g}',17,MUTED).move_to(ax.c2p(v,yr[0])+DOWN*.20))
        for v in [yr[0],yr[1]]:
            marks.add(tex(f'{v:g}',17,MUTED).next_to(ax.c2p(xr[0],v),LEFT,buff=.12))
        marks.add(tex(labels[0],25).next_to(ax.c2p(xr[1],yr[0]),RIGHT,buff=.16),tex(labels[1],25).next_to(ax.c2p(xr[0],yr[1]),UP,buff=.12))
        self.add(ax,marks);return ax

    def note(self,text,color=MUTED):
        obj=jp(text,21,color).move_to([0,2.67,0]);self.add(obj);return obj

    def formula(self,*parts):
        obj=MathTex(*parts,font_size=30).move_to([0,-2.5,0])
        if obj.width>12:obj.scale_to_fit_width(12)
        self.add(obj);return obj

    def equation_change(self, old, new, *correspondence):
        duration=self.beat_cues()[-1]['end']
        self.beat(phases=[('equation transition',.6,lambda:ReplacementTransform(old,new)),
                          ('explain equation',duration-.6,lambda:AnimationGroup(Circumscribe(new,color=MUTED,buff=.09),*correspondence))])

    def slider(self,tr,lo,hi,pos,label,color=PRIOR,width=3.2):
        rail=NumberLine(x_range=[lo,hi,hi-lo],length=width,include_ticks=False,color=MUTED).move_to(pos)
        dot=Dot(color=color,radius=.07).add_updater(lambda m:m.move_to(rail.n2p(tr.get_value())))
        nums=number(label+'=',(lambda: np.floor(tr.get_value()+1e-8)) if label=='N' else tr.get_value,np.array(pos)+UP*.35,color,0 if label=='N' else 2)
        g=VGroup(rail,dot,nums);self.add(g);return g

    def dots(self,ax,n,kind='line'):
        x,t=(X,T) if kind=='line' else (XR,TR)
        g=VGroup()
        for i,(a,b) in enumerate(zip(x,t)):
            dot=Dot(ax.c2p(a,b),radius=.048,color=DATA)
            dot.add_updater(lambda m,i=i:m.set_opacity(float(np.clip(n.get_value()-i,0,1))))
            g.add(dot)
        self.add(g);return g

    def candidates(self,ax,n,kind='line',alpha=None,beta=None,phase=None):
        grid=np.linspace(ax.x_range[0],ax.x_range[1],100) if kind=='line' else np.linspace(0,1,150)
        return always_redraw(lambda:VGroup(*[path(ax,grid,phi(grid,kind)@w,c,2.2,.7) for w,c in zip(samples(n.get_value(),alpha.get_value() if alpha else 2,beta.get_value() if beta else 25,kind,phase.get_value() if phase else 0),COLORS)]))

    def aid_card(self, label):
        """Temporarily replace the body while retaining the chapter recap."""
        saved = [m for m in self.mobjects if m is not self.subtitle]
        header = [m for m in saved if m.get_center()[1] > 3]
        self.clear()
        frame = VGroup()
        label = jp(label,23).move_to([-4.85,2.02,0],aligned_edge=LEFT)
        self.add(*header,frame,label)
        return saved,frame,label

    def restore_body(self, saved):
        self.clear()
        self.add(*saved)

    def parameter_recap(self):
        saved,frame,label = self.aid_card('復習: 1.2 パラメータのベイズ更新')
        # 1.2 parameters(): same beta(2,2) prior, three heads, and colours.
        ax = Axes(x_range=[0,1,.5],y_range=[0,2.6,1],x_length=5.5,y_length=2.3,
                  tips=False,axis_config={'color':MUTED,'include_ticks':False}).move_to([-1,.05,0])
        xx = np.linspace(0,1,201)
        prior = path(ax,xx,6*xx*(1-xx),DATA)
        like = path(ax,xx,xx**3,NOISE)
        product = path(ax,xx,6*xx**4*(1-xx),PRIOR)
        post = path(ax,xx,30*xx**4*(1-xx),PRIOR)
        legends = VGroup(jp('事前密度',22,DATA),jp('尤度',22,NOISE),jp('事後密度',22,PRIOR)).arrange(DOWN,buff=.25).move_to([3.2,.3,0])
        formula = tex(r'p(\theta)\,L(\theta)\quad\longrightarrow\quad p(\theta\mid D)',29).move_to([0,-1.7,0])
        mass = jp('面積を1へ',22,PRIOR).move_to([3.2,-1.05,0])
        coin = jp('コインが3回とも表の例',19,MUTED).move_to([-1,1.48,0])
        theta = tex(r'\theta',26).next_to(ax.x_axis,RIGHT,buff=.1)
        density_objects = VGroup(ax,prior,like,legends,formula,coin,theta)
        self.add(density_objects)
        # 2.3 geometry(): green contours, blue points, green/purple principal axes.
        ga = Axes(x_range=[-3,3,1],y_range=[-2.2,2.2,1],x_length=4.2,y_length=3.08,
                  tips=False,axis_config={'color':MUTED,'include_ticks':False}).move_to([-1.75,-.25,0])
        unit = np.array([np.cos(np.linspace(0,TAU,100)),np.sin(np.linspace(0,TAU,100))])
        angle=.6;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        mat=rot@np.diag([1.6,.65])
        ring=path(ga,*unit,POST)
        ellipse=path(ga,*(mat@unit),POST)
        points=np.random.default_rng(233).normal(size=(50,2))*.65
        dots=VGroup(*[Dot(ga.c2p(*p),radius=.025,color=DATA) for p in points])
        target=VGroup(*[Dot(ga.c2p(*p),radius=.025,color=DATA) for p in points@mat.T])
        axes=VGroup(*[Arrow(ga.c2p(0,0),ga.c2p(*mat[:,i]),buff=0,color=c,stroke_width=3)
                      for i,c in enumerate([POST,PRIOR])])
        xlabels=VGroup(tex('x_1',26).next_to(ga.x_axis,RIGHT,buff=.1),tex('x_2',26).next_to(ga.y_axis,UP,buff=.08))
        wlabels=VGroup(tex('w_0',26).move_to(xlabels[0]),tex('w_1',26).move_to(xlabels[1]))
        mapping=VGroup(jp('2つの係数へ',23),tex(r'(x_1,x_2)\ \to\ (w_0,w_1)',29),
                       tex(r'\mu\ \to\ m_N',29,MEAN),tex(r'\Sigma\ \to\ S_N',29,POST)).arrange(DOWN,buff=.28).move_to([2.45,.15,0])
        center=Dot(ga.c2p(0,0),color=MEAN,radius=.07)
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        def switch():
            # AnimationGroup can lift children out of density_objects. Clear all
            # body objects explicitly, retaining only the title, card and captions.
            header = [m for m in saved if m.get_center()[1] > 3]
            self.clear()
            new_label=jp('復習: 2.3 ガウス分布',23).move_to([-4.85,2.02,0],aligned_edge=LEFT)
            self.add(*header,frame,new_label,self.subtitle,ga,ring,dots,xlabels)
            return Wait()
        self.beat(phases=[
            ('R1.2 recall prior and likelihood',a*.24,lambda:Wait()),
            ('R1.2 multiply',a*.26,lambda:Transform(prior,product)),
            ('R1.2 normalize',a*.50,lambda:AnimationGroup(Transform(prior,post),FadeIn(mass))),
            ('R2.3 change source',b*.22,switch),
            ('R2.3 ellipse and axes',b*.78,lambda:AnimationGroup(Transform(ring,ellipse),Transform(dots,target),Create(axes))),
            ('R map coordinates to weights',c*.55,lambda:AnimationGroup(Transform(xlabels,wlabels),FadeIn(mapping),FadeIn(center))),
            ('R connect center and spread',c*.45,lambda:AnimationGroup(Circumscribe(center,color=YELLOW),Circumscribe(ring,color=YELLOW))),
        ])
        self.restore_body(saved)

    def transpose_aid(self):
        saved,_,_ = self.aid_card('補足: 転置は、列を行へ')
        blue,purple,yellow = '#58C4DD','#9A72AC','#FFFF00'
        source=Matrix([['1','x_1'],['1','x_2'],['1','x_3']],h_buff=1.1,v_buff=.65).scale(.75).move_to([-2.6,.05,0])
        target=Matrix([['1','1','1'],['x_1','x_2','x_3']],h_buff=1.1,v_buff=.7).scale(.75).move_to([2.2,.05,0])
        for j,col in enumerate(source.get_columns()):col.set_color([blue,purple][j])
        for i,row in enumerate(target.get_rows()):row.set_color([blue,purple][i])
        self.add(source,tex(r'\Phi',31).move_to([-2.6,1.3,0]),tex(r'\Phi^T',31).move_to([2.2,1.3,0]))
        moving=source.get_entries().copy()
        brackets=target.get_brackets()
        size1=tex(r'3\times2',30,blue).move_to([-2.6,-1.45,0])
        size2=tex(r'2\times3',30,'#83C167').move_to([2.2,-1.45,0])
        arrow=Arrow([- .8,-1.45,0],[.5,-1.45,0],buff=0,color=yellow)
        self.add(size1)
        a,b=[self.sentence_duration(i) for i in range(2)]
        def fold_column(j):
            # Start the working copy in the empty gap between the two matrices;
            # moving it through the source would cross the other source column.
            for i in range(3):
                moving[2*i+j].set_x(-.25)
                self.add(moving[2*i+j])
            return AnimationGroup(*[Transform(moving[2*i+j],target.get_entries()[3*j+i])
                                    for i in range(3)])
        self.beat(phases=[
            ('V08c identify columns',a*.485,lambda:AnimationGroup(Circumscribe(source.get_columns()[0],color=yellow),FadeIn(brackets))),
            ('V08c constant column to row',a*.2575,lambda:fold_column(0)),
            ('V08c input column to row',a*.2575,lambda:fold_column(1)),
            ('V08c dimensions',b*.65,lambda:AnimationGroup(GrowArrow(arrow),FadeIn(size2))),
            ('V08c preserve entries',b*.35,lambda:Circumscribe(moving,color=yellow)),
        ])
        self.restore_body(saved)

    def projection_aid(self):
        saved,_,_ = self.aid_card('補足: 予測へ写した分散')
        blue,yellow,green = '#58C4DD','#FFFF00','#83C167'
        self.add(jp('説明用の例・中心を0へ移す',19,MUTED).move_to([2.8,2.02,0]))
        ax=Axes(x_range=[-2.7,2.7,1],y_range=[-1.7,1.7,1],x_length=4.05,y_length=2.55,
                tips=False,axis_config={'color':MUTED,'include_ticks':False}).move_to([-2.25,-.15,0])
        theta=np.linspace(0,TAU,65);xy=np.array([2*np.cos(theta),np.sin(theta)])
        ring=path(ax,*xy,blue)
        self.add(ax,ring,tex(r'S=\begin{pmatrix}4&0\\0&1\end{pmatrix}',28,blue).move_to([-2.25,1.38,0]))
        # This is the Mahalanobis-radius-one contour, not the support of the cloud.
        self.add(jp('輪は等密度線',18,MUTED).move_to([-2.25,-1.82,0]))
        hx=VGroup(*[DashedLine(ax.c2p(x,y),ax.c2p(x,-1.45),color=yellow,stroke_width=1)
                    for x,y in xy[:,::8].T if abs(y+1.45)>.01])
        hy=VGroup(*[DashedLine(ax.c2p(x,y),ax.c2p(2.4,y),color=yellow,stroke_width=1)
                    for x,y in xy[:,::8].T if abs(x-2.4)>.01])
        shadowx=VGroup(Line(ax.c2p(-2,-1.45),ax.c2p(2,-1.45),color=yellow,stroke_width=5),
                       *[tex(str(v),18,green).next_to(ax.c2p(v,-1.45),DOWN,buff=.09) for v in [-2,0,2]])
        shadowy=VGroup(Line(ax.c2p(2.4,-1),ax.c2p(2.4,1),color=yellow,stroke_width=5),
                       *[tex(str(v),18,green).next_to(ax.c2p(2.4,v),RIGHT,buff=.09) for v in [-1,0,1]])
        horizontal=VGroup(tex(r'\phi=(1,0)^T',29,yellow),jp('標準偏差 2　分散 4',23,green)).arrange(DOWN,buff=.2).move_to([2.5,.95,0])
        vertical=VGroup(tex(r'\phi=(0,1)^T',29,yellow),jp('標準偏差 1　分散 1',23,green)).arrange(DOWN,buff=.2).move_to([2.5,-.3,0])
        result=tex(r'\mathrm{Var}(\phi^Tw)=\phi^TS\phi',28,green).move_to([2.45,-1.42,0])
        general=VGroup(tex(r'\phi=c\,u,\quad\|u\|=1',26,yellow),tex(r'\phi^TS\phi=c^2(u^TSu)',27,green)).arrange(RIGHT,buff=.4).move_to([0,-2.55,0])
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('V09b show horizontal projection',a,lambda:AnimationGroup(Create(hx),Create(shadowx))),
            ('V09b horizontal standard deviation and variance',b*.43,lambda:FadeIn(horizontal)),
            ('V09b vertical projection',b*.57,lambda:AnimationGroup(FadeOut(hx),Create(hy),Create(shadowy),FadeIn(vertical))),
            ('V09b quadratic variance',c*.57,lambda:FadeIn(result)),
            ('V09b general length factor',c*.43,lambda:FadeIn(general)),
        ])
        self.restore_body(saved)

    def question(self):
        ax=self.axes(xr=(-1,1.8,.5));n=ValueTracker(2);a=ValueTracker(.15);b=ValueTracker(.65)
        dots=self.dots(ax,n)
        target=DashedLine(ax.c2p(FORECAST_X,-1.45),ax.c2p(FORECAST_X,1.45),color=YELLOW,stroke_width=2)
        target_label=tex(r'x_*=1.5',26,YELLOW).next_to(target,UP,buff=.06)
        line=always_redraw(lambda:path(ax,[-1,1.8],[a.get_value()-b.get_value(),a.get_value()+1.8*b.get_value()]))
        self.beat(Create(target),FadeIn(target_label),end_sentence=1);self.add(line)
        s0=self.slider(a,-.5,.6,[-2.4,-2.4,0],'w_0',MEAN)
        s1=self.slider(b,-.3,1.3,[2.4,-2.4,0],'w_1',YELLOW)
        self.beat(a.animate.set_value(.45));self.beat(b.animate.set_value(1.1))
        cloud=self.candidates(ax,n)
        estimate=float(predict([FORECAST_X],2)[0][0])
        result=Dot(ax.c2p(FORECAST_X,estimate),radius=.08,color=MEAN)
        result_label=VGroup(jp('予測平均',20,MEAN),tex(f'{estimate:.3f}',26,MEAN)).arrange(RIGHT,buff=.13).move_to([2.5,2.6,0])
        self.beat(FadeOut(line),FadeIn(cloud),FadeIn(result),FadeIn(result_label))
        self.remove(s0,s1);f=self.formula(r't=w_0+w_1x+\epsilon,\qquad',r'\epsilon\sim\mathcal N(0,\beta^{-1})');f[1].set_color(NOISE)
        noise=VGroup(*[Line(ax.c2p(x,.15+.65*x),ax.c2p(x,t),color=NOISE,stroke_width=5) for x,t in zip(X[:2],T[:2])])
        self.beat(Create(noise));self.beat(Indicate(dots[:2],scale_factor=1.1),Circumscribe(f,color=MUTED,buff=.09))

    def map(self):
        left=self.axes((-3.25,.15,0),4.4,3.5,(-1.5,1.5,1),(-1.5,1.5,1),('w_0','w_1'))
        right=self.axes((3.15,.15,0),4.4,3.5,yr=(-2.4,2.4,1))
        a=ValueTracker(-.4);b=ValueTracker(.3);alpha=ValueTracker(2)
        dot=always_redraw(lambda:Dot(left.c2p(a.get_value(),b.get_value()),color=YELLOW,radius=.08))
        line=always_redraw(lambda:path(right,[-1,1],[a.get_value()-b.get_value(),a.get_value()+b.get_value()],YELLOW))
        self.add(dot,line);f=self.formula(r'y(x)=',r'w_0',r'+',r'w_1x');f[1].set_color(MEAN);f[3].set_color(YELLOW)
        self.beat(Indicate(dot),Indicate(line));self.beat(a.animate.set_value(.45));self.beat(b.animate.set_value(1.0))
        self.parameter_recap()
        cs=always_redraw(lambda:contours(left,np.zeros(2),np.eye(2)/alpha.get_value(),PRIOR))
        n=ValueTracker(0);cloud=self.candidates(right,n,alpha=alpha)
        self.remove(dot,line,f);self.formula(r'p(w\mid\alpha)=\mathcal N(w\mid0,\alpha^{-1}I)').set_color(PRIOR)
        self.beat(Create(cs),FadeIn(cloud));self.slider(alpha,2,10,[0,2.53,0],r'\alpha',width=3)
        self.beat(alpha.animate.set_value(10));self.beat(alpha.animate.set_value(2))

    def update(self):
        ax=self.axes((-3.2,.15,0),4.5,3.5,(-1.5,1.5,1),(-1.5,1.5,1),('w_0','w_1'))
        data=self.axes((3.15,.15,0),4.5,3.5,yr=(-2.4,2.4,1))
        n=ValueTracker(0);seen=ValueTracker(0);self.dots(data,seen)
        ellipse=always_redraw(lambda:contours(ax,*posterior(n.get_value())))
        self.add(ellipse);cloud=self.candidates(data,n);self.add(cloud)
        f=self.formula(r'p(w\mid\mathbf t)\propto',r'p(\mathbf t\mid w)',r'p(w)');f[1].set_color(NOISE);f[2].set_color(PRIOR)
        self.beat(seen.animate.set_value(1))
        def stripe(i):
            yy=np.linspace(-1.5,1.5,100); xx=T[i]-X[i]*yy
            return VGroup(*[path(ax,xx+offset,yy,NOISE,2,.18) for offset in np.linspace(-.2,.2,13)])
        st=stripe(0);self.beat(Create(st))
        slope=ValueTracker(-.7)
        pair=lambda:np.array([T[0]-X[0]*slope.get_value(),slope.get_value()])
        dot=always_redraw(lambda:Dot(ax.c2p(*pair()),color=YELLOW,radius=.08))
        ln=always_redraw(lambda:path(data,[-1,1],phi([-1,1])@pair(),YELLOW,4))
        self.add(dot,ln);self.beat(slope.animate.set_value(1.2))
        self.remove(dot,ln);self.beat(n.animate.set_value(1),FadeOut(st))
        st2=stripe(1);self.add(st2);self.beat(n.animate.set_value(2),seen.animate.set_value(2),FadeOut(st2))
        self.remove(f);f=self.formula(r'p(w\mid\mathbf t)=\mathcal N(w\mid',r'm_N',',',r'S_N',')');f[1].set_color(MEAN);f[3].set_color(POST)
        self.beat(Indicate(ellipse),Circumscribe(f,color=MUTED,buff=.09))

    def sequence(self):
        ax=self.axes((-3.2,.25,0),4.5,3.3,(-.35,.5,.2),(0,1.2,.4),('w_0','w_1'))
        data=self.axes((3.15,.25,0),4.5,3.3)
        n=ValueTracker(2);self.dots(data,n)
        cs=always_redraw(lambda:contours(ax,*posterior(n.get_value())));cloud=self.candidates(data,n)
        self.add(cs,cloud);self.slider(n,0,20,[0,2.55,0],'N',DATA)
        self.beat(n.animate.set_value(4));self.beat(n.animate.set_value(8));self.beat(n.animate.set_value(20))
        f=VGroup(tex(r'\Phi',34),jp('の各行：',25),tex(r'(1,x_n)',34),tex(r'\qquad\phi(x)=(1,x)^T',34)).arrange(RIGHT,buff=.12).move_to([0,-2.5,0])
        self.add(f)
        self.beat(Circumscribe(f,color=MUTED,buff=.09))
        g=MathTex(r'S_N^{-1}=',r'\alpha I',r'+',r'\beta\Phi^T\Phi',font_size=32).move_to([0,-2.4,0]);g[1].set_color(PRIOR);g[3].set_color(DATA)
        duration=self.beat_cues()[-1]['end']
        self.beat(phases=[
            ('show precision equation',.6,lambda:ReplacementTransform(f,g)),
            ('explain precision equation',duration-.85,lambda:Circumscribe(g,color=MUTED,buff=.09)),
            ('clear formula before transpose',.25,lambda:FadeOut(g)),
        ])
        self.transpose_aid()
        restored=MathTex(r'S_N^{-1}=',r'\alpha I',r'+',r'\beta\Phi^T\Phi',font_size=32).move_to([0,-2.4,0])
        restored[1].set_color(PRIOR);restored[3].set_color(DATA)
        self.add(restored)
        h=tex(r'm_N=\beta S_N\Phi^T\mathbf t\qquad\text{sequential}=\text{batch}',30).move_to([0,-2.4,0])
        duration=self.beat_cues()[-1]['end']
        self.beat(phases=[
            ('clear precision equation',.25,lambda:FadeOut(restored)),
            ('show posterior mean',.35,lambda:FadeIn(h)),
            ('compare sequential and batch',duration-.6,
             lambda:AnimationGroup(Circumscribe(h,color=MUTED,buff=.09),Indicate(cs))),
        ])

    def regularization(self):
        ax=self.axes((-1.8,.25,0),6.7,3.6);n=ValueTracker(4);alpha=ValueTracker(2);beta=ValueTracker(25)
        self.dots(ax,n);grid=np.linspace(-1,1,100)
        mean=lambda:posterior(4,alpha.get_value(),beta.get_value())[0]
        line=always_redraw(lambda:path(ax,grid,phi(grid)@mean()));self.add(line)
        residuals=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,float((phi([x])@mean())[0])),color=YELLOW,stroke_width=2) for x,t in zip(X[:4],T[:4])]))
        self.add(residuals)
        self.slider(alpha,2,60,[4.2,1.5,0],r'\alpha',width=2.4)
        self.slider(beta,25,100,[4.2,.25,0],r'\beta',NOISE,width=2.4)
        bars=always_redraw(lambda:VGroup(*[Rectangle(width=.42,height=max(.008,v*v*1.3),stroke_width=0,fill_color=PRIOR,fill_opacity=.8).move_to([3.7+j,-1.1+v*v*.65,0]) for j,v in enumerate(mean())]))
        self.add(bars,tex('w_0^2',21,PRIOR).move_to([3.7,-1.4,0]),tex('w_1^2',21,PRIOR).move_to([4.7,-1.4,0]))
        f=self.formula(r'w_{\rm MAP}=m_N');self.beat(Indicate(line),Circumscribe(f,color=MUTED,buff=.09))
        g=MathTex(r'-\ln p(w\mid\mathbf t)=',r'\frac\beta2\sum_n(t_n-w^T\phi_n)^2',r'+',r'\frac\alpha2w^Tw',r'+C',font_size=28).move_to([0,-2.5,0]);g[1].set_color(YELLOW);g[3].set_color(PRIOR)
        self.equation_change(f,g,Indicate(residuals),Indicate(bars,color=PRIOR))
        self.beat(alpha.animate.set_value(60))
        self.add(tex(r'\lambda=\alpha/\beta',28,PRIOR).move_to([4.2,2.5,0]))
        note=self.note('事前を強くすると、係数がゼロへ近づく',PRIOR).move_to([-1.8,2.6,0])
        self.beat(alpha.animate.set_value(2));self.remove(note);self.note('ノイズの精度を上げると、点を強く信じる',NOISE).move_to([-1.8,2.6,0])
        self.beat(beta.animate.set_value(100))
        q=tex(r'p(w\mid\alpha)\propto\exp\!\left(-\frac\alpha2\sum_j|w_j|^q\right)\quad(q>0)',29).move_to([0,-2.5,0])
        q.shift(UP*.2)
        self.equation_change(g,q)

    def prediction(self):
        ax=self.axes(width=8.5);n=ValueTracker(2);cursor=ValueTracker(-.8);noise=ValueTracker(0)
        self.dots(ax,n);grid=np.linspace(-1,1,150)
        calc=lambda:predict(grid,n.get_value())
        cloud=self.candidates(ax,n);self.add(cloud)
        guide=always_redraw(lambda:Line(ax.c2p(cursor.get_value(),-1.6),ax.c2p(cursor.get_value(),1.6),color=MUTED,stroke_width=1))
        self.add(guide);self.beat(cursor.animate.set_value(.3))
        f=self.formula(r'p(t_*\mid x_*,\mathbf t)=\int',r'p(t_*\mid x_*,w)',r'p(w\mid\mathbf t)',r'\,dw');f[1].set_color(NOISE);f[2].set_color(POST)
        self.beat(Indicate(cloud),Circumscribe(f,color=MUTED,buff=.09))
        latent=always_redraw(lambda:band(ax,grid,calc()[0],calc()[1],POST,.35));mean=always_redraw(lambda:path(ax,grid,calc()[0]))
        self.beat(FadeIn(latent),FadeOut(cloud),FadeIn(mean))
        outer=always_redraw(lambda:band(ax,grid,calc()[0],calc()[1]+noise.get_value()/25,NOISE,.16));self.add(outer)
        self.remove(f);f=self.formula(r'\sigma_N^2(x)=',r'\beta^{-1}',r'+',r'\phi(x)^TS_N\phi(x)');f[1].set_color(NOISE);f[3].set_color(POST)
        self.beat(noise.animate.set_value(1))
        self.note('緑：関数の不確かさ　橙：新しい観測の不確かさ')
        self.beat(Circumscribe(f[3],color=MUTED,buff=.09))
        self.projection_aid()
        self.beat(cursor.animate.set_value(.95));self.beat(n.animate.set_value(20))

    def curves(self):
        ax=self.axes(xr=(0,1,.5),yr=(-1.65,1.65,1),width=8.8)
        grid=np.linspace(0,1,160);n=ValueTracker(1)
        bases=VGroup(*[path(ax,grid,phi(grid,'rbf')[:,j],PRIOR,1.5,.65) for j in range(9)])
        f=self.formula(r'y(x,w)=\sum_{j=1}^9w_j\phi_j(x),\qquad\phi_j(x)=e^{-(x-\mu_j)^2/(2s^2)}')
        basis_note=VGroup(jp('中心',21),tex(r'\mu_j',24,PRIOR),jp('と幅',21),tex('s=0.14',24,PRIOR),jp('は固定',21)).arrange(RIGHT,buff=.13).move_to([0,2.65,0]);self.add(basis_note)
        self.beat(LaggedStart(*[Create(g) for g in bases],lag_ratio=.1));self.remove(bases,*bases,basis_note)
        self.dots(ax,n,'rbf');self.add(path(ax,grid,np.sin(2*PI*grid),POST,2,.5))
        calc=lambda:predict(grid,n.get_value(),kind='rbf')
        fill=always_redraw(lambda:band(ax,grid,calc()[0],calc()[2],MEAN,.2));mean=always_redraw(lambda:path(ax,grid,calc()[0]));self.add(fill,mean)
        label=number('N=',lambda:np.floor(n.get_value()+1e-8),[0,2.65,0],DATA,0);self.add(label)
        self.beat(n.animate.set_value(2));self.beat(n.animate.set_value(4));self.beat(n.animate.set_value(25))
        phase=ValueTracker(0);cloud=self.candidates(ax,n,'rbf',phase=phase)
        self.beat(FadeOut(fill),FadeIn(cloud));self.beat(phase.animate.set_value(PI/2))

    def smoother(self):
        ax=self.axes((0,1.05,0),8.5,1.7,(0,1,.5),(-1.5,1.5,1))
        ka=self.axes((0,-1.1,0),8.5,1.35,(0,1,.5),(-.3,.7,.2),(r'x_n',r'k'))
        grid=np.linspace(0,1,160);n=ValueTracker(25);x=ValueTracker(.5)
        self.dots(ax,n,'rbf');self.add(path(ax,grid,predict(grid,25,kind='rbf')[0]))
        weights=lambda:kernel(x.get_value(),XR)[0]
        curve=always_redraw(lambda:path(ka,grid,kernel(x.get_value(),grid)[0],YELLOW,2))
        bars=always_redraw(lambda:VGroup(*[Line(ka.c2p(a,0),ka.c2p(a,b+1e-9),color=YELLOW if b>=0 else PRIOR,stroke_width=3) for a,b in zip(XR,weights())]))
        dot=always_redraw(lambda:Dot(ax.c2p(x.get_value(),weights()@TR),color=YELLOW,radius=.08))
        guide=always_redraw(lambda:DashedLine(ax.c2p(x.get_value(),-1.5),ax.c2p(x.get_value(),1.5),color=YELLOW))
        self.add(curve,bars,dot,guide)
        f=self.formula(r'y(x,m_N)=\sum_n',r'k(x,x_n)',r't_n');f[1].set_color(YELLOW);f[2].set_color(DATA)
        self.beat(x.animate.set_value(.6));self.beat(x.animate.set_value(.15))
        self.remove(f);f=self.formula(r'k(x,x_n)=\beta\phi(x)^TS_N\phi(x_n)');f.set_color(YELLOW)
        self.beat(x.animate.set_value(.85));self.beat(x.animate.set_value(.45))
        self.remove(f);f=self.formula(r"\operatorname{cov}[y(x),y(x')]=\beta^{-1}k(x,x')")
        self.beat(x.animate.set_value(.65));self.remove(f)
        f=number(r'\sum_n k(x,x_n)=',lambda:float(weights().sum()),[0,-2.5,0],YELLOW,4);self.add(f)
        self.beat(x.animate.set_value(.05))

    def limits(self):
        ax=self.axes(xr=(-.8,1.8,.5),yr=(-1.7,1.7,1),width=9)
        grid=np.linspace(-.8,1.8,260);n=ValueTracker(25);x=ValueTracker(.5)
        self.dots(ax,n,'rbf');mu,var,total=predict(grid,25,kind='rbf')
        self.add(band(ax,grid,mu,total,NOISE,.2),path(ax,grid,mu))
        focus=always_redraw(lambda:Line(ax.c2p(x.get_value(),-1.7),ax.c2p(x.get_value(),1.7),color=YELLOW,stroke_width=2));self.add(focus)
        f=self.formula(r'\phi(x)\to0\quad\Longrightarrow\quad m_N^T\phi(x)\to0,\quad\sigma_N^2(x)\to\beta^{-1}')
        self.beat(x.animate.set_value(1.15));self.beat(x.animate.set_value(1.75))
        note=self.note('予測の幅は、モデルの仮定にも依存する',YELLOW)
        self.beat(x.animate.set_value(-.75))
        self.remove(f);f=self.formula(r'\beta\ \mathrm{known}:\ \mathcal N\qquad\longrightarrow\qquad (w,\beta)\ \mathrm{unknown}:\ \mathrm{Student}\ t')
        self.beat(Circumscribe(f,color=MUTED,buff=.09));self.remove(f);f=self.formula(r'k(x,z)=\psi(x)^T\psi(z),\qquad\psi(x)=\sqrt\beta\,S_N^{1/2}\phi(x)')
        self.beat(x.animate.set_value(.5),Circumscribe(f,color=MUTED,buff=.09));self.remove(f)
        # Return to the same held-out input and observation as the opening.
        self.clear()
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]))
        answer_ax=self.axes(xr=(-1,1.8,.5),yr=(-1.7,1.7,1),width=9)
        count=ValueTracker(2)
        self.dots(answer_ax,count)
        xx=np.linspace(-1,1.8,160)
        result=lambda:predict(xx,count.get_value())
        answer_band=always_redraw(lambda:band(answer_ax,xx,result()[0],result()[2],NOISE,.27))
        answer_line=always_redraw(lambda:path(answer_ax,xx,result()[0],MEAN))
        cursor=DashedLine(answer_ax.c2p(FORECAST_X,-1.65),answer_ax.c2p(FORECAST_X,1.65),color=YELLOW,stroke_width=2)
        forecast=tex(r'x_*=1.5',26,YELLOW).next_to(cursor,UP,buff=.04)
        mu2,_,var2=predict([FORECAST_X],2)
        mu20,_,var20=predict([FORECAST_X],20)
        two=jp(f'2点: 平均{mu2[0]:.3f}　予測の幅±{np.sqrt(var2[0]):.3f}',22,MEAN).move_to([0,2.65,0])
        twenty=jp(f'20点: 平均{mu20[0]:.3f}　予測の幅±{np.sqrt(var20[0]):.3f}',22,POST).move_to([0,2.65,0])
        heldout=Dot(answer_ax.c2p(FORECAST_X,HELD_OUT_T),color=DATA,radius=.085)
        heldout_label=tex(f't_*={HELD_OUT_T:.3f}',26,DATA).move_to([2.8,-2.2,0])
        self.add(answer_line,cursor,forecast,two)
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('show two-point uncertainty',a,lambda:FadeIn(answer_band)),
            ('show twenty-point target',min(.5,b*.12),lambda:AnimationGroup(FadeOut(two),FadeIn(twenty))),
            ('add data and narrow prediction',b-min(.5,b*.12),lambda:count.animate.set_value(20)),
            ('check same held-out observation',c,lambda:AnimationGroup(FadeIn(heldout),FadeIn(heldout_label))),
        ])
