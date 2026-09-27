"""PRML 3.3: linked visual experiments in Manim Community."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from bayesian_model import X,T,XR,TR,CENTERS,phi,posterior,predict,samples,kernel

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
    return VGroup(*[path(ax,*(m[:,None]+r*np.linalg.cholesky(c)@unit),color,2,.8) for r in [.6,1.2,1.8]])

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

    def slider(self,tr,lo,hi,pos,label,color=PRIOR,width=3.2):
        rail=NumberLine(x_range=[lo,hi,hi-lo],length=width,include_ticks=False,color=MUTED).move_to(pos)
        dot=Dot(color=color,radius=.07).add_updater(lambda m:m.move_to(rail.n2p(tr.get_value())))
        nums=number(label+'=',tr.get_value,np.array(pos)+UP*.35,color)
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
        grid=np.linspace(-1,1,100) if kind=='line' else np.linspace(0,1,150)
        return always_redraw(lambda:VGroup(*[path(ax,grid,phi(grid,kind)@w,c,1.7,.55) for w,c in zip(samples(n.get_value(),alpha.get_value() if alpha else 2,beta.get_value() if beta else 25,kind,phase.get_value() if phase else 0),COLORS)]))

    def question(self):
        ax=self.axes();n=ValueTracker(0);a=ValueTracker(.15);b=ValueTracker(.65)
        dots=self.dots(ax,n);line=always_redraw(lambda:path(ax,[-1,1],[a.get_value()-b.get_value(),a.get_value()+b.get_value()]))
        self.beat(n.animate.set_value(2));self.add(line)
        s0=self.slider(a,-.5,.6,[-2.4,-2.4,0],'w_0',MEAN)
        s1=self.slider(b,-.3,1.3,[2.4,-2.4,0],'w_1',YELLOW)
        self.beat(a.animate.set_value(.45));self.beat(b.animate.set_value(1.1))
        cloud=self.candidates(ax,n);self.beat(FadeOut(line),FadeIn(cloud))
        self.remove(s0,s1);f=self.formula(r't=w_0+w_1x+\epsilon,\qquad',r'\epsilon\sim\mathcal N(0,\beta^{-1})');f[1].set_color(NOISE)
        noise=VGroup(*[Line(ax.c2p(x,.15+.65*x),ax.c2p(x,t),color=NOISE,stroke_width=5) for x,t in zip(X[:2],T[:2])])
        self.beat(Create(noise));self.beat(Indicate(dots[:2],scale_factor=1.1),Indicate(f,scale_factor=1.02))

    def map(self):
        left=self.axes((-3.25,.15,0),4.4,3.5,(-1.5,1.5,1),(-1.5,1.5,1),('w_0','w_1'))
        right=self.axes((3.15,.15,0),4.4,3.5,yr=(-2.4,2.4,1))
        a=ValueTracker(-.4);b=ValueTracker(.3);alpha=ValueTracker(2)
        dot=always_redraw(lambda:Dot(left.c2p(a.get_value(),b.get_value()),color=YELLOW,radius=.08))
        line=always_redraw(lambda:path(right,[-1,1],[a.get_value()-b.get_value(),a.get_value()+b.get_value()],YELLOW))
        self.add(dot,line);f=self.formula(r'y(x)=',r'w_0',r'+',r'w_1x');f[1].set_color(MEAN);f[3].set_color(YELLOW)
        self.beat(Indicate(dot),Indicate(line));self.beat(a.animate.set_value(.45));self.beat(b.animate.set_value(1.0))
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
        f=self.formula(r'p(w\mid t)\propto',r'p(t\mid w)',r'p(w)');f[1].set_color(NOISE);f[2].set_color(PRIOR)
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
        self.remove(f);f=self.formula(r'p(w\mid t)=\mathcal N(w\mid',r'm_N',',',r'S_N',')');f[1].set_color(MEAN);f[3].set_color(POST)
        self.beat(Indicate(ellipse),Indicate(f))

    def sequence(self):
        ax=self.axes((-3.2,.25,0),4.5,3.3,(-1.4,1.4,1),(-1.4,1.4,1),('w_0','w_1'))
        data=self.axes((3.15,.25,0),4.5,3.3)
        n=ValueTracker(2);self.dots(data,n)
        cs=always_redraw(lambda:contours(ax,*posterior(n.get_value())));cloud=self.candidates(data,n)
        self.add(cs,cloud);self.slider(n,0,20,[0,2.55,0],'N',DATA)
        self.beat(n.animate.set_value(4));self.beat(n.animate.set_value(8));self.beat(n.animate.set_value(20))
        f=self.formula(r'\Phi=\begin{pmatrix}1&x_1\\1&x_N\end{pmatrix}\ (\cdots),\quad\phi(x)=(1,x)^T')
        self.beat(Write(f))
        g=MathTex(r'S_N^{-1}=',r'\alpha I',r'+',r'\beta\Phi^T\Phi',font_size=32).move_to([0,-2.4,0]);g[1].set_color(PRIOR);g[3].set_color(DATA)
        self.beat(ReplacementTransform(f,g))
        h=tex(r'm_N=\beta S_N\Phi^Tt\qquad\text{sequential}=\text{batch}',30).move_to([0,-2.4,0])
        self.beat(ReplacementTransform(g,h),Indicate(cs))

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
        f=self.formula(r'w_{\rm MAP}=m_N');self.beat(Indicate(line),Indicate(f))
        g=MathTex(r'-\ln p(w\mid t)=',r'\frac\beta2\sum_n(t_n-w^T\phi_n)^2',r'+',r'\frac\alpha2w^Tw',r'+C',font_size=28).move_to([0,-2.5,0]);g[1].set_color(YELLOW);g[3].set_color(PRIOR)
        self.beat(ReplacementTransform(f,g),Indicate(residuals),Indicate(bars))
        self.beat(alpha.animate.set_value(60))
        self.add(tex(r'\lambda=\alpha/\beta',28,PRIOR).move_to([4.2,2.5,0]))
        note=self.note('事前を強くすると、係数がゼロへ近づく',PRIOR)
        self.beat(alpha.animate.set_value(2));self.remove(note);self.note('ノイズの精度を上げると、点を強く信じる',NOISE)
        self.beat(beta.animate.set_value(100))
        q=tex(r'p(w\mid\alpha)\propto\exp\!\left(-\frac\alpha2\sum_j|w_j|^q\right)\quad(q>0)',29).move_to([0,-2.5,0])
        self.beat(ReplacementTransform(g,q))

    def prediction(self):
        ax=self.axes(width=8.5);n=ValueTracker(2);cursor=ValueTracker(-.8);noise=ValueTracker(0)
        self.dots(ax,n);grid=np.linspace(-1,1,150)
        calc=lambda:predict(grid,n.get_value())
        cloud=self.candidates(ax,n);self.add(cloud)
        guide=always_redraw(lambda:Line(ax.c2p(cursor.get_value(),-1.6),ax.c2p(cursor.get_value(),1.6),color=MUTED,stroke_width=1))
        self.add(guide);self.beat(cursor.animate.set_value(.3))
        f=self.formula(r'p(t_*\mid x_*,t)=\int',r'p(t_*\mid x_*,w)',r'p(w\mid t)',r'\,dw');f[1].set_color(NOISE);f[2].set_color(POST)
        self.beat(Indicate(cloud),Write(f))
        latent=always_redraw(lambda:band(ax,grid,calc()[0],calc()[1],POST,.35));mean=always_redraw(lambda:path(ax,grid,calc()[0]))
        self.beat(FadeIn(latent),FadeOut(cloud),FadeIn(mean))
        outer=always_redraw(lambda:band(ax,grid,calc()[0],calc()[1]+noise.get_value()/25,NOISE,.16));self.add(outer)
        self.remove(f);f=self.formula(r'\sigma_N^2(x)=',r'\beta^{-1}',r'+',r'\phi(x)^TS_N\phi(x)');f[1].set_color(NOISE);f[3].set_color(POST)
        self.beat(noise.animate.set_value(1))
        self.note('緑：関数の不確かさ　橙：新しい観測の不確かさ')
        self.beat(cursor.animate.set_value(.95));self.beat(n.animate.set_value(20))

    def curves(self):
        ax=self.axes(xr=(0,1,.5),yr=(-2.2,2.2,1),width=8.8)
        grid=np.linspace(0,1,160);n=ValueTracker(1)
        bases=VGroup(*[path(ax,grid,phi(grid,'rbf')[:,j],PRIOR,1.5,.65) for j in range(9)])
        f=self.formula(r'y(x,w)=\sum_{j=1}^9w_j\phi_j(x),\qquad\phi_j(x)=e^{-(x-\mu_j)^2/(2s^2)}')
        self.beat(LaggedStart(*[Create(g) for g in bases],lag_ratio=.1));self.remove(bases)
        self.dots(ax,n,'rbf');self.add(path(ax,grid,np.sin(2*PI*grid),POST,2,.5))
        calc=lambda:predict(grid,n.get_value(),kind='rbf')
        fill=always_redraw(lambda:band(ax,grid,calc()[0],calc()[2],MEAN,.2));mean=always_redraw(lambda:path(ax,grid,calc()[0]));self.add(fill,mean)
        label=number('N=',n.get_value,[0,2.65,0],DATA,0);self.add(label)
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
        self.beat(Write(f));self.remove(f);f=self.formula(r'k(x,z)=\psi(x)^T\psi(z),\qquad\psi(x)=\sqrt\beta\,S_N^{1/2}\phi(x)')
        self.beat(x.animate.set_value(.5),Write(f));self.remove(f)
        f=self.formula(r'p(w)\ \longrightarrow\ p(w\mid t)\ \longrightarrow\ p(t_*\mid x_*,t)')
        self.beat(Write(f),Indicate(focus))
