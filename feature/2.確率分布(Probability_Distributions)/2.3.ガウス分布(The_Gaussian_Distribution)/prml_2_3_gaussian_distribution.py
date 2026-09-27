"""PRML 2.3 — linked numerical experiments in Manim Community."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from gaussian_model import *

BG='#10141F'
BLUE=ManimColor('#58B5ED'); GREEN=ManimColor('#77D49A')
YELLOW=ManimColor('#FFE079'); RED=ManimColor('#FF6B77')
PURPLE=ManimColor('#C29AFF'); MUTED=ManimColor('#A8B2C5')


def line(points,color=BLUE,width=3):
    return VMobject().set_points_as_corners(points).set_stroke(color,width)


def coords(ax,xy):
    xy=np.asarray(xy); o=ax.c2p(0,0)
    return o+xy[:,0,None]*(ax.c2p(1,0)-o)+xy[:,1,None]*(ax.c2p(0,1)-o)


def curve(ax,fn,lo=None,hi=None,color=GREEN):
    u=np.linspace(ax.x_range[0] if lo is None else lo,ax.x_range[1] if hi is None else hi,241)
    return line(coords(ax,np.c_[u,fn(u)]),color)


def outline(ax,cov,radius=1,color=GREEN):
    return line(coords(ax,ellipse(cov,radius)),color)


def readout(label,getter,pos,color=WHITE,places=2,size=25):
    num=DecimalNumber(getter(),num_decimal_places=places,font_size=size,color=color)
    g=VGroup(tex(label,size,color),num).arrange(RIGHT,buff=.12).move_to(pos)
    anchor=num.get_left().copy()
    num.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return g


class PRML23GaussianDistribution(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.shape,self.clt,self.geometry,self.restrictions,self.conditioning,
                                   self.linear_bayes,self.estimation,self.bayesian,self.robust,self.periodic,self.mixtures]):
            self.begin(i)
            if not self.audio_entry:
                raise RuntimeError('Generate matching narration before rendering')
            self.top=None; self.bottom=None
            method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml23_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def equation(self,s,bottom=False,size=29):
        m=tex(s,size).move_to([0,-2.55 if bottom else 2.55,0])
        if m.width>12.6: raise ValueError(f'Equation too wide: {s}')
        attr='bottom' if bottom else 'top'; old=getattr(self,attr)
        setattr(self,attr,m)
        return ReplacementTransform(old,m) if old is not None else FadeIn(m)

    def note(self,s,pos=(0,-2.5,0),color=MUTED,size=23):
        return jp(s,size,color).move_to(pos)

    def ax(self,x=(-4,4,2),y=(0,.65,.2),pos=(0,.1,0),w=9,h=3.3,xlabel='x',ylabel='p(x)'):
        ax=Axes(x_range=x,y_range=y,x_length=w,y_length=h,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.2,'include_ticks':False}).move_to(pos)
        labs=VGroup()
        for v in np.arange(x[0],x[1]+1e-6,x[2]):
            labs.add(tex(f'{v:g}',17,MUTED).move_to(ax.c2p(v,y[0])+DOWN*.22))
        for v in np.arange(y[0],y[1]+1e-6,y[2]):
            if abs(v)>1e-8: labs.add(tex(f'{v:g}',17,MUTED).next_to(ax.c2p(x[0],v),LEFT,buff=.1))
        labs.add(tex(xlabel,22).next_to(ax.c2p(x[1],y[0]),RIGHT,buff=.2))
        labs.add(tex(ylabel,22).next_to(ax.c2p(x[0],y[1]),UP,buff=.12))
        self.add(ax,labs); ax.labels=labs
        return ax

    def slider(self,t,lo,hi,pos,label,color=YELLOW,width=3):
        axis=NumberLine(x_range=[lo,hi],length=width,include_ticks=False,color=MUTED).move_to(pos)
        knob=Dot(axis.n2p(t.get_value()),radius=.07,color=color)
        knob.add_updater(lambda m:m.move_to(axis.n2p(t.get_value())))
        labelmob=tex(label,24,color).next_to(axis,LEFT,buff=.18)
        value=readout('',t.get_value,axis.get_right()+RIGHT*.5,color,size=21)
        return VGroup(axis,knob,labelmob,value)

    def cloud(self,ax,getter,base=BASE_POINTS,color=BLUE):
        dots=VGroup(*[Dot(radius=.026,color=color) for _ in base])
        def update(m):
            for dot,p in zip(m,coords(ax,getter())): dot.move_to(p)
        dots.add_updater(update); update(dots)
        return dots

    def histogram(self,ax,data,bins,color=BLUE):
        counts,edges=np.histogram(data,bins=bins,density=True)
        bars=VGroup()
        for h,l,r in zip(counts,edges[:-1],edges[1:]):
            pts=coords(ax,[[l,0],[l,h],[r,h],[r,0],[l,0]])
            bars.add(Polygon(*pts,stroke_width=.3,stroke_color=BG,fill_color=color,fill_opacity=.6))
        return bars

    def shape(self):
        ax=self.ax(x=(-5,5,2),y=(0,.65,.2),w=9,h=3.1,pos=(0,.05,0))
        mu=ValueTracker(0); sd=ValueTracker(.75)
        graph=always_redraw(lambda:curve(ax,lambda x:normal(x,mu.get_value(),sd.get_value())))
        points=np.random.default_rng(2301).normal(0,.75,60)
        dots=VGroup(*[Dot(ax.c2p(x,.015+(i%3)*.017),radius=.035,color=BLUE) for i,x in enumerate(points)])
        self.add(dots)
        self.beat(Create(graph),self.equation(r'\mathcal N(x\mid\mu,\sigma^2)'))
        self.add(self.slider(mu,-1.5,1.5,(-2.7,-2.25,0),r'\mu',YELLOW,2.5),self.slider(sd,.55,1.4,(2.5,-2.25,0),r'\sigma',PURPLE,2.5))
        marker=always_redraw(lambda:DashedLine(ax.c2p(mu.get_value(),0),ax.c2p(mu.get_value(),normal(mu.get_value(),mu.get_value(),sd.get_value())),color=YELLOW))
        self.add(marker)
        self.beat(mu.animate.set_value(1.2),FadeOut(dots))
        self.beat(sd.animate.set_value(1.35))
        edge=ValueTracker(-.2)
        def area():
            x=np.linspace(-.5,edge.get_value(),81)
            p=coords(ax,np.r_[np.c_[x,normal(x,mu.get_value(),sd.get_value())],[[edge.get_value(),0],[-.5,0]]])
            return Polygon(*p,stroke_width=0,fill_color=BLUE,fill_opacity=.4)
        shade=always_redraw(area); self.add(shade)
        self.beat(edge.animate.set_value(2.5),self.equation(r'P(a<x<b)=\int_a^b p(x)\,dx'))
        self.remove(shade)
        z=ValueTracker(.2)
        pair=always_redraw(lambda:VGroup(*[Dot(ax.c2p(mu.get_value()+sg*z.get_value()*sd.get_value(),normal(mu.get_value()+sg*z.get_value()*sd.get_value(),mu.get_value(),sd.get_value())),color=YELLOW,radius=.06) for sg in [-1,1]]))
        self.add(pair)
        formula=MathTex(r'\mathcal N(x\mid',r'\mu',r',',r'\sigma^2',r')=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]',font_size=29).move_to([0,2.55,0])
        formula[1].set_color(YELLOW); formula[3].set_color(PURPLE)
        self.beat(z.animate.set_value(1.7),ReplacementTransform(self.top,formula));self.top=formula
        self.beat(mu.animate.set_value(0),sd.animate.set_value(.8))

    def clt(self):
        ax=self.ax(x=(0,1,.25),y=(0,4.6,1),xlabel=r'\bar x',h=3.2)
        bars=self.histogram(ax,CLT_MEANS[1],np.linspace(0,1,31))
        self.beat(FadeIn(bars),self.equation(r'\bar x=(x_1+\cdots+x_N)/N\qquad N=1'))
        self.beat(Transform(bars,self.histogram(ax,CLT_MEANS[2],np.linspace(0,1,31))),self.equation(r'N=2\qquad \mathbb E[\bar x]=1/2'))
        gauss=curve(ax,lambda x:normal(x,.5,np.sqrt(1/120)),color=YELLOW)
        self.beat(Transform(bars,self.histogram(ax,CLT_MEANS[10],np.linspace(0,1,31))),Create(gauss),self.equation(r'N=10\qquad \mathrm{var}[\bar x]=1/(12N)'))
        self.remove(ax,ax.labels,bars,gauss)
        ax=self.ax(x=(-4,4,2),y=(0,.48,.2),xlabel='z',h=3.2)
        std=lambda n:(CLT_MEANS[n]-.5)/np.sqrt(1/(12*n))
        bars=self.histogram(ax,std(10),np.linspace(-4,4,41))
        self.beat(FadeIn(bars),self.equation(r'z=\frac{\bar x-1/2}{\sqrt{1/(12N)}}'))
        gauss=curve(ax,normal,color=YELLOW)
        self.beat(Transform(bars,self.histogram(ax,std(32),np.linspace(-4,4,41))),Create(gauss),self.equation(r'N=32\qquad z\ \longrightarrow\ \mathcal N(0,1)'))
        self.beat(Indicate(gauss,color=YELLOW),FadeIn(self.note('独立・同分布・有限の分散')))

    def geometry(self):
        ax=self.ax(x=(-4.6,4.6,2),y=(-3.2,3.2,2),w=5.52,h=3.84,xlabel='x_1',ylabel='x_2',pos=(-1.9,0,0))
        s1=ValueTracker(1);s2=ValueTracker(1);angle=ValueTracker(0)
        mat=lambda:rotation(angle.get_value())@np.diag([s1.get_value(),s2.get_value()])
        dots=self.cloud(ax,lambda:BASE_POINTS@mat().T)
        rings=always_redraw(lambda:VGroup(*[outline(ax,mat()@mat().T,r) for r in [1,2]]))
        self.beat(FadeIn(dots),Create(rings),self.equation(r'\mu=0\qquad \Sigma=I'))
        self.add(self.slider(s1,.65,1.6,(3.5,1.2,0),r'\sqrt{\lambda_1}',GREEN,1.5),self.slider(s2,.65,1.6,(3.5,.3,0),r'\sqrt{\lambda_2}',PURPLE,1.5))
        self.beat(s1.animate.set_value(1.6),s2.animate.set_value(.65),self.equation(r'\Sigma=R\,\mathrm{diag}(\lambda_1,\lambda_2)R^T'))
        self.beat(angle.animate.set_value(.6))
        axes=always_redraw(lambda:VGroup(*[Arrow(ax.c2p(0,0),ax.c2p(*mat()[:,i]),buff=0,color=c,stroke_width=4) for i,c in enumerate([GREEN,PURPLE])]))
        self.beat(Create(axes),self.equation(r'\Sigma u_i=\lambda_i u_i\qquad p(x)/p(\mu)=e^{-1/2}',bottom=True,size=27))
        t=ValueTracker(0)
        dot=always_redraw(lambda:Dot(ax.c2p(*(mat()@np.array([np.cos(t.get_value()),np.sin(t.get_value())]))),color=YELLOW,radius=.08))
        self.add(dot)
        self.beat(t.animate.set_value(2*np.pi),self.equation(r'\Delta^2=(x-\mu)^T\Sigma^{-1}(x-\mu)=\sum_i y_i^2/\lambda_i',bottom=True,size=26))
        self.beat(angle.animate.set_value(0),s1.animate.set_value(1),s2.animate.set_value(1),self.equation(r'\mathcal N(x\mid\mu,\Sigma)=\frac{e^{-\Delta^2/2}}{(2\pi)^{D/2}|\Sigma|^{1/2}}',size=29),self.equation(r'z_i=y_i/\sqrt{\lambda_i}\quad\Rightarrow\quad\Delta^2=\sum_i z_i^2',bottom=True))

    def restrictions(self):
        ax=self.ax(x=(-4,4,2),y=(-3,3,1),pos=(-2.2,0,0),w=5.2,h=3.9,xlabel='x_1',ylabel='x_2')
        off=ValueTracker(.85); diag=ValueTracker(1.8)
        cov=lambda:np.array([[diag.get_value(),off.get_value()],[off.get_value(),.8]])
        points=BASE_POINTS@np.linalg.cholesky(cov()).T
        dots=self.cloud(ax,lambda:points)
        rings=always_redraw(lambda:VGroup(*[outline(ax,cov(),r) for r in [1,1.8]]))
        self.add(dots,rings)
        matrix=Matrix([[r'\sigma_1^2',r'\sigma_{12}'],[r'\sigma_{12}',r'\sigma_2^2']],element_to_mobject=lambda s:tex(s,28)).move_to([3.2,.6,0])
        matrix.get_entries()[1].set_color(YELLOW);matrix.get_entries()[2].set_color(YELLOW)
        self.beat(FadeIn(matrix),self.equation(r'\Sigma=\mathbb E[(x-\mu)(x-\mu)^T]'))
        self.beat(off.animate.set_value(0),self.equation(r'\Sigma=\mathrm{diag}(\sigma_1^2,\ldots,\sigma_D^2)'))
        self.beat(diag.animate.set_value(.8),self.equation(r'\Sigma=\sigma^2I'))
        self.remove(matrix)
        counts=VGroup(jp('一般形',23,GREEN),tex(r'D(D+3)/2',28,GREEN),jp('対角形',23,BLUE),tex('2D',28,BLUE),jp('等方形',23,PURPLE),tex('D+1',28,PURPLE)).arrange(DOWN,buff=.16).move_to([3.4,0,0])
        self.beat(FadeIn(counts),self.equation(r'D+\frac{D(D+1)}2=\frac{D(D+3)}2'))
        replacement=VGroup(jp('一般形',23,GREEN),tex('5150',28,GREEN),jp('対角形',23,BLUE),tex('200',28,BLUE),jp('等方形',23,PURPLE),tex('101',28,PURPLE)).arrange(DOWN,buff=.16).move_to(counts)
        self.beat(Transform(counts,replacement),self.equation(r'D=100'))
        self.beat(off.animate.set_value(.85),diag.animate.set_value(1.8),FadeIn(self.note('相関を表す自由さと、必要なパラメータ数')))

    def conditioning(self):
        ax=self.ax(x=(-3,3,1),y=(-2.5,2.5,1),pos=(-3.2,.1,0),w=4.4,h=3.5,xlabel='x_a',ylabel='x_b')
        dens=self.ax(x=(-3,3,2),y=(0,.55,.2),pos=(3.1,.1,0),w=4.4,h=3.5,xlabel='x_a',ylabel='p')
        rings=VGroup(*[outline(ax,COV,r) for r in [.7,1.4,2]])
        self.beat(Create(rings),self.equation(r'\Sigma=\begin{pmatrix}1.4&0.85\\0.85&1\end{pmatrix}'))
        b=ValueTracker(-1)
        cut=always_redraw(lambda:Line(ax.c2p(-3,b.get_value()),ax.c2p(3,b.get_value()),color=YELLOW))
        conditional_curve=always_redraw(lambda:curve(dens,lambda x:normal(x,*self.cond_sd(b.get_value())),color=YELLOW))
        self.add(cut)
        self.beat(Create(conditional_curve),self.equation(r'p(x_a\mid x_b)=\frac{p(x_a,x_b)}{p(x_b)}'))
        self.add(readout('x_b=',b.get_value,(-3,-2.2,0),YELLOW))
        self.beat(b.animate.set_value(1.2))
        self.beat(b.animate.set_value(-.5),self.equation(r'\mu_{a|b}=\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(x_b-\mu_b)',size=27),self.equation(r'\Sigma_{a|b}=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}=0.6775',bottom=True,size=27))
        sweep=ValueTracker(-2.4)
        scan=always_redraw(lambda:Line(ax.c2p(-3,sweep.get_value()),ax.c2p(3,sweep.get_value()),color=BLUE,stroke_opacity=.5))
        self.add(scan)
        # Integrate joint density from -infinity to the moving scan height.
        from scipy.special import ndtr
        partial=always_redraw(lambda:curve(dens,lambda x:normal(x,0,np.sqrt(1.4))*ndtr((sweep.get_value()-.85/1.4*x)/np.sqrt(1-.85**2/1.4)),color=BLUE))
        self.beat(sweep.animate.set_value(2.4),Create(partial),self.equation(r'p(x_a)=\int p(x_a,x_b)\,dx_b'))
        self.remove(scan,partial)
        marginal=curve(dens,lambda x:normal(x,0,np.sqrt(1.4)),color=BLUE)
        self.add(marginal)
        self.beat(b.animate.set_value(1),self.equation(r'p(x_a)=\mathcal N(x_a\mid\mu_a,\Sigma_{aa})',bottom=True),FadeIn(self.note('周辺',pos=(4.5,1.5,0),color=BLUE,size=21)),FadeIn(self.note('条件付き',pos=(4.5,1.9,0),color=YELLOW,size=21)))

    def cond_sd(self,b):
        m,v=conditional(COV,b);return m,np.sqrt(v)

    def linear_bayes(self):
        ax=self.ax(x=(-3.2,3.2,1),y=(-5.2,5.2,2),pos=(-3,.1,0),w=4.4,h=3.4,xlabel='x',ylabel='y')
        a=ValueTracker(.6);noise=ValueTracker(.9);obs=ValueTracker(1.3)
        base=BASE_POINTS
        dots=self.cloud(ax,lambda:np.c_[base[:,0],a.get_value()*base[:,0]+noise.get_value()*base[:,1]],base=base)
        straight=always_redraw(lambda:curve(ax,lambda x:a.get_value()*x,color=GREEN))
        self.beat(FadeIn(dots),Create(straight),self.equation(r'x\sim\mathcal N(0,1),\quad y=ax+\epsilon,\quad\epsilon\sim\mathcal N(0,s^2)'))
        self.add(self.slider(a,.6,1.4,(3,1.2,0),'a',GREEN,2),self.slider(noise,.35,.9,(3,.3,0),'s',PURPLE,2))
        self.beat(a.animate.set_value(1.4),self.equation(r'\mathbb E[y]=0,\qquad\mathrm{var}[y]=a^2+s^2',bottom=True))
        cut=always_redraw(lambda:Line(ax.c2p(-2.6,obs.get_value()),ax.c2p(2.6,obs.get_value()),color=YELLOW))
        self.beat(Create(cut),self.equation(r'y=1.3\quad\Longrightarrow\quad p(x\mid y)'))
        # The sliders move below the shared posterior plot.
        for mob in list(self.mobjects):
            if isinstance(mob,VGroup) and len(mob)==4: self.remove(mob)
        den=self.ax(x=(-2.5,2.5,1),y=(0,1.75,.5),pos=(3.2,.1,0),w=4.1,h=3.1,xlabel='x',ylabel='p')
        prior=curve(den,normal,color=BLUE)
        post=always_redraw(lambda:curve(den,lambda x:normal(x,linear_posterior(a.get_value(),noise.get_value(),obs.get_value())[0],np.sqrt(linear_posterior(a.get_value(),noise.get_value(),obs.get_value())[1])),color=YELLOW))
        self.beat(Create(prior),Create(post),self.equation(r'p(x\mid y)\propto p(y\mid x)p(x)'))
        self.beat(noise.animate.set_value(.35),self.equation(r'\mathrm{var}[x\mid y]^{-1}=1+a^2/s^2',bottom=True))
        self.beat(obs.animate.set_value(-.5),self.equation(r'S=(\Lambda+A^TLA)^{-1},\quad m=S\{A^TL(y-b)+\Lambda\mu\}',bottom=True,size=25),self.equation(r'p(x)=\mathcal N(\mu,\Lambda^{-1}),\quad p(y\mid x)=\mathcal N(Ax+b,L^{-1})',size=25))

    def estimation(self):
        ax=self.ax(x=(-3,3,1),y=(0,.65,.2),w=8.8,h=3.1)
        mu=ValueTracker(-1);sd=ValueTracker(1.2)
        g=always_redraw(lambda:curve(ax,lambda x:normal(x,mu.get_value(),sd.get_value())))
        dots=VGroup(*[Dot(ax.c2p(x,.025+(i%2)*.025),radius=.045,color=BLUE) for i,x in enumerate(DATA)])
        self.add(dots,g)
        self.beat(mu.animate.set_value(-.6),self.equation(r'\log p(X\mid\mu,\sigma^2)=-\frac N2\log(2\pi\sigma^2)-\frac{\sum_n(x_n-\mu)^2}{2\sigma^2}',size=27))
        residuals=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,.09+.02*i),ax.c2p(mu.get_value(),.09+.02*i),color=YELLOW,stroke_width=1.4) for i,x in enumerate(DATA)]))
        self.add(residuals)
        self.beat(mu.animate.set_value(DATA.mean()),self.equation(r'\mu_{ML}=\frac1N\sum_n x_n',bottom=True))
        self.beat(sd.animate.set_value(DATA.std()),self.equation(r'\Sigma_{ML}=\frac1N\sum_n(x_n-\mu_{ML})(x_n-\mu_{ML})^T',bottom=True,size=27))
        self.remove(g,residuals,dots,ax,ax.labels)
        bx=self.ax(x=(0,6000,2000),y=(0,1.2,.3),xlabel='M',ylabel=r'\overline{\hat\sigma^2}',w=8.8,h=3.1)
        count=ValueTracker(20)
        means=np.cumsum(BIAS)/np.arange(1,len(BIAS)+1)
        progress=always_redraw(lambda:line(coords(bx,np.c_[np.arange(1,int(count.get_value())+1),means[:int(count.get_value())]]),BLUE,2))
        true=Line(bx.c2p(0,1),bx.c2p(6000,1),color=MUTED)
        expect=DashedLine(bx.c2p(0,.75),bx.c2p(6000,.75),color=YELLOW)
        self.add(progress,true,expect)
        self.beat(count.animate.set_value(6000),self.equation(r'N=4\qquad \mathbb E[\hat\sigma^2_{ML}]=\frac{N-1}{N}\sigma^2'),self.equation(r'\sigma^2=1\qquad M=6000',bottom=True))
        corrected=line(coords(bx,np.c_[np.arange(1,6001),means*4/3]),GREEN,2)
        progress.clear_updaters()
        self.beat(Transform(progress,corrected),self.equation(r'\hat\sigma^2=\frac1{N-1}\sum_n(x_n-\bar x)^2',bottom=True))
        # Do not let the running-estimate updater overwrite the corrected curve.
        progress.clear_updaters();self.remove(progress,corrected,true,expect,bx,bx.labels)
        numberline=NumberLine(x_range=[-2,2,1],length=9,include_numbers=True,font_size=22).move_to([0,0,0]);self.add(numberline)
        old=DATA[:4].mean();new=DATA[:5].mean();marker=Dot(numberline.n2p(old),color=YELLOW,radius=.09)
        incoming=Dot(numberline.n2p(DATA[4]),color=BLUE,radius=.09)
        self.add(marker,incoming,self.note('新しい観測',pos=numberline.n2p(DATA[4])+UP*.6,color=BLUE))
        self.beat(marker.animate.move_to(numberline.n2p(new)),self.equation(r'\mu^{(N)}=\mu^{(N-1)}+\frac{x_N-\mu^{(N-1)}}N',bottom=True),self.equation(r'N=5'))

    def bayesian(self):
        ax=self.ax(x=(-2,2.5,1),y=(0,2.3,.5),xlabel=r'\mu',ylabel=r'p(\mu\mid X)',h=3.2)
        mean=ValueTracker(0);variance=ValueTracker(1)
        post=always_redraw(lambda:curve(ax,lambda x:normal(x,mean.get_value(),np.sqrt(variance.get_value())),color=PURPLE))
        prior=curve(ax,normal,color=MUTED).set_opacity(.45)
        self.add(prior)
        self.beat(Create(post),self.equation(r'p(\mu)=\mathcal N(\mu\mid0,1)\qquad N=0'))
        m,v=posterior(1)
        self.beat(mean.animate.set_value(m),variance.animate.set_value(v),self.equation(r'\sigma^2=0.36\quad\text{(known)}\qquad N=1'))
        marks=VGroup(*[Dot(ax.c2p(x,.02+.035*(i%2)),color=BLUE,radius=.035) for i,x in enumerate(BAYES_DATA)])
        m,v=posterior(10)
        self.beat(mean.animate.set_value(m),variance.animate.set_value(v),FadeIn(marks),self.equation(r'N=10\qquad p(\mu\mid X)\propto p(X\mid\mu)p(\mu)'))
        self.beat(Indicate(post,color=PURPLE),self.equation(r'\frac1{\sigma_N^2}=\frac1{\sigma_0^2}+\frac N{\sigma^2}',bottom=True),self.equation(r'\mu_N=\sigma_N^2\left(\frac{\mu_0}{\sigma_0^2}+\frac{\sum_n x_n}{\sigma^2}\right)'))
        self.remove(ax,ax.labels,post,prior,marks)
        ax=self.ax(x=(0,12,3),y=(0,.48,.2),xlabel=r'\lambda',ylabel=r'p(\lambda\mid X)',h=3.2)
        gamma_prior=curve(ax,lambda x:gamma_posterior(x,0),color=MUTED)
        gamma_post=curve(ax,lambda x:gamma_posterior(x,10),color=PURPLE)
        self.add(gamma_prior)
        self.beat(Transform(gamma_prior,gamma_post),self.equation(r'\lambda=1/\sigma^2,\quad p(\lambda\mid X)=\mathrm{Gam}(\lambda\mid a_N,b_N)',size=27),self.equation(r'a_N=a_0+N/2,\qquad b_N=b_0+\tfrac12\sum_n(x_n-\mu)^2',bottom=True,size=27))
        self.beat(Indicate(gamma_prior,color=PURPLE),self.equation(r'p(\mu,\lambda)=\mathcal N(\mu\mid\mu_0,(\beta\lambda)^{-1})\,\mathrm{Gam}(\lambda\mid a,b)',size=27),self.equation(r'p(\mu,\Lambda)=\mathcal N(\mu\mid\mu_0,(\beta\Lambda)^{-1})\,\mathcal W(\Lambda\mid W,\nu)',bottom=True,size=27))

    def robust(self):
        ax=self.ax(x=(-3,8,2),y=(0,.9,.3),w=9,h=3.2)
        outlier=ValueTracker(0)
        dots=VGroup(*[Dot(ax.c2p(x,.02+.035*(i%2)),color=BLUE,radius=.035) for i,x in enumerate(ROBUST_DATA)])
        od=always_redraw(lambda:Dot(ax.c2p(outlier.get_value(),.04),color=RED,radius=.075))
        gc=always_redraw(lambda:curve(ax,lambda x:normal(x,*fit_at(outlier.get_value())[:2]),color=GREEN))
        self.add(dots,od,gc)
        self.beat(outlier.animate.set_value(7),self.equation(r'\text{Gaussian: }\ -\log p(x)\sim(x-\mu)^2'))
        tc=always_redraw(lambda:curve(ax,lambda x:student(x,*fit_at(outlier.get_value())[2:],3),color=PURPLE))
        self.add(self.note('ガウス',pos=(3.5,1.55,0),color=GREEN),self.note('t 分布（自由度 3）',pos=(3.5,1.1,0),color=PURPLE))
        self.beat(Create(tc),self.equation(r'\nu=3\quad\text{(fixed)}',bottom=True))
        self.remove(dots,od,gc,tc)
        for mob in list(self.mobjects):
            if isinstance(mob,Text) and mob.get_center()[0]>2: self.remove(mob)
        components=VGroup(*[curve(ax,lambda x,s=s:normal(x,0,s),color=MUTED).set_opacity(.38) for s in [.5,.8,1.3,2.2]])
        tcurve=curve(ax,lambda x:student(x,0,1,3),color=PURPLE)
        self.beat(Create(components),Create(tcurve),self.equation(r'p(x)=\int_0^\infty\mathcal N(x\mid\mu,\tau^{-1})\,\mathrm{Gam}(\tau\mid a,b)\,d\tau',size=25),self.equation(r'\nu=2a,\quad\lambda=a/b',bottom=True))
        self.remove(components,tcurve)
        nu=ValueTracker(3)
        dynamic=always_redraw(lambda:curve(ax,lambda x:student(x,0,1,nu.get_value()),color=PURPLE));self.add(dynamic)
        target=curve(ax,normal,color=GREEN).set_opacity(.5)
        self.beat(nu.animate.set_value(40),Create(target),self.equation(r'\mathrm{St}(x\mid\mu,\lambda,\nu)=\frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)}\sqrt{\frac\lambda{\pi\nu}}\left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-(\nu+1)/2}',size=25),self.equation(r'\nu\to\infty:\quad\mathcal N(\mu,\lambda^{-1})',bottom=True))
        self.beat(nu.animate.set_value(3),self.equation(r'\mathrm{var}[x]=\frac\nu{\nu-2}\lambda^{-1}\quad(\nu>2)',bottom=True))
        self.remove(dynamic,target);outlier.set_value(7);self.add(dots,od,gc,tc)
        self.beat(outlier.animate.set_value(0),self.equation(r'\nu=3\qquad\text{Gaussian / Student }t',bottom=True))

    def periodic(self):
        center=np.array([-2.6,0.,0.]);radius=1.35
        circle=Circle(radius=radius,color=MUTED).move_to(center)
        pt=lambda a:center+radius*np.array([np.cos(a),np.sin(a),0])
        d1=Dot(pt(np.deg2rad(5)),color=BLUE,radius=.075);d2=Dot(pt(np.deg2rad(355)),color=YELLOW,radius=.075)
        wrong=Arrow(center,pt(np.pi),buff=0,color=RED)
        self.add(circle,d1,d2)
        eq=self.equation(r'(5^\circ+355^\circ)/2=180^\circ')
        self.beat(GrowArrow(wrong),eq)
        vectors=VGroup(Arrow(center,d1.get_center(),buff=0,color=BLUE),Arrow(center,d2.get_center(),buff=0,color=YELLOW))
        good=Arrow(center,(d1.get_center()+d2.get_center())/2,buff=0,color=GREEN)
        self.beat(FadeOut(wrong),GrowArrow(vectors[0]),GrowArrow(vectors[1]),GrowArrow(good),self.equation(r'\bar v=\frac1N\sum_n(\cos\theta_n,\sin\theta_n)'))
        self.beat(Indicate(good,color=GREEN),self.equation(r'\bar\theta=\mathrm{atan2}\left(\sum_n\sin\theta_n,\sum_n\cos\theta_n\right)=0',bottom=True,size=27))
        self.remove(vectors,good,d1,d2)
        concentration=ValueTracker(0);direction=ValueTracker(0)
        def polar():
            t=np.linspace(0,2*np.pi,241);r=radius+.6*von_mises(t,direction.get_value(),concentration.get_value())
            return line(center+np.c_[r*np.cos(t),r*np.sin(t),np.zeros(len(t))],PURPLE)
        graph=always_redraw(polar)
        self.add(self.note('円からの距離が密度',pos=(3,.8,0),size=24),self.slider(concentration,0,5,(3,-.2,0),'m',PURPLE,2.2))
        self.beat(Create(graph),self.equation(r'p(\theta+2\pi)=p(\theta)'),self.equation(r'p(\theta\mid\theta_0,m)=\frac{e^{m\cos(\theta-\theta_0)}}{2\pi I_0(m)}',bottom=True))
        self.beat(concentration.animate.set_value(5))
        self.beat(direction.animate.set_value(2*np.pi),self.equation(r'0\leq\theta<2\pi\qquad\int_0^{2\pi}p(\theta)\,d\theta=1'))

    def mixtures(self):
        ax=self.ax(x=(-4,4,2),y=(0,.5,.2),w=9,h=3.2)
        dots=VGroup(*[Dot(ax.c2p(x,.012+(i%3)*.013),color=BLUE,radius=.025) for i,x in enumerate(MIX_DATA)])
        single=curve(ax,lambda x:normal(x,MIX_DATA.mean(),MIX_DATA.std()),color=GREEN)
        self.add(dots)
        self.beat(Create(single),self.equation(r'K=1'))
        weight=ValueTracker(.45)
        mixture_curve=always_redraw(lambda:curve(ax,lambda x:mixture(x,weight.get_value()),color=PURPLE))
        components=always_redraw(lambda:VGroup(curve(ax,lambda x:weight.get_value()*normal(x,-1.7,.55),color=BLUE),curve(ax,lambda x:(1-weight.get_value())*normal(x,1.5,.7),color=YELLOW)))
        self.beat(FadeOut(single),Create(components),Create(mixture_curve),self.equation(r'p(x)=\sum_{k=1}^K\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)'))
        self.add(self.slider(weight,.2,.65,(0,-2.15,0),r'\pi_1',BLUE,3.5))
        self.beat(weight.animate.set_value(.65),self.equation(r'\pi_k\geq0,\qquad\sum_k\pi_k=1'))
        self.remove(dots)
        point=ValueTracker(-1.7)
        marker=always_redraw(lambda:DashedLine(ax.c2p(point.get_value(),0),ax.c2p(point.get_value(),mixture(point.get_value(),weight.get_value())),color=WHITE))
        self.add(marker)
        self.beat(self.equation(r'\gamma_k(x)=\frac{\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)}{\sum_l\pi_l\mathcal N(x\mid\mu_l,\Sigma_l)}'),Indicate(marker))
        barbase=np.array([-1.5,-2.15,0])
        for mob in list(self.mobjects):
            if isinstance(mob,VGroup) and len(mob)==4: self.remove(mob)
        def rbar():
            r=float(responsibility(point.get_value(),weight.get_value()))
            return VGroup(Line(barbase,barbase+RIGHT*3*r,color=BLUE,stroke_width=18),Line(barbase+RIGHT*3*r,barbase+RIGHT*3,color=YELLOW,stroke_width=18))
        bar=always_redraw(rbar);self.add(bar,readout(r'\gamma_1=',lambda:responsibility(point.get_value(),weight.get_value()),(-3.7,-2.15,0),BLUE),readout(r'\gamma_2=',lambda:1-responsibility(point.get_value(),weight.get_value()),(3.7,-2.15,0),YELLOW))
        self.beat(point.animate.set_value(1.7))
        self.beat(point.animate.set_value(0),self.equation(r'\log p(X)=\sum_n\log\left[\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)\right]',size=28))
