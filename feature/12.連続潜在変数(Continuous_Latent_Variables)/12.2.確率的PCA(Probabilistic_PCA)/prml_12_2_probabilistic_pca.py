"""PRML 12.2: the same projection error, different Gaussian density."""
import json
from pathlib import Path
import numpy as np
from manim import *

from narration_content import SCENES
from pca_model import POINTS, MEAN, EIGENVALUES, EIGENVECTORS, THETA
from ppca_model import (U, W, SIGMA2, LAMBDA1, LAMBDA2, A, B, DENSITY_RATIO,
                        posterior, em_fit, missing_x2_given_x1)
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST

BG = '#101923'
BLUE = '#58C4DD'
YELLOW = '#FFFF00'
GREEN = '#83C167'
PURPLE = '#9A72AC'
RED = '#FC6255'
MUTED = '#A6B6C5'
WHITE = '#F4F7FA'
GX, GY, SCALE = -4.37, -.21, .78
V = EIGENVECTORS[:, 1]
EM = em_fit()


def P(v):
    return np.array([GX+SCALE*float(v[0]), GY+SCALE*float(v[1]), 0])


def axes():
    g=VGroup()
    g.add(Line(P([-2.0,0]),P([6.75,0]),color=MUTED,stroke_width=1.7),
          Line(P([0,-2.25]),P([0,4.0]),color=MUTED,stroke_width=1.7))
    for x in (-2,0,2,4,6):
        q=P([x,0]);g.add(Line(q+UP*.045,q+DOWN*.045,color=MUTED,stroke_width=1.2),
                         jp(str(x),13,MUTED).move_to(q+DOWN*.23))
    for y in (-2,0,2,4):
        if y==0:continue
        q=P([0,y]);g.add(Line(q+LEFT*.045,q+RIGHT*.045,color=MUTED,stroke_width=1.2),
                         jp(str(y),13,MUTED).move_to(q+LEFT*.23))
    g.add(tex('x_1',20,MUTED).move_to(P([6.86,0])+DOWN*.16),
          tex('x_2',20,MUTED).move_to(P([0,4.12])+LEFT*.18))
    return g


def principal_line(theta=THETA):
    u=np.array([np.cos(theta),np.sin(theta)])
    low,high=-12.,12.
    for i,bounds in enumerate(((-2.0,6.65),(-2.23,3.9))):
        if abs(u[i])<1e-8:continue
        a,b=sorted(((bounds[0]-MEAN[i])/u[i],(bounds[1]-MEAN[i])/u[i]))
        low,high=max(low,a),min(high,b)
    return Line(P(MEAN+low*u),P(MEAN+high*u),color=PURPLE,stroke_width=3.3)


def scatter():
    return VGroup(*[Dot(P(x),radius=.073,color=BLUE) for x in POINTS])


def ellipse(cov, level=1, color=GREEN, opacity=.6):
    eigen, vectors=np.linalg.eigh(cov)
    order=np.argsort(eigen)[::-1]
    eigen=eigen[order];vectors=vectors[:,order]
    angle=np.arctan2(vectors[1,0],vectors[0,0])
    return Ellipse(width=2*SCALE*level*np.sqrt(eigen[0]),
                   height=max(.04,2*SCALE*level*np.sqrt(max(0,eigen[1]))),
                   color=color,stroke_width=2,stroke_opacity=opacity).rotate(angle).move_to(P(MEAN))


def side_axes(xlabel,ylabel,xticks,yticks,xlim,ylim):
    x0,x1,y0,y1=2.32,6.10,-1.70,.80
    g=VGroup(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.5),
             Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.5))
    def at(x,y):
        return np.array([x0+(x-xlim[0])/(xlim[1]-xlim[0])*(x1-x0),
                         y0+(y-ylim[0])/(ylim[1]-ylim[0])*(y1-y0),0])
    for x,label in xticks:
        q=at(x,ylim[0]);g.add(Line(q+UP*.04,q+DOWN*.04,color=MUTED,stroke_width=1.0),
                              jp(label,13,MUTED).move_to(q+DOWN*.20))
    for y,label in yticks:
        if y==ylim[0]:continue
        q=at(xlim[0],y);g.add(Line(q+LEFT*.04,q+RIGHT*.04,color=MUTED,stroke_width=1),
                              jp(label,13,MUTED).move_to(q+LEFT*.27))
    g.add(jp(xlabel,15,MUTED).move_to([5.96,-2.25,0]),
          jp(ylabel,15,MUTED).move_to([2.37,1.10,0]))
    return g,at


def curve(at,func,lo,hi,color=GREEN,count=90):
    pts=[at(float(t),float(func(t))) for t in np.linspace(lo,hi,count)]
    return VMobject().set_points_as_corners(pts).set_stroke(color,width=2.5)


def error_panel(bi):
    g,at=side_axes('候補','再構成誤差',[(0,'A'),(1,'B')],[(0,'0'),(.5,'0.5'),(1,'1')],(-.5,1.5),(0,1.1))
    for i,color in enumerate((GREEN,YELLOW)):
        x=at(i,0)
        if bi>=2:g.add(Dot(x,radius=.095,color=color),jp('0',17,color).move_to(x+UP*.28))
        else:g.add(jp('?',20,color).move_to(at(i,.5)))
    return g


def prior_panel(z,show_far=False):
    g,at=side_axes('潜在値 z','密度', [(-3,'-3'),(0,'0'),(3,'3')],[(0,'0'),(.2,'0.2'),(.4,'0.4')],(-3.5,3.5),(0,.45))
    normal=lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi)
    g.add(curve(at,normal,-3.5,3.5,GREEN),Dot(at(z,normal(z)),radius=.09,color=YELLOW))
    if show_far:g.add(jp('遠い位置ほど少ない',16,PURPLE).move_to([4.2,-2.55,0]))
    return g


def ratio_panel(progress=1):
    g,at=side_axes('主軸距離','密度比',[(0,'0'),(3,'3'),(6,'6')],[(0,'0'),(.5,'0.5'),(1,'1')],(0,6.2),(0,1.05))
    fn=lambda t: np.exp(-t*t/(2*LAMBDA1))
    if progress>0:
        g.add(curve(at,fn,0,6,GREEN),Dot(at(0,1),radius=.085,color=GREEN))
    if progress>1:g.add(Dot(at(6,DENSITY_RATIO),radius=.09,color=YELLOW),
                        jp('約0.0103',16,YELLOW).move_to([5.34,-.98,0]))
    return g


def recap_panel():
    g=VGroup(Rectangle(width=4.12,height=2.82,stroke_color=YELLOW,stroke_width=1,
                       fill_opacity=0).move_to([4.18,-.42,0]),
             jp('復習: 2.3 ガウス分布',18,WHITE).move_to([4.18,.67,0]),
             Ellipse(width=2.6,height=.80,color=GREEN,stroke_width=2).rotate(THETA).move_to([4.18,-.42,0]),
             jp('共分散で幅と向きを決める',16,GREEN).move_to([4.18,-1.52,0]))
    return g


def variance_panel(mode):
    names=['潜在','雑音','合計']; vals=[LAMBDA1-SIGMA2,SIGMA2,LAMBDA1]
    g,at=side_axes('成分','主軸の分散',list(zip(range(3),names)),[(0,'0'),(2,'2'),(4,'4')],(-.5,2.5),(0,4.4))
    for i,(v,c) in enumerate(zip(vals,(YELLOW,RED,GREEN))):
        if i>mode:continue
        base=at(i,0);end=at(i,v)
        g.add(Rectangle(width=.40,height=max(.025,end[1]-base[1]),fill_color=c,fill_opacity=.8,
                        stroke_width=0).move_to((base+end)/2),
              jp(f'{v:.3f}',14,c).move_to(end+UP*.14))
    return g


def eigen_panel(bi):
    g,at=side_axes('主軸番号','分散',[(0,'1'),(1,'2')],[(0,'0'),(2,'2'),(4,'4')],(-.5,1.5),(0,4.4))
    for i,(v,c) in enumerate(zip((LAMBDA1,LAMBDA2),(GREEN,RED))):
        if bi<1:continue
        base,end=at(i,0),at(i,v)
        g.add(Rectangle(width=.43,height=max(.03,end[1]-base[1]),fill_color=c,fill_opacity=.8,
                        stroke_width=0).move_to((base+end)/2),
              jp(f'{v:.3f}',15,c).move_to(end+UP*.16))
    return g


def sigma_knob(sigma2):
    y=-2.58
    left,right=2.55,4.55
    value=np.clip((float(sigma2)-.04)/.96,0,1)
    g=VGroup(Line([left,y,0],[right,y,0],color=MUTED,stroke_width=2),
             Dot([left+(right-left)*value,y,0],radius=.105,color=YELLOW),
             tex(r'\sigma^2='+f'{sigma2:.3f}',20,YELLOW).move_to([5.32,y,0]))
    return g


def posterior_panel(sigma2,show=True):
    g,at=side_axes('潜在値 z','相対密度',[(-2,'-2'),(0,'0'),(2,'2')],[(0,'0'),(.5,'0.5'),(1,'1')],(-2.5,2.5),(0,1.08))
    g.add(curve(at,lambda t:np.exp(-t*t/2),-2.5,2.5,PURPLE))
    if show:
        observation=MEAN+2.5*U+.4*V
        m,v=posterior(observation,sigma2=sigma2)
        g.add(curve(at,lambda t:np.exp(-(t-m)**2/(2*v)),-2.5,2.5,GREEN),
              Dot(at(m,1),radius=.08,color=YELLOW))
    return g


def em_panel(step):
    g,at=side_axes('更新回数','線の角度',[(0,'0'),(4,'4'),(8,'8')],[(0,'0°'),(20,'20°'),(40,'40°')],(0,8),(0,43))
    vals=[r['angle'] for r in EM[:9]]
    pts=[at(i,float(a)) for i,a in enumerate(vals[:step+1])]
    if len(pts)>1:g.add(VMobject().set_points_as_corners(pts).set_stroke(GREEN,width=2.6))
    g.add(Dot(pts[-1],radius=.09,color=YELLOW),
          jp(f'{vals[step]:.1f}°',15,YELLOW).move_to([5.2,.95,0]))
    return g


def concept_panel(bi):
    g=VGroup()
    if bi<2:
        g.add(jp('条件付き雑音',18,MUTED).move_to([4.2,.63,0]),
              Circle(radius=.40,color=RED,stroke_width=2).move_to([3.25,-.40,0]),
              jp('等方: 同じ幅',16,RED).move_to([3.25,-1.40,0]))
        if bi>=1:g.add(Ellipse(width=1.25,height=.65,color=YELLOW,stroke_width=2).move_to([5.2,-.40,0]),
                       jp('因子分析: 軸別の幅',16,YELLOW).move_to([5.15,-1.40,0]))
    else:
        for i,label in enumerate(('方向1','方向2')):
            y=.35-i*.95
            g.add(jp(label,17,MUTED).move_to([2.9,y,0]),
                  Rectangle(width=2.35 if i==0 else (.18 if bi>=3 else 1.4),height=.22,
                            fill_color=GREEN if i==0 else PURPLE,fill_opacity=.85,stroke_width=0).move_to([4.75,y,0]))
        g.add(jp('不要な方向を0へ',16,YELLOW).move_to([4.45,-1.75,0]))
    return g


FORMULAS=[
 [r'D=2,\ M=1',r'A=\mu,\ B=\mu+6u_1',r'J(A)=J(B)=0',r'\|B-\mu\|=6',r'p(x)\ ?'],
 [r'z\in\mathbb R',r'p(z)=\mathcal N(z\mid0,1)',r'Wz+\mu',r'|z|\uparrow\ \Rightarrow\ p(z)\downarrow',r'\|W\|\simeq1.972',r'W\to-W,\ z\to-z',r'z\to x\ ?'],
 [r'x=Wz+\mu+\epsilon',r'\epsilon\sim\mathcal N(0,\sigma^2I)',r'z\to Wz+\mu\to x',r'p(x\mid z)',r'z\sim\mathcal N(0,1)',r'p(x)=\int p(x\mid z)p(z)\,dz',r'\mathrm{PPCA}'],
 [r'2.3\quad \mathcal N(\mu,C)',r'Wz\quad+\quad\epsilon',r'C=WW^T+\sigma^2I',r'u_2^TCu_2=\sigma^2',r'u_1^TCu_1=\|W\|^2+\sigma^2',r'p(x)=\int p(x\mid z)p(z)\,dz',r'p(x)=\mathcal N(x\mid\mu,C)'],
 [r'\mu_{\rm ML}=\bar x',r'\lambda_1=3.932,\ \lambda_2=0.046',r'\sigma^2_{\rm ML}=\lambda_2',r'W=u_1\sqrt{\lambda_1-\sigma^2}',r'\operatorname{span}(W)=\operatorname{span}(u_1)'],
 [r'x\to z\ ?',r'u_1^T(x-\mu)',r'p(z\mid x)',r'\mathbb E[z\mid x]={W^T(x-\mu)\over W^TW+\sigma^2}',r'\sigma^2=1',r'\sigma^2\simeq0.046',r'\operatorname{var}[z\mid x]={\sigma^2\over W^TW+\sigma^2}'],
 [r'z_n\ ?',r'E: \ \mathbb E[z_n],\ \mathbb E[z_n^2]',r'M: \ W,\sigma^2',r'E\to M\to E\to M',r'0^\circ\to34.9^\circ\to35.4^\circ',r'x_2\ ?',r'p(x_2\mid x_1)'],
 [r'\sigma^2I',r'\Psi=\operatorname{diag}(\psi_1,\ldots,\psi_D)',r'M\ ?',r'w_i\to0',r'D=2,\ M=1'],
 [r'J(A)=J(B)=0',r'\|B-A\|=6',r'{p(B)\over p(A)}=e^{-6^2/(2\lambda_1)}\simeq0.0103',r'p(x)=\mathcal N(x\mid\mu,C)',r'12.3'],
]


def main_extra(si,bi,theta,sigma2):
    g=VGroup()
    if si in (0,8):
        if bi>=1 or si==8:
            for pos,label,color,shift in ((A,'A',GREEN,UP*.28+LEFT*.10),(B,'B',YELLOW,UP*.20+RIGHT*.20)):
                g.add(Dot(P(pos),radius=.105,color=color),jp(label,18,color).move_to(P(pos)+shift))
            if bi>=3 or si==8:g.add(ellipse(np.outer(W,W)+SIGMA2*np.eye(2),2,GREEN,.7))
    elif si==1:
        z=[0,0,1.0,-2.5,3.04,-1.0,0][bi]
        pos=MEAN+W*z
        g.add(Dot(P(pos),radius=.105,color=YELLOW),
              jp(f'z={z:g}',16,YELLOW).move_to(P(pos)+(DOWN*.28+RIGHT*.28 if z>2 else UP*.28+RIGHT*.28)))
        if bi==5:
            g.add(Arrow(P(MEAN),P(MEAN+W*.7),buff=0,color=GREEN,stroke_width=3),
                  Arrow(P(MEAN),P(MEAN-W*.7),buff=0,color=PURPLE,stroke_width=3))
    elif si==2:
        z=[0,0,1.,1.,1.2,1.2,0][bi]
        center=MEAN+W*z
        g.add(Circle(radius=SCALE*np.sqrt(SIGMA2)*2,color=RED,stroke_width=2).move_to(P(center)),
              Dot(P(center),radius=.075,color=YELLOW))
        if bi==3:
            for disp in ([.13,.04],[-.10,.13],[.02,-.17]):
                g.add(Dot(P(center+np.array(disp)),radius=.065,color=RED))
        if bi>=5:
            rng=np.random.default_rng(122)
            for zz,eps in zip(rng.normal(0,1,22),rng.normal(0,np.sqrt(SIGMA2),(22,2))):
                pt=MEAN+W*zz+eps
                g.add(Dot(P(pt),radius=.045,color=GREEN))
            g.add(ellipse(np.outer(W,W)+SIGMA2*np.eye(2),2,GREEN,.65))
    elif si in (3,4):
        if si==3 and bi<2:
            g.add(ellipse(np.outer(W,W),1,PURPLE,.8))
        else:
            g.add(ellipse(np.outer(W,W)+SIGMA2*np.eye(2),1,GREEN,.78),
                  ellipse(np.outer(W,W)+SIGMA2*np.eye(2),2,GREEN,.48))
    elif si==5:
        observed=MEAN+2.5*U+.4*V
        projection=MEAN+2.5*U
        zmean,_=posterior(observed,sigma2=sigma2)
        conditional=MEAN+W*zmean
        g.add(Dot(P(observed),radius=.095,color=BLUE),
              Line(P(observed),P(projection),color=RED,stroke_width=2),
              Dot(P(projection),radius=.079,color=YELLOW))
        if bi>=2:g.add(Dot(P(conditional),radius=.085,color=GREEN),
                       Line(P(projection),P(conditional),color=GREEN,stroke_width=2))
    elif si==6:
        if bi>=5:
            x1=2.5;y=missing_x2_given_x1(x1)
            g.add(Line(P([x1,-.4]),P([x1,1.9]),color=YELLOW,stroke_width=2),
                  Dot(P([x1,y]),radius=.1,color=GREEN),
                  jp('x₂ 欠測',16,YELLOW).move_to(P([x1,1.9])+UP*.25))
        else:
            step=[0,0,1,4,8,8,8][bi]
            record=EM[step]
            w=record['w']
            for pt in POINTS[::2]:
                q=MEAN+w*posterior(pt,w=w,sigma2=record['sigma2'])[0]
                g.add(Line(P(pt),P(q),color=RED,stroke_width=1.5,stroke_opacity=.65),
                      Dot(P(q),radius=.05,color=YELLOW))
    return g


def side_panel(si,bi,sigma2):
    if si==0:return error_panel(bi)
    if si==1:return prior_panel([0,0,1,-2.5,3.04,-1,0][bi],bi>=3)
    if si==2:return prior_panel([0,0,1,1,1.2,1.2,0][bi])
    if si==3:return recap_panel() if bi==0 else variance_panel(min(2,max(0,bi-1)))
    if si==4:return eigen_panel(bi)
    if si==5:return VGroup(posterior_panel(sigma2,bi>=2),sigma_knob(sigma2)) if bi>=4 else posterior_panel(sigma2,bi>=2)
    if si==6:return em_panel([0,0,1,4,8,8,8][bi])
    if si==7:return concept_panel(bi)
    return ratio_panel(0 if bi==0 else 1 if bi==1 else 2)


class PRML122ProbabilisticPCA(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            angle=ValueTracker(0. if si==6 else THETA)
            sigma=ValueTracker(1.)
            self.add(axes(),always_redraw(lambda:principal_line(angle.get_value())),scatter())
            self.add(jp('説明用の13点',14,MUTED).move_to([-5.45,-2.42,0]))
            old=None
            for bi,beat in enumerate(story['beats']):
                if si==5 and bi>=5:sigma.set_value(SIGMA2)
                target_angle=THETA
                if si==6:
                    target_angle=np.radians(EM[[0,0,1,4,8,8,8][bi]]['angle'])
                if si==4 and bi==0:target_angle=0
                formula=tex(FORMULAS[si][bi],26,YELLOW).move_to([4.20,2.48,0])
                if formula.width>4.35:formula.scale(4.35/formula.width)
                note=jp('カーネルPCAへ' if si==8 and bi==4 else beat['visual_note'],16,MUTED).move_to([4.20,1.87,0])
                if note.width>4.3:note.scale(4.3/note.width)
                panel=side_panel(si,bi,sigma.get_value())
                extras=main_extra(si,bi,target_angle,sigma.get_value())
                state=VGroup(formula,note,panel,extras)
                animations=[]
                if old is None:animations.append(FadeIn(state))
                else:animations.append(ReplacementTransform(old,state))
                if abs(angle.get_value()-target_angle)>1e-6:animations.append(angle.animate.set_value(target_angle))
                self.beat(*animations,end_sentence=1,
                          actions=[dict(name='図を更新',start=0,end=self.durations[self.beat_index])])
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml122_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
