"""PRML 10.6: a confident boundary that opens away from training data."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, log_expit
from scipy.stats import gamma
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex
from variational_logistic_model import XY, PHI, T, QUERY, NEAR, iterate, prediction, local_bound, hierarchical, lam

BG='#10141F'
BLUE=ManimColor('#58C4DD'); GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
FIT=iterate(); MEAN=FIT['mean']; COV=FIT['covariance']; HISTORY=FIT['history']
NEAR_P=prediction(NEAR,MEAN,COV); FAR_P=prediction(QUERY,MEAN,COV)
HYPER=hierarchical()

def J(s,size=19,color=WHITE):
    return jp(str(s),size,color)

def path(points,color,width=3):
    m=VMobject().set_points_as_corners([np.array([a,b,0.]) for a,b in points])
    return m.set_stroke(color,width=width)

def plot_x(x):return -5.55+(x+2.5)*6.75/5
def plot_y(y):return -2.35+(y+5)*4.65/10
def P(x,y):return np.array([plot_x(x),plot_y(y),0.])

def plot_base():
    g=VGroup()
    g.add(Line(P(-2.5,-5),P(2.5,-5),color=MUTED,stroke_width=1.6),
          Line(P(-2.5,-5),P(-2.5,5),color=MUTED,stroke_width=1.6))
    for x in (-2,0,2):
        pt=P(x,-5)
        g.add(Line(pt,pt+DOWN*.09,color=MUTED,stroke_width=1),
              J(str(x),15,MUTED).move_to(pt+DOWN*.28))
    for y in (-4,0,4):
        pt=P(-2.5,y)
        g.add(Line(pt,pt+LEFT*.09,color=MUTED,stroke_width=1),
              J(str(y),15,MUTED).move_to(pt+LEFT*.36))
    g.add(J('横位置 x',16,MUTED).move_to([1.30,-2.69,0]),
          J('縦位置 y',16,MUTED).move_to([-5.52,2.56,0]))
    for (x,y),t in zip(XY,T):
        center=P(x,y)
        if t==1:
            g.add(Circle(radius=.09,color=RED,stroke_width=2.4).move_to(center))
        else:
            g.add(Line(center+UP*.075+LEFT*.075,center+DOWN*.075+RIGHT*.075,color=BLUE,stroke_width=2.4),
                  Line(center+UP*.075+RIGHT*.075,center+DOWN*.075+LEFT*.075,color=BLUE,stroke_width=2.4))
    return g

def kappa_prob(x,y):
    phi=np.array([1.,x,y]); mu=phi@MEAN; variance=phi@COV@phi
    return float(expit(mu/np.sqrt(1+np.pi*variance/8)))

def contour(level):
    ys=np.linspace(-5,5,101); pts=[]
    for y in ys:
        if kappa_prob(-2.5,y)<=level<=kappa_prob(2.5,y):
            x=brentq(lambda a:kappa_prob(a,y)-level,-2.5,2.5)
            pts.append(P(x,y))
    if len(pts)<2:return VMobject().set_opacity(0)
    return VMobject().set_points_as_corners(pts).set_stroke(GREEN if level==.75 else YELLOW if level==.9 else BLUE if level==.25 else PURPLE,width=2.7)

CONTOURS=[contour(v) for v in (.1,.25,.75,.9)]

def boundary(wy):
    # The mean x-weight is fixed for a transparent one-direction perturbation.
    ys=np.linspace(-5,5,101)
    points=[P(float(x),float(y)) for y in ys if -2.5 <= (x:=-wy*y/MEAN[1]) <=2.5]
    return VMobject().set_points_as_corners(points).set_stroke(PURPLE,width=1.8,opacity=.82)

BOUNDARIES=[boundary(v) for v in (-1.45,-.75,0,.75,1.45)]

def overlay(si,bi):
    g=VGroup()
    show_mean=si in (0,3,4,5,7) and (si!=0 or bi>=3)
    show_contour=(si==0 and bi>=4) or (si==5 and bi>=3) or si==7
    show_samples=(si==0 and bi in (5,6)) or (si==1 and bi==0) or (si==3 and bi in (5,6)) or (si==5 and bi in (1,2,6)) or (si==7 and bi in (2,4))
    for source in CONTOURS:
        mob=source.copy()
        if not show_contour:mob.set_opacity(0)
        g.add(mob)
    for j,source in enumerate(BOUNDARIES):
        mob=source.copy()
        if not show_samples:mob.set_opacity(0)
        if j==2 and show_mean:mob.set_stroke(GREEN,width=3.2,opacity=1)
        g.add(mob)
    if si==0:
        y=0 if bi<2 else 4 if bi>=2 else 0
    elif si in (5,7):
        y=[0,0,2,0,4,4,4,4][bi] if si==5 else 4
    else:y=0
    marker=Circle(radius=.085,stroke_color=YELLOW,stroke_width=3,
                  fill_color=YELLOW,fill_opacity=0 if y==0 else 1).move_to(P(1.4,y))
    if si not in (0,5,7) or (si==0 and bi==0):marker.set_opacity(0)
    g.add(marker)
    guide=Line(P(1.4,0),P(1.4,y),color=YELLOW,stroke_width=1.5)
    if y==0 or si not in (0,5,7):guide.set_opacity(0)
    g.add(guide)
    return g

def tiny_axes(x0,x1,y0,y1,x_name,y_name,x_ticks,y_ticks):
    g=VGroup(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.2),
             Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.2))
    for x,label in x_ticks:
        g.add(Line([x,y0,0],[x,y0-.065,0],color=MUTED,stroke_width=1),
              J(label,13,MUTED).move_to([x,y0-.24,0]))
    for y,label in y_ticks:
        g.add(Line([x0,y,0],[x0-.065,y,0],color=MUTED,stroke_width=1),
              J(label,13,MUTED).move_to([x0-.30,y,0]))
    if x_name:g.add(J(x_name,15,MUTED).move_to([x1+.07,y0-.52,0]))
    if y_name:g.add(J(y_name,15,MUTED).move_to([x0,y1+.36,0]))
    return g

def probability_panel(si,bi):
    g=VGroup(J('同じ横位置 x = 1.4',18,MUTED).move_to([4.45,.89,0]))
    x0,x1=3.65,6.05
    g.add(tiny_axes(x0,x1,-1.93,-.01,'確率','',[(x0,'0'),((x0+x1)/2,'0.5'),(x1,'1')],[]))
    rows=[('代表',.959890,GREEN),('近い点',.936412,BLUE),('上の点',.743531,YELLOW)]
    if (si==0 and bi<4) or (si==7 and bi<1):rows=rows[:1]
    elif si==5 and bi<3:rows=rows[:1]
    elif si==5 and bi<4:rows=rows[:2]
    for i,(name,value,color) in enumerate(rows):
        yy=-.35-i*.63
        g.add(J(name,17,color).move_to([2.92,yy+.07,0]),
              Line([x0,yy,0],[x0+(x1-x0)*value,yy,0],color=color,stroke_width=13),
              J(f'{value:.3f}',17,color).move_to([5.55,yy+.22,0]))
    if si==5 and bi==2:location='移動中: (1.4, 2)'
    elif (si==0 and bi<2) or (si==5 and bi in (0,1,3)):location='近い点: (1.4, 0)'
    else:location='上の点: (1.4, 4)'
    g.add(J(location,16,MUTED).move_to([4.45,-2.43,0]))
    return g

def sigmoid_panel(bi):
    if bi==0:
        recap=J('復習: 4.5',18,WHITE).move_to([4.45,.92,0])
        g=VGroup(SurroundingRectangle(recap,color=YELLOW,buff=.09,stroke_width=1.5),recap)
    else:g=VGroup(J('1点の尤度と局所下界',18,MUTED).move_to([4.45,.92,0]))
    x0,x1=2.92,6.04;y0,y1=-2.18,.45
    g.add(tiny_axes(x0,x1,y0,y1,'スコア','確率',[(x0,'-4'),((x0+x1)/2,'0'),(x1,'4')],[(y0,'0'),(y0+(y1-y0)*.5,'0.5'),(y1,'1')]))
    xs=np.linspace(-4,4,100); xi=1 if bi<5 else 2.3
    coords=lambda x,v:(x0+(x+4)*(x1-x0)/8,y0+v*(y1-y0))
    g.add(path([coords(x,float(expit(x))) for x in xs],RED,3))
    low=path([coords(x,float(local_bound(x,xi))) for x in xs],BLUE,3)
    if bi<4:low.set_opacity(0)
    g.add(low)
    if bi>=4:
        g.add(Dot([*coords(xi,float(expit(xi))),0],radius=.07,color=YELLOW),
              J(f'ξ = {xi:.1f}',17,YELLOW).move_to([4.5,-2.60,0]))
    return g

def factor_panel(bi):
    if bi>=6:
        g=VGroup(J('対数の結合密度',18,MUTED).move_to([4.45,.89,0]))
        x0,x1=3.08,6.02;y0,y1=-2.06,.34
        g.add(tiny_axes(x0,x1,y0,y1,'横の重み','対数',[(x0,'0'),(4.55,'2'),(x1,'4')],[(y0,'-8'),(-.86,'-4'),(y1,'0')]))
        xs=np.linspace(0,4,90)
        def values(weight):
            score=PHI[:,1]*weight
            exact=np.sum(np.where(T==1,log_expit(score),log_expit(-score)))-.2*weight**2
            xi=FIT['xi']; l=lam(xi)
            approx=np.sum(log_expit(xi)-xi/2+l*xi**2+(T-.5)*score-l*score**2)-.2*weight**2
            return exact,approx
        actual=[];lower=[]
        for w in xs:
            a,b=values(w)
            if a>=-8:actual.append((x0+w*(x1-x0)/4,y0+(a+8)*(y1-y0)/8))
            if b>=-8:lower.append((x0+w*(x1-x0)/4,y0+(b+8)*(y1-y0)/8))
        g.add(path(actual,RED,2.9),path(lower,BLUE,2.9),
              J('赤: 元の形　青: 下界',15,MUTED).move_to([4.25,-2.57,0]))
        return g
    g=VGroup(J('各点の −対数下界',18,MUTED).move_to([4.45,.89,0]))
    x0,x1=3.05,6.05;y0=-1.94
    g.add(tiny_axes(x0,x1,y0,.38,'観測の番号','量',[(x0,'1'),((x0+x1)/2,'5'),(x1,'10')],[(y0,'0'),(-.78,'0.03'),(.38,'0.06')]))
    shown=min(10,max(1,bi*2))
    score=PHI@MEAN;xi=FIT['xi'];l=lam(xi)
    log_terms=log_expit(xi)-xi/2+l*xi**2+(T-.5)*score-l*score**2
    heights=-log_terms*(.38-y0)/.06
    for i,h in enumerate(heights):
        x=x0+.10+i*(x1-x0-.20)/9
        bar=Line([x,y0,0],[x,y0+h,0],color=BLUE if i<shown else MUTED,stroke_width=10)
        if i>=shown:bar.set_opacity(.25)
        g.add(bar)
    return g

def weight_panel(bi):
    g=VGroup(J('重みの候補',18,MUTED).move_to([4.45,.9,0]))
    x0,x1=2.94,6.05;y0,y1=-2.05,.32
    g.add(tiny_axes(x0,x1,y0,y1,'横重み','縦重み',[(x0,'0'),(4.45,'2'),(x1,'4')],[(y0,'-2'),(-.85,'0'),(y1,'2')]))
    cx=2.94+MEAN[1]*(x1-x0)/4; cy=-.85
    radius_x=.55 if bi>=1 else 1.00
    radius_y=.70 if bi>=4 else 1.00
    ell=Ellipse(width=2*radius_x,height=2*radius_y,color=BLUE,stroke_width=2.5).move_to([cx,cy,0])
    g.add(ell,Dot([cx,cy,0],radius=.08,color=YELLOW))
    if bi==3:
        g.add(Line([cx-.20,cy-.61,0],[cx+.20,cy+.61,0],color=YELLOW,stroke_width=2.2),
              Dot([cx-.20,cy-.61,0],radius=.055,color=YELLOW),
              Dot([cx+.20,cy+.61,0],radius=.055,color=YELLOW))
    if bi>=5:
        for dx,dy in [(-.35,-.5),(.22,.47),(.4,-.31),(-.18,.32)]:
            g.add(Dot([cx+dx,cy+dy,0],radius=.055,color=PURPLE))
    g.add(J('入力方向へ映した幅' if bi==3 else '中心と幅を持つガウス',16,GREEN).move_to([4.47,-2.48,0]))
    return g

def moment_panel():
    g=VGroup(J('1番の観測の二乗平均',18,MUTED).move_to([4.45,.90,0]))
    x0,x1=3.48,6.07;y0=-1.98
    g.add(tiny_axes(x0,x1,y0,.37,'量','',[(x0,'0'),(4.68,'6'),(x1,'13')],[]))
    mu=float(PHI[0]@MEAN)
    variance=float(PHI[0]@COV@PHI[0])
    values=[('中心の二乗',mu**2,GREEN),('分散',variance,BLUE),('合計',mu**2+variance,YELLOW)]
    for i,(label,value,color) in enumerate(values):
        y=-.23-i*.59
        g.add(J(label,16,color).move_to([2.73,y+.1,0]),
              Line([x0,y,0],[x0+(x1-x0)*value/13,y,0],color=color,stroke_width=12),
              J(f'{value:.1f}',16,color).move_to([5.67,y+.21,0]))
    g.add(J('合計の平方根が ξ',16,MUTED).move_to([4.48,-2.48,0]))
    return g

def lower_panel(bi):
    g=VGroup(J('モデル証拠の下界',18,MUTED).move_to([4.46,.9,0]))
    x0,x1=3.03,6.03;y0,y1=-2.03,.34
    g.add(tiny_axes(x0,x1,y0,y1,'更新回数','下界',[(x0,'0'),(4.52,'5'),(x1,'10')],[(y0,'-4.8'),(-.91,'-4.2'),(y1,'-3.6')]))
    indices=np.arange(min(11,len(HISTORY)))
    coords=[(x0+i*(x1-x0)/10,y0+(HISTORY[i]+4.8)*(y1-y0)/1.2) for i in indices]
    count=[1,2,3,4,6,8,11,11][bi]
    g.add(path(coords[:count] if count>1 else [coords[0],coords[0]],GREEN,3))
    g.add(Dot([*coords[count-1],0],radius=.075,color=YELLOW))
    g.add(J(f'{HISTORY[count-1]:.3f}',17,GREEN).move_to([4.5,-2.48,0]))
    return g

def gamma_panel(bi):
    g=VGroup(J('事前の強さ α',18,MUTED).move_to([4.46,.9,0]))
    x0,x1=3.0,6.03;y0,y1=-2.05,.35
    g.add(tiny_axes(x0,x1,y0,y1,'α','密度',[(x0,'0'),(4.52,'1.5'),(x1,'3')],[(y0,'0'),(-.87,'0.5'),(y1,'1')]))
    xs=np.linspace(.01,3,90)
    coord=lambda x,v:(x0+x/3*(x1-x0),y0+v*(y1-y0))
    g.add(path([coord(x,float(gamma.pdf(x,a=2,scale=.5))) for x in xs],PURPLE,2.5))
    if bi>=4:
        g.add(path([coord(x,float(gamma.pdf(x,a=3.5,scale=1/HYPER[-1,2]))) for x in xs],GREEN,2.8))
    g.add(J('紫: 事前　緑: 更新後',16,MUTED).move_to([4.5,-2.49,0]))
    return g

def side_panel(si,bi):
    if si==4 and bi in (3,4):return moment_panel()
    return [probability_panel,sigmoid_panel,factor_panel,weight_panel,
            lower_panel,probability_panel,gamma_panel,probability_panel][si](si,bi) if si in (0,5,7) else [None,sigmoid_panel,factor_panel,weight_panel,lower_panel,None,gamma_panel][si](bi)

FORMULAS=[
 ['',r'\phi=(1,x,y)^T',r'(1.4,0)\to(1.4,4)',r'\sigma(m^T\phi)=0.960',r'E_q[\sigma(w^T\phi)]=0.744','','',''],
 ['',r'p(w)=N(0,\alpha^{-1}I)',r'p(t)=\int p(t\mid w)p(w)dw','',r'B(a,\xi)\leq\sigma(a)',r'p(t\mid w)=e^{at}\sigma(-a)',r'\log B=C+a/2-\lambda a^2',''],
 ['',r'\xi_1,\ldots,\xi_{10}',r'h(w,\xi)p(w)\leq p(t,w)',r'\log\prod_n h_n=\sum_n\log h_n',r'\log h=C+w^Tb-w^TAw/2','',r'\log h+\log p(w)\leq\log p(t,w)',''],
 [r'q(w)=N(w\mid m_N,S_N)',r'm_N=S_N\sum_n(t_n-1/2)\phi_n',r'S_N^{-1}=S_0^{-1}+2\sum_n\lambda_n\phi_n\phi_n^T',r'\mathrm{Var}(w^T\phi)=\phi^TS_N\phi','','','',''],
 ['',r'\xi\;\to q(w)',r'\xi_n^2=E_q[(w^T\phi_n)^2]',r'E[a^2]=E[a]^2+\mathrm{Var}(a)',r'\xi_n^2=\phi_n^T(S_N+m_Nm_N^T)\phi_n','',r'\mathcal{L}:-4.755\to-3.607',''],
 ['', '',r'\mathrm{Var}(a)=\phi^TS_N\phi',r'(1.4,0):\;0.936',r'(1.4,4):\;0.744',r'E_q[\sigma(a)]\ne\sigma(E_q[a])','',''],
 [r'\alpha=0.4',r'\alpha\sim\mathrm{Gam}(a_0,b_0)',r'q(w,\alpha)=q(w)q(\alpha)',r'S_N^{-1}=E[\alpha]I+2\sum_n\lambda_n\phi_n\phi_n^T',r'q(\alpha)=\mathrm{Gam}(a_N,b_N)','',r'E[\alpha]=0.837',r'\alpha=0.4'],
 ['',r'\sigma(m_N^T\phi)=0.960',r'E_q[\sigma(w^T\phi)]=0.744','','','','',''],
]
# Formula labels are mathematics only; prose screen annotations use Japanese.

class PRML106VariationalLogisticRegression(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            self.add(plot_base())
            self.overlay=overlay(si,0);self.add(self.overlay)
            self.side=side_panel(si,0);self.add(self.side)
            self.note=None;self.formula=None
            for bi in range(len(story['beats'])):self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml106_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def run_beat(self,si,bi):
        if self.note is not None:self.remove(self.note)
        self.note=J(SCENES[si]['beats'][bi]['visual_note'],19,MUTED).move_to([4.45,2.52,0])
        if self.note.width>4.25:self.note.scale(4.25/self.note.width)
        self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        self.formula=None
        formula=FORMULAS[si][bi]
        if formula:
            self.formula=tex(formula,25,YELLOW).move_to([4.45,1.79,0])
            if self.formula.width>4.05:self.formula.scale(4.05/self.formula.width)
            self.add(self.formula)
        target_overlay=overlay(si,bi)
        target_side=side_panel(si,bi)
        duration=self.durations[self.beat_index]
        self.beat(phases=[('panel',.48,lambda:Succession(FadeOut(self.side),FadeIn(target_side))),
                          ('graph',duration-.48,lambda:Transform(self.overlay,target_overlay))])
        self.side=target_side
