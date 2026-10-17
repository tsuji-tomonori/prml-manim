"""PRML 10.7: order dependence, cavity, tilted distribution, revisited sites."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex
from ep_model import OBS, ORDER, REVERSE, exact_moments, exact_unnormalized, run, gaussian, true_factor

BG='#10141F'
BLUE=ManimColor('#58C4DD'); GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
FWD=run(ORDER,1); REV=run(REVERSE,1); EP=run(ORDER,8); EXACT=exact_moments()
PRIOR=dict(mean=0.,variance=100.)
FWD4=dict(mean=float(FWD['history'][3]['posterior_mean']),variance=float(FWD['history'][3]['posterior_variance']))
SWEEPS=[dict(mean=float(EP['history'][5*i+4]['posterior_mean']),
             variance=float(EP['history'][5*i+4]['posterior_variance'])) for i in range(8)]


def J(s,size=19,color=WHITE): return jp(str(s),size,color)

def path(points,color,width=3,opacity=1):
    mob=VMobject().set_points_as_corners([np.array([x,y,0.]) for x,y in points])
    return mob.set_stroke(color,width=width,opacity=opacity)

# The main axes remain visible through all 8 scenes, including the closing beat.
X0,X1=-5.55,1.45; Y0,Y1=-1.73,1.55
T0,T1=-5.,8.

def PX(t): return X0+(t-T0)/(T1-T0)*(X1-X0)
def PY(y): return Y0+y/.85*(Y1-Y0)
def P(t,y): return np.array([PX(t),PY(y),0.])

def axes():
    g=VGroup(Line(P(T0,0),P(T1,0),color=MUTED,stroke_width=1.5),
             Line(P(T0,0),P(T0,.85),color=MUTED,stroke_width=1.5))
    for t in (-5,0,5):
        pt=P(t,0)
        g.add(Line(pt,pt+DOWN*.07,color=MUTED,stroke_width=1),
              J(str(t),15,MUTED).move_to(pt+DOWN*.25))
    for v,label in ((0,'0'),(.4,'0.4'),(.8,'0.8')):
        pt=P(T0,v)
        g.add(Line(pt,pt+LEFT*.06,color=MUTED,stroke_width=1),
              J(label,14,MUTED).move_to(pt+LEFT*.32))
    g.add(tex(r'\theta',26,MUTED).move_to([X1+.20,Y0-.20,0]),
          J('密度',16,MUTED).move_to([X0+.13,Y1+.36,0]))
    # Observation marks are meaningful data marks, not a decorative plot frame.
    g.add(J('観測',16,MUTED).move_to([X0+.16,-2.62,0]))
    for i,x in enumerate(OBS):
        point=np.array([PX(float(x)),-2.54,0])
        g.add(Line(point+UP*.09+LEFT*.07,point+DOWN*.09+RIGHT*.07,color=RED if i==4 else BLUE,stroke_width=2.2),
              Line(point+UP*.09+RIGHT*.07,point+DOWN*.09+LEFT*.07,color=RED if i==4 else BLUE,stroke_width=2.2))
    g.add(J('−3',14,RED).move_to([PX(-3),-2.83,0]),
          J('4.2〜5.4',14,BLUE).move_to([PX(4.8),-2.83,0]))
    return g


def density_values(kind):
    ts=np.linspace(T0,T1,165)
    if kind=='exact': vals=exact_unnormalized(ts)/EXACT['evidence']
    elif kind=='prior': vals=gaussian(ts,0,100)
    elif kind=='moment': vals=gaussian(ts,EXACT['mean'],EXACT['variance'])
    elif kind=='fwd4': vals=gaussian(ts,**FWD4)
    elif kind=='forward': vals=gaussian(ts,FWD['mean'],FWD['variance'])
    elif kind=='reverse': vals=gaussian(ts,REV['mean'],REV['variance'])
    elif kind=='ep': vals=gaussian(ts,EP['mean'],EP['variance'])
    elif kind=='rev1': vals=gaussian(ts,float(REV['history'][0]['posterior_mean']),float(REV['history'][0]['posterior_variance']))
    elif kind.startswith('sweep'):
        state=SWEEPS[int(kind[-1])-1]; vals=gaussian(ts,**state)
    else: raise ValueError(kind)
    return list(zip(ts,np.clip(vals,0,.85)))


def curve(kind,color,width=3,opacity=1):
    return path([(PX(float(t)),PY(float(y))) for t,y in density_values(kind)],color,width,opacity)


def main_overlay(si,bi):
    g=VGroup()
    exact_on=(si in (1,2,6,7) and bi>=2) or (si==5 and bi>=6)
    if exact_on:g.add(curve('exact',RED,2.0,.8))
    if si==0:
        if bi<=1:kind='prior'
        elif bi==2:kind='fwd4'
        elif bi==3:kind='forward'
        elif bi==4:kind='rev1'
        else:kind='reverse'
    elif si==1:kind='moment' if bi>=5 else 'prior'
    elif si==2:kind='ep' if bi>=3 else 'reverse'
    elif si==3:kind='reverse' if bi<=1 else 'forward'
    elif si==4:kind='fwd4'
    elif si==5:kind=['forward','forward','forward','sweep1','sweep2','sweep3','ep','ep'][bi]
    elif si==6:kind='ep'
    else:kind=['prior','reverse','ep','ep','ep','ep','ep','ep'][bi]
    if si==0 and bi>=6 or si==7 and bi>=1:
        other='forward' if kind=='reverse' else 'reverse'
        g.add(curve(other,PURPLE,2,.68))
    g.add(curve(kind,BLUE if kind!='prior' else PURPLE,3.2))
    if si==3 and bi in (2,3):
        # After removing the outlier site from the one-pass forward result,
        # the remaining four observations recover this cavity Gaussian.
        g.add(curve('fwd4',GREEN,2.4,.82))
    if si==4:
        g.add(curve('forward',GREEN,2.4,.90))
    if si==7 and bi>=3:
        for value,color in ((EXACT['mean'],GREEN),(EP['mean'],YELLOW)):
            pt=P(value,0)
            g.add(Line(pt,pt+UP*.16,color=color,stroke_width=2))
    return g


def tiny_axes(x0,x1,y0,y1,xname,yname,xticks,yticks):
    g=VGroup(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.25),
             Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.25))
    for x,label in xticks:
        g.add(Line([x,y0,0],[x,y0-.06,0],color=MUTED,stroke_width=1),
              J(label,13,MUTED).move_to([x,y0-.22,0]))
    for y,label in yticks:
        g.add(Line([x0,y,0],[x0-.06,y,0],color=MUTED,stroke_width=1),
              J(label,13,MUTED).move_to([x0-.27,y,0]))
    if xname:g.add(J(xname,15,MUTED).move_to([x1+.12,y0-.47,0]))
    if yname:g.add(J(yname,15,MUTED).move_to([x0-.22,y1+.24,0]))
    return g


def centers_panel(si,bi):
    g=VGroup(J('同じ5点から得た中心',18,MUTED).move_to([4.52,.91,0]))
    left,right=3.35,5.8
    def xp(v):return left+(v-4)/1.45*(right-left)
    g.add(tiny_axes(left,right,-1.89,.35,'中心','',[(xp(4.0),'4.0'),(xp(4.7),'4.7'),(xp(5.4),'5.4')],[]))
    rows=[('順方向',FWD['mean'],BLUE),('逆順',REV['mean'],PURPLE),('EP',EP['mean'],YELLOW),('真の事後',EXACT['mean'],GREEN)]
    count=(0 if si==0 and bi<3 or si==7 and bi==0 else
           1 if si==0 and bi<5 else 2 if si==0 or si==7 and bi==1 else
           3 if si==7 and bi==2 else 4)
    for i,(name,value,color) in enumerate(rows[:count]):
        y=.16-i*.51
        g.add(J(name,15,color).move_to([2.76,y,0]),
              Line([left,y,0],[xp(value),y,0],color=color,stroke_width=7),
              Dot([xp(value),y,0],radius=.055,color=color),
              J(f'{value:.3f}',15,color).move_to([6.03,y,0]))
    return g


def factor_panel(bi):
    g=VGroup(J('左端の点が与える因子',18,MUTED).move_to([4.5,.91,0]))
    x0,x1=3.15,6.05;y0,y1=-1.87,.42
    g.add(tiny_axes(x0,x1,y0,y1,'θ','高さ',[(x0,'-5'),((x0+x1)/2,'0'),(x1,'5')],[(y0,'0'),(y0+(y1-y0)*.4,'0.1'),(y0+(y1-y0)*.8,'0.2')]))
    coords=lambda t,v:(x0+(t+5)/10*(x1-x0),y0+min(v,.25)/.25*(y1-y0))
    ts=np.linspace(-5,5,101)
    actual=[coords(t,float(true_factor(t,-3))) for t in ts]
    g.add(path(actual,RED,2.7))
    if bi>=2:
        # At the right-hand posterior mass this site is nearly flat.
        g.add(path([coords(t,.0403) for t in ts],BLUE,2.4))
    g.add(J('赤: 元の因子　青: 右側での近似',14,MUTED).move_to([4.56,-2.47,0]))
    return g


def tiles_panel(bi):
    g=VGroup(J('近似因子を1枚外す',18,MUTED).move_to([4.5,.91,0]))
    labels=['事前','4.2','4.8','5.1','5.4','−3']
    xs=[3.0+i*.59 for i in range(6)]
    selected=5
    for i,(label,x) in enumerate(zip(labels,xs)):
        color=PURPLE if i==0 else RED if i==selected else BLUE
        if i==selected and bi>=1:color=YELLOW
        if i==selected and bi>=2:g.add(Line([x,-.58,0],[x,-1.18,0],color=YELLOW,stroke_width=1.6))
        g.add(J(label,15,color).move_to([x,-.4 if i!=selected or bi<2 else -1.32,0]),
              Line([x-.20,-.68,0],[x+.20,-.68,0],color=color,stroke_width=4,
                   stroke_opacity=.15 if i==selected and bi>=2 else 1))
    g.add(J('残り: 事前 + 右側の4点',16,GREEN).move_to([4.52,-1.84,0]))
    return g


def tilted_panel(bi):
    rho=float(FWD['history'][-1]['clean_probability'])
    if bi<4:
        g=VGroup(J('2つの候補の中心',18,MUTED).move_to([4.5,.91,0]))
        x0,x1=3.16,6.05;y=-1.78
        xp=lambda t:x0+(t+3)/10.5*(x1-x0)
        g.add(Line([x0,y,0],[x1,y,0],color=MUTED,stroke_width=1.3),
              J('θ',15,MUTED).move_to([x1+.11,y-.43,0]))
        for t,label in [(-3,'-3'),(0,'0'),(5,'5')]:
            x=xp(t)
            g.add(Line([x,y,0],[x,y-.07,0],color=MUTED,stroke_width=1.1),
                  J(label,14,MUTED).move_to([x,y-.24,0]))
        row=FWD['history'][-1]
        cm,cv,x=row['cavity_mean'],row['cavity_variance'],row['observation']
        clean_mean=cm+cv/(cv+1)*(x-cm)
        g.add(J('信号側',16,RED).move_to([xp(clean_mean),-.54,0]),
              J('背景側',16,BLUE).move_to([xp(cm),-.54,0]),
              Line([xp(clean_mean),-1.38,0],[xp(clean_mean),y,0],color=RED,stroke_width=2.3),
              Line([xp(cm),-1.38,0],[xp(cm),y,0],color=BLUE,stroke_width=2.3),
              Dot([xp(clean_mean),-1.38,0],radius=.065,color=RED),
              Dot([xp(cm),-1.38,0],radius=.065,color=BLUE))
        if bi>=2:
            mean=row['tilted_mean']
            g.add(Dot([xp(mean),-1.08,0],radius=.075,color=GREEN),
                  J('重み付きの支点',15,GREEN).move_to([4.60,-2.49,0]))
        if bi>=3:
            sd=float(np.sqrt(row['tilted_variance']))
            g.add(Line([xp(mean-sd),-1.08,0],[xp(mean+sd),-1.08,0],
                       color=GREEN,stroke_width=2.0))
        return g
    g=VGroup(J('左端が信号である重み',18,MUTED).move_to([4.5,.91,0]))
    x0,x1=3.14,6.05;y=-.45
    g.add(tiny_axes(x0,x1,-1.85,.1,'確率','',[(x0,'0'),((x0+x1)/2,'0.5'),(x1,'1')],[]))
    g.add(Line([x0,y,0],[x0+max(.05,(x1-x0)*rho),y,0],color=YELLOW,stroke_width=10),
          J(f'{rho:.4f}',22,YELLOW).move_to([4.7,y+.08,0]))
    g.add(J('この5点・順方向の初回',15,MUTED).move_to([4.5,-2.47,0]))
    return g

def sweep_panel(bi):
    g=VGroup(J('中心を何巡も見直す',18,MUTED).move_to([4.5,.91,0]))
    x0,x1=3.12,6.02;y0,y1=-1.83,.43
    g.add(tiny_axes(x0,x1,y0,y1,'巡目','中心',[(x0,'1'),(4.77,'5'),(x1,'8')],[(y0,'4.8'),(-.58,'5.0'),(y1,'5.2')]))
    coords=lambda i,v:(x0+(i-1)/7*(x1-x0),y0+(v-4.8)/.4*(y1-y0))
    shown=[1,1,1,1,2,3,8,8][bi]
    pts=[coords(i+1,float(SWEEPS[i]['mean'])) for i in range(shown)]
    if shown>1:g.add(path(pts,GREEN,2.6))
    for x,y in pts:g.add(Dot([x,y,0],radius=.06,color=YELLOW))
    g.add(J(f'{SWEEPS[shown-1]["mean"]:.3f}',18,GREEN).move_to([4.56,-2.48,0]))
    return g


def limit_panel(bi):
    g=VGroup(J('中心と幅は別に確かめる',18,MUTED).move_to([4.5,.91,0]))
    x0,x1=3.2,5.9
    g.add(tiny_axes(x0,x1,-1.91,.44,'分散','',[(x0,'0'),(4.47,'0.3'),(x1,'0.6')],[]))
    for i,(label,value,color) in enumerate([('EP',EP['variance'],BLUE),('真の事後',EXACT['variance'],RED)]):
        y=.05-i*.75
        g.add(J(label,16,color).move_to([2.69,y,0]),
              Line([x0,y,0],[x0+(x1-x0)*value/.6,y,0],color=color,stroke_width=10),
              J(f'{value:.3f}',16,color).move_to([6.03,y,0]))
    return g


def recap_panel(bi):
    if bi==0:
        title=J('復習: 1.6 KLダイバージェンス',17,WHITE).move_to([4.52,.91,0])
        g=VGroup(title,J('説明用の例',14,MUTED).move_to([5.48,.54,0]))
        g.add(tiny_axes(3.20,6.02,-1.84,.28,'位置','確率',
                        [(3.47,'0'),(4.17,'1'),(4.87,'2'),(5.57,'3')],
                        [(-1.84,'0'),(-1.30,'0.2'),(-.76,'0.4')]))
        for i,(a,b) in enumerate(((.4,.2),(.3,.4),(.2,.3),(.1,.1))):
            x=3.4+i*.70
            g.add(Line([x,-1.84,0],[x,-1.84+2.7*a,0],color=BLUE,stroke_width=11),
                  Line([x+.17,-1.84,0],[x+.17,-1.84+2.7*b,0],color=PURPLE,stroke_width=10))
        g.add(J('重みを置く分布で差が変わる',15,MUTED).move_to([4.5,-2.43,0]))
    else:
        title=J('復習: 2.3 ガウス分布',18,WHITE).move_to([4.5,.91,0])
        g=VGroup(title,J('説明用の例',14,MUTED).move_to([5.48,.54,0]))
        x0,x1=3.18,6.02;y0,y1=-1.84,.23
        g.add(tiny_axes(x0,x1,y0,y1,'位置','密度',
                        [(x0,'-3'),((x0+x1)/2,'0'),(x1,'3')],
                        [(y0,'0'),(y0+(y1-y0)*.5,'0.2'),(y1,'0.4')]))
        xs=np.linspace(-3,3,95)
        topos=lambda t,v:(x0+(t+3)/6*(x1-x0),y0+min(v,.4)/.4*(y1-y0))
        g.add(path([topos(t,float(gaussian(t,0,1))) for t in xs],BLUE,2.7),
              Dot([4.6,y0,0],radius=.06,color=YELLOW),
              J('平均が中心、分散が広がり',15,MUTED).move_to([4.5,-2.43,0]))
    g.add(SurroundingRectangle(g,color=YELLOW,buff=.10,stroke_width=1.1))
    return g

def moment_panel(bi):
    g=VGroup(J('平均と分散を合わせる',18,MUTED).move_to([4.5,.91,0]),
             J('説明用の例',14,MUTED).move_to([5.48,.54,0]))
    x0,x1=3.18,6.05;y0,y1=-1.82,.18
    xs=np.linspace(3.52,5.71,4)
    g.add(tiny_axes(x0,x1,y0,y1,'位置','重み',
                    [(float(x),str(i)) for i,x in enumerate(xs)],
                    [(y0,'0'),(y0+.60,'0.2'),(y0+1.20,'0.4')]))
    for x,w in zip(xs,[.12,.23,.42,.23]):
        g.add(Line([x,y0,0],[x,y0+3.0*w,0],color=BLUE,stroke_width=12),
              Dot([x,y0+3.0*w,0],radius=.05,color=YELLOW))
    g.add(J('確率で重み付けした支点',15,GREEN).move_to([4.5,-2.43,0]))
    return g

def side_panel(si,bi):
    if si in (0,7):return centers_panel(si,bi)
    if si==1:return recap_panel(bi) if bi<2 else moment_panel(bi)
    if si==2:return factor_panel(bi)
    if si==3:return tiles_panel(bi)
    if si==4:return tilted_panel(bi)
    if si==5:return sweep_panel(bi)
    return limit_panel(bi)

FORMULAS=[
 ['',r'x=4.2,4.8,5.1,5.4,-3',r'q_0(\theta)=\mathcal N(0,100)',r'\mu_{\mathrm{ADF},\to}=5.131',r'\mu_{\mathrm{ADF},\leftarrow}=4.327',r'5.131-4.327=0.804','',''],
 ['',r'\mathrm{KL}(p\Vert q)',r'\mathrm{KL}(q\Vert p)\;\leftrightarrow\;\mathrm{KL}(p\Vert q)',r'E_p[\theta]',r'\operatorname{Var}_p[\theta]',r'E_q[\theta]=E_p[\theta]',r'\operatorname{Var}_q[\theta]=\operatorname{Var}_p[\theta]','',''],
 [r'p(D,\theta)=p(\theta)\prod_{n=1}^5 f_n(\theta)',r'q(\theta)\propto p(\theta)\prod_{n=1}^5\tilde f_n(\theta)','','','','','',''],
 ['',r'q^{\setminus j}(\theta)\propto q(\theta)/\tilde f_j(\theta)','','',r'f_j(\theta)=p(x_j\mid\theta)',r'\tau_{\setminus j}=\tau-\tau_j',r'\tau=1/v',''],
 ['',r'\hat p_j(\theta)\propto f_j(\theta)q^{\setminus j}(\theta)',r'Z_j=\int f_jq^{\setminus j}\,d\theta',r'E_{\hat p_j}[\theta]',r'\operatorname{Var}_{\hat p_j}[\theta]',r'\rho_j=0.0016','','',''],
 ['',r'\tilde f_j^{\rm new}=Z_jq^{\rm new}/q^{\setminus j}','','',r'1\to2\to3\to\cdots',r'5.131\to4.976\to4.881',r'\mu_{\rm EP}=4.879',r'\tau_j<0',''],
 ['', '',r'\mathrm{KL}(q\Vert p)\; /\;\mathrm{KL}(p\Vert q)','',r'v_{\rm EP}=0.309,\quad v_p=0.568','','',''],
 ['',r'5.131\; /\;4.327',r'\mu_{\rm EP}=4.879',r'\mu_p=4.865',r'|4.879-4.865|=0.014','','',''],
]

class PRML107ExpectationPropagation(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            self.add(axes())
            if si==4:self.add(J('青: 残り　緑: 本物を掛けた後',16,MUTED).move_to([-2.02,2.25,0]))
            self.overlay=main_overlay(si,0);self.add(self.overlay)
            self.side=side_panel(si,0);self.add(self.side)
            self.note=None;self.formula=None
            for bi in range(len(story['beats'])):self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml107_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def run_beat(self,si,bi):
        if self.note is not None:self.remove(self.note)
        self.note=None
        if not (si==1 and bi<2):
            self.note=J(SCENES[si]['beats'][bi]['visual_note'],19,MUTED).move_to([4.52,2.52,0])
            if self.note.width>4.25:self.note.scale(4.25/self.note.width)
            self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        self.formula=None
        formula=FORMULAS[si][bi]
        if formula:
            self.formula=tex(formula,24,YELLOW).move_to([4.52,1.82,0])
            if self.formula.width>4.08:self.formula.scale(4.08/self.formula.width)
            self.add(self.formula)
        target_overlay=main_overlay(si,bi)
        target_side=side_panel(si,bi)
        duration=self.durations[self.beat_index]
        self.beat(phases=[('panel',.48,lambda:Succession(FadeOut(self.side),FadeIn(target_side))),
                          ('graph',duration-.48,lambda:Transform(self.overlay,target_overlay))])
        self.side=target_side
