"""PRML 11.5: one long, almost-energy-preserving proposal."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene,jp,tex
from hmc_model import leapfrog,START_Z,START_R,EPSILON,STEPS,energy,gaussian_density,acceptance,leapfrog_anisotropic

BG='#10141F'
BLUE=ManimColor('#58C4DD');GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00');PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255');MUTED=ManimColor('#A8B2C5')
STATES=leapfrog(START_Z,START_R)
ANISO=leapfrog_anisotropic()
PHASES={0:[0,0,0,0,0,0],1:[0,0,0,0,0,0],2:[0,3,7,12,16,18],3:[0,1,3,7,12,18],4:[0,4,8,12,16,18],5:[18,18,18,18,18,18],6:[0,0,0,6,12,18],7:[0,0,0,0,0,0],8:[0,0,8,18,18,18]}
FORMULAS=[
 [r'p(z)=\mathcal N(z\mid0,1)',r'z_0=-2,\ z_1\simeq2.02',r'\Delta z\simeq4.02',r'A\simeq0.999735',r'(z,r)\longrightarrow(z\prime,r\prime)',r'r_0=0.5'],
 [r'r\sim\mathcal N(0,1)',r'r_0=0.5',r'(z,r)',r'E(z)=-\log p(z)+C',r'H=E(z)+K(r)',r'K(r)=r^2/2'],
 [r'p(z)\propto e^{-E(z)}',r'E(z)\downarrow,\ K(r)\uparrow',r'E(z)\simeq0',r'E(z)\uparrow,\ K(r)\downarrow',r'H=E(z)+K(r)',r'\Delta H\simeq0'],
 [r'F(z)=-E\prime(z)',r'E\prime(z)=z',r'r\leftarrow r-\varepsilon E\prime(z)/2',r'z\leftarrow z+\varepsilon r',r'r\leftarrow r-\varepsilon E\prime(z)/2',r'\varepsilon=0.15,\ L=18'],
 [r'H(z,r)=H_0',r'(z,r)\to(z\prime,r\prime)',r'\det J=1',r'p(z,r)\propto e^{-H(z,r)}',r'+\varepsilon\ \longleftrightarrow\ -\varepsilon',r'\Delta H\ne0'],
 [r'(z\prime,r\prime)\simeq(2.02345,0.39520)',r'A=\min(1,e^{H_0-H_1})',r'\varepsilon\uparrow\ \Rightarrow\ |\Delta H|\uparrow',r'p(x)T(x\to y)=p(y)T(y\to x)',r'A=0\Rightarrow z\prime=z',r'r\sim\mathcal N(0,1)'],
 [r'H=H_0',r'r\sim\mathcal N(0,1)',r'r\to\tilde r',r'z\to z\prime',r'\varepsilon\in[a,b]',r'(z,r)\to(z\prime,r\prime)'],
 [r'\sigma_{\max}=2,\ \sigma_{\min}=0.4',r'\sigma_{\max}/\sigma_{\min}=5',r'\varepsilon<\sigma_{\min}',r'L=O(\sigma_{\max}/\sigma_{\min})',r'N_{\rm RW}=O((\sigma_{\max}/\sigma_{\min})^2)',r'5:25'],
 [r'z_0=-2,\ r_0=0.5',r'\varepsilon=0.15,\ L=18',r'z_1=2.02345,\ \Delta H=0.000265',r'u=0.5<A=0.999735',r'\mathrm{HMC}:(z,r)\to(z\prime,r\prime)',r'Z=\int e^{-E(z)}dz']]

def J(s,size=19,color=WHITE):return jp(str(s),size,color)
def coord(z,y):return np.array([-2.52+1.14*z,-1.73+y,0.])
def curve(points,color=RED,width=3,opacity=1):
    m=VMobject();m.set_points_as_corners(points);m.set_stroke(color,width=width,opacity=opacity);return m

def axis(mode):
    g=VGroup()
    if mode=='ellipse':
        g.add(Line([-6.05,-1.75,0],[1.05,-1.75,0],color=MUTED,stroke_width=1.5),Line([-2.50,-2.95,0],[-2.50,1.12,0],color=MUTED,stroke_width=1.5))
        for n in (-1,0,1):
            x=-2.50+1.30*n;q=np.array([x,-1.75,0]);g.add(Line(q,q+DOWN*.06,color=MUTED,stroke_width=1),J(n,14,MUTED).move_to(q+DOWN*.23+(RIGHT*.23 if n==0 else ORIGIN)))
        for n in (-.1,.1):
            y=-1.75+2.65*n;q=np.array([-2.50,y,0]);g.add(Line(q,q+LEFT*.07,color=MUTED,stroke_width=1),J(f'{n:g}',14,MUTED).move_to(q+LEFT*.30))
        g.add(tex('z_1',23,MUTED).move_to([1.18,-1.72,0]),tex('z_2',23,MUTED).move_to([-2.49,1.31,0]))
        for rad,opa in [(.6,.45),(1.0,.85)]:
            pts=[[-2.50+2.60*rad*np.cos(t),-1.75+1.06*rad*np.sin(t),0] for t in np.linspace(0,2*np.pi,120)]
            g.add(curve(pts,RED,2.6,opa))
        return g
    if mode=='phase':
        g.add(Line(coord(-2.7,.94*1.4),coord(2.7,.94*1.4),color=MUTED,stroke_width=1.5),Line(coord(0,-.8),coord(0,3.4),color=MUTED,stroke_width=1.5))
        for n in (-1,0,1):
            q=coord(n,.94*1.4);g.add(Line(q,q+DOWN*.06,color=MUTED,stroke_width=1),J(n,14,MUTED).move_to(q+DOWN*.33+(RIGHT*.23 if n==0 else ORIGIN)))
        for n in (-1,1):
            q=coord(0,.94*(n+1.4));g.add(Line(q,q+LEFT*.07,color=MUTED,stroke_width=1),J(n,14,MUTED).move_to(q+LEFT*.29))
        g.add(tex('z',23,MUTED).move_to(coord(2.78,.94*1.4)),tex('r',23,MUTED).move_to(coord(0,3.54)))
        for h,opa in [(1.0,.45),(2.125,.85)]:
            radius=np.sqrt(2*h)
            pts=[coord(radius*np.cos(t),.94*(radius*np.sin(t)+1.4)) for t in np.linspace(0,2*np.pi,130)]
            g.add(curve(pts,RED,2.5,opa))
        return g
    is_energy=mode=='energy'
    ymax=2.8 if is_energy else .44
    scl=1.02 if is_energy else 6.6
    g.add(Line(coord(-2.75,0),coord(2.75,0),color=MUTED,stroke_width=1.5),Line(coord(-2.75,0),coord(-2.75,2.85),color=MUTED,stroke_width=1.5))
    for n in (-2,-1,0,1,2):
        q=coord(n,0);g.add(Line(q,q+DOWN*.07,color=MUTED,stroke_width=1),J(n,14,MUTED).move_to(q+DOWN*.22))
    vals=(0,1,2) if is_energy else (0,.2,.4)
    for n in vals:
        q=coord(-2.75,scl*n);g.add(Line(q,q+LEFT*.07,color=MUTED,stroke_width=1),J(f'{n:g}',14,MUTED).move_to(q+LEFT*.28))
    g.add(tex('z',23,MUTED).move_to(coord(2.84,-.02)),tex('E(z)' if is_energy else 'p(z)',23,MUTED).move_to(coord(-2.6,3.04)))
    xs=np.linspace(-2.35 if is_energy else -2.7,2.35 if is_energy else 2.7,180)
    ys=[.5*x*x if is_energy else gaussian_density(x) for x in xs]
    g.add(curve([coord(float(x),scl*y) for x,y in zip(xs,ys)],RED))
    return g

def phase_point(index):
    z,r=STATES[index];return Dot(coord(z,.94*(r+1.4)),radius=.10,color=BLUE)
def position_point(index,mode='density'):
    z,r=STATES[index]
    y=1.02*energy(z,0) if mode=='energy' else 6.6*gaussian_density(z)
    return Dot(coord(z,y),radius=.10,color=BLUE)

def shear_polygon(bi):
    amount=[0,.16,.47,.47,-.47,.47][bi]
    points=[[3.35,-1.58,0],[4.50,-1.58,0],[4.50+amount,-.42,0],[3.35+amount,-.42,0]]
    return Polygon(*points,fill_color=BLUE,fill_opacity=.26,stroke_color=YELLOW,stroke_width=2.5)

def trajectory_points(si,bi):
    if bi==0:return None
    if si==7:
        indices=[0,1,4,8,12,18]
        lo,hi=indices[bi-1:bi+1]
        states=ANISO
        points=[[-2.50+1.30*v[0][0],-1.75+2.65*v[0][1],0] for v in states[lo:hi+1]]
    elif si in (2,3,4,6,8):
        lo,hi=PHASES[si][bi-1:bi+1]
        if si in (4,6):points=[coord(float(z),.94*(float(r)+1.4)) for z,r in STATES[lo:hi+1]]
        elif si in (2,3):points=[coord(float(z),1.02*energy(float(z),0)) for z,r in STATES[lo:hi+1]]
        else:points=[coord(float(z),6.6*gaussian_density(float(z))) for z,r in STATES[lo:hi+1]]
    else:return None
    return points if hi>lo else None

def side_diagram(si,bi):
    """A small quantitative graphic with labelled axes or bars, never a text placard."""
    g=VGroup()
    if si==0:
        g.add(J('短い提案を反復',18,BLUE).move_to([4.18,.78,0]))
        xs=[3.05,3.49,3.24,3.69,3.40,3.83,3.55]
        for a,b in zip(xs[:-1],xs[1:]):
            g.add(Arrow([a,-.16,0],[b,-.16,0],buff=0,color=BLUE,stroke_width=2.2,max_tip_length_to_length_ratio=.20))
        g.add(J('1回の長い候補',18,YELLOW).move_to([4.18,-.72,0]))
        g.add(Arrow([3.05,-1.18,0],[5.52,-1.18,0],buff=0,color=YELLOW,stroke_width=3.2,max_tip_length_to_length_ratio=.10))
        g.add(Dot([3.05,-1.18,0],radius=.09,color=BLUE),Dot([5.52,-1.18,0],radius=.10,color=GREEN))
        g.add(J('距離 約4.02',18,GREEN).move_to([4.18,-1.78,0]))
        if bi>=3:g.add(J('採用確率 約99.97%',18,GREEN).move_to([4.18,-2.32,0]))
        return g
    if si in (2,3,5,8):
        # Two energy columns share a zero line. Height comes from the same computed state.
        idx=PHASES[si][bi]
        z,r=STATES[idx]
        vals=[energy(z,0),energy(0,r)]
        if si==5 and bi==2:
            z,r=leapfrog(START_Z,START_R,epsilon=1.4,steps=STEPS)[-1]
            vals=[energy(z,0),energy(0,r)]
        for j,(v,color,name) in enumerate(zip(vals,[BLUE,YELLOW],['位置 E','運動 K'])):
            x=3.20+j*1.84
            g.add(Rectangle(width=.80,height=max(.025,.75*v),fill_color=color,fill_opacity=.7,stroke_width=0).move_to([x,-1.55+.75*v/2,0]))
            g.add(J(name,18,color).move_to([x,-1.82,0]))
            g.add(J(f'{v:.3f}',15,WHITE).move_to([x,-1.52+.75*v+.16,0]))
        g.add(Line([2.55,-1.55,0],[5.78,-1.55,0],color=MUTED,stroke_width=1.5))
        g.add(Line([2.58,-1.55,0],[2.58,.27,0],color=MUTED,stroke_width=1.2))
        for n in (0,1,2):
            yy=-1.55+.75*n
            g.add(Line([2.58,yy,0],[2.65,yy,0],color=MUTED,stroke_width=1),J(n,13,MUTED).move_to([2.37,yy,0]))
        g.add(tex('E,K',18,MUTED).move_to([2.58,.48,0]))
        if si==8:
            g.add(J('合計 H = E + K',18,GREEN).move_to([4.18,.75,0]))
        elif si==3:
            labels=['傾きと逆向き','半歩','運動量の半歩','位置の全歩','運動量の半歩','18回の軌道']
            g.add(J(labels[bi],19,PURPLE if bi==0 else YELLOW).move_to([4.18,.58,0]))
        else:
            g.add(J('刻み1.4の別実験' if si==5 and bi==2 else '合計 H = E + K',18,GREEN).move_to([4.18,.75,0]))
            if si==5 and bi==2:
                g.add(Line([3.24,-2.23,0],[5.18,-2.23,0],color=MUTED,stroke_width=1.1))
                g.add(Line([3.24,-2.23,0],[3.24+1.94*.1195015909691799/.2,-2.23,0],color=RED,stroke_width=6))
                g.add(J('0',12,MUTED).move_to([3.24,-2.48,0]),J('0.2',12,MUTED).move_to([5.18,-2.48,0]))
                g.add(J('差 0.1195',15,RED).move_to([4.25,-2.54,0]))
        return g
    if si==1 and bi>=2:
        g.add(J('位相空間',18,MUTED).move_to([4.18,.8,0]))
        g.add(Line([2.80,-1.25,0],[5.66,-1.25,0],color=MUTED,stroke_width=1.5),Line([4.18,-2.48,0],[4.18,.25,0],color=MUTED,stroke_width=1.5))
        for n in (-2,0,2):
            xx=4.18+.52*n
            g.add(Line([xx,-1.25,0],[xx,-1.31,0],color=MUTED,stroke_width=1),J(n,13,MUTED).move_to([xx+(.19 if n==0 else 0),-1.49,0]))
        for n in (-1,0,1):
            yy=-1.25+1.02*n
            g.add(Line([4.18,yy,0],[4.10,yy,0],color=MUTED,stroke_width=1),J(n,13,MUTED).move_to([3.91,yy+(.17 if n==0 else 0),0]))
        g.add(tex('z',22,MUTED).move_to([5.83,-1.25,0]),tex('r',22,MUTED).move_to([4.18,.43,0]))
        g.add(Dot([4.18+.52*START_Z,-1.25+1.02*START_R,0],radius=.10,color=BLUE))
        g.add(J('(-2, 0.5)',16,BLUE).move_to([5.10,-2.35,0]))
        return g
    if si in (1,6):
        g.add(J('運動量の分布',18,MUTED).move_to([4.18,.58,0]))
        g.add(Line([2.65,-1.62,0],[5.78,-1.62,0],color=MUTED,stroke_width=1.5),Line([2.65,-1.62,0],[2.65,.42,0],color=MUTED,stroke_width=1.5))
        xs=np.linspace(-2.5,2.5,100)
        g.add(curve([[4.18+.59*x,-1.62+4.55*gaussian_density(x),0] for x in xs],YELLOW,2.8))
        for n in (-2,0,2):
            x=4.18+.59*n;g.add(Line([x,-1.62,0],[x,-1.68,0],color=MUTED,stroke_width=1),J(n,13,MUTED).move_to([x,-1.86,0]))
        for n in (0,.2,.4):
            yy=-1.62+4.55*n
            g.add(Line([2.65,yy,0],[2.72,yy,0],color=MUTED,stroke_width=1),J(f'{n:g}',12,MUTED).move_to([2.39,yy,0]))
        g.add(tex('r',22,MUTED).move_to([5.96,-1.61,0]),tex('p(r)',22,MUTED).move_to([2.67,.59,0]))
        marker=.5 if si==1 or bi<2 else [-1.3,.2,1.0,-.6][min(bi-2,3)]
        g.add(Dot([4.18+.59*marker,-1.62+4.55*gaussian_density(marker),0],radius=.09,color=GREEN))
        g.add(J(f'r = {marker:g}',17,GREEN).move_to([4.18,-2.23,0]))
        return g
    if si==4:
        # Figure 11.14's shear. The boundary is a data region, not a decorative frame.
        g.add(J('位相空間の領域',18,MUTED).move_to([4.18,.8,0]))
        g.add(shear_polygon(bi))
        g.add(Line([2.83,-1.82,0],[5.73,-1.82,0],color=MUTED,stroke_width=1.2),Line([2.83,-1.82,0],[2.83,.31,0],color=MUTED,stroke_width=1.2))
        for n,x in [(0,2.83),(1,4.00),(2,5.17)]:g.add(J(n,13,MUTED).move_to([x,-2.02,0]))
        for n,y in [(0,-1.82),(1,-.66)]:g.add(J(n,13,MUTED).move_to([2.63,y,0]))
        g.add(tex('z',20,MUTED).move_to([5.91,-1.82,0]),tex('r',20,MUTED).move_to([2.82,.53,0]))
        g.add(J('面積 = 1.33',17,GREEN).move_to([4.18,-2.37,0]))
        return g
    # Scene 8: a small scale comparison with named axes and tick values.
    g.add(J('探索の目安',18,MUTED).move_to([4.18,.79,0]))
    bars=[5,25]
    for j,(v,c,label) in enumerate(zip(bars,[YELLOW,PURPLE],['力学','等方提案'])):
        x=3.25+j*1.90
        g.add(Rectangle(width=.72,height=1.54*v/25,fill_color=c,fill_opacity=.74,stroke_width=0).move_to([x,-1.70+1.54*v/50,0]))
        g.add(J(label,17,c).move_to([x,-2.02,0]),J(v,17,WHITE).move_to([x,-1.50+1.54*v/25,0]))
    g.add(Line([2.66,-1.70,0],[5.80,-1.70,0],color=MUTED,stroke_width=1.4),Line([2.66,-1.70,0],[2.66,-.09,0],color=MUTED,stroke_width=1.2))
    for n in (0,5,25):
        yy=-1.70+1.54*n/25
        g.add(Line([2.66,yy,0],[2.73,yy,0],color=MUTED,stroke_width=1),J(n,12,MUTED).move_to([2.41,yy,0]))
    g.add(J('回数',15,MUTED).move_to([2.62,.15,0]))
    g.add(J('幅の比 5',17,GREEN).move_to([4.18,-2.43,0]))
    return g

def figure(si,bi):
    g=VGroup()
    if si==7:
        index=[0,1,4,8,12,18][min(bi,5)]
        x,y=ANISO[index][0]
        pts=[[-2.50+1.30*v[0][0],-1.75+2.65*v[0][1],0] for v in ANISO[:index+1]]
        if len(pts)>1:g.add(curve(pts,YELLOW,2.4,.8))
        g.add(Dot([-2.50+1.30*x,-1.75+2.65*y,0],radius=.10,color=BLUE))
    elif si in (4,6):
        idx=PHASES[si][bi]
        g.add(phase_point(idx))
        if si==6 and bi in (1,2):
            z=-2.0;r=[-.4,1.3][bi-1]
            g.add(Dot(coord(z,.94*(r+1.4)),radius=.09,color=GREEN))
    else:
        mode='energy' if si in (2,3,5) else 'density'
        idx=PHASES[si][bi]
        g.add(position_point(idx,mode))
        if si==3 and bi==0:
            x=-2.;y=2.
            g.add(Line(coord(-2.25,1.02*2.5),coord(-1.75,1.02*1.5),color=YELLOW,stroke_width=2.2))
            g.add(Arrow(coord(-1.62,1.02*y)+UP*.79,coord(-2.22,1.02*y)+UP*.79,buff=0,color=YELLOW,stroke_width=2.5,max_tip_length_to_length_ratio=.14))
            g.add(Arrow(coord(-2.22,1.02*y)+UP*.45,coord(-1.62,1.02*y)+UP*.45,buff=0,color=PURPLE,stroke_width=2.5,max_tip_length_to_length_ratio=.14))
        if si==3 and bi==1:
            x=1.5;y=1.125
            g.add(Dot(coord(x,1.02*y),radius=.08,color=YELLOW))
            g.add(Line(coord(1.15,1.02*(y-.525)),coord(1.85,1.02*(y+.525)),color=YELLOW,stroke_width=2.2))
            g.add(Arrow(coord(1.7,1.02*y)+UP*.65,coord(1.1,1.02*y)+UP*.65,buff=0,color=YELLOW,stroke_width=2.5,max_tip_length_to_length_ratio=.14))
        if si in (0,8) and bi>=1:g.add(Dot(coord(STATES[-1,0],6.6*gaussian_density(STATES[-1,0])),radius=.10,color=GREEN))
        if si in (2,3,5,8) and bi>=1:
            vals=[coord(float(z),1.02*.5*float(z*z)) for z,r in STATES[:idx+1]] if mode=='energy' else [coord(float(z),6.6*gaussian_density(float(z))) for z,r in STATES[:idx+1]]
            if len(vals)>1:g.add(curve(vals,YELLOW,3,.8))
    g.add(side_diagram(si,bi))
    return g

class PRML115HybridMonteCarlo(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            mode='ellipse' if si==7 else 'phase' if si in (4,6) else 'energy' if si in (2,3,5) else 'density'
            self.plot=axis(mode);self.add(self.plot)
            self.visual=figure(si,0);self.add(self.visual)
            self.formula=None;self.note=None
            for bi,beat in enumerate(story['beats']):
                if self.formula is not None:self.remove(self.formula,self.note)
                if bi>0 and (si,bi-1) in ((1,0),(3,0)):self.remove(review)
                self.formula=tex(FORMULAS[si][bi],27,PURPLE if (si,bi)==(3,0) else YELLOW).move_to([4.18,2.27,0])
                if self.formula.width>4.75:self.formula.scale(4.75/self.formula.width)
                note=beat['visual_note']
                if (si,bi)==(1,0):note='中心と幅を運動量へ'
                if (si,bi)==(3,0):note='接線の傾きから力へ'
                named={(1,3):'ポテンシャルエネルギー',(1,4):'運動エネルギーとハミルトニアン',(3,2):'リープフロッグ離散化',(5,3):'詳細釣り合い',(8,4):'ハイブリッドモンテカルロアルゴリズム'}
                note=named.get((si,bi),note)
                self.note=J(note,19,MUTED).move_to([4.18,1.52,0])
                if self.note.width>4.7:self.note.scale(4.7/self.note.width)
                self.add(self.formula,self.note)
                if (si,bi) in ((1,0),(3,0)):
                    review=J('復習: 2.3 ガウス分布' if si==1 else '復習: 3.1 勾配',18,YELLOW).move_to([4.18,1.08,0])
                    self.add(review)
                # Switch labels and diagrams atomically. Morphing Japanese glyphs
                # makes them unreadable during a long spoken beat.
                self.remove(self.visual)
                self.visual=figure(si,bi)
                self.add(self.visual)
                duration=self.durations[self.beat_index]
                points=trajectory_points(si,bi)
                moving=[];tracer=None;target_dot=None;ghost=None;target_polygon=None
                if points:
                    target_dot=next(m for m in self.visual.submobjects if isinstance(m,Dot))
                    target_dot.set_opacity(0)
                    tracer=Dot(points[0],radius=.09,color=BLUE)
                    path=VMobject().set_points_as_corners(points)
                    self.add(tracer)
                    moving.append(MoveAlongPath(tracer,path))
                if si==4 and bi>0:
                    target_polygon=next(m for m in self.visual[-1].submobjects if isinstance(m,Polygon))
                    target_polygon.set_opacity(0)
                    ghost=shear_polygon(bi-1)
                    self.add(ghost)
                    moving.append(Transform(ghost,shear_polygon(bi)))
                if not moving:
                    moving=[ShowPassingFlash(self.plot[-1].copy().set_stroke(YELLOW,width=5),time_width=.13)]
                action='せん断' if ghost is not None else '軌道を進む' if tracer is not None else '曲線をたどる'
                self.beat(*moving,actions=[dict(name=action,start=0,end=duration)])
                if tracer is not None:
                    self.remove(tracer)
                    target_dot.set_opacity(1)
                if ghost is not None:
                    self.remove(ghost)
                    target_polygon.set_fill(BLUE,opacity=.26).set_stroke(YELLOW,width=2.5,opacity=1)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml115_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
