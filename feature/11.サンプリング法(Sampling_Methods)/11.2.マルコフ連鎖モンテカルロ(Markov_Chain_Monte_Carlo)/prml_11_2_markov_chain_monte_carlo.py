"""PRML 11.2: a walker tests whether it can discover the missed mode."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene,jp,tex
from mcmc_model import target,truth,walk,summary,START,THRESHOLD,SCALES

BG='#10141F'
BLUE=ManimColor('#58C4DD'); GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
X0,X1=-5.40,1.46; Y0,Y1=-1.76,1.36; Z0,Z1=-4.6,3.5
ROWS={s:summary(s) for s in SCALES}

def J(value,size=18,color=WHITE):return jp(str(value),size,color)
def X(z):return X0+(z-Z0)/(Z1-Z0)*(X1-X0)
def Y(v):return Y0+v/.83*(Y1-Y0)
def P(z,v):return np.array([X(z),Y(v),0.])
def curve(points,color,width=2.7,opacity=1):
    mob=VMobject();mob.set_points_as_corners([np.array([x,y,0]) for x,y in points]);mob.set_stroke(color,width=width,opacity=opacity);return mob

def density_axes():
    g=VGroup(Line([X0,Y0,0],[X1,Y0,0],color=MUTED,stroke_width=1.4),Line([X0,Y0,0],[X0,Y1,0],color=MUTED,stroke_width=1.4))
    for z in (-4,-2,0,1,2,3):
        x=X(z);g.add(Line([x,Y0,0],[x,Y0-.06,0],color=MUTED,stroke_width=1),J(z,14,MUTED).move_to([x,Y0-.23,0]))
    for value in (0,.4,.8):
        y=Y(value);g.add(Line([X0-.06,y,0],[X0,y,0],color=MUTED,stroke_width=1),J(f'{value:g}',13,MUTED).move_to([X0-.34,y,0]))
    g.add(tex('z',22,MUTED).move_to([X1+.17,Y0-.18,0]),J('密度',16,MUTED).move_to([X0+.05,Y1+.25,0]))
    return g

def target_plot():
    z=np.linspace(Z0,Z1,150)
    return curve([(X(x),Y(float(target(x)))) for x in z],RED,3.2)

def area():
    z=np.linspace(1,Z1,70)
    return Polygon(P(1,0),*[P(float(x),float(target(x))) for x in z],P(Z1,0),color=RED,stroke_width=0,fill_color=RED,fill_opacity=.18)

def threshold():
    return VGroup(DashedLine([X(1),Y0,0],[X(1),Y1-.08,0],dash_length=.11,color=YELLOW,stroke_width=1.3),J('z > 1',15,YELLOW).move_to([X(1.7),1.1,0]))

def walk_index(si,bi):
    if si==0:return [1,8,24,24,24,24][bi],.35
    if si==1:return [1,11,13,14,14,24][bi],.35
    if si==2:return [11,12,13,17,18,23][bi],.35
    if si==4:return [1183,421,37,37,2500,6000][bi],[.35,.8,2.2,2.2,.35,2.2][bi]
    if si==5:return [46,60,8000,8200,2200,2350][bi],.35
    if si==8:return [24,24,37,1000,5500,9000][bi],[.35,.35,2.2,2.2,2.2,2.2][bi]
    return min(24,4+bi*3),.8

def markov_trace(si,bi):
    if si==6:
        old=-1.;candidate=[-.6,-.2,0.,.4,.1,0.][bi]
        a=P(old,float(target(old)));b=P(candidate,float(target(candidate)))
        return VGroup(Dot(a,radius=.085,color=BLUE),Dot(b,radius=.065,color=PURPLE),
                      Arrow(a,b,buff=.12,stroke_width=2.5,max_tip_length_to_length_ratio=.12,color=PURPLE))
    index,scale=walk_index(si,bi)
    states,candidates,uniforms,accepted=walk(scale)
    # The low horizontal strip is the chain's position over a short window;
    # dots on the target curve show the current and proposed states.
    end=max(1,index);start=max(0,end-22)
    g=VGroup()
    shown=states[start:end+1]
    for j,z in enumerate(shown):
        if Z0<=z<=Z1:
            g.add(Dot([X(float(z)),-1.36+(j%3)*.065,0],radius=.030,color=GREEN if (j and accepted[start+j-1]) else BLUE))
    current=float(states[36]) if si==8 and bi>=2 else float(states[end-1])
    previous=current; proposed=float(candidates[end-1])
    g.add(Dot(P(np.clip(current,Z0,Z1),float(target(current))),radius=.087,color=BLUE))
    if si==8:
        if bi>=2:
            destination=float(states[37])
            g.add(Arrow(P(current,float(target(current))),P(destination,float(target(destination))),buff=.12,stroke_width=2.8,max_tip_length_to_length_ratio=.12,color=GREEN))
            g.add(Dot(P(destination,float(target(destination))),radius=.070,color=GREEN))
        return g
    if Z0<=proposed<=Z1:
        q=P(proposed,float(target(proposed)))
        g.add(Dot(q,radius=.058,color=GREEN if accepted[end-1] else PURPLE))
        if abs(proposed-previous)>.08:
            h=P(previous,float(target(previous)))
            g.add(Arrow(h,q,buff=.10,stroke_width=2.2,max_tip_length_to_length_ratio=.14,color=GREEN if accepted[end-1] else PURPLE))
    return g

def flow_panel(si,bi):
    g=VGroup()
    if si==3:
        left=.82;right=.18;forward=.09;back=.41
        if bi==0:
            # Directed three-node chain recalls §8.2's graphical notation.
            for x,label in [(2.97,'前'),(4.30,'今'),(5.65,'次')]:g.add(Dot([x,-.18,0],radius=.13,color=BLUE),J(label,16).move_to([x,-.56,0]))
            g.add(Arrow([3.13,-.18,0],[4.12,-.18,0],buff=0,color=YELLOW,stroke_width=2),Arrow([4.46,-.18,0],[5.49,-.18,0],buff=0,color=YELLOW,stroke_width=2))
            g.add(J('復習: 8.2 条件付き独立',17,YELLOW).move_to([4.32,1.12,0]))
            g.add(SurroundingRectangle(g,buff=.16,color=YELLOW,stroke_width=1.2))
        else:
            g.add(J('左の状態',18,BLUE).move_to([3.14,.93,0]),J('右の状態',18,RED).move_to([5.58,.93,0]))
            g.add(J('0.82',20,BLUE).move_to([3.12,.51,0]),J('0.18',20,RED).move_to([5.58,.51,0]))
            g.add(Arrow([3.45,-.15,0],[5.19,-.15,0],color=GREEN,stroke_width=4,buff=.10),Arrow([5.19,-.65,0],[3.45,-.65,0],color=PURPLE,stroke_width=4,buff=.10))
            g.add(J('0.82 × 0.09 = 0.0738',15,GREEN).move_to([4.32,-1.11,0]),J('0.18 × 0.41 = 0.0738',15,PURPLE).move_to([4.32,-1.49,0]))
            g.add(J('説明用の2状態例',16,YELLOW).move_to([4.32,-2.10,0]))
    else:
        g.add(J('右へ偏る候補',18,BLUE).move_to([4.32,1.0,0]))
        g.add(J('今の位置',17,BLUE).move_to([3.08,.42,0]),J('候補',17,RED).move_to([5.49,.42,0]))
        g.add(Arrow([3.42,-.13,0],[5.13,-.13,0],color=GREEN,stroke_width=4,buff=.06),Arrow([5.13,-.63,0],[3.42,-.63,0],color=PURPLE,stroke_width=3,buff=.06))
        g.add(J('順方向 q(z′|z)',16,GREEN).move_to([4.30,-1.09,0]),J('逆方向 q(z|z′)',16,PURPLE).move_to([4.30,-1.47,0]))
        if bi>=2:g.add(J('採用後の流れは同じ',17,YELLOW).move_to([4.30,-2.08,0]))
    return g

def ellipse_plot(bi):
    g=VGroup()
    # A two-dimensional, correlated Gaussian; ellipse widths are explanatory.
    g.add(Line([X0+.3,-1.57,0],[X1,-1.57,0],color=MUTED,stroke_width=1.5),Line([X0+.55,-1.80,0],[X0+.55,1.28,0],color=MUTED,stroke_width=1.5))
    for v,x in [(-2,-3.6),(0,-1.9),(2,-.2)]:
        g.add(Line([x,-1.57,0],[x,-1.64,0],color=MUTED,stroke_width=1),J(v,13,MUTED).move_to([x,-1.82,0]))
    for v,y in [(-1,-.78),(0,-.12),(1,.54)]:
        g.add(Line([X0+.49,y,0],[X0+.55,y,0],color=MUTED,stroke_width=1),J(v,13,MUTED).move_to([X0+.23,y,0]))
    g.add(tex('z_1',22,MUTED).move_to([X1+.18,-1.67,0]),tex('z_2',22,MUTED).move_to([X0+.47,1.5,0]))
    ellipse=Ellipse(width=5.2,height=.75,color=RED,stroke_width=3).rotate(PI/7).move_to([-2.03,-.03,0]);g.add(ellipse)
    radius=[.48,1.12,.38,.38,.50,.50][bi]
    g.add(Circle(radius=radius,color=BLUE,stroke_width=2.4).move_to([-2.35,-.16,0]))
    g.add(Dot([-2.35,-.16,0],radius=.075,color=BLUE))
    return g

def right_numbers(si,bi):
    g=VGroup()
    def line(y,a,b,color=WHITE):
        g.add(J(a,18,MUTED).move_to([3.35,y,0]),J(b,21,color).move_to([5.34,y,0]))
    if si in (0,1,2,5,8):
        index,scale=walk_index(si,bi);r=ROWS[scale]
        line(.92,'歩幅 σ',f'{scale:g}',BLUE)
        if si==0:
            line(.35,'開始位置','z = -1',BLUE)
            if bi>=2:line(-.22,'最初の24歩','右側 0点',YELLOW)
            if bi>=3:line(-.79,'真の右側確率','0.180',RED)
        elif si in (1,2):
            states,candidates,uniforms,accepted=walk(scale)
            old=float(states[index-1]);candidate=float(candidates[index-1])
            line(.35,'現在地',f'{old:.2f}',BLUE)
            line(-.22,'候補',f'{candidate:.2f}',PURPLE)
            line(-.79,'採否','採用' if accepted[index-1] else '棄却',GREEN if accepted[index-1] else RED)
        else:
            line(.35,'初到達',f"{r['first_right_step']}歩",YELLOW)
            line(-.22,'採用率',f"{r['acceptance_rate']:.3f}",GREEN)
            line(-.79,'右側比率',f"{r['right_estimate']:.3f}",RED)
            if si==5 and bi>=2:line(-1.34,'隣接相関',f"{r['right_lag1_correlation']:.3f}",PURPLE)
            if si==8:g.add(J('真の右側確率 0.180',17,YELLOW).move_to([4.40,-2.05,0]))
    elif si==4:
        g.add(J('歩幅',14,BLUE).move_to([3.17,1.33,0]),J('初到達',14,YELLOW).move_to([4.38,1.33,0]),J('採用率',14,GREEN).move_to([5.65,1.33,0]))
        limit=min(bi+1,3)
        for i,s in enumerate(SCALES[:limit]):
            r=ROWS[s]; y=.94-i*.59
            g.add(J(f'σ={s:g}',17,BLUE).move_to([3.17,y,0]),J(f"{r['first_right_step']}歩",17,YELLOW).move_to([4.38,y,0]),J(f"{r['acceptance_rate']:.3f}",16,GREEN).move_to([5.65,y,0]))
        if bi>=4:
            g.add(J('初期2000歩を除く',16,MUTED).move_to([4.35,-1.1,0]))
            g.add(J('σ=0.35 → 0.449',17,PURPLE).move_to([4.35,-1.52,0]),J('σ=2.2 → 0.179',17,YELLOW).move_to([4.35,-1.96,0]))
    elif si==7:
        g.add(J('2次元の説明用例',18,MUTED).move_to([4.35,.98,0]))
        g.add(J('青: 等方ガウス提案',18,BLUE).move_to([4.35,.30,0]))
        g.add(J('赤: 細長い目標分布',18,RED).move_to([4.35,-.30,0]))
        g.add(J('大きい円 → 棄却が増える',16,PURPLE).move_to([4.35,-1.10,0]))
        g.add(J('小さい円 → 移動が遅い',16,GREEN).move_to([4.35,-1.55,0]))
    return g

FORMULAS=[
 [r'P(z>1)=0.180',r'z_0=-1',r'0/24=0',r'p(z)\downarrow',r'11.1\to11.2',r'P(z>1)\;?'],
 [r'q(z\prime\mid z)=q(z\mid z\prime)',r'\tilde p(z\prime)>\tilde p(z)',r'0<A<1',r'z^{(t+1)}=z^{(t)}',r'z^{(0)},z^{(1)},\ldots',r'q(z\prime\mid z)=q(z\mid z\prime)'],
 [r'q(z\prime\mid z)=\mathcal N(z\prime\mid z,\sigma^2)',r'A=\min(1,\tilde p(z\prime)/\tilde p(z))',r'\tilde p(z\prime)<\tilde p(z)',r'\tilde p(z\prime)>\tilde p(z)',r'\frac{\tilde p(z\prime)/Z}{\tilde p(z)/Z}=\frac{\tilde p(z\prime)}{\tilde p(z)}',r'p(z)\;?'],
 [r'p(z^{(t+1)}\mid z^{(0:t)})=p(z^{(t+1)}\mid z^{(t)})',r'p(a)T(a,b)',r'p(a)T(a,b)=p(b)T(b,a)',r'pT=p',r'\lim_{t\to\infty}p(z^{(t)})=p(z)',r'\sigma=0.35'],
 [r'\sigma=0.35',r'\sigma=0.8',r'\sigma=2.2',r'r_{\rm a}\uparrow\;\not\Rightarrow\;v\uparrow',r'\hat P=\frac1{10000}\sum_{t=2001}^{12000}\mathbf 1[z^{(t)}>1]',r'0.179\approx0.180'],
 [r'z^{(t+1)}\approx z^{(t)}',r'0,0,0,\ldots,1,1,1',r'\rho_1\approx0.999',r't,\;t+M,\;t+2M',r't>2000',r'N=10000\;\not\Rightarrow\;N_{\rm eff}=10000'],
 [r'q(z\prime\mid z)\neq q(z\mid z\prime)',r'p(z)q(z\prime\mid z)',r'A=\min\!\left(1,\frac{\tilde p(z\prime)q(z\mid z\prime)}{\tilde p(z)q(z\prime\mid z)}\right)',r'p(z)T(z,z\prime)=p(z\prime)T(z\prime,z)',r'q(z\prime\mid z)=q(z\mid z\prime)',r'\tilde p(z),\;q(z\prime\mid z)'],
 [r'p(z_1,z_2)',r'\rho\uparrow',r'\rho\downarrow',r'O((\sigma_{\max}/\sigma_{\min})^2)',r'q(z\prime\mid z)',r'\sigma=2.2'],
 [r'z_0=-1,\;\sigma=0.35',r'P(z>1)=0.180',r'\hat P(z>1)=0.179',r'p(a)T(a,b)=p(b)T(b,a)',r't\to\infty',r'11.3\;p(z_i\mid z_{\setminus i})']
]

class PRML112MarkovChainMonteCarlo(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            if si==7:
                self.visual=ellipse_plot(0)
                self.add(self.visual)
            else:
                self.add(density_axes(),area(),target_plot(),threshold())
                self.visual=markov_trace(si,0)
                self.add(self.visual)
            self.side=flow_panel(si,0) if si in (3,6) else right_numbers(si,0)
            self.add(self.side)
            self.note=None;self.formula=None
            for bi in range(len(story['beats'])):self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml112_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
    def run_beat(self,si,bi):
        wide=(si,bi) in {(3,0),(4,4),(6,2)}
        if self.note is not None:self.remove(self.note)
        self.note=None
        if not wide:
            self.note=J(SCENES[si]['beats'][bi]['visual_note'],17,MUTED).move_to([4.34,2.42,0])
            if self.note.width>3.9:self.note.scale(3.9/self.note.width)
            self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        f=FORMULAS[si][bi]
        self.formula=tex(f,27 if wide else 23,YELLOW).move_to([0,2.37,0] if wide else [4.35,1.76,0])
        limit=9.0 if wide else 3.8
        if self.formula.width>limit:self.formula.scale(limit/self.formula.width)
        self.add(self.formula)
        new_visual=ellipse_plot(bi) if si==7 else markov_trace(si,bi)
        new_side=flow_panel(si,bi) if si in (3,6) else right_numbers(si,bi)
        duration=self.durations[self.beat_index]
        switch=(si==1 and bi in (1,2,3,4)) or (si==2 and bi in (1,2,3))
        if switch:
            index,scale=walk_index(si,bi)
            accepted=bool(walk(scale)[3][index-1])
            focus=new_visual[-3] if si==1 and bi==4 else new_visual[-2]
            color=BLUE if si==1 and bi==4 else GREEN if accepted else PURPLE
            self.beat(phases=[
                ('表示切替',.5,lambda:AnimationGroup(Succession(FadeOut(self.side),FadeIn(new_side)),Succession(FadeOut(self.visual),FadeIn(new_visual)),lag_ratio=0)),
                ('候補を強調',duration-.5,lambda:Indicate(focus,color=color,scale_factor=1.3))])
            self.visual=new_visual
        else:
            self.beat(phases=[
                ('表示切替',.5,lambda:Succession(FadeOut(self.side),FadeIn(new_side))),
                ('移動',duration-.5,lambda:Transform(self.visual,new_visual))])
        self.side=new_side
