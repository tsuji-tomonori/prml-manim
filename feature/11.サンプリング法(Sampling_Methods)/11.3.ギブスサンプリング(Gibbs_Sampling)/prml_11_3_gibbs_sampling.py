"""PRML 11.3: every accepted Gibbs step can still move slowly."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene,jp,tex
from gibbs_model import RHO,COND_SD,chain

BG='#10141F'
BLUE=ManimColor('#58C4DD'); GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
DATA=chain(); OVER=chain(alpha=-.8)

# Fixed axes have space for their names and numbers; all ellipses lie inside them.
def J(value,size=19,color=WHITE):return jp(str(value),size,color)
def pos(x,y):return np.array([-2.38+1.00*x,-.15+.56*y,0.])
def path(points,color,width=2.7,opacity=1.):
    mob=VMobject();mob.set_points_as_corners(points);mob.set_stroke(color,width=width,opacity=opacity);return mob

def cloud_axes():
    g=VGroup(Line(pos(-3.2,-3),pos(3.2,-3),stroke_width=1.4,color=MUTED),Line(pos(-3.2,-3),pos(-3.2,3),stroke_width=1.4,color=MUTED))
    for k in (-2,0,2):
        q=pos(k,-3);g.add(Line(q,q+DOWN*.08,stroke_width=1,color=MUTED),J(k,15,MUTED).move_to(q+DOWN*.25))
        q=pos(-3.2,k);g.add(Line(q,q+LEFT*.08,stroke_width=1,color=MUTED),J(k,15,MUTED).move_to(q+LEFT*.29))
    g.add(tex('z_1',25,MUTED).move_to(pos(3.43,-3.10)),tex('z_2',25,MUTED).move_to(pos(-3.18,3.35)))
    return g

def cloud():
    g=VGroup()
    angles=np.linspace(0,2*np.pi,145)
    for radius,opacity in [(1.,.34),(1.7,.58),(2.25,.9)]:
        a=radius*np.sqrt(1+RHO)*np.cos(angles)
        b=radius*np.sqrt(1-RHO)*np.sin(angles)
        x=(a+b)/np.sqrt(2);y=(a-b)/np.sqrt(2)
        g.add(path([pos(float(u),float(v)) for u,v in zip(x,y)],RED,2.7,opacity))
    return g

def trace(points,end,color=BLUE):
    end=max(0,min(end,len(points)-1))
    g=VGroup()
    # Keep only the last 40 half-steps legible; first and final points remain.
    start=max(0,end-40)
    for k in range(start,end):
        p=pos(*points[k]);q=pos(*points[k+1])
        g.add(Line(p,q,color=color,stroke_width=2.6,stroke_opacity=.65))
    for k in range(start,end+1,2):
        g.add(Dot(pos(*points[k]),radius=.033,color=color,fill_opacity=.7))
    g.add(Dot(pos(*points[end]),radius=.092,color=GREEN if end else BLUE))
    return g

def cut_line(y,which=0):
    g=VGroup()
    if which==0:
        g.add(DashedLine(pos(-3.0,y),pos(3.0,y),color=YELLOW,stroke_width=2,dash_length=.11))
        m=RHO*y;g.add(Dot(pos(m,y),radius=.085,color=YELLOW))
    else:
        g.add(DashedLine(pos(y,-2.8),pos(y,2.8),color=YELLOW,stroke_width=2,dash_length=.11))
        m=RHO*y;g.add(Dot(pos(y,m),radius=.085,color=YELLOW))
    return g

def alpha_slider(value):
    left,right,y=-4.28,-1.78,1.72
    x=left+(value+1)*(right-left)/2
    g=VGroup(Line([left,y,0],[right,y,0],color=MUTED,stroke_width=2))
    for v in (-1,0,1):
        q=left+(v+1)*(right-left)/2
        g.add(Line([q,y-.07,0],[q,y+.07,0],color=MUTED,stroke_width=1.3),J(v,14,MUTED).move_to([q,y-.24,0]))
    g.add(Dot([x,y,0],radius=.09,color=PURPLE),tex(r'\alpha',24,PURPLE).move_to([left-.23,y+.02,0]))
    return g

def marker_pair():
    a=pos(-1.95,-1.95);b=pos(-2.05,-1.95)
    return VGroup(Dot(a,radius=.09,color=BLUE),Dot(b,radius=.085,color=GREEN),Arrow(a+UP*.3,b+UP*.3,buff=.03,color=PURPLE,stroke_width=2))

def graph_nodes(bi):
    g=VGroup();centers=[(-4.3,.05),(-5.65,1.12),(-3.0,1.12),(-5.65,-1.08),(-3.0,-1.08),(-1.65,1.12)]
    edges=[(1,0),(2,0),(0,3),(0,4),(5,4)]
    for a,b in edges:
        q=np.array([*centers[a],0]);r=np.array([*centers[b],0])
        if bi==1:
            unit=(r-q)/np.linalg.norm(r-q)
            g.add(Line(q+.20*unit,r-.20*unit,stroke_width=2.4,color=YELLOW if a==0 or b==0 else MUTED))
        else:
            g.add(Arrow(q,r,buff=.21,stroke_width=2.4,color=YELLOW if bi>=2 and (a==0 or b==0) else MUTED,max_tip_length_to_length_ratio=.12))
    for k,(x,y) in enumerate(centers):
        color=YELLOW if k==0 else BLUE if bi>=1 and (k in (1,2,3,4) or (k==5 and bi>=2)) else MUTED
        g.add(Dot([x,y,0],radius=.18,color=color),J(f'z{k}',16,color).move_to([x,y-.37,0]))
    if bi==0:
        g.add(J('復習: 8.3 マルコフブランケット',18,YELLOW).move_to([-3.76,1.87,0]))
        g.add(Rectangle(width=5.10,height=4.25,color=YELLOW,stroke_width=1.2,fill_opacity=0).move_to([-3.76,-.02,0]))
    elif bi>=2:g.add(J('親・子・子の他の親',18,YELLOW).move_to([-3.76,1.87,0]))
    return g

def figure(si,bi):
    if si==6:return graph_nodes(bi)
    g=VGroup()
    if si in (0,8):
        end=[0,0,0,40,40,40][bi] if si==0 else [0,40,40,40,40,40][bi]
        g.add(trace(DATA,end))
        if bi>=3 or si==8:g.add(J('開始',15,BLUE).move_to(pos(-2,-2)+LEFT*.29+DOWN*.26))
    elif si==1:
        y=[-2.,-2.,-1.4,-1.4,-2.,-2.][bi]
        g.add(cut_line(y),trace(DATA,[0,0,0,0,1,1][bi]))
    elif si==2:
        end=[0,1,2,2,4,6][bi];g.add(trace(DATA,end))
        if bi in (0,1,2):g.add(cut_line(float(DATA[end,1 if bi==0 else 0]),0 if bi==0 else 1))
    elif si==3:
        g.add(trace(DATA,[0,0,0,2,4,4][bi]))
        if bi<3:g.add(cut_line(-2.))
    elif si==4:
        g.add(trace(DATA,[0,10,40,80,80,80][bi]))
        if bi in (0,1,4):g.add(cut_line(-2.))
    elif si==5:
        g.add(trace(OVER,[0,0,1,2,10,20][bi],PURPLE),marker_pair(),alpha_slider(0. if bi==0 else -.8))
    elif si==7:
        g.add(trace(DATA,[0,2,4,4,6,40][bi]))
        if bi>=1:
            pt=np.random.default_rng(77).multivariate_normal([0,0],[[1,RHO],[RHO,1]])
            g.add(Dot(pos(*pt),radius=.11,color=PURPLE),Arrow(pos(-2,-2),pos(*pt),buff=.16,color=PURPLE,stroke_width=2,max_tip_length_to_length_ratio=.12))
    return g

def mini_conditional(bi,over=False,show_title=True):
    # Axis names and numerical ticks remain visible at 480p.
    x0,x1=2.52,5.72;y0=-1.92
    g=VGroup(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.3),Line([x0,y0,0],[x0,.56,0],color=MUTED,stroke_width=1.3))
    for v in (-.4,0,.4):
        x=4.10+3.45*v
        g.add(Line([x,y0,0],[x,y0-.07,0],color=MUTED,stroke_width=1),J(f'{v:g}',14,MUTED).move_to([x,y0-.24,0]))
    for v in (0,1,2):
        y=y0+1.05*v
        g.add(Line([x0-.07,y,0],[x0,y,0],color=MUTED,stroke_width=1),J(v,14,MUTED).move_to([x0-.27,y,0]))
    g.add(tex(r'z_i-\mu_i',23,MUTED).move_to([5.36,y0-.60,0]))
    if show_title:g.add(J('条件付き密度',16,MUTED).move_to([3.0,.82,0]))
    xx=np.linspace(-.46,.46,120)
    yy=np.exp(-.5*(xx/COND_SD)**2)/(COND_SD*np.sqrt(2*np.pi))
    g.add(path([np.array([4.10+3.45*float(x),y0+1.05*float(y),0]) for x,y in zip(xx,yy)],GREEN,2.8))
    if bi>=1:g.add(Dot([4.10,y0,0],radius=.072,color=YELLOW))
    if over and bi>=2:
        g.add(Dot([4.10+3.45*.25,y0+.09,0],radius=.065,color=BLUE),Dot([4.10-3.45*.20,y0+.09,0],radius=.065,color=PURPLE))
    return g

def mini_lag():
    x0,x1=2.65,5.88;y0=-1.94;y1=.55
    g=VGroup(Line([x0,y0,0],[x1,y0,0],stroke_width=1.3,color=MUTED),Line([x0,y0,0],[x0,y1,0],stroke_width=1.3,color=MUTED))
    for k in (0,1,2,4,8):
        x=x0+k/8*(x1-x0);g.add(Line([x,y0,0],[x,y0-.07,0],color=MUTED,stroke_width=1),J(k,14,MUTED).move_to([x,y0-.22,0]))
    for value in (0,.5,1):
        y=y0+value*(y1-y0);g.add(Line([x0-.07,y,0],[x0,y,0],color=MUTED,stroke_width=1),J(f'{value:g}',14,MUTED).move_to([x0-.28,y,0]))
    g.add(J('巡目',15,MUTED).move_to([5.40,y0-.59,0]),J('相関',15,MUTED).move_to([x0,y1+.22,0]))
    g.add(path([np.array([x0+k/8*(x1-x0),y0+RHO**(2*k)*(y1-y0),0]) for k in np.linspace(0,8,80)],PURPLE,2.8))
    return g

FORMULAS=[
 [r'\rho=0.98',r'z^{(0)}=(-2,-2)',r'A=1',r'z^{(20)}\approx(-2.22,-2.09)',r'A=1',r'p(z_1,z_2)'],
 [r'\Sigma_{12}=0.98',r'z_2=-2',r'p(z_1\mid z_2)',r'\sigma^2=1-\rho^2=0.0396',r'\sigma\approx0.199',r'z_1\mid z_2'],
 [r'z_1\sim p(z_1\mid z_2)',r'z_2\sim p(z_2\mid z_1^{(t+1)})',r'z_1\to z_2',r'z_1\to z_2\to z_3',r'z_1\to z_2',r'p(z_1,z_2)'],
 [r'p(z_2)',r'p(z_1\mid z_2)',r'p(z_1,z_2)=p(z_2)p(z_1\mid z_2)',r'A=1',r'p(z_i\mid z_{-i})>0',r'\rho=0.98'],
 [r'\sigma_c=0.199',r'z_1\to z_2',r'z^{(40)}\approx(-0.96,-1.03)',r'\rho^2=0.9604',r'O((L/l)^2)',r'A=1'],
 [r'\alpha=0',r'\alpha=-0.8',r'z_i^{\prime}=\mu_i+\alpha(z_i-\mu_i)+\sigma_i\sqrt{1-\alpha^2}\nu',r'\sigma_i^2',r'-1<\alpha<1',r'p(z_i\mid z_{-i})'],
 [r'8.3',r'z_i\mid z_{-i}',r'z_i\leftrightarrow z_j',r'p(z_i\mid z_{-i})',r'p(z_i\mid z_{-i})',r'z_i'],
 [r'\rho=0.98',r'z_1\to z_2',r'(z_1,z_2)\sim p(z_1,z_2)',r'p(z_B\mid z_{-B})',r'B\subseteq\{1,\ldots,M\}',r'B'],
 [r'z^{(0)}=(-2,-2)',r'z^{(20)}\approx(-2.22,-2.09)',r'\sigma_c=0.199',r'A=1',r'A=1',r'11.4']
]

def side_panel(si,bi):
    g=VGroup()
    if si in (1,5):
        g.add(mini_conditional(bi,over=si==5,show_title=not(si==1 and bi==0)))
        if si==1 and bi==0:
            g.add(Rectangle(width=4.35,height=3.72,color=YELLOW,stroke_width=1.2,fill_opacity=0).move_to([4.18,-.72,0]))
            g.add(J('復習: 2.3 多変量ガウス分布',16,YELLOW).move_to([4.18,1.00,0]))
    elif si==4:g.add(mini_lag())
    else:
        labels={
        0:[('開始', '(-2,-2)'),('相関','0.98'),('採用率','100%'),('20巡後','(-2.22,-2.09)'),('条件付き幅','0.199'),('目標','赤い雲')],
        2:[('更新中','横'),('更新中','縦'),('1巡','横→縦'),('3変数','1→2→3'),('順番','固定'),('次の問い','不変性')],
        3:[('縦の周辺','保つ'),('横の条件付き','引き直す'),('同時分布','積で復元'),('採用比','1'),('到達性','正の条件付き'),('この例','到達可能')],
        6:[('中心','黄色'),('ブランケット','青色'),('親・子','点灯'),('1変数','引き直す'),('必要な条件','引けること'),('課題','連続標本の相関')],
        7:[('相関','0.98'),('軸に沿う','短い移動'),('2変数まとめて','同時更新'),('実問題','同時抽出が難しい'),('組','重複も可'),('冒頭の点','20巡後')],
        8:[('開始','(-2,-2)'),('20巡後','(-2.22,-2.09)'),('条件付き幅','0.199'),('採用率','100%'),('速さ','別の問題'),('次節','11.4')]
        }
        a,b=labels[si][bi]
        g.add(J(a,18,MUTED).move_to([3.30,-.10,0]),J(b,22,YELLOW if bi>=3 else GREEN).move_to([4.95,-.10,0]))
        if si in (0,8) and bi>=3:
            g.add(J('同じ乱数の種: 43',15,MUTED).move_to([4.10,-.68,0]))
    return g

class PRML113GibbsSampling(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            if si!=6:self.add(cloud_axes(),cloud())
            self.visual=figure(si,0);self.side=side_panel(si,0)
            self.add(self.visual,self.side)
            self.note=None;self.formula=None
            for bi in range(len(story['beats'])):self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml113_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def run_beat(self,si,bi):
        if self.note is not None:self.remove(self.note)
        note=SCENES[si]['beats'][bi]['visual_note']
        if (si,bi)==(1,0):note='雲の向きを思い出す'
        if (si,bi)==(6,0):note='近所を思い出す'
        self.note=J(note,18,MUTED).move_to([4.15,2.45,0])
        if self.note.width>4.15:self.note.scale(4.15/self.note.width)
        self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        formula=FORMULAS[si][bi]
        # Formula stays beside the cloud, not on top of a curve or axis.
        self.formula=tex(formula,26,YELLOW).move_to([4.15,1.58,0])
        if self.formula.width>4.05:self.formula.scale(4.05/self.formula.width)
        self.add(self.formula)
        new_visual=figure(si,bi);new_side=side_panel(si,bi)
        duration=self.durations[self.beat_index]
        if si==6:
            self.beat(phases=[
                ('表示切替',.45,lambda:AnimationGroup(Succession(FadeOut(self.side),FadeIn(new_side)),Succession(FadeOut(self.visual),FadeIn(new_visual)),lag_ratio=0)),
                ('説明',duration-.45,lambda:Wait())])
            self.visual=new_visual
        else:
            self.beat(phases=[
                ('表示切替',.45,lambda:Succession(FadeOut(self.side),FadeIn(new_side))),
                ('移動',duration-.45,lambda:Transform(self.visual,new_visual))])
        self.side=new_side
