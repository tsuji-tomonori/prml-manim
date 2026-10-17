"""PRML 10.5: move one tangent until it bounds an uncertain prediction."""
import json
from pathlib import Path
import numpy as np
from manim import *
from local_bound_model import sigmoid, bound, density, upper_sigmoid, optimum, tangent_exp, lam
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex

BG='#10141F'
BLUE=ManimColor('#58C4DD')
GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00')
PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255')
MUTED=ManimColor('#A8B2C5')
XI_OPT, F_OPT=optimum()

FORMULAS=[
 ['',r'a\sim\mathcal N(1,4)',r'\sigma(a)=1/(1+e^{-a})',r'\sigma(E[a])=0.731',r'E[\sigma(a)]=0.648','','',r'I=E[\sigma(a)]'],
 ['',r'\sigma(a)\,p(a)',r'B(a,\xi)\leq\sigma(a)','','','','',''],
 [r'f(x)=e^{-x}',r'\xi=1',r'f\prime(\xi)=-e^{-\xi}',r'y(x)\leq f(x)',r'y=f(\xi)+f\prime(\xi)(x-\xi)',r'f\prime(\xi)=-e^{-\xi}',r'x=1',r'f(1)=\max y(1)'],
 [r'\eta=f\prime(\xi)<0',r'y=\eta x-g(\eta)',r'g(\eta)',r'g(\eta)=\max_x\{\eta x-f(x)\}',r'g(\eta)=\max_x\{\eta x-f(x)\}',r'x=1',r'f(x)=\max_\eta\{\eta x-g(\eta)\}',''],
 ['',r'\log\sigma(x)',r'\log\sigma(x)\leq\eta x-g(\eta)', '',r'\log\sigma(x)=x/2-\log(2\cosh(x/2))',r'(+2.5)^2=(-2.5)^2=6.25',r'u=x^2',''],
 [r'\xi=2.5',r'B(x,\xi)\leq\sigma(x)',r'\lambda(\xi)=\tanh(\xi/2)/(4\xi)',r'\eta=-\lambda(\xi)',r'B=\sigma(\xi)e^{(x-\xi)/2-\lambda(\xi)(x^2-\xi^2)}',r'\log B=C+x/2-\lambda x^2','',''],
 [r'a\sim\mathcal N(1,4)',r'I=\int\sigma(a)p(a)\,da',r'F(\xi)=\int B(a,\xi)p(a)\,da',r'I\geq F(\xi)',r'F(1)=0.568',r'\xi^*=2.332,\quad F(\xi^*)=0.600',r'I-F(\xi^*)\simeq0.048',''],
 [r'a\sim\mathcal N(1,4)',r'\sigma(E[a])=0.731',r'I=0.648',r'F(\xi^*)=0.600',r'F(\xi^*)\leq I', '','',r'0.600<0.648<0.731'],
]

# Values are display states, not additional numerical claims.
XI_BY_BEAT=[
 [2.5]*8,[2.5]*8,
 [1.0,1.0,.65,1.75,1.0,2.4,.65,1.0],
 [.65,1.0,1.75,2.3,1.5,.65,1.0,1.0],
 [2.5]*8,
 [2.5,2.5,2.5,2.5,2.5,2.5,3.4,2.5],
 [2.5,2.5,2.5,2.5,1.0,XI_OPT,XI_OPT,XI_OPT],
 [XI_OPT]*8,
]
CURSOR_BY_BEAT=[
 [1.,-2.,1.,1.,-1.8,0.,2.2,1.],
 [-3.,-1.5,1.,2.2,-2.2,1.8,-.7,0.],
 [1.,1.,.65,1.75,1.,2.4,1.,1.],
 [1.2,1.8,2.2,1.0,2.7,1.,1.,1.],
 [-3.,-1.5,1.4,-1.2,0.,2.5,-2.5,2.5],
 [2.5,-2.5,1.,0.,2.5,-1.,3.4,1.],
 [-2.,1.,0.,1.,1.,2.2,-1.,1.],
 [1.,-2.,1.,-1.,0.,2.2,1.,1.],
]

def J(value,size=20,color=WHITE):
    return jp(str(value),size,color)

def xy(x,y,kind='sigmoid'):
    return np.array([-5.05+(x+6)*.5 if kind=='sigmoid' else -5.05+x*1.5,
                     -2.10+3.3*y if kind=='sigmoid' else -2.10+3.0*y,0])

def path_from_values(xs,ys,kind,color,width=3):
    points=[xy(float(x),float(y),kind) for x,y in zip(xs,ys) if -0.001<=y<=1.06]
    if len(points)<2:
        return VGroup()
    return VMobject().set_points_as_corners(points).set_stroke(color,width=width)

def main_axes(kind):
    base=VGroup()
    right=6 if kind=='sigmoid' else 4
    base.add(Line(xy(-6 if kind=='sigmoid' else 0,0,kind),xy(right,0,kind),color=MUTED,stroke_width=1.5),
             Line(xy(-6 if kind=='sigmoid' else 0,0,kind),xy(-6 if kind=='sigmoid' else 0,1.06,kind),color=MUTED,stroke_width=1.5))
    for x in ([-6,-3,0,3,6] if kind=='sigmoid' else [0,1,2,3,4]):
        pos=xy(x,0,kind)
        base.add(Line(pos,pos+DOWN*.08,color=MUTED,stroke_width=1),
                 J(str(x),15,MUTED).move_to(pos+DOWN*.26))
    for y in ([0,.5,1] if kind=='sigmoid' else [0,.5,1]):
        pos=xy(-6 if kind=='sigmoid' else 0,y,kind)
        base.add(Line(pos,pos+LEFT*.08,color=MUTED,stroke_width=1))
        if y>0:
            base.add(J(str(y),14,MUTED).move_to(pos+LEFT*.32))
    base.add(J('スコア' if kind=='sigmoid' else '入力',16,MUTED).move_to([1.70,-2.43,0]),
             J('確率' if kind=='sigmoid' else '高さ',16,MUTED).move_to([-5.08,1.66,0]))
    return base

def main_trace(si,bi):
    """Keep a stable submobject order while curves and contact marks move."""
    kind='exp' if si in (2,3) else 'sigmoid'
    xi=XI_BY_BEAT[si][bi]
    cursor=CURSOR_BY_BEAT[si][bi]
    xs=np.linspace(0,4,180) if kind=='exp' else np.linspace(-6,6,280)
    ys=np.exp(-xs) if kind=='exp' else sigmoid(xs)
    original=path_from_values(xs,ys,kind,RED,3.5)
    if kind=='exp':
        tangent=path_from_values(xs,tangent_exp(xs,xi),kind,YELLOW,3.2)
        contact=Dot(xy(xi,np.exp(-xi),kind),radius=.07,color=YELLOW)
        guide=Line(xy(1,0,kind),xy(1,np.exp(-1),kind),color=GREEN,stroke_width=1.8)
        guide.set_opacity(1 if bi>=6 else 0)
        return VGroup(original,tangent,contact,guide)

    # All sigmoid beats have exactly the same six slots.  A single yellow dot
    # must never be transformed into a two-dot VGroup: it creates filled wedges.
    if si==4 and bi in (1,2):
        secondary=path_from_values(xs,upper_sigmoid(xs,1),kind,PURPLE,3)
    elif si>=5:
        secondary=path_from_values(xs,bound(xs,xi),kind,BLUE,3.1)
    else:
        secondary=path_from_values(xs,ys,kind,BLUE,3.1)
        secondary.set_opacity(0)
    if si==4 and bi not in (1,2):
        secondary=path_from_values(xs,ys,kind,PURPLE,3)
        secondary.set_opacity(0)
    contacts_visible=si==5 or (si==4 and bi>=5)
    contact_xi=2.5 if si==4 else xi
    left=Dot(xy(-contact_xi,float(sigmoid(-contact_xi))),radius=.074,color=YELLOW)
    right=Dot(xy(contact_xi,float(sigmoid(contact_xi))),radius=.074,color=YELLOW)
    if not contacts_visible:
        left.set_opacity(0)
        right.set_opacity(0)
    cursor=float(np.clip(cursor,-5.8,5.8))
    point=xy(cursor,float(sigmoid(cursor)))
    marker=Dot(point,radius=.083,color=YELLOW)
    if contacts_visible:
        marker.set_opacity(0)
    guide=Line(xy(cursor,0),point,color=YELLOW,stroke_width=1.35)
    if not (si in (0,7) and bi in (1,2,3,4,5)):
        guide.set_opacity(0)
    return VGroup(original,secondary,left,right,marker,guide)

def density_panel(scene_index):
    g=VGroup(J('スコアの分布',18,MUTED).move_to([4.48,.67,0]))
    x0,x1=2.85,6.15
    y0=-1.53
    g.add(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.2),
          Line([x0,y0,0],[x0,.05,0],color=MUTED,stroke_width=1.2))
    xx=np.linspace(-6,6,120)
    curve=VMobject().set_points_as_corners([[x0+(x+6)*(x1-x0)/12,y0+6.2*density(x),0] for x in xx]).set_stroke(BLUE if scene_index==0 else PURPLE,width=3)
    g.add(curve)
    for v in (-6,0,6):
        px=x0+(v+6)*(x1-x0)/12
        g.add(Line([px,y0,0],[px,y0-.07,0],color=MUTED,stroke_width=1),
              J(str(v),14,MUTED).move_to([px,y0-.25,0]))
    for v in (0.1,0.2):
        yy=y0+6.2*v
        g.add(Line([x0,yy,0],[x0-.07,yy,0],color=MUTED,stroke_width=1),
              J(str(v),13,MUTED).move_to([x0-.32,yy,0]))
    g.add(J('a',17,MUTED).move_to([6.49,y0-.02,0]),
          J('密度',15,MUTED).move_to([2.56,.17,0]))
    return g

def side_panel(si,bi):
    g=VGroup()
    if si in (0,6):
        g.add(density_panel(si))
        if si==0:
            label=['前節の問い','中心1、標準偏差2','復習: 4.5','中心だけ 0.731',
                   '山全体 0.648','低いスコアにも重み','積分は難しい','下から押さえる？'][bi]
            color=YELLOW if bi in (3,4) else GREEN
        else:
            label=['同じ山','幅ごとの面積','下界の面積','赤 ≥ 青',
                   'F(1) = 0.568','F(2.332) = 0.600','真の面積 0.648','3つの値を比較'][bi]
            color=BLUE if bi in (2,4,5) else GREEN
        g.add(J(label,19,color).move_to([4.47,-2.36,0]))
        if si==0 and bi==2:
            recap=J('復習: 4.5',17,WHITE).move_to([4.47,1.10,0])
            g.add(SurroundingRectangle(recap,color=YELLOW,buff=.12,stroke_width=1.5),recap)
        if (si==6 and bi in (1,2,3,4,5,6)) or (si==0 and bi==2):
            # Small density-weighted bars carry the integral meaning; all share x.
            xx=np.linspace(-5,5,13)
            heights=density(xx)*(bound(xx,XI_BY_BEAT[si][bi]) if si==6 and bi in (2,3,4,5,6) else sigmoid(xx))
            for x,h in zip(xx,heights):
                px=2.85+(x+6)*3.3/12
                g.add(Line([px,-1.53,0],[px,-1.53+6.2*float(h),0],
                           color=BLUE if si==6 and bi in (2,3,4,5,6) else YELLOW,stroke_width=4,stroke_opacity=.75))
        return g
    if si==7:
        g.add(J('同じ山・同じ曲線',19,MUTED).move_to([4.45,.78,0]))
        vals=[(.731,'中心だけ',YELLOW),(.648,'真の平均',RED),(.600,'下界',BLUE)]
        shown=1 if bi<2 else 2 if bi==2 else 3
        for j,(value,label,color) in enumerate(vals[:shown]):
            y=.05-j*.92
            g.add(J(label,17,color).move_to([3.24,y+.2,0]),
                  Line([2.89,y-.18,0],[2.89+2.9*value,y-.18,0],color=color,stroke_width=17),
                  J(f'{value:.3f}',18,color).move_to([6.05,y-.18,0]))
        g.add(Line([2.9,-2.52,0],[5.8,-2.52,0],color=MUTED,stroke_width=1.2),
              J('0',14,MUTED).move_to([2.9,-2.76,0]),
              J('0.5',14,MUTED).move_to([4.35,-2.76,0]),
              J('1',14,MUTED).move_to([5.8,-2.76,0]),
              J('平均確率',15,MUTED).move_to([4.45,-2.28,0]))
        return g
    if si==1:
        g.add(J('大きなモデルの一部分',18,MUTED).move_to([4.48,.77,0]))
        positions=[(3.0,-.2),(4.45,-.2),(5.9,-.2)]
        colors=[PURPLE,RED,GREEN]
        labels=['事前','シグモイド','ほかの因子']
        for i,(x,y) in enumerate(positions):
            g.add(Dot([x,y,0],radius=.11,color=colors[i]),J(labels[i],16,colors[i]).move_to([x,y+.44,0]))
        for i in range(2):
            g.add(Arrow([positions[i][0]+.2,-.2,0],[positions[i+1][0]-.2,-.2,0],
                        buff=0,tip_length=.12,color=MUTED,stroke_width=2))
        g.add(J(['どこが難しい？','1因子を切り出す','全スコアで下に','局所変分法',
                 '接点を探す','別の曲線で練習','向きを確認','接線へ'][bi],20,YELLOW).move_to([4.48,-1.18,0]))
        g.add(J('元の因子 ≥ 置き換え',17,BLUE).move_to([4.48,-2.12,0]))
        return g
    if si in (2,3):
        xi=XI_BY_BEAT[si][bi]
        g.add(J('接点のつまみ',18,MUTED).move_to([4.48,.8,0]))
        g.add(Line([2.95,.05,0],[6.0,.05,0],color=MUTED,stroke_width=2))
        for val in (0,1,2,3):
            px=2.95+val*3.05/3
            g.add(Line([px,.05,0],[px,-.04,0],color=MUTED,stroke_width=1),
                  J(str(val),14,MUTED).move_to([px,-.24,0]))
        px=2.95+xi*3.05/3
        g.add(Dot([px,.05,0],radius=.12,color=YELLOW),
              tex(r'\xi='+f'{xi:.2f}',24,YELLOW).move_to([4.48,-.76,0]),
              J('接する場所',16,MUTED).move_to([4.48,-1.28,0]))
        g.add(J('1での高さを比較' if (si==2 and bi>=6) or si==3 else '線全体が下界',19,GREEN)
              .move_to([4.48,-2.18,0]))
        return g
    if si==4:
        if bi<5:
            g.add(J('対数を取ると接線は上',19,MUTED).move_to([4.48,.8,0]),
                  J('曲線  ≤  上界',23,PURPLE).move_to([4.48,-.35,0]),
                  J('欲しいのは下界',20,YELLOW).move_to([4.48,-1.65,0]))
            return g
        g.add(J('同じ二乗へ折りたたむ',19,MUTED).move_to([4.48,.8,0]))
        for x,label in [(3.13,'−2.5'),(5.78,'+2.5')]:
            g.add(Dot([x,-.15,0],radius=.1,color=YELLOW),J(label,19,YELLOW).move_to([x,-.55,0]),
                  Arrow([x,-.75,0],[4.45,-1.55,0],buff=.13,tip_length=.12,color=MUTED,stroke_width=2))
        g.add(Dot([4.45,-1.72,0],radius=.13,color=GREEN),
              J('二乗すると 6.25',20,GREEN).move_to([4.48,-2.3,0]))
        return g
    # Scene 6: a shared xi and two contact points, or the quadratic coefficient.
    xi=XI_BY_BEAT[si][bi]
    g.add(J('共通の接点',18,MUTED).move_to([4.48,.8,0]),
          tex(r'\xi='+f'{xi:.2f}',27,YELLOW).move_to([4.48,.2,0]),
          Line([2.98,-.68,0],[6.02,-.68,0],color=MUTED,stroke_width=1.8))
    for sign in (-1,1):
        px=4.5+sign*xi*.42
        g.add(Dot([px,-.68,0],radius=.1,color=YELLOW),
              tex(('-' if sign<0 else '+')+r'\xi',24,YELLOW).move_to([px,-1.1,0]))
    g.add(tex(r'\lambda(\xi)='+f'{lam(xi):.3f}',24,BLUE).move_to([4.48,-1.9,0]),
          J('青い下界が2点で接する',17,GREEN).move_to([4.48,-2.49,0]))
    return g

class PRML105LocalVariationalMethods(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={entry['id']:entry for entry in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            kind='exp' if si in (2,3) else 'sigmoid'
            self.add(main_axes(kind))
            self.trace=main_trace(si,0)
            self.side=side_panel(si,0)
            self.note=None
            self.formula=None
            self.add(self.trace,self.side)
            for bi in range(len(story['beats'])):
                self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml105_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def run_beat(self,si,bi):
        if self.note is not None:self.remove(self.note)
        note=SCENES[si]['beats'][bi]['visual_note']
        self.note=J(note,19,MUTED).move_to([4.48,2.51,0])
        if self.note.width>4.2:self.note.scale(4.2/self.note.width)
        self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        self.formula=None
        formula=FORMULAS[si][bi]
        if formula:
            self.formula=tex(formula,26,YELLOW).move_to([4.48,1.89,0])
            if self.formula.width>4.15:self.formula.scale(4.15/self.formula.width)
            self.add(self.formula)
        target_trace=main_trace(si,bi)
        target_side=side_panel(si,bi)
        duration=self.durations[self.beat_index]
        self.beat(phases=[('panel',.55,lambda:Succession(FadeOut(self.side),FadeIn(target_side))),
                          ('graph',duration-.55,lambda:Transform(self.trace,target_trace))])
        self.side=target_side
