"""PRML 11.6: from a missed distant mode to a bridge of nearby ratios."""
import json
from math import exp, pi, sqrt
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex
from partition_model import ALPHAS, EXPERIMENT, energy, partition

BG='#10141F'
BLUE=ManimColor('#58C4DD'); YELLOW=ManimColor('#FFFF00')
GREEN=ManimColor('#83C167'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')

def J(s,size=19,color=WHITE):return jp(str(s),size,color)
def T(s,size=26,color=WHITE):return tex(s,size,color)
def X(z):return -6.0 + .75*(z+3)
def Y(value,normal=False):return -1.9 + (5.0 if normal else .90)*value
def gauss(z,alpha=0,normal=False):
    if normal:return exp(-(z-4*alpha)**2/2)/sqrt(2*pi)
    return exp(-float(energy(alpha,z)))
def path_curve(alpha,color,normal=False,opacity=1):
    xs=np.linspace(-3,7,200)
    m=VMobject().set_points_as_corners([[X(float(z)),Y(gauss(float(z),alpha,normal),normal),0] for z in xs])
    m.set_stroke(color,width=3.6,opacity=opacity)
    return m

def plot_axes(normal=False):
    g=VGroup()
    x0,x1=X(-3),X(7); base=Y(0)
    g.add(Line([x0,base,0],[x1,base,0],color=MUTED,stroke_width=1.6),
          Line([x0,base,0],[x0,1.00,0],color=MUTED,stroke_width=1.6))
    for z in (-2,0,2,4,6):
        x=X(z)
        g.add(Line([x,base,0],[x,base-.08,0],color=MUTED,stroke_width=1.2),J(z,14,MUTED).move_to([x,base-.27,0]))
    values=(0,.2,.4) if normal else (0,1,2,3)
    for v in values:
        yy=Y(v,normal)
        if yy>1.0:continue
        g.add(Line([x0,yy,0],[x0-.07,yy,0],color=MUTED,stroke_width=1.1),J(f'{v:g}',14,MUTED).move_to([x0-.33,yy,0]))
    g.add(T('z',22,MUTED).move_to([x1+.2,base+.05,0]),
          J('密度' if normal else '重み',18,MUTED).move_to([x0+.2,1.35,0]))
    return g

def sample_dots():
    xs=EXPERIMENT['direct_samples']
    return VGroup(*[Dot([X(float(x)),-2.60+.05*(i%3),0],radius=.024,color=BLUE) for i,x in enumerate(xs)])

def label_lines(si):
    g=VGroup()
    if si in (5,6):
        g.add(Line([-5.15,1.89,0],[-4.76,1.89,0],color=BLUE,stroke_width=3),J('基準',17,BLUE).move_to([-4.37,1.89,0]),
              Line([-.98,1.89,0],[-.59,1.89,0],color=PURPLE,stroke_width=3),J('対象',17,PURPLE).move_to([-.20,1.89,0]))
        g.add(Line([-3.6,1.89,0],[-3.21,1.89,0],color=GREEN,stroke_width=3),J('中間',17,GREEN).move_to([-2.80,1.89,0]))
    else:
        g.add(Line([-5.15,1.89,0],[-4.76,1.89,0],color=BLUE,stroke_width=3),J('基準 G',17,BLUE).move_to([-4.22,1.89,0]),
              Line([-.95,1.89,0],[-.56,1.89,0],color=PURPLE,stroke_width=3),J('対象 E',17,PURPLE).move_to([.02,1.89,0]))
    return g

# Right-hand diagrams share the same labelled axes and leave the subtitle band clear.
def small_axes(xlabel,ylabel,top,ticks):
    x0,x1=2.65,6.18; y0,y1=-1.85,.76
    g=VGroup(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.4),
             Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.4))
    for v in ticks:
        y=y0+(v/top)*(y1-y0)
        g.add(Line([x0-.07,y,0],[x0,y,0],color=MUTED,stroke_width=1),J(f'{v:g}',13,MUTED).move_to([x0-.31,y,0]))
    g.add(J(ylabel,15,MUTED).move_to([2.40,1.05,0]),J(xlabel,15,MUTED).move_to([5.30,-2.55,0]))
    return g,(x0,x1,y0,y1)

def bar_panel(values,names,top,ticks,ylabel,visible=None):
    g,box=small_axes('種類',ylabel,top,ticks)
    x0,x1,y0,y1=box
    n=len(values);centers=np.linspace(3.25,5.63,n)
    for k,(v,name,x) in enumerate(zip(values,names,centers)):
        color=[BLUE,PURPLE,GREEN][k%3]
        if visible is not None and k>=visible:
            g.add(J(name,15,MUTED).move_to([x,-2.17,0]),J('?',20,MUTED).move_to([x,-.50,0]))
            continue
        height=(v/top)*(y1-y0)
        g.add(Rectangle(width=.55,height=height,fill_color=color,fill_opacity=.70,stroke_width=0).move_to([x,y0+height/2,0]))
        g.add(J(name,15,color).move_to([x,-2.17,0]),J(f'{v:.3f}' if v<4 else f'{v:.2f}',15,color).move_to([x,y0+height+.20,0]))
    return g

def weight_panel():
    g,box=small_axes('標本番号','重み',20,(0,10,20));x0,x1,y0,y1=box
    values=np.sort(EXPERIMENT['direct_weights'])
    for i,v in enumerate(values):
        x=x0+.13+(i+1)/80*(x1-x0-.32)
        h=min(v/20,1)*(y1-y0)
        g.add(Line([x,y0,0],[x,y0+h,0],color=YELLOW if i>73 else BLUE,stroke_width=2.0 if i>73 else 1.3))
    for n,x in [(1,x0+.2),(40,(x0+x1)/2),(80,x1-.15)]:g.add(J(n,13,MUTED).move_to([x,-2.14,0]))
    return g

def alpha_plot_point(alpha):
    x0,x1,y0,y1=2.65,6.18,-1.85,.76
    return [x0+.15+(x1-x0-.30)*alpha,
            y0+partition(alpha)/8*(y1-y0),0]

def alpha_panel_static():
    g,box=small_axes('α','面積',8,(0,4,8));x0,x1,y0,y1=box
    pts=[alpha_plot_point(float(a)) for a in np.linspace(0,1,100)]
    g.add(VMobject().set_points_as_corners(pts).set_stroke(PURPLE,width=2.3))
    for a in (0,.5,1):
        xx=alpha_plot_point(a)[0]
        g.add(J(f'{a:g}',13,MUTED).move_to([xx,-2.14,0]))
    return g

def ratios_panel(visible=8):
    g,box=small_axes('段階','隣の比',3,(0,1,2,3));x0,x1,y0,y1=box
    for i,v in enumerate(EXPERIMENT['ratios']):
        x=x0+.24+(i+.5)*(x1-x0-.35)/8
        h=float(v)/3*(y1-y0)
        color=GREEN if i<visible else MUTED
        g.add(Rectangle(width=.22,height=h,fill_color=color,fill_opacity=.8,stroke_width=0).move_to([x,y0+h/2,0]))
        g.add(J(i+1,12,MUTED).move_to([x,-2.14,0]))
    return g

def recap_panel():
    g=VGroup(J('復習: 2.1 期待値',19,YELLOW).move_to([4.36,.80,0]))
    xs=[3.1,4.25,5.4];values=[1,2,3];weights=[.2,.3,.5]
    for x,v,w in zip(xs,values,weights):
        g.add(Dot([x,-.15,0],radius=.13+.15*w,color=BLUE),J(v,16,WHITE).move_to([x,-.55,0]),J(f'{w:.1f}',15,YELLOW).move_to([x,.25,0]))
    g.add(J('値 × 確率 → 平均',18,GREEN).move_to([4.3,-1.32,0]))
    g.add(Rectangle(width=4.55,height=3.10,stroke_color=YELLOW,stroke_width=1.1,fill_opacity=0).move_to([4.30,-.42,0]))
    return g

FORMULAS=[
 [r'Z_E/Z_G\;?',r'G(z)=z^2/2',r'L=80',r'0.241\ne 3',r'Z_E/Z_G',r'0.241\longrightarrow 3\;?'],
 [r'e^{-E(z)}',r'Z_E=\int e^{-E(z)}dz',r'p_E=e^{-E}/Z_E',r'Z_G\simeq2.507',r'Z_E\simeq7.520',r'Z_E/Z_G'],
 [r'p(D\mid M)',r'p(M\mid D)',r'Z_E/Z_G=3',r'\int e^{-E(z)}dz',r'e^{-E(z)}/e^{-G(z)}',r'\mathbb E_G[\cdot]'],
 [r'\mathbb E[X]',r'e^{-E}=e^{-G}e^{-E+G}',r'Z_E/Z_G=\mathbb E_G[e^{-E+G}]',r'w(z)=e^{-E(z)+G(z)}',r'\hat R={1\over L}\sum_{l=1}^{L}w(z^{(l)})',r'p_G,\,p_E'],
 [r'\max z^{(l)}\simeq2.432',r'w(4)\gg w(0)',r'\hat R\simeq0.241',r'p_G\not\simeq p_E',r'L\uparrow',r'G\longrightarrow E'],
 [r'\alpha:0\longrightarrow1',r'E_\alpha=(1-\alpha)G+\alpha E',r'p_\alpha=e^{-E_\alpha}/Z_\alpha',r'\alpha=k/8',r'p_0,p_1,\ldots,p_8',r'\Delta\alpha\ll1'],
 [r'Z_{k+1}/Z_k',r'Z_8/Z_0=\prod_{k=0}^{7}Z_{k+1}/Z_k',r'\prod_{k=0}^{7}\hat R_k\simeq2.963',r'8\times80=640',r'{1\over L}\sum_l T(z^{(l)},z)',r'z^{(l)}\sim p_\alpha'],
 [r'L=80,\ \max z\simeq2.432',r'\hat R_1\simeq0.241',r'\hat R_8\simeq2.963',r'Z_E/Z_G=3',r'\alpha:0\longrightarrow1',r'Z_E/Z_G\simeq 3']]

def curve_position(si,bi):
    if si==5:return [0,.125,.25,.5,.75,1][bi]
    if si==6:return [.0,.125,.25,.5,.75,1][bi]
    return 0

def side(si,bi):
    if si in (0,4,7):
        values=[EXPERIMENT['direct'],3,EXPERIMENT['bridge']]
        names=['直接','厳密','橋']
        count=3 if si==7 else 2
        visible=0 if si==0 and bi<3 else count
        return bar_panel(values[:count],names[:count],3.3,(0,1,2,3),'比',visible)
    if si==1:return bar_panel([partition(0),partition(1)],['基準','対象'],8,(0,4,8),'面積',1 if bi<4 else 2)
    if si==2:return bar_panel([1,3],['A=G','B=E'],3.3,(0,1,2,3),'事後の比')
    if si==3:return recap_panel() if bi==0 else weight_panel()
    if si==5:return alpha_panel_static()
    return ratios_panel([1,2,8,8,8,8][bi])

class PRML116PartitionFunction(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            normal=si in (5,6)
            axes=plot_axes(normal);self.add(axes)
            self.add(label_lines(si))
            base=path_curve(0,BLUE,normal,opacity=.6 if normal else .9)
            target=path_curve(1,PURPLE,normal,opacity=.6 if normal else .9)
            self.add(base,target)
            bridge=None
            bridge_tracker=ValueTracker(0) if normal else None
            if normal:
                bridge=always_redraw(lambda: path_curve(bridge_tracker.get_value(),GREEN,True,opacity=1))
                self.add(bridge)
            samples=sample_dots() if si in (0,3,4,7) else VGroup()
            if si in (3,4,7):self.add(samples)
            formula=None;note=None;panel=None
            for bi,beat in enumerate(story['beats']):
                if formula is not None:
                    self.remove(formula,note)
                    if si!=5:self.remove(panel)
                formula=T(FORMULAS[si][bi],27,YELLOW).move_to([4.32,2.36,0])
                if formula.width>4.65:formula.scale(4.65/formula.width)
                note_text='値と確率をつなぐ' if (si,bi)==(3,0) else beat['visual_note']
                note=J(note_text,18,MUTED).move_to([4.31,1.67,0])
                if note.width>4.55:note.scale(4.55/note.width)
                self.add(formula,note)
                if si!=5 or bi==0:
                    panel=side(si,bi)
                    self.add(panel)
                if si==5 and bi==0:
                    side_dot=always_redraw(lambda: Dot(alpha_plot_point(bridge_tracker.get_value()),radius=.09,color=YELLOW))
                    side_number=DecimalNumber(0,num_decimal_places=3,font_size=18,color=YELLOW)
                    side_number.add_updater(lambda m: m.set_value(bridge_tracker.get_value()).move_to(
                        [alpha_plot_point(bridge_tracker.get_value())[0],
                         min(1.05,alpha_plot_point(bridge_tracker.get_value())[1]+.25),0]))
                    self.add(side_dot,side_number)
                anim=[]
                if normal:
                    a=curve_position(si,bi)
                    # The density and its peak marker share one tracker throughout the motion.
                    marker=always_redraw(lambda: Dot([X(4*bridge_tracker.get_value()),
                        Y(gauss(4*bridge_tracker.get_value(),bridge_tracker.get_value(),True),True),0],
                        radius=.09,color=YELLOW))
                    self.add(marker)
                    anim.append(bridge_tracker.animate.set_value(a))
                elif si==0 and bi==2:
                    self.add(samples)
                    anim.append(LaggedStart(*[FadeIn(d) for d in samples],lag_ratio=.025))
                else:
                    highlighted=target if si in (1,2,4) and bi in (2,3,4) else base
                    anim.append(ShowPassingFlash(highlighted.copy().set_stroke(YELLOW,width=6),time_width=.18))
                self.beat(*anim,actions=[dict(name='分布を動かす' if normal else '曲線を強調',start=0,end=self.durations[self.beat_index])])
                if normal:self.remove(marker)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml116_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
