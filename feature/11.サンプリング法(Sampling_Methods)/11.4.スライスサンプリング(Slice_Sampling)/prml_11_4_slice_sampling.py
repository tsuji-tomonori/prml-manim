"""PRML 11.4: a horizontal slice finds the width of a one-dimensional move."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene,jp,tex
from slice_model import f,boundary

BG='#10141F'
BLUE=ManimColor('#58C4DD');GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00');PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255');MUTED=ManimColor('#A8B2C5')

def J(value,size=18,color=WHITE):return jp(str(value),size,color)
def P(z,h):return np.array([-2.64+.95*z,-1.84+2.9*h,0.])
def path(points,color=RED,width=2.8,opacity=1.):
    m=VMobject();m.set_points_as_corners(points);m.set_stroke(color,width=width,opacity=opacity);return m

def twin(z):
    return float(np.exp(-.5*((z+1.5)/.55)**2)+.7*np.exp(-.5*((z-1.5)/.65)**2))

def slice_segments(height,double=False):
    if not double:
        b=boundary(height);return [(-b,b)]
    xs=np.linspace(-3.5,3.5,1401)
    inside=[twin(x)>=height for x in xs]
    spans=[];start=None
    for i,on in enumerate(inside):
        if on and start is None:start=i
        if start is not None and (not on or i==len(xs)-1):
            spans.append((float(xs[start]),float(xs[i-1 if not on else i])));start=None
    return spans

def plot_axes(double=False):
    g=VGroup()
    g.add(Line(P(-3.5,0),P(3.5,0),color=MUTED,stroke_width=1.5))
    g.add(Line(P(-3.5,0),P(-3.5,1.32),color=MUTED,stroke_width=1.5))
    for v in (-3,-2,-1,0,1,2,3):
        q=P(v,0);g.add(Line(q,q+DOWN*.07,color=MUTED,stroke_width=1),J(v,14,MUTED).move_to(q+DOWN*.23))
    for v in (.5,1):
        q=P(-3.5,v);g.add(Line(q,q+LEFT*.07,color=MUTED,stroke_width=1),J(f'{v:g}',14,MUTED).move_to(q+LEFT*.30))
    g.add(tex('z',24,MUTED).move_to(P(3.66,-.02)),tex('f(z)',24,MUTED).move_to(P(-3.42,1.43)))
    xs=np.linspace(-3.5,3.5,210)
    vals=[twin(x) if double else float(f(x)) for x in xs]
    g.add(path([P(float(x),y) for x,y in zip(xs,vals)]))
    return g

def cross(x,y):
    q=P(x,y);return VGroup(Line(q+LEFT*.09+UP*.09,q+RIGHT*.09+DOWN*.09,color=RED,stroke_width=2.7),Line(q+LEFT*.09+DOWN*.09,q+RIGHT*.09+UP*.09,color=RED,stroke_width=2.7))

def slice_line(height,double=False):
    g=VGroup(DashedLine(P(-3.5,height),P(3.5,height),color=YELLOW,stroke_opacity=.42,stroke_width=1.4,dash_length=.12))
    for a,b in slice_segments(height,double):
        g.add(Line(P(a,height),P(b,height),color=YELLOW,stroke_width=5))
    return g

def marker(z,color=BLUE):
    return Dot(P(z,0),radius=.10,color=color)

def bracket(left,right,color=PURPLE):
    y=-2.52
    a=np.array([P(left,0)[0],y,0]);b=np.array([P(right,0)[0],y,0])
    return VGroup(Line(a,b,color=color,stroke_width=3),Line(a+UP*.10,a+DOWN*.10,color=color,stroke_width=2),Line(b+UP*.10,b+DOWN*.10,color=color,stroke_width=2))

def area_under_curve():
    xs=np.linspace(-3.5,3.5,180)
    points=[P(-3.5,0),*[P(float(x),float(f(x))) for x in xs],P(3.5,0)]
    return Polygon(*points,color=YELLOW,fill_color=YELLOW,fill_opacity=.10,stroke_width=0)

def vertical_band(z):
    zz=[z-.055,z+.055]
    a=P(zz[0],0);b=P(zz[1],0);c=P(zz[1],float(f(zz[1])));d=P(zz[0],float(f(zz[0])))
    return Polygon(a,b,c,d,color=YELLOW,fill_color=YELLOW,fill_opacity=.30,stroke_width=0)

def invisible(mob):
    mob.set_opacity(0)
    return mob


def figure(si,bi):
    """The same geometric slots in every beat keep Manim transforms stable."""
    double=si==6
    g=VGroup()
    g.add(marker(-1.5 if double else 0))
    target_visible=(si==0 and bi>=2) or (si==4 and bi>=3) or (si==5 and bi>=2) or (si==8 and bi>=4)
    target=marker(1.4,GREEN)
    g.add(target if target_visible else invisible(target))
    vertical=Line(P(0,0),P(0,1.25),color=BLUE,stroke_width=2.3)
    g.add(vertical if si==1 and bi>=1 else invisible(vertical))
    area=area_under_curve()
    g.add(area if si==2 and bi<=1 else invisible(area))
    band=vertical_band(1.4)
    g.add(band if (si==1 and bi==0) or (si==2 and 2<=bi<=4) else invisible(band))
    heights={5:[.95,.6,.15,.05,.15,.95],6:[.5,.5,.02,.5,.5,.02],7:[.15,.15,.6,.6,.15,.15]}
    height=heights.get(si,[.15]*7)[bi]
    visible=(si in (3,4,5,6,7)) or (si==0 and bi>=3) or (si==1 and bi>=3) or (si==2 and bi>=5) or (si==8 and bi>=1)
    dashed=DashedLine(P(-3.5,height),P(3.5,height),color=YELLOW,stroke_opacity=.42,stroke_width=1.4,dash_length=.12)
    g.add(dashed if visible else invisible(dashed))
    spans=slice_segments(height,double)
    first=Line(P(*[spans[0][0],height]),P(*[spans[0][1],height]),color=YELLOW,stroke_width=5)
    g.add(first if visible else invisible(first))
    if len(spans)>1:
        second=Line(P(spans[1][0],height),P(spans[1][1],height),color=YELLOW,stroke_width=5)
        g.add(second if visible else invisible(second))
    else:
        g.add(invisible(Line(P(-3.4,height),P(-3.39,height),color=YELLOW,stroke_width=5)))
    intervals={3:[(-.2,.3),(-2.2,.3),(-2.2,2.3),(-2.2,2.3),(-2.2,2.3),(-2.2,2.3)],
               4:[(-2.2,2.3),(-2.2,2.0),(-2.2,2.0),(-2.2,2.0),(-2.2,2.0),(-2.2,2.0)],
               7:[(-.2,.3),(-1.2,1.3),(-1.2,1.3),(-1.2,1.3),(-1.2,1.3),(-1.2,1.3)]}
    bracket_visible=si in intervals or (si==0 and bi>=4) or (si==8 and bi>=2)
    if si in intervals:limits=intervals[si][bi]
    elif si==8:limits=(-2.2,2.3 if bi<3 else 2.0)
    else:limits=(-2.2,2.3)
    br=bracket(*limits)
    g.add(br if bracket_visible else invisible(br))
    candidate=None
    if si==1 and bi==2:candidate=(0,.72,YELLOW)
    elif si==1 and bi>=3:candidate=(0,.15,YELLOW)
    elif si==2 and bi==5:candidate=(1.4,.15,GREEN)
    elif si==4:candidate=(2,.15,PURPLE) if bi<=1 else (1.4,.15,GREEN)
    elif si==6 and bi>=1:candidate=(1.5,0,PURPLE)
    elif si==8 and bi>=3:candidate=(2,.15,PURPLE) if bi==3 else (1.4,.15,GREEN)
    dot=Dot(P(candidate[0],candidate[1]),radius=.09,color=candidate[2]) if candidate else Dot(P(0,.15),radius=.09,color=YELLOW)
    g.add(dot if candidate else invisible(dot))
    reject=(si==4 and bi>=1) or (si==8 and bi>=3)
    symbol=cross(2,.15)
    g.add(symbol if reject else invisible(symbol))
    return g

def number_line(left=-2.2,right=2.3,current=0,candidate=None):
    g=VGroup();y=-.90
    def X(z):return 4.13+.51*z
    g.add(Line([X(-3),y,0],[X(3),y,0],color=MUTED,stroke_width=1.4))
    for z in (-2,0,2):
        x=X(z);g.add(Line([x,y-.07,0],[x,y+.07,0],color=MUTED,stroke_width=1),J(z,14,MUTED).move_to([x,y-.23,0]))
    g.add(tex('z',22,MUTED).move_to([5.91,y-.03,0]))
    g.add(Line([X(left),y+.18,0],[X(right),y+.18,0],color=PURPLE,stroke_width=3))
    g.add(Dot([X(current),y+.18,0],radius=.07,color=BLUE))
    if candidate is not None:g.add(Dot([X(candidate),y+.18,0],radius=.08,color=GREEN if candidate==1.4 else RED))
    g.add(J('候補の区間',16,MUTED).move_to([4.13,-1.68,0]))
    return g

def height_slider(height):
    g=VGroup();x=4.13;y0=-1.88;scale=1.36
    g.add(Line([x,y0,0],[x,y0+1.25*scale,0],color=MUTED,stroke_width=1.6))
    for value in (0,.5,1):
        y=y0+value*scale
        g.add(Line([x-.08,y,0],[x+.08,y,0],color=MUTED,stroke_width=1),J(f'{value:g}',14,MUTED).move_to([x-.30,y,0]))
    y=y0+height*scale
    g.add(tex('u',22,MUTED).move_to([x+.28,-.04,0]))
    return g

def mini_two_coordinates(bi):
    g=VGroup();x0,x1=2.70,5.75;y0,y1=-1.85,.57
    g.add(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.3),Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.3))
    for value in (-2,0,2):
        x=4.20+value*.65;y=-.62+value*.50
        g.add(Line([x,y0,0],[x,y0-.07,0],color=MUTED,stroke_width=1),J(value,13,MUTED).move_to([x,y0-.23,0]))
        g.add(Line([x0-.07,y,0],[x0,y,0],color=MUTED,stroke_width=1),J(value,13,MUTED).move_to([x0-.25,y,0]))
    g.add(tex('z_1',20,MUTED).move_to([5.96,y0-.04,0]),tex('z_2',20,MUTED).move_to([x0,y1+.27,0]))
    angles=np.linspace(0,2*np.pi,90)
    for rad,opacity in [(.55,.55),(1.,.85)]:
        pts=[]
        for t in angles:
            u=rad*np.cos(t);v=rad*np.sin(t)
            pts.append(np.array([4.18+u*.93+v*.20,-.60+u*.38+v*.46,0]))
        g.add(path(pts,RED,2,opacity))
    g.add(Dot([4.10,-.66,0],radius=.07,color=BLUE))
    if bi>=1:g.add(Line([3.65,-.67,0],[4.58,-.67,0],color=YELLOW,stroke_width=2.4))
    if bi>=2:g.add(Line([4.48,-1.13,0],[4.48,.02,0],color=YELLOW,stroke_width=2.4))
    return g

FORMULAS=[
 [r'w=0.5',r'z=0',r'0.5<1.4',r'z^{\prime}=1.4',r'u=0.15',r'z=0\to1.4'],
 [r'2.1',r'f(0)=1.25',r'0<u<f(z)',r'u=0.15',r'S_u=\{z:f(z)>u\}',r'p(z)'],
 [r'0<u<f(z)',r'p(z,u)=1/Z',r'f(z)\,dz',r'\int_0^{f(z)}du=f(z)',r'p(z)=\int p(z,u)du=f(z)/Z',r'u\to z\to u',r'z\sim p(z\mid u)'],
 [r'w=0.5',r'f(z_{\min})>u',r'f(z_{\max})>u',r'[z_{\min},z_{\max}]=[-2.2,2.3]',r'w=0.5',r'z^{\prime}\sim U(z_{\min},z_{\max})'],
 [r'f(2.0)=0.135<0.15',r'z_{\max}=2.0',r'f(1.4)=0.193>0.15',r'z^{\prime}=1.4',r'z^{(t)}\in[z_{\min},z_{\max}]',r'u=0.15'],
 [r'u=0.95',r'u=0.6',r'u=0.15',r'u=0.05',r'u\mapsto S_u',r'w=0.5'],
 [r'11.13',r'u=0.5',r'u=0.02',r'p(z\to z^{\prime})',r'w=0.5',r'z_1,z_2'],
 [r'p(z_1\mid z_2)',r'z_1\sim p(z_1\mid z_2)',r'z_2\sim p(z_2\mid z_1)',r'f_i(z_i)\propto p(z_i\mid z_{-i})',r'w=0.5',r'z=0'],
 [r'z=0,\ w=0.5',r'u=0.15',r'[-2.2,2.3]',r'2.0\to1.4',r'S_u=\{z:f(z)>u\}',r'11.5']
]

def side(si,bi):
    g=VGroup()
    label={
      0:['11.2の歩幅','頂上から出発','予想の間','意外な次点','横線が手がかり','最後に再会'],
      1:['復習: 2.1 密度と面積','現在点の高さ','補助変数 u','今回の高さ','黄色がスライス','面積を数える'],
      2:['領域内は一様','曲線下で同じ濃さ','細い縦帯','積分は縦に足す','周辺分布','交互に更新','直接は難しい'],
      3:['最初は幅0.5','左へ拡張','右へ拡張','両端は外','初期幅は上限でない','候補を引く'],
      4:['候補 2.0 は外','右端を縮小','候補 1.4 は内','1.4 を採用','古い点を残す','高さで範囲を探す'],
      5:['頂上付近','少し広い','裾まで届く','さらに広い','高さのつまみ','次は2山'],
      6:['2山の模式図','高い切り口','低い切り口','詳細釣り合い','移動は保証しない','多変数へ'],
      7:['11.3の断面','横を更新','縦を更新','比例する密度','相関は残る','冒頭へ'],
      8:['同じ山と点','同じ高さ','同じ拡張','同じ候補','問いの答え','次は11.5']
    }[si][bi]
    g.add(J(label,20,YELLOW).move_to([4.19,.45,0]))
    if si==7:
        g.add(mini_two_coordinates(bi))
    elif si==5:
        g.add(height_slider([.95,.6,.15,.05,.15,.95][bi]))
    else:
        if si in (3,4,8):
            span={3:[(-.2,.3),(-2.2,.3),(-2.2,2.3),(-2.2,2.3),(-2.2,2.3),(-2.2,2.3)],4:[(-2.2,2.3),(-2.2,2.0),(-2.2,2.0),(-2.2,2.0),(-2.2,2.0),(-2.2,2.0)],8:[(-.2,.3),(-.2,.3),(-2.2,2.3),(-2.2,2.0),(-2.2,2.0),(-2.2,2.0)]}[si][bi]
            candidate=(2.0 if si==4 and bi<=1 else 1.4 if (si==4 and bi>=2) or (si==8 and bi>=3) else None)
            g.add(number_line(*span,candidate=candidate))
        else:
            g.add(number_line(-.2,.3))
    if si==1 and bi==0:
        g.add(Line([2.38,1.07,0],[5.98,1.07,0],color=YELLOW,stroke_width=2))
    return g

class PRML114SliceSampling(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            self.add(plot_axes(si==6))
            self.visual=figure(si,0);self.side=side(si,0)
            self.add(self.visual,self.side)
            self.slider_value=None
            if si==5:
                self.slider_value=ValueTracker(.95)
                self.slider_dot=Dot([4.13,-1.88+1.36*.95,0],radius=.11,color=YELLOW)
                self.slider_dot.add_updater(lambda m:m.move_to([4.13,-1.88+1.36*self.slider_value.get_value(),0]))
                self.slider_label=VGroup(tex('u=',20,YELLOW),DecimalNumber(.95,num_decimal_places=2,font_size=23,color=YELLOW))
                def place_slider_label(m):
                    m[1].set_value(self.slider_value.get_value())
                    m.arrange(RIGHT,buff=.05).move_to([4.92,-1.88+1.36*self.slider_value.get_value(),0])
                self.slider_label.add_updater(place_slider_label)
                place_slider_label(self.slider_label)
                self.add(self.slider_dot,self.slider_label)
            self.note=None;self.formula=None
            for bi in range(len(story['beats'])):self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml114_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def run_beat(self,si,bi):
        if self.note is not None:self.remove(self.note)
        note=SCENES[si]['beats'][bi]['visual_note']
        if (si,bi)==(1,0):note='曲線下の面積を思い出す'
        self.note=J(note,17,MUTED).move_to([4.18,2.43,0])
        if self.note.width>4.05:self.note.scale(4.05/self.note.width)
        self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        self.formula=tex(FORMULAS[si][bi],25,YELLOW).move_to([4.18,1.58,0])
        if self.formula.width>4.10:self.formula.scale(4.10/self.formula.width)
        self.add(self.formula)
        next_visual=figure(si,bi);next_side=side(si,bi)
        self.remove(self.side)
        self.side=next_side
        self.add(self.side)
        # The graph moves through the whole spoken beat; the numeric label is
        # already correct when the first sentence begins.
        if (si,bi) in ((4,2),(8,4)):
            emphasis=Dot(P(1.4,.15),radius=.095,color=GREEN)
            self.add(emphasis)
            self.beat(Transform(self.visual,next_visual),actions=[dict(name='移動',start=0,end=self.durations[self.beat_index])])
            self.remove(emphasis)
        elif si in (5,6):
            duration=self.durations[self.beat_index]
            height=([.95,.6,.15,.05,.15,.95] if si==5 else [.5,.5,.02,.5,.5,.02])[bi]
            def move():
                main=Transform(self.visual,next_visual)
                if si==5:
                    knob=self.slider_value.animate.set_value(height)
                    return AnimationGroup(main,knob,lag_ratio=0)
                return main
            def trace_highlight():
                lines=[self.visual[6]]
                if si==6 and len(slice_segments(height,True))>1:lines.append(self.visual[7])
                return AnimationGroup(*[ShowPassingFlash(line.copy().set_stroke(width=7),time_width=.22)
                                        for line in lines],lag_ratio=0)
            self.beat(phases=[('高さを動かす',2.4,move),('範囲を強調',duration-2.4,trace_highlight)])
        else:
            self.beat(Transform(self.visual,next_visual),actions=[dict(name='移動',start=0,end=self.durations[self.beat_index])])
