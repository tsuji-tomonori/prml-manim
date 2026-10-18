"""PRML 14.5: conditional mixtures on one persistent 20-point example."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from manim import *
from narration_content import SCENES
from scene_support import NarratedScene,jp,tex
from make_voicevox_narration import MANIFEST
from mixture_model import X,T,QUERY,fit_mixture,fit_gate,lower_gate,normal_pdf

BG='#111823';BLUE='#58C4DD';YELLOW='#FFFF00';GREEN='#83C167'
PURPLE='#9A72AC';PURPLE_BRIGHT='#C9A5E4';RED='#FC6255';MUTED='#A8B3C0'
HISTORY,RESPONSIBILITY=fit_mixture()
FIT=HISTORY[-1]
GATE=fit_gate(RESPONSIBILITY)


def lab(s,x,y,size=18,color=WHITE):
    return jp(s,size,color).move_to([x,y,0])


def math(s,x,y,size=25,color=YELLOW,max_width=12.0):
    m=tex(s,size,color).move_to([x,y,0])
    if m.width>max_width:m.scale(max_width/m.width)
    return m


def xx(v):return -5.40+2.52*(v+1)
def yy(v):return -1.76+1.16*(v+1.5)


def x_label(s,x,y,size=13):return lab(s,x,y,size,MUTED)


def model_stage(si,bi):
    if si==3:
        return HISTORY[[0,0,1,3,8,20,20][bi]]
    return FIT


def query_x(si,bi):
    if si==4:return [-.75,-.2,.75,.75,.75,.75,.75][bi]
    if si==5:return [-.75,-.4,-.1,.25,.5,.75,.75][bi]
    if si==1:return [-.7,-.5,-.25,0,.2,.45,.75][bi]
    return QUERY


def plot(si,bi,state):
    g=VGroup()
    axis=VGroup(
        Line([xx(-1),yy(-1.5),0],[xx(1.02),yy(-1.5),0],color=MUTED,stroke_width=1.5),
        Line([xx(-1),yy(-1.5),0],[xx(-1),yy(1.55),0],color=MUTED,stroke_width=1.5))
    g.add(axis)
    for v in (-1,0,1):
        g.add(Line([xx(v),yy(-1.5)-.06,0],[xx(v),yy(-1.5)+.06,0],color=MUTED,stroke_width=1),
              x_label(str(v),xx(v),yy(-1.5)-.23))
        g.add(Line([xx(-1)-.06,yy(v),0],[xx(-1)+.06,yy(v),0],color=MUTED,stroke_width=1),
              x_label(str(v),xx(-1)-.26,yy(v)))
    g.add(lab('入力',-4.33,-2.19,16,MUTED),math('x',-3.80,-2.19,19,MUTED),
          lab('観測',-5.05,2.18,16,MUTED),math('t',-4.60,2.18,19,MUTED))
    # Both line segments stay inside the plotting axes.
    line_colors=(BLUE,RED)
    lines=[]
    for k in range(2):
        w=state['weights'][k]
        line=Line([xx(-1),yy(w[0]-w[1]),0],[xx(1),yy(w[0]+w[1]),0],
                  color=line_colors[k],stroke_width=3.3)
        lines.append(line);g.add(line)
    dots=[]
    for i,(x,t) in enumerate(zip(X,T)):
        if si==2 and bi>=3:
            col=interpolate_color(ManimColor(BLUE),ManimColor(RED),float(RESPONSIBILITY[i,1]))
        else:col=GREEN
        dot=Dot([xx(float(x)),yy(float(t)),0],radius=.071,color=col)
        dots.append(dot);g.add(dot)
    xq=query_x(si,bi)
    qline=DashedLine([xx(xq),yy(-1.48),0],[xx(xq),yy(1.48),0],dash_length=.10,
                     color=YELLOW,stroke_width=1.4)
    qdot=Dot([xx(xq),yy(-1.5),0],radius=.083,color=YELLOW)
    g.add(qline,qdot)
    if si==0 and bi<3:
        qline.set_opacity(0);qdot.set_opacity(0)
    if si in (0,4,5,8) and not (si==0 and bi<3):
        g.add(lab(f'x = {xq:.2f}',xx(xq),2.17,17,YELLOW))
    else:g.add(VGroup())
    return g,lines,qdot,dots


def density_panel(x,weights,lower_weight=.35,compare=False):
    """Horizontal density, vertical target: literal Gaussian widths and areas."""
    g=VGroup();left=1.57;bottom=-1.72;width=3.68;top=1.76
    tx=lambda t:bottom+(t+1.5)*(top-bottom)/3
    dx=lambda d:left+width*d/4.0
    axes=VGroup(Line([left,bottom,0],[left+width,bottom,0],color=MUTED,stroke_width=1.4),
                Line([left,bottom,0],[left,top,0],color=MUTED,stroke_width=1.4))
    g.add(axes)
    for v in (0,1,2,3,4):
        g.add(Line([dx(v),bottom-.05,0],[dx(v),bottom+.05,0],color=MUTED,stroke_width=1),
              x_label(str(v),dx(v),bottom-.22,12))
    for v in (-1,0,1):
        g.add(Line([left-.05,tx(v),0],[left+.05,tx(v),0],color=MUTED,stroke_width=1),
              x_label(str(v),left-.25,tx(v),12))
    g.add(lab('密度',4.83,-2.18,15,MUTED),math('t',1.30,2.10,18,MUTED),
          lab(f'入力 x = {x:.2f}',3.32,2.12,17,YELLOW))
    means=weights[:,0]+weights[:,1]*x
    values=np.linspace(-1.48,1.48,360)
    for k,weight in enumerate((1-lower_weight,lower_weight)):
        curve=VMobject(stroke_color=(BLUE,RED)[k],stroke_width=3.0)
        curve.set_points_smoothly([[dx(float(weight*normal_pdf(t,means[k],FIT['precision']))),tx(float(t)),0] for t in values])
        g.add(curve)
    if compare:
        old=VMobject(stroke_color=PURPLE_BRIGHT,stroke_width=2.5)
        old.set_points_smoothly([[dx(float(.35*normal_pdf(t,means[1],FIT['precision']))),tx(float(t)),0] for t in values])
        g.add(old)
        g.add(lab('固定 35%',5.85,-.18,15,PURPLE_BRIGHT),lab('ゲート 7.7%',5.85,-.50,15,RED))
        focus=g[-3]
    else:
        g.add(lab(f'青 {100*(1-lower_weight):.1f}%',5.84,1.59,15,BLUE),
              lab(f'赤 {100*lower_weight:.1f}%',5.84,1.29,15,RED))
        focus=g[-2]
    return g,focus



def prediction_panel():
    g=VGroup(lab('どちらへ何%？',3.48,1.81,21,YELLOW))
    for i,(name,color) in enumerate((('上の線',BLUE),('下の線',RED))):
        y=.71-i*1.07
        g.add(lab(name,2.16,y,18,color),
              Line([3.07,y,0],[5.33,y,0],color=MUTED,stroke_width=1.5),
              lab('？',5.58,y,25,color))
    return g,g[-1]

def weights_panel(upper=.65,lower=.35,caption='混合係数'):
    g=VGroup(lab(caption,3.27,1.85,19,YELLOW))
    for i,(name,weight,color) in enumerate((('青',upper,BLUE),('赤',lower,RED))):
        y=.87-i*1.00
        g.add(lab(name,1.63,y,18,color),
              Line([2.0,y,0],[5.45,y,0],color=MUTED,stroke_width=1.3),
              Line([2.0,y,0],[2.0+3.45*weight,y,0],color=color,stroke_width=9),
              lab(f'{100*weight:.1f}%',5.34,y+.26,16,color))
    g.add(lab('青 + 赤 = 100%',3.53,-1.33,17,MUTED))
    return g,g[3]


def responsibility_panel(si,bi):
    if bi<3:
        return weights_panel(.5,.5,'担当はまだ不明')
    i=0 if bi in (3,6) else 13
    up=float(RESPONSIBILITY[i,0]);lo=float(RESPONSIBILITY[i,1])
    g,focus=weights_panel(up,lo,'この点の負担率')
    g.add(lab('上の点' if i==0 else '下の点',3.3,-1.85,17,YELLOW))
    return g,focus



def weighted_error_panel():
    """Small conceptual card for the weighted squared residual."""
    g=VGroup(lab('ずれに重りを掛ける',3.45,1.93,19,YELLOW))
    y0=-.75;y1=.66;x=3.31
    g.add(Line([1.82,y0,0],[5.33,y0,0],color=BLUE,stroke_width=2.4),
          Dot([x,y1,0],radius=.105,color=GREEN),
          DashedLine([x,y0,0],[x,y1,0],dash_length=.13,color=YELLOW,stroke_width=2.5),
          lab('観測',4.11,y1,17,GREEN),lab('予測',4.57,y0-.32,17,BLUE))
    sq=Square(side_length=.92,color=YELLOW,stroke_width=2.2).move_to([2.25,.58,0])
    g.add(sq,lab('ずれの二乗',2.25,-.08,16,YELLOW),
          lab('負担率で面積を調整',3.52,-1.63,17,BLUE),
          lab('説明図',5.31,1.93,14,MUTED))
    return g,sq

def em_panel(bi):
    stage=[0,0,1,3,8,20,20][bi]
    g=VGroup(lab('E → M を繰り返す',3.35,1.86,19,YELLOW))
    g.add(Line([1.8,1.0,0],[5.5,1.0,0],color=MUTED,stroke_width=1.5),
          Dot([1.8+3.7*stage/20,1.0,0],radius=.13,color=YELLOW),
          lab('0',1.8,.75,13,MUTED),lab('20',5.5,.75,13,MUTED),
          lab(f'{stage} 回',3.48,.75,18,YELLOW))
    for k,(label,color) in enumerate((('青の線',BLUE),('赤の線',RED))):
        g.add(lab(label,2.25,-.13-.75*k,18,color),
              lab(f'切片 {HISTORY[stage]["weights"][k,0]:+.2f}',4.63,-.13-.75*k,17,color))
    return g,g[2]


def gate_panel(x,blend=1.,footer=True):
    g=VGroup();left=1.60;bottom=-1.62;width=3.85;height=3.30
    gx=lambda v:left+width*(v+1)/2
    gy=lambda v:bottom+height*v
    g.add(Line([left,bottom,0],[left+width,bottom,0],color=MUTED,stroke_width=1.4),
          Line([left,bottom,0],[left,bottom+height,0],color=MUTED,stroke_width=1.4))
    for v in (-1,0,1):
        g.add(Line([gx(v),bottom-.05,0],[gx(v),bottom+.05,0],color=MUTED,stroke_width=1),
              x_label(str(v),gx(v),bottom-.21,12))
    for v in (0,.5,1):
        g.add(Line([left-.05,gy(v),0],[left+.05,gy(v),0],color=MUTED,stroke_width=1),
              x_label(f'{v:g}',left-.24,gy(v),12))
    g.add(lab('入力 x',4.92,-2.12,15,MUTED),lab('赤の係数',2.28,2.05,16,RED))
    xs=np.linspace(-1,1,100)
    fixed=VMobject(stroke_color=PURPLE,stroke_width=2.2)
    fixed.set_points_as_corners([[gx(v),gy(.35),0] for v in xs])
    g.add(fixed)
    ys=(1-blend)*.35+blend*lower_gate(xs,GATE)
    curve=VMobject(stroke_color=RED,stroke_width=3.0)
    curve.set_points_smoothly([[gx(v),gy(float(y)),0] for v,y in zip(xs,ys)])
    g.add(curve)
    yq=float((1-blend)*.35+blend*lower_gate(x,GATE))
    marker=Dot([gx(x),gy(yq),0],radius=.105,color=YELLOW)
    g.add(marker)
    if footer:g.add(lab(f'固定 35%  →  {yq*100:.1f}%',3.65,-2.50,17,YELLOW))
    return g,marker


def logistic_panel(bi):
    g=VGroup();left=1.58;bottom=-1.60;width=3.85;height=3.22
    gx=lambda v:left+width*(v+1)/2
    gy=lambda v:bottom+height*v
    g.add(Line([left,bottom,0],[left+width,bottom,0],color=MUTED,stroke_width=1.4),
          Line([left,bottom,0],[left,bottom+height,0],color=MUTED,stroke_width=1.4))
    for v in (-1,0,1):
        g.add(Line([gx(v),bottom-.05,0],[gx(v),bottom+.05,0],color=MUTED,stroke_width=1),
              x_label(str(v),gx(v),bottom-.21,12))
    for v in (0,.5,1):
        g.add(Line([left-.05,gy(v),0],[left+.05,gy(v),0],color=MUTED,stroke_width=1),
              x_label(f'{v:g}',left-.24,gy(v),12))
    g.add(lab('入力 x',4.91,-2.12,15,MUTED),lab('クラス1の確率',2.50,2.03,16,YELLOW),
          lab('分類の説明用',4.55,2.05,14,MUTED))
    xs=np.linspace(-1,1,100)
    y1=expit(6*(xs+.25));y2=expit(-6*(xs-.25));ym=.5*(y1+y2)
    focus=None
    for j,(ys,col) in enumerate(((y1,BLUE),(y2,RED),(ym,GREEN))):
        curve=VMobject(stroke_color=col,stroke_width=3.0 if j==2 else 2.2)
        curve.set_points_smoothly([[gx(v),gy(float(y)),0] for v,y in zip(xs,ys)])
        if bi<2 and j>0:curve.set_opacity(0)
        elif bi==2 and j==2:curve.set_opacity(0)
        g.add(curve);focus=curve if (bi>=3 and j==2) or (bi<3 and j==0) else focus
    g.add(lab('混合',4.72,1.13,15,GREEN) if bi>=3 else VGroup())
    return g,focus


def tree_panel(bi):
    g=VGroup(lab('確率で枝を選ぶ',3.43,1.99,18,YELLOW))
    root=Dot([3.48,1.37,0],radius=.11,color=YELLOW);g.add(root)
    left=Dot([2.27,.19,0],radius=.1,color=BLUE);right=Dot([4.64,.19,0],radius=.1,color=RED)
    g.add(Line(root.get_center(),left.get_center(),color=BLUE,stroke_width=2.8),
          Line(root.get_center(),right.get_center(),color=RED,stroke_width=2.8),left,right)
    g.add(lab('70%',2.47,1.03,16,BLUE),lab('30%',4.49,1.03,16,RED))
    if bi>=2:
        leaf2=Dot([3.90,-1.10,0],radius=.09,color=GREEN);leaf3=Dot([5.35,-1.10,0],radius=.09,color=PURPLE_BRIGHT)
        g.add(Line(right.get_center(),leaf2.get_center(),color=GREEN,stroke_width=2.4),
              Line(right.get_center(),leaf3.get_center(),color=PURPLE_BRIGHT,stroke_width=2.4),leaf2,leaf3)
        g.add(lab('60%',4.08,-.54,15,GREEN),lab('40%',5.17,-.54,15,PURPLE_BRIGHT),
              lab('葉1 70%',2.12,-1.55,15,BLUE),lab('葉2 18%',3.89,-1.55,15,GREEN),
              lab('葉3 12%',5.45,-1.55,15,PURPLE_BRIGHT))
    else:g.add(lab('左の葉',2.25,-.63,16,BLUE),lab('右の葉',4.65,-.63,16,RED))
    if bi!=2:g.add(lab('入力で枝の割合が変わる',3.56,-2.43,16,YELLOW))
    return g,root


def right_panel(si,bi,state):
    if si in (0,):
        if bi>=4:return density_panel(QUERY,state['weights'],.35)
        return prediction_panel()
    if si==1:return density_panel(query_x(si,bi),state['weights'],.35)
    if si==2:return responsibility_panel(si,bi)
    if si==3:return weighted_error_panel() if bi==1 else em_panel(bi)
    if si==4:return density_panel(query_x(si,bi),state['weights'],.35)
    if si==5:
        blend=[0,.12,.25,.5,.75,1,1][bi]
        return gate_panel(query_x(si,bi),blend,footer=bi!=2)
    if si==6:return logistic_panel(bi)
    if si==7:return tree_panel(bi)
    if bi<2:return density_panel(QUERY,state['weights'],.35)
    return density_panel(QUERY,state['weights'],float(lower_gate(QUERY,GATE)),compare=True)

FORMULAS={
    (1,4):r'p(t\mid x)=\pi_1\mathcal N_1+\pi_2\mathcal N_2',
    (2,3):r'\gamma_{nk}=\frac{\pi_k\mathcal N_k}{\sum_j\pi_j\mathcal N_j}',
    (3,3):r'\min_{w_k}\sum_n\gamma_{nk}(t_n-w_k^\top\phi_n)^2',
    (3,4):r'\pi_k=\frac1N\sum_n\gamma_{nk}',
    (5,2):r'p(t\mid x)=\sum_k\pi_k(x)p_k(t\mid x)',
    (6,3):r'p(t\mid\phi)=\sum_k\pi_k y_k^t(1-y_k)^{1-t}',
    (7,2):r'0.30\times0.60=0.18,\qquad0.30\times0.40=0.12',
    (8,2):r'\pi_2:0.35\longrightarrow0.077',
}


def make_state(si,bi):
    state=model_stage(si,bi)
    note=SCENES[si]['beats'][bi]['visual_note']
    heading=lab(note,0,2.61,18,YELLOW)
    if note.startswith('復習:'):
        heading.add(SurroundingRectangle(heading,color=YELLOW,stroke_width=1,buff=.08))
    graph,lines,qdot,dots=plot(si,bi,state)
    side,focus=right_panel(si,bi,state)
    formula=math(FORMULAS[(si,bi)],0,-2.48,30,YELLOW,11.4) if (si,bi) in FORMULAS else VGroup()
    # The caption area is below y=-2.95; equations stay above it.
    whole=VGroup(heading,graph,side,formula)
    if si==3 and bi in (1,2,3):focus=lines[0] if bi<3 else lines[1]
    if si in (0,4,8) and bi in (2,3,4):focus=qdot
    return whole,focus


class PRML145ConditionalMixtures(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            old=None
            for bi,_ in enumerate(story['beats']):
                state,focus=make_state(si,bi)
                duration=float(self.boundaries[bi]-(self.boundaries[bi-1] if bi else 0))
                entrance=min(3.0,duration*.40)
                def second(a=state,f=focus):
                    self.clear();self.add(self.main_title,a,self.subtitle)
                    return Succession(Indicate(f,color=f.get_color(),scale_factor=1.035),
                                      Indicate(f,color=f.get_color(),scale_factor=1.025))
                if old is None:
                    self.beat(phases=[('図を導入',entrance,lambda a=state:FadeIn(a)),
                                      ('対応を追う',duration-entrance,second)])
                else:
                    def update(a=old,b=state):
                        return AnimationGroup(
                            Transform(a[1],b[1]),
                            FadeOut(a[0]),FadeIn(b[0]),
                            FadeOut(a[2]),FadeIn(b[2]),
                            FadeOut(a[3]),FadeIn(b[3]),lag_ratio=0)
                    self.beat(phases=[('図を更新',entrance,update),
                                      ('対応を追う',duration-entrance,second)])
                self.clear();self.add(self.main_title,state,self.subtitle)
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml145_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
