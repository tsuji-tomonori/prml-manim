"""Animated PRML 12.4: curved reconstruction and non-Gaussian independence."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from nonlinear_model import (POINTS, LINEAR, CURVED, MEAN, AXIS, T, curve,
                             SOURCE_1, SOURCE_2, MIXED, LINEAR_MSE, CURVED_MSE)
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST

BG='#111823'; BLUE='#58C4DD'; GREEN='#83C167'; YELLOW='#FFFF00'; PURPLE='#9A72AC'; RED='#FC6255'; MUTED='#A8B3C0'
FORMULAS=[
 [r'D=2,\quad M=1',r'x\to z\to\hat x',r'\hat x_{\rm PCA}=\bar x+uu^T(x-\bar x)',r'{1\over N}\sum_n\|x_n-\hat x_n\|^2\simeq0.289',r'\mathbb E[x\mid z]=Wz+\mu',r'JP:分布か写像を変える'],
 [r'JP:2音源から2観測',r'x=Az',r'\det A\ne0',r'p(z)=\prod_j p(z_j)',r'z=A^{-1}x',r'JP:独立な向きはどこか'],
 [r'\operatorname{cov}(x_1,x_2)=0',r'JP:無相関でも依存する',r'RR^T=I',r'p(z_j)={1\over\pi\cosh z_j}',r'JP:非ガウスな独立成分',r'JP:次は写像を曲げる'],
 [r'x\to z\to y',r'E={1\over2}\sum_n\|y(x_n)-x_n\|^2',r'JP:復習: 5.3 誤差逆伝播',r'M<D',r'JP:中間層だけ非線形',r'JP:前後にも非線形層'],
 [r'x\to F_1\to z\to F_2\to\hat x',r'z=F_1(x),\quad\hat x=F_2(z)',r'f(z)=(z,0.48(z^2-1.08))',r'{1\over N}\sum_n\|x_n-f(z_n)\|^2\simeq0.0104',r'JP:固定した説明用写像',r'JP:曲線をどう学ぶか'],
 [r'JP:主曲線',r'z=g_f(x)',r'JP:投影から平均へ',r'\mathbb E[x\mid g_f(x)=z]=f(z)',r'JP:直線から曲線へ',r'JP:密度は別に必要'],
 [r'JP:確率的PCAの混合',r'z_1,\ldots,z_K',r'JP:生成位相写像',r'p(x)={1\over K}\sum_{k=1}^{K}p(x\mid z_k)',r'JP:図12.21',r'JP:曲線に確率を置く'],
 [r'N=13,\quad M=1',r'JP:直線 0.289',r'JP:曲線 0.0104',r'JP:独立性と非線形写像',r'JP:説明用の固定写像',r'JP:未知の点はどうか'],
]

def main_axes():
    a=Axes(x_range=[-2.2,2.2,1],y_range=[-.8,1.6,.5],x_length=5.75,y_length=4.15,
           tips=False,axis_config={'stroke_color':MUTED,'stroke_width':1.7,'include_numbers':False})
    a.move_to([-3.52,-.05,0])
    labels=VGroup(jp('横座標',15,MUTED).next_to(a.x_axis.get_end(),RIGHT,buff=.10).shift(DOWN*.12),
                  jp('縦座標',15,MUTED).next_to(a.y_axis.get_end(),UP,buff=.08))
    for x in [-2,0,2]:
        labels.add(jp(str(x),13,MUTED).next_to(a.c2p(x,-.8),DOWN,buff=.08))
    for y in [0,1]:
        labels.add(jp(str(y),13,MUTED).next_to(a.c2p(-2.2,y),LEFT,buff=.08))
    return a,labels

def line_from_points(coords,color,width=2.7):
    v=VMobject(stroke_color=color,stroke_width=width)
    v.set_points_as_corners(coords)
    return v

def left_overlay(a,si,bi):
    m=VGroup()
    show_line=(si==0 and bi>=2) or si==3 or (si==5 and bi<2) or (si==7 and bi in (1,))
    show_curve=(si==4 and bi>=2) or (si==5 and bi>=2) or si==6 or (si==7 and bi>=2)
    if show_line:
        ends=[MEAN+s*AXIS for s in [-2.1,2.1]]
        m.add(Line(a.c2p(*ends[0]),a.c2p(*ends[1]),color=PURPLE,stroke_width=3))
        if (si==0 and bi>=3) or (si==3 and bi>=3) or (si==7 and bi==1):
            for i in [0,2,5,7,10,12]:
                m.add(Line(a.c2p(*POINTS[i]),a.c2p(*LINEAR[i]),color=RED,stroke_width=2.5))
    if show_curve:
        coords=curve(np.linspace(-1.86,1.86,80))
        m.add(line_from_points([a.c2p(*xy) for xy in coords],GREEN,3.5))
        if (si==4 and bi>=3) or (si==5 and bi>=2) or (si==7 and bi>=2):
            for i in [0,2,5,7,10,12]:
                m.add(Line(a.c2p(*POINTS[i]),a.c2p(*CURVED[i]),color=RED,stroke_width=2.3))
    if si==5 and bi>=1:
        k=3+bi%7
        m.add(Dot(a.c2p(*POINTS[k]),radius=.105,color=YELLOW),
              Dot(a.c2p(*CURVED[k]),radius=.095,color=GREEN))
    if si==6 and bi>=2:
        for t in np.linspace(-1.65,1.65,9):
            m.add(Dot(a.c2p(*curve(t)),radius=.055,color=YELLOW))
    return m

def plot_points(a,opacity=1):
    return VGroup(*[Dot(a.c2p(*xy),radius=.067,color=BLUE,fill_opacity=opacity) for xy in POINTS])

def panel_label(text,y=1.22):
    return jp(text,18,MUTED).move_to([3.63,y,0])

def wave_panel(stage):
    group=VGroup()
    upper=(SOURCE_1,SOURCE_2) if stage in (0,4) else (MIXED[0],MIXED[1])
    names=('音源 1','音源 2') if stage in (0,4) else ('マイク 1','マイク 2')
    for j,(values,name,color) in enumerate(zip(upper,names,(BLUE,PURPLE))):
        ybase=.50-j*1.58
        x0=1.35; width=4.42; height=.88
        baseline=Line([x0,ybase,0],[x0+width,ybase,0],color=MUTED,stroke_width=1.2)
        v=line_from_points([[x0+i/71*width,ybase+float(value)*height/2.1,0]
                            for i,value in enumerate(values)],color,2.3)
        group.add(baseline,v,jp(name,16,color).move_to([1.75,ybase+.61,0]),
                  jp('0',12,MUTED).move_to([x0,ybase-.18,0]),
                  jp('72',12,MUTED).move_to([x0+width,ybase-.18,0]))
    group.add(jp('時刻',14,MUTED).move_to([6.04,-1.55,0]),jp('振幅',14,MUTED).move_to([1.0,.50,0]))
    if stage>=2:
        group.add(tex(r'A=\begin{pmatrix}1&.55\\ .45&1\end{pmatrix}',23,YELLOW).move_to([3.63,-2.02,0]))
    return group


def tail_panel():
    g=VGroup()
    x0=3.65;y0=-1.52;sx=.60;sy=5.1
    g.add(Line([1.80,y0,0],[5.55,y0,0],color=MUTED,stroke_width=1.4),
          Line([1.80,y0,0],[1.80,.63,0],color=MUTED,stroke_width=1.4),
          jp('密度',14,MUTED).move_to([1.59,.88,0]),
          jp('値',14,MUTED).move_to([5.79,y0+.05,0]))
    for z in [-3,0,3]:
        x=x0+sx*z
        g.add(Line([x,y0-.06,0],[x,y0+.06,0],color=MUTED,stroke_width=1),
              jp(str(z),12,MUTED).move_to([x,y0-.24,0]))
    for y in [0,.2,.4]:
        yy=y0+sy*y
        g.add(Line([1.74,yy,0],[1.86,yy,0],color=MUTED,stroke_width=1),
              jp(f'{y:.1f}',12,MUTED).move_to([1.51,yy,0]))
    z=np.linspace(-3,3,90)
    gaussian=np.exp(-z*z/2)/np.sqrt(2*np.pi)
    heavy=1/(np.pi*np.cosh(z))
    g.add(line_from_points([[x0+sx*t,y0+sy*v,0] for t,v in zip(z,gaussian)],BLUE,2.8),
          line_from_points([[x0+sx*t,y0+sy*v,0] for t,v in zip(z,heavy)],PURPLE,3.0))
    g.add(jp('ガウス',15,BLUE).move_to([2.93,1.03,0]),
          jp('裾が重い',15,PURPLE).move_to([4.42,1.03,0]))
    return g

def dependence_panel(stage):
    g=VGroup()
    x0=3.45; y0=-.36; sx=1.23;sy=.75
    g.add(Line([1.45,y0,0],[5.9,y0,0],color=MUTED,stroke_width=1.4),
          Line([x0,-2.1,0],[x0,1.05,0],color=MUTED,stroke_width=1.4),
          jp('横',14,MUTED).move_to([5.98,y0-.18,0]),jp('縦',14,MUTED).move_to([x0-.2,1.15,0]))
    for v in [-1,0,1]:
        x=x0+sx*v
        g.add(Line([x,y0-.06,0],[x,y0+.06,0],color=MUTED,stroke_width=1),
              jp(str(v),12,MUTED).move_to([x,y0-.24,0]))
    for v in [0,1]:
        y=y0+sy*v
        g.add(Line([x0-.06,y,0],[x0+.06,y,0],color=MUTED,stroke_width=1),
              jp(str(v),12,MUTED).move_to([x0-.23,y,0]))
    xs=np.linspace(-1.5,1.5,9); ys=xs*xs-np.mean(xs*xs)
    for x,y in zip(xs,ys): g.add(Dot([x0+sx*x,y0+sy*y,0],radius=.06,color=BLUE))
    if stage>=1:
        g.add(Line([x0-1.55,y0+.14,0],[x0+1.55,y0+.14,0],color=PURPLE,stroke_width=2.2))
    if stage>=2:
        g.add(jp('平均の傾き 0',17,YELLOW).move_to([3.65,-1.77,0]))
    g.add(jp('説明用の対称な9点',14,MUTED).move_to([4.6,-2.22,0]))
    return g

def network_panel(stage):
    g=VGroup()
    columns=[[(1.52,.55),(1.52,-.56)],[(2.68,.76),(2.68,0),(2.68,-.76)],[(3.72,0)],[(4.77,.76),(4.77,0),(4.77,-.76)],[(5.82,.55),(5.82,-.56)]]
    if stage<5: columns=[columns[0],columns[2],columns[-1]]
    for c1,c2 in zip(columns,columns[1:]):
        for x,y in c1:
            for xx,yy in c2:g.add(Line([x,y,0],[xx,yy,0],color='#607286',stroke_width=1.5))
    for k,col in enumerate(columns):
        color=BLUE if k==0 else GREEN if k==len(columns)-1 else YELLOW if k==len(columns)//2 else PURPLE
        for x,y in col:g.add(Circle(radius=.095,color=color,stroke_width=2.3).move_to([x,y,0]))
    g.add(jp('入力',16,BLUE).move_to([1.52,-1.43,0]),jp('潜在',16,YELLOW).move_to([3.72,-1.43,0]),
          jp('再構成',16,GREEN).move_to([5.80,-1.43,0]))
    if stage==2:
        g.add(Arrow([5.70,-1.02,0],[1.75,-1.02,0],buff=.10,color=YELLOW,stroke_width=2.5))
    elif stage>=5:g.add(jp('前後の層で曲げる',16,YELLOW).move_to([3.65,-1.93,0]))
    return g

def latent_panel(stage):
    g=VGroup()
    g.add(Line([1.32,-1.22,0],[5.9,-1.22,0],color=MUTED,stroke_width=1.6),
          jp('潜在変数',16,MUTED).move_to([3.65,-2.11,0]))
    for t in [-1.8,-.9,0,.9,1.8]:
        x=3.65+t*1.17
        g.add(Line([x,-1.14,0],[x,-1.30,0],color=MUTED),
              jp(f'{t:.1f}',13,MUTED).move_to([x,-1.49,0]))
    if stage>=1:
        k=1+(stage*2)%11
        x=3.65+float(POINTS[k,0])*1.17
        g.add(Dot([x,-1.22,0],radius=.12,color=YELLOW),
              jp('曲線へ戻す' if stage>=3 else '点の横座標',16,YELLOW).move_to([3.65,.20,0]),
              Arrow([x,.04,0],[x,-.96,0],buff=.08,color=YELLOW,stroke_width=2.5))
    return g

def principal_panel(stage):
    g=latent_panel(0)
    x=3.65+(.28*max(0,stage-1))
    g.add(Dot([x,-1.22,0],radius=.11,color=YELLOW),
          Arrow([x,-.30,0],[x,-1.02,0],buff=.07,color=YELLOW,stroke_width=2.4))
    if stage>=2:
        g.add(jp('近くの点を平均',17,YELLOW).move_to([3.65,.77,0]))
    if stage>=3:
        g.add(jp('投影 → 平均 → 更新',17,GREEN).move_to([3.65,.19,0]))
    return g

def gtm_panel(stage):
    g=VGroup()
    g.add(panel_label('説明用の潜在格子9点',1.05))
    for k,t in enumerate(np.linspace(-1.65,1.65,9)):
        x=1.55+k*.53
        g.add(Dot([x,-.10,0],radius=.055,color=YELLOW))
        if stage>=1:
            xy=curve(t)
            yy=-1.50+.43*float(xy[1])
            g.add(Dot([x,yy,0],radius=.06,color=GREEN),
                  Line([x,-.10,0],[x,yy+.10,0],color='#607286',stroke_width=1.2))
        if stage>=2:
            g.add(Circle(radius=.13,color=BLUE,stroke_opacity=.48,stroke_width=1).move_to([x,yy,0]))
    for k,value in [(0,'-1.65'),(4,'0'),(8,'1.65')]:
        g.add(jp(value,12,MUTED).move_to([1.55+k*.53,.17,0]))
    g.add(jp('潜在空間',15,YELLOW).move_to([1.97,.56,0]))
    if stage>=1:g.add(jp('観測空間',15,GREEN).move_to([1.97,-1.93,0]))
    if stage>=3:g.add(jp('密度を平均',18,GREEN).move_to([4.57,-2.06,0]))
    return g

def score_panel(stage):
    g=VGroup()
    g.add(panel_label('平均二乗再構成距離',1.11))
    origin=np.array([1.55,-1.7,0]); width=3.95
    g.add(Line(origin,origin+RIGHT*width,color=MUTED,stroke_width=1.5))
    for t,v in enumerate([0,.1,.2,.3]):
        x=origin[0]+v/.3*width
        g.add(Line([x,-1.63,0],[x,-1.78,0],color=MUTED),
              jp(f'{v:.1f}',13,MUTED).move_to([x,-2.02,0]))
    if stage>=1:
        g.add(jp('PCA',17,PURPLE).move_to([1.23,.30,0]),
              Line([origin[0],.30,0],[origin[0]+LINEAR_MSE/.3*width,.30,0],color=PURPLE,stroke_width=9),
              jp('0.289',16,PURPLE).move_to([5.82,.30,0]))
    if stage>=2:
        g.add(jp('曲線',17,GREEN).move_to([1.23,-.55,0]),
              Line([origin[0],-.55,0],[origin[0]+CURVED_MSE/.3*width,-.55,0],color=GREEN,stroke_width=9),
              jp('0.0104',16,GREEN).move_to([2.30,-.55,0]))
    return g

def panel(si,bi):
    if si==0:return score_panel(0 if bi<3 else 1)
    if si==1:return wave_panel(0 if bi==0 else 1 if bi<4 else 4)
    if si==2:return tail_panel() if bi in (3,4) else dependence_panel(bi)
    if si==3:return network_panel(bi)
    if si==4:return latent_panel(bi)
    if si==5:return principal_panel(bi)
    if si==6:return gtm_panel(bi)
    return score_panel(0 if bi<1 else 1 if bi<2 else 2)

class PRML124NonlinearLatentVariables(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            axes,labels=main_axes()
            self.add(axes,labels,plot_points(axes, .42 if si in (1,2) else 1),
                     jp('説明用の13点',15,MUTED).move_to([-5.50,2.28,0]))
            old=None
            for bi,beat in enumerate(story['beats']):
                formula=(jp(FORMULAS[si][bi][3:],20,YELLOW) if FORMULAS[si][bi].startswith('JP:') else tex(FORMULAS[si][bi],25,YELLOW)).move_to([3.70,2.42,0])
                if formula.width>4.95:formula.scale(4.95/formula.width)
                note=jp(beat['visual_note'],16,MUTED).move_to([3.70,1.81,0])
                if note.width>4.9:note.scale(4.9/note.width)
                state=VGroup(formula,note,panel(si,bi),left_overlay(axes,si,bi))
                anim=FadeIn(state) if old is None else AnimationGroup(FadeOut(old),FadeIn(state),lag_ratio=0)
                self.beat(anim,end_sentence=1,actions=[dict(name='図を更新',start=0,end=self.sentence_duration(0))])
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml124_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
