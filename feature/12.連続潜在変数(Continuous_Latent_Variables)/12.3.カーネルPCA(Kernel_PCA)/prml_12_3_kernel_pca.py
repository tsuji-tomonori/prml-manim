"""PRML 12.3 — kernel PCA of four diagonal groups."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from kernel_pca_model import (POINTS,LABELS,N,LINEAR_EIGENVALUES,FEATURE_EIGENVALUES,
                              GRAM,CENTERED_GRAM,GRAM_EIGENVALUES,SCORES,NEW_POINT,NEW_SCORE,
                              PREIMAGE_POINT,PREIMAGE_LHS,PREIMAGE_RHS)
from scene_support import NarratedScene,jp,tex
from make_voicevox_narration import MANIFEST

BG='#101923'; BLUE='#58C4DD'; YELLOW='#FFFF00'; GREEN='#83C167'; PURPLE='#9A72AC'; RED='#FC6255'; MUTED='#A6B6C5'; WHITE='#F4F7FA'
O=np.array([-4.05,0.05,0]); S=1.38

def P(x): return O+S*np.array([float(x[0]),float(x[1]),0.])

def main_axes():
    g=VGroup(Line(P([-1.66,0]),P([1.66,0]),color=MUTED,stroke_width=1.8),
             Line(P([0,-1.48]),P([0,1.48]),color=MUTED,stroke_width=1.8))
    for x in (-1,1):
        q=P([x,0]);g.add(Line(q+UP*.048,q+DOWN*.048,color=MUTED,stroke_width=1.3),
                        jp(str(x),14,MUTED).move_to(q+DOWN*.22))
    for y in (-1,1):
        q=P([0,y]);g.add(Line(q+LEFT*.048,q+RIGHT*.048,color=MUTED,stroke_width=1.3),
                        jp(str(y),14,MUTED).move_to(q+LEFT*.22))
    g.add(tex('x_1',22,MUTED).move_to(P([1.73,-.14])),
          tex('x_2',22,MUTED).move_to(P([-.17,1.62])))
    return g

def points():
    return VGroup(*[Dot(P(x),radius=.064,color=GREEN if c>0 else PURPLE)
                    for x,c in zip(POINTS,LABELS)])

def side_axes(xlabel,ylabel,xrange,yrange,xticks,yticks):
    x0,x1,y0,y1=1.63,6.17,-1.78,1.05
    def q(x,y): return np.array([x0+(x-xrange[0])/(xrange[1]-xrange[0])*(x1-x0),
                                y0+(y-yrange[0])/(yrange[1]-yrange[0])*(y1-y0),0.])
    g=VGroup(Line(q(xrange[0],yrange[0]),q(xrange[1],yrange[0]),color=MUTED,stroke_width=1.5),
             Line(q(xrange[0],yrange[0]),q(xrange[0],yrange[1]),color=MUTED,stroke_width=1.5))
    for x,label in xticks:
        v=q(x,yrange[0]);g.add(Line(v+UP*.045,v+DOWN*.045,color=MUTED,stroke_width=1),
                              jp(label,13,MUTED).move_to(v+DOWN*.19))
    for y,label in yticks:
        if abs(y-yrange[0])<1e-9:continue
        v=q(xrange[0],y);g.add(Line(v+LEFT*.045,v+RIGHT*.045,color=MUTED,stroke_width=1),
                               jp(label,13,MUTED).move_to(v+LEFT*.25))
    g.add(jp(xlabel,15,MUTED).move_to([5.25,-2.61,0]),
          jp(ylabel,15,MUTED).move_to([2.0,1.30,0]))
    return g,q

def score_panel(mode='linear', reveal=True, new=False):
    title={'linear':'横座標','product':'座標の積','kernel':'第1主成分の得点'}[mode]
    left,right,y=1.65,6.15,-.62
    def q(value,raise_by=0):return np.array([left+(float(value)+2.25)/4.5*(right-left),y+raise_by,0.])
    g=VGroup(Line([left,y,0],[right,y,0],color=MUTED,stroke_width=1.8),
             jp(title,16,MUTED).move_to([4.15,-1.27,0]))
    for value in (-2,-1,0,1,2):
        v=q(value)
        g.add(Line(v+UP*.055,v+DOWN*.055,color=MUTED,stroke_width=1),
              jp(str(value),14,MUTED).move_to(v+DOWN*.22))
    if not reveal:
        for value in (-1.2,1.2):g.add(jp('?',22,YELLOW).move_to(q(value,.4)))
        return g
    if mode=='linear':values=POINTS[:,0]
    elif mode=='product':values=POINTS[:,0]*POINTS[:,1]
    else:values=SCORES
    for i,(value,label) in enumerate(zip(values,LABELS)):
        vertical=.24+.17*(i%4)+(.09 if label>0 else -.09)
        g.add(Dot(q(value,vertical),radius=.062,color=GREEN if label>0 else PURPLE))
    if new:
        g.add(Dot(q(NEW_SCORE,.92),radius=.10,color=YELLOW),
              jp('新点: 約1.018',16,YELLOW).move_to([4.90,.80,0]))
    g.add(jp('縦のずらしは表示用',14,MUTED).move_to([4.15,-1.76,0]))
    return g

def variance_panel(kind='feature',amount=3):
    if kind=='feature':
        vals=FEATURE_EIGENVALUES[[1,0,2]] # x1², sqrt(2)xy, x2²
        labels=(r'x_1^2',r'\sqrt2 x_1x_2',r'x_2^2'); maximum=3.2;yticks=[(0,'0'),(1,'1'),(2,'2'),(3,'3')]
        ylabel='分散'
    elif kind=='linear':
        vals=LINEAR_EIGENVALUES; labels=(r'x_1',r'x_2');maximum=1.7;yticks=[(0,'0'),(1,'1')];ylabel='分散'
    else:
        vals=GRAM_EIGENVALUES[:3];labels=(r'1',r'2',r'3');maximum=50;yticks=[(0,'0'),(20,'20'),(40,'40')];ylabel='固有値'
    g,q=side_axes('特徴の方向' if kind=='feature' else '方向の番号',ylabel,(-.5,len(vals)-.5),(0,maximum),
                  ([] if kind=='feature' else [(i,str(i+1)) for i in range(len(vals))]),yticks)
    colors=(MUTED,GREEN,MUTED) if kind=='feature' else (GREEN,PURPLE,MUTED)
    for i,(v,c) in enumerate(zip(vals,colors)):
        if i>=amount:break
        b=q(i,0);t=q(i,v)
        g.add(Rectangle(width=.50,height=max(.025,t[1]-b[1]),fill_color=c,fill_opacity=.83,stroke_width=0).move_to((b+t)/2),
              jp(f'{v:.3f}',15,c).move_to(t+UP*.18))
        if kind=='feature':g.add(tex(labels[i],21,c).move_to([q(i,0)[0],-2.20,0]))
    return g

def sign_panel(stage):
    g=VGroup()
    for sx in (-1,1):
        for sy in (-1,1):
            x=3.15+1.24*sx;y=-.38+.73*sy
            color=GREEN if sx*sy>0 else PURPLE
            g.add(Dot([x,y,0],radius=.15,color=color),
                  tex('+' if sx*sy>0 else '-',23,color).move_to([x+.32,y,0]))
    g.add(jp('座標の符号',17,MUTED).move_to([4.45,1.09,0]),
          tex(r'x_1x_2',28,YELLOW).move_to([4.45,-1.75,0]))
    if stage>=1:g.add(Arrow([4.2,.50,0],[4.2,-1.18,0],buff=.14,color=YELLOW,stroke_width=3))
    return g

def recap_panel():
    # The narrow yellow outline is the shared recap-card convention; there is no graph frame.
    g=VGroup(Rectangle(width=4.46,height=2.67,stroke_color=YELLOW,stroke_width=1,fill_opacity=0).move_to([4.15,-.30,0]),
             jp('復習: 6.1 特徴と内積',18,WHITE).move_to([4.15,.78,0]),
             jp('入力',17,BLUE).move_to([2.62,.06,0]),
             tex(r'x,z',25,BLUE).move_to([2.64,-.47,0]),
             Arrow([3.07,-.45,0],[3.72,-.45,0],buff=.03,color=YELLOW,stroke_width=3),
             jp('特徴',17,GREEN).move_to([4.39,.06,0]),
             tex(r'\phi(x),\phi(z)',22,GREEN).move_to([4.55,-.47,0]),
             jp('対応成分を掛けて足す',16,YELLOW).move_to([4.15,-1.35,0]))
    return g

def comparison_panel(stage):
    g=VGroup()
    g.add(Dot([2.20,.32,0],radius=.14,color=BLUE),Dot([5.72,.32,0],radius=.14,color=PURPLE),
          tex(r'x=(1.2,1)',22,BLUE).move_to([2.95,.88,0]),
          tex(r'z=(-1.2,1)',22,PURPLE).move_to([5.04,.88,0]))
    if stage>=1:
        for j,(lab,val,col) in enumerate(((r'x_1z_1',r'-1.44',RED),(r'x_2z_2',r'1',BLUE))):
            y=-.50-j*.60
            g.add(tex(lab,23,col).move_to([2.77,y,0]),tex(val,23,col).move_to([4.38,y,0]))
        g.add(Line([4.00,-.19,0],[4.00,-1.43,0],color=MUTED,stroke_width=1))
    if stage>=2:g.add(tex(r'x^Tz=-0.44',24,YELLOW).move_to([4.38,-1.78,0]))
    return g

def heatmap(centered=False,selected=False):
    matrix=CENTERED_GRAM if centered else GRAM
    g=VGroup()
    left=1.93;top=1.10;step=.245
    scale=max(float(np.max(np.abs(matrix))),1)
    for i in range(N):
        for j in range(N):
            value=float(matrix[i,j])/scale
            color=(GREEN if value>=0 else PURPLE) if centered else BLUE
            opacity=.22+.68*abs(value)
            g.add(Square(side_length=step*.98,fill_color=color,fill_opacity=opacity,
                         stroke_color=BG,stroke_width=.35).move_to([left+step*j,top-step*i,0]))
    if selected:
        g.add(Square(side_length=step*.99,stroke_color=YELLOW,stroke_width=2.7,fill_opacity=0).move_to([left,top,0]))
    for t in (1,8,16):
        g.add(jp(str(t),13,MUTED).move_to([left+step*(t-1),top+.31,0]),
              jp(str(t),13,MUTED).move_to([left-.30,top-step*(t-1),0]))
    g.add(tex('m',19,MUTED).move_to([6.20,top+.32,0]),
          tex('n',19,MUTED).move_to([left-.32,-2.05,0]))
    return g

def dimension_panel():
    g=VGroup(jp('通常の主成分分析',16,BLUE).move_to([2.74,1.05,0]),
             jp('カーネル主成分分析',16,GREEN).move_to([5.05,1.05,0]))
    for left,count,col in ((2.37,2,BLUE),(4.46,8,GREEN)):
        side=.25 if count==2 else .15
        for i in range(count):
            for j in range(count):
                g.add(Square(side_length=side,stroke_color=col,stroke_width=1.5,
                             fill_color=col,fill_opacity=.22).move_to([left+side*j,-.58+side*i,0]))
    g.add(tex(r'D\times D=2\times2',22,BLUE).move_to([2.86,-1.37,0]),
          tex(r'N\times N=16\times16',22,GREEN).move_to([5.06,-1.37,0]),
          jp('右は16×16から縮めて表示',14,MUTED).move_to([4.88,-2.16,0]))
    return g

def manifold_panel(stage):
    g,q=side_axes('第2特徴の値','特徴の積',(-2.15,2.15),(0,4.3),[(-2,'−2'),(0,'0'),(2,'2')],[(0,'0'),(2,'2'),(4,'4')])
    xs=np.linspace(-2.05,2.05,65)
    g.add(VMobject().set_points_as_corners([q(x,x*x) for x in xs]).set_stroke(GREEN,width=3))
    original=q(np.sqrt(PREIMAGE_LHS),PREIMAGE_LHS)
    projected=q(np.sqrt(PREIMAGE_LHS),PREIMAGE_RHS)
    if stage>=1:
        g.add(Dot(original,radius=.10,color=BLUE),jp('元の特徴',15,BLUE).move_to(original+LEFT*.55+UP*.15))
    if stage>=2:
        g.add(DashedLine(original,projected,dash_length=.08,color=RED,stroke_width=2),
              Dot(projected,radius=.10,color=YELLOW),
              jp('投影先',15,YELLOW).move_to(projected+RIGHT*.48+DOWN*.12))
    return g

def contour_group(new=False):
    g=VGroup()
    for level,color in ((1.5,GREEN),(-1.5,PURPLE),(.7,GREEN),(-.7,PURPLE)):
        for sign in (-1,1):
            xarr=np.linspace(.37,1.58,80)*sign
            pts=[]
            for x in xarr:
                y=level/(np.sqrt(2)*x)
                if abs(y)<=1.43:pts.append(P([x,y]))
            if len(pts)>1:g.add(VMobject().set_points_as_corners(pts).set_stroke(color,width=2.1,opacity=.68))
    if new:g.add(Dot(P(NEW_POINT),radius=.11,color=YELLOW),
                 jp('新点',15,YELLOW).move_to(P(NEW_POINT)+RIGHT*.41+UP*.18))
    return g

FORMULAS=[
 [r'D=2,\ N=16',r'\mathrm{PCA}:\ S u_i=\lambda_i u_i',r'x_1\ ? ',r'\lambda_1=1.4464',r'\lambda_2=1.0025',r'x_1x_2\ ?'],
 [r'(+, +),(-,-)',r'x_1x_2>0',r'x_1x_2',r'x_1x_2=0',r'\sqrt2 x_1x_2',r'\phi(x)\ ?'],
 [r'\phi=(x_1^2,\sqrt2x_1x_2,x_2^2)^T',r'\phi_c=\phi-\bar\phi',r'C={1\over N}\sum_n\phi_{cn}\phi_{cn}^T',r'(0.037,2.900,0.010)',r'C v_1=\lambda_1v_1',r'\dim\phi\uparrow'],
 [r'6.1\quad\phi(x)^T\phi(z)',r'x=(1.2,1),\ z=(-1.2,1)',r'(x^Tz)^2=0.1936',r'\phi(x)^T\phi(z)',r'k(x,z)=(x^Tz)^2',r'K_{nm}=k(x_n,x_m)'],
 [r'K_{nm}=k(x_n,x_m)',r'\lambda_{\max}(K)\simeq50.004',r'7.513-3.475-3.475+3.097',r'\widetilde K_{11}\simeq3.660',r'\widetilde K=HKH',r'\sum_m\widetilde K_{nm}=0'],
 [r'\widetilde K a_i=N\lambda_i a_i',r'46.401/16\simeq2.900',r'a_i^T\widetilde K a_i=1',r'y_1(x)=\sum_n a_{1n}\widetilde k(x,x_n)',r'[-1.901,-1.505]\quad[1.505,1.901]',r'\operatorname{rank}(\widetilde K)\leq N'],
 [r'x_*=(0.9,0.8)',r'\widetilde k(x_*,x_n)',r'y_1(x_*)\simeq1.018',r'\sqrt2x_1x_2=c',r'k_G(x,z)=e^{-\|x-z\|^2/0.1}',r'y_1(x)'],
 [r'D\times D\quad N\times N',r'\operatorname{rank}(\widetilde K)\leq N',r'\phi(x)\to\phi_{\rm proj}',r'\phi_2^2=2\phi_1\phi_3',r'3.613\neq2.900',r'\phi^{-1}(\phi_{\rm proj})\ ?'],
 [r'N=16',r'\lambda_1=1.4464',r'y_1=\sqrt2x_1x_2',r'\widetilde K a_1=N\lambda_1a_1',r'k(x,z)=(x^Tz)^2',r'12.4\quad ?'],
]

def panel_for(si,bi):
    if si==0:
        return score_panel('linear',bi>=3)
    if si==1:
        return sign_panel(bi) if bi<2 else score_panel('product',True)
    if si==2:
        return variance_panel('feature',1 if bi<2 else 3)
    if si==3:
        return recap_panel() if bi==0 else comparison_panel(min(2,bi-1))
    if si==4:
        return heatmap(centered=bi>=3,selected=bi in (2,3))
    if si==5:
        return variance_panel('gram') if bi<3 else score_panel('kernel')
    if si==6:
        return score_panel('kernel',True,new=bi>=2)
    if si==7:
        return dimension_panel() if bi<2 else manifold_panel(0 if bi==2 else 1 if bi==3 else 2)
    return score_panel('linear' if bi<2 else 'kernel')

def left_extra(si,bi):
    if si in (0,8) and (bi>=2 if si==0 else bi>=1):
        return VGroup(Line(P([-1.55,0]),P([1.55,0]),color=YELLOW,stroke_width=3.5))
    if si==6 and bi>=3:return contour_group(new=bi>=0)
    if si==7 and bi>=2:
        return VGroup(Circle(radius=.108,stroke_color=BLUE,stroke_width=2.6,fill_opacity=0).move_to(P(PREIMAGE_POINT)))
    return VGroup()

class PRML123KernelPCA(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            self.add(main_axes(),points(),jp('説明用の16点（二次カーネル）' if si==6 else '説明用の16点',14,MUTED).move_to([-4.30,-2.47,0]))
            old=None
            for bi,beat in enumerate(story['beats']):
                formula=tex(FORMULAS[si][bi],26,YELLOW).move_to([4.1,2.43,0])
                if formula.width>4.65:formula.scale(4.65/formula.width)
                note=jp(beat['visual_note'],17,MUTED).move_to([4.1,1.81,0])
                if note.width>4.7:note.scale(4.7/note.width)
                state=VGroup(formula,note,panel_for(si,bi),left_extra(si,bi))
                if old is None:
                    animation=FadeIn(state)
                elif (si==3 and bi==1) or (si==7 and bi>=2):
                    animation=Succession(FadeOut(old),FadeIn(state))
                else:
                    animation=ReplacementTransform(old,state)
                self.beat(animation,end_sentence=1,actions=[dict(name='図を更新',start=0,end=self.durations[self.beat_index])])
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml123_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
