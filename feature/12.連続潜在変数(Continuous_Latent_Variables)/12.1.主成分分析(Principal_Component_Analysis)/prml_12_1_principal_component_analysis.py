"""A single 13-point cloud shows PCA's two equivalent objectives."""
import json
from pathlib import Path
import numpy as np
from manim import *

from narration_content import SCENES
from pca_model import (POINTS, CENTERED, MEAN, COV, EIGENVALUES, EIGENVECTORS,
                       THETA, direction, projected, reconstruction, variance, error)
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST

BG = '#101923'
BLUE = '#58C4DD'
YELLOW = '#FFFF00'
GREEN = '#83C167'
PURPLE = '#9A72AC'
RED = '#FC6255'
MUTED = '#A6B6C5'
WHITE = '#F4F7FA'
GX, GY, SCALE = -3.30, -.15, .88


def P(v):
    return np.array([GX + SCALE * float(v[0]), GY + SCALE * float(v[1]), 0])


def graph_axes():
    g = VGroup()
    # Deliberate open axes; no decorative panel or shadow behind the graph.
    g.add(Line(P([-2, 0]), P([4.2, 0]), color=MUTED, stroke_width=2),
          Line(P([0, -2.15]), P([0, 2.38]), color=MUTED, stroke_width=2))
    for x in (-2, 0, 2, 4):
        q = P([x, 0]); g.add(Line(q+UP*.055, q+DOWN*.055, color=MUTED, stroke_width=1.5))
        g.add(jp(str(x), 14, MUTED).move_to(q+DOWN*.24))
    for y in (-2, 0, 2):
        q = P([0, y]); g.add(Line(q+LEFT*.055, q+RIGHT*.055, color=MUTED, stroke_width=1.5))
        if y:
            g.add(jp(str(y), 14, MUTED).move_to(q+LEFT*.27))
    g.add(tex('x_1', 21, MUTED).move_to(P([4.5, -.08])+DOWN*.13),
          tex('x_2', 21, MUTED).move_to(P([-.23, 2.60])))
    return g


def clipped_line(theta, offset=0):
    u = direction(theta); n = np.array([-u[1], u[0]])
    center = MEAN + offset*n
    low, high = -10., 10.
    for index, bounds in enumerate(((-1.95, 4.12), (-2.12, 2.27))):
        if abs(u[index]) < 1e-8:
            continue
        a, b = sorted(((bounds[0]-center[index])/u[index], (bounds[1]-center[index])/u[index]))
        low, high = max(low, a), min(high, b)
    return Line(P(center+low*u), P(center+high*u), color=PURPLE, stroke_width=4)


def scatter():
    return VGroup(*[Dot(P(x), radius=.083, color=BLUE) for x in POINTS])


def reconstruction_marks(theta, offset=0, m=None):
    if m is None:
        u = direction(theta); n = np.array([-u[1], u[0]])
        anchor = MEAN + offset*n
        xhat = anchor + np.outer((POINTS-anchor) @ u, u)
    else:
        xhat = reconstruction(int(np.clip(round(m), 0, 2)))
    g = VGroup()
    for i in range(len(POINTS)):
        if i % 2 == 0 and np.linalg.norm(POINTS[i]-xhat[i]) > .035:
            g.add(Line(P(POINTS[i]), P(xhat[i]), color=RED, stroke_width=2.3, stroke_opacity=.78))
    for x in xhat:
        g.add(Dot(P(x), radius=.066, color=YELLOW))
    return g


def small_axes(x_name, y_name, x_ticks, y_ticks, xlim, ylim):
    x0, x1, y0, y1 = 2.85, 6.00, -1.76, .90
    g = VGroup(Line([x0,y0,0], [x1,y0,0], color=MUTED, stroke_width=1.8),
               Line([x0,y0,0], [x0,y1,0], color=MUTED, stroke_width=1.8))
    def at(x,y):
        return np.array([x0+(x-xlim[0])/(xlim[1]-xlim[0])*(x1-x0),
                         y0+(y-ylim[0])/(ylim[1]-ylim[0])*(y1-y0), 0])
    for x, label in x_ticks:
        q=at(x,ylim[0]);g.add(Line(q+UP*.045,q+DOWN*.045,color=MUTED,stroke_width=1))
        g.add(jp(label,13,MUTED).move_to(q+DOWN*.20))
    for y, label in y_ticks:
        q=at(xlim[0],y);g.add(Line(q+LEFT*.045,q+RIGHT*.045,color=MUTED,stroke_width=1))
        if y != ylim[0]:g.add(jp(label,13,MUTED).move_to(q+LEFT*.30))
    g.add(jp(x_name,15,MUTED).move_to([x1-.02,y0-.50,0]),
          jp(y_name,15,MUTED).move_to([x0-.08,y1+.25,0]))
    return g, at


def angle_panel(theta_tracker, kind='variance'):
    y_name={'variance':'分散','error':'誤差','both':'分散・誤差'}[kind]
    g, at=small_axes('角度',y_name,[(0,'0°'),(45,'45°'),(90,'90°')],
                     [(0,'0'),(2,'2'),(4,'4')],(0,90),(0,4.15))
    angles=np.linspace(0,np.pi/2,91)
    def curve(fn,color):
        pts=[at(np.degrees(a),fn(a)) for a in angles]
        return VMobject().set_points_as_corners(pts).set_stroke(color,width=2.7)
    if kind in ('variance','both'):
        g.add(curve(variance,GREEN))
    if kind in ('error','both'):
        g.add(curve(error,RED))
    active=variance if kind != 'error' else error
    marker=always_redraw(lambda: Dot(at(float(np.degrees(theta_tracker.get_value())),
                                      active(theta_tracker.get_value())),radius=.093,color=YELLOW))
    legend=VGroup(jp('緑: 影の分散',14,GREEN),jp('赤: 再構成誤差',14,RED)).arrange(DOWN,aligned_edge=LEFT,buff=.08)
    if kind=='both':g.add(legend.move_to([4.60,1.31,0]))
    else:g.add(jp('影の分散' if kind=='variance' else '再構成誤差',16,GREEN if kind=='variance' else RED).move_to([4.55,1.30,0]))
    return g,marker


def error_bars(m_mode=False, chosen=None, visible=None):
    if m_mode:
        values=[sum(EIGENVALUES), EIGENVALUES[1], 0.]
        names=['0','1','2'];top=4.2;yticks=[(0,'0'),(2,'2'),(4,'4')]
        x_name='M';x_range=(-.5,2.5)
    else:
        values=[error(0),error(THETA)]
        names=['横','主成分'];top=1.5;yticks=[(0,'0'),(.75,'0.75'),(1.5,'1.5')]
        x_name='方法';x_range=(-.5,1.5)
    g,at=small_axes(x_name,'誤差',list(zip(range(len(names)),names)),yticks,x_range,(0,top))
    for i,value in enumerate(values):
        if visible is not None and i>=visible:
            g.add(jp('?',22,MUTED).move_to(at(i,top*.46)))
            continue
        base=at(i,0);end=at(i,max(value,.025 if value else .012))
        color=YELLOW if chosen==i else (RED if i==0 else GREEN)
        g.add(Rectangle(width=.36,height=max(.025,end[1]-base[1]),fill_color=color,
                        fill_opacity=.82,stroke_width=0).move_to((base+end)/2))
        g.add(jp(f'{value:.3f}' if value else '0',14,color).move_to(end+UP*.16))
    return g


def offset_panel(offset):
    g,at=small_axes('ずれ','誤差',[(0,'0'),(.5,'0.5'),(1,'1')],
                    [(0,'0'),(.75,'0.75'),(1.5,'1.5')],(0,1),(0,1.5))
    pts=[at(t,error(THETA)+t*t) for t in np.linspace(0,1,61)]
    g.add(VMobject().set_points_as_corners(pts).set_stroke(RED,width=2.7))
    g.add(Dot(at(offset,error(THETA)+offset*offset),radius=.09,color=YELLOW))
    return g


def covariance_panel():
    g=VGroup(jp('平均からのずれを表に',17,GREEN).move_to([4.35,.80,0]),
             tex(r'S=\begin{pmatrix}2.629&1.835\\1.835&1.349\end{pmatrix}',27,WHITE).move_to([4.35,-.55,0]),
             jp('対角: それぞれの広がり',15,BLUE).move_to([4.35,-1.55,0]),
             jp('非対角: 同時の動き',15,PURPLE).move_to([4.35,-2.03,0]))
    return g


def recap_panel():
    """Recreate the ellipse/major-axis motif from PRML 2.3 geometry()."""
    g=VGroup()
    g.add(Rectangle(width=3.88,height=2.95,stroke_color=YELLOW,stroke_width=1,
                    fill_opacity=0).move_to([4.35,-.55,0]))
    g.add(jp('復習: 2.3 ガウス分布',17,WHITE).move_to([4.35,.54,0]))
    cloud=Ellipse(width=2.70,height=.82,color='#77D49A',stroke_width=2).rotate(THETA).move_to([4.35,-.51,0])
    g.add(cloud,Line([3.16,-1.36,0],[5.55,.34,0],color='#C29AFF',stroke_width=3),
          Dot([4.35,-.51,0],radius=.075,color='#FFE079'))
    g.add(jp('楕円の長い向き',16,'#77D49A').move_to([4.35,-1.59,0]))
    return g


def score_panel():
    g,at=small_axes('点番号','影',[(1,'1'),(7,'7'),(13,'13')],
                    [(-3,'-3'),(0,'0'),(3,'3')],(1,13),(-3.2,3.2))
    scores=CENTERED @ EIGENVECTORS[:,0]
    for i,score in enumerate(scores,1):
        g.add(Dot(at(i,float(score)),radius=.063,color=GREEN))
    return g


def whiten_panel():
    g,at=small_axes('第1軸','第2軸',[(-2,'-2'),(0,'0'),(2,'2')],
                    [(-2,'-2'),(0,'0'),(2,'2')],(-3,3),(-3,3))
    white=CENTERED @ EIGENVECTORS / np.sqrt(EIGENVALUES)
    for xy in white:
        g.add(Dot(at(*xy),radius=.058,color=GREEN))
    return g


FORMULAS=[
    [r'D=2\longrightarrow M=1',r'N=13',r'\theta=0^\circ',r'J(0^\circ)\simeq1.349',r'J(35.4^\circ)\simeq0.046',r'\theta\;?'],
    [r'\bar x+tu',r'\bar x=(1.15,0.35)',r'\bar x+tu',r'x_n-\bar x',r'z_n=u^T(x_n-\bar x)',r'\|u\|=1'],
    [r'\theta:0^\circ\to90^\circ',r'V(0^\circ)\simeq2.629',r'\theta\uparrow',r'V(35.4^\circ)\simeq3.932',r'V(90^\circ)\simeq1.349',r'\max_\theta V(\theta)'],
    [r'\mathrm{Cov}(x)',r'x_n-\bar x',r'S={1\over N}\sum_n(x_n-\bar x)(x_n-\bar x)^T',r'V(u)=u^TSu',r'2.629\to3.932',r'u^Tu=1'],
    [r'u^Tu=1',r'V(\theta)=u^TSu',r'\theta\simeq35.4^\circ',r'Su=\lambda u',r'\lambda_1\simeq3.932',r'\max V\;?'],
    [r'\hat x_n=\bar x+z_nu',r'J={1\over N}\sum_n\|x_n-\hat x_n\|^2',r'J(0^\circ)\simeq1.349',r'\theta\uparrow',r'V+J=\mathrm{tr}(S)\simeq3.98',r'\max V\Longleftrightarrow\min J'],
    [r'M=0',r'M=1',r'M=2',r'J=\sum_{i=M+1}^{D}\lambda_i',r'D=2,\;M=1',r'J_1\simeq0.046'],
    [r'z_n=u_1^T(x_n-\bar x)',r'\hat x_n=\bar x+z_nu_1',r'M\uparrow',r'y=\Lambda^{-1/2}U^T(x-\bar x)',r'\mathrm{Cov}(y)=I',r'M<D'],
    [r'J(0^\circ)\simeq1.349',r'\theta:0^\circ\to35.4^\circ',r'V\simeq3.932,\;J\simeq0.046',r'J=\lambda_2',r'\max V=\min J',r'12.2\;?'],
]


def angle_target(si,bi):
    if si==0:return [0,0,0,0,THETA,THETA][bi]
    if si==1:return THETA
    if si==2:return [0,0,np.radians(20),THETA,np.pi/2,THETA][bi]
    if si==3:return [THETA,THETA,THETA,THETA,0,THETA][bi]
    if si==4:return [0,np.radians(18),THETA,THETA,THETA,THETA][bi]
    if si==5:return [THETA,THETA,0,np.radians(20),THETA,THETA][bi]
    if si==6:return THETA
    if si==7:return THETA
    return [0,THETA,THETA,THETA,THETA,THETA][bi]


class PRML121PrincipalComponentAnalysis(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            theta=ValueTracker(angle_target(si,0))
            offset=ValueTracker(1.0 if si==1 else 0.0)
            count=ValueTracker(0 if si==6 else 1)
            reveal=ValueTracker(0 if si==0 else 1)
            self.add(graph_axes())
            line=always_redraw(lambda: clipped_line(theta.get_value(),offset.get_value()).set_opacity(
                0 if si==6 and round(count.get_value()) != 1 else 1))
            marks=always_redraw(lambda: reconstruction_marks(theta.get_value(),offset.get_value(),
                                                             count.get_value() if si==6 else None).set_opacity(reveal.get_value()))
            self.add(line,marks,scatter(),Dot(P(MEAN),radius=.09,color=GREEN))
            self.add(jp('平均',15,GREEN).move_to(P(MEAN)+UP*.25+LEFT*.30))
            plot_marker=None
            if si in (2,4,5):
                panel,plot_marker=angle_panel(theta,'both' if si==5 else 'variance')
                self.add(panel,plot_marker)
            elif si in (3,7):pass
            elif si in (1,6,0,8):pass
            current_formula=None;current_note=None;old_panel=None
            for bi,beat in enumerate(story['beats']):
                if si==0 and bi==3:
                    # The projected points are bright throughout the sentence
                    # that names them; the prior beat still leaves room to guess.
                    reveal.set_value(1)
                if si==6:
                    # M is a count. Switch its reconstruction at the start of
                    # the spoken beat, so the image agrees with the subtitle.
                    count.set_value([0,1,2,1,1,1][bi])
                if current_formula is not None:self.remove(current_formula,current_note)
                current_formula=tex(FORMULAS[si][bi],27,YELLOW).move_to([4.42,2.45,0])
                if current_formula.width>4.60:current_formula.scale(4.60/current_formula.width)
                current_note=jp(beat['visual_note'],17,MUTED).move_to([4.42,1.90,0])
                if current_note.width>4.55:current_note.scale(4.55/current_note.width)
                self.add(current_formula,current_note)
                if old_panel is not None:self.remove(old_panel)
                if si==0:
                    old_panel=error_bars(chosen=0 if bi<4 else 1,visible=1 if bi<4 else 2)
                elif si==1:
                    off=[1,.8,0,0,0,0][bi];old_panel=offset_panel(off)
                elif si==6:
                    old_panel=error_bars(m_mode=True,chosen=[0,1,2,1,1,1][bi])
                elif si==3:
                    old_panel=recap_panel() if bi==0 else covariance_panel()
                elif si==7:
                    old_panel=whiten_panel() if bi>=3 else score_panel()
                elif si==8:
                    old_panel=error_bars(chosen=0 if bi==0 else 1)
                else:old_panel=None
                if old_panel is not None:self.add(old_panel)
                animations=[]
                target=angle_target(si,bi)
                if abs(theta.get_value()-target)>1e-5:animations.append(theta.animate.set_value(target))
                if si==1:
                    off=[1,.8,0,0,0,0][bi]
                    if abs(offset.get_value()-off)>1e-5:animations.append(offset.animate.set_value(off))
                moving=bool(animations)
                if not animations:
                    animations=[ShowPassingFlash(line.copy().set_stroke(YELLOW,width=6),time_width=.22)]
                self.beat(*animations,end_sentence=1 if moving else None,
                          actions=[dict(name='投影を動かす',start=0,end=self.durations[self.beat_index])])
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml121_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
