"""PRML 11.1: finding a missed probability mass with sampling algorithms."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene,jp,tex
from sampling_model import target,proposal,truth,envelope,introductory_draws,importance_draws,rejection_draws,resampled_draws

BG='#10141F'
BLUE=ManimColor('#58C4DD'); GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
INTRO=introductory_draws(); IMP,WEIGHTS=importance_draws(); REJ_Z,REJ_U,ACCEPTED=rejection_draws(); RES=resampled_draws()
K,_,_=envelope(); TRUE=truth()
X0,X1=-5.35,1.55;Y0,Y1=-1.80,1.38;Z0,Z1=-4.6,3.5

def J(s,size=18,color=WHITE):
    mob=jp(str(s),size,color)
    return mob

def X(z):return X0+(z-Z0)/(Z1-Z0)*(X1-X0)
def Y(v,ymax):return Y0+v/ymax*(Y1-Y0)
def P(z,v,ymax):return [X(z),Y(v,ymax),0]
def path(points,color,width=2.8,opacity=1):
    return VMobject().set_points_as_corners([np.array([x,y,0]) for x,y in points]).set_stroke(color,width=width,opacity=opacity)
def density_curve(f,ymax,color,width=3):
    zs=np.linspace(Z0,Z1,135)
    return path([(X(z),Y(min(float(f(z)),ymax),ymax)) for z in zs],color,width)
def y_limit(si):return 3.6 if si==4 else 2.6 if si==3 else 1.0

def axes(si):
    ymax=y_limit(si)
    g=VGroup(Line([X0,Y0,0],[X1,Y0,0],stroke_width=1.5,color=MUTED),Line([X0,Y0,0],[X0,Y1,0],stroke_width=1.5,color=MUTED))
    for z in (-4,-2,0,1,2,3):
        x=X(z)
        g.add(Line([x,Y0,0],[x,Y0-.065,0],color=MUTED,stroke_width=1),J(str(z),14,MUTED).move_to([x,Y0-.24,0]))
    ticks=((0,'0'),(.5,'0.5'),(1,'1')) if ymax==1 else ((0,'0'),(1,'1'),(2,'2')) if ymax==2.6 else ((0,'0'),(1.5,'1.5'),(3,'3'))
    for v,label in ticks:
        g.add(Line([X0,Y(v,ymax),0],[X0-.07,Y(v,ymax),0],color=MUTED,stroke_width=1),J(label,13,MUTED).move_to([X0-.32,Y(v,ymax),0]))
    g.add(tex('z',24,MUTED).move_to([X1+.22,Y0-.17,0]),J('密度',16,MUTED).move_to([X0-.05,Y1+.30,0]))
    return g

def right_area(ymax):
    zs=np.linspace(1,Z1,70)
    pts=[P(1,0,ymax)]+[P(z,float(target(z)),ymax) for z in zs]+[P(Z1,0,ymax)]
    return Polygon(*pts,color=RED,stroke_width=0,fill_color=RED,fill_opacity=.20)

def dots(values,y=-1.53,color=BLUE,radius=.034,limit=None):
    g=VGroup()
    for i,z in enumerate(values[:limit] if limit else values):
        if Z0<=z<=Z1:
            yy=y+(i%3)*.09
            g.add(Dot([X(float(z)),yy,0],radius=radius,color=color))
    return g

def main_overlay(si,bi):
    ymax=y_limit(si)
    g=VGroup()
    if si in (3,4):
        k=K if si==3 else 12-(12-K)*min(bi,2)/2
        g.add(density_curve(lambda z:k*proposal(z),ymax,BLUE,2.8))
        g.add(J(f'包絡線  k = {k:.3f}',16,BLUE).move_to([-3.73,1.75,0]))
    elif si==2 and bi<4:
        # One-axis exponential illustration; same horizontal coordinates, changed units.
        u=[.21,.43,.70,.88][bi]
        xv=-np.log(1-u)
        xs=np.linspace(0,min(xv,Z1),60)
        g.add(Polygon(P(0,0,ymax),*[P(z,float(np.exp(-z)),ymax) for z in xs],P(min(xv,Z1),0,ymax),
                      color=YELLOW,stroke_width=0,fill_color=YELLOW,fill_opacity=.18))
        g.add(density_curve(lambda z:np.exp(-max(z,0)) if z>=0 else 0,ymax,RED))
        g.add(J('指数分布',16,RED).move_to([-1.3,1.75,0]))
    else:
        g.add(density_curve(proposal,ymax,BLUE,2.5))
        g.add(J('提案 q',16,BLUE).move_to([-4.13,1.77,0]))
    if si!=2 or bi>=4:
        if (si==0 and bi>=3) or (si in (1,5,6,7,8) and bi>=1):g.add(right_area(ymax))
        g.add(density_curve(target,ymax,RED,3.2))
        g.add(J('目標 p',16,RED).move_to([-1.35 if si in (3,4) else -2.75,1.77,0]))
    if si in (0,7,8):
        n=24 if si!=0 else (0 if bi==0 else min(24,4+bi*4))
        if n:g.add(dots(INTRO[:n]))
        if si==7 and bi>=1:
            # Importance weights are strictly a property of the sampled positions.
            for z in INTRO[::3]:
                val=min(.30,float(target(z)/proposal(z))*.055)
                g.add(Line([X(float(z)),-1.36,0],[X(float(z)),-1.36+val,0],color=YELLOW,stroke_width=2))
    elif si==3:
        if bi>=1:
            for i in range(9):
                z=float(REJ_Z[i]);u=float(REJ_U[i]*K*proposal(z))
                if Z0<=z<=Z1:
                    g.add(Dot(P(z,u,ymax),radius=.048,color=GREEN if ACCEPTED[i] else PURPLE))
        if bi>=4:g.add(dots(REJ_Z[ACCEPTED][:60],color=GREEN,radius=.025))
    elif si==5:
        if bi>=0:
            g.add(dots(IMP[:min(80,8+bi*12)],color=BLUE,radius=.022))
        if bi>=2:
            for z,w in zip(IMP[:min(40,8+bi*7)],WEIGHTS[:min(40,8+bi*7)]):
                height=min(.36,float(w)*5)
                g.add(Line([X(float(z)),-1.30,0],[X(float(z)),-1.30+height,0],color=YELLOW,stroke_width=2))
    elif si==6 and bi>=1:g.add(dots(RES[:min(90,15+bi*14)],color=GREEN,radius=.021))
    elif si==1 and bi>=1:
        for z in (-2,-.7,.6,1.4,2.1):
            color=YELLOW if z>1 else BLUE
            g.add(Dot([X(z),-1.45,0],radius=.07,color=color))
            if bi>=2:g.add(J('1' if z>1 else '0',13,color).move_to([X(z),-1.16,0]))
    elif si==2 and bi<4:
        g.add(Dot(P(1.2,float(np.exp(-1.2)),ymax),radius=.075,color=YELLOW))
    if si in (0,7,8) or (si==1 and bi>=1) or (si==5 and bi>=3):
        x=X(1)
        g.add(DashedLine([x,Y0,0],[x,Y1,0],dash_length=.11,color=YELLOW,stroke_width=1.1))
        g.add(J('z > 1',15,YELLOW).move_to([X(1.55),1.13,0]))
    return g

def mini_axes(x0=2.82,x1=5.94,y0=-1.65,y1=.52,top=.22):
    g=VGroup(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.2),Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.2))
    for v in (0,.1,.2):
        y=y0+(v/top)*(y1-y0)
        g.add(Line([x0-.05,y,0],[x0,y,0],color=MUTED,stroke_width=1),J(f'{v:g}',13,MUTED).move_to([x0-.31,y,0]))
    g.add(J('右側の確率',15,MUTED).move_to([2.98,.87,0]))
    return g

def probability_panel(entries,shown):
    g=mini_axes()
    positions=np.linspace(3.35,5.52,len(entries))
    for i,((name,v,color),x) in enumerate(zip(entries,positions)):
        if i>=shown:continue
        top=-1.65+v/.22*2.17
        g.add(Line([x,-1.65,0],[x,top,0],color=color,stroke_width=26),J(name,14,color).move_to([x,-1.93,0]),J(f'{v:.3f}',15,color).move_to([x,top+.16,0]))
    return g

def cdf_panel(bi):
    x0,x1=2.90,5.85;y0,y1=-1.60,.62
    g=VGroup(J('逆変換法',17,MUTED).move_to([4.42,.99,0]),Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.2),Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.2))
    for val in (0,1,2,3):
        x=x0+val/3*(x1-x0)
        g.add(Line([x,y0,0],[x,y0-.05,0],color=MUTED,stroke_width=1),J(str(val),13,MUTED).move_to([x,y0-.20,0]))
    for val in (0,.5,1):
        y=y0+val*(y1-y0)
        g.add(Line([x0-.05,y,0],[x0,y,0],color=MUTED,stroke_width=1),J(f'{val:g}',13,MUTED).move_to([x0-.26,y,0]))
    g.add(tex('y',23,MUTED).move_to([x1+.20,y0-.18,0]),tex('F(y)',22,MUTED).move_to([x0-.17,y1+.27,0]))
    xs=np.linspace(0,3,80)
    g.add(path([(x0+x/3*(x1-x0),y0+(1-np.exp(-x))*(y1-y0)) for x in xs],RED,2.5))
    u=[.21,.43,.70,.88,.60,.40,.76][bi]
    xv=-np.log(1-u)
    xx=x0+xv/3*(x1-x0);yy=y0+u*(y1-y0)
    g.add(Line([x0,yy,0],[xx,yy,0],color=YELLOW,stroke_width=1.4),Line([xx,y0,0],[xx,yy,0],color=YELLOW,stroke_width=1.4),Dot([xx,yy,0],radius=.055,color=YELLOW))
    return g

def rejection_panel(bi):
    if bi<4:
        x0,x1=3.04,5.83;y0,y1=-1.48,.45
        g=VGroup(J('候補点の高さで採否を決める',17,MUTED).move_to([4.42,.92,0]),
                 Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.2),
                 Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.2))
        for x,label in ((x0,'0'),(4.43,'1'),(x1,'2')):
            g.add(Line([x,y0,0],[x,y0-.05,0],color=MUTED,stroke_width=1),J(label,13,MUTED).move_to([x,y0-.19,0]))
        for y,label in ((y0,'0'),(-.53,'1')):
            g.add(Line([x0-.05,y,0],[x0,y,0],color=MUTED,stroke_width=1),J(label,13,MUTED).move_to([x0-.24,y,0]))
        g.add(tex('z',22,MUTED).move_to([x1+.18,y0-.18,0]),tex('u',22,MUTED).move_to([x0-.18,y1+.20,0]),
              Dot([4.34,-.62,0],radius=.07,color=YELLOW),J('採用？',16,YELLOW).move_to([4.44,-2.12,0]))
        return g
    g=VGroup(J('棄却サンプリング',18,MUTED).move_to([4.42,.92,0]))
    rows=[('提案', '2000点',BLUE),('採用','236点',GREEN),('採択率','0.120',YELLOW)]
    for i,(a,b,c) in enumerate(rows):
        y=.28-i*.65
        g.add(J(a,17,MUTED).move_to([3.34,y,0]),J(b,22,c).move_to([5.11,y,0]))
    g.add(J('上の余りを捨てる',16,PURPLE).move_to([4.45,-2.05,0]))
    return g

def dimension_panel(bi):
    if bi in (2,3):
        x0,x1=3.11,5.83;y0,y1=-1.55,.45
        g=VGroup(J('適応的棄却サンプリング',16,MUTED).move_to([4.43,.91,0]),
                 Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.2),
                 Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1.2))
        px=lambda x:x0+(x+1.6)/3.2*(x1-x0)
        py=lambda v:y0+(v+1.5)/2.4*(y1-y0)
        for x,label in ((-1,'-1'),(0,'0'),(1,'1')):
            g.add(Line([px(x),y0,0],[px(x),y0-.05,0],color=MUTED,stroke_width=1),J(label,13,MUTED).move_to([px(x),y0-.20,0]))
        for v,label in ((-1,'-1'),(0,'0')):
            g.add(Line([x0-.05,py(v),0],[x0,py(v),0],color=MUTED,stroke_width=1),J(label,13,MUTED).move_to([x0-.25,py(v),0]))
        g.add(tex('z',22,MUTED).move_to([x1+.17,y0-.17,0]),tex(r'\log p',19,MUTED).move_to([x0-.08,y1+.21,0]))
        xs=np.linspace(-1.6,1.6,75)
        g.add(path([(px(x),py(-x*x/2)) for x in xs],RED,2.6))
        for x in (-.9,.9):
            line=lambda t,x=x:-x*x/2-x*(t-x)
            g.add(path([(px(t),py(line(t))) for t in np.linspace(-1.6,1.6,35) if -1.5<=line(t)<=.9],BLUE,1.6,.70))
            g.add(Dot([px(x),py(-x*x/2),0],radius=.05,color=YELLOW))
        g.add(J('接線の上側で包む',14,GREEN).move_to([4.43,-2.08,0]))
        return g
    if bi<4:
        g=VGroup(J('包絡線の倍率を動かす',17,MUTED).move_to([4.42,.90,0]))
        x0,x1=3.12,5.83;y=-.56
        g.add(Line([x0,y,0],[x1,y,0],color=MUTED,stroke_width=2))
        for v,label in ((7,'7'),(9,'9'),(12,'12')):
            x=x0+(v-7)/5*(x1-x0)
            g.add(Line([x,y-.08,0],[x,y+.08,0],color=MUTED,stroke_width=1),J(label,14,MUTED).move_to([x,y-.29,0]))
        k=12-(12-K)*min(bi,2)/2
        x=x0+(k-7)/5*(x1-x0)
        g.add(Dot([x,y,0],radius=.11,color=YELLOW),tex('k',23,YELLOW).move_to([x,y+.31,0]),
              J(f'k = {k:.3f}',19,YELLOW).move_to([4.43,-1.70,0]))
        return g
    g=VGroup(J('別の思考実験: 高次元ガウス',16,MUTED).move_to([4.43,.90,0]))
    for i,(d,ratio) in enumerate(((1,1/1.01),(100,1/1.01**100),(1000,1/1.01**1000))):
        y=.28-i*.74
        ratio_label = f'{ratio:.7f}' if d == 1000 else f'{ratio:.4f}'
        g.add(J(f'{d}次元',16,MUTED).move_to([3.40,y,0]),Line([4.08,y,0],[4.08+1.7*ratio,y,0],color=YELLOW,stroke_width=11),J(ratio_label,16,YELLOW).move_to([6.48,y,0]))
    g.add(J('幅の比 = 1.01',15,MUTED).move_to([4.44,-2.08,0]))
    return g

def resample_panel(bi):
    entries=[('生の数',11/200,BLUE),('重み',float(np.dot(WEIGHTS,IMP>1)),YELLOW),('選び直し',33/200,GREEN)]
    g=probability_panel(entries,1 if bi==0 else 3)
    if bi>=2:g.add(J('異なる位置: 90 / 200',15,MUTED).move_to([4.39,-2.26,0]))
    return g

def recap_panel():
    g=VGroup(J('復習: 10.7 期待値伝播',18,WHITE).move_to([4.40,.92,0]),J('分布の形を近似',16,MUTED).move_to([4.40,.43,0]))
    x0,x1=3.03,5.75;y0,y1=-1.62,.05
    g.add(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1),Line([x0,y0,0],[x0,y1,0],color=MUTED,stroke_width=1))
    xs=np.linspace(-2.5,2.5,80)
    f=lambda x:.55*np.exp(-.5*((x+.55)/.55)**2)+.22*np.exp(-.5*((x-1.1)/.38)**2)
    h=lambda x:.49*np.exp(-.5*((x+.2)/.90)**2)
    for func,color in ((f,RED),(h,BLUE)):
        g.add(path([(x0+(x+2.5)/5*(x1-x0),y0+func(x)*2.6) for x in xs],color,2.2))
    g.add(J('元の分布',14,RED).move_to([3.57,-2.12,0]),J('近似',14,BLUE).move_to([5.10,-2.12,0]))
    g.add(SurroundingRectangle(g,color=YELLOW,buff=.12,stroke_width=.9))
    return g

def side_panel(si,bi):
    if si==0:return probability_panel([('24点',0,BLUE),('真の値',TRUE,RED)],1 if bi<3 else 2)
    if si==1:return recap_panel() if bi==0 else probability_panel([('標本',0,BLUE),('面積',TRUE,RED)],1 if bi<2 else 2)
    if si==2:return cdf_panel(bi)
    if si==3:return rejection_panel(bi)
    if si==4:return dimension_panel(bi)
    if si==5:
        g=probability_panel([('生の数',11/200,BLUE),('重点',float(np.dot(WEIGHTS,IMP>1)),YELLOW),('真の値',TRUE,RED)],1 if bi<3 else 2 if bi<5 else 3)
        g.add(J('重点サンプリング',15,YELLOW).move_to([4.45,1.12,0]))
        return g
    if si==6:return resample_panel(bi)
    if si==7:return probability_panel([('24点',0,BLUE),('真の値',TRUE,RED)],1 if bi<3 else 2)
    return probability_panel([('24点',0,BLUE),('真の値',TRUE,RED),('棄却',float(np.mean(REJ_Z[ACCEPTED]>1)),GREEN),('重点',float(np.dot(WEIGHTS,IMP>1)),YELLOW)],1 if bi<2 else 2 if bi==2 else 3 if bi==3 else 4)

FORMULAS=[
 [r'P(z>1)=?',r'L=24',r'0/24=0',r'P(z>1)=0.180',r'10.7\to11.1',r'\tilde p(z)',r'0\to0.180\;?'],
 ['',r'f(z)=\mathbf1[z>1]',r'E_p[f]=\int f(z)p(z)\,dz',r'\operatorname{Var}(\bar f)=\operatorname{Var}(f)/L',r'u\sim U(0,1)',r'F(y)=\int_{-\infty}^{y}p(t)\,dt',''],
 [r'u=F(y)',r'y=F^{-1}(u)',r'F(y)=1-e^{-\lambda y}',r'y=-\ln(1-u)/\lambda','','',''],
 [r'kq(z)\geq\tilde p(z)',r'z\sim q,\;u\sim U(0,kq(z))',r'u\leq\tilde p(z)',r'z_{\rm accepted}\sim p',r'P(\rm accept)=1/k=0.120',r'236/2000',r'1-1/k=0.880'],
 [r'k\downarrow',r'P(\rm accept)=\int\tilde p(z)dz/k',r'\log p(z)',r'(\log p)\prime\prime\leq0',r'1/1.01^{1000}=0.0000477','',''],
 [r'L=200',r'r_l=p(z_l)/q(z_l)',r'r_l=p(z_l)/q(z_l)',r'w_l=r_l/\sum_m r_m',r'\hat P=\sum_lw_l\mathbf1[z_l>1]',r'\hat P=0.164',r'z_l\sim q,\quad w_l\propto p/q'],
 [r'L=200',r'\mathrm{SIR}',r'33/200=0.165',r'L\to\infty',r'E_p[f]\approx\sum_lw_lf(z_l)','',''],
 [r'0/24=0',r'\sum_{z_l>1}w_l=0',r'\sum_{z_l>1}w_l=0',r'q(z)>0',r'q(z)>0',r'N_{\rm eff}=1/\sum_lw_l^2\approx86',''],
 [r'0/24=0',r'P(z>1)=0.180',r'\hat P_{\rm reject}=0.195',r'\hat P_{\rm IS}=0.164',r'q(z)>0','',''],
]

class PRML111BasicSamplingAlgorithms(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            self.add(axes(si))
            self.overlay=main_overlay(si,0);self.add(self.overlay)
            self.side=side_panel(si,0);self.add(self.side)
            self.note=None;self.formula=None
            for bi in range(len(story['beats'])):self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml111_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
    def run_beat(self,si,bi):
        if self.note is not None:self.remove(self.note)
        self.note=J(SCENES[si]['beats'][bi]['visual_note'],18,MUTED).move_to([4.40,2.48,0])
        if self.note.width>4.10:self.note.scale(4.10/self.note.width)
        self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        self.formula=None
        formula=FORMULAS[si][bi]
        if formula:
            self.formula=tex(formula,24,YELLOW).move_to([4.40,1.77,0])
            if self.formula.width>4.0:self.formula.scale(4.0/self.formula.width)
            self.add(self.formula)
        next_overlay=main_overlay(si,bi);next_side=side_panel(si,bi)
        duration=self.durations[self.beat_index]
        self.beat(phases=[('panel',.5,lambda:Succession(FadeOut(self.side),FadeIn(next_side))),('graph',duration-.5,lambda:AnimationGroup(FadeOut(self.overlay),FadeIn(next_overlay)))])
        self.side=next_side
        self.overlay=next_overlay
