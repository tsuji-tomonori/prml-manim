"""PRML 14.2: bootstrap committees and why correlated errors remain."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST
from committee_model import (TRAIN_X, TRAIN_Y, BOOTSTRAP_INDICES, GRID, QUERY_X,
                             truth, predict, comparison, orthogonal_errors,
                             orthogonal_expectation)

BG='#111823'; BLUE='#58C4DD'; YELLOW='#FFFF00'; GREEN='#83C167'
PURPLE='#9A72AC'; RED='#FC6255'; MUTED='#A8B3C0'
COLORS=(BLUE,PURPLE,'#E9966B','#79A8F8','#B5D768')
VALUES=predict(np.array([QUERY_X]))[0]
STATS=comparison()

def label(s,x,y,size=18,color=WHITE):return jp(s,size,color).move_to([x,y,0])
def note(s):return label(s,0,2.53,21,YELLOW)
def formula(s,y=-2.43,size=26):
    m=tex(s,size,YELLOW).move_to([0,y,0])
    if m.width>11.6:m.scale(11.6/m.width)
    return m

def recap(s):
    h=label(s,-4.05,2.51,18,WHITE)
    return VGroup(h,SurroundingRectangle(h,color=YELLOW,stroke_width=1.2,buff=.10))

def linegraph(xmin=-1,xmax=1,ymin=-3.0,ymax=2.3,x0=-4.65,x1=4.18,y0=-1.50,y1=1.78,
              ylabel='予測',xlabel='入力'):
    def point(x,y):return np.array([x0+(x-xmin)/(xmax-xmin)*(x1-x0),y0+(y-ymin)/(ymax-ymin)*(y1-y0),0.])
    g=VGroup(Line(point(xmin,ymin),point(xmax,ymin),color=MUTED,stroke_width=1.5),
             Line(point(xmin,ymin),point(xmin,ymax),color=MUTED,stroke_width=1.5))
    for x in (-1,0,1):
        px=point(x,ymin)[0]
        g.add(Line([px,y0-.06,0],[px,y0+.06,0],color=MUTED,stroke_width=1),
              label(str(x),px,y0-.27,13,MUTED))
    for y in (-2,0,2):
        if y<ymin or y>ymax:continue
        py=point(xmin,y)[1]
        g.add(Line([x0-.06,py,0],[x0+.06,py,0],color=MUTED,stroke_width=1),
              label(str(y),x0-.30,py,12,MUTED))
    g.add(label(ylabel,x0-.43,y1+.19,15,MUTED),label(xlabel,x1-.42,y0-.27,15,MUTED))
    return g,point

def curve(points,color,width=2.7,dashed=False):
    m=VMobject(stroke_color=color,stroke_width=width).set_points_smoothly(points)
    return DashedVMobject(m,num_dashes=55) if dashed else m

def regression_panel(si,bi):
    messages={
      0:['同じ入力で5本の予測','x = 0.45 を固定','予測が割れる','平均を予想','平均は 0.965','真値は 0.949','5分の1になる？','原因を追う'],
      1:['予測を束ねる','復習: 3.2 バイアス・バリアンス','平均でも残るずれ','5本を同じ軸で平均','コミッティの式','入力を動かす','多様性はどこから？','標本を引き直す'],
      3:['1点の成功から全体へ','曲線と真値のずれ','ずれを二乗','入力に沿って平均','個別平均 0.0845','コミッティ 0.0258','比は 0.306','何が消えた？'],
      7:['同じ x = 0.45','同じ5本の予測','平均は 0.965','全体でも改善','消えるのは散らばり','5分の1は条件付き','バギングとコミッティ','次はブースティング']}
    g=VGroup(recap(messages[si][bi]) if si==1 and bi==1 else note(messages[si][bi]))
    axis,p=linegraph()
    g.add(axis)
    xs=np.linspace(-1,1,150)
    yy=predict(xs)
    show_candidates=si!=0 or bi>=2
    if show_candidates:
        for j in range(5):g.add(curve([p(x,y) for x,y in zip(xs,yy[:,j])],('#FF6B77' if si==1 and bi==1 else COLORS[j]),1.8))
    truth_visible=(si in (3,7)) or (si==0 and bi>=4) or (si==1 and bi>=1)
    if truth_visible:g.add(curve([p(x,y) for x,y in zip(xs,truth(xs))],('#77D49A' if si==1 and bi==1 else RED),2.3,True))
    mean_visible=(si==0 and bi>=4) or (si==1 and bi>=3) or (si==3 and bi>=5) or si==7
    if mean_visible:g.add(curve([p(x,y) for x,y in zip(xs,yy.mean(axis=1))],GREEN,3.4))
    qx=p(QUERY_X,-3.0)[0]
    g.add(DashedLine([qx,-1.49,0],[qx,1.79,0],color=YELLOW,dash_length=.09,stroke_width=1.7),
          label('0.45',qx,-1.85,14,YELLOW))
    dots=[]
    if show_candidates:
        for j,v in enumerate(VALUES):
            d=Dot(p(QUERY_X,float(v)),radius=.083,color=('#FF6B77' if si==1 and bi==1 else COLORS[j]));g.add(d);dots.append(d)
    if truth_visible:g.add(Dot(p(QUERY_X,float(truth(QUERY_X))),radius=.09,color=('#77D49A' if si==1 and bi==1 else RED)))
    if mean_visible:
        d=Dot(p(QUERY_X,float(VALUES.mean())),radius=.12,color=GREEN);g.add(d);dots.append(d)
    # A separate readout enlarges the five heights at the fixed input.
    if si in (0,7):
        g.add(label('x = 0.45',5.14,1.91,15,YELLOW),
              Line([4.75,-.72,0],[4.75,1.50,0],color=MUTED,stroke_width=1.3))
        def gauge(v):return -.72+(float(v)-.75)/.40*2.22
        for mark in (.8,1.0,1.1):
            gy=gauge(mark)
            g.add(Line([4.69,gy,0],[4.81,gy,0],color=MUTED,stroke_width=1),
                  label(f'{mark:.1f}',5.17,gy,12,MUTED))
        if show_candidates:
            for j,v in enumerate(VALUES):g.add(Dot([4.75,gauge(v),0],radius=.064,color=COLORS[j]))
        if truth_visible:g.add(Dot([4.75,gauge(truth(QUERY_X)),0],radius=.08,color=RED))
        if mean_visible:g.add(Dot([4.75,gauge(VALUES.mean()),0],radius=.105,color=GREEN))
        g.add(label('赤 真値　緑 平均',5.12,-1.15,13,MUTED))
    elif si==3:
        # Small named axis for mean squared error, outside the regression plot.
        g.add(label('二乗誤差',5.30,1.73,15,YELLOW),
              Line([4.60,-1.30,0],[6.12,-1.30,0],color=MUTED,stroke_width=1.2),
              Line([4.60,-1.30,0],[4.60,1.42,0],color=MUTED,stroke_width=1.2))
        for v in (0,.05,.10):
            yy=-1.30+2.5*v/.10
            g.add(Line([4.55,yy,0],[4.65,yy,0],color=MUTED,stroke_width=1),
                  label(f'{v:g}',4.33,yy,11,MUTED))
        for x,name,value,color,shown in ((5.0,'個別',STATS['average_mse'],BLUE,bi>=4),
                                         (5.72,'平均',STATS['committee_mse'],GREEN,bi>=5)):
            if shown:
                height=2.5*value/.10
                g.add(Rectangle(width=.28,height=height,stroke_width=0,fill_color=color,fill_opacity=.9).move_to([x,-1.30+height/2,0]))
            g.add(label(name,x,-1.60,13,color))
        if bi in (2,3):
            g.add(Square(side_length=.33,stroke_color=RED,fill_color=RED,fill_opacity=.25).move_to([5.45,.55,0]),
                  label('ずれ²',5.45,.15,14,RED))
    else:
        g.add(label('5本',5.10,1.53,15,('#FF6B77' if si==1 and bi==1 else BLUE)) if show_candidates else label('5本の予測？',5.10,1.53,15,BLUE))
        if truth_visible:g.add(label('真値',5.10,1.15,15,('#77D49A' if si==1 and bi==1 else RED)))
        if mean_visible:g.add(label('平均',5.10,.77,15,GREEN))
    if si==0 and bi in (2,3):g.add(label('0.786 〜 1.112',0,-2.38,18,YELLOW))
    elif si==0 and bi>=4:g.add(label('平均 0.965　真値 0.949',0,-2.38,18,YELLOW))
    elif si==1 and bi==4:g.add(formula(r'y_{\rm COM}(x)=\frac1M\sum_{m=1}^{M}y_m(x)'))
    elif si==1 and bi in (1,2):g.add(label('データを替えると曲線が揺れる',0,-2.38,18,YELLOW))
    elif si==3 and bi in (1,2):g.add(formula(r'y_m(x)=h(x)+\epsilon_m(x)',size=24))
    elif si==3 and bi==3:g.add(formula(r'E_{\rm AV}=\frac1M\sum_m E_x[\epsilon_m(x)^2]',size=24))
    elif si==3 and bi>=4:
        g.add(label('個別 0.0845　→　平均 0.0258',0,-2.38,19,YELLOW))
    elif si==7:g.add(label('真値 0.949　平均 0.965　誤差比 0.306',0,-2.38,18,YELLOW))
    else:g.add(label('同じ入力で、予測の高さを比べる',0,-2.38,18,YELLOW))
    focus=dots[-1] if dots else g[1]
    return g,focus

def bootstrap_panel(bi):
    notes=['12点の観測','補足: ブートストラップ標本','重複と欠落','5回引き直す',
           '5本の曲線','新しい観測ではない','バギング','誤差も似やすい']
    if bi>=4:
        g,focus=regression_panel(1,3)
        g.remove(g[0],g[-1]);g.add(note(notes[bi]))
        g.add(label('各標本で同じ5次回帰を学習',0,-2.38,18,YELLOW))
        return g,focus
    g=VGroup(note(notes[bi]))
    xs=np.linspace(-5,5,12)
    for i,x in enumerate(xs):
        c=BLUE if bi<2 else (YELLOW if i in BOOTSTRAP_INDICES[1] else MUTED)
        d=Dot([x,.60,0],radius=.105,color=c)
        g.add(d,label(str(i+1),x,.24,13,MUTED))
    g.add(label('元の12点',0,1.43,19,BLUE))
    if bi>=1:
        counts=np.bincount(BOOTSTRAP_INDICES[1],minlength=12)
        for i,(x,count) in enumerate(zip(xs,counts)):
            if count:
                for j in range(int(count)):
                    g.add(Dot([x,-.56-j*.30,0],radius=.09,color=YELLOW))
            else:g.add(label('0',x,-.56,14,MUTED))
        g.add(label('2組目で選んだ回数',0,-1.87,18,YELLOW))
    if bi==3:
        for m in range(5):
            g.add(Dot([-3.0+m*.8,-2.38,0],radius=.09,color=COLORS[m]))
        g.add(label('5組の標本',1.9,-2.38,18,YELLOW))
    else:g.add(label('戻してから引く：重複してよい',0,-2.38,18,YELLOW))
    return g,g[1]

def variance_panel(bi):
    notes=['予測の上下を分ける','平均からのずれ','二乗誤差を分ける','散らばりは消える',
           '差は 0.0587','全員が同じなら','共通のずれは残る','5分の1の条件は？']
    g=VGroup(note(notes[bi]))
    q=VALUES
    lo=.65; hi=1.23; x0=-3.0; scale=2.9/(hi-lo)
    def py(y):return -1.3+(y-lo)*scale
    g.add(Line([x0,-1.3,0],[x0,1.8,0],color=MUTED,stroke_width=1.4))
    for v in (.7,.9,1.1):
        y=py(v);g.add(Line([x0-.08,y,0],[x0+.08,y,0],color=MUTED,stroke_width=1),
                      label(f'{v:.1f}',x0-.41,y,13,MUTED))
    g.add(label('予測値',x0-.32,2.03,15,MUTED))
    for j,v in enumerate(q):
        x=-1.70+j*.86
        d=Dot([x,py(float(v)),0],radius=.115,color=COLORS[j]);g.add(d)
        g.add(DashedLine([x,py(float(v)),0],[x,py(float(q.mean())),0],color=YELLOW,stroke_width=1.4))
        g.add(label(str(j+1),x,-1.65,14,MUTED))
    g.add(Line([-2.15,py(float(q.mean())),0],[3.2,py(float(q.mean())),0],color=GREEN,stroke_width=2.5),
          DashedLine([-2.15,py(float(truth(QUERY_X))),0],[3.2,py(float(truth(QUERY_X))),0],color=RED,stroke_width=1.8,dash_length=.10),
          label('平均',3.82,py(float(q.mean())),16,GREEN),
          label('真値',3.82,py(float(truth(QUERY_X)))-.32,16,RED))
    if bi==3:g.add(formula(r'E_{\rm AV}-E_{\rm COM}=E_x\!\left[\frac1M\sum_m(y_m-y_{\rm COM})^2\right]',size=24))
    elif bi in (4,5):g.add(label('0.0845 − 0.0258 = 0.0587',0,-2.40,20,YELLOW))
    else:g.add(label('黄色: 5本の平均からの距離',0,-2.40,18,YELLOW))
    return g,g[2]

def orthogonal_panel(bi):
    notes=['説明用の別の誤差','5本は違う形','2本の積を調べる','積の平均は 0',
           '5分の1の条件','1 → 0.2','本数だけでは足りない','共通の癖を混ぜる']
    g=VGroup(note(notes[bi]))
    ax,p=linegraph(ymin=-2.4,ymax=2.4,ylabel='誤差',x0=-4.65,x1=3.65)
    g.add(ax)
    xs=np.linspace(-1,1,170); er=orthogonal_errors(0.,xs)
    if bi in (2,3):
        product=er[:,0]*er[:,1]
        for x,v in zip(xs[::4],product[::4]):
            a=p(x,0);b=p(x,v)
            g.add(Line(a,b,color=YELLOW if v>=0 else RED,stroke_width=4.2,stroke_opacity=.42))
        for j in (0,1):g.add(curve([p(x,y) for x,y in zip(xs,er[:,j])],COLORS[j],2.1))
        g.add(label('黄: 正の積',4.75,1.25,15,YELLOW),label('赤: 負の積',4.75,.82,15,RED))
    else:
        for j in range(5):g.add(curve([p(x,y) for x,y in zip(xs,er[:,j])],COLORS[j],1.8))
        mean=er.mean(axis=1)
        g.add(curve([p(x,y) for x,y in zip(xs,mean)],GREEN,3.1))
        g.add(label('5本の平均',4.78,1.2,16,GREEN),label('各誤差',4.55,.75,16,WHITE))
    if bi in (2,3):g.add(formula(r'E_x[\epsilon_m\epsilon_l]=0\quad(m\ne l)',size=25))
    elif bi in (4,5):g.add(formula(r'E_{\rm COM}=E_{\rm AV}/M=1/5',size=26))
    else:g.add(label('別の波の積は、正負が打ち消し合う',0,-2.40,18,YELLOW))
    return g,g[-3]

def correlation_panel(rho,bi):
    g=VGroup(note('共通する誤差を増やす'))
    ax,p=linegraph(ymin=-2.4,ymax=2.4,ylabel='誤差',x0=-4.65,x1=2.55)
    g.add(ax)
    xs=np.linspace(-1,1,120); er=orthogonal_errors(rho,xs)
    for j in range(5):g.add(curve([p(x,y) for x,y in zip(xs,er[:,j])],COLORS[j],1.5))
    g.add(curve([p(x,y) for x,y in zip(xs,er.mean(axis=1))],GREEN,3.1))
    g.add(label('個別平均',4.36,1.45,17,BLUE),label('1.00',4.36,1.1,19,BLUE),
          label('委員会',4.36,.45,17,GREEN),
          label(f'{orthogonal_expectation(rho):.2f}',4.36,.05,21,GREEN))
    sx=-3.0+6*rho
    if bi==3:
        g.add(formula(r'E_{\rm COM}/E_{\rm AV}=\rho+(1-\rho)/5',-2.38,24))
    else:
        g.add(Line([-3.0,-2.35,0],[3.0,-2.35,0],color=MUTED,stroke_width=2),
              Dot([sx,-2.35,0],radius=.12,color=YELLOW),
              label('0',-3.,-2.62,13,MUTED),label('1',3.,-2.62,13,MUTED),
              label(f'共通成分 ρ = {rho:.2f}',0,-1.99,18,YELLOW))
    return g

class PRML142Committees(NarratedScene):
    def correlated_errors(self):
        """Move one shared-variance knob while curves and the error readout follow it."""
        rho=ValueTracker(0.)
        ax,p=linegraph(ymin=-2.4,ymax=2.4,ylabel='誤差',x0=-4.65,x1=2.55)
        xs=np.linspace(-1,1,120)
        candidates=VGroup(*[
            always_redraw(lambda j=j:curve(
                [p(x,y) for x,y in zip(xs,orthogonal_errors(rho.get_value(),xs)[:,j])],
                COLORS[j],1.5)) for j in range(5)])
        mean=always_redraw(lambda:curve(
            [p(x,y) for x,y in zip(xs,orthogonal_errors(rho.get_value(),xs).mean(axis=1))],
            GREEN,3.1))
        committee=DecimalNumber(.2,num_decimal_places=2,font_size=23,color=GREEN).move_to([4.36,.05,0])
        committee.add_updater(lambda mob:mob.set_value(orthogonal_expectation(rho.get_value())))
        rho_number=DecimalNumber(0,num_decimal_places=2,font_size=19,color=YELLOW)
        rho_number.add_updater(lambda mob:mob.set_value(rho.get_value()))
        rho_readout=VGroup(jp('共通成分',17,YELLOW),tex(r'\rho=',19,YELLOW),rho_number)
        rho_readout.arrange(RIGHT,buff=.08).move_to([0,-1.99,0])
        pointer=always_redraw(lambda:Dot([-3.+6.*rho.get_value(),-2.35,0],radius=.12,color=YELLOW))
        slider=VGroup(Line([-3.,-2.35,0],[3.,-2.35,0],color=MUTED,stroke_width=2),
                      pointer,label('0',-3.,-2.62,13,MUTED),label('1',3.,-2.62,13,MUTED),
                      rho_readout)
        eq=formula(r'E_{\rm COM}/E_{\rm AV}=\rho+(1-\rho)/5',-2.38,24)
        self.add(note('共通する誤差を増やす'),ax,candidates,mean,
                 label('個別平均',4.36,1.45,17,BLUE),label('1.00',4.36,1.1,19,BLUE),
                 label('委員会',4.36,.45,17,GREEN),committee,slider)
        for bi,target in enumerate([0.,.2,.8,.8,1.,.8,.2,0.]):
            if bi==3:
                self.remove(slider)
                self.add(eq)
            elif bi==4:
                self.remove(eq)
                self.add(slider)
            duration=self.boundaries[bi]-(self.boundaries[bi-1] if bi else 0)
            enter=min(4.,duration*.55)
            if bi==0:
                first=lambda:Indicate(pointer,color=YELLOW,scale_factor=1.12)
            elif bi==3:
                first=lambda:Indicate(eq,color=YELLOW,scale_factor=1.02)
            else:
                first=lambda target=target:rho.animate.set_value(target)
            self.beat(phases=[('つまみと曲線',enter,first),
                              ('誤差を確認',duration-enter,
                               lambda:Indicate(committee,color=GREEN,scale_factor=1.04))])

    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            if si==6:
                self.correlated_errors()
                self.timeline[-1]['end']=float(self.time)
                continue
            old=None
            for bi,_ in enumerate(story['beats']):
                state,focus=(bootstrap_panel(bi) if si==2 else
                             variance_panel(bi) if si==4 else
                             orthogonal_panel(bi) if si==5 else
                             regression_panel(si,bi))
                duration=self.boundaries[bi]-(self.boundaries[bi-1] if bi else 0)
                enter=min(2.,duration*.35)
                if old is None:
                    transition=FadeIn(state,run_time=enter)
                else:
                    # Avoid fading two Japanese notes/formulas through each other.
                    self.remove(old)
                    self.add(state)
                    transition=Indicate(focus,color=focus.get_color(),scale_factor=1.035,run_time=enter)
                self.beat(phases=[('図を更新',enter,lambda a=transition:a),
                                  ('対応を追う',duration-enter,
                                   lambda f=focus:Indicate(f,color=f.get_color(),scale_factor=1.035))])
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml142_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
