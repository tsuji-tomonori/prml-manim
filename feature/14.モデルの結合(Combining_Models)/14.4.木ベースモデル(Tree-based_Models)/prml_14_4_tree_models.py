"""PRML 14.4: one 4x4 example through greedy growth and pruning."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST
from tree_model import X, T, QUERY, grow, predict, cost

BG='#111823'; BLUE='#58C4DD'; YELLOW='#FFFF00'; GREEN='#83C167'
PURPLE='#9A72AC'; RED='#FC6255'; MUTED='#A8B3C0'
STAGES=grow()

def lab(s,x,y,size=18,color=WHITE):
    return jp(s,size,color).move_to([x,y,0])

def eq(s,x=3.0,y=-2.37,size=26):
    m=tex(s,size,YELLOW).move_to([x,y,0])
    return m.scale(min(1,6.7/m.width))

def px(x):return -5.27+1.05*x
def py(y):return -1.85+.93*y

def stage_for(si,bi):
    if si==0:return 4 if bi>=4 else 0
    if si==1:return 1 if bi>=3 else 0
    if si==2:return 2 if bi>=2 else 1
    if si==3:return 2
    if si==4:return 3 if bi<2 else 4
    if si==5:return 2 if bi>=5 else 4
    if si==6:return 2
    if si==7:return 2
    return 4 if bi<2 else 2

def plot(si,bi,stage):
    g=VGroup()
    ax=VGroup(Line([px(0),py(0),0],[px(4.15),py(0),0],color=MUTED,stroke_width=1.7),
              Line([px(0),py(0),0],[px(0),py(4.1),0],color=MUTED,stroke_width=1.7))
    g.add(ax)
    for v in (0,2,4):
        g.add(Line([px(v),py(0)-.055,0],[px(v),py(0)+.055,0],color=MUTED,stroke_width=1.1),
              lab(str(v),px(v),py(0)-.24,13,MUTED))
        g.add(Line([px(0)-.055,py(v),0],[px(0)+.055,py(v),0],color=MUTED,stroke_width=1.1),
              lab(str(v),px(0)-.24,py(v),13,MUTED))
    g.add(lab('位置',-.62,-1.86,16,MUTED),tex('x_1',21,MUTED).move_to([-.24,-1.86,0]),
          lab('高さ',-5.18,2.23,16,MUTED),tex('x_2',21,MUTED).move_to([-4.75,2.23,0]))
    if si==6:g.add(lab('分類例: 8以上=赤',-2.89,2.23,15,YELLOW))
    # These lines are data partitions, not graph decoration.
    splits=[(0,2,0,4),(1,2,0,2),(1,3,2,4),(0,3,3,4)]
    boundaries=[]
    for index,(axis,t,lo,hi) in enumerate(splits[:stage]):
        if axis==0:line=Line([px(t),py(lo),0],[px(t),py(hi),0],color=(YELLOW,GREEN,PURPLE,RED)[index],stroke_width=3.0)
        else:line=Line([px(lo),py(t),0],[px(hi),py(t),0],color=(YELLOW,GREEN,PURPLE,RED)[index],stroke_width=3.0)
        boundaries.append(line);g.add(line)
    if si==1 and bi==2:
        line=Line([px(0),py(2),0],[px(4),py(2),0],color=PURPLE,stroke_width=2.7)
        boundaries.append(line);g.add(line)
    dots=[]
    for (x,y),v in zip(X,T):
        color=(RED if v>=8 else BLUE) if si==6 else (BLUE if v==2 else PURPLE if v==6 else GREEN if v==8 else RED)
        dot=Dot([px(x),py(y),0],radius=.083,color=color)
        dots.append(dot);g.add(dot)
        # Values sit above points, with room between the 1.05-unit columns.
        label_x,label_y=(px(x)+.04,py(y)+.38) if v==14 else (px(x)+.17,py(y)+.15)
        g.add(lab(str(int(v)),label_x,label_y,14,color))
    query=Circle(radius=.22,color=YELLOW,stroke_width=2.2).move_to([px(3.5),py(3.5),0])
    if si in (0,4,5,8) or (si==7 and bi==3):g.add(query)
    return g,(boundaries[-1] if boundaries else query),query

def small_tree(stage):
    g=VGroup()
    root=lab('位置 ≤ 2 ?',3.16,1.34,19,YELLOW)
    g.add(root)
    if stage==0:
        g.add(lab('全16点 → 平均 5.875',3.13,.35,18,GREEN))
        return g,root
    left=lab('左',1.72,.56,17,BLUE);right=lab('右',4.73,.56,17,GREEN)
    g.add(Line([2.62,1.12,0],[1.72,.72,0],color=MUTED,stroke_width=1.6),
          Line([3.68,1.12,0],[4.73,.72,0],color=MUTED,stroke_width=1.6),left,right)
    if stage==1:
        g.add(lab('平均 4',1.72,-.23,17,PURPLE),lab('平均 8.75',4.73,-.23,17,GREEN))
        return g,root
    g.add(lab('高さ ≤ 2 ?',1.72,-.16,16,YELLOW),
          Line([1.54,-.36,0],[1.10,-.85,0],color=MUTED,stroke_width=1.5),
          Line([1.90,-.36,0],[2.31,-.85,0],color=MUTED,stroke_width=1.5),
          lab('2',1.10,-1.05,20,BLUE),lab('6',2.31,-1.05,20,PURPLE))
    if stage==2:
        g.add(lab('8.75',4.73,-.22,22,GREEN))
        return g,root
    g.add(lab('高さ ≤ 3 ?',4.73,-.17,16,YELLOW),
          Line([4.45,-.37,0],[4.13,-.83,0],color=MUTED,stroke_width=1.5),
          Line([4.98,-.37,0],[5.39,-.83,0],color=MUTED,stroke_width=1.5),
          lab('8',4.13,-1.05,20,GREEN))
    if stage==3:
        g.add(lab('11',5.39,-1.05,20,RED))
        return g,root
    g.add(lab('位置 ≤ 3 ?',5.39,-1.03,14,YELLOW),
          Line([5.18,-1.20,0],[4.93,-1.51,0],color=MUTED,stroke_width=1.2),
          Line([5.60,-1.20,0],[5.91,-1.51,0],color=MUTED,stroke_width=1.2),
          lab('8',4.89,-1.70,18,GREEN),lab('14',5.91,-1.70,18,RED))
    return g,root

def mini_impurity():
    g=VGroup();ox=2.05;oy=-1.28;w=3.8;h=2.58
    x=lambda p:ox+w*p
    y=lambda v:oy+h*v/.75
    g.add(Line([ox,oy,0],[ox+w,oy,0],color=MUTED,stroke_width=1.4),
          Line([ox,oy,0],[ox,oy+h,0],color=MUTED,stroke_width=1.4),
          lab('青の割合',5.04,-1.73,15,MUTED),lab('混ざり具合',2.69,1.58,15,MUTED))
    for p in (0,.5,1):
        g.add(Line([x(p),oy-.055,0],[x(p),oy+.055,0],color=MUTED,stroke_width=1),
              lab(str(p).rstrip('0').rstrip('.') if p else '0',x(p),oy-.22,12,MUTED))
    for v in (0,.5):
        g.add(Line([ox-.055,y(v),0],[ox+.055,y(v),0],color=MUTED,stroke_width=1),
              lab(str(v).rstrip('0').rstrip('.') if v else '0',ox-.24,y(v),12,MUTED))
    ps=np.linspace(.001,.999,80)
    entropy=-ps*np.log(ps)-(1-ps)*np.log(1-ps)
    gini=2*ps*(1-ps)
    curve1=VMobject(stroke_color=YELLOW,stroke_width=2.6).set_points_smoothly([[x(p),y(v),0] for p,v in zip(ps,entropy)])
    curve2=VMobject(stroke_color=GREEN,stroke_width=2.6).set_points_smoothly([[x(p),y(v),0] for p,v in zip(ps,gini)])
    g.add(curve1,curve2,lab('交差エントロピー',4.20,1.67,13,YELLOW),lab('ジニ指数',4.8,.48,13,GREEN))
    return g,curve1

def right_side(si,bi,stage):
    g=VGroup();focus=None
    if si==6:
        panel,focus=mini_impurity();g.add(panel)
    elif si==5:
        lam=0 if bi<2 else 18
        c3=cost(STAGES[2][0],lam);c5=cost(STAGES[4][0],lam)
        burden_label=lab('葉の負担  λ =',3.38,1.62,20,YELLOW)
        burden_number=DecimalNumber(lam,num_decimal_places=0,color=YELLOW,font_size=23).next_to(burden_label,RIGHT,buff=.09)
        g.add(VGroup(burden_label,burden_number))
        g.add(Line([2.35,1.08,0],[5.20,1.08,0],color=MUTED,stroke_width=2),
              Dot([2.35+2.85*lam/18,1.08,0],radius=.12,color=YELLOW),
              lab('0',2.35,.87,12,MUTED),lab('18',5.20,.87,12,MUTED))
        for j,(label,value,color) in enumerate((('3葉',c3,GREEN),('5葉',c5,PURPLE))):
            yy=.48-j*.91
            tip=2.35+2.75*value/100
            places=1 if j==0 else 0
            number=DecimalNumber(value,num_decimal_places=places,color=color,font_size=24).move_to([tip+.30,yy,0])
            g.add(lab(label,1.80,yy,18,color),number)
            bar=Line([2.35,yy,0],[2.35+2.75*value/100,yy,0],color=color,stroke_width=16)
            g.add(bar);focus=bar
        g.add(Line([2.35,-1.02,0],[5.10,-1.02,0],color=MUTED,stroke_width=1.1),
              lab('費用',1.81,-1.02,13,MUTED))
        for value in (0,50,100):
            xx=2.35+2.75*value/100
            g.add(Line([xx,-1.08,0],[xx,-.96,0],color=MUTED,stroke_width=1),
                  lab(str(value),xx,-1.25,12,MUTED))
        g.add(lab('費用 = 残差 + 葉の負担',3.65,-1.60,16,MUTED))
    elif si==3:
        g.add(lab('右の葉: 8, 8, 8, 8',3.58,1.54,17,GREEN),lab('8, 8, 8, 14',3.65,1.07,17,RED),
              lab('平均 8.75',3.52,.40,23,YELLOW))
        baseline=Line([1.7,-1.05,0],[5.9,-1.05,0],color=MUTED,stroke_width=1.2);g.add(baseline)
        g.add(Line([1.7,-1.05,0],[1.7,.21,0],color=MUTED,stroke_width=1.1),
              lab('値',1.57,.42,13,MUTED),lab('観測点',3.72,-1.65,14,MUTED))
        for value in (8,14):
            yy=-1.05+(value-7)*.16
            g.add(Line([1.64,yy,0],[1.76,yy,0],color=MUTED,stroke_width=1),
                  lab(str(value),1.45,yy,12,MUTED))
        for index in (1,4,8):
            xx=1.9+(index-1)*.52
            g.add(Line([xx,-1.11,0],[xx,-.99,0],color=MUTED,stroke_width=1),
                  lab(str(index),xx,-1.31,12,MUTED))
        for i,v in enumerate([8]*7+[14]):
            x=1.9+i*.52;h=(v-7)*.16
            g.add(Line([x,-1.05,0],[x,-1.05+h,0],color=RED if v==14 else GREEN,stroke_width=10))
        mean_y=-1.05+(8.75-7)*.16
        g.add(Line([1.75,mean_y,0],[5.95,mean_y,0],color=YELLOW,stroke_width=1.6),
              Square(side_length=.12,color=YELLOW,stroke_width=1.5).move_to([2.04,mean_y-.06,0]),
              Square(side_length=.84,color=RED,stroke_width=1.6).move_to([5.92,mean_y+.42,0]),
              lab('ズレの二乗',5.55,.35,14,RED))
        focus=g[-2]
    elif si in (0,8):
        full=14;pruned=8.75
        g.add(lab('同じ入力の次回観測',3.60,1.65,18,YELLOW))
        if si==0 and bi<4:
            g.add(lab('観測値は？',3.55,1.12,22,BLUE),lab('予測は？',3.55,.18,26,YELLOW));focus=g[-1]
            if bi==0:g.add(lab('木ベースモデル',3.55,-1.27,18,GREEN))
        else:
            g.add(lab('真値 8',3.52,1.12,22,BLUE))
            g.add(lab(f'予測 {full:g}' if stage==4 else f'予測 {pruned:g}',3.56,.22,24,RED if stage==4 else GREEN),
                  lab('ズレ 6' if stage==4 else 'ズレ 0.75',3.56,-.58,22,RED if stage==4 else GREEN))
            focus=g[-1]
    else:
        panel,focus=small_tree(stage);g.add(panel)
    return g,focus

FORMULAS={
    (1,4):r'Q_0=153.75\quad\longrightarrow\quad Q_1=63.5',
    (2,4):r'y_R=\frac{7\cdot8+14}{8}=8.75',
    (3,3):r'y_\tau=\frac{1}{N_\tau}\sum_{x_n\in R_\tau}t_n',
    (3,4):r'Q_\tau=\sum_{x_n\in R_\tau}(t_n-y_\tau)^2',
    (4,3):r'Q(T_5)=0',
    (5,1):r'C(T)=\sum_\tau Q_\tau(T)+\lambda|T|',
    (5,3):r'C(T_5)=0+18\cdot5=90',
    (5,4):r'C(T_3)=31.5+18\cdot3=85.5',
    (6,2):r'H=-\sum_k p_k\ln p_k',
    (6,3):r'G=\sum_k p_k(1-p_k)',
    (8,1):r'|14-8|=6',
    (8,2):r'|8.75-8|=0.75',
}

def make_state(si,bi):
    stage=stage_for(si,bi)
    state=VGroup()
    note=SCENES[si]['beats'][bi]['visual_note']
    if (si,bi)==(5,0):note='復習: 1.1 正則化'
    if (si,bi)==(6,1):note='復習: 1.6 エントロピー'
    heading=lab(note,0,2.63,19,YELLOW)
    state.add(heading)
    if (si,bi) in ((5,0),(6,1)):
        state.add(SurroundingRectangle(heading,color=YELLOW,stroke_width=1,buff=.10))
    graph,split,query=plot(si,bi,stage);state.add(graph)
    side,focus=right_side(si,bi,stage);state.add(side)
    if (si,bi)==(7,3):focus=query
    if (si,bi) in FORMULAS:state.add(eq(FORMULAS[(si,bi)]))
    elif si==0 and bi<4:state.add(eq(r'x=(3.5,3.5)\quad t^{\prime}=?',x=2.9))
    elif si in (0,4,8):state.add(eq(r'x=(3.5,3.5)\quad t^{\prime}=8',x=2.9))
    elif si==2:state.add(lab('3領域  ↔  3葉',3.0,-2.37,22,YELLOW))
    elif si==7 and bi==4:state.add(lab('分類回帰木（CART）',3.0,-2.37,21,YELLOW))
    elif si==7:state.add(eq(r'x_1\le2\quad\longrightarrow\quad x_2\le2'))
    return state,focus if focus is not None else split

class PRML144TreeModels(NarratedScene):
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
                entrance=min(1.7,duration*.29)
                if old is None:
                    self.beat(phases=[('図を導入',entrance,lambda a=FadeIn(state):a),
                                      ('注目する量',duration-entrance,lambda f=focus:Indicate(f,color=f.get_color(),scale_factor=1.025))])
                elif si==5 and bi==2:
                    # Let the spoken first sentence finish at lambda=0, then move
                    # the knob and both cost bars during the sentence about 18.
                    first=min(duration-2.1,max(1.,self.beat_cues()[0]['end']))
                    def move_costs(a=old,b=state):
                        def change_cost(m,start,finish,y):
                            return UpdateFromAlphaFunc(m,lambda obj,alpha:
                                (obj.set_value(start+(finish-start)*alpha),
                                 obj.move_to([2.35+2.75*(start+(finish-start)*alpha)/100+.30,y,0])))
                        return AnimationGroup(
                            Transform(a[2][2],b[2][2].copy()),
                            Transform(a[2][7],b[2][7].copy()),
                            Transform(a[2][10],b[2][10].copy()),
                            ChangeDecimalToValue(a[2][0][1],18),
                            change_cost(a[2][6],31.5,85.5,.48),
                            change_cost(a[2][9],0,90,-.43),
                            Succession(FadeOut(a[0]),FadeIn(b[0])),
                            FadeOut(a[3]),lag_ratio=0)
                    self.beat(phases=[('0の費用を確認',first,lambda a=old:Indicate(a[2][2],color=YELLOW,scale_factor=1.1)),
                                      ('つまみと費用を動かす',duration-first,move_costs)])
                else:
                    def update_figure(a=old,b=state,seconds=entrance):
                        def parts(s):
                            plot_index=2 if isinstance(s[1],SurroundingRectangle) else 1
                            rest=VGroup(*s[1:plot_index],*s[plot_index+1:])
                            return s[plot_index],rest
                        def swap(old_part,new_part):
                            return Succession(FadeOut(old_part,run_time=.12),
                                              FadeIn(new_part,run_time=.28),
                                              Wait(max(.01,seconds-.40)))
                        old_plot,old_rest=parts(a);new_plot,new_rest=parts(b)
                        return AnimationGroup(swap(a[0],b[0]),swap(old_rest,new_rest),
                            AnimationGroup(FadeOut(old_plot),FadeIn(new_plot),lag_ratio=0,run_time=seconds),lag_ratio=0)
                    self.beat(phases=[('図を更新',entrance,update_figure),
                                      ('注目する量',duration-entrance,lambda f=focus:Indicate(f,color=f.get_color(),scale_factor=1.025))])
                self.clear();self.add(self.main_title,state,self.subtitle)
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml144_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
