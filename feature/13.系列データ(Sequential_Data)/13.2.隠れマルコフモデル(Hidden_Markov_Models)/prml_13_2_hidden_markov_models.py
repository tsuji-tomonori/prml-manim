"""PRML 13.2: a hidden weather state reconsidered by future observations."""
import json
from pathlib import Path
import numpy as np
from manim import *

from narration_content import SCENES
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST
from hmm_model import OBS, PATH_OBS, infer, em_step, viterbi

BG = '#111823'
BLUE = '#58C4DD'
YELLOW = '#FFFF00'
GREEN = '#83C167'
PURPLE = '#9A72AC'
RED = '#FC6255'
MUTED = '#A8B3C0'
DATA = infer(OBS)
PATH_DATA = infer(PATH_OBS)
EM = em_step(OBS)
VIT = viterbi(PATH_OBS)


def label(value, x, y, size=20, color=WHITE):
    return jp(value, size, color).move_to([x, y, 0])


def mathline(value, y=-2.45, size=27):
    mob = tex(value, size, YELLOW).move_to([0, y, 0])
    if mob.width > 11.8:
        mob.scale(11.8/mob.width)
    return mob


def weather_row(obs=OBS, upto=5, y=1.55, focus_day=1):
    group = VGroup()
    focus = None
    xs = [-5.25, -2.62, 0, 2.62, 5.25]
    for i,(x,wet) in enumerate(zip(xs,obs)):
        group.add(label(f'{i+1}日目',x,y+.70,16,MUTED))
        color = BLUE if wet else YELLOW
        visible = i < upto
        circle = Circle(radius=.30,stroke_color=color if visible else MUTED,
                        stroke_width=2,fill_color=color if visible else BG,
                        fill_opacity=.13 if visible else 0).move_to([x,y,0])
        group.add(circle,label(('雨' if wet else '晴') if visible else '？',
                               x,y,20,color if visible else MUTED))
        if i:
            group.add(Arrow([xs[i-1]+.37,y,0],[x-.37,y,0],
                            buff=0,color=MUTED,stroke_width=1.5,tip_length=.10))
        if i==focus_day:
            focus=circle
            group.add(Circle(radius=.39,stroke_color=PURPLE,stroke_width=2).move_to([x,y,0]))
    return group,focus


def probability_bars(wet, heading='2日目の隠れた状態', y=.05, axis_label='確率（%）'):
    group=VGroup(label(heading,0,.82,21))
    left,length=-1.65,4.55
    focus=None
    for i,(name,value,color) in enumerate((('乾き',1-wet,PURPLE),('湿り',wet,GREEN))):
        yy=y-i*.80
        group.add(label(name,-3.07,yy,18,color),
                  Line([left,yy,0],[left+length,yy,0],color=MUTED,stroke_width=1.25))
        bar=Line([left,yy,0],[left+length*value,yy,0],color=color,stroke_width=18)
        group.add(bar,label(f'{value*100:.1f}%',4.10,yy,18,color))
        if i==1:focus=bar
    base=y-1.55
    for v in (0,.5,1):
        x=left+length*v
        group.add(Line([x,base+.12,0],[x,base-.12,0],color=MUTED,stroke_width=1.1),
                  label(str(int(100*v)),x,base-.35,14,MUTED))
    group.add(Line([left,base+.12,0],[left+length,base+.12,0],color=MUTED,stroke_width=1.3),
              label(axis_label,5.15,base-.35,15,MUTED))
    return group,focus


def opening_panel(bi, ending=False):
    if ending:
        upto=[2,2,5,5,5,5,5][bi]
        wet=DATA['filter'][1,1] if bi<3 else DATA['gamma'][1,1]
    else:
        upto=[2,2,2,5,5,5,5][bi]
        wet=.5 if bi<2 else DATA['filter'][1,1] if bi<4 else DATA['gamma'][1,1]
    row,day=weather_row(upto=upto)
    bars,bar=probability_bars(wet)
    group=VGroup(row,bars)
    if bi in (2,3) and not ending:
        group.add(label('その日までの情報',0,-2.43,19,BLUE))
    elif bi>=4 or ending and bi>=3:
        group.add(label('全5日を見た情報' if wet<.5 else 'その日までの情報',0,-2.43,19,YELLOW))
    else:
        group.add(label('予想してから確かめる',0,-2.43,19,YELLOW))
    return group,bar if bi>=2 else day


def hidden_chain(bi):
    group=VGroup()
    xs=[-5.15,-2.58,0,2.58,5.15]
    group.add(label('隠れた状態',-5.35,2.60,18,PURPLE),
              label('観測した天気',-5.35,-1.30,18,BLUE))
    for i,x in enumerate(xs):
        state=Circle(radius=.30,color=PURPLE,stroke_width=2,
                     fill_color=PURPLE,fill_opacity=.15).move_to([x,1.23,0])
        obs=Circle(radius=.28,color=BLUE,stroke_width=2,
                   fill_color=BLUE,fill_opacity=.13).move_to([x,-.56,0])
        group.add(state,tex(f'z_{{{i+1}}}',21,PURPLE).move_to([x,1.23,0]),
                  obs,label('雨' if OBS[i] else '晴',x,-.56,18,BLUE if OBS[i] else YELLOW),
                  Arrow([x,.88,0],[x,-.19,0],buff=0,color=BLUE,stroke_width=2,tip_length=.11))
        if i:
            group.add(Arrow([xs[i-1]+.36,1.23,0],[x-.36,1.23,0],buff=0,
                            color=YELLOW,stroke_width=2,tip_length=.11))
    focus=group[-1]
    if bi==1:
        group.add(label('残る 0.9',-1.30,1.78,17,YELLOW),label('移る 0.1',1.20,1.78,17,YELLOW))
    elif bi==2:
        group.add(label('乾き → 雨 0.2',-3.45,-1.85,17,BLUE),
                  label('湿り → 雨 0.8',3.20,-1.85,17,GREEN))
    elif bi==3:
        group.add(mathline(r'\pi,\quad A_{jk}=p(z_n=k\mid z_{n-1}=j),\quad p(x_n\mid z_n)',-2.35,24))
    elif bi==4:
        group.add(mathline(r'p(z_1)\prod_{n=2}^N p(z_n\mid z_{n-1})\prod_{n=1}^N p(x_n\mid z_n)',-2.35,24))
    elif bi>=5:
        group.add(label('1日の断面は混合、横の矢印が時系列を結ぶ',0,-2.35,20,GREEN))
    return group,focus


def forward_panel(bi):
    upto=1 if bi<=0 else 2
    row,focus=weather_row(upto=upto)
    wet=[.5,.26,DATA['filter'][1,1],DATA['filter'][1,1],
         DATA['filter'][1,1],DATA['filter'][1,1],DATA['filter'][1,1]][bi]
    if bi==-1:
        wet=.5
    elif bi==0:
        wet=.2
    bars,bar=probability_bars(wet,'左から進む状態の確率')
    group=VGroup(row,bars)
    if bi==1:
        group.add(label('晴の後 → 乾き 0.74、湿り 0.26',0,-2.43,18,YELLOW))
    elif bi==3:
        group.add(mathline(r'\alpha_n(k)=p(x_{1:n},z_n=k)',-2.43,27))
    elif bi==4:
        group.add(mathline(r'\alpha_n(k)=p(x_n\mid k)\sum_j\alpha_{n-1}(j)A_{jk}',-2.43,25))
    elif bi==5:
        group.add(mathline(r'p(z_n=k\mid x_{1:n})=\alpha_n(k)/\sum_l\alpha_n(l)',-2.43,25))
    else:
        group.add(label('前向き: 遷移 → 観測 → 合計を1へ',0,-2.43,18,YELLOW))
    return group,bar if bi>=2 else focus


def recap_sumproduct():
    group=VGroup()
    heading=label('復習: 8.4 積和アルゴリズム',-3.98,2.34,20)
    group.add(heading,SurroundingRectangle(heading,color=YELLOW,stroke_width=1.4,buff=.10))
    left=[[-3.8,.95,0],[-3.8,-.25,0]]
    right=[.40,.35,0]
    for point,color,word in zip(left,(BLUE,PURPLE),('道 1','道 2')):
        group.add(Dot(point,color=color,radius=.16),label(word,point[0]-.80,point[1],18,color),
                  Arrow(np.array(point)+RIGHT*.28,np.array(right)+LEFT*.30,buff=0,
                        color=color,stroke_width=3,tip_length=.14))
    group.add(Dot(right,color=GREEN,radius=.20),label('合流',1.35,.35,20,GREEN),
              mathline(r'\sum_j\alpha_{n-1}(j)A_{jk}',-1.73,30),
              label('各道で掛ける → 同じ終点で足す',0,-2.36,18,YELLOW))
    return group,group[-3]


def backward_panel(bi):
    if bi==0:return recap_sumproduct()
    upto=5
    row,focus=weather_row(upto=upto)
    if bi<6:
        back_index={1:4,2:3,3:2,4:1,5:1}[bi]
        weights=DATA['backward'][back_index]
        wet=float(weights[1]/weights.sum())
        bars,bar=probability_bars(wet,'右から届く重みの比較',axis_label='相対重み（%）')
    else:
        wet=float(DATA['gamma'][1,1])
        bars,bar=probability_bars(wet,'前後を合わせた状態確率')
    group=VGroup(row,bars)
    if bi in (2,3,4):
        xs=[-5.25,-2.62,0,2.62,5.25]
        group.add(Arrow([xs[back_index+1]-.40,2.66,0],[xs[back_index]+.40,2.66,0],
                        buff=0,color=YELLOW,stroke_width=2,tip_length=.13))
    if bi==1:
        group.add(label('右端の後向き重みは、両状態で 1',0,-2.43,18,YELLOW))
    elif bi==3:
        group.add(mathline(r'\beta_n(k)=p(x_{n+1:N}\mid z_n=k)',-2.43,27))
    elif bi==5:
        group.add(label('後の晴れ3日分が、2日目へ届いた',0,-2.43,18,YELLOW))
    elif bi==6:
        group.add(mathline(r'\gamma_n(k)\propto\alpha_n(k)\beta_n(k)',-2.43,26))
    else:
        group.add(label('後向き: 未来の観測を順に取り込む',0,-2.43,18,YELLOW))
    return group,bar


def lattice(obs=OBS, phase=5, path=None):
    group=VGroup()
    xs=[-4.75,-2.38,0,2.38,4.75]
    ys=[.85,-.62]
    group.add(label('乾き',-5.90,ys[0],17,PURPLE),label('湿り',-5.90,ys[1],17,GREEN))
    for i,x in enumerate(xs):
        group.add(label(f'{i+1}日目',x,2.15,15,MUTED),
                  label('雨' if obs[i] else '晴',x,1.57,17,BLUE if obs[i] else YELLOW))
        for k,y in enumerate(ys):
            node=Circle(radius=.18,stroke_color=PURPLE if k==0 else GREEN,
                        fill_color=PURPLE if k==0 else GREEN,
                        fill_opacity=.18,stroke_width=2).move_to([x,y,0])
            group.add(node)
        if 0<i<phase:
            for j in range(2):
                for k in range(2):
                    group.add(Line([xs[i-1]+.20,ys[j],0],[x-.20,ys[k],0],
                                   color=MUTED,stroke_width=.9,stroke_opacity=.45))
    if path is not None:
        color=GREEN if path==VIT['path'] else BLUE
        for i,k in enumerate(path):
            group.add(Circle(radius=.27,stroke_color=color,stroke_width=2.7).move_to([xs[i],ys[k],0]))
            if i:
                group.add(Line([xs[i-1]+.27,ys[path[i-1]],0],[xs[i]-.27,ys[k],0],
                               color=color,stroke_width=3.5))
    return group


def likelihood_panel(bi):
    phase=min(5,max(1,bi+1))
    lat=lattice(phase=phase)
    group=VGroup(lat)
    count=2**phase
    group.add(label(f'候補の経路: {count}本',0,-1.55,20,YELLOW))
    if bi>=3:
        group.add(mathline(r'p(X)=\sum_k\alpha_N(k)=0.035282',-2.34,26))
    else:
        group.add(label('同じ終点への経路を、その場で足す',0,-2.34,19,GREEN))
    return group,lat[-1]


def em_panel(bi):
    row,day=weather_row()
    group=VGroup(row)
    if bi<3:
        bars,focus=probability_bars(DATA['gamma'][1,1],'2日目の期待度数')
        group.add(bars)
        if bi==2:
            group.add(mathline(r'\gamma_n(k),\quad\xi_n(j,k)',-2.43,27))
        else:
            group.add(label('断定の代わりに事後の重みを数える',0,-2.43,18,YELLOW))
        return group,focus
    old=.9
    new=float(EM['a'][0,0])
    shown=old if bi<5 else new
    group.add(label('乾きからの遷移確率',0,.95,20),
              label('乾きへ残る',-3.80,.34,18,PURPLE),
              label('湿りへ移る',-3.80,-.56,18,GREEN))
    left,length=-2.10,4.8
    for i,(value,color) in enumerate(((shown,PURPLE),(1-shown,GREEN))):
        y=.34-i*.9
        group.add(Line([left,y,0],[left+length,y,0],color=MUTED,stroke_width=1.3),
                  Line([left,y,0],[left+length*value,y,0],color=color,stroke_width=17),
                  label(f'{value:.3f}',3.72,y,17,color))
    for v in (0,.5,1):
        x=left+length*v
        group.add(Line([x,-1.36,0],[x,-1.58,0],color=MUTED,stroke_width=1),
                  label(str(v),x,-1.83,14,MUTED))
    group.add(Line([left,-1.36,0],[left+length,-1.36,0],color=MUTED,stroke_width=1))
    if bi==4:
        group.add(mathline(r'A_{jk}^{new}=\frac{\sum_n\xi_n(j,k)}{\sum_{n,l}\xi_n(j,l)}',-2.40,24))
    elif bi>=5:
        group.add(label('観測列の確率: 0.035282 → 0.081029',0,-2.43,18,YELLOW))
    else:
        group.add(label('期待回数: 乾→乾 3.476 ／ 乾→湿 0.096',0,-2.43,18,YELLOW))
    return group,group[-1]


def viterbi_panel(bi):
    local=tuple(np.argmax(PATH_DATA['gamma'],axis=1))
    path=None if bi==0 else local if bi<4 else VIT['path']
    lat=lattice(PATH_OBS,phase=5,path=path)
    group=VGroup(lat)
    if bi==1:
        group.add(label('各日の最多: 最後は湿り 55.4%',0,-2.25,19,BLUE))
    elif bi>=4:
        group.add(label('列全体の最尤: 5日とも乾き',0,-2.25,19,GREEN))
    elif bi==3:
        group.add(mathline(r'\omega_n(k)=\log p(x_n\mid k)+\max_j[\omega_{n-1}(j)+\log A_{jk}]',-2.25,31))
    else:
        group.add(label('各状態へ来た最大の道だけ残す',0,-2.25,19,YELLOW))
    return group,lat[-1]


def panel(si,bi):
    if si==0:return opening_panel(bi)
    if si==1:return hidden_chain(bi)
    if si==2:return forward_panel(bi)
    if si==3:return backward_panel(bi)
    if si==4:return likelihood_panel(bi)
    if si==5:return em_panel(bi)
    if si==6:return viterbi_panel(bi)
    return opening_panel(bi,ending=True)


class PRML132HiddenMarkovModels(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            old=None
            for bi,_ in enumerate(story['beats']):
                state,focus=panel(si,bi)
                if si==2 and bi==0:
                    initial,_=forward_panel(-1)
                    self.add(initial)
                    duration=self.boundaries[bi]
                    first=self.sentence_duration(0)
                    self.beat(phases=[('半分ずつ',first,lambda:Wait()),
                                      ('晴れで更新',duration-first,
                                       lambda a=initial,b=state:
                                       Succession(FadeOut(a,run_time=.15),FadeIn(b,run_time=.20),Wait(.20)))])
                    old=state
                    continue
                transition=(FadeIn(state) if old is None else
                            Succession(FadeOut(old,run_time=.20),FadeIn(state,run_time=.20),Wait(.40)))
                duration=self.boundaries[bi]-(self.boundaries[bi-1] if bi else 0)
                switch=min(1.0,duration*.18)
                self.beat(phases=[('表示切替',switch,lambda a=transition:a),
                                  ('注目箇所',duration-switch,
                                   lambda f=focus,c=(focus.get_color() if isinstance(focus,Line) and si in (0,2,3,5,7) else YELLOW):Indicate(f,color=c,scale_factor=1.025))])
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml132_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
