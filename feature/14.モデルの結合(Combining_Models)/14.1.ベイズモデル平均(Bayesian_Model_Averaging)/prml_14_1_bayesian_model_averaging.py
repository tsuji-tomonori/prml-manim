"""PRML 14.1: four observed colours reveal where the model selector sits."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST
from model_average import OBSERVATIONS, P_RED, posterior, predictive_red

BG='#111823'; BLUE='#58C4DD'; YELLOW='#FFFF00'; GREEN='#83C167'
PURPLE='#9A72AC'; RED='#FC6255'; MUTED='#A8B3C0'

def label(s,x,y,size=19,color=WHITE):
    return jp(s,size,color).move_to([x,y,0])

def formula(s,y=-2.43,size=26):
    m=tex(s,size,YELLOW).move_to([0,y,0])
    if m.width>11.6:m.scale(11.6/m.width)
    return m

def note(s):return label(s,0,2.53,21,YELLOW)

def recap(s):
    heading=label(s,-4.45,2.51,19,WHITE)
    return VGroup(heading,SurroundingRectangle(heading,color=YELLOW,stroke_width=1.2,buff=.11))

def balls(y=1.15,xs=(-2.5,-.85,.8,2.45),r=.22):
    g=VGroup(); dots=[]
    for i,(x,c) in enumerate(zip(xs,OBSERVATIONS)):
        dot=Dot([x,y,0],radius=r,color=RED if c=='赤' else BLUE)
        dots.append(dot);g.add(dot,label(str(i+1),x,y-.43,14,MUTED))
    return g,dots

def card(h,x,y,prob=True):
    color=BLUE if h=='A' else PURPLE
    g=VGroup(label(h,x,y+.32,27,color))
    if prob:
        p=float(P_RED[h]); start=x-.58; width=1.16
        g.add(Rectangle(width=width*p,height=.16,stroke_width=0,fill_color=RED,fill_opacity=.9).move_to([start+width*p/2,y-.04,0]),
              Rectangle(width=width*(1-p),height=.16,stroke_width=0,fill_color=BLUE,fill_opacity=.9).move_to([start+width*(1+p)/2,y-.04,0]),
              label(f'赤 {p:.0%}',x,y-.37,15,color))
    return g

def prob_axis(y,x0=-1.6,x1=3.15):
    g=VGroup(Line([x0,y,0],[x1,y,0],color=MUTED,stroke_width=1.5))
    for v in (0,.5,1):
        x=x0+(x1-x0)*v
        g.add(Line([x,y-.07,0],[x,y+.07,0],color=MUTED,stroke_width=1),label(f'{v:g}',x,y-.29,13,MUTED))
    g.add(label('赤の確率',5.16,y-.29,14,MUTED))
    return g

def probability_bar(name,value,y,color,show_axis=False):
    x0=-1.6; span=4.75
    g=VGroup(label(name,-3.45,y,19,color),Rectangle(width=max(.005,span*value),height=.22,
        stroke_width=0,fill_color=color,fill_opacity=.9).move_to([x0+span*value/2,y,0]),
        label(f'{value*100:.1f}%',4.08,y,19,color))
    if show_axis:g.add(prob_axis(y-.39))
    return g

def opening_panel(bi,ending=False):
    g=VGroup();g.add(note((['同じ4個から、2つの予測？','次の1個を予想','同じ2枚の候補',
     '1個ごとなら 50%','箱全体なら 76.5%','選択はいつ？','選択を置く場所が鍵'] if not ending else
     ['同じ赤赤赤青','2枚の候補も同じ','4つの選択','1つの選択','モデルの結合 ／ ベイズモデル平均化','選択を置く単位','次はコミッティ'])[bi]))
    obs,dots=balls(y=1.20);g.add(obs)
    if bi>=2 or ending:g.add(card('A',-4.65,.25),card('B',4.65,.25))
    if ending or bi>=3:
        g.add(probability_bar('点ごと',.5,-.69,BLUE))
    if ending or bi>=4:
        g.add(probability_bar('集合全体',float(predictive_red()),-1.48,GREEN,True))
    if ending and bi in (2,3):
        if bi==2:
            for x in (-2.5,-.85,.8,2.45):g.add(Arrow([x,1.92,0],[x,1.49,0],buff=0,color=BLUE,stroke_width=2,tip_length=.1))
        else:g.add(Arrow([0,2.03,0],[0,1.53,0],buff=0,color=YELLOW,stroke_width=2,tip_length=.13))
    if ending or bi>=5:g.add(label('同じ観測、異なる選択の仮定',0,-2.59,19,YELLOW))
    return g,dots[min(bi//2,3)]

def local_panel(bi):
    g=VGroup(recap('復習: 9.2 点ごとの成分') if bi==0 else note(['復習: 9.2 点ごとの成分','2枚の確率カード','各点に選択 zₙ','選択は別々','赤は 40% + 10%','次も選び直す','モデルの結合'][bi]))
    xs=(-4.2,-1.4,1.4,4.2)
    obs,dots=balls(y=-.35,xs=xs);g.add(obs)
    choices=(['A','B','A','B'] if bi in (3,4) else ['A','A','B','B'])
    if bi==0:
        # Shrunk reproduction of the two component peaks from §9.2.
        for mu,sigma,color in ((-1.1,.9,RED),(1.1,1.2,BLUE)):
            x=np.linspace(mu-2.5,mu+2.5,70)
            g.add(VMobject(stroke_color=color,stroke_width=2.7).set_points_smoothly(
                [[v,-1.70+2.1*np.exp(-.5*((v-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)),0] for v in x]))
        g.add(label('9.2 混合ガウス',0,-2.32,18,YELLOW))
    else:
        for i,x in enumerate(xs):
            h=choices[i]; c=BLUE if h=='A' else PURPLE
            g.add(label(h,x,1.28,23,c),Arrow([x,.95,0],[x,.06,0],buff=0,color=c,stroke_width=2,tip_length=.12))
        if bi==1:g.add(card('A',-2.0,-1.75),card('B',2.0,-1.75))
        elif bi==2:g.add(formula(r'z_1,z_2,z_3,z_4',-2.37,28))
        elif bi==4:g.add(label('0.5 × 0.8   +   0.5 × 0.2   =   0.5',0,-2.35,20,YELLOW))
        elif bi==5:g.add(label('次の選択も A / B は半分ずつ',0,-2.35,20,YELLOW))
        else:g.add(label('1個につき1つの選択',0,-2.35,20,YELLOW))
    return g,dots[min(bi//2,3)]

def branch_panel(bi):
    g=VGroup(note(['どのカードから赤？','2本の道筋','40% と 10%','周辺化で足す','見えない z を足す','原典はガウスの山','4個へ広げる'][bi]))
    g.add(card('A',-3.2,1.30),card('B',3.2,1.30))
    red=Dot([0,-.55,0],radius=.27,color=RED);g.add(red,label('赤',0,-1.12,19,RED))
    left=Arrow([-2.8,.55,0],[-.47,-.33,0],buff=0,color=BLUE,stroke_width=3,tip_length=.16)
    right=Arrow([2.8,.55,0],[.47,-.33,0],buff=0,color=PURPLE,stroke_width=3,tip_length=.16)
    g.add(left,right)
    if bi>=2:g.add(label('0.5 × 0.8 = 0.40',-3.05,-.54,17,BLUE),label('0.5 × 0.2 = 0.10',3.08,-.54,17,PURPLE))
    if bi in (3,4):g.add(formula(r'p(x)=\sum_z p(x,z)',-2.37,28))
    elif bi==5:g.add(formula(r'p(x)=\sum_k\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k)',-2.37,26))
    else:g.add(label('2本の道が、同じ赤へ',0,-2.37,20,YELLOW))
    return g,red

def product_panel(bi):
    g=VGroup(note(['4個には4つの選択','各点で先に足す','赤も青も 0.5','足してから掛ける','和は積の内側','0.5を4回','箱の上に選択を移す'][bi]))
    xs=(-4.3,-1.43,1.43,4.3)
    obs,dots=balls(y=.35,xs=xs);g.add(obs)
    for i,x in enumerate(xs):
        if bi<6:
            g.add(label('A / B',x,1.60,18,BLUE if i%2==0 else PURPLE),
                  Arrow([x,1.29,0],[x,.73,0],buff=0,color=YELLOW,stroke_width=2,tip_length=.10))
            if bi>=1:g.add(label('0.5',x,-.65,19,GREEN))
        else:
            g.add(Arrow([0,1.58,0],[x,.75,0],buff=0,color=YELLOW,stroke_width=1.8,tip_length=.12))
    if bi==6:g.add(label('h',0,1.93,25,YELLOW))
    if bi>=3 and bi<6:
        for x in (-2.85,0,2.85):g.add(label('×',x,-.65,28,YELLOW))
    if bi in (4,):g.add(formula(r'p(X)=\prod_n\sum_{z_n}p(x_n,z_n)',-2.34,27))
    elif bi==5:g.add(label('0.5 × 0.5 × 0.5 × 0.5 = 0.0625',0,-2.34,20,GREEN))
    elif bi==6:g.add(label('次は 1つの選択を共有',0,-2.34,20,YELLOW))
    else:g.add(label('各個体の選択を独立に行う',0,-2.34,19,YELLOW))
    return g,dots[min(bi//2,3)]

def global_panel(bi):
    g=VGroup(note(['箱全体で1枚','共通のモデル h','Aの世界 ／ Bの世界','事前は半分ずつ','和は集合の外側','ベイズモデル平均化','観測はどちらを支持？'][bi]))
    h=Dot([0,1.45,0],radius=.25,color=YELLOW);g.add(h,label('h',0,1.45,18,BG))
    obs,dots=balls(y=-.55,xs=(-4.25,-1.42,1.42,4.25));g.add(obs)
    for x in (-4.25,-1.42,1.42,4.25):
        g.add(Arrow([0,1.13,0],[x,-.22,0],buff=0,color=YELLOW,stroke_width=1.7,tip_length=.12))
    if bi>=2:g.add(card('A',-3.4,-1.42),card('B',3.4,-1.42))
    if bi==4:g.add(formula(r'p(X)=\sum_h p(X\mid h)p(h)',-2.41,27))
    elif bi==5:g.add(label('1つの h がデータ集合全体を生成',0,-2.40,20,YELLOW))
    elif bi==3:g.add(label('事前  A: 0.5    B: 0.5',0,-2.40,20,PURPLE))
    else:g.add(label('選択は箱全体で1回だけ',0,-2.40,20,YELLOW))
    return g,h

def likelihood_panel(bi):
    g=VGroup(recap('復習: 1.2 事前 × 尤度') if bi==0 else note(['復習: 1.2 事前 × 尤度','Aなら 0.1024','Bなら 0.0064',
      '尤度比は 16 : 1','事後も 16 : 1','Bもまだ残る','次の赤は？'][bi]))
    obs,dots=balls(y=1.52);g.add(obs)
    if bi==0:
        a,b=.5,.5; basis='事前確率'
    elif bi in (1,2,3):
        a,b=.1024,.0064; basis='順序付き4個の尤度'
    else:
        a,b=16/17,1/17; basis='モデルの事後確率'
    maxv=1 if bi in (0,4,5,6) else .11
    g.add(label(basis,0,.83,18,YELLOW))
    # The same axis gets new scale labels when we move from likelihood to posterior.
    for h,y,value,color in (('A',.23,a,BLUE),('B',-.67,b,PURPLE)):
        if bi==1 and h=='B':continue
        span=5.6*value/maxv
        g.add(label(h,-3.6,y,20,color),Rectangle(width=max(span,.007),height=.24,
            stroke_width=0,fill_color=color,fill_opacity=.95).move_to([-2.65+span/2,y,0]),
            label(f'{value:.4f}' if bi in (1,2,3) else f'{value:.3f}',3.75,y,17,color))
    g.add(Line([-2.65,-1.16,0],[2.95,-1.16,0],color=MUTED,stroke_width=1.4),
          label('0',-2.65,-1.47,13,MUTED),label(f'{maxv:g}',2.95,-1.47,13,MUTED),
          label('確率',3.55,-1.47,14,MUTED))
    if bi==0:g.add(label('事前',-2.9,-2.35,18,PURPLE),label('×',-1.65,-2.35,20,YELLOW),
        label('尤度',-.5,-2.35,18,YELLOW),label('→',.72,-2.35,20,YELLOW),
        label('事後',2.0,-2.35,18,GREEN))
    elif bi in (1,2):g.add(label('A: 0.8³ × 0.2     B: 0.2³ × 0.8',0,-2.35,19,YELLOW))
    elif bi==4:g.add(formula(r'p(h\mid X)\propto p(X\mid h)p(h)',-2.36,27))
    else:g.add(label('同じ4個から候補の重みを更新',0,-2.35,19,YELLOW))
    return g,dots[min(bi//2,3)]

def predictive_panel(bi):
    g=VGroup(note(['事後の重みを予測へ','重み付きで足す','13/17 が答え','50% と 76.5%','選択の単位が違う','観測を増やすと？','区別できる候補なら'][bi]))
    g.add(card('A',-3.75,1.40),card('B',3.75,1.40))
    if bi>=1:g.add(label('16/17',-3.75,.18,18,GREEN),label('1/17',3.75,.18,18,GREEN),
       Arrow([-3.15,.10,0],[-.9,-.65,0],buff=0,color=BLUE,stroke_width=2,tip_length=.13),
       Arrow([3.15,.10,0],[.9,-.65,0],buff=0,color=PURPLE,stroke_width=2,tip_length=.13))
    result_bar=None
    if bi>=2:
        result_bar=probability_bar('モデル平均',float(predictive_red()),-1.08,GREEN,True)
        g.add(result_bar)
    if bi in (3,4):g.add(label('点ごとの選択なら 50%',0,-2.38,20,BLUE))
    elif bi==1:g.add(formula(r'p(R\mid X)=\sum_h p(R\mid h,X)p(h\mid X)',-2.40,26))
    elif bi==2:g.add(formula(r'\frac{16}{17}\,0.8+\frac{1}{17}\,0.2=\frac{13}{17}',-2.39,27))
    elif bi==6:g.add(label('候補が同じなら、データでも区別できない',0,-2.38,19,YELLOW))
    else:g.add(label('赤の予測確率',0,-2.40,19,YELLOW))
    return g,result_bar[1] if bi==2 else g[1]

def gaussian_panel(bi):
    g=VGroup(note(['離散例から連続量へ','各点に zₙ','ガウス混合は点ごと','モデルの形を候補に','全体には h が1つ','和の場所を比べる','予測にも同じ区別'][bi]))
    x0=-5.15; y0=-1.32; width=10.2; height=3.15
    g.add(Line([x0,y0,0],[x0+width,y0,0],color=MUTED,stroke_width=1.4),
          Line([x0,y0,0],[x0,y0+height,0],color=MUTED,stroke_width=1.4),
          label('観測値',5.60,y0-.22,15,MUTED),label('密度',-5.55,1.99,15,MUTED))
    for v in (-3,0,3):
        x=x0+(v+4)*width/8
        g.add(Line([x,y0-.08,0],[x,y0+.08,0],color=MUTED,stroke_width=1),label(str(v),x,y0-.32,13,MUTED))
    for v in (0,.5,1):
        y=y0+v*2.4
        g.add(Line([x0-.08,y,0],[x0+.08,y,0],color=MUTED,stroke_width=1),label(f'{v:g}',x0-.34,y,12,MUTED))
    curves=[]
    for mu,color in ((-1.5,BLUE),(1.5,PURPLE)):
        xs=np.linspace(-4,4,140)
        yy=np.exp(-.5*((xs-mu)/.62)**2)/(.62*np.sqrt(2*np.pi))
        curve=VMobject(stroke_color=color,stroke_width=3.3).set_points_smoothly(
            [[x0+(x+4)*width/8,y0+y*2.4,0] for x,y in zip(xs,yy)])
        curves.append(curve);g.add(curve)
    if bi>=2:
        xs=np.linspace(-4,4,140)
        yy=.5*np.exp(-.5*((xs+1.5)/.62)**2)/(.62*np.sqrt(2*np.pi))
        yy+=.5*np.exp(-.5*((xs-1.5)/.62)**2)/(.62*np.sqrt(2*np.pi))
        g.add(VMobject(stroke_color=GREEN,stroke_width=3).set_points_smoothly(
            [[x0+(x+4)*width/8,y0+y*2.4,0] for x,y in zip(xs,yy)]))
    if bi in (3,4,5):
        xs=np.linspace(-4,4,140)
        gamma=.8
        yy=.5*gamma/(np.pi*((xs+1.5)**2+gamma**2))
        yy+=.5*gamma/(np.pi*((xs-1.5)**2+gamma**2))
        cauchy=VMobject(stroke_color=YELLOW,stroke_width=2.6).set_points_smoothly(
            [[x0+(x+4)*width/8,y0+y*2.4,0] for x,y in zip(xs,yy)])
        g.add(DashedVMobject(cauchy,num_dashes=42))
    for i,x in enumerate((-1.9,-1.1,1.1,1.9)):
        px=x0+(x+4)*width/8
        g.add(Dot([px,y0+.08,0],radius=.07,color=RED if i<2 else BLUE))
        if bi in (1,2):
            g.add(tex(f'z_{{{i+1}}}',18,YELLOW).move_to([px,-.83,0]),
                  Arrow([px,-1.01,0],[px,-1.18,0],buff=0,color=YELLOW,stroke_width=1.4,tip_length=.06))
    if bi in (3,4,5):
        g.add(Line([-3.45,1.62,0],[3.45,1.62,0],color=YELLOW,stroke_width=1.5),
              Line([-3.45,1.62,0],[-3.45,1.39,0],color=YELLOW,stroke_width=1.5),
              Line([3.45,1.62,0],[3.45,1.39,0],color=YELLOW,stroke_width=1.5),
              tex('h',22,YELLOW).move_to([0,1.91,0]))
    if bi==2:g.add(formula(r'p(x)=\sum_k\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k)',-2.41,26))
    elif bi==4:g.add(formula(r'p(X)=\sum_h p(X\mid h)p(h)',-2.41,27))
    elif bi in (3,5):g.add(label('緑: ガウス混合  ／  黄破線: コーシー混合',0,-2.40,18,YELLOW))
    else:g.add(label('赤・青: 成分     緑: 混合密度',0,-2.40,18,YELLOW))
    return g,curves[min(bi%2,1)]

def panel(si,bi):
    return (opening_panel(bi) if si==0 else local_panel(bi) if si==1 else
            branch_panel(bi) if si==2 else product_panel(bi) if si==3 else
            global_panel(bi) if si==4 else likelihood_panel(bi) if si==5 else
            predictive_panel(bi) if si==6 else gaussian_panel(bi) if si==7 else
            opening_panel(bi,True))

class PRML141BayesianModelAveraging(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            old=None
            for bi,_ in enumerate(story['beats']):
                state,focus=panel(si,bi)
                duration=self.boundaries[bi]-(self.boundaries[bi-1] if bi else 0)
                enter=min(.8,duration*.18)
                if old is None:
                    transition=FadeIn(state,run_time=enter)
                else:
                    # Replace the state at the beat boundary so old and new
                    # annotations cannot cover each other during a fade.
                    self.remove(old)
                    self.add(state)
                    transition=Indicate(focus,color=focus.get_color(),
                                        scale_factor=1.04,run_time=enter)
                self.beat(phases=[('図を更新',enter,lambda a=transition:a),
                                  ('対応を追う',duration-enter,
                                   lambda f=focus:Indicate(f,color=f.get_color(),scale_factor=1.04))])
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml141_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
