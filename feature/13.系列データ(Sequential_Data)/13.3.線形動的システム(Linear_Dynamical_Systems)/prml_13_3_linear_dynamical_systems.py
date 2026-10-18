"""PRML 13.3: a noisy position sensor and the linear dynamical system."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex
from make_voicevox_narration import MANIFEST
from lds_model import OBS, infer, SENSOR_VARIANCE

BG='#111823'
BLUE='#58C4DD'
YELLOW='#FFFF00'
GREEN='#83C167'
PURPLE='#9A72AC'
RED='#FC6255'
MUTED='#A8B3C0'
DATA=infer()
LOW=infer(sensor_variance=.04)


def label(value,x,y,size=19,color=WHITE):
    return jp(value,size,color).move_to([x,y,0])


def formula(value,y=-2.54,size=26):
    item=tex(value,size,YELLOW).move_to([0,y,0])
    if item.width>11.5:item.scale(11.5/item.width)
    return item


def note(value,color=YELLOW):
    return label(value,0,2.48,19,color)


def timeline_axes():
    g=VGroup()
    x0,y0=-5.10,-1.75
    g.add(Line([x0,y0,0],[5.05,y0,0],color=MUTED,stroke_width=1.6),
          Line([x0,y0,0],[x0,2.12,0],color=MUTED,stroke_width=1.6))
    for t in range(1,6):
        xx=-4.25+(t-1)*2.12
        g.add(Line([xx,y0-.09,0],[xx,y0+.09,0],color=MUTED,stroke_width=1),
              label(str(t),xx,-2.03,14,MUTED))
    for v in (0,2,4):
        yy=y0+v*.91
        g.add(Line([x0-.10,yy,0],[x0+.10,yy,0],color=MUTED,stroke_width=1),
              label(str(v),-5.45,yy,14,MUTED))
    g.add(label('回目',5.64,-2.04,15,MUTED),label('位置',-5.72,2.08,15,MUTED))
    return g


def tx(i):return -4.25+i*2.12

def ty(v):return -1.75+float(v)*.91


def time_panel(si,bi):
    g=VGroup(timeline_axes())
    obs_count={0:[3,3,4,5,5,5],5:[1,2,3,4,5,5],6:[5]*6,7:[5]*6,9:[5]*6}[si][bi]
    filter_count=0
    smooth_count=0
    if si==0 and bi>=4:filter_count=4
    if si==5:filter_count=min(bi+1,5)
    if si==6:filter_count=5;smooth_count=[0,0,4,4,5,5][bi]
    if si==7:filter_count=5;smooth_count=5 if bi>=2 else 0
    if si==9:filter_count=3 if bi<3 else 4;smooth_count=5 if bi>=4 else 0
    obs_points=[];filter_points=[];smooth_points=[];filter_dots=[];smooth_dots=[]
    for i in range(obs_count):
        color=RED if i==3 else BLUE
        pos=[tx(i),ty(OBS[i]),0]
        dot=Dot(pos,radius=.095,color=color)
        obs_points.append(dot);g.add(dot)
    if filter_count:
        for i in range(filter_count):
            pos=[tx(i),ty(DATA['mean'][i]),0]
            filter_points.append(pos)
            dot=Dot(pos,radius=.105,color=GREEN)
            filter_dots.append(dot);g.add(dot)
            if i and i<filter_count:g.add(Line(filter_points[i-1],pos,color=GREEN,stroke_width=2.5))
    if smooth_count:
        for i in range(smooth_count):
            pos=[tx(i),ty(DATA['smooth_mean'][i]),0]
            smooth_points.append(pos)
            dot=Dot(pos,radius=.075,color=YELLOW)
            smooth_dots.append(dot);g.add(dot)
            if i:g.add(Line(smooth_points[i-1],pos,color=YELLOW,stroke_width=2))
    if si==0 and bi==1:
        g.add(label('？',tx(3),ty(.7),25,YELLOW))
    if si==6 and bi in (2,3):
        g.add(Arrow([tx(4)-.15,1.92,0],[tx(3)+.15,1.92,0],
                    buff=0,color=YELLOW,stroke_width=2,tip_length=.12))
    if si==0:
        messages=['センサーの観測を順に開く','次はどの位置？','4回目に大きな測定','5回目は再び近くへ','測定 3.5 ／ 推定 1.09','なぜ一部だけ取り入れる？']
    elif si==5:
        messages=['観測 0.1 → 推定 0.05','観測 0.2 → 推定 0.10','観測 0.5 → 推定 0.22',
                  '観測 3.5 → 推定 1.09','観測 0.9 → 推定 1.04','前向き: その時点までの観測']
    elif si==6:
        messages=['5回目から見直す','13.2と同じ後向きの手がかり','4回目: 1.09 → 1.05',
                  '次の時点の修正を戻す','緑: 前向き ／ 黄: 平滑化','使える観測の範囲が違う']
    elif si==7:
        messages=['真の位置は見えない','期待値は重み付き平均','Eステップ: 分布を推定',
                  'Mステップ: 雑音を更新','初期共分散は P₀','線形ガウスなら計算できる']
    else:
        messages=['同じ5回の観測','4回目の予測は 0.22','測定は 3.5、雑音は大きい',
                  '利得 0.264 → 推定 1.09','平滑化後は 1.05','幅の比較が答え']
    heading=note(messages[bi])
    g.add(heading)
    if si==2 and bi==0:
        g.add(SurroundingRectangle(heading,color=YELLOW,stroke_width=1.2,buff=.10))
    if si==9 and bi in (1,2,3):
        g.add(label('予測',-4.25,-2.50,16,PURPLE),label(f'{DATA["prior_mean"][3]:.2f}',-3.35,-2.50,16,PURPLE),
              label('測定',-.80,-2.50,16,BLUE),label('3.50',.16,-2.50,16,BLUE),
              label('推定',2.20,-2.50,16,GREEN),label('1.09',3.13,-2.50,16,GREEN))
    else:
        g.add(label('青・赤: 測定',-3.83,-2.50,16,BLUE),label('緑: 前向き',-.30,-2.50,16,GREEN),
              label('黄: 平滑化',3.13,-2.50,16,YELLOW))
    if si==0:
        focus=(filter_dots[3] if bi>=4 and len(filter_dots)>3 else
               obs_points[3] if bi==2 and len(obs_points)>3 else obs_points[-1])
    elif si==5:
        focus=filter_dots[min(bi,4)]
    elif si==6:
        focus=smooth_dots[3] if bi>=2 and len(smooth_dots)>3 else filter_dots[3]
    elif si==9:
        focus=(smooth_dots[3] if bi>=4 and len(smooth_dots)>3 else
               filter_dots[3] if bi>=3 and len(filter_dots)>3 else obs_points[3])
    else:
        focus=filter_dots[3] if len(filter_dots)>3 else obs_points[3]
    return g,focus


def chain_panel(bi):
    g=VGroup()
    xs=[-4.6,-2.3,0,2.3,4.6]
    g.add(note(['前節の鎖を連続位置へ','隠れた位置と測定','位置の変化に雑音','測定にも雑音',
                '状態と測定で幅が違う','測定の方が大きく揺れる'][bi]))
    for i,x in enumerate(xs):
        g.add(Circle(radius=.28,color=PURPLE,stroke_width=2).move_to([x,1.05,0]),
              tex(f'z_{{{i+1}}}',20,PURPLE).move_to([x,1.05,0]),
              Circle(radius=.28,color=BLUE,stroke_width=2).move_to([x,-.72,0]),
              label(f'{OBS[i]:.1f}',x,-.72,17,BLUE if i!=3 else RED),
              Arrow([x,.70,0],[x,-.37,0],buff=0,color=BLUE,stroke_width=2,tip_length=.1))
        if i:
            g.add(Arrow([xs[i-1]+.35,1.05,0],[x-.35,1.05,0],buff=0,color=YELLOW,stroke_width=2,tip_length=.1))
    g.add(label('真の位置',-5.58,1.05,17,PURPLE),label('測定',-5.58,-.72,17,BLUE))
    texts=['13.2: 離散状態 → 13.3: 連続位置',
           r'z_n \longrightarrow x_n',r'z_n=Az_{n-1}+w_n',r'x_n=Cz_n+v_n',
           r'\Gamma=0.04,\quad \Sigma=0.64',r'\Sigma>\Gamma']
    g.add(formula(texts[bi],-2.30,25) if bi else label(texts[bi],0,-2.30,20,YELLOW))
    return g,g[-1]


def density_axes():
    g=VGroup()
    x0,y0=-5.15,-1.65
    g.add(Line([x0,y0,0],[5.05,y0,0],color=MUTED,stroke_width=1.5),
          Line([x0,y0,0],[x0,1.83,0],color=MUTED,stroke_width=1.5))
    for v in (-1,0,1,2,3,4):
        x=dex(v)
        g.add(Line([x,y0-.08,0],[x,y0+.08,0],color=MUTED,stroke_width=1),label(str(v),x,-1.99,14,MUTED))
    for v in (0,1,2):
        y=y0+1.35*v
        g.add(Line([x0-.08,y,0],[x0+.08,y,0],color=MUTED,stroke_width=1),
              label(f'{v:g}',-5.53,y,13,MUTED))
    g.add(label('位置',5.58,-1.98,15,MUTED),label('密度',-5.56,2.05,15,MUTED))
    return g


def dex(v):return -5.15+(v+1.5)*1.66

def gaussian(mean,var,color):
    xs=np.linspace(-1.5,4.55,140)
    ys=np.exp(-.5*(xs-mean)**2/var)/np.sqrt(2*np.pi*var)
    return VMobject(stroke_color=color,stroke_width=3.2).set_points_smoothly(
        [[dex(x),-1.65+1.35*y,0] for x,y in zip(xs,ys)])


def density_panel(si,bi):
    g=VGroup(density_axes())
    if si==2:
        pred_mean=DATA['prior_mean'][3];pred_var=DATA['prior_var'][3]
        if bi==0:
            pred_mean=DATA['mean'][2];pred_var=DATA['var'][2]
        if bi in (0,1,2):
            g.add(gaussian(DATA['mean'][2],DATA['var'][2],BLUE))
        if bi>=3:g.add(gaussian(pred_mean,pred_var,PURPLE))
        messages=['復習: 2.3 ガウス分布','3回目の中心 0.22、分散 0.19','次も中心は 0.22',
                  '状態雑音で幅が広がる','予測の分散 0.23','ここへ測定 3.5 が届く']
        if bi==4:g.add(formula(r'P_{n-1}=AV_{n-1}A^{\mathsf T}+\Gamma',-2.54,24))
    elif si==3:
        g.add(gaussian(DATA['prior_mean'][3],DATA['prior_var'][3],PURPLE),
              gaussian(OBS[3],SENSOR_VARIANCE,BLUE))
        if bi>=1:g.add(gaussian(DATA['mean'][3],DATA['var'][3],GREEN))
        messages=['紫: 予測 ／ 青: 測定','重なりから緑の山へ','中心は測定側へ少し移る',
                  '緑の山は予測より狭い','カルマンフィルタ','どれだけ近づく？']
    elif si==4:
        low=bi in (2,)
        d=LOW if low else DATA
        sensor=.04 if low else SENSOR_VARIANCE
        g.add(gaussian(DATA['prior_mean'][3],DATA['prior_var'][3],PURPLE),
              gaussian(OBS[3],sensor,BLUE),
              gaussian(d['mean'][3],d['var'][3],GREEN))
        messages=['赤: 予測誤差 3.28','予測 + 利得 × 誤差','測定雑音 0.04 → 利得 0.62',
                  '測定雑音 0.64 → 利得 0.264','予測 0.22 → 推定 1.09','更新後の山は狭い']
        if bi==1:g.add(formula(r'\mu_n=A\mu_{n-1}+K_n(x_n-CA\mu_{n-1})',-2.56,23))
        if bi==4:g.add(formula(r'K_n=P_{n-1}/(P_{n-1}+\Sigma)',-2.56,24))
        if bi==5:g.add(formula(r'V_n=(1-K_nC)P_{n-1}',-2.56,24))
        if bi in (2,3):
            slider_y=-2.45
            g.add(Line([-2.6,slider_y,0],[1.9,slider_y,0],color=MUTED,stroke_width=2),
                  Dot([-2.6+4.5*(sensor-.04)/.60,slider_y,0],radius=.11,color=YELLOW),
                  label('測定分散',-4.18,slider_y,15,BLUE),
                  label(f'{sensor:.2f}',2.60,slider_y,16,BLUE),
                  label(f'K = {d["gain"][3]:.3f}',4.45,slider_y,16,YELLOW))
    else:
        g.add(gaussian(2.7,1.5,PURPLE),gaussian(2.15,.13,BLUE),gaussian(3.53,.13,BLUE))
        if bi>=1:
            xs=np.linspace(-1.5,4.55,140)
            yy=.55*np.exp(-.5*(xs-2.20)**2/.12)/np.sqrt(2*np.pi*.12)
            yy+=.45*np.exp(-.5*(xs-3.46)**2/.12)/np.sqrt(2*np.pi*.12)
            g.add(VMobject(stroke_color=GREEN,stroke_width=3).set_points_smoothly(
                [[dex(x),-1.65+1.35*y,0] for x,y in zip(xs,yy)]))
        if bi>=2:
            for i,x in enumerate(np.linspace(-.5,4.3,16)):
                color=GREEN if abs(x-2.15)<.8 or abs(x-3.53)<.8 else MUTED
                g.add(Dot([dex(x),-1.33-.18*(i%2),0],radius=.055,color=color))
        messages=['測定の候補が2つ','山の数が増えていく','分布を粒子で表す',
                  '観測に合う点へ重み','粒子フィルタ','今回の条件なら山1つで十分']
        if bi==3:g.add(formula(r'w_n^{(l)}\propto p(x_n\mid z_n^{(l)})',-2.53,24))
    heading=note(messages[bi])
    g.add(heading)
    if si==2 and bi==0:
        g.add(SurroundingRectangle(heading,color=YELLOW,stroke_width=1.2,buff=.10))
    if (si==3 or si==4) and not (si==4 and bi in (1,2,3,4,5)):
        g.add(label('紫: 予測',-3.70,-2.45,16,PURPLE),label('青: 測定',-.05,-2.45,16,BLUE),label('緑: 更新',3.30,-2.45,16,GREEN))
    elif si==2 and bi!=4:
        g.add(label('中心 = 平均、幅 = 分散',0,-2.45,18,YELLOW))
    elif si==8 and bi!=3:
        g.add(label('峰が分かれると、山1つでは表せない',0,-2.45,18,YELLOW))
    return g,g[-1]


def panel(si,bi):
    if si in (0,5,6,7,9):return time_panel(si,bi)
    if si==1:return chain_panel(bi)
    return density_panel(si,bi)


class PRML133LinearDynamicalSystems(NarratedScene):
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
                if si==4 and bi==3:
                    # Evolve the actual Gaussian parameters, rather than
                    # interpolating SVG paths (which creates false double peaks).
                    old[7].set_opacity(0)
                    old[8].set_opacity(0)
                    old[9].become(note('測定分散を戻す'))
                    value=DecimalNumber(.04,num_decimal_places=2,font_size=25,color=BLUE).move_to([2.60,-2.45,0])
                    k_label=label('K =',3.64,-2.45,16,YELLOW)
                    k_value=DecimalNumber(float(LOW['gain'][3]),num_decimal_places=3,
                                          font_size=25,color=YELLOW).move_to([4.83,-2.45,0])
                    self.add(value,k_label,k_value)
                    prior=float(DATA['prior_var'][3])
                    center=float(DATA['prior_mean'][3])
                    observation=float(OBS[3])
                    def sensor(alpha):
                        return .04+.60*alpha
                    def curve_motion():
                        return AnimationGroup(
                            UpdateFromAlphaFunc(old[2],lambda m,a:m.become(gaussian(observation,sensor(a),BLUE))),
                            UpdateFromAlphaFunc(old[3],lambda m,a:m.become(
                                gaussian(center+prior/(prior+sensor(a))*(observation-center),
                                         (1-prior/(prior+sensor(a)))*prior,GREEN))),
                            UpdateFromAlphaFunc(old[5],lambda m,a:m.move_to([-2.6+4.5*a,-2.45,0])),
                            UpdateFromAlphaFunc(value,lambda m,a:m.set_value(sensor(a))),
                            UpdateFromAlphaFunc(k_value,lambda m,a:m.set_value(prior/(prior+sensor(a)))))
                    def finish_motion():
                        self.remove(old,*old.submobjects,value,k_label,k_value)
                        self.add(state)
                        return Indicate(focus,color=YELLOW,scale_factor=1.015)
                    move=min(4.2,duration*.62)
                    self.beat(phases=[('つまみと曲線',move,curve_motion),
                                      ('利得を確認',duration-move,finish_motion)])
                    old=state
                    continue
                transition=(FadeIn(state) if old is None else
                            Succession(FadeOut(old,run_time=.18),FadeIn(state,run_time=.25),Wait(.15)))
                switch=min(.9,duration*.15)
                self.beat(phases=[('図の更新',switch,lambda a=transition:a),
                                  ('注目箇所',duration-switch,
                                   lambda f=focus:Indicate(f,color=YELLOW,scale_factor=1.015))])
                old=state
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml133_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
