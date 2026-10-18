"""PRML 14.3: follow a single misclassified point through three AdaBoost rounds."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from scene_support import NarratedScene,jp,tex
from make_voicevox_narration import MANIFEST
from boosting_model import X,T,QUERY_INDEX,rounds

BG='#111823';BLUE='#58C4DD';YELLOW='#FFFF00';GREEN='#83C167'
PURPLE='#9A72AC';RED='#FC6255';MUTED='#A8B3C0'
STATES=rounds(); MAP=lambda x:-5.22+.91*x

def txt(s,x,y,size=18,color=WHITE):return jp(s,size,color).move_to([x,y,0])
def formula(s,y=-2.39,size=26):
    m=tex(s,size,YELLOW).move_to([0,y,0]);return m.scale(min(1,11.8/m.width))
def color(t):return RED if t<0 else BLUE

def data_panel(si,bi):
    weight_stage=(0 if si==0 else (1 if si==1 and bi>=5 else 0 if si==1 else
                  1 if si==2 and bi<5 else 2 if si==2 else
                  2 if si==3 else 0 if si==8 and bi<2 else 2 if si==8 else 1))
    weights=np.ones(10)/10 if weight_stage==0 else np.array(STATES[weight_stage-1]['weight_after'])
    if si==0: shown=[] if bi<2 else [0]
    elif si==1:shown=[0]
    elif si==2:shown=[0] if bi<1 else [1]
    elif si==3:shown=[1] if bi<2 else [2]
    elif si==4:shown=[0,1,2]
    elif si==5:shown=[0] if bi==0 else [0,1] if bi==1 else [0,1,2]
    else: shown=[0] if bi<2 else [0,1,2]
    g=VGroup()
    headline=txt(SCENES[si]['beats'][bi]['visual_note'],0,2.48,22,YELLOW)
    if (si==1 and bi==1):
        headline=txt('復習: 1.5 分類の損失',0,2.48,20,WHITE)
        g.add(SurroundingRectangle(headline,color=YELLOW,stroke_width=1.2,buff=.10))
    g.add(headline)
    axis=Line([MAP(0),.22,0],[MAP(10),.22,0],color=MUTED,stroke_width=1.6);g.add(axis)
    for x in (0,2,4,6,8,10):
        xx=MAP(x)
        g.add(Line([xx,.16,0],[xx,.30,0],color=MUTED,stroke_width=1),txt(str(x),xx,-.06,13,MUTED))
    g.add(txt('入力 x',MAP(10)+.56,.22,15,MUTED),txt('赤',-5.67,1.00,16,RED),txt('青',-5.67,.62,16,BLUE))
    circles=[]
    for j,(x,t) in enumerate(zip(X,T)):
        xx=MAP(x)
        dot=Dot([xx,.90,0],radius=.12,color=color(t));g.add(dot)
        r=.14+.11*(weights[j]-.036)/(.25-.036)
        ring=Circle(radius=max(.17,min(.27,r)),color=YELLOW,stroke_width=1.2).set_fill(opacity=0).move_to([xx,.90,0])
        ring.set_stroke(opacity=.45 if weight_stage==0 else .78);g.add(ring);circles.append(ring)
    query=Dot([MAP(X[QUERY_INDEX]),.90,0],radius=.103,color=RED);g.add(query)
    g.add(Triangle(color=YELLOW,fill_opacity=1).scale(.07).rotate(PI).move_to([MAP(4.5),.52,0]))
    g.add(txt('注目 x = 4.5',5.08,1.90,16,YELLOW))
    boundary=[]
    for index in shown:
        state=STATES[index];k=state['k'];xx=MAP(k)
        if len(shown)>1:
            yy=-.52-.32*index
            label_y=yy
            g.add(Line([MAP(0),yy,0],[xx,yy,0],color=color(state['left']),stroke_width=2.3),
                  Line([xx,yy,0],[MAP(10),yy,0],color=color(-state['left']),stroke_width=2.3))
            tick=Line([xx,yy-.14,0],[xx,yy+.14,0],color=(PURPLE,YELLOW,GREEN)[index],stroke_width=3)
            g.add(tick,Dot([MAP(X[QUERY_INDEX]),yy,0],radius=.07,color=color(state['prediction'][QUERY_INDEX])))
            g.add(txt(f'{index+1}本目',-5.77,yy,14,(PURPLE,YELLOW,GREEN)[index]))
            boundary.append(tick)
        else:
            line=Line([xx,.34,0],[xx,1.45,0],color=GREEN,stroke_width=2.4)
            g.add(line);boundary.append(line)
            left='赤' if state['left']<0 else '青'
            right='青' if state['left']<0 else '赤'
            g.add(txt(left,(MAP(0)+xx)/2,-.69,18,color(state['left'])),
                  txt(right,(xx+MAP(10))/2,-.69,18,color(-state['left'])))
    if si in (1,2,3):
        g.add(txt('点の重み',-5.35,-1.29,14,MUTED))
        for j,w in enumerate(weights):
            xx=MAP(X[j]);h=.10+.60*w/.25
            bar=Line([xx,-1.99,0],[xx,-1.99+h,0],color=YELLOW,stroke_width=8)
            g.add(bar)
        g.add(txt('0',4.40,-1.97,12,MUTED),txt('0.25',4.55,-1.35,12,MUTED))
    right_focus=None
    if si==4:
        g.add(Line([4.25,-1.84,0],[6.07,-1.84,0],color=MUTED,stroke_width=1),
              Line([4.25,-1.84,0],[4.25,-.91,0],color=MUTED,stroke_width=1))
        for value in (0,1,2):
            yy=-1.84+.70*value/2
            g.add(Line([4.20,yy,0],[4.30,yy,0],color=MUTED,stroke_width=1),txt(str(value),4.03,yy,11,MUTED))
    if si==4 and bi!=4:
        for j,state in enumerate(STATES):
            x=4.59+j*.54
            h=.70*state['alpha']/2
            bar=Line([x,-1.84,0],[x,-1.84+h,0],color=(PURPLE,YELLOW,GREEN)[j],stroke_width=12)
            g.add(bar)
            if j==2:right_focus=bar
            g.add(txt(str(j+1),x,-1.99,13,MUTED))
        g.add(txt('票の強さ',5.14,-.48,15,YELLOW))
    elif si==4:
        g.add(txt('誤り率 → 票',5.13,-.45,15,YELLOW))
        for x,epsilon,alpha in ((4.73,.5,0),(5.67,.125,1.946)):
            h=.75*alpha/1.946
            bar=Line([x,-1.78,0],[x,-1.78+max(.015,h),0],color=GREEN,stroke_width=11)
            g.add(bar)
            if epsilon<.5:right_focus=bar
            g.add(txt(f'{epsilon:g}',x,-1.99,13,MUTED),txt(f'{alpha:g}',x,-1.83+h+.17,13,GREEN))
    if si in (5,8):
        q=QUERY_INDEX
        score=sum(STATES[j]['alpha']*STATES[j]['prediction'][q] for j in shown)
        result='赤' if score<0 else '青'
        g.add(txt('判定',5.06,.64,16,MUTED),txt(result,5.06,.13,27,color(-1 if score<0 else 1)))
        g.add(txt(f'得点 {score:+.3f}',5.08,-.44,15,GREEN if score<0 else RED))
    f={
      (1,2):r'J_m=\sum_nw_n^{(m)}I(y_m(x_n)\ne t_n)',
      (1,4):r'w_n^{(m+1)}=w_n^{(m)}e^{\alpha_m I(y_m(x_n)\ne t_n)}',
      (2,3):r'\epsilon_2=0.125',
      (3,3):r'\epsilon_3\simeq0.214',
      (4,1):r'\epsilon_m=\frac{\sum_nw_n^{(m)}I(y_m(x_n)\ne t_n)}{\sum_nw_n^{(m)}}',
      (4,2):r'\alpha_m=\ln\frac{1-\epsilon_m}{\epsilon_m}',
      (4,3):r'(\alpha_1,\alpha_2,\alpha_3)=(1.386,1.946,1.299)',
      (4,5):r'Y_3(x)=\operatorname{sign}\sum_{m=1}^{3}\alpha_my_m(x)',
      (5,0):r'+1.386',
      (5,1):r'+1.386-1.946=-0.560',
      (5,2):r'+1.386-1.946-1.299=-1.859',
      (5,3):r'Y_3(4.5)=-1',
      (8,3):r'\sum_{m=1}^{3}\alpha_my_m(4.5)=-1.859',
    }
    if (si,bi) in f:g.add(formula(f[(si,bi)]))
    else:
        if si==0 and bi>=3:g.add(formula(r'x=4.5,\quad t=-1,\quad y_1(4.5)=+1'))
        elif si in (1,2,3):g.add(formula(r'w_n^{(1)}=0.1\quad\longrightarrow\quad w_n^{(m+1)}'))
        elif si==8 and bi>=2:g.add(formula(r'Y_3(4.5)=-1\quad'))
    focus=(circles[QUERY_INDEX] if si in (0,5,8) else
           right_focus if si==4 and right_focus is not None else
           circles[4 if si==1 else 2] if si in (1,2) else
           boundary[-1] if boundary else query)
    return g,focus,dict(rings=circles,weights=weights,boundaries=boundary,k=STATES[shown[-1]]['k'] if shown else None)

def loss_panel(si,bi):
    g=VGroup();headline=txt(SCENES[si]['beats'][bi]['visual_note'],0,2.50,22,YELLOW)
    if si==7 and bi==0:
        headline=txt('復習: 4.3 交差エントロピー',0,2.50,20,WHITE)
        g.add(SurroundingRectangle(headline,color=YELLOW,stroke_width=1.2,buff=.10))
    g.add(headline)
    x0,x1=-4.60,3.77;y0,y1=-1.65,1.72
    P=lambda z,v:np.array([x0+(z+2.1)/4.3*(x1-x0),y0+v/6.4*(y1-y0),0])
    g.add(Line(P(-2.1,0),P(2.2,0),color=MUTED,stroke_width=1.5),
          Line(P(-2.1,0),P(-2.1,6.4),color=MUTED,stroke_width=1.5))
    for z in (-2,-1,0,1,2):
        q=P(z,0);g.add(Line(q+[0,-.07,0],q+[0,.07,0],color=MUTED,stroke_width=1),
                     txt(str(z),q[0],-1.96,13,MUTED))
    for v in (0,1,3,6):
        q=P(-2.1,v);g.add(Line(q+[-.07,0,0],q+[.07,0,0],color=MUTED,stroke_width=1),
                     txt(str(v),-4.94,q[1],13,MUTED))
    g.add(txt('余白 z',4.38,-1.63,16,MUTED),txt('損失',-5.03,1.93,16,MUTED))
    z=np.linspace(-1.82,2.15,100)
    expcurve=VMobject(stroke_color=GREEN,stroke_width=3).set_points_smoothly([P(a,np.exp(-a)) for a in z])
    g.add(expcurve)
    crosscurve=VMobject(stroke_color=RED,stroke_width=2.7).set_points_smoothly([P(a,np.log1p(np.exp(-a))/np.log(2)) for a in z])
    if si==7 or bi>=1:g.add(crosscurve)
    g.add(txt('指数誤差',5.13,1.19,16,GREEN))
    if si==7 or bi>=1:g.add(txt('交差エントロピー',5.19,.73,15,RED))
    if si==6:
        zv=(0 if bi<2 else -1 if bi<4 else -1.6)
    else:zv=(-1 if bi<3 else -1.65)
    q=P(zv,np.exp(-zv));dot=Dot(q,radius=.10,color=RED);g.add(dot)
    g.add(txt('最初の赤い点',5.12,-.05,15,RED))
    if si==6 and bi==1:g.add(formula(r'E=\sum_n e^{-t_nf(x_n)}'))
    elif si==6 and bi==4:g.add(formula(r'w_n^{(m+1)}=w_n^{(m)}e^{\alpha_m I(y_m(x_n)\ne t_n)}'))
    elif si==6 and bi==2:g.add(formula(r'e^0=1,\qquad e^1\simeq2.718'))
    elif si==7 and bi>=2:g.add(formula(r'e^{-z}\quad,\quad\frac{\ln(1+e^{-z})}{\ln 2}'))
    return g,dot if bi in (2,3,4,5) else expcurve,dict(dot=dot,z=zv)

class PRML143Boosting(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si);old=None;old_meta=None
            for bi,_ in enumerate(story['beats']):
                state,focus,meta=loss_panel(si,bi) if si in (6,7) else data_panel(si,bi)
                duration=self.boundaries[bi]-(self.boundaries[bi-1] if bi else 0)
                enter=min(2.2,duration*.32)
                if old is None:
                    self.beat(phases=[('図を導入',enter,lambda a=FadeIn(state):a),
                                      ('注目する量',duration-enter,lambda f=focus:Indicate(f,color=f.get_color(),scale_factor=1.05))])
                else:
                    motions=[]
                    if 'rings' in meta and not np.allclose(meta['weights'],old_meta['weights']):
                        motions += [Transform(a,b.copy()) for a,b in zip(old_meta['rings'],meta['rings'])]
                    if (meta.get('k') is not None and old_meta.get('k') is not None and
                        meta['k']!=old_meta['k'] and len(meta['boundaries'])==len(old_meta['boundaries'])==1 and
                        si in (2,3)):
                        motions.append(Transform(old_meta['boundaries'][0],meta['boundaries'][0].copy()))
                    if 'dot' in meta and meta['z']!=old_meta['z']:
                        motions.append(Transform(old_meta['dot'],meta['dot'].copy()))
                    if motions:
                        old_state=old
                        def finish(new=state,prior=old_state,target=focus):
                            self.remove(prior);self.add(new)
                            return Indicate(target,color=target.get_color(),scale_factor=1.05)
                        self.beat(phases=[('境界と重みを動かす',enter,lambda parts=motions:AnimationGroup(*parts,lag_ratio=0)),
                                          ('結果を確認',duration-enter,finish)])
                    else:
                        self.remove(old);self.add(state)
                        self.beat(phases=[('図を更新',enter,lambda f=focus:Indicate(f,color=f.get_color(),scale_factor=1.07)),
                                          ('注目する量',duration-enter,lambda f=focus:Indicate(f,color=f.get_color(),scale_factor=1.05))])
                old=state;old_meta=meta
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml143_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')
