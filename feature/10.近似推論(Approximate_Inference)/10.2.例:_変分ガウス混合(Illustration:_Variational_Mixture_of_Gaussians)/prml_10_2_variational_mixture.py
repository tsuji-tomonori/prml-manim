"""Six candidates, two surviving components: a narrated PRML 10.2 experiment."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t, norm
from manim import *
from mixture_model import fit, posterior, expected_logs, responsibilities, observations, K, ALPHA0
from make_voicevox_narration import MANIFEST
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex

BG='#10141F'
BLUE=ManimColor('#58C4DD')
GREEN=ManimColor('#83C167')
YELLOW=ManimColor('#FFFF00')
PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255')
MUTED=ManimColor('#A8B2C5')
COLORS=[PURPLE,BLUE,RED,YELLOW,GREEN,ManimColor('#FF9E63')]
X,HISTORY=fit()
_,STRONG_HISTORY=fit(alpha0=10.)

ITERATIONS=[
    [0,0,0,30,0,0,0],
    [0,0,0,0,1,1,1],
    [0,0,0,0,0,0,0],
    [0,1,1,2,4,5,5],
    [0,2,5,10,30,30,30],
    [0,30,30,30,30,30,30],
    [30,30,30,30,30,30],
    [0,2,10,30,30,30],
]
NOTES=[
    ['前節の更新を、点の所属へ','変分ガウス混合：候補は6個','6個とも残る？','結果：目立つ山は2個','データも候補数も同じ','6個のまま追う','なぜ4個が小さくなる？'],
    ['復習: 9.2 ガウス混合','所属とパラメータが未知','所属とパラメータを分ける','事前分布を置く','一方を固定して他方を更新','点ごとの所属／全点で共有','境目の点は何票渡す？'],
    ['境目の1点を調べる','重み・幅・距離の採点','正の重み→合計1','複数の山へ票を分ける','点を動かすと割合も変わる','1点の合計／80点の合計','票を集めると山は？'],
    ['届いた票の合計','事前の中心と点の中心','復習: 2.2 ディリクレ分布','復習: 2.3 ガウス分布','中心と幅が動く','精度と楕円の幅','票を返して繰り返す'],
    ['所属→パラメータ','2巡目','10巡目','更新を下界で測る','30巡目','固定した条件での上昇','下界は証拠の下の値'],
    ['冒頭の6本の重み','票と事前を足す','ゼロ票でも厳密な0ではない','事前 α₀ = 10','小さな事前 α₀ = 0.001','中心と精度も事前へ','新しい点はどう予測？'],
    ['新しい点を置く','パラメータも不確か','候補を平均する','予測は t 分布の混合','データが多いとガウスへ','同じ2個の山を使う'],
    ['冒頭と同じ条件','票が大きな山へ','重みは約0.5ずつ','下界が上昇して安定','この例での答え','10.3 線形回帰へ'],
]
FORMULAS=[
    ['',r'N=80,\quad K=6',r'K=6',r'2/6',r'N=80,\ K=6',r'K=6',''],
    [r'x=0.15',r'z_n,\ \pi,\ \mu,\ \Lambda',r'q(Z)q(\pi,\mu,\Lambda)',
     r'p(\pi),\ p(\mu,\Lambda)','',r'z_n\ ;\ \pi,\mu,\Lambda',''],
    ['',r'\ln\rho_{nk}=E[\ln\pi_k]+\cdots',r'\rho_{nk}>0',
     r'r_{nk}=\rho_{nk}/\sum_j\rho_{nj}',r'\sum_kr_{nk}=1',r'\sum_kr_{nk}=1,\quad N_k=\sum_nr_{nk}',''],
    [r'N_k=\sum_nr_{nk}',r'm_k=\frac{\beta_0m_0+N_k\bar x_k}{\beta_0+N_k}',
     r'\alpha_k=\alpha_0+N_k',r'\Lambda=\Sigma^{-1}',r'\beta_k=\beta_0+N_k',r'\Lambda=\Sigma^{-1}',''],
    [r'0\rightarrow1',r'1\rightarrow2',r'2\rightarrow10',
     r'\mathcal L_0=-254.56',r'\mathcal L_{30}=-193.89',r'\mathcal L_{t+1}\geq\mathcal L_t',r'\mathcal L\leq\ln p(X)'],
    [r'K=6',r'E[\pi_k]=\frac{\alpha_0+N_k}{K\alpha_0+N}',
     r'N_k\simeq0\Rightarrow E[\pi_k]\simeq0.0000125',
     r'\alpha_0=10',r'\alpha_0=0.001',r'q(\mu_k,\Lambda_k)\rightarrow p(\mu_k,\Lambda_k)',''],
    [r'\tilde x',r'q(\pi)q(\mu,\Lambda)',r'\int p(\tilde x\mid\theta)q(\theta)d\theta',
     r'\sum_kE[\pi_k]\,\mathrm{St}_k(\tilde x)',r'\nu_k+1-D',''],
    [r'N=80,\ K=6',r'N_k=\sum_nr_{nk}',r'0.500,\ 0.500',
     r'-254.56\rightarrow-193.89',r'4\times0.0000125',r'10.3'],
]

def J(s,size=20,color=WHITE):
    return jp(str(s),size,color)

def coord(x,y):
    return np.array([-1.85+1.03*x,.1+.8*y,0.])

def graph_axes():
    g=VGroup(Line(coord(-3.7,-2.3),coord(3.65,-2.3),color=MUTED,stroke_width=1.25),
             Line(coord(-3.7,-2.3),coord(-3.7,2.45),color=MUTED,stroke_width=1.25))
    for value in (-2,0,2):
        at=coord(value,-2.3)
        g.add(Line(at+DOWN*.07,at+UP*.07,color=MUTED,stroke_width=1.0),
              J(str(value),13,MUTED).move_to(at+DOWN*.23))
    g.add(tex('x_1',22,MUTED).move_to(coord(3.75,-2.3)+RIGHT*.22),
          tex('x_2',22,MUTED).move_to(coord(-3.7,2.5)+UP*.15))
    return g

def dots(probe=None):
    group=VGroup(*[Dot(coord(*x),radius=.023,color=MUTED,fill_opacity=.72)
                   for x in X])
    if probe is not None:
        group.add(Dot(coord(*probe),radius=.085,color=YELLOW),
                  Circle(radius=.13,color=YELLOW,stroke_width=1.8).move_to(coord(*probe)))
    return group

def contours(state):
    group=VGroup()
    for k in range(K):
        c=state['covariance'][k]
        vals,vecs=np.linalg.eigh(c)
        root=vecs@np.diag(np.sqrt(np.maximum(vals,1e-7)))
        angles=np.linspace(0,2*np.pi,49)
        points=[coord(*(state['mean'][k]+root@np.array([np.cos(a),np.sin(a)])))
                for a in angles]
        opacity=.9*min(1,float(state['weights'][k])/.08)
        curve=VMobject().set_points_as_corners(points)
        curve.set_stroke(COLORS[k],width=2.6,opacity=opacity)
        group.add(curve)
    return group

def bars(weights,caption='混合係数の平均'):
    group=VGroup(J(caption,19,MUTED).move_to([4.22,1.48,0]))
    baseline=-1.65
    group.add(Line([2.7,baseline,0],[6.3,baseline,0],color=MUTED,stroke_width=1.2))
    for k,w in enumerate(weights):
        x=3.04+.56*k
        height=max(.016,2.65*float(w))
        group.add(Line([x,baseline,0],[x,baseline+height,0],color=COLORS[k],stroke_width=21,
                       stroke_opacity=.85),
                  J(str(k+1),15,COLORS[k]).move_to([x,-1.93,0]))
    return group

def responsibility_bars(x,p):
    r=responsibilities(np.asarray(x)[None,:],p)[0]
    group=bars(r,'この点の負担率')
    group.add(J('合計 1',17,GREEN).move_to([4.3,-2.38,0]))
    return group

def raw_score_bars(x,p):
    elnpi,elnlam=expected_logs(p)
    diff=np.asarray(x)-p['mean']
    distance=np.einsum('ki,kij,kj->k',diff,p['w'],diff)
    logrho=elnpi+.5*elnlam-.5*(2*np.log(2*np.pi)+2/p['beta']+p['nu']*distance)
    rho=np.exp(logrho)
    group=VGroup(J('指数で得た正の重み',18,MUTED).move_to([4.3,1.48,0]),
                 Line([2.7,-1.65,0],[6.3,-1.65,0],color=MUTED,stroke_width=1.2))
    for k,v in enumerate(rho):
        xbar=3.04+.56*k
        group.add(Line([xbar,-1.65,0],[xbar,-1.65+190*v,0],
                       color=COLORS[k],stroke_width=21),
                  J(str(k+1),15,COLORS[k]).move_to([xbar,-1.93,0]))
    group.add(J(f'合計 {rho.sum():.4f}',17,PURPLE).move_to([4.3,-2.38,0]))
    return group

def recap_92():
    """Only the 9.2 x=0.15 weighted hills, redrawn without importing that scene."""
    centers=(-1.1,1.1);sigmas=(.9,1.2);weights=(.65,.35)
    xs=np.linspace(-3,3,81)
    group=VGroup(J('境目の1点',18,MUTED).move_to([4.35,1.48,0]),
                 Line([2.84,-.55,0],[6.16,-.55,0],color=MUTED,stroke_width=1.1))
    for mu,sigma,weight,color in zip(centers,sigmas,weights,(RED,BLUE)):
        yy=weight*norm.pdf(xs,loc=mu,scale=sigma)
        points=[np.array([2.84+(x+3)*3.32/6,-.55+5*y,0]) for x,y in zip(xs,yy)]
        group.add(VMobject().set_points_as_corners(points).set_stroke(color,width=2.7))
    x=2.84+(0.15+3)*3.32/6
    group.add(Dot([x,-.55,0],radius=.065,color=YELLOW))
    for y,amount,color,label in ((-1.16,.564,RED,'赤 0.564'),(-1.76,.436,BLUE,'青 0.436')):
        group.add(J(label,17,color).move_to([3.35,y,0]),
                  Line([4.25,y,0],[4.25+2.55*amount,y,0],color=color,stroke_width=14))
    return group

def recap_23(sigma):
    """The 2.3 center-and-width operation as a compact one-curve card."""
    xs=np.linspace(-2.6,2.6,81)
    color=BLUE if sigma>1 else GREEN
    points=[np.array([4.5+3.15*x/5.2,-1.35+3.2*norm.pdf(x,scale=sigma),0]) for x in xs]
    curve=VMobject().set_points_as_corners(points).set_stroke(color,width=3)
    return VGroup(J('同じ中心、違う幅',18,MUTED).move_to([4.4,1.48,0]),
                  Line([2.9,-1.35,0],[6.1,-1.35,0],color=MUTED,stroke_width=1.1),
                  Line([4.5,-1.35,0],[4.5,1.02,0],color=YELLOW,stroke_width=1,
                       stroke_opacity=.65),curve,
                  J(f'幅 {sigma:.2f}',18,color).move_to([4.4,-1.88,0]))

def recap_22(updated):
    """Dirichlet counts: the 2.2 prior receives fractional observations."""
    counts=np.full(K,ALPHA0)+ (HISTORY[1]['nk'] if updated else 0)
    group=VGroup(J('票を加えたパラメータ' if updated else '各成分の事前',18,MUTED)
                 .move_to([4.35,1.48,0]),
                 Line([2.8,-1.65,0],[6.25,-1.65,0],color=MUTED,stroke_width=1.1))
    for k,value in enumerate(counts):
        x=3.04+.56*k
        group.add(Line([x,-1.65,0],[x,-1.65+max(.02,.08*value),0],
                       color=COLORS[k] if updated else PURPLE,stroke_width=21),
                  J(str(k+1),15,COLORS[k]).move_to([x,-1.94,0]))
    group.add(J('合計 80.006' if updated else '各成分 0.001',17,
                GREEN if updated else PURPLE).move_to([4.35,-2.36,0]))
    return group

def bound_chart(upto):
    vals=np.array([HISTORY[t]['bound'] for t in range(31)])
    group=VGroup(J('変分下界',19,MUTED).move_to([4.3,1.5,0]),
                 Line([2.7,-1.65,0],[6.25,-1.65,0],color=MUTED,stroke_width=1.2))
    points=[np.array([2.72+3.45*t/30,-1.4+2.6*(vals[t]-vals[0])/(vals[30]-vals[0]),0])
            for t in range(upto+1)]
    if upto==0:
        points.append(points[0]+RIGHT*.001)
    path=VMobject().set_points_as_corners(points)
    path.set_stroke(GREEN,width=3)
    group.add(path,Dot(points[-1],radius=.065,color=YELLOW),
              J(f'{vals[upto]:.2f}',17,GREEN).move_to([4.3,-2.12,0]))
    return group

def predictive(show_average=False):
    p=posterior(X,HISTORY[30]['r'])
    xs=np.linspace(-3.65,3.65,100)
    values=np.zeros_like(xs);plug=np.zeros_like(xs)
    for k in range(K):
        df=p['nu'][k]+1-2
        scale_matrix=np.linalg.inv(p['w'][k])*((1+p['beta'][k])/(df*p['beta'][k]))
        scale=np.sqrt(scale_matrix[0,0])
        pi=p['alpha'][k]/p['alpha'].sum()
        values+=pi*t.pdf(xs,df,loc=p['mean'][k,0],scale=scale)
        fixed=np.sqrt(np.linalg.inv(p['nu'][k]*p['w'][k])[0,0])
        plug+=pi*norm.pdf(xs,loc=p['mean'][k,0],scale=fixed)
    def pt(x,v):return coord(x,-2.22+7.5*v)
    base=VGroup(Line(coord(-3.7,-2.22),coord(3.65,-2.22),color=MUTED,stroke_width=1.2))
    for x in X[:,0]:
        base.add(Line(pt(x,-.025),pt(x,0),color=MUTED,stroke_width=.8,stroke_opacity=.45))
    for vv,color,visible in ((plug,PURPLE,True),(values,GREEN,show_average)):
        line=VMobject().set_points_as_corners([pt(x,v) for x,v in zip(xs,vv)])
        base.add(line.set_stroke(color,width=3,opacity=1 if visible else 0))
    green_legend=J('緑: 分布で平均',18,GREEN).move_to([.25,2.35,0])
    green_legend.set_opacity(1 if show_average else 0)
    base.add(J('紫: パラメータを固定',18,PURPLE).move_to([-3.5,2.35,0]),
             green_legend,
             tex('x_1',22,MUTED).move_to(coord(3.8,-2.22)))
    return base

def tail_bars():
    z=3.5
    p=posterior(X,HISTORY[30]['r'])
    predicted=fixed=0.
    for k in range(K):
        df=p['nu'][k]+1-2
        scale=np.sqrt((np.linalg.inv(p['w'][k])*(1+p['beta'][k])/(df*p['beta'][k]))[0,0])
        plug=np.sqrt(np.linalg.inv(p['nu'][k]*p['w'][k])[0,0])
        weight=p['alpha'][k]/p['alpha'].sum()
        predicted+=weight*t.pdf(z,df,loc=p['mean'][k,0],scale=scale)
        fixed+=weight*norm.pdf(z,loc=p['mean'][k,0],scale=plug)
    group=VGroup(J('右端の密度',19,MUTED).move_to([4.28,1.48,0]),
                 Line([2.9,-1.65,0],[6.2,-1.65,0],color=MUTED,stroke_width=1.2))
    for xc,value,color,label in ((3.5,fixed,PURPLE,'固定'),(5.,predicted,GREEN,'予測')):
        group.add(Line([xc,-1.65,0],[xc,-1.65+2.4*value/predicted,0],
                       color=color,stroke_width=25),
                  J(label,17,color).move_to([xc,-1.97,0]))
    group.add(J(f'{predicted/fixed:.2f}倍',20,GREEN).move_to([4.3,-2.42,0]))
    return group

class PRML102VariationalMixture(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si in range(8):
            self.begin(si)
            self.setup_scene(si)
            for bi in range(len(SCENES[si]['beats'])):
                self.run_beat(si,bi)
            assert self.beat_index==len(SCENES[si]['beats'])
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml102_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def setup_scene(self,si):
        self.note=None;self.formula=None
        if si==6:
            self.density=predictive()
            self.add(self.density)
            self.current=VGroup()
            self.side=bars(HISTORY[30]['weights'])
        else:
            self.add(graph_axes(),dots())
            self.current=contours(HISTORY[0 if si in (0,1,2,3,4,7) else 30])
            self.side=recap_92() if si==1 else bars(HISTORY[0 if si in (0,1,2,3,4,7) else 30]['weights'])
        self.side_mode='recap92' if si==1 else 'bars'
        self.add(self.current,self.side)

    def run_beat(self,si,bi):
        note=J(NOTES[si][bi],18,MUTED).move_to([4.3,2.48,0])
        recap=(si,bi) in ((1,0),(3,2),(3,3))
        if note.width>(3.7 if recap else 4.2):note.scale((3.7 if recap else 4.2)/note.width)
        if self.note is not None:self.remove(self.note)
        self.note=note;self.add(note)
        if hasattr(self,'recap_border') and self.recap_border is not None:
            self.remove(self.recap_border)
        self.recap_border=None
        if recap:
            self.recap_border=Rectangle(width=4.0,height=5.35,color=YELLOW,
                                        stroke_width=1.0).move_to([4.5,.08,0])
            self.add(self.recap_border)
        formula=FORMULAS[si][bi]
        if self.formula is not None:self.remove(self.formula)
        self.formula=None
        if formula:
            f=tex(formula,22,YELLOW).move_to([4.3,1.94,0])
            max_width=3.7 if recap else 4.15
            if f.width>max_width:f.scale(max_width/f.width)
            self.formula=f;self.add(f)
        iteration=ITERATIONS[si][bi]
        if si==5 and bi==0:iteration=0
        state=HISTORY[iteration]
        if si==5 and bi==3:
            state=STRONG_HISTORY[30]
        side_mode=('responsibility' if si==2 else
                   'recap92' if si==1 and bi==0 else
                   'recap22' if si==3 and bi==2 else
                   'recap23' if si==3 and bi==3 else
                   'bound' if si==4 else
                   'tail' if si==6 and bi>=3 else 'bars')
        if si==2:
            probe=[[-.25,0.],[-.08,.12],[0.,.05],[.05,.05],[.95,.4],[0.,.05],[0.,.05]][bi]
            probe_posterior=posterior(X,HISTORY[0]['r'])
            side=responsibility_bars(probe,probe_posterior)
            if bi==2:
                self.remove(self.side)
                self.side=raw_score_bars(probe,probe_posterior)
                self.side_mode='raw'
                self.add(self.side)
            probe_mob=VGroup(Dot(coord(*probe),radius=.075,color=YELLOW),
                             Circle(radius=.13,color=YELLOW,stroke_width=1.6).move_to(coord(*probe)))
            if not hasattr(self,'probe'):
                self.probe=probe_mob;self.add(self.probe)
                probe_animation=Wait()
            else:probe_animation=Transform(self.probe,probe_mob)
        else:
            probe_animation=Wait()
            side=(recap_92() if si==1 and bi==0 else
                  recap_22(True) if si==3 and bi==2 else
                  recap_23(.55) if si==3 and bi==3 else
                  bound_chart(iteration) if si==4 else
                  tail_bars() if si==6 and bi>=3 else bars(state['weights']))
            if si==3 and bi==2:
                self.remove(self.side)
                self.side=recap_22(False)
                self.side_mode='recap22'
                self.add(self.side)
            if si==3 and bi==3:
                self.remove(self.side)
                self.side=recap_23(1.1)
                self.side_mode='recap23'
                self.add(self.side)
        target=contours(state) if si!=6 else VGroup()
        movements=[]
        if si!=6:movements.append(Transform(self.current,target))
        side_transform=(self.side_mode==side_mode or
                        (self.side_mode=='raw' and side_mode=='responsibility'))
        cues=self.beat_cues()
        action_span=cues[-1]['end']-cues[0]['start']
        fade_time=min(1.2,action_span/3)
        movements.append(Transform(self.side,side) if side_transform else
                         Succession(FadeOut(self.side,run_time=fade_time*.4),
                                    FadeIn(side,run_time=fade_time*.6),
                                    Wait(action_span-fade_time)))
        if si==2 and bi>0:movements.append(probe_animation)
        if si==6:
            # A highlighted candidate location moves while the predictive curves remain visible.
            if not hasattr(self,'new_point'):
                self.new_point=Dot(coord(0.,-2.22),radius=.075,color=YELLOW)
                self.add(self.new_point)
            target_x=[-.9,-.45,0.,3.5,3.5,3.5][bi]
            movements.append(self.new_point.animate.move_to(coord(target_x,-2.22)))
            if bi==3:movements.append(Transform(self.density,predictive(True)))
        self.beat(*movements)
        if not side_transform:self.side=side
        self.side_mode=side_mode
