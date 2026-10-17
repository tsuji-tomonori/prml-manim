"""PRML 10.4: one ambiguous count connects global updates to local messages."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import poisson, gamma
from manim import *
from poisson_model import COUNTS, PROBE_INDEX, A0, B0, history
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex

BG = '#10141F'
BLUE = ManimColor('#58C4DD')
GREEN = ManimColor('#83C167')
YELLOW = ManimColor('#FFFF00')
PURPLE = ManimColor('#9A72AC')
RED = ManimColor('#FC6255')
MUTED = ManimColor('#A8B2C5')
STATES = history()
ITERATIONS = [
    [0, 0, 0, 0, 18, 18, 18],
    [0, 0, 0, 0, 0, 0, 0],
    [18, 18, 18, 18, 18, 18, 18],
    [18, 18, 18, 18, 18, 18, 18],
    [0, 0, 18, 18, 18, 18, 18],
    [0, 1, 2, 3, 5, 18, 18],
    [18, 18, 18, 18, 18, 18, 18],
    [0, 18, 18, 18, 18, 18, 18],
]
NOTES = [
    ['10.3の問いを広げる','出所不明の12件','3回の出所を予想','最初の仮の重み','更新後の意外な重み','同じ条件で追う','どの量を渡す？'],
    ['10.2の所属と山','復習: 2.4 指数型分布族','観測と見えない所属','潜在変数とパラメータ','所属が見える場合','所属を消すと混合','残す量は何か'],
    ['率と回数','十分統計量','ポアソンの式','分布で平均する','平均と対数の順序','2台の得点を正規化','誘導された因子化'],
    ['3回の分配','期待値の補足','12件を足す','担当件数を足す','今回の集計','訂正された一般形','共役事前へ'],
    ['率の事前分布','形状と率の条件','回数の和を渡す','担当件数を渡す','2つの分布が動く','更新後の平均','所属へ返す'],
    ['初期の責務','率を更新','所属を更新','もう1巡','5巡目','18巡目','距離だけでは決まらない'],
    ['有向グラフ','復習: 8.2 マルコフブランケット','親は混合比','子と子の他の親','近所の項だけを足す','変分メッセージ伝播','共役指数型の条件'],
    ['同じ12件','最初と更新後','2つの率と3回','往復で渡した量','一般の共通形','近似の範囲','次の局所近似へ'],
]
FORMULAS = [
    ['', '', '', r'q_{8,2}=0.39', r'q_{8,2}\simeq0.77', '', ''],
    ['', r'p=h\,g\,\exp(\eta^Tu)', '', '', '', '', ''],
    [r'\eta=\log\lambda', r'u(x)=x', r'p(x\mid\lambda)=e^{-\lambda}\lambda^x/x!',
     r'E[\lambda],\ E[\log\lambda]', r'E[\log\lambda]\ne\log E[\lambda]',
     r'r_{n,k}\propto\exp\{x_nE[\log\lambda_k]-E[\lambda_k]\}', r'q(Z)=\prod_nq(z_n)'],
    ['', r'E[r_{n,k}x_n]', r'\sum_n r_{n,k}x_n', r'N_k=\sum_n r_{n,k}', '',
     r'\nu_N\chi_N=\nu_0\chi_0+\sum_nE[u_n]', ''],
    [r'\lambda_k\sim\mathrm{Gamma}(a_{0k},b_{0k})', '', r'a_k=a_{0k}+\sum_n r_{n,k}x_n',
     r'b_k=b_{0k}+\sum_n r_{n,k}', '', r'E[\lambda_k]=a_k/b_k', ''],
    [r'q_{8,2}=0.39', '', '', '', r'q_{8,2}\simeq0.76', r'q_{8,2}\simeq0.77', ''],
    ['', '', '', '', r'\log q_j^*=E[\log p_j+\sum_c\log p_c]+C',
     '', ''],
    ['', r'0.39\ \longrightarrow\ 0.77', r'E[\lambda_1]=0.85,\quad E[\lambda_2]=5.44',
     '', '', '', ''],
]


def J(s, size=20, color=WHITE):
    return jp(s, size, color)


def xpos(x):
    return -5.65 + .765 * x


def left_axes():
    g = VGroup(Line([xpos(0)-.15,-1.95,0],[xpos(8)+.22,-1.95,0],color=MUTED,stroke_width=1.4),
               J('同じ12件の回数',19,MUTED).move_to([-2.62,2.33,0]),
               J('平均率のポアソン形',16,MUTED).move_to([-2.62,1.75,0]))
    for k in range(9):
        xx=xpos(k)
        g.add(Line([xx,-1.95,0],[xx,-2.02,0],color=MUTED,stroke_width=1.1),
              J(str(k),16,MUTED).move_to([xx,-2.26,0]))
    g.add(J('回数',16,MUTED).move_to([.85,-2.26,0]))
    return g


def data_visual(state, reveal=True):
    """Curves and all 12 observations share the same horizontal count scale."""
    g=VGroup()
    for k,color in enumerate((BLUE,PURPLE)):
        points=[[xpos(x),-.18+3.1*poisson.pmf(x,state['mean'][k]),0] for x in range(9)]
        line=VMobject().set_points_as_corners(points).set_stroke(color,width=3,opacity=.95 if reveal else .22)
        g.add(line)
    g.add(Line([xpos(0)-.10,-.18,0],[xpos(8)+.12,-.18,0],color=MUTED,stroke_width=1.1))
    seen={}
    for i,x in enumerate(COUNTS.astype(int)):
        rank=seen.get(x,0);seen[x]=rank+1
        color=interpolate_color(BLUE,PURPLE,float(state['r'][i,1])) if reveal else MUTED
        y=-1.60+rank*.23
        marker=[xpos(x),y,0]
        g.add(Dot(marker,radius=.092 if i==PROBE_INDEX else .069,
                  color=YELLOW if i==PROBE_INDEX and not reveal else color))
        if i==PROBE_INDEX:
            g.add(Circle(radius=.145,color=YELLOW,stroke_width=2).move_to(marker))
    q=float(state['r'][PROBE_INDEX,1])
    left=xpos(3)-.27
    g.add(Line([left,-1.32,0],[left+.54*(1-q),-1.32,0],
               color=BLUE if reveal else MUTED,stroke_width=8),
          Line([left+.54*(1-q),-1.32,0],[left+.54,-1.32,0],
               color=PURPLE if reveal else MUTED,stroke_width=8))
    g.add(J('3回',17,YELLOW).move_to([xpos(3),-2.69,0]))
    return g


def side_base(title):
    return VGroup(J(title,20,MUTED).move_to([4.43,1.08,0]))


def side_prob(state, value=None, title='3回の所属の重み'):
    q=float(state['r'][PROBE_INDEX,1] if value is None else value)
    g=side_base(title)
    for y,label,color,prob in [(0.3,'低頻度',BLUE,1-q),(-.85,'高頻度',PURPLE,q)]:
        g.add(J(label,18,color).move_to([3.06,y+.25,0]),
              Line([2.92,y-.08,0],[2.92+2.9*prob,y-.08,0],color=color,stroke_width=22),
              J(f'{prob:.2f}',20,color).move_to([5.95,y-.08,0]))
    return g


def side_family(stage):
    g=side_base('共通形から計数へ')
    labels=[('回数 3',BLUE),('対数率',PURPLE),('掛ける',YELLOW),('正の重み',GREEN)]
    for i,(label,color) in enumerate(labels):
        g.add(J(label,19,color).move_to([3.04+i*.86,-.2,0]))
    g.add(Arrow([3.0,-.63,0],[5.66,-.63,0],buff=0,color=YELLOW,stroke_width=2.2),
          J('指数 → 正規化',19,GREEN).move_to([4.35,-1.25,0]))
    captions=['10.2の責務','指数で正の重み','見えない所属','率は2つ','所属が見えると','混合は別の形','必要な量を探す']
    g.add(J(captions[stage],18,BLUE if stage<3 else GREEN).move_to([4.35,-2.08,0]))
    return g


def side_scores(state, stage):
    if stage==4:
        g=side_base('説明用の例: 順序の違い')
        g.add(J('率の候補 1, 4',19,BLUE).move_to([4.38,.38,0]),
              J('対数を先に',18,YELLOW).move_to([3.05,-.22,0]),
              J('0, 1.39 → 平均 0.69',18,GREEN).move_to([5.08,-.22,0]),
              J('平均を先に',18,YELLOW).move_to([3.05,-1.12,0]),
              J('2.5 → 対数 0.92',18,PURPLE).move_to([5.08,-1.12,0]),
              J('2候補は等しい重み',16,MUTED).move_to([4.4,-2.08,0]))
        return g
    g=side_prob(state,title='率の分布から所属へ')
    captions=['対数率','回数','ポアソンの重み','分布で平均','平均の順序','正規化後の責務','各件で因子化']
    g.add(J(captions[stage],18,YELLOW if stage<5 else GREEN).move_to([4.35,-1.96,0]))
    return g


def side_stats(state, stage):
    g=side_base('責務を掛けて集計')
    kinds=[('担当件数',state['effective_n'],8),('回数の和',state['effective_sum'],35)]
    key=0 if stage in (0,3) else 1
    title,values,limit=kinds[key]
    g.add(J(title,18,YELLOW).move_to([4.35,.48,0]))
    for y,label,color,val in [(0,'低頻度',BLUE,values[0]),(-.92,'高頻度',PURPLE,values[1])]:
        g.add(J(label,17,color).move_to([3.06,y+.1,0]),
              Line([2.91,y-.28,0],[2.91+2.72*val/limit,y-.28,0],color=color,stroke_width=20),
              J(f'{val:.2f}',19,color).move_to([5.96,y-.28,0]))
    if stage==1:
        g.add(J('3 × 0.77 ≈ 2.30',18,YELLOW).move_to([4.35,-2.06,0]))
    elif stage==4:
        g.add(J('担当 6.44 / 5.56 件',17,GREEN).move_to([4.35,-2.06,0]))
    return g


def side_gamma(state, prior=False):
    a,b=(A0,B0) if prior else (state['a'],state['b'])
    g=side_base('率の事前分布' if prior else '更新後の率の分布')
    x0,x1=2.7,6.20;y0=-1.43
    g.add(Line([x0,y0,0],[x1,y0,0],color=MUTED,stroke_width=1.2))
    xs=np.linspace(.02,10.,100)
    ys=np.vstack([gamma.pdf(xs,a[k],scale=1/b[k]) for k in range(2)])
    scale=1.98/max(ys.max(),.1)
    for k,color in enumerate((BLUE,PURPLE)):
        points=[[x0+(x1-x0)*x/10,y0+scale*y,0] for x,y in zip(xs,ys[k])]
        g.add(VMobject().set_points_as_corners(points).set_stroke(color,width=3))
    for k,xx in enumerate([0,5,10]):
        pos=x0+(x1-x0)*xx/10
        g.add(Line([pos,y0,0],[pos,y0-.08,0],color=MUTED,stroke_width=1),
              J(str(xx),15,MUTED).move_to([pos,y0-.31,0]))
    g.add(J('発生率',16,MUTED).move_to([4.35,-2.12,0]))
    return g


def side_update(state, index):
    g=side_prob(state,title=f'交互更新 {index}巡目')
    g.add(J(f'平均率  {state["mean"][0]:.2f}  /  {state["mean"][1]:.2f}',18,GREEN)
          .move_to([4.4,-2.07,0]))
    return g


def side_graph(stage):
    g=side_base('3回の所属の近所')
    nodes={'mix':([3.15,.3,0],'混合比',BLUE),
           'rate':([5.6,.3,0],'発生率',PURPLE),
           'z':([3.15,-1.13,0],'所属',YELLOW),
           'x':([5.6,-1.13,0],'観測 3',GREEN)}
    def edge(a,b):
        return Arrow(nodes[a][0],nodes[b][0],buff=.19,color=MUTED,stroke_width=2.3,tip_length=.14)
    g.add(edge('mix','z'),edge('rate','x'),edge('z','x'))
    for key,(position,label,color) in nodes.items():
        active=stage==0 or stage==1 or (stage==2 and key=='mix') or (stage==3 and key in ('rate','x','z')) or stage>=4
        label_pos={'mix':[3.15,.73,0],'rate':[5.6,.73,0],
                   'z':[2.55,-1.13,0],'x':[5.6,-1.59,0]}[key]
        g.add(Dot(position,radius=.16,color=color if active else MUTED),
              J(label,16,color if active else MUTED).move_to(label_pos))
    return g


def side_visual(si,bi,state):
    if si==0:
        if bi<3:
            g=side_base('2台の候補')
            g.add(J('低頻度 ？  高頻度',23,YELLOW).move_to([4.4,-.2,0]))
            return g
        return side_prob(state,title='3回をどちらへ？')
    if si==1:return side_family(bi)
    if si==2:return side_scores(state,bi)
    if si==3:return side_stats(state,bi)
    if si==4:return side_gamma(state,prior=bi<2)
    if si==5:return side_update(state,ITERATIONS[si][bi])
    if si==6:return side_graph(bi)
    return side_prob(state,title='同じ3回の答え')


class PRML104ExponentialFamily(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for si,story in enumerate(SCENES):
            self.begin(si)
            self.add(left_axes())
            self.current=data_visual(STATES[ITERATIONS[si][0]],reveal=si!=0)
            self.side=side_visual(si,0,STATES[ITERATIONS[si][0]])
            self.note=None;self.formula=None
            self.add(self.current,self.side)
            for bi in range(len(story['beats'])):
                self.run_beat(si,bi)
            self.timeline[-1]['end']=float(self.time)
        (Path(config.media_dir)/'prml104_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def run_beat(self,si,bi):
        if self.note is not None:self.remove(self.note)
        self.note=J(NOTES[si][bi],19,MUTED).move_to([4.4,2.48,0])
        if self.note.width>4.2:self.note.scale(4.2/self.note.width)
        self.add(self.note)
        if self.formula is not None:self.remove(self.formula)
        self.formula=None
        formula=FORMULAS[si][bi]
        if formula:
            self.formula=tex(formula,23,YELLOW).move_to([4.4,1.79,0])
            if self.formula.width>4.1:self.formula.scale(4.1/self.formula.width)
            self.add(self.formula)
        state=STATES[ITERATIONS[si][bi]]
        target=data_visual(state,reveal=si!=0 or bi>=3)
        side=side_visual(si,bi,state)
        # Settle labels quickly; let the data and density marks move for the
        # rest of the spoken beat.  A full-beat Text morph is hard to read.
        total=self.durations[self.beat_index]
        self.beat(phases=[('panel',.6,lambda:Succession(FadeOut(self.side),FadeIn(side))),
                          ('counts',total-.6,lambda:Transform(self.current,target))])
        self.side=side
