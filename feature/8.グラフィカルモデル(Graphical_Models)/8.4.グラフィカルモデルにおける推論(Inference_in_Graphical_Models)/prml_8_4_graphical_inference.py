"""PRML 8.4: exact inference on a small chain, then its tree generalization."""
import json
from pathlib import Path
import numpy as np
from manim import *
from chain_model import TABLES, candidates, sum_messages
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex

BLUE = ManimColor('#58C4DD')
YELLOW = ManimColor('#FFFF00')
GREEN = ManimColor('#83C167')
PURPLE = ManimColor('#9A72AC')
RED = ManimColor('#FC6255')
MUTED = ManimColor('#A8B2C5')
BG = '#10141F'


def J(value, size=24, color=WHITE):
    return jp(str(value), size, color)


def bar(value, maximum, color, x, y, width=3.0, height=.25):
    # A semantic bar: its length encodes a weight or a probability.
    length = width * value / maximum
    rect = Rectangle(width=length, height=height, stroke_width=0,
                     fill_color=color, fill_opacity=.86).move_to([x+length/2,y,0])
    number = J(f'{value:g}', 20, color).next_to(rect, RIGHT, buff=.13)
    return VGroup(rect, number)


def trace_paths(paths, color=GREEN):
    # Highlight an edge without scaling it across the endpoint state labels.
    return AnimationGroup(*[ShowPassingFlash(path.copy().set_stroke(color, width=6),
                                             time_width=.55) for path in paths],
                          lag_ratio=.18)


class PRML84GraphicalInference(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.manifest = {e['id']: e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline = []
        for i, method in enumerate((self.question, self.observe, self.factors,
                                    self.left_message, self.right_message,
                                    self.tree, self.maximize, self.backtrack, self.return_to_chain)):
            self.begin(i)
            self.note = None
            method()
            assert self.beat_index == len(SCENES[i]['beats'])
            self.timeline[-1]['end'] = float(self.time)
        (Path(config.media_dir) / 'prml84_timeline.json').write_text(
            json.dumps(self.timeline, ensure_ascii=False, indent=2)+'\n')

    def note_anim(self, message, color=MUTED, math=False):
        new = tex(message, 28, color) if math else J(message, 23, color)
        if new.width > 12.2:
            new.scale_to_fit_width(12.2)
        new.move_to([0,-2.38,0])
        if self.note is not None:
            self.remove(self.note)
        self.note = new
        return FadeIn(new, rate_func=lambda t: smooth(min(1,6*t)))

    def say(self, message, *anims, color=MUTED, math=False):
        self.beat(self.note_anim(message,color,math), *anims)

    def tag(self, message, color=WHITE, x=0):
        obj = J(message, 22, color).move_to([x,2.48,0])
        self.add(obj)
        return obj

    def graph(self, y=.55, positions=None):
        xs = positions or [-4.8,-1.6,1.6,4.8]
        self.points = [np.array([x,y,0]) for x in xs]
        wires = VGroup(*[Line(self.points[i]+RIGHT*.32,self.points[i+1]+LEFT*.32,
                              color=MUTED,stroke_width=2) for i in range(3)])
        circles = VGroup(*[Circle(radius=.33,stroke_color=BLUE if i==3 else WHITE,
                                  stroke_width=2,fill_opacity=0).move_to(pt)
                           for i,pt in enumerate(self.points)])
        labels = VGroup(*[tex(f'x_{i+1}',25,BLUE if i==3 else WHITE).next_to(c,UP,buff=.17)
                          for i,c in enumerate(circles)])
        self.state_labels = [J('1' if i==3 else '?',24,BLUE if i==3 else WHITE).move_to(pt).set_z_index(5)
                             for i,pt in enumerate(self.points)]
        group=VGroup(wires,circles,labels,*self.state_labels)
        self.add(group)
        self.circles=circles
        return group

    def set_state(self, i, value, color=GREEN, rate_func=smooth):
        # Keep the glyph legible throughout the narration beat. Morphing '?' into
        # a digit produces an unreadable intermediate shape at 480p.
        switch = lambda t: 0 if rate_func(t) < .08 else 1
        return Transform(self.state_labels[i],J(value,24,color).move_to(self.points[i]).set_z_index(5),
                         rate_func=switch)

    def mark_observed(self):
        mark=J('観測',20,BLUE).next_to(self.circles[3],RIGHT,buff=.20)
        self.add(mark)
        return mark

    def table(self, index, x, y=-.85):
        matrix=TABLES[index]
        cells=VGroup()
        for row in range(2):
            for col in range(2):
                px=x+(col-.5)*.53
                py=y+(.5-row)*.47
                square=Square(side_length=.50,stroke_color=MUTED,stroke_width=1,
                              fill_color=BG,fill_opacity=1).move_to([px,py,0])
                value=J(int(matrix[row,col]),21).move_to(square).set_z_index(5)
                cells.add(VGroup(square,value))
        row_labels=VGroup(*[J(row,16,MUTED).move_to([x-.76,y+(.5-row)*.47,0])
                            for row in range(2)])
        col_labels=VGroup(*[J(col,16,MUTED).move_to([x+(col-.5)*.53,y+.65,0])
                            for col in range(2)])
        name=tex(r'\psi_{'+str(index+1)+','+str(index+2)+'}',24).move_to([x,y+1.02,0])
        return VGroup(cells,name,row_labels,col_labels)

    def candidate_rows(self, y_top=1.84, x=-4.95, gap=.46):
        ordered=sorted(candidates(),key=lambda row:(-row[0][1],row[0]))
        groups=[]
        for i,(state,weight) in enumerate(ordered):
            y=y_top-i*gap
            squares=VGroup(*[Square(side_length=.26,stroke_width=1.2,
                                    stroke_color=BLUE if j==3 else MUTED,
                                    fill_color=(BLUE if v else BG),fill_opacity=1).move_to([x+j*.34,y,0])
                             for j,v in enumerate(state)])
            weightbar=bar(weight,48,GREEN if state[1] else PURPLE,x+1.75,y,width=2.65,height=.21)
            groups.append(VGroup(squares,weightbar))
        return VGroup(*groups),ordered

    def question(self):
        self.graph()
        self.mark_observed()
        self.say('8.3：局所更新は全体の最良を保証しない',Indicate(self.circles[1],color=YELLOW))
        self.say('4個の二値ランプをつないだ鎖',LaggedStart(*[Indicate(c,color=BLUE) for c in self.circles],lag_ratio=.22))
        self.say('右端だけ観測：1',Indicate(self.state_labels[3],color=BLUE))
        q=J('2番目は？',24,YELLOW).move_to([-1.6,-.55,0]);self.add(q)
        self.say('予想：2番目だけと4個全部',Indicate(self.circles[1],color=YELLOW))
        self.remove(q)
        bars=VGroup(bar(.658,1,GREEN,-4.6,-1.3,width=3.0),bar(.342,1,PURPLE,1.2,-1.3,width=3.0))
        labs=VGroup(J('2番目＝1',20,GREEN).move_to([-3.1,-.82,0]),J('2番目＝0',20,PURPLE).move_to([2.7,-.82,0]))
        late=lambda t:smooth(max(0,min(1,(t-.52)/.48)))
        early=lambda t:smooth(min(1,t/.52))
        self.say('単独は1、全体の最良では0',FadeIn(bars,rate_func=early),
                 FadeIn(labs,rate_func=early),self.set_state(0,1,rate_func=late),
                 self.set_state(1,0,rate_func=late),self.set_state(2,0,rate_func=late),color=GREEN)
        self.say('同じ観測で、なぜ違う？',Indicate(bars[0],color=GREEN),
                 Indicate(self.circles[1],color=RED),color=RED)
        self.say('足す答えと、1つ選ぶ答え',Indicate(bars[0],color=GREEN),Indicate(self.circles[1],color=YELLOW))

    def observe(self):
        self.graph(y=1.1)
        self.mark_observed()
        recap=J('復習: 1.2 確率の正規化',21,WHITE).move_to([-4.65,2.4,0])
        frame=SurroundingRectangle(recap,buff=.14,stroke_color=YELLOW,stroke_width=1.3)
        weights=VGroup(bar(2,5,BLUE,-3.7,-.55,width=2.8),bar(3,5,PURPLE,.7,-.55,width=2.8))
        normalized=tex(r'\frac25+\frac35=1',32,GREEN).move_to([0,-1.42,0])
        self.say('復習 1.2：正の重みを合計で割る',FadeIn(recap),Create(frame),
                 FadeIn(weights),FadeIn(normalized,rate_func=lambda t:smooth(min(1,2*t))),color=YELLOW)
        self.remove(normalized)
        count=J('16通り',28,WHITE).move_to([0,-1.45,0]);self.add(count)
        self.say('観測前は2の4乗＝16通り',Indicate(count,color=YELLOW))
        self.remove(frame,recap,weights)
        half=J('右端が1の8通り',27,BLUE).move_to([0,-1.45,0])
        self.say('右端が0の候補を除く',
                 Transform(count,half,rate_func=lambda t: 0 if t < .12 else 1),
                 Indicate(self.state_labels[3],color=BLUE))
        arrow=Arrow(self.points[3]+DOWN*.35,self.points[2]+DOWN*.35,color=YELLOW,buff=.05)
        self.say('観測に合う候補だけ残す',GrowArrow(arrow))
        self.remove(count)
        formula=tex(r'p(x\mid y)=\frac{p(y\mid x)p(x)}{p(y)}',31,GREEN).move_to([0,-.8,0]);self.add(formula)
        self.say('ベイズの更新も、合う重みを集めて割る',Indicate(formula,color=YELLOW))
        self.remove(formula,arrow)
        count2=J('長い鎖では候補が急増',24,PURPLE).move_to([0,-.8,0])
        self.say('候補を全部並べる？',FadeIn(count2))
        self.say('隣同士の3枚の表へ',LaggedStart(*[Indicate(c,color=YELLOW) for c in self.circles],lag_ratio=.25))

    def factors(self):
        self.graph(y=1.05)
        tables=[self.table(i,[-3.2,0,3.2][i],y=-.85) for i in range(3)]
        self.say('隣同士に1枚ずつ因子表',LaggedStart(*[FadeIn(t) for t in tables],lag_ratio=.2))
        self.say('行＝左、列＝右',Indicate(tables[0][0][1][1],color=YELLOW))
        self.say('1、0、0、1 → 4×4×3＝48',self.set_state(0,1),self.set_state(1,0),
                 self.set_state(2,0),Indicate(tables[0][0][2][1],color=YELLOW),
                 Indicate(tables[1][0][0][1],color=YELLOW),Indicate(tables[2][0][1][1],color=YELLOW),color=GREEN)
        self.say(r'w(x)=\psi_{12}\psi_{23}\psi_{34}',Indicate(tables[0][0][2][1],color=GREEN),
                 Indicate(tables[1][0][0][1],color=GREEN),Indicate(tables[2][0][1][1],color=GREEN),
                 color=YELLOW,math=True)
        ten=J('10個 → 1024通り',24,PURPLE).move_to([0,-1.82,0]);self.add(ten)
        self.say('候補の総数は指数的に増える',Indicate(ten,color=PURPLE))
        self.say(r'ab+ac=a(b+c)',Indicate(tables[0][1],color=YELLOW),math=True)
        self.say('端の2候補を先に足す',Indicate(tables[0][0][0][1],color=YELLOW),Indicate(tables[0][0][2][1],color=YELLOW))

    def left_message(self):
        self.graph(y=1.0)
        table=self.table(0,-3.2,-.55);self.add(table)
        formula=tex(r'\mu_L(x_2)=\sum_{x_1}\psi_{12}(x_1,x_2)',29,YELLOW).move_to([2.1,-1.25,0])
        self.say('Σ：固定した列を縦に足す',FadeIn(formula),Indicate(table[0][0][1],color=YELLOW),
                 Indicate(table[0][2][1],color=YELLOW),color=YELLOW)
        self.remove(formula)
        b0=bar(5,8,BLUE,-3.9,-1.48,width=2.4)
        self.say('2番目＝0：1＋4＝5',FadeIn(b0),Indicate(table[0][0][1],color=YELLOW),Indicate(table[0][2][1],color=YELLOW))
        b1=bar(8,8,GREEN,.6,-1.48,width=2.4)
        self.say('2番目＝1：4＋4＝8',FadeIn(b1),Indicate(table[0][1][1],color=YELLOW),Indicate(table[0][3][1],color=YELLOW))
        arrow=Arrow(self.points[0]+DOWN*.32,self.points[1]+DOWN*.32,buff=.08,color=GREEN)
        self.say('左の札：(5, 8)',GrowArrow(arrow),color=GREEN)
        self.say('札は確率ではなく部分和',Indicate(b0,color=BLUE),Indicate(b1,color=GREEN))
        self.say('次の表へ掛けて、また足す',Indicate(self.circles[2],color=YELLOW))
        self.say('右端の観測からも札を戻す',Indicate(self.state_labels[3],color=BLUE))

    def right_message(self):
        self.graph(y=1.05)
        left=VGroup(bar(5,18,BLUE,-5.3,-.8,width=1.5),bar(8,18,BLUE,-5.3,-1.35,width=1.5))
        right=VGroup(bar(15,18,PURPLE,1.2,-.8,width=1.5),bar(18,18,PURPLE,1.2,-1.35,width=1.5))
        self.say('観測した1の列だけ使う',Indicate(self.state_labels[3],color=BLUE))
        table=self.table(2,3.2,-.6)
        fifteen=J('2番目＝0 → 15',22,PURPLE).move_to([-2.6,-1.35,0])
        self.say('右から：(3, 3) → 15',FadeIn(table),FadeIn(fifteen))
        self.remove(table)
        middle_table=self.table(1,0,-.4)
        self.remove(fifteen)
        self.say('2番目＝1の右の札は18',
                 FadeIn(middle_table,rate_func=lambda t:smooth(min(1,3*t))),
                 FadeIn(right,rate_func=lambda t:smooth(min(1,3*t))))
        formula=tex(r'p(x_2\mid x_4=1)\propto\mu_L(x_2)\mu_R(x_2)',27,YELLOW).move_to([0,-1.98,0])
        self.say('0は75、1は144',FadeIn(left),FadeIn(formula),
                 Indicate(self.circles[1],color=YELLOW))
        self.remove(formula)
        result=VGroup(bar(75,144,PURPLE,-4.2,-1.98,width=2.5,height=.20),
                      bar(144,144,GREEN,.5,-1.98,width=2.5,height=.20))
        self.say('合計219。1の確率は144/219',FadeIn(result),color=GREEN)
        self.remove(left,right,middle_table,result)
        rows,_=self.candidate_rows(y_top=.45,gap=.31)
        self.say('8通りを直接数えても同じ',FadeIn(rows),color=GREEN)
        self.remove(rows)
        factors=VGroup(*[Square(side_length=.16,stroke_color=YELLOW,fill_color=YELLOW,
                                fill_opacity=1).move_to([(self.points[i][0]+self.points[i+1][0])/2,1.05,0])
                         for i in range(3)])
        self.say('次は木の枝へ',FadeIn(factors),Indicate(self.circles[1],color=YELLOW))

    def tree(self):
        self.graph(y=1.13)
        squares=VGroup(*[Square(side_length=.18,stroke_color=YELLOW,fill_color=YELLOW,
                                fill_opacity=1).move_to([(self.points[i][0]+self.points[i+1][0])/2,1.13,0])
                         for i in range(3)])
        self.say('丸＝変数、四角＝因子',FadeIn(squares),color=YELLOW)
        branch=Line(self.points[1]+DOWN*.3,[-1.6,-.80,0],color=MUTED)
        leaf=Circle(radius=.3,stroke_color=BLUE,fill_color=BG,fill_opacity=1).move_to([-1.6,-1.08,0])
        branch_square=Square(side_length=.18,stroke_color=YELLOW,fill_color=YELLOW,fill_opacity=1).move_to([-1.6,-.45,0])
        self.say('枝を足しても、輪がなければ木',Create(branch),FadeIn(leaf),FadeIn(branch_square))
        incoming=Dot(color=YELLOW,radius=.09).move_to(leaf)
        self.say('因子は他の状態を足して送る',MoveAlongPath(incoming,Line(leaf.get_center(),self.points[1])),
                 Indicate(branch_square,color=YELLOW))
        self.say('変数は他の枝の札を掛ける',Indicate(self.circles[1],color=GREEN))
        left=Dot(color=BLUE,radius=.09).move_to(self.points[0])
        self.say('葉から中心へ集める',MoveAlongPath(left,Line(self.points[0],self.points[2])),
                 Indicate(self.circles[2],color=GREEN))
        back=Dot(color=PURPLE,radius=.09).move_to(self.points[3])
        self.say('逆方向にも各辺を1回通す',MoveAlongPath(back,Line(self.points[3],self.points[0])))
        self.say('次は全体の最良の1本',Indicate(self.circles[1],color=YELLOW))

    def trellis(self):
        xs=[-4.8,-1.6,1.6,4.8]
        ys=[1.40,-.15]
        cells=VGroup()
        for i,x in enumerate(xs):
            for v,y in enumerate(ys):
                shape=Square(side_length=.48,stroke_color=BLUE if i==3 else MUTED,
                             stroke_width=2,fill_color=BG,fill_opacity=1).move_to([x,y,0])
                number=J(v,22,BLUE if i==3 else WHITE).move_to(shape).set_z_index(5)
                cells.add(VGroup(shape,number))
        labels=VGroup(*[tex(f'x_{i+1}',24).move_to([x,2.2,0]) for i,x in enumerate(xs)])
        rows=VGroup(J('0',20,MUTED).move_to([-5.62,ys[0],0]),
                    J('1',20,MUTED).move_to([-5.62,ys[1],0]))
        self.add(cells,labels,rows)
        self.trellis_cells=cells
        self.trellis_pts=lambda i,v:cells[2*i+v].get_center()
        return cells

    def maximize(self):
        self.trellis()
        self.say('全体で最も重い1経路を探す',Indicate(self.trellis_cells[7][1],color=BLUE))
        rule=tex(r'M_{n+1}(x_{n+1})=\max_{x_n}M_n(x_n)\psi_{n,n+1}',26,YELLOW).move_to([0,-1.6,0])
        self.say('同じ因子表で、和を最大に替える',FadeIn(rule),Indicate(self.trellis_cells[2][1],color=YELLOW),
                 Indicate(self.trellis_cells[3][1],color=YELLOW))
        first=VGroup(J('4',23,GREEN).next_to(self.trellis_cells[2],DOWN,buff=.18),
                     J('4',23,GREEN).next_to(self.trellis_cells[3],DOWN,buff=.18))
        self.say('2番目への最大札：(4, 4)',FadeIn(first),color=GREEN)
        second=VGroup(J('16',23,GREEN).next_to(self.trellis_cells[4],DOWN,buff=.18),
                      J('12',23,GREEN).next_to(self.trellis_cells[5],DOWN,buff=.18))
        self.say('3番目への最大札：(16, 12)',FadeIn(second),color=GREEN)
        self.remove(rule)
        last=bar(48,48,GREEN,1.1,-1.5,width=3)
        self.say('右端1への最大は48',FadeIn(last),Indicate(self.trellis_cells[7][1],color=GREEN))
        self.remove(last)
        logrule=tex(r'\log\max_x\prod\psi=\max_x\sum\log\psi',28,YELLOW).move_to([0,-1.5,0])
        self.say('対数なら積が和に変わる',FadeIn(logrule),Indicate(first,color=YELLOW),Indicate(second,color=YELLOW))
        self.say('数だけでなく、前の状態も記録',Indicate(self.trellis_cells[4][1],color=YELLOW))

    def backtrack(self):
        self.trellis()
        nodes=self.trellis_cells
        # x4=1 <- x3=0 <- x2=0 <- x1=1
        chosen=[1,0,0,1]
        paths=[]
        for i in range(3):
            a=self.trellis_pts(i,chosen[i]);b=self.trellis_pts(i+1,chosen[i+1])
            paths.append(Line(a+RIGHT*.25,b+LEFT*.25,color=YELLOW,stroke_width=3))
        self.say('右端1の直前は3番目の0',Create(paths[2]),
                 Indicate(nodes[7][1],color=BLUE),Indicate(nodes[4][1],color=YELLOW))
        self.say('さらに0、その前は1',Create(paths[1]),Create(paths[0]))
        weight=J('4 × 4 × 3 = 48',27,GREEN).move_to([0,-1.55,0])
        self.say('最良の道：1、0、0、1',FadeIn(weight),trace_paths(paths),color=GREEN)
        marginal=J('4経路の和＝144',22,GREEN).move_to([-3.85,-1.55,0])
        self.say('周辺は和、最良の道は1経路',FadeIn(marginal),trace_paths([paths[1]],YELLOW))
        self.say('各点の最大を混ぜない',Indicate(nodes[2][1],color=YELLOW),Indicate(nodes[4][1],color=YELLOW))
        self.say('8.3の局所更新とは違う',trace_paths(paths))
        self.say('単独は1、最良の道では0',Indicate(nodes[3][1],color=GREEN),Indicate(nodes[2][1],color=YELLOW))

    def return_to_chain(self):
        self.graph(y=2.00)
        self.mark_observed()
        rows,order=self.candidate_rows(y_top=1.23,gap=.39)
        self.say('同じ4個、同じ観測、同じ因子表',FadeIn(rows),color=BLUE)
        first_values=[row[1][1] for row in rows[:4]]
        total1=J('1の4経路：36×4＝144',23,GREEN).move_to([4.35,.55,0])
        self.say('2番目＝1は4本で144',
                 AnimationGroup(*[Indicate(value,color=GREEN) for value in first_values],lag_ratio=.18),
                 FadeIn(total1))
        total0=J('0の4経路：合計75',23,PURPLE).move_to([4.35,-.7,0])
        best_index=next(i for i,(state,w) in enumerate(order) if state==(1,0,0,1))
        self.say('2番目＝0は75、その中の1本が48',FadeIn(total0),
                 Indicate(rows[best_index][1][1],color=YELLOW))
        answer=J('単独：1  ／  全体：1、0、0、1',23,GREEN).move_to([3.85,-1.55,0])
        self.say('単独なら1、全部なら1、0、0、1',FadeIn(answer),
                 self.set_state(0,1),self.set_state(1,0),self.set_state(2,0),color=GREEN)
        # Indicate can promote children of a VGroup to scene-level mobjects.
        # Rebuild the same chain before introducing the loop so no candidate bar
        # remains beneath the new edge.
        self.clear()
        self.add(J(SCENES[8]['title'],34).move_to([0,3.35,0]))
        self.graph(y=2.00)
        for i,value in enumerate((1,0,0)):
            self.state_labels[i].become(J(value,24,GREEN).move_to(self.points[i]).set_z_index(5))
        self.mark_observed()
        self.add(answer)
        loop=CubicBezier(self.points[0]+DOWN*.33,self.points[0]+DOWN*1.25,
                         self.points[3]+DOWN*1.25,self.points[3]+DOWN*.33,
                         color=RED,stroke_width=2)
        self.say('輪ができると、単純な伝播は保証されない',Create(loop),color=RED)
        clique=J('大きなクリークでは計算も大きい',22,PURPLE).move_to([0,-.65,0])
        self.say('一般のグラフはクリークをまとめる',trace_paths([loop],YELLOW),FadeIn(clique))
        factors=VGroup(*[Square(side_length=.16,stroke_color=YELLOW,fill_color=YELLOW,
                                fill_opacity=1).move_to([(self.points[i][0]+self.points[i+1][0])/2,2,0])
                         for i in range(3)])
        self.say('局所の表を渡して答えへ',FadeIn(factors),Indicate(answer,color=GREEN))
