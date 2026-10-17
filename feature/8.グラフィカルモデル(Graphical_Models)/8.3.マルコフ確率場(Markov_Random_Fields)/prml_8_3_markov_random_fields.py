"""PRML 8.3: one noisy pixel, local factors and an undirected image model."""
import itertools
import json
import math
from pathlib import Path

from manim import *
import numpy as np

from image_model import BETA, ETA, FLIPS, TARGET, icm, observed, original
from make_voicevox_narration import MANIFEST
from narration_content import SCENES
from scene_support import NarratedScene, jp, tex

BG = '#10141F'
BLUE = ManimColor('#58C4DD')
YELLOW = ManimColor('#FFFF00')
GREEN = ManimColor('#83C167')
PURPLE = ManimColor('#9A72AC')
RED = ManimColor('#FC6255')
MUTED = ManimColor('#A8B2C5')
BLACK_CELL = ManimColor('#26344F')
WHITE_CELL = ManimColor('#E7EFF7')

_fade_in = FadeIn
def FadeIn(mobject, **kwargs):
    kwargs.setdefault('rate_func', lambda t: smooth(min(1., 10*t)))
    return _fade_in(mobject, **kwargs)

_fade_out = FadeOut
def FadeOut(mobject, **kwargs):
    kwargs.setdefault('rate_func', lambda t: smooth(min(1., 10*t)))
    return _fade_out(mobject, **kwargs)

_replace = ReplacementTransform
def ReplacementTransform(source, target, **kwargs):
    kwargs.setdefault('rate_func', lambda t: smooth(min(1., 4*t)))
    return _replace(source, target, **kwargs)

_create = Create
def Create(mobject, **kwargs):
    kwargs.setdefault('rate_func', lambda t: smooth(min(1., 5*t)))
    return _create(mobject, **kwargs)


def J(value, x=0, y=0, size=24, color=WHITE):
    return jp(value, size, color).move_to((x, y, 0))


def M(value, x=0, y=0, size=28, color=WHITE):
    return tex(value, size, color).move_to((x, y, 0))


def flash(mob, color=YELLOW):
    # Preserve pixel values, node labels, and the semantic colors of factors.
    # Indicate's recoloring made the entire binary picture one flat color.
    return Wiggle(mob, scale_value=1.045, rotation_angle=0, n_wiggles=1)


def pixel_image(data, center=(-3.45, 0, 0), step=.345, border=True):
    cells = VGroup()
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            x = center[0] + (col-(data.shape[1]-1)/2)*step
            y = center[1] + ((data.shape[0]-1)/2-row)*step
            cell = Square(side_length=step*.96, stroke_color=BG if border else None,
                          stroke_width=.7 if border else 0,
                          fill_color=BLACK_CELL if data[row,col] == -1 else WHITE_CELL,
                          fill_opacity=1).move_to((x,y,0))
            cells.add(cell)
    return cells


def pixel_at(cells, row, col):
    return cells[row*12+col]


def target_ring(cells, color=YELLOW):
    return SurroundingRectangle(pixel_at(cells, *TARGET), color=color, buff=.04, stroke_width=3)


def vertex(label, xy, color=BLUE, filled=False):
    circle = Circle(radius=.32, stroke_color=color, stroke_width=2.5,
                    fill_color=color if filled else BG, fill_opacity=1).move_to((*xy,0))
    letter = M(label, xy[0], xy[1], 25, BG if filled else color)
    return VGroup(circle, letter)


def link(a, b, color=MUTED):
    a0 = a[0].get_center()
    b0 = b[0].get_center()
    direction = (b0-a0)/np.linalg.norm(b0-a0)
    return Line(a0+.34*direction,b0-.34*direction,color=color,stroke_width=3)


def energy_bar(name, value, y, color):
    start = 1.05
    scale = .67
    base = Line((start,y,0),(start+3.2,y,0),color=MUTED,stroke_width=1.2)
    amount = Line((start,y,0),(start+scale*(value+2.2),y,0),color=color,stroke_width=11)
    label = J(name,2.28,y+.38,22,color)
    number = M(f'{value:+.1f}',4.72,y,27,color)
    return VGroup(base,amount,label,number)


class PRML83MarkovRandomFields(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.manifest = {e['id']: e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline = []
        methods = [self.opening, self.separation, self.blanket, self.factors,
                   self.local_choice, self.restoration, self.partition,
                   self.moralization, self.return_to_pixel]
        for i, method in enumerate(methods):
            self.begin(i)
            self.side_note = None
            self.equation = None
            method()
            assert self.beat_index == len(SCENES[i]['beats'])
            self.timeline[-1]['end'] = float(self.time)
        Path(config.media_dir,'prml83_timeline.json').write_text(
            json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def side(self, value, color=WHITE, y=2.22, size=23):
        if self.side_note is not None:
            self.remove(self.side_note)
        self.side_note = J(value,3.24,y,size,color)
        return FadeIn(self.side_note)

    def eq(self, value, color=WHITE, size=29, y=-2.22):
        mob = M(value,3.15,y,max(size,30),color)
        if self.equation is None:
            action = FadeIn(mob)
        else:
            action = ReplacementTransform(self.equation,mob)
        self.equation = mob
        return action

    def opening(self):
        y = observed()
        pic = pixel_image(y)
        ring = target_ring(pic)
        self.beat(FadeIn(pic),self.side('12×12 の観測画像',BLUE))
        self.beat(Create(ring),self.side('白い穴を発見',YELLOW))
        self.beat(flash(ring,YELLOW),self.side('白のまま？ 黒に戻す？',YELLOW))
        neighbours = [(4,5),(6,5),(5,4),(5,6)]
        marks = VGroup(*[SurroundingRectangle(pixel_at(pic,*q),color=PURPLE,buff=.01,stroke_width=2)
                         for q in neighbours])
        self.beat(FadeIn(marks),self.side('隣の4画素は黒',PURPLE))
        counter = J('反転 14 / 144 画素',3.20,.78,25,RED)
        self.beat(FadeIn(counter),flash(ring,RED),self.side('自作の固定データ',BLUE))
        self.beat(flash(marks,PURPLE),self.side('近所だけで判断できる？',YELLOW))
        self.beat(flash(ring,BLUE),self.side('矢印のない図を読む',GREEN))

    def separation(self):
        a = vertex('A',(-5.0,0),BLUE)
        c1 = vertex('c_1',(-2.35,1.45),YELLOW)
        c2 = vertex('c_2',(-2.35,-1.45),YELLOW)
        b = vertex('B',(.25,0),GREEN)
        edges = VGroup(link(a,c1),link(c1,b),link(a,c2),link(c2,b))
        graph = VGroup(edges,a,c1,c2,b)
        self.beat(FadeIn(graph),self.side('8.2 の合流点から',BLUE))
        self.beat(flash(edges[:2],YELLOW),flash(edges[2:],PURPLE),self.side('2本の道がある',YELLOW))
        crossed1 = Cross(c1,stroke_color=RED,stroke_width=4)
        self.beat(FadeIn(crossed1),edges[:2].animate.set_opacity(.25),self.side('上の道だけ遮断',RED))
        crossed2 = Cross(c2,stroke_color=RED,stroke_width=4)
        self.beat(FadeIn(crossed2),edges[2:].animate.set_opacity(.25),self.side('すべての道を遮断',GREEN))
        self.beat(self.eq(r'A\perp B\mid C',GREEN),flash(VGroup(c1,c2),YELLOW),self.side('観測集合 C で分離',GREEN))
        self.beat(flash(graph,BLUE),self.side('矢印の合流はない',BLUE))
        self.beat(flash(VGroup(c1,c2),PURPLE),self.side('画素の近所を囲もう',PURPLE))

    def blanket(self):
        core = vertex('x_i',(-3.5,0),YELLOW)
        positions = [(-3.5,1.55),(-3.5,-1.55),(-5.08,0),(-1.92,0)]
        adjacent = [vertex(f'x_{j}',p,PURPLE) for j,p in enumerate(positions,1)]
        far = vertex('z',(.05,1.55),MUTED)
        observation = vertex('y_i',(-5.15,-1.50),BLUE,True)
        lines = VGroup(*[link(core,v) for v in adjacent],link(adjacent[3],far),
                       link(core,observation,BLUE))
        graph = VGroup(lines,core,*adjacent,far,observation)
        self.beat(FadeIn(graph),self.side('中心と4近傍',YELLOW))
        self.beat(flash(lines[:4],PURPLE),self.side('遠い点へは隣を通る',PURPLE))
        around = VGroup(*[SurroundingRectangle(v,color=GREEN,buff=.08,stroke_width=2)
                          for v in [*adjacent,observation]])
        self.beat(FadeIn(around),lines[4].animate.set_opacity(.22),self.side('隣4つと観測が境界',GREEN))
        self.beat(self.eq(r'p(x_i\mid x_{-i},y)=p(x_i\mid x_{N(i)},y_i)',WHITE,22),
                  flash(around,GREEN),self.side('遠くは追加情報にならない',GREEN))
        self.beat(flash(VGroup(core,far),BLUE),self.side('直接つながらない2点',BLUE))
        y=observed();pic=pixel_image(y)
        ring=target_ring(pic)
        marks=VGroup(*[SurroundingRectangle(pixel_at(pic,*q),color=PURPLE,buff=.01,stroke_width=2)
                       for q in [(4,5),(6,5),(5,4),(5,6)]])
        self.beat(FadeOut(graph),FadeOut(around),FadeIn(pic),Create(ring),FadeIn(marks),
                  self.side('白い穴も近所を見る',YELLOW))
        self.beat(flash(ring,YELLOW),self.side('局所の相性を数へ',PURPLE))

    def factors(self):
        v = [vertex(f'x_{i}',xy,c) for i,(xy,c) in enumerate([
            ((-4.7,1.25),BLUE),((-2.7,1.25),PURPLE),((-4.7,-.85),GREEN),((-2.7,-.85),YELLOW)],1)]
        edges = VGroup(*[link(v[i],v[j]) for i,j in [(0,1),(0,2),(1,2),(1,3),(2,3)]])
        graph = VGroup(edges,*v)
        self.beat(FadeIn(graph),self.side('線のある組を探す',BLUE))
        triangle = VGroup(edges[0],edges[1],edges[2])
        self.beat(flash(triangle,GREEN),self.side('3点の最大クリーク',GREEN))
        missing = J('左上と右下は線なし',3.18,-.18,22,RED)
        self.beat(flash(v[0],RED),flash(v[3],RED),FadeIn(missing),self.side('欠けた対角線',RED))
        weight = VGroup(J('同じ色',1.55,.95,22,BLUE),J('大きい重み',4.0,.95,22,GREEN),
                        J('違う色',1.55,-.1,22,PURPLE),J('小さい重み',4.0,-.1,22,YELLOW))
        self.beat(FadeOut(missing),FadeIn(weight),self.eq(r'p(x)=Z^{-1}\prod_C\psi_C(x_C)',WHITE,28),self.side('局所の重みを掛ける',BLUE))
        # R card: the area rectangles evoke 1.2's red/blue boxes.
        frame = Rectangle(width=10.0,height=4.3,stroke_color=YELLOW,stroke_width=1.5).move_to((0,0,0))
        label = J('復習: 1.2 条件付き確率',-3.2,1.68,22,WHITE)
        orange=ManimColor('#FF862F')
        red=Rectangle(width=2.30,height=1.36,stroke_color=RED,stroke_width=2).move_to((-2.5,.12,0))
        blue=Rectangle(width=4.50,height=1.36,stroke_color=BLUE,stroke_width=2).move_to((1.2,.12,0))
        red_orange=Rectangle(width=2.27,height=1.00,fill_color=orange,fill_opacity=.8,stroke_width=0).move_to((-2.5,.28,0))
        blue_orange=Rectangle(width=4.47,height=.27,fill_color=orange,fill_opacity=.8,stroke_width=0).move_to((1.2,.65,0))
        red_green=Rectangle(width=2.27,height=.33,fill_color=GREEN,fill_opacity=.62,stroke_width=0).move_to((-2.5,-.39,0))
        blue_green=Rectangle(width=4.47,height=1.06,fill_color=GREEN,fill_opacity=.62,stroke_width=0).move_to((1.2,-.02,0))
        review=VGroup(frame,label,red_green,blue_green,red_orange,blue_orange,red,blue,
                      J('赤い箱 0.3',-2.5,-.85,19,RED),J('青い箱 0.7',1.2,-.85,19,BLUE))
        norm=VGroup(Line((-2.0,-1.55,0),(1.7,-1.55,0),color=orange,stroke_width=14),
                    J('オレンジだけを全体 1 へ',0,-1.90,20,orange))
        self.beat(FadeOut(graph),FadeOut(weight),FadeOut(self.equation),
                  Succession(FadeIn(review),FadeIn(norm)),
                  self.side('合計で割って確率へ',YELLOW,y=2.62))
        self.equation=None
        self.beat(FadeOut(review),FadeOut(norm),FadeIn(graph),FadeIn(weight),
                  self.eq(r'\psi_C=\exp(-E_C)',GREEN),self.side('低いエネルギーほど重い',GREEN))
        self.beat(flash(weight,YELLOW),self.side('白い穴の2候補へ',YELLOW))

    def local_choice(self):
        pic = pixel_image(observed())
        ring = target_ring(pic)
        self.beat(FadeIn(pic),Create(ring),self.side('黒 = −1、白 = +1',BLUE))
        marks = VGroup(*[SurroundingRectangle(pixel_at(pic,*q),color=PURPLE,buff=.01,stroke_width=2)
                         for q in [(4,5),(6,5),(5,4),(5,6)]])
        self.beat(FadeIn(marks),self.side('観測と隣の相性',PURPLE))
        knob = VGroup(J('β = 1',1.85,.88,24,PURPLE),J('η = 2.1',4.25,.88,24,BLUE))
        self.beat(FadeIn(knob),self.side('隣より観測を重く',BLUE))
        self.beat(self.eq(r'E_i=-\beta x_i\sum_{j\in N(i)}x_j-\eta x_i y_i',WHITE,24),flash(ring,YELLOW),self.side('関係する項だけ残す',YELLOW))
        white = energy_bar('白の候補',1.9,.15,RED)
        self.beat(FadeIn(white),self.side('白なら +1.9',RED))
        black = energy_bar('黒の候補',-1.9,-1.18,GREEN)
        self.beat(FadeIn(black),self.side('黒なら −1.9',GREEN))
        restored = pixel_image(observed())
        pixel_at(restored,*TARGET).set_fill(BLACK_CELL,opacity=1)
        self.beat(ReplacementTransform(pic,restored),flash(black,GREEN),self.side('低い黒を選ぶ',GREEN))

    def restoration(self):
        y=observed();restored,history,changes=icm(y)
        pic=pixel_image(y)
        self.beat(FadeIn(pic),self.side('最初は観測画像',BLUE))
        scan=SurroundingRectangle(pixel_at(pic,0,0),color=YELLOW,buff=.02,stroke_width=2)
        self.beat(Succession(Create(scan),scan.animate.move_to(pixel_at(pic,5,5))),
                  self.side('左上から1画素ずつ',YELLOW))
        final_pic=pixel_image(history[1])
        self.beat(ReplacementTransform(pic,final_pic),FadeOut(scan),self.side('1巡で8画素を更新',GREEN))
        marker=target_ring(final_pic,GREEN)
        self.beat(Create(marker),self.side('2巡目は変化0',GREEN))
        bars=VGroup(J('観測 130 / 144',3.10,.95,23,RED),J('復元 138 / 144',3.10,-.1,23,GREEN))
        self.beat(FadeIn(bars),self.side('一致率 95.8%',GREEN))
        self.beat(flash(bars,YELLOW),self.side('自作画像の結果',BLUE))
        self.beat(flash(marker,PURPLE),self.side('局所解で止まる可能性',PURPLE))

    def partition(self):
        # Separate 2x2 Ising example; every displayed bar comes from its potential.
        states=list(itertools.product((-1,1),repeat=4))
        vals=[math.exp(.13*(s[0]*s[1]+s[0]*s[2]+s[1]*s[3]+s[2]*s[3])
                       +.10*sum(s)) for s in states]
        bars=VGroup(*[Line((-5.55+i*.35,-1.55,0),(-5.55+i*.35,-1.55+v,0),
                           color=BLUE if i!=10 else YELLOW,stroke_width=12)
                      for i,v in enumerate(vals)])
        base=Line((-5.85,-1.55,0),(.15,-1.55,0),color=MUTED,stroke_width=1)
        label=J('説明用の2×2',-3.0,1.65,21,BLUE)
        self.beat(FadeIn(bars),FadeIn(base),FadeIn(label),self.side('4画素なら16通り',BLUE))
        self.beat(flash(bars,YELLOW),self.eq(r'Z=\sum_x\prod_C\psi_C(x_C)',WHITE,29),self.side('全候補の重みを足す',YELLOW))
        count=J('12×12 → 2の144乗通り',3.15,.28,23,RED)
        self.beat(FadeIn(count),self.side('全列挙は難しい',RED))
        self.beat(flash(bars[10],YELLOW),self.eq(r'\frac{p(x_i=+1\mid\cdots)}{p(x_i=-1\mid\cdots)}',WHITE,27),self.side('2候補の比では Z が消える',GREEN))
        self.beat(self.eq(r'p(x,y)=Z^{-1}\exp\{-E(x,y)\}',WHITE,28),flash(bars,GREEN),self.side('積はエネルギーの和へ',GREEN))
        self.beat(flash(count,RED),self.side('学習では Z も変わる',RED))
        self.beat(flash(bars,BLUE),self.side('有向の因子との違い',BLUE))

    def moralization(self):
        a=vertex('a',(-5.0,1.15),BLUE)
        b=vertex('b',(-2.6,1.15),PURPLE)
        c=vertex('c',(-3.8,-1.05),GREEN)
        arrows=VGroup(Arrow(a[0].get_center(),c[0].get_center(),buff=.35,color=MUTED),
                      Arrow(b[0].get_center(),c[0].get_center(),buff=.35,color=MUTED))
        chain_edges=VGroup(Arrow(a[0].get_center(),b[0].get_center(),buff=.35,color=MUTED),
                           Arrow(b[0].get_center(),c[0].get_center(),buff=.35,color=MUTED))
        self.beat(FadeIn(chain_edges),FadeIn(a),FadeIn(b),FadeIn(c),
                  self.side('矢印の鎖なら隣へ',BLUE))
        self.beat(ReplacementTransform(chain_edges,arrows),self.side('2つの親が合流',YELLOW))
        self.beat(self.eq(r'p(a)p(b)p(c\mid a,b)',WHITE,28),flash(c,GREEN),self.side('因子は3変数を含む',GREEN))
        parent_link=link(a,b,YELLOW)
        self.beat(Create(parent_link),self.side('親同士に線を足す',YELLOW))
        plain=VGroup(link(a,c),link(b,c))
        self.beat(ReplacementTransform(arrows,plain),self.side('道徳化して三角形',GREEN))
        self.beat(flash(parent_link,RED),self.side('元の独立は図から読めない',RED))
        self.beat(flash(VGroup(plain,parent_link),PURPLE),self.side('画素の格子へ戻る',PURPLE))

    def return_to_pixel(self):
        y=observed();result,_,_=icm(y)
        pic=pixel_image(y);ring=target_ring(pic)
        self.beat(FadeIn(pic),Create(ring),self.side('同じ観測画像',BLUE))
        marks=VGroup(*[SurroundingRectangle(pixel_at(pic,*q),color=PURPLE,buff=.01,stroke_width=2)
                       for q in [(4,5),(6,5),(5,4),(5,6)]])
        self.beat(FadeIn(marks),self.side('隣4つと観測の白',PURPLE))
        white=energy_bar('白',1.9,.25,RED);black=energy_bar('黒',-1.9,-1.05,GREEN)
        self.beat(FadeIn(white),FadeIn(black),self.side('白 +1.9 / 黒 −1.9',GREEN))
        one=y.copy();one[TARGET]=-1
        chosen=pixel_image(one)
        self.beat(ReplacementTransform(pic,chosen),flash(black,GREEN),self.side('同じ画素を黒へ',GREEN))
        final=pixel_image(result)
        self.beat(ReplacementTransform(chosen,final),self.side('130 → 138 / 144 画素',GREEN))
        self.beat(flash(ring,PURPLE),self.side('近所が判断を支える',PURPLE))
        self.beat(flash(ring,BLUE),self.side('次はグラフ上の推論へ',BLUE))
