"""PRML 8.2: a fuel gauge, three arrow shapes, and d-separation."""
import json
from pathlib import Path
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from gauge_model import P_G0,P_F0_G0,P_F0_G0_B0,JOINT_G0

BG='#10141F'
BLUE=ManimColor('#58C4DD'); YELLOW=ManimColor('#FFFF00')
GREEN=ManimColor('#83C167'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
_fade_in=FadeIn
def FadeIn(mobject,**kwargs):
    kwargs.setdefault('rate_func',lambda t:smooth(min(1.,10*t)))
    return _fade_in(mobject,**kwargs)
_fade_out=FadeOut
def FadeOut(mobject,**kwargs):
    kwargs.setdefault('rate_func',lambda t:smooth(min(1.,10*t)))
    return _fade_out(mobject,**kwargs)
_replacement_transform=ReplacementTransform
def ReplacementTransform(source,target,**kwargs):
    kwargs.setdefault('rate_func',lambda t:smooth(min(1.,4*t)))
    return _replacement_transform(source,target,**kwargs)

def J(s,x=0,y=0,size=24,color=WHITE):
    return jp(s,size,color).move_to((x,y,0))
def M(s,x=0,y=0,size=28,color=WHITE):
    return tex(s,size,color).move_to((x,y,0))
def dot(symbol,name,x,y,color,observed=False):
    circle=Circle(radius=.36,stroke_color=color,stroke_width=2.6,
                  fill_color=color if observed else BG,fill_opacity=1 if observed else 0).move_to((x,y,0))
    letter=M(symbol,x,y,26,BG if observed else color)
    label=J(name,x,y+.64 if y>0 else y-.65,20,color)
    return VGroup(circle,letter,label)
def edge(a,b,color=MUTED):
    return Arrow(a[0].get_center(),b[0].get_center(),buff=.42,color=color,
                 stroke_width=2.5,max_tip_length_to_length_ratio=.30)
def seen(node,color=YELLOW):
    return AnimationGroup(node[0].animate.set_fill(color,opacity=1),node[1].animate.set_color(BG),
                          rate_func=lambda t:smooth(min(1.,4*t)))
def flash(m,color=YELLOW):
    return Indicate(m,color=color,scale_factor=1.04)
def bar(label,value,y,color,x=.40,width=4.55,scale=1.):
    base=Line((x,y,0),(x+width,y,0),color=MUTED,stroke_width=1.2)
    amount=Line((x,y,0),(x+width*float(value)/scale,y,0),color=color,stroke_width=12)
    tag=J(label,x+.02,y+.30,21,color).align_to(base,LEFT)
    num=M(f'{float(value):.3f}'.rstrip('0').rstrip('.'),x+width+.47,y,23,color)
    return VGroup(base,amount,tag,num)
def gauge():
    b=dot('B','電池',-5.15,1.17,BLUE)
    f=dot('F','燃料',-2.25,1.17,PURPLE)
    g=dot('G','燃料計',-3.70,-1.02,YELLOW)
    bg=edge(b,g);fg=edge(f,g)
    return VGroup(bg,fg,b,f,g),dict(b=b,f=f,g=g,bg=bg,fg=fg)
def fork():
    r=dot('R','雨',-3.70,1.36,BLUE)
    d=dot('D','道',-5.15,-.95,YELLOW)
    u=dot('U','傘',-2.25,-.95,GREEN)
    rd=edge(r,d);ru=edge(r,u)
    return VGroup(rd,ru,r,d,u),dict(r=r,d=d,u=u,rd=rd,ru=ru)
def chain():
    a=dot('a','左',-5.2,.15,BLUE)
    c=dot('c','真ん中',-3.7,.15,YELLOW)
    b=dot('b','右',-2.2,.15,GREEN)
    ac=edge(a,c);cb=edge(c,b)
    return VGroup(ac,cb,a,c,b),dict(a=a,c=c,b=b,ac=ac,cb=cb)

class PRML82ConditionalIndependence(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        methods=[self.opening,self.definition,self.common_parent,self.middle_chain,
                 self.collider,self.numbers,self.long_paths,self.applications,self.return_to_gauge]
        for i,method in enumerate(methods):
            self.begin(i)
            self.side_note=None;self.equation=None
            method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml82_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def side(self,s,color=WHITE,y=2.30):
        if self.side_note is not None:self.remove(self.side_note)
        self.side_note=J(s,2.55,y,23,color)
        return FadeIn(self.side_note)
    def eq(self,s,color=WHITE,size=31):
        m=M(s,2.65,-2.25,size,color).set_stroke(color=color,width=.8,opacity=1)
        if self.equation is None:anim=FadeIn(m)
        else:anim=ReplacementTransform(self.equation,m)
        self.equation=m
        return anim

    def opening(self):
        g,n=gauge(); prior=bar('燃料切れ・事前',.1,.95,PURPLE)
        self.beat(FadeIn(g),FadeIn(prior),self.side('8.1 の矢印から先へ',BLUE))
        self.beat(flash(n['g']),self.side('燃料計は「空」',YELLOW))
        self.beat(seen(n['g']),flash(prior,PURPLE),self.side('燃料切れは増える？',YELLOW))
        after=bar('空表示の下',float(P_F0_G0),.10,YELLOW)
        self.beat(FadeIn(after),self.eq(r'p(F=0\mid G=0)\simeq0.257',YELLOW),self.side('10% → 25.7%',YELLOW))
        final=bar('電池切れも判明',float(P_F0_G0_B0),-.75,GREEN)
        self.beat(seen(n['b'],BLUE),FadeIn(final),self.eq(r'p(F=0\mid G=0,B=0)\simeq0.111',WHITE,30),self.side('予想と逆の 11.1%',GREEN),start_sentence=1)
        self.beat(flash(after,YELLOW),flash(final,GREEN),self.side('25.7% → 11.1%',GREEN))
        self.beat(flash(n['bg'],BLUE),flash(n['fg'],PURPLE),self.side('道はいつ開き、いつ閉じる？',YELLOW))

    def definition(self):
        # Semantic 2-by-2 probability bars; c is fixed within each row.
        frame=Rectangle(width=10.8,height=4.55,stroke_color=YELLOW,stroke_width=1.5).move_to((0,.02,0))
        review_label=J('復習: 1.2 条件付き確率',-3.25,1.82,24,WHITE)
        red_box=Rectangle(width=2.4,height=2.0,stroke_color=RED,stroke_width=2).move_to((-3.10,-.25,0))
        blue_box=Rectangle(width=5.6,height=2.0,stroke_color=BLUE,stroke_width=2).move_to((1.20,-.25,0))
        orange=ManimColor('#FF862F')
        oranges=VGroup(Rectangle(width=2.38,height=1.49,fill_color=orange,fill_opacity=.72,stroke_width=0).move_to((-3.10,0,0)),
                       Rectangle(width=5.58,height=.39,fill_color=orange,fill_opacity=.72,stroke_width=0).move_to((1.20,.55,0)))
        apples=VGroup(Rectangle(width=2.38,height=.49,fill_color=GREEN,fill_opacity=.52,stroke_width=0).move_to((-3.10,-1.00,0)),
                      Rectangle(width=5.58,height=1.59,fill_color=GREEN,fill_opacity=.52,stroke_width=0).move_to((1.20,-.45,0)))
        box_labels=VGroup(J('赤い箱 0.3',-3.10,-1.65,21,RED),J('青い箱 0.7',1.20,-1.65,21,BLUE))
        red_box.set_z_index(2);blue_box.set_z_index(2);box_labels.set_z_index(3)
        review=VGroup(frame,review_label,red_box,blue_box,oranges,box_labels)
        self.beat(FadeIn(review),FadeIn(apples))
        self.beat(FadeOut(apples),flash(oranges,orange))
        labels=VGroup(M('b=0',1.4,1.85,23,BLUE),M('b=1',4.35,1.85,23,PURPLE),
                      M('c=0',-.8,.75,23,YELLOW),M('c=1',-.8,-1.15,23,GREEN))
        vals=[(.3,1.4,.60,BLUE),(.3,4.35,.60,PURPLE),(.8,1.4,-1.30,BLUE),(.8,4.35,-1.30,PURPLE)]
        cells=VGroup(*[Line((x-1,y,0),(x-1+2*v,y,0),color=color,stroke_width=15) for v,x,y,color in vals])
        bases=VGroup(*[Line((x-1,y,0),(x+1,y,0),color=MUTED,stroke_width=1) for v,x,y,color in vals])
        header=J('説明用の例：a=1 の割合',2.65,2.42,22,YELLOW)
        row=SurroundingRectangle(VGroup(cells[0],cells[1]),color=YELLOW,buff=.28,stroke_width=2)
        self.beat(FadeOut(review),FadeIn(labels),FadeIn(bases),FadeIn(cells),FadeIn(header),Create(row),self.side('b を変えても同じ高さ',GREEN,y=-2.30))
        self.beat(FadeOut(header),self.eq(r'p(a\mid b,c)=p(a\mid c)',WHITE),self.side('追加情報が要らない',GREEN))
        product_m=MathTex(r'p(a,b\mid c)=',r'p(a\mid c)',r'p(b\mid c)',font_size=29).move_to((2.65,-2.25,0))
        product_m.set_stroke(color=WHITE,width=.8,opacity=1)
        product_m[1].set_color(GREEN);product_m[2].set_color(BLUE)
        product=ReplacementTransform(self.equation,product_m)
        self.equation=product_m
        self.beat(product,flash(cells,BLUE),self.side('条件付きの積に分かれる',BLUE))
        second=SurroundingRectangle(VGroup(cells[2],cells[3]),color=GREEN,buff=.28,stroke_width=2)
        self.beat(ReplacementTransform(row,second),flash(labels[3],GREEN),self.side('c=1 でも同じ関係',GREEN))
        self.beat(flash(second,GREEN),self.side('図だけで読める？',YELLOW))

    def common_parent(self):
        g,n=fork();b1=bar('雨の日・道がぬれる',.9,.95,YELLOW)
        b2=bar('雨の日・傘を持つ',.8,-.10,GREEN)
        self.beat(FadeIn(g),FadeIn(b1),FadeIn(b2),self.side('8.1 の雨・道・傘',BLUE))
        self.beat(flash(n['rd'],YELLOW),flash(n['ru'],GREEN),self.side('雨が2つを一緒に動かす',BLUE))
        same=bar('道が乾いても傘',.8,-1.15,GREEN)
        self.beat(seen(n['r'],BLUE),FadeIn(same),flash(b2,GREEN),self.side('雨を固定：傘 0.8',BLUE))
        sunny=VGroup(bar('晴れ・道がぬれる',.1,.95,YELLOW),bar('晴れ・傘',.2,-.10,GREEN),bar('晴れ・乾いても傘',.2,-1.15,GREEN))
        self.beat(FadeOut(b1),FadeOut(b2),FadeOut(same),FadeIn(sunny),self.side('晴れでも同じ',GREEN))
        self.beat(n['rd'].animate.set_opacity(.22),n['ru'].animate.set_opacity(.22),self.side('共通の親で道が閉じる',RED))
        self.beat(self.eq(r'D\perp U\mid R',WHITE),flash(n['d']),flash(n['u']),self.side('雨を知れば独立',GREEN))
        self.beat(flash(n['rd']),flash(n['ru']),self.side('次は矢印を鎖にする',YELLOW))

    def middle_chain(self):
        g,n=chain()
        self.beat(FadeIn(g),self.side('左 → 真ん中 → 右',BLUE))
        self.beat(Succession(flash(n['ac'],BLUE),flash(n['cb'],GREEN)),self.side('真ん中を通って届く',GREEN))
        left0=bar('左の状態 0',.62,.8,BLUE);left1=bar('左の状態 1',.62,-.35,PURPLE)
        self.beat(seen(n['c']),FadeIn(left0),FadeIn(left1),self.side('真ん中 c=1 を観測',YELLOW))
        factors=MathTex(r'p(a,c,b)=',r'p(a)',r'p(c\mid a)',r'p(b\mid c)',font_size=28).move_to((2.65,-2.25,0))
        factors.set_stroke(color=WHITE,width=.8,opacity=1)
        factors[1].set_color(BLUE);factors[2].set_color(YELLOW);factors[3].set_color(GREEN)
        self.equation=factors
        self.beat(FadeIn(factors),self.side('右は真ん中だけを見る',BLUE))
        self.beat(n['ac'].animate.set_opacity(.22),n['cb'].animate.set_opacity(.22),self.eq(r'a\perp b\mid c',WHITE),self.side('鎖も真ん中で閉じる',GREEN))
        self.beat(flash(n['ac']),flash(n['cb']),self.side('親と鎖は真ん中で閉じる',YELLOW))
        car,_=gauge()
        self.beat(FadeOut(g),FadeOut(left0),FadeOut(left1),FadeOut(self.equation),FadeIn(car),self.side('車では矢印が合流',YELLOW))

    def collider(self):
        g,n=gauge();prior=bar('燃料切れ',.1,.75,PURPLE)
        same=bar('電池切れでも燃料切れ',.1,-.35,BLUE)
        self.beat(FadeIn(g),FadeIn(prior),FadeIn(same),self.eq(r'B\perp F',PURPLE),self.side('電池と燃料は初め独立',PURPLE))
        self.beat(flash(n['bg'],BLUE),flash(n['fg'],PURPLE),self.side('2本の先端が合流',YELLOW))
        after=bar('空表示だけで',float(P_F0_G0),-1.4,YELLOW)
        self.beat(seen(n['g']),FadeIn(after),self.eq(r'p(F=0\mid G=0)\simeq0.257',YELLOW),self.side('空の原因が2つある',YELLOW))
        final=bar('電池切れも判明',float(P_F0_G0_B0),-.35,GREEN)
        self.beat(seen(n['b'],BLUE),FadeOut(same),FadeIn(final),self.eq(r'p(F=0\mid G=0,B=0)\simeq0.111',WHITE,30),self.side('電池が空表示を説明',BLUE))
        self.beat(self.eq(r'p(F\mid G,B)\ne p(F\mid G)',RED,30),flash(n['fg'],RED),self.side('合流点を見て道が開く',RED))
        self.beat(flash(after,YELLOW),flash(final,GREEN),self.side('説明し去る',GREEN))
        self.beat(flash(after,YELLOW),self.side('4状態で数字を追う',GREEN))

    def numbers(self):
        g,n=gauge();self.add(g)
        entries=[((1,1),BLUE),((1,0),PURPLE),((0,1),YELLOW),((0,0),GREEN)]
        ys=[1.55,.60,-.35,-1.30]
        rows=VGroup(*[bar(f'B={b}, F={f}',JOINT_G0[b,f],y,color,x=.50,width=4.2,scale=.2)
                      for ((b,f),color),y in zip(entries,ys)])
        self.beat(FadeIn(rows),self.side('4状態の重み・目盛 0.2',BLUE))
        self.beat(flash(rows,YELLOW),self.eq(r'p(G=0)=0.315',YELLOW),self.side('4つを足して 0.315',YELLOW))
        self.beat(flash(rows[1],PURPLE),flash(rows[3],GREEN),self.eq(r'p(F=0,G=0)=0.081',GREEN),self.side('燃料切れ側 0.081',GREEN))
        self.beat(self.eq(r'p(F=0\mid G=0)=0.081/0.315\simeq0.257',YELLOW,27),seen(n['g']),self.side('空表示で 25.7%',YELLOW))
        pair=VGroup(bar('F=1 の重み',.72,.45,YELLOW,x=.50,width=4.2),bar('F=0 の重み',.09,-.60,GREEN,x=.50,width=4.2))
        self.beat(FadeOut(rows),FadeIn(pair),FadeOut(self.equation),flash(n['b'],BLUE),self.side('B=0 固定・共通因子を省く',GREEN))
        self.equation=None
        self.beat(seen(n['b'],BLUE),self.eq(r'0.09/(0.72+0.09)\simeq0.111',WHITE,31),self.side('重みを割り算する',GREEN))
        self.beat(flash(pair[1],GREEN),self.side('10%より少し高い',GREEN))

    def long_paths(self):
        # Fig. 8.22's upper path, plus an explicit second route to test "all paths".
        a=dot('a','始点',-5.25,1.2,BLUE); e=dot('e','合流',-3.95,1.2,YELLOW)
        f=dot('f','分岐',-2.65,1.2,PURPLE); b=dot('b','終点',-1.35,1.2,GREEN)
        c=dot('c','子孫',-3.95,-.20,YELLOW); c[2].move_to((-3.10,.15,0))
        h=dot('h','別の道',-4.60,-1.85,BLUE)
        links=VGroup(edge(a,e),edge(f,e),edge(f,b),edge(e,c),edge(a,h),edge(h,b))
        graph=VGroup(links,a,e,f,b,c,h)
        top=J('上の道：閉',2.65,.8,24,YELLOW);lower=J('下の道：開',2.65,-.25,24,BLUE)
        self.beat(FadeIn(graph),FadeIn(top),FadeIn(lower),self.side('合流点の先に子がいる',YELLOW))
        self.beat(flash(links[0],BLUE),flash(links[1],PURPLE),flash(links[4],BLUE),self.side('2本の道をたどる',BLUE))
        top_open=J('上の道：開',2.65,.8,24,GREEN)
        low_closed=J('下の道：閉',2.65,-.25,24,RED)
        self.beat(Succession(
            AnimationGroup(seen(h,BLUE),ReplacementTransform(lower,low_closed)),
            AnimationGroup(seen(c),ReplacementTransform(top,top_open),flash(e,YELLOW))),
            self.side('通常の点は閉じ・子孫は開く',GREEN))
        top_closed=J('上の道：閉',2.65,.8,24,RED)
        self.beat(seen(f,PURPLE),ReplacementTransform(top_open,top_closed),self.side('2本とも閉じる',RED),start_sentence=1)
        self.beat(self.eq(r'a\perp b\mid\{c,f,h\}',WHITE),flash(top_closed),flash(low_closed),self.side('全経路が閉じて d 分離',GREEN))
        self.beat(flash(graph,PURPLE),self.side('数値の偶然は別',PURPLE))
        self.beat(flash(graph,BLUE),self.side('図で計算を軽くできる',BLUE))

    def applications(self):
        mu=dot(r'\mu','平均',-3.65,1.55,BLUE)
        xs=[dot(f'x_{i}',f'観測{i}',x,-.65,GREEN) for i,x in [(1,-5.2),(2,-3.65),(3,-2.1)]]
        links=VGroup(*[edge(mu,x) for x in xs]); graph=VGroup(links,mu,*xs)
        self.beat(FadeIn(graph),self.side('共通の平均から観測へ',BLUE))
        self.beat(seen(mu,BLUE),self.eq(r'p(D\mid\mu)=\prod_n p(x_n\mid\mu)',WHITE,30),flash(links),self.side('観測ごとの積に分ける',GREEN))
        self.beat(mu[0].animate.set_fill(BG,opacity=0),mu[1].animate.set_color(BLUE),self.eq(r'p(D)\ne\prod_n p(x_n)',YELLOW,30),self.side('平均を隠すと結び付きが戻る',YELLOW))
        z=dot('z','クラス',-3.65,1.55,PURPLE)
        ng=VGroup(*[edge(z,x) for x in xs],z,*[x.copy() for x in xs])
        self.beat(FadeOut(graph),FadeIn(ng),self.eq(r'p(x\mid z)=\prod_i p(x_i\mid z)',WHITE,30),self.side('同じ形で特徴を分類',PURPLE))
        self.beat(seen(z,PURPLE),flash(VGroup(*ng[4:]),GREEN),self.side('独立の仮定には代価がある',PURPLE))
        # Parent, child, and co-parent make the local blanket.
        core=dot('x','注目',-3.75,.25,YELLOW)
        par=dot('p','親',-5.1,1.65,BLUE); co=dot('q','共同の親',-2.0,1.65,PURPLE)
        child=dot('y','子',-3.1,-1.15,GREEN)
        blanket=VGroup(edge(par,core),edge(core,child),edge(co,child),par,co,core,child)
        blanket_label=J('親・子・共同の親',2.65,-2.25,25,YELLOW)
        self.beat(FadeOut(ng),FadeOut(self.equation),FadeIn(blanket),FadeIn(blanket_label),self.side('子の共同の親も必要',YELLOW))
        self.beat(flash(blanket,GREEN),self.side('マルコフブランケット',GREEN))

    def return_to_gauge(self):
        g,n=gauge();prior=bar('事前・燃料切れ',.1,.9,PURPLE)
        self.beat(FadeIn(g),FadeIn(prior),self.side('最初と同じ車',BLUE))
        self.beat(flash(n['g']),self.side('計器を見る前は道が閉じる',PURPLE))
        wet=bar('空表示だけ',float(P_F0_G0),-.05,YELLOW)
        self.beat(seen(n['g']),FadeIn(wet),self.side('空で道が開く：25.7%',YELLOW))
        cell=bar('電池切れも判明',float(P_F0_G0_B0),-1.0,GREEN)
        self.beat(seen(n['b'],BLUE),FadeIn(cell),self.side('電池が空の理由を説明',BLUE))
        self.beat(self.eq(r'0.257\longrightarrow0.111',WHITE,32),flash(wet,YELLOW),flash(cell,GREEN),self.side('同じ値で逆転を確認',GREEN))
        self.beat(flash(n['bg'],BLUE),flash(n['fg'],PURPLE),self.side('合流点だけは観測で開く',YELLOW))
        self.beat(flash(g,GREEN),self.side('次は矢印のない図へ',GREEN))
