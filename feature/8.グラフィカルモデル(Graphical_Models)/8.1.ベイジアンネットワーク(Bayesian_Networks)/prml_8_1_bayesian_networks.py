"""PRML 8.1: a wet road and the arrows of a Bayesian network."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene,jp,tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from bayesian_model import RESULT,DAYS,GAUSSIAN

BG='#10141F'
BLUE=ManimColor('#58C4DD');YELLOW=ManimColor('#FFFF00')
GREEN=ManimColor('#83C167');PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255');MUTED=ManimColor('#A8B2C5')
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

def J(s,x=0,y=2.42,size=24,color=WHITE):return jp(s,size,color).move_to((x,y,0))
def M(s,x=0,y=2.42,size=29,color=WHITE):return tex(s,size,color).move_to((x,y,0))
def highlight(m,color=YELLOW):return Indicate(m,color=color,scale_factor=1.06)

def node(symbol,name,position,color,observed=False):
    c=Circle(radius=.4,stroke_color=color,stroke_width=2.5,fill_color=color if observed else BG,
             fill_opacity=1 if observed else 0).move_to(position)
    letter=M(symbol,position[0],position[1],27,BG if observed else color)
    tag=J(name,position[0],position[1]+.72 if symbol in ('R','w') else position[1]-.70,21,color)
    return VGroup(c,letter,tag)

def network(full=False,observed=False):
    r=node('R','雨',(-3.15,1.05,0),BLUE)
    d=node('D','道',(-4.45,-.95,0),YELLOW,observed)
    u=node('U','傘',(-1.85,-.95,0),GREEN)
    rd=Arrow(r[0].get_center(),d[0].get_center(),buff=.45,color=MUTED,stroke_width=2.1,max_tip_length_to_length_ratio=.13)
    ru=Arrow(r[0].get_center(),u[0].get_center(),buff=.45,color=MUTED,stroke_width=2.1,max_tip_length_to_length_ratio=.13)
    du=Arrow(d[0].get_center(),u[0].get_center(),buff=.45,color=RED,stroke_width=2.1,max_tip_length_to_length_ratio=.13)
    links=VGroup(rd,ru,du)
    group=VGroup(links,r,d,u)
    if not full:du.set_opacity(0)
    return group,dict(r=r,d=d,u=u,rd=rd,ru=ru,du=du)

def bar(label,value,y,color,x=.9,width=4.1):
    base=Line((x,y,0),(x+width,y,0),color=MUTED,stroke_width=1.2)
    amount=Line((x,y,0),(x+width*value,y,0),color=color,stroke_width=13)
    title=J(label,x+.2,y+.40,22,color).align_to(base,LEFT)
    number=M(f'{value:.3f}'.rstrip('0').rstrip('.'),x+width+.47,y,24,color)
    return VGroup(base,amount,title,number)

def note(scene,m):
    if scene.current_note is not None:scene.remove(scene.current_note)
    scene.current_note=m
    return FadeIn(m)

def formula(s,x=2.35,y=-2.2,size=27,color=WHITE):return M(s,x,y,size,color)

class PRML81BayesianNetworks(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        methods=[self.opening,self.chain_rule,self.missing_link,self.posterior,
                 self.sampling,self.regression,self.discrete,self.gaussian,self.return_to_question]
        for i,method in enumerate(methods):
            self.begin(i);self.current_note=None;method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml81_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def opening(self):
        g,n=network()
        self.beat(FadeIn(g),note(self,J('7章の予測から、8章の「つながり」へ',color=BLUE)))
        b=bar('雨の日の傘',.8,1.05,GREEN)
        self.beat(FadeIn(b),highlight(n['r']),note(self,J('雨なら傘は 0.8',color=GREEN)))
        guess=M(r'p(U=1\mid D=1)\;?\;0.8',2.7,-.4,31,YELLOW)
        self.beat(FadeIn(guess),highlight(n['d']),note(self,J('道がぬれた日も 0.8？',color=YELLOW)))
        dry=J('晴れでも散水でぬれる',2.8,-1.5,24,RED)
        self.beat(FadeIn(dry),highlight(n['d'],RED),note(self,J('「ぬれた道」と「雨」は違う',color=RED)))
        obs=Circle(radius=.48,color=YELLOW,stroke_width=2.5).move_to(n['d'][0])
        self.beat(FadeOut(dry),FadeIn(obs),n['d'][0].animate.set_fill(YELLOW,opacity=1),note(self,J('見えたのは道だけ',color=YELLOW)))
        self.beat(highlight(n['r'],BLUE),highlight(n['u'],GREEN),note(self,J('雨を見ずに、傘を求める',color=GREEN)))
        joint=formula(r'p(R,D,U)',2.8,-2.05,32,PURPLE)
        self.beat(FadeOut(guess),FadeIn(joint),highlight(g,PURPLE),note(self,J('3つの出来事を1つに',color=PURPLE)))

    def chain_rule(self):
        g,n=network(full=True)
        review=J('復習 1.2：積の規則',2.45,1.45,25,BLUE)
        review_formula=M(r'p(a,b)=p(a)p(b\mid a)',2.45,.55,29,BLUE)
        pair=VGroup(Circle(radius=.29,color=BLUE,stroke_width=2).move_to((1.3,-.7,0)),
                    M('a',1.3,-.7,24,BLUE),
                    Arrow((1.65,-.7,0),(3.55,-.7,0),buff=0,color=YELLOW,stroke_width=2),
                    Circle(radius=.29,color=GREEN,stroke_width=2).move_to((3.9,-.7,0)),
                    M('b',3.9,-.7,24,GREEN))
        self.beat(FadeIn(g),FadeIn(review),FadeIn(review_formula),FadeIn(pair),note(self,J('同時確率を2段に分ける',color=BLUE)))
        self.beat(FadeOut(review),FadeOut(review_formula),FadeOut(pair),highlight(n['r']),highlight(n['d']),highlight(n['u']),note(self,J('雨・道・傘の3節点',color=YELLOW)))
        f1=formula(r'p(R,D,U)=p(U\mid R,D)p(R,D)',2.0,-2.05,29)
        self.beat(FadeIn(f1),highlight(n['du'],RED),note(self,J('傘には雨と道が条件',color=RED)))
        f2=formula(r'p(R,D,U)=p(R)p(D\mid R)p(U\mid R,D)',2.0,-2.05,29)
        self.beat(ReplacementTransform(f1,f2),highlight(n['rd']),note(self,J('積の規則をもう1回',color=BLUE)))
        self.beat(highlight(n['rd'],BLUE),highlight(n['ru'],GREEN),highlight(n['du'],RED),note(self,J('条件から子へ矢印',color=GREEN)))
        order=J('順序を変えると別の分解',2.3,.9,24,PURPLE)
        self.beat(FadeIn(order),highlight(g,PURPLE),note(self,J('全結合はどの分布にも使える',color=PURPLE)))
        self.beat(highlight(n['du'],RED),note(self,J('この赤い矢印は必要？',color=RED)))

    def missing_link(self):
        g,n=network(full=True)
        self.add(g)
        self.beat(highlight(n['du'],RED),note(self,J('道から傘へ、直接の矢印',color=RED)))
        self.beat(FadeOut(n['du']),note(self,J('雨が同じなら道は追加情報？',color=YELLOW)))
        f=formula(r'p(U\mid R,D)=p(U\mid R)',2.6,-1.9,30,GREEN)
        self.beat(FadeIn(f),highlight(n['r'],BLUE),highlight(n['u'],GREEN),note(self,J('傘の親は雨だけ',color=GREEN)))
        f2=formula(r'p(R,D,U)=p(R)p(D\mid R)p(U\mid R)',2.2,-1.9,29)
        self.beat(ReplacementTransform(f,f2),highlight(n['rd']),highlight(n['ru']),note(self,J('親ごとの因子を掛ける',color=YELLOW)))
        general=formula(r'p(x)=\prod_k p(x_k\mid\mathrm{pa}_k)',2.5,-1.9,28,PURPLE)
        self.beat(ReplacementTransform(f2,general),highlight(n['d'],YELLOW),note(self,J('積は節点ごとの掛け算',color=PURPLE)))
        cycle=CurvedArrow(n['u'][0].get_center()+RIGHT*.15,n['r'][0].get_center()+RIGHT*.15,angle=-1.1,color=RED,stroke_width=2)
        self.beat(Create(cycle),note(self,J('矢印の輪は作らない',color=RED)))
        self.beat(FadeOut(cycle),n['d'][0].animate.set_fill(YELLOW,opacity=1),highlight(n['u'],GREEN),note(self,J('道だけ見て傘を計算',color=YELLOW)))

    def posterior(self):
        g,n=network(observed=True)
        self.add(g)
        prior1=bar('雨の経路',.3,.95,BLUE,x=2.15,width=2.6)
        prior2=bar('晴れの経路',.7,-.38,PURPLE,x=2.15,width=2.6)
        self.beat(FadeIn(prior1),FadeIn(prior2),highlight(n['d']),note(self,J('補足：見えない雨を2経路で足す',color=YELLOW)))
        b1=bar('雨かつ道',.27,.95,BLUE,x=2.15,width=2.6)
        b2=bar('晴れかつ道',.07,-.38,PURPLE,x=2.15,width=2.6)
        self.beat(ReplacementTransform(prior1,b1),ReplacementTransform(prior2,b2),note(self,J('0.27 と 0.07',color=YELLOW)))
        sumline=formula(r'p(D=1)=0.27+0.07=0.34',2.55,-2.1,28,YELLOW)
        self.beat(FadeIn(sumline),highlight(b1),highlight(b2),note(self,J('2本で道の全体は 0.34',color=YELLOW)))
        post=formula(r'p(R=1\mid D=1)=0.27/0.34=0.794',2.65,-2.1,25,BLUE)
        self.beat(ReplacementTransform(sumline,post),highlight(n['r'],BLUE),note(self,J('ぬれた道のうち雨は約 0.794',color=BLUE)))
        pb1=bar('雨側の傘',.216,.95,GREEN,x=2.15,width=2.6)
        pb2=bar('晴れ側の傘',.014,-.38,GREEN,x=2.15,width=2.6)
        self.beat(ReplacementTransform(b1,pb1),ReplacementTransform(b2,pb2),highlight(n['u'],GREEN),note(self,J('傘まで進むと 0.216 と 0.014',color=GREEN)))
        answer=formula(r'p(U=1\mid D=1)=0.230/0.340=0.676',2.55,-2.1,29,GREEN)
        self.beat(ReplacementTransform(post,answer),highlight(pb1),highlight(pb2),note(self,J('答えは約 0.676',color=GREEN)))
        self.beat(highlight(n['rd']),highlight(n['ru']),note(self,J('次は矢印を順に進む',color=YELLOW)))

    def sampling(self):
        g,n=network()
        self.add(g)
        self.beat(highlight(n['r'],BLUE),note(self,J('親から子へ、雨を先に抽選',color=BLUE)))
        self.beat(highlight(n['rd'],YELLOW),highlight(n['ru'],GREEN),note(self,J('雨を見て道と傘を抽選',color=YELLOW)))
        state=VGroup(J('雨',1.15,1.15,26,BLUE),J('ぬれた道',2.7,1.15,26,YELLOW),J('傘あり',4.9,1.15,26,GREEN))
        product=formula(r'0.3\times0.9\times0.8=0.216',2.7,-1.85,28,GREEN)
        self.beat(FadeIn(state),FadeIn(product),highlight(n['r']),note(self,J('1日の組にも確率がある',color=GREEN)))
        sample=J('固定乱数・1000日',2.7,.15,24,PURPLE)
        days=ValueTracker(0)
        wet_prefix=np.r_[0,np.cumsum(DAYS[:,1])]
        umbrella_prefix=np.r_[0,np.cumsum(np.logical_and(DAYS[:,1],DAYS[:,2]))]
        wet_count=Integer(0,font_size=26,color=YELLOW).move_to((4.0,-.65,0))
        umbrella_count=Integer(0,font_size=26,color=GREEN).move_to((4.0,-1.25,0))
        wet_count.add_updater(lambda m:m.set_value(int(wet_prefix[min(1000,int(days.get_value()))])))
        umbrella_count.add_updater(lambda m:m.set_value(int(umbrella_prefix[min(1000,int(days.get_value()))])))
        count=VGroup(J('ぬれた道',2.15,-.65,25,YELLOW),wet_count,J('日',4.55,-.65,24,YELLOW),
                     J('そのうち傘あり',2.15,-1.25,24,GREEN),umbrella_count,J('日',4.55,-1.25,24,GREEN))
        self.add(count)
        self.beat(FadeOut(state),FadeOut(product),FadeIn(sample),days.animate.set_value(1000),highlight(g),note(self,J('標本の個数と理論値を区別',color=YELLOW)))
        ancestor=J('祖先サンプリング',2.7,1.35,26,BLUE)
        self.beat(FadeOut(sample),FadeIn(ancestor),highlight(n['r']),highlight(n['d']),highlight(n['u']),note(self,J('親が先なら条件がそろう',color=BLUE)))
        caution=J('因果と読むには生成過程の仮定',2.65,-2.0,23,RED)
        self.beat(FadeIn(caution),highlight(n['rd'],RED),note(self,J('矢印だけで因果は証明できない',color=RED)))
        self.beat(FadeOut(caution),highlight(n['r'],BLUE),note(self,J('既存の回帰も図で読もう',color=YELLOW)))

    def regression(self):
        # The plate is the source's repetition notation, not a decorative backing.
        xax=Axes(x_range=[-1,1,1],y_range=[-1,1,1],x_length=4,y_length=2,
                 tips=False,axis_config=dict(color=MUTED,include_ticks=False)).shift(RIGHT*2.7+DOWN*.5)
        c1=xax.plot(lambda x:.28+.37*x-.18*x*x,x_range=[-1,1,.04],color=PURPLE)
        c2=xax.plot(lambda x:.10+.55*x-.04*x*x,x_range=[-1,1,.04],color=GREEN)
        recap=J('復習 3.3：重みの候補 → 曲線',2.5,1.65,23,BLUE)
        self.beat(FadeIn(xax),Create(c1),Create(c2),FadeIn(recap),note(self,J('重みを1点に固定しない',color=BLUE)))
        self.remove(xax,c1,c2,recap)
        w=node('w','潜在',(-2.15,1.35,0),PURPLE)
        t=node('t_n','観測',(-2.15,-1.1,0),GREEN)
        wt=Arrow(w[0].get_center(),t[0].get_center(),buff=.43,color=MUTED,stroke_width=2)
        self.beat(FadeIn(w),FadeIn(t),GrowArrow(wt),note(self,J('隠れた重みから観測へ',color=PURPLE)))
        plate=Rectangle(width=3.15,height=2.15,stroke_color=BLUE,stroke_width=1.6).move_to((-2.15,-1.35,0))
        count=J('N',-3.35,-2.15,22,BLUE)
        self.beat(Create(plate),FadeIn(count),highlight(t),note(self,J('意味のある枠で反復を表す',color=BLUE)))
        x=Dot((-3.1,-.7,0),radius=.085,color=YELLOW)
        xn=M(r'x_n',-3.55,-.45,25,YELLOW)
        xt=Arrow(x.get_center(),t[0].get_center(),buff=.44,color=YELLOW,stroke_width=1.8)
        self.beat(FadeIn(x),FadeIn(xn),GrowArrow(xt),t[0].animate.set_fill(GREEN,opacity=1),note(self,J('既知の入力、観測した出力',color=GREEN)))
        prior=formula(r'p(w\mid t)\propto p(w)\prod_n p(t_n\mid w)',2.4,-2.0,26,PURPLE)
        self.beat(FadeIn(prior),highlight(w),note(self,J('重みは観測後も分布',color=PURPLE)))
        tx=node('t_*','予測',(1.3,-1.05,0),GREEN)
        newx=Dot((.35,-.45,0),radius=.085,color=YELLOW)
        arrows=VGroup(Arrow(w[0].get_center(),tx[0].get_center(),buff=.44,color=MUTED,stroke_width=1.8),Arrow(newx.get_center(),tx[0].get_center(),buff=.43,color=YELLOW,stroke_width=1.8))
        self.beat(FadeOut(prior),FadeIn(tx),FadeIn(newx),Create(arrows),note(self,J('枠の外へ新しい入力と出力',color=GREEN)))
        no_prior=J('入力の確率は未定義',2.9,1.2,24,RED)
        self.beat(FadeIn(no_prior),highlight(newx,RED),note(self,J('入力を含む生成モデルではない',color=RED)))

    def discrete(self):
        xs=[-5.2,-3.9,-2.6,-1.3]
        nodes=VGroup(*[Circle(radius=.25,color=BLUE,stroke_width=2).move_to((x,.9,0)) for x in xs])
        names=VGroup(*[M(f'x_{i+1}',x,.9,22,BLUE) for i,x in enumerate(xs)])
        chart=VGroup()
        for i in range(16):
            box=Square(side_length=.37,stroke_width=1.1,stroke_color=MUTED,
                       fill_color=YELLOW if i<15 else BG,fill_opacity=.9 if i<15 else 0)
            box.move_to((2.0+(i%4)*.58,1.4-(i//4)*.58,0))
            chart.add(box)
        self.beat(FadeIn(nodes),FadeIn(names),FadeIn(chart),note(self,J('二値変数4個で16通り',color=YELLOW)))
        count=M(r'2^4-1=15',3.05,-1.55,32,YELLOW)
        self.beat(FadeIn(count),highlight(chart),note(self,J('和が1なので自由な値は15個',color=YELLOW)))
        all_links=VGroup()
        for i in range(4):
            for j in range(i+1,4):
                h=.14*(j-i)
                all_links.add(CurvedArrow((xs[i],1.22+h,0),(xs[j],1.22+h,0),angle=-.35*(j-i),color=PURPLE,stroke_width=1.3,tip_length=.12))
        all_links.set_z_index(-1)
        self.beat(FadeIn(all_links),highlight(count),note(self,J('全結合なら一般の分布',color=PURPLE)))
        chain=VGroup(*[Arrow((xs[i]+.27,.9,0),(xs[i+1]-.27,.9,0),buff=0,color=GREEN,stroke_width=2) for i in range(3)])
        chart7=VGroup(*[chart[i].copy().set_fill(GREEN,opacity=.9) for i in range(7)])
        chart7.arrange_in_grid(rows=2,cols=4,buff=.2).move_to(chart)
        seven=M(r'1+3\times2=7',3.05,-1.55,31,GREEN)
        self.beat(FadeOut(all_links),FadeIn(chain),ReplacementTransform(chart,chart7),ReplacementTransform(count,seven),note(self,J('鎖にすると7個',color=GREEN)))
        four=VGroup(*[chart7[i].copy().set_fill(BLUE,opacity=.9) for i in range(4)])
        four.arrange(RIGHT,buff=.22).move_to(chart7)
        four_text=M(r'4',3.05,-1.55,33,BLUE)
        self.beat(FadeOut(chain),ReplacementTransform(chart7,four),ReplacementTransform(seven,four_text),note(self,J('矢印なしなら4個',color=BLUE)))
        labels=VGroup(J('全結合 15',2.7,.2,23,YELLOW),J('鎖 7',2.7,-.45,23,GREEN),J('独立 4',2.7,-1.1,23,BLUE))
        self.beat(FadeOut(four),FadeOut(four_text),FadeIn(labels),highlight(nodes),note(self,J('15 → 7 → 4：仮定は強まる',color=YELLOW)))
        self.beat(highlight(nodes,BLUE),note(self,J('連続変数にも同じ図法？',color=BLUE)))

    def gaussian(self):
        names=VGroup(*[M(f'x_{i}',-5.0+1.6*(i-1),1.05,30,BLUE if i==1 else GREEN) for i in (1,2,3)])
        circles=VGroup(*[Circle(radius=.37,color=BLUE if i==0 else GREEN,stroke_width=2).move_to(names[i]) for i in range(3)])
        links=VGroup(*[Arrow(circles[i].get_center(),circles[i+1].get_center(),buff=.4,color=MUTED,stroke_width=2) for i in (0,1)])
        ax=Axes(x_range=[-3,3,1],y_range=[-3,3,1],x_length=5.0,y_length=3.6,tips=False,
                axis_config=dict(color=MUTED,include_ticks=False,stroke_width=1.2)).shift(RIGHT*2.55+DOWN*.55)
        axis_labels=VGroup(M(r'x_1',ax.c2p(3,-3)[0]+.27,ax.c2p(3,-3)[1]-.2,22,BLUE),
                           M(r'x_3',ax.c2p(-3,3)[0]-.22,ax.c2p(-3,3)[1]+.16,22,GREEN))
        bell_axis=Line((.3,-1.3,0),(5.1,-1.3,0),color=MUTED,stroke_width=1.2)
        bell=FunctionGraph(lambda z:-1.3+1.4*np.exp(-.5*((z-2.7)/.8)**2),x_range=[.3,5.1,.04],color=BLUE,stroke_width=2.6)
        bell_label=J('中心 0・幅 1',2.7,.65,23,BLUE)
        self.beat(FadeIn(names),FadeIn(circles),FadeIn(links),FadeIn(bell_axis),Create(bell),FadeIn(bell_label),note(self,J('復習 2.3：中心と幅',color=BLUE)))
        first=M(r'x_1\sim\mathcal N(0,1)',-3.45,-.65,28,BLUE)
        parent=ValueTracker(-1.)
        values=VGroup(*[
            always_redraw(lambda factor=factor,x=x,color=color:
                DecimalNumber(factor*parent.get_value(),num_decimal_places=2,font_size=27,color=color).move_to((x,-.05,0)))
            for factor,x,color in [(1.,-5.,BLUE),(.8,-3.4,YELLOW),(.48,-1.8,GREEN)]
        ])
        center_labels=VGroup(J('値',-5.,.30,18,BLUE),J('中心',-3.4,.30,18,YELLOW),J('中心',-1.8,.30,18,GREEN))
        self.beat(FadeIn(first),FadeIn(values),FadeIn(center_labels),highlight(circles[0],BLUE),note(self,J('最初は平均0、分散1',color=BLUE)))
        second=M(r'x_2=0.8x_1+\varepsilon_2',-3.45,-.65,27,YELLOW)
        self.beat(ReplacementTransform(first,second),parent.animate.set_value(1.),highlight(links[0],YELLOW),note(self,J('親が子の中心を動かす',color=YELLOW)))
        third=M(r'x_3=0.6x_2+\varepsilon_3',-3.45,-.65,27,GREEN)
        self.beat(ReplacementTransform(second,third),parent.animate.set_value(-.5),highlight(links[1],GREEN),note(self,J('次の子も同じ仕組み',color=GREEN)))
        points=VGroup(*[Dot(ax.c2p(float(x1),float(x3)),radius=.024,color=GREEN,fill_opacity=.48)
                       for x1,_,x3 in GAUSSIAN[:240]])
        self.beat(FadeOut(bell_axis),FadeOut(bell),FadeOut(bell_label),FadeIn(ax),FadeIn(axis_labels),
                  LaggedStart(*[FadeIn(dot) for dot in points],lag_ratio=.005),note(self,J('順に作ると点の雲になる',color=GREEN)))
        cov=M(r'\operatorname{cov}(x_1,x_3)=0.48',-3.35,-1.8,25,PURPLE)
        self.beat(FadeIn(cov),highlight(points,PURPLE),note(self,J('直接の矢印がなくても相関',color=PURPLE)))
        self.beat(highlight(links[0]),highlight(links[1]),note(self,J('途中の節点がつながりを伝える',color=YELLOW)))

    def return_to_question(self):
        g,n=network(observed=True)
        self.add(g)
        initial=bar('雨の日の傘',.8,1.05,GREEN)
        self.beat(FadeIn(initial),highlight(n['d']),note(self,J('冒頭と同じぬれた道',color=YELLOW)))
        prior=J('雨 0.3   道：雨 0.9 / 晴れ 0.1',2.5,.05,23,BLUE)
        self.beat(FadeIn(prior),highlight(n['rd']),note(self,J('確率表も冒頭と同じ',color=BLUE)))
        b1=bar('雨かつ道',.27,.75,BLUE,x=2.1,width=2.7)
        b2=bar('晴れかつ道',.07,-.62,PURPLE,x=2.1,width=2.7)
        self.beat(FadeOut(initial),FadeOut(prior),FadeIn(b1),FadeIn(b2),note(self,J('道がぬれる2経路',color=YELLOW)))
        a1=bar('雨と道と傘',.216,.75,GREEN,x=2.1,width=2.7)
        a2=bar('晴れと道と傘',.014,-.62,GREEN,x=2.1,width=2.7)
        self.beat(ReplacementTransform(b1,a1),ReplacementTransform(b2,a2),highlight(n['u'],GREEN),note(self,J('傘ありの2経路を足す',color=GREEN)))
        answer=formula(r'(0.216+0.014)/0.34=0.676',2.6,-2.05,29,GREEN)
        self.beat(FadeIn(answer),highlight(a1),highlight(a2),note(self,J('ぬれた道なら傘は約 0.676',color=GREEN)))
        condition=J('0.8 は雨を知っているとき',2.7,-1.35,23,PURPLE)
        self.beat(FadeIn(condition),highlight(n['r'],BLUE),note(self,J('0.8 と 0.676 は別の条件',color=PURPLE)))
        self.beat(highlight(n['rd']),highlight(n['ru']),note(self,J('次は図から独立性を読む',color=YELLOW)))
