"""PRML 1.5: probabilities, losses and actions as linked visual experiments."""
from pathlib import Path
import json
import numpy as np
from manim import *
from narrated_scene import NarratedScene
from visual_support import jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from decision_model import (joint, posterior, mistake_parts, optimal_boundary,
                            reject_bounds, reject_fraction, corrected_posterior,
                            combine_posteriors, gaussian, mixture, MEAN, MEDIAN,
                            MODE, VARIANCE, regression_risk)

BLUE_C1 = '#58B5ED'
ORANGE_C2 = '#FFB45B'
GREEN_ACTION = '#77D49A'
YELLOW_LOSS = '#FFE079'
PURPLE_HOLD = '#C29AFF'
MUTED = '#A8B2C5'
BG = '#10141F'
AID_INPUT = '#58C4DD'
AID_OPERATION = '#FFFF00'
AID_RESULT = '#83C167'
AID_PRIOR = '#9A72AC'


def curve(ax, f, low, high, color=BLUE_C1, n=201):
    x = np.linspace(low, high, n)
    y = np.asarray(f(x)) + np.zeros_like(x)
    origin = ax.c2p(0, 0)
    points = origin + x[:,None]*(ax.c2p(1,0)-origin) + y[:,None]*(ax.c2p(0,1)-origin)
    return VMobject().set_points_as_corners(points).set_stroke(color, 3)


def area(ax, f, low, high, color, opacity=.35):
    if high <= low:
        high = low+1e-5
    xs = np.linspace(low,high,91)
    ys = np.asarray(f(xs))+np.zeros_like(xs)
    return Polygon(ax.c2p(low,0), *[ax.c2p(x,y) for x,y in zip(xs,ys)], ax.c2p(high,0),
                   stroke_width=0, fill_color=color, fill_opacity=opacity)


def readout(label, getter, pos, color=WHITE, places=3, size=24):
    prefix = tex(label, size, color)
    number = DecimalNumber(getter(),num_decimal_places=places,font_size=size,color=color)
    group = VGroup(prefix,number).arrange(RIGHT,buff=.12).move_to(pos)
    anchor = number.get_left().copy()
    number.add_updater(lambda m: m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group


def probability_bar(getter, pos=(0,.45,0), width=9, height=.65):
    def bars():
        p=float(getter())
        a=Rectangle(width=max(.0001,width*p),height=height,stroke_width=0,fill_color=BLUE_C1,fill_opacity=1)
        b=Rectangle(width=max(.0001,width*(1-p)),height=height,stroke_width=0,fill_color=ORANGE_C2,fill_opacity=1)
        a.move_to(np.array(pos)+LEFT*width/2,aligned_edge=LEFT)
        b.move_to(np.array(pos)+RIGHT*width/2,aligned_edge=RIGHT)
        return VGroup(a,b)
    return always_redraw(bars)


class PRML15DecisionTheory(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.question,self.errors,self.losses,self.reject,
                                   self.models,self.priors,self.regression,self.other_losses,self.recap]):
            self.begin(i)
            method()
            if self.beat_index != len(self.story['beats']):
                raise RuntimeError(f"Unused beats in {self.story['id']}")
            self.timeline[-1]['end']=float(self.time)
        out=Path(config.media_dir)/'prml15_timeline.json'
        out.write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def formula_at(self, expression, size=31, pos=(0,-2.55,0), colors=None):
        if self.formula is not None:
            self.remove(self.formula)
        term_colors = {r'p(x,C_1)': BLUE_C1, r'p(x,C_2)': ORANGE_C2,
                       r'p(C_k\mid x)': BLUE_C1, r'L_{kj}': YELLOW_LOSS,
                       r'\mathrm{Var}[t\mid x]': PURPLE_HOLD}
        # TeX colors keep one complete SVG group. Nested substring isolation in
        # this CE version drops glyphs following an isolated term in fractions.
        template=TexTemplate()
        template.add_to_preamble(r'\usepackage{xcolor}')
        colored=expression
        for term,color in term_colors.items():
            colored=colored.replace(term, r'{\color[HTML]{'+color[1:]+'}'+term+'}')
        self.formula=MathTex(colored,font_size=size,tex_template=template).move_to(pos)
        if colors:
            for text,color in colors.items():
                self.formula.set_color_by_tex(text,color,substring=False)
        if self.formula.width>12.6:
            raise ValueError('Formula too wide: '+expression)
        self.add(self.formula)
        return self.formula

    def note(self, text, pos=(0,2.43,0), color=MUTED, size=23):
        return jp(text,size,color).move_to(pos)

    def axes(self,x=(-5,5,1),y=(0,1,.5),width=10,height=2.8,center=(0,.3,0),xlabel='x',ylabel=None):
        ax=Axes(x_range=x,y_range=y,x_length=width,y_length=height,tips=False,
                axis_config=dict(color=MUTED,stroke_width=1.3,include_ticks=False)).move_to(center)
        labels=VGroup()
        for v in np.arange(x[0],x[1]+1e-6,x[2]):
            labels.add(tex(f'{v:g}',17,MUTED).next_to(ax.c2p(v,y[0]),DOWN,buff=.12))
        for v in np.arange(y[0],y[1]+1e-6,y[2]):
            labels.add(tex(f'{v:g}',17,MUTED).next_to(ax.c2p(x[0],v),LEFT,buff=.12))
        labels.add(tex(xlabel,23).next_to(ax.c2p(x[1],y[0]),RIGHT,buff=.2))
        if ylabel:
            labels.add(tex(ylabel,24).next_to(ax.c2p(x[0],y[1]),UP,buff=.12))
        self.add(ax,labels)
        return ax,VGroup(ax,labels)

    def slider(self,tracker,low,high,pos=(0,-1.95,0),label='c',width=5,ticks=None,color=YELLOW_LOSS):
        rail=NumberLine(x_range=[low,high,1],length=width,include_ticks=False,color=MUTED).move_to(pos)
        group=VGroup(rail,tex(label,25,color).next_to(rail,LEFT,buff=.25))
        for value in ticks or [low,high]:
            group.add(tex(f'{value:g}',16,MUTED).next_to(rail.n2p(value),DOWN,buff=.15))
        knob=Dot(radius=.075,color=color)
        knob.add_updater(lambda m:m.move_to(rail.n2p(tracker.get_value())))
        group.add(knob)
        self.add(group)
        return group

    def legend(self,first=r'p(x,C_1)',second=r'p(x,C_2)',y=2.25):
        group=VGroup(tex(first,26,BLUE_C1),tex(second,26,ORANGE_C2)).arrange(RIGHT,buff=1).move_to([0,y,0])
        self.add(group)
        return group

    def boundary(self,ax,getter,height,color=GREEN_ACTION):
        return always_redraw(lambda:DashedLine(ax.c2p(getter(),0),ax.c2p(getter(),height),
                                              color=color,dash_length=.1,stroke_width=2.5))

    def wipe_body(self):
        # Retain the title and subtitle while replacing the visual experiment.
        self.remove(*[m for m in self.mobjects if m is not self.header and m is not self.subtitle])
        self.add(self.header)
        self.formula=None

    def review_card(self, title):
        frame = RoundedRectangle(width=10.4, height=4.45, corner_radius=.12,
                                 color=AID_OPERATION, stroke_width=1.2).move_to([0,.1,0])
        # Empty Pango space glyphs must not enlarge the visible label bounds.
        label = VGroup(*[g for g in jp(title, 23) if g.has_points()])
        label.move_to([-4.85,2.02,0], aligned_edge=LEFT)
        self.add(frame,label)
        return label

    def bayes_recap(self):
        """Reuse the existing Bayes beat, with 1.2 bayes()'s area operation."""
        body = [m for m in self.mobjects if m is not self.header and m is not self.subtitle]
        old_formula = self.formula
        self.wipe_body()
        label = self.review_card('復習: 1.2 ベイズ更新')
        red, blue, green = '#FF6B77', '#58B5ED', '#77D49A'
        legend = VGroup(jp('赤い箱由来',20,red),jp('青い箱由来',20,blue),
                        jp('観測：オレンジ',20)).arrange(RIGHT,buff=.5).move_to([0,1.4,0])
        self.add(legend)
        width, height, left, bottom = 7, 1.7, -3.5, -.9
        def block(w,h,x,y,color,opacity=.65):
            return Rectangle(width=w,height=h,stroke_color=color,stroke_width=1.5,
                             fill_color=color,fill_opacity=opacity).move_to([x+w/2,y+h/2,0])
        cells = VGroup()
        for x,w,p,color in [(left,width*.3,.75,red),(left+width*.3,width*.7,.2,blue)]:
            cells.add(block(w,height*p,x,bottom,color),
                      block(w,height*(1-p),x,bottom+height*p,green,.2))
        self.add(cells)
        joint_weights = np.array([.3*.75,.7*.2])
        strip, normalized = VGroup(), VGroup()
        for k,color in enumerate([red,blue]):
            strip.add(block(width*joint_weights[k],height,left+width*joint_weights[:k].sum(),bottom,color))
            normalized.add(block(width*joint_weights[k]/joint_weights.sum(),height,
                                 left+width*joint_weights[:k].sum()/joint_weights.sum(),bottom,color))
        mapping = VGroup(jp('箱の種類 → クラス',22),jp('果物の観測 → 画像',22)).arrange(RIGHT,buff=.7).move_to([0,-1.65,0])
        formula = tex(r'p(C_k\mid x)=\frac{p(x\mid C_k)p(C_k)}{p(x)}\qquad(1.77)',32).move_to([0,-2.58,0])
        a,b,c = [self.sentence_duration(i) for i in range(3)]
        # audio_query at speedScale=1.08: 合計 starts 2.0444 s into sentence 2.
        self.beat(phases=[
            ('R1.2 identify prior and posterior',a,lambda:Indicate(label,color=WHITE,scale_factor=1.02)),
            ('R1.2 select observed fruit',.8,lambda:AnimationGroup(FadeOut(cells[1]),FadeOut(cells[3]))),
            ('R1.2 collect equal-height areas',2.0444-.8,
             lambda:AnimationGroup(Transform(cells[0],strip[0]),Transform(cells[2],strip[1]))),
            ('R1.2 normalize by observed total',b-2.0444,
             lambda:AnimationGroup(Transform(cells[0],normalized[0]),Transform(cells[2],normalized[1]))),
            ('R1.2 map boxes and fruit to classes and image',c,
             lambda:AnimationGroup(FadeIn(mapping),FadeIn(formula))),
        ])
        self.wipe_body()
        self.add(*body)
        self.formula = old_formula

    def prior_factor_aid(self):
        """V06b: show proportional factors, remove one prior, then normalize."""
        body = [m for m in self.mobjects if m is not self.header and m is not self.subtitle]
        old_formula = self.formula
        self.wipe_body()
        self.review_card('補足：事前確率は一回分だけ残す')
        self.add(jp('前提：クラスを固定すると画像と検査は独立',20).move_to([0,1.48,0]))
        def token(name,formula,color):
            box = RoundedRectangle(width=2.65,height=.62,corner_radius=.08,
                                   stroke_color=color,stroke_width=1.4)
            content = VGroup(jp(name,17,color),tex(formula,23,color)).arrange(RIGHT,buff=.14)
            return VGroup(box,content)
        image_factor = token('画像の尤度',r'p(x_I\mid C_k)',AID_INPUT).move_to([-.15,.6,0])
        blood_factor = token('検査の尤度',r'p(x_B\mid C_k)',AID_INPUT).move_to([-.15,-.3,0])
        prior1 = token('事前',r'p(C_k)',AID_PRIOR).move_to([3,.6,0])
        prior2 = token('事前',r'p(C_k)',AID_PRIOR).move_to([3,-.3,0])
        prefixes = VGroup(tex(r'p(C_k\mid x_I)\propto',26).move_to([-3.2,.6,0]),
                          tex(r'p(C_k\mid x_B)\propto',26).move_to([-3.2,-.3,0]))
        crosses = VGroup(*[tex(r'\times',27,AID_OPERATION).move_to([1.43,y,0]) for y in [.6,-.3]])
        self.add(prefixes,image_factor,blood_factor,crosses)
        slash = Line(prior2.get_corner(DL),prior2.get_corner(UR),color=AID_OPERATION,stroke_width=3)
        result = MathTex(r'w_k=',r'p(x_I\mid C_k)',r'p(x_B\mid C_k)',r'p(C_k)',font_size=29)
        result[1:3].set_color(AID_INPUT);result[3].set_color(AID_PRIOR)
        result.move_to([0,-1.45,0])
        normalized = tex(r'p(C_k\mid x_I,x_B)=\frac{w_k}{\sum_j w_j}\quad\Longrightarrow\quad\sum_k p(C_k\mid x_I,x_B)=1',29,AID_RESULT).move_to([0,-2.57,0])
        a,b = [self.sentence_duration(i) for i in range(2)]
        # Accent-phrase starts: 事前確率 1.9359 s; 除いて 0.7750 s;
        # 全クラス 1.8607 s. The matching PCM cues supply the sentence boundaries.
        self.beat(phases=[
            ('V06b two posterior factors',1.9359,lambda:Indicate(prefixes,scale_factor=1.02)),
            ('V06b one prior in each posterior',a-1.9359,lambda:AnimationGroup(FadeIn(prior1),FadeIn(prior2))),
            ('V06b identify duplicate prior',.775,lambda:Indicate(prior2,color=AID_OPERATION,scale_factor=1.04)),
            ('V06b cancel one prior',1.8607-.775,lambda:AnimationGroup(Create(slash),prior2.animate.set_opacity(.15))),
            ('V06b collect remaining factors',.7,lambda:AnimationGroup(FadeIn(result[0]),
                TransformFromCopy(image_factor[1][1],result[1]),TransformFromCopy(blood_factor[1][1],result[2]),
                TransformFromCopy(prior1[1][1],result[3]))),
            ('V06b normalize over all classes',b-1.8607-.7,lambda:FadeIn(normalized)),
        ])
        self.wipe_body()
        self.add(*body)
        self.formula = old_formula

    def question(self):
        p=ValueTracker(.08)
        bar=probability_bar(p.get_value)
        self.add(bar,self.note('説明用の例：画像から行動へ',color=BLUE_C1))
        numbers=VGroup(readout(r'p(C_1\mid x)=',p.get_value,(-2,1.3,0),BLUE_C1),
                       readout(r'p(C_2\mid x)=',lambda:1-p.get_value(),(2,1.3,0),ORANGE_C2))
        self.add(numbers)
        self.beat(Circumscribe(bar,color=BLUE_C1,buff=.08))
        a=self.note('病気として対応',(-2,-.7,0),BLUE_C1,28)
        b=self.note('健康と判断',(2,-.7,0),ORANGE_C2,28)
        self.beat(FadeIn(a,shift=UP*.15),FadeIn(b,shift=UP*.15))
        eq=self.formula_at(r'x\quad\longrightarrow\quad p(C_k\mid x)\quad\longrightarrow\quad a_j')
        self.beat(Indicate(eq))
        self.input_value=readout('x=',lambda:.5*np.log((1-p.get_value())/p.get_value()),(0,-1.7,0),GREEN_ACTION,2)
        self.add(self.input_value)
        self.beat(p.animate.set_value(.85),start_sentence=1)
        self.bayes_recap()
        self.formula_at(r'p(C_k\mid x)=\frac{p(x\mid C_k)p(C_k)}{p(x)}\qquad(1.77)',34)
        self.remove(self.input_value)
        inference=self.note('推論：確率を求める',(-2,-1.6,0),BLUE_C1)
        decision=self.note('決定：行動を選ぶ',(2,-1.6,0),GREEN_ACTION)
        self.beat(FadeIn(inference),FadeIn(decision))

    def errors(self):
        ax,_=self.axes(y=(0,.23,.1),height=2.65,center=(0,.3,0))
        self.legend()
        c1=curve(ax,lambda x:joint(x,1),-5,5,BLUE_C1)
        c2=curve(ax,lambda x:joint(x,2),-5,5,ORANGE_C2)
        self.beat(Create(c1),Create(c2))
        boundary=ValueTracker(-1.3)
        line=self.boundary(ax,boundary.get_value,.23)
        regions=VGroup(self.note('左：病気と判断',(-2,1.92,0),BLUE_C1,19),self.note('右：健康と判断',(2,1.92,0),ORANGE_C2,19))
        self.beat(FadeIn(line),FadeIn(regions))
        shades=always_redraw(lambda:VGroup(area(ax,lambda x:joint(x,2),-5,boundary.get_value(),ORANGE_C2),
                                          area(ax,lambda x:joint(x,1),boundary.get_value(),5,BLUE_C1)))
        self.add(shades)
        values=VGroup(readout(r'P(\mathrm{mistake})=',lambda:sum(mistake_parts(boundary.get_value())),(0,-1.85,0),YELLOW_LOSS))
        self.add(values)
        self.beat(Indicate(shades.copy(),remover=True,scale_factor=1,color=YELLOW_LOSS))
        self.beat(boundary.animate.set_value(1.5))
        self.beat(boundary.animate.set_value(0),end_sentence=1)
        eq=self.formula_at(r'P(\mathrm{mistake})=\int_{R_1}p(x,C_2)\,dx+\int_{R_2}p(x,C_1)\,dx\quad(1.78)',29)
        self.beat(Indicate(eq))
        eq=self.formula_at(r'j^*(x)=\arg\max_k p(C_k\mid x)\qquad P(\mathrm{correct})=\sum_k\int_{R_k}p(x,C_k)\,dx',27)
        self.beat(Indicate(eq,color=GREEN_ACTION))

    def risk_bars(self,cost,p=.08,center=(2.5,.2,0)):
        # Dynamic linear vertical scale; both bars always share the same scale.
        def bars():
            r1,r2=1-p,cost.get_value()*p
            top=max(1.8,r1*1.2,r2*1.2)
            result=VGroup()
            for x,r,col in [(center[0]-1,r1,BLUE_C1),(center[0]+1,r2,ORANGE_C2)]:
                rect=Rectangle(width=.7,height=max(.002,2.1*r/top),stroke_width=0,fill_color=col,fill_opacity=.85)
                rect.move_to([x,-.9,0],aligned_edge=DOWN)
                result.add(rect)
            return result
        g=always_redraw(bars)
        self.add(g,readout(r'R_1=',lambda:1-p,(center[0]-1,1.7,0),BLUE_C1,2),
                 readout(r'R_2=',lambda:p*cost.get_value(),(center[0]+1,1.7,0),ORANGE_C2,2))
        self.add(self.note('病気と判断',(center[0]-1,-1.3,0),BLUE_C1,19),
                 self.note('健康と判断',(center[0]+1,-1.3,0),ORANGE_C2,19))
        choice=always_redraw(lambda:SurroundingRectangle(g[0 if 1-p<p*cost.get_value() else 1],color=GREEN_ACTION,buff=.1))
        self.add(choice)
        return g

    def losses(self):
        cost=ValueTracker(1)
        self.add(self.note('行：本当のクラス　／　列：判断',(-2.8,2.35,0),size=21))
        head=VGroup(self.note('病気',(-3.1,1.6,0),BLUE_C1),self.note('健康',(-1.4,1.6,0),ORANGE_C2),
                    self.note('病気',(-4.8,.7,0),BLUE_C1),self.note('健康',(-4.8,-.2,0),ORANGE_C2))
        cells=VGroup(tex('0',35).move_to([-3.1,.7,0]),tex('1',35).move_to([-3.1,-.2,0]),tex('0',35).move_to([-1.4,-.2,0]))
        number=DecimalNumber(1,num_decimal_places=0,font_size=35,color=YELLOW_LOSS).move_to([-1.4,.7,0])
        number.add_updater(lambda m:m.set_value(cost.get_value()).move_to([-1.4,.7,0]))
        self.add(head,cells,number)
        self.beat(Indicate(head))
        slider=self.slider(cost,1,20,pos=(-2.9,-1.3,0),width=3.2,label='L_{12}',ticks=[1,20])
        self.beat(Indicate(cells))
        bars=self.risk_bars(cost)
        eq=self.formula_at(r'R_1=0\cdot0.08+1\cdot0.92\qquad R_2=L_{12}\cdot0.08',29)
        self.beat(Indicate(eq))
        self.beat(cost.animate.set_value(20))
        self.remove(slider)
        self.add(self.note('見逃しの損失：原文例 1000',(-2.8,-1.3,0),YELLOW_LOSS,20))
        self.beat(cost.animate.set_value(1000),end_sentence=1)
        eq=self.formula_at(r'R_j(x)=\sum_k L_{kj}p(C_k\mid x)\quad j^*(x)=\arg\min_j R_j(x)\quad(1.81)',28)
        global_risk=tex(r'\mathbb{E}[L]=\sum_k\sum_j\int_{R_j}L_{kj}p(x,C_k)\,dx\quad(1.80)',26).move_to([0,-1.84,0])
        self.add(global_risk)
        self.beat(Indicate(eq))
        self.wipe_body()
        ax,_=self.axes(y=(0,.23,.1),height=2.65)
        self.legend()
        self.add(curve(ax,lambda x:joint(x,1),-5,5,BLUE_C1),curve(ax,lambda x:joint(x,2),-5,5,ORANGE_C2))
        cost.set_value(1)
        line=self.boundary(ax,lambda:optimal_boundary(cost.get_value()),.23)
        band=always_redraw(lambda:area(ax,lambda x:np.full_like(x,.23),-5,optimal_boundary(cost.get_value()),BLUE_C1,.1))
        self.add(line,band,readout('L_{12}=',cost.get_value,(0,-1.85,0),YELLOW_LOSS,1))
        self.formula_at(r'C_1\ \mathrm{if}\ p(C_1\mid x)>\frac{1}{1+L_{12}}',31)
        self.beat(cost.animate.set_value(20))

    def reject(self):
        ax,_=self.axes(y=(0,1,.5))
        self.legend(r'p(C_1\mid x)',r'p(C_2\mid x)')
        self.add(curve(ax,posterior,-5,5,BLUE_C1),curve(ax,lambda x:1-posterior(x),-5,5,ORANGE_C2))
        dot=Dot(ax.c2p(0,.5),color=YELLOW_LOSS)
        self.beat(Indicate(dot,scale_factor=2))
        theta=ValueTracker(.6)
        line=always_redraw(lambda:DashedLine(ax.c2p(-5,theta.get_value()),ax.c2p(5,theta.get_value()),color=PURPLE_HOLD))
        def band():
            a,b=reject_bounds(theta.get_value())
            return area(ax,lambda x:np.ones_like(x),max(-5,a),min(5,b),PURPLE_HOLD,.24)
        shade=always_redraw(band)
        self.add(shade,line,readout(r'\theta=',theta.get_value,(-2,-1.85,0),PURPLE_HOLD,2),
                 readout(r'P(\mathrm{reject})=',lambda:reject_fraction(theta.get_value()),(2,-1.85,0),PURPLE_HOLD,3))
        eq=self.formula_at(r'\max_k p(C_k\mid x)\leq\theta\quad\Longrightarrow\quad\mathrm{reject}',32)
        self.beat(Indicate(eq,color=PURPLE_HOLD))
        self.beat(theta.animate.set_value(.9))
        self.beat(theta.animate.set_value(.55))
        d=self.sentence_duration(0)
        self.beat(phases=[('theta below half',d,lambda:theta.animate.set_value(.4)),
                          ('theta one',self.sentence_duration(1),lambda:theta.animate.set_value(1))])
        eq=self.formula_at(r'R_{\mathrm{reject}}(x)=\sum_k L_{k,\mathrm{reject}}p(C_k\mid x)',32)
        self.beat(Indicate(eq,color=PURPLE_HOLD))

    def models(self):
        ax,axgroup=self.axes(y=(0,1,.5))
        legend=self.legend()
        factor=ValueTracker(0)
        # The vertical unit expands continuously from 0.25 to 1 as densities
        # become posterior probabilities, avoiding a nearly empty initial plot.
        scale=lambda:.25+.75*factor.get_value()
        for tick,value in zip(list(axgroup[1][11:14]),[0,.5,1]):
            number=DecimalNumber(value*scale(),num_decimal_places=2,font_size=17,color=MUTED)
            position=tick.get_center().copy()
            number.move_to(position)
            number.add_updater(lambda m,value=value,position=position:m.set_value(value*scale()).move_to(position))
            self.remove(tick)
            axgroup[1].remove(tick)
            self.add(number)
        def density(k):
            return lambda x: ((1-factor.get_value())*joint(x,k)+factor.get_value()*(posterior(x) if k==1 else 1-posterior(x)))/scale()
        curves=always_redraw(lambda:VGroup(curve(ax,density(1),-5,5,BLUE_C1),curve(ax,density(2),-5,5,ORANGE_C2)))
        self.add(curves)
        eq=self.formula_at(r'p(x\mid C_k),p(C_k)\ \longrightarrow\ p(x,C_k)',33)
        self.beat(Indicate(eq))
        self.remove(legend)
        self.legend(r'p(C_1\mid x)',r'p(C_2\mid x)')
        self.formula_at(r'p(C_k\mid x)=\frac{p(x,C_k)}{\sum_j p(x,C_j)}\qquad(1.82),(1.83)',32)
        self.beat(factor.animate.set_value(1),start_sentence=1)
        eq=self.formula_at(r'x\ \longrightarrow\ p(C_k\mid x)\ \longrightarrow\ a_j',34)
        self.beat(Indicate(eq,color=GREEN_ACTION))
        labels=VGroup(Line(ax.c2p(-5,.8),ax.c2p(0,.8),color=BLUE_C1,stroke_width=8),
                      Line(ax.c2p(0,.2),ax.c2p(5,.2),color=ORANGE_C2,stroke_width=8))
        curves.clear_updaters()
        self.formula_at(r'x\ \longrightarrow\ f(x)\in\{C_1,C_2\}',34)
        self.beat(Transform(curves,labels),start_sentence=1)
        self.remove(curves)
        self.add(curve(ax,posterior,-5,5,BLUE_C1),curve(ax,lambda x:1-posterior(x),-5,5,ORANGE_C2))
        probe=ValueTracker(.02)
        point=always_redraw(lambda:Dot(ax.c2p(probe.get_value(),1-posterior(probe.get_value())),color=GREEN_ACTION))
        self.add(point,readout(r'p(C_2\mid x)=',lambda:1-posterior(probe.get_value()),(0,-1.85,0),ORANGE_C2,2))
        self.formula_at(r'\mathrm{label}\ C_2\qquad 0.51\ \longrightarrow\ 0.99',31)
        self.beat(probe.animate.set_value(float(np.log(99)/2)))
        self.wipe_body()
        ax,_=self.axes(y=(0,.26,.1))
        self.add(curve(ax,lambda x:joint(x,1)+joint(x,2),-5,5,GREEN_ACTION))
        probe.set_value(1)
        self.add(self.note('入力そのものの密度',color=GREEN_ACTION),self.boundary(ax,probe.get_value,.26),
                 readout('p(x)=',lambda:joint(probe.get_value(),1)+joint(probe.get_value(),2),(-2,-1.85,0),GREEN_ACTION,4),
                 readout(r'p(C_2\mid x)=',lambda:1-posterior(probe.get_value()),(2,-1.85,0),ORANGE_C2,4))
        self.formula_at(r'p(x)=\sum_k p(x\mid C_k)p(C_k)',32)
        self.beat(probe.animate.set_value(4.3),start_sentence=1)

    def priors(self):
        prior=ValueTracker(.5)
        p=lambda:corrected_posterior(.8,.5,prior.get_value())[0]
        bar=probability_bar(p)
        self.add(bar,readout(r'p_{\rm new}(C_1\mid x)=',p,(0,1.35,0),BLUE_C1,4),
                 self.note('学習時は２クラスを同じ数だけ集めた',size=23))
        rail=self.slider(prior,.01,.5,pos=(0,-.9,0),label=r'p_{\rm new}(C_1)',ticks=[.01,.5])
        self.beat(Circumscribe(bar,color=YELLOW_LOSS,buff=.08))
        self.beat(prior.animate.set_value(.01))
        eq=self.formula_at(r'p_{\rm new}(C_k\mid x)\propto p_{\rm train}(C_k\mid x)\frac{p_{\rm new}(C_k)}{p_{\rm train}(C_k)}',31)
        self.add(self.note('前提：クラスごとの入力分布は同じ',(0,-1.7,0),MUTED,21))
        self.beat(Indicate(eq))
        self.wipe_body()
        image_box=VGroup(RoundedRectangle(width=3.4,height=1,color=BLUE_C1),jp('画像の情報',27,BLUE_C1)).move_to([-2.5,1.3,0])
        blood_box=VGroup(RoundedRectangle(width=3.4,height=1,color=ORANGE_C2),jp('血液検査の情報',27,ORANGE_C2)).move_to([2.5,1.3,0])
        self.beat(FadeIn(image_box),FadeIn(blood_box))
        eq=self.formula_at(r'p(x_I,x_B\mid C_k)=p(x_I\mid C_k)p(x_B\mid C_k)\quad(1.84)',30)
        condition=self.note('本当のクラスを固定したときの独立',(0,.2,0),YELLOW_LOSS)
        likelihood_note=self.note('尤度：各クラスでの測定結果の現れやすさ',(0,-.65,0),MUTED,22)
        self.add(condition,likelihood_note)
        self.beat(Indicate(eq))
        self.remove(likelihood_note)
        eq=self.formula_at(r'p(C_k\mid x_I,x_B)\propto\frac{p(C_k\mid x_I)p(C_k\mid x_B)}{p(C_k)}\quad(1.85)',31)
        self.beat(Indicate(eq,color=YELLOW_LOSS))
        self.prior_factor_aid()
        evidence=ValueTracker(0)
        result=lambda:combine_posteriors(.5,.2+.4*evidence.get_value(),.2)[0]
        bar=probability_bar(result,pos=(0,-.8,0),width=8,height=.55)
        self.add(bar,readout(r'p(C_1\mid x_I,x_B)=',result,(0,-1.55,0),BLUE_C1,3),
                 self.note('事前 0.20　画像のみ 0.50　血液のみ 0.60',(0,2.35,0),size=22))
        self.beat(evidence.animate.set_value(1),start_sentence=1)

    def regression(self):
        ax,ag=self.axes(x=(-3,3,1),y=(0,.8,.4),xlabel='t',height=2.25,center=(0,.75,0))
        for tick in ag[1][:7]:
            tick.shift(UP*.4)
        mu=ValueTracker(0)
        pred=ValueTracker(-1.7)
        density=always_redraw(lambda:curve(ax,lambda t:gaussian(t,mu.get_value(),.6),-3,3,BLUE_C1))
        self.add(density,self.note('入力を固定したときの分布',color=BLUE_C1))
        self.beat(Indicate(density.copy(),remover=True,scale_factor=1))
        line=self.boundary(ax,pred.get_value,.8,YELLOW_LOSS)
        self.add(line,readout('y=',pred.get_value,(-2,-1.65,0),YELLOW_LOSS,2),
                 readout(r'\mathbb{E}[(y-t)^2\mid x]=',lambda:(pred.get_value()-mu.get_value())**2+.36,(2,-1.65,0),YELLOW_LOSS,3))
        self.beat(pred.animate.set_value(1.7))
        # Under the same horizontal t-axis, a downward density of loss contributions.
        xs=np.arange(-2.9,3,.2)
        def loss_rects():
            result=VGroup()
            for t in xs:
                val=(pred.get_value()-t)**2*gaussian(t,mu.get_value(),.6)
                width=np.linalg.norm(ax.c2p(.19,0)-ax.c2p(0,0))
                rect=Rectangle(width=width,height=max(.001,.35*val),stroke_width=0,fill_color=YELLOW_LOSS,fill_opacity=.6)
                rect.move_to(ax.c2p(t,0)+DOWN*.02,aligned_edge=UP)
                result.add(rect)
            return result
        bars=always_redraw(loss_rects)
        self.add(bars)
        eq=self.formula_at(r'\mathbb{E}[L]=\iint (y(x)-t)^2p(x,t)\,dx\,dt\quad(1.86),(1.87)',28)
        self.beat(Indicate(eq))
        self.beat(pred.animate.set_value(0),end_sentence=2)
        eq=self.formula_at(r'y^*(x)=\int t\,p(t\mid x)\,dt=\mathbb{E}[t\mid x]\qquad(1.89)',32)
        self.beat(Indicate(eq,color=GREEN_ACTION))
        eq=self.formula_at(r'\mathbb{E}[(y-t)^2\mid x]=(y-\mathbb{E}[t\mid x])^2+\mathrm{Var}[t\mid x]',30)
        self.add(self.note('平均で予測しても、分散 0.36 は残る',(0,-2.03,0),GREEN_ACTION,21))
        self.beat(phases=[('move away from mean',self.sentence_duration(0),lambda:pred.animate.set_value(.65)),
                          ('return to mean',self.sentence_duration(1),lambda:pred.animate.set_value(0))])
        pred.set_value(0)
        self.wipe_body()
        ax,_=self.axes(x=(-2,2,1),y=(-1.5,1.5,1),xlabel='x',ylabel='t',height=2.7)
        x=ValueTracker(-1.5)
        mean=lambda u:.55*np.sin(1.2*u)
        self.add(curve(ax,mean,-2,2,GREEN_ACTION),self.note('条件付き平均をつないだ回帰曲線',color=GREEN_ACTION))
        slice_curve=always_redraw(lambda:VMobject().set_points_as_corners([
            ax.c2p(x.get_value()+.45*gaussian(t,mean(x.get_value()),.35),t) for t in np.linspace(-1.5,1.5,101)]).set_stroke(BLUE_C1,3))
        dot=always_redraw(lambda:Dot(ax.c2p(x.get_value(),mean(x.get_value())),color=YELLOW_LOSS))
        self.add(slice_curve,dot)
        self.formula_at(r'\mathbb{E}[L]=\int(y-\mathbb{E}[t\mid x])^2p(x)\,dx+\int\mathrm{Var}[t\mid x]p(x)\,dx',26)
        self.beat(x.animate.set_value(1.3))

    def other_losses(self):
        ax,ag=self.axes(x=(-3,3.5,1),y=(0,.65,.3),xlabel='t',height=2.6)
        pred=ValueTracker(MEAN)
        dens=curve(ax,mixture,-3,3.5,BLUE_C1)
        marker=self.boundary(ax,pred.get_value,.65,YELLOW_LOSS)
        self.add(dens,marker,self.note('同じ条件付き分布に、二つの山',color=BLUE_C1),readout('y=',pred.get_value,(0,-1.8,0),YELLOW_LOSS,3))
        self.beat(Indicate(marker.copy(),remover=True,scale_factor=1))
        self.wipe_body()
        ax,_=self.axes(x=(-2,2,1),y=(0,4,2),xlabel='y-t',height=2.8)
        q=ValueTracker(2)
        loss=always_redraw(lambda:curve(ax,lambda z:np.abs(z)**q.get_value(),-2,2,YELLOW_LOSS))
        self.add(loss,readout('q=',q.get_value,(0,2.35,0),YELLOW_LOSS,2))
        eq=self.formula_at(r'\mathbb{E}[L_q]=\iint |y(x)-t|^q p(x,t)\,dx\,dt\qquad(1.91)',29)
        self.beat(Indicate(eq))
        self.beat(q.animate.set_value(1))
        self.wipe_body()
        ax,_=self.axes(x=(-3,3.5,1),y=(0,.65,.3),xlabel='t',height=2.6)
        median_note=self.note('平均 → 中央値：面積を半分に分ける',color=GREEN_ACTION)
        self.add(curve(ax,mixture,-3,3.5,BLUE_C1),median_note)
        marker=self.boundary(ax,pred.get_value,.65,YELLOW_LOSS)
        shade=always_redraw(lambda:area(ax,mixture,-3,pred.get_value(),GREEN_ACTION,.3))
        self.add(marker,shade,readout('y=',pred.get_value,(0,-1.8,0),YELLOW_LOSS,3))
        self.formula_at(r'\int_{-\infty}^{y^*}p(t\mid x)\,dt=\frac12\qquad(q=1)',32)
        self.beat(pred.animate.set_value(MEDIAN),start_sentence=1)
        self.remove(median_note)
        self.add(self.note('最頻値：狭い許容幅に入る確率',color=PURPLE_HOLD))
        window=ValueTracker(.8)
        self.remove(shade)
        pred.set_value(MODE)
        windowshade=always_redraw(lambda:area(ax,mixture,MODE-window.get_value(),MODE+window.get_value(),PURPLE_HOLD,.4))
        self.add(windowshade)
        self.formula_at(r'\arg\max_y P(|t-y|<\varepsilon\mid x)\ \xrightarrow[\varepsilon\to0]{}\ \mathrm{mode}',31)
        self.beat(window.animate.set_value(.12),start_sentence=1)
        eq=self.formula_at(r'q=2:\ \mathrm{mean}\qquad q=1:\ \mathrm{median}\qquad\varepsilon\to0:\ \mathrm{mode}',29)
        self.beat(Indicate(eq,color=GREEN_ACTION))

    def recap(self):
        cost=ValueTracker(1)
        self.add(probability_bar(lambda:.08,pos=(0,1.1,0),width=9,height=.45),self.note('病気の確率は、ずっと 8%',color=BLUE_C1))
        # Static first risk and dynamic second risk, with a moving choice marker.
        self.add(tex(r'R_1=0.92',34,BLUE_C1).move_to([-2,-.2,0]),
                 readout(r'R_2=',lambda:.08*cost.get_value(),(2,-.2,0),ORANGE_C2,2,34))
        choice=always_redraw(lambda:SurroundingRectangle(jp('病気と判断' if .92<.08*cost.get_value() else '健康と判断',27),color=GREEN_ACTION).move_to([0,-1.25,0]))
        label=always_redraw(lambda:jp('病気と判断' if .92<.08*cost.get_value() else '健康と判断',27,GREEN_ACTION).move_to([0,-1.25,0]))
        self.add(choice,label)
        self.formula_at(r'j^*(x)=\arg\min_j\sum_k L_{kj}p(C_k\mid x)',34)
        self.beat(Indicate(label.copy(),remover=True,scale_factor=1))
        self.beat(cost.animate.set_value(20),end_sentence=1)
        self.beat(Indicate(self.formula,color=GREEN_ACTION))
        self.remove(choice,label)
        next_topic=self.note('次へ：不確かさを、情報量で測る',(0,-1.25,0),PURPLE_HOLD,27)
        self.beat(FadeIn(next_topic))
