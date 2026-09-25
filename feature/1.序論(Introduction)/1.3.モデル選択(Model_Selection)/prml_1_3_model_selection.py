"""PRML 1.3: measured model selection experiments in Manim Community."""
from pathlib import Path
import json
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
import model_selection as model

BLUE_DATA = ManimColor('#58B5ED')
ORANGE_DATA = ManimColor('#FFB45B')
MODEL_RED = ManimColor('#FF6B77')
GREEN_TEST = ManimColor('#77D49A')
PURPLE_TERM = ManimColor('#C29AFF')
YELLOW_ERROR = ManimColor('#FFE079')
MUTED = ManimColor('#A8B2C5')
BG = '#10141F'


def polyline(points, color, width=3):
    return VMobject().set_points_as_corners(points).set_stroke(color, width)


def curve(ax, w, color=MODEL_RED):
    return polyline([ax.c2p(x,y) for x,y in zip(model.GRID, model.predict(w,model.GRID))], color)


def dots(ax, x, t, color=BLUE_DATA):
    return VGroup(*[Dot(ax.c2p(a,b), radius=.055, color=color) for a,b in zip(x,t)])


def residuals(ax, w, x, t, color=YELLOW_ERROR):
    return VGroup(*[Line(ax.c2p(a,b),ax.c2p(a,c),color=color,stroke_width=2)
                   for a,b,c in zip(x,t,model.predict(w,x))])


def number(label, getter, position, color=WHITE, places=3, size=24):
    prefix = tex(label,size,color)
    n = DecimalNumber(getter(),num_decimal_places=places,font_size=size,color=color)
    group = VGroup(prefix,n).arrange(RIGHT,buff=.12).move_to(position)
    anchor=n.get_left().copy()
    n.add_updater(lambda m: m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group


class PRML13ModelSelection(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.timeline=[]
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        for i, method in enumerate([self.question,self.validation,self.roles,self.cross_validation,
                                     self.leave_one_out,self.cost,self.aic,self.limits]):
            self.begin(i)
            if self.audio_entry is None:
                raise RuntimeError('Generate matching WhiteCUL narration before rendering')
            method()
            assert self.beat_index == len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        out=Path(config.media_dir)/'prml13_timeline.json'
        out.write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def legend(self, entries):
        g=VGroup(*[VGroup(Dot(radius=.045,color=c),jp(s,18,c)).arrange(RIGHT,buff=.10)
                   for s,c in entries]).arrange(RIGHT,buff=.55).move_to([0,2.75,0])
        self.add(g)
        return g

    def graph(self, center=(-.7,.3,0), width=8, height=3.45):
        ax=Axes(x_range=[0,1,.25],y_range=[-1.5,1.5,.5],x_length=width,y_length=height,
                tips=False,axis_config={'color':MUTED,'stroke_width':1.2})
        ax.move_to(center)
        labels=VGroup(tex('x',22).next_to(ax.c2p(1,0),RIGHT,buff=.12),
                      tex(r't,\ y',22).next_to(ax.c2p(0,1.5),UP,buff=.12))
        for x in [0,.5,1]: labels.add(tex(str(x),17,MUTED).next_to(ax.c2p(x,-1.5),DOWN,buff=.1))
        for y in [-1,1]: labels.add(tex(str(y),17,MUTED).next_to(ax.c2p(0,y),LEFT,buff=.1))
        self.add(ax,labels)
        return ax,VGroup(ax,labels)

    def slider(self, tracker, lo=1, hi=9, y=-2.55, label='d', color=MODEL_RED):
        line=NumberLine(x_range=[lo,hi,1],length=7.1,include_ticks=True,include_numbers=False,
                        color=MUTED,stroke_width=2).move_to([-.5,y,0])
        group=VGroup(line,tex(label,27,color).next_to(line,LEFT,buff=.25))
        for n in range(lo,hi+1): group.add(tex(str(n),17,MUTED).next_to(line.n2p(n),DOWN,buff=.13))
        knob=Dot(line.n2p(tracker.get_value()),color=color,radius=.085)
        knob.add_updater(lambda m:m.move_to(line.n2p(tracker.get_value())))
        group.add(knob)
        self.add(group)
        return group

    def cursor(self, ax, w_getter, x=.0):
        scan=ValueTracker(x)
        dot=always_redraw(lambda:Dot(ax.c2p(scan.get_value(),model.predict(w_getter(),[scan.get_value()])[0]),
                                   radius=.075,color=YELLOW_ERROR))
        self.add(dot)
        return scan,dot

    def question(self):
        self.legend([('訓練',BLUE_DATA),('予測',MODEL_RED),('ずれ',YELLOW_ERROR)])
        ax,_=self.graph()
        points=dots(ax,model.X,model.T)
        d=ValueTracker(1)
        w=lambda:model.interpolated_weights(d.get_value())
        line=always_redraw(lambda:curve(ax,w()))
        errs=always_redraw(lambda:residuals(ax,w(),model.X,model.T))
        self.slider(d)
        badge=jp('整数で候補を比較',18,MUTED).move_to([4.5,-2.55,0])
        self.add(badge)
        self.beat(LaggedStart(*[FadeIn(p) for p in points],lag_ratio=.08))
        self.add(line)
        scan,probe=self.cursor(ax,w)
        self.beat(scan.animate.set_value(1))
        self.remove(probe)
        self.beat(d.animate.set_value(3),start_sentence=1)
        formula=tex(r'y=w_0+w_1x+w_2x^2+w_3x^3',28,MODEL_RED).move_to([0,-1.97,0])
        self.beat(Write(formula),Create(errs))
        self.remove(formula)
        self.beat(d.animate.set_value(9))
        self.beat(d.animate.set_value(5))
        self.beat(d.animate.set_value(9))
        q=jp('未知の点でも、当たる？',28,ORANGE_DATA).move_to([0,-1.98,0])
        self.beat(Write(q))

    def validation(self):
        legend=self.legend([('訓練',BLUE_DATA),('検証',ORANGE_DATA),('予測',MODEL_RED)])
        ax,axes_group=self.graph()
        train=dots(ax,model.X,model.T)
        valid=dots(ax,model.XV,model.TV,ORANGE_DATA)
        d=ValueTracker(9); w=lambda:model.interpolated_weights(d.get_value())
        line=always_redraw(lambda:curve(ax,w()))
        errs=always_redraw(lambda:residuals(ax,w(),model.XV,model.TV,ORANGE_DATA))
        slider=self.slider(d)
        self.add(train,line)
        self.beat(LaggedStart(*[FadeIn(p) for p in valid],lag_ratio=.04))
        self.beat(Create(errs))
        self.beat(d.animate.set_value(3))
        formula=MathTex(r'E_{\rm val}', '=',r'\frac1{N_{\rm val}}\sum_n',r'(y(x_n)-t_n)^2',font_size=29)
        formula[0].set_color(ORANGE_DATA); formula[3].set_color(ORANGE_DATA)
        formula.move_to([0,-2.03,0])
        self.beat(Write(formula),Indicate(errs,color=ORANGE_DATA))
        # Same central space becomes a score plot. New points are computed, never hand drawn.
        self.clear()
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]),formula)
        self.legend([('訓練誤差',BLUE_DATA),('検証誤差',ORANGE_DATA)])
        ex=Axes(x_range=[1,9,1],y_range=[0,.22,.05],x_length=8.4,y_length=3.35,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.3}).move_to([0,.35,0])
        labels=VGroup(jp('平均二乗誤差',19,MUTED).move_to([-4,2.3,0]),tex('d',24).next_to(ex.c2p(9,0),RIGHT))
        for v in [1,3,5,7,9]: labels.add(tex(str(v),18,MUTED).next_to(ex.c2p(v,0),DOWN,buff=.12))
        for v in [0,.1,.2]: labels.add(tex(str(v),18,MUTED).next_to(ex.c2p(1,v),LEFT,buff=.12))
        self.add(ex,labels)
        tr=VGroup(*[Dot(ex.c2p(int(k),float(v)),color=BLUE_DATA,radius=.06) for k,v in zip(model.DEGREES,model.TRAIN)])
        va=VGroup(*[Dot(ex.c2p(int(k),float(v)),color=ORANGE_DATA,radius=.06) for k,v in zip(model.DEGREES,model.VALID)])
        trline=polyline([p.get_center() for p in tr],BLUE_DATA)
        valine=polyline([p.get_center() for p in va],ORANGE_DATA)
        self.beat(Create(trline),LaggedStart(*[FadeIn(p) for p in tr],lag_ratio=.2))
        self.beat(Create(valine),LaggedStart(*[FadeIn(p) for p in va],lag_ratio=.2))
        connectors=VGroup(*[Line(tr[i].get_center(),va[i].get_center(),color=PURPLE_TERM,stroke_width=3) for i in [6,7,8]])
        self.beat(LaggedStart(*[Create(l) for l in connectors],lag_ratio=.25))
        choice=SurroundingRectangle(va[model.SELECTED-1],color=YELLOW_ERROR,buff=.12)
        result=tex(r'd^*=3,\quad E_{\rm val}=%.4f'%model.VALID[2],27,ORANGE_DATA).move_to([0,-2.6,0])
        self.beat(Create(choice),Write(result))

    def roles(self):
        role_labels=['訓練：係数を学ぶ','検証：モデルを選ぶ','テスト：最後に測る']
        colors=[BLUE_DATA,ORANGE_DATA,GREEN_TEST]
        rows=VGroup()
        for i,(label,color) in enumerate(zip(role_labels,colors)):
            y=1.65-i*1.4
            blocks=VGroup(*[Square(side_length=.27,color=color,fill_color=color,fill_opacity=.65) for _ in range(10)])
            blocks.arrange(RIGHT,buff=.07).move_to([-2,y,0])
            txt=jp(label,26,color).move_to([2.45,y,0])
            rows.add(VGroup(blocks,txt))
        self.beat(FadeIn(rows[0],shift=RIGHT*.4))
        self.beat(FadeIn(rows[1],shift=RIGHT*.4))
        box=RoundedRectangle(width=3.6,height=.65,color=GREEN_TEST,fill_color=BG,fill_opacity=1).move_to(rows[2][0])
        locked=jp('選択が終わるまで未使用',20,GREEN_TEST).move_to(box)
        self.beat(FadeIn(rows[2]),FadeIn(box),Write(locked))
        loop=CurvedArrow([4,.7,0],[4,-.2,0],angle=-TAU*.8,color=ORANGE_DATA)
        self.beat(Create(loop),Indicate(rows[1],color=ORANGE_DATA))
        warning=jp('選ぶほど、検証にも合わせ込める',25,YELLOW_ERROR).move_to([0,-2.35,0])
        self.beat(Write(warning),Indicate(box,color=GREEN_TEST))
        self.clear()
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]))
        self.legend([('訓練',BLUE_DATA),('選択済みの三次式',MODEL_RED),('最終テスト',GREEN_TEST)])
        ax,_=self.graph()
        self.add(dots(ax,model.X,model.T),curve(ax,model.WEIGHTS[2]))
        td=dots(ax,model.XT,model.TT,GREEN_TEST)
        score=tex(r'E_{\rm test}=%.4f'%model.TEST_MSE,30,GREEN_TEST).move_to([0,-2.2,0])
        self.beat(LaggedStart(*[FadeIn(p) for p in td],lag_ratio=.02),Write(score),start_sentence=1)
        scan,probe=self.cursor(ax,lambda:model.WEIGHTS[2])
        self.beat(scan.animate.set_value(1))
        self.remove(probe)
        note=VGroup(tex(r'd,\ \lambda',34,PURPLE_TERM),jp('学習方法の設定',22,PURPLE_TERM)).arrange(RIGHT,buff=.4).move_to([0,-2.7,0])
        self.beat(Write(note))

    def fold_blocks(self, count=4, y=2.3):
        blocks=VGroup(*[Rectangle(width=8/count-.06,height=.33,color=BLUE_DATA,fill_color=BLUE_DATA,fill_opacity=.5) for _ in range(count)])
        blocks.arrange(RIGHT,buff=.06).move_to([0,y,0]); self.add(blocks)
        return blocks

    def cross_validation(self):
        self.legend([('その回の訓練',BLUE_DATA),('その回の評価',ORANGE_DATA)])
        ax,_=self.graph(center=(-2,-.15,0),width=6.4,height=3.05)
        points=dots(ax,model.XC,model.TC)
        self.add(points)
        blocks=self.fold_blocks()
        self.beat(LaggedStart(*[Indicate(p,color=BLUE_DATA) for p in points],lag_ratio=.025))
        line=curve(ax,model.CV_WEIGHTS[0]); errs=residuals(ax,model.CV_WEIGHTS[0],model.XC[model.FOLDS[0]],model.TC[model.FOLDS[0]],ORANGE_DATA)
        self.beat(Create(line),blocks[0].animate.set_fill(ORANGE_DATA,1),
                  *[points[int(i)].animate.set_color(ORANGE_DATA) for i in model.FOLDS[0]])
        bx=Axes(x_range=[.5,4.5,1],y_range=[0,.18,.05],x_length=3.4,y_length=2.65,tips=False,
                axis_config={'color':MUTED,'stroke_width':1}).move_to([3.4,-.15,0])
        score_head=jp('各回の平均二乗誤差',19,ORANGE_DATA).move_to([3.4,1.65,0])
        score_ticks=VGroup(*[tex(str(y),16,MUTED).next_to(bx.c2p(.5,y),LEFT,buff=.1) for y in [0,.1]])
        self.add(bx,score_head,score_ticks)
        bars=VGroup()
        for i,score in enumerate(model.CV_SCORES):
            height=score/.18*2.65
            bar=Rectangle(width=.45,height=height,color=ORANGE_DATA,fill_opacity=.65).move_to(bx.c2p(i+1,0),aligned_edge=DOWN)
            label=tex(f'{score:.3f}',19,ORANGE_DATA).next_to(bar,UP,buff=.12)
            idx=tex(str(i+1),18,MUTED).next_to(bx.c2p(i+1,0),DOWN,buff=.13)
            bars.add(VGroup(bar,label,idx))
        self.beat(Create(errs),GrowFromEdge(bars[0][0],DOWN),FadeIn(bars[0][1:]))
        for fold in range(1,4):
            held=set(model.FOLDS[fold])
            self.beat(Transform(line,curve(ax,model.CV_WEIGHTS[fold])),
                      Transform(errs,residuals(ax,model.CV_WEIGHTS[fold],model.XC[model.FOLDS[fold]],model.TC[model.FOLDS[fold]],ORANGE_DATA)),
                      *[p.animate.set_color(ORANGE_DATA if i in held else BLUE_DATA) for i,p in enumerate(points)],
                      blocks[fold-1].animate.set_fill(BLUE_DATA,.5),blocks[fold].animate.set_fill(ORANGE_DATA,1),
                      GrowFromEdge(bars[fold][0],DOWN),FadeIn(bars[fold][1:]))
        mean=model.CV_SCORES.mean()
        mean_line=DashedLine(bx.c2p(.5,mean),bx.c2p(4.5,mean),color=YELLOW_ERROR)
        formula=tex(r'E_{\rm CV}=(E_1+E_2+E_3+E_4)/4=%.4f'%mean,25,ORANGE_DATA).move_to([0,-2.18,0])
        self.beat(Create(mean_line),Write(formula))
        ratio=tex(r'\frac{S-1}{S}=\frac34\qquad d^*_{\rm CV}=3',25,BLUE_DATA).move_to([0,-2.7,0])
        # Explicitly record the candidate scores used to select, without reading the final test set.
        comparison=VGroup(*[VGroup(tex(str(d),19),tex(f'{v:.3f}',19,ORANGE_DATA)).arrange(DOWN,buff=.1)
                           for d,v in zip(model.CV_DEGREES,model.CV_MEANS)]).arrange(RIGHT,buff=.4).move_to([3.35,.2,0])
        self.remove(*bx.get_family(),mean_line,*bars.get_family(),*score_ticks.get_family(),score_head)
        self.add(jp('次数と交差検証の平均誤差',18,ORANGE_DATA).move_to([3.4,1.65,0]))
        self.beat(Write(ratio),FadeIn(comparison))

    def leave_one_out(self):
        self.legend([('訓練へ',BLUE_DATA),('評価へ',ORANGE_DATA)])
        blocks=self.fold_blocks(24,y=2.2)
        # Fixed random ordering; adjacent blocks are not adjacent x intervals.
        marker=SurroundingRectangle(VGroup(*blocks[:6]),color=ORANGE_DATA,buff=.045)
        count=ValueTracker(18)
        readout=number(r'N_{\rm train}=',count.get_value,[3.6,.4,0],BLUE_DATA,0,29)
        self.add(readout)
        ax,_=self.graph(center=(-2,-.2,0),width=6.2,height=3)
        points=dots(ax,model.XC,model.TC); self.add(points)
        line=curve(ax,model.CV_WEIGHTS[0]); self.add(line)
        fraction=tex(r'S=4:\quad 18/24',30).move_to([1,-2.3,0])
        self.beat(Create(marker),Write(fraction),*[points[int(i)].animate.set_color(ORANGE_DATA) for i in model.FOLDS[0]])
        f6=tex(r'S=6:\quad 20/24',30).move_to(fraction)
        six_groups=np.array_split(model.PERMUTATION,6)
        six_w=model.cross_validate(groups=six_groups)[0][0]
        self.beat(Transform(marker,SurroundingRectangle(VGroup(*blocks[:4]),color=ORANGE_DATA,buff=.045)),count.animate.set_value(20),TransformMatchingTex(fraction,f6),Transform(line,curve(ax,six_w)),
                  *[p.animate.set_color(ORANGE_DATA if i in six_groups[0] else BLUE_DATA) for i,p in enumerate(points)])
        f24=tex(r'S=N=24:\quad 23/24',30).move_to(f6)
        self.beat(Transform(marker,SurroundingRectangle(blocks[0],color=ORANGE_DATA,buff=.045)),count.animate.set_value(23),TransformMatchingTex(f6,f24),Transform(line,curve(ax,model.LOO_WEIGHTS[0])),
                  *[p.animate.set_color(BLUE_DATA) for p in points])
        self.remove(line)
        turn=ValueTracker(0)
        index=lambda:min(23,int(turn.get_value()))
        moving_curve=always_redraw(lambda:curve(ax,model.LOO_WEIGHTS[index()]))
        moving_point=always_redraw(lambda:Dot(points[int(model.PERMUTATION[index()])].get_center(),radius=.085,color=ORANGE_DATA))
        marker.add_updater(lambda m:m.become(SurroundingRectangle(blocks[index()],color=ORANGE_DATA,buff=.045)))
        self.add(moving_curve,moving_point)
        self.beat(turn.animate.set_value(8))
        self.beat(turn.animate.set_value(23))
        marker.clear_updaters()
        final=curve(ax,model.fit(model.XC,model.TC,model.CV_SELECTED),GREEN_TEST)
        self.beat(FadeOut(moving_curve),FadeOut(moving_point),FadeOut(marker),Create(final))
        note=jp('学習に使う量・評価のばらつき・計算量',23,YELLOW_ERROR).move_to([0,-2.8,0])
        self.beat(Write(note))

    def cost(self):
        self.legend([('1マス＝1候補',BLUE_DATA),('交差検証＝候補ごとに4回',ORANGE_DATA)])
        def grid(layers, rows):
            group=VGroup()
            for layer in range(layers):
                for r in range(rows):
                    for c in range(5):
                        box=Square(side_length=.61,color=BLUE_DATA,fill_color=BG,fill_opacity=1 if layers>1 else .25)
                        box.move_to([-4.4+c*.74+layer*.3,1.25-r*.74+layer*.25,0]); group.add(box)
            return group
        g=grid(1,1)
        self.beat(LaggedStart(*[FadeIn(b) for b in g],lag_ratio=.15))
        g20=grid(1,4)
        label=tex(r'd\quad\times\quad\lambda',34,PURPLE_TERM).move_to([-2.9,-2,0])
        self.beat(ReplacementTransform(g,g20),Write(label))
        formula=tex(r'5\times4=20',40).move_to([3,.9,0])
        self.beat(Write(formula))
        counter=ValueTracker(0)
        runs=number(r'\mathrm{runs}=',counter.get_value,[3,-.25,0],ORANGE_DATA,0,32)
        scan=SurroundingRectangle(g20[0],color=ORANGE_DATA,buff=.025)
        scan.add_updater(lambda m:m.become(SurroundingRectangle(g20[min(19,int(counter.get_value()/4))],color=ORANGE_DATA,buff=.025)))
        self.add(runs,scan)
        f80=tex(r'20\times4=80',32,ORANGE_DATA).move_to([3,-1.25,0])
        self.beat(counter.animate.set_value(80),Write(f80))
        scan.clear_updaters(); self.remove(scan)
        g60=grid(3,4)
        self.beat(ReplacementTransform(g20,g60),counter.animate.set_value(240),
                  Transform(formula,tex(r'5\times4\times3=60',34).move_to(formula)),
                  Transform(f80,tex(r'60\times4=240',32,ORANGE_DATA).move_to(f80)))
        general=tex(r'k^h\ \mathrm{candidates}\quad\Longrightarrow\quad S k^h\ \mathrm{runs}',34,YELLOW_ERROR).move_to([0,-2.55,0])
        self.beat(Write(general))
        self.beat(Indicate(runs,color=YELLOW_ERROR),Indicate(g60,color=YELLOW_ERROR))

    def aic(self):
        formula=MathTex(r'\ln p(\mathcal D\mid\mathbf w_{\rm ML})','-', 'M',font_size=34)
        formula[0].set_color(BLUE_DATA);formula[2].set_color(PURPLE_TERM)
        formula.move_to([0,2.6,0])
        ax=Axes(x_range=[1,9,1],y_range=[-16,6,5],x_length=8.5,y_length=3.5,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.2}).move_to([0,.25,0])
        labels=VGroup(jp('スコア：大きいほどよい',19,MUTED).move_to([-3.55,2.2,0]))
        for d in [1,3,5,7,9]:labels.add(tex(str(d),18,MUTED).next_to(ax.c2p(d,-16),DOWN,buff=.1))
        for v in [-15,-10,-5,0,5]:labels.add(tex(str(v),17,MUTED).next_to(ax.c2p(1,v),LEFT,buff=.1))
        labels.add(tex('d',24).next_to(ax.c2p(9,-16),RIGHT,buff=.15))
        self.add(ax,labels)
        likelihood=polyline([ax.c2p(int(d),float(v)) for d,v in zip(model.DEGREES,model.LOG_LIKELIHOOD)],BLUE_DATA)
        self.beat(Create(likelihood))
        likelihood_label=tex(r'p(\mathcal D\mid\mathbf w)',35,BLUE_DATA).move_to([0,-2.3,0])
        meaning=VGroup(tex(r'\mathcal D',24,BLUE_DATA),jp('：訓練データ',20),tex(r'\mathbf w',24,BLUE_DATA),jp('：係数',20)).arrange(RIGHT,buff=.13).move_to([0,-2.8,0])
        self.beat(Write(likelihood_label),Write(meaning))
        self.beat(ReplacementTransform(likelihood_label,formula[0]))
        self.remove(meaning)
        self.add(formula[0])
        fixed=tex(r'\ln p=-\frac N2\ln(2\pi\sigma^2)-\frac{\sum_n(y(x_n)-t_n)^2}{2\sigma^2}',28,BLUE_DATA).move_to([0,-2.3,0])
        note=VGroup(tex(r'\sigma=0.25',22,MUTED),jp('既知の標準偏差',18,MUTED)).arrange(RIGHT,buff=.25).move_to([0,-2.8,0])
        self.beat(Write(fixed),Write(note))
        mapping=tex(r'M=d+1\qquad d=3\Rightarrow M=4',30,PURPLE_TERM).move_to([0,-2.3,0])
        self.remove(fixed,note)
        self.beat(Write(mapping),FadeIn(formula[2]))
        strength=ValueTracker(0)
        values=lambda:model.LOG_LIKELIHOOD-strength.get_value()*model.PARAMETERS
        adjusted=always_redraw(lambda:polyline([ax.c2p(int(d),float(v)) for d,v in zip(model.DEGREES,values())],ORANGE_DATA))
        penalties=always_redraw(lambda:VGroup(*[Line(ax.c2p(int(d),float(top)),ax.c2p(int(d),float(bottom)-1e-6),
                         color=PURPLE_TERM,stroke_width=2) for d,top,bottom in zip(model.DEGREES,model.LOG_LIKELIHOOD,values())]))
        self.add(adjusted,penalties)
        eqno=tex('(1.73)',20,MUTED).move_to([5.3,2.6,0])
        self.beat(FadeIn(formula[1]),FadeIn(eqno),strength.animate.set_value(.5))
        selected=Dot(ax.c2p(3,float(model.AIC_SCORE[2])),radius=.085,color=YELLOW_ERROR)
        self.beat(strength.animate.set_value(1),FadeIn(selected),end_sentence=1)
        aic=tex(r'\mathrm{AIC}=-2\ln p(\mathcal D\mid\mathbf w_{\rm ML})+2M\quad\to\min',29,ORANGE_DATA).move_to([0,-2.8,0])
        self.beat(Write(aic),Indicate(selected,color=YELLOW_ERROR))

    def limits(self):
        self.legend([('観測',BLUE_DATA),('最適な係数の一本',MODEL_RED),('別の係数（模式図）',PURPLE_TERM)])
        ax,ag=self.graph()
        data=dots(ax,model.X,model.T); line=curve(ax,model.WEIGHTS[2])
        self.add(data,line)
        self.beat(Indicate(line,color=MODEL_RED))
        variants=VGroup()
        for shift in np.linspace(-.18,.18,7):
            w=model.WEIGHTS[2].copy();w[0]+=shift;w[1]-=shift*.8
            variants.add(curve(ax,w,PURPLE_TERM).set_stroke(opacity=.35,width=2))
        self.beat(LaggedStart(*[Create(v) for v in variants],lag_ratio=.1))
        note=jp('一本の最適解と、係数の不確かさ',26,YELLOW_ERROR).move_to([0,-2.15,0])
        self.beat(Write(note),Indicate(variants,color=PURPLE_TERM))
        road=VGroup(tex(r'\mathrm{BIC}\ \to\ 4.4.1',28,PURPLE_TERM),jp('ベイズ的モデル比較 → 3.4',23,GREEN_TEST)).arrange(RIGHT,buff=.7).move_to([0,-2.75,0])
        self.beat(Write(road))
        self.beat(LaggedStart(*[Indicate(v,color=PURPLE_TERM,scale_factor=1.025) for v in variants],lag_ratio=.2))
        self.remove(*variants.get_family(),note,road)
        valid=dots(ax,model.XV,model.TV,ORANGE_DATA)
        question=jp('未知の点を予測するために、選ぶ',27,ORANGE_DATA).move_to([0,-2.15,0])
        self.beat(FadeIn(valid),Write(question))
        roles=VGroup(*[jp(s,23,c) for s,c in [('学ぶ',BLUE_DATA),('選ぶ',ORANGE_DATA),('最後に測る',GREEN_TEST)]]).arrange(RIGHT,buff=.8).move_to([0,-2.75,0])
        self.beat(LaggedStart(*[Write(r) for r in roles],lag_ratio=.5))
