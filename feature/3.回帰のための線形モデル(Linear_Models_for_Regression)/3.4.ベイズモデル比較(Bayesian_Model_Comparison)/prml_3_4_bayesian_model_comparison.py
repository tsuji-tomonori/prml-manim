"""PRML 3.4 — linked visual experiments in Manim Community."""
import json
from pathlib import Path
import numpy as np
from manim import *
from caption_layout import jp, tex, caption_mobject
from narration_content import SCENES, estimated_duration
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry
import bayesian_model as bm

BG='#10141F'
BLUE=ManimColor('#58B5ED')
PURPLE=ManimColor('#C29AFF')
GOLD=ManimColor('#FFE079')
GREEN=ManimColor('#77D49A')
RED=ManimColor('#FF6B77')
MUTED=ManimColor('#A8B2C5')
COLORS=[BLUE,GREEN,PURPLE]

def curve(ax,xs,ys,color=BLUE,width=3):
    return VMobject().set_points_as_corners([ax.c2p(float(x),float(y)) for x,y in zip(xs,ys)]).set_stroke(color,width)

def graph(ax,fn,lo,hi,color=BLUE):
    xs=np.linspace(lo,hi,241)
    return curve(ax,xs,fn(xs),color)

def area(ax,fn,lo,hi,color=GOLD):
    xs=np.linspace(lo,max(lo+.00001,hi),121)
    points=[ax.c2p(lo,0)]+[ax.c2p(float(x),float(fn(x))) for x in xs]+[ax.c2p(hi,0)]
    return Polygon(*points,stroke_width=0,fill_color=color,fill_opacity=.35)

def readout(label,getter,pos,color=WHITE,places=3):
    prefix=tex(label,25,color)
    num=DecimalNumber(getter(),num_decimal_places=places,font_size=25,color=color)
    g=VGroup(prefix,num).arrange(RIGHT,buff=.14).move_to(pos)
    anchor=num.get_left().copy()
    num.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return g

def pulse(m,color=GOLD):
    return Circumscribe(m,color=color,buff=.1)

class PRML34BayesianModelComparison(Scene):
    def construct(self):
        self.camera.background_color=BG
        self.timeline=[]
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        methods=[self.question,self.integral,self.models,self.occam,self.data_space,
                 self.regression,self.mixture,self.expected,self.caution]
        for i,method in enumerate(methods):
            self.begin(i)
            if not self.audio_entry:
                raise RuntimeError('Generate matching narration before rendering')
            method()
            assert self.beat_index==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path('media/prml34_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def ax(self,xr,yr,center=(0,.15,0),width=9,height=3.25):
        return Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
            axis_config={'color':MUTED,'stroke_width':1.4,'include_numbers':True,'font_size':19}).move_to(center)

    def formula(self,*parts,colors=None,y=-2.48,size=31):
        m=MathTex(*parts,font_size=size).move_to([0,y,0])
        if colors:
            for part,c in zip(m,colors):part.set_color(c)
        if m.width>12.7:raise ValueError('Formula too wide')
        return m

    def legend(self,items,y=2.25):
        return VGroup(*[VGroup(Line(LEFT*.18,RIGHT*.18,color=c,stroke_width=4),jp(t,21,c)).arrange(RIGHT,buff=.13)
                        for t,c in items]).arrange(RIGHT,buff=.5).move_to([0,y,0])

    def slider(self,tr,lo,hi,label,pos=(0,2.25,0),width=4):
        rail=Line(LEFT*width/2,RIGHT*width/2,color=MUTED).move_to(pos)
        dot=Dot(color=GOLD,radius=.07)
        dot.add_updater(lambda m:m.move_to(rail.point_from_proportion(np.clip((tr.get_value()-lo)/(hi-lo),0,1))))
        number=readout(label,tr.get_value,[pos[0]+width/2+1.3,pos[1],0],GOLD,2)
        return VGroup(rail,dot,number)

    def recap_card(self, title):
        """Temporarily replace the body; keep the header and PCM caption clock."""
        saved=[m for m in self.mobjects if m is not self.subtitle]
        header=[m for m in saved if m.get_center()[1]>2.65]
        self.clear()
        label=jp(title,23).move_to([-4.85,2.02,0],aligned_edge=LEFT)
        self.add(*header,label)
        return saved

    def restore_recap(self, saved):
        self.clear()
        self.add(*saved)

    def roles_recap(self):
        saved=self.recap_card('復習: 1.3 訓練・検証・テスト')
        orange=ManimColor('#FFB45B')
        rows=VGroup()
        # Same square data strips and colours as 1.3 roles().
        for i,(label,color) in enumerate(zip(['訓練：係数を学ぶ','検証：モデルを選ぶ','テスト：最後に測る'],[BLUE,orange,GREEN])):
            y=1.12-i*.8
            blocks=VGroup(*[Square(side_length=.22,color=color,fill_opacity=.65) for _ in range(10)])
            blocks.arrange(RIGHT,buff=.06).move_to([-2.45,y,0])
            rows.add(VGroup(blocks,jp(label,23,color).move_to([1.25,y,0])))
        arrow=Arrow([- .9,.32,0],[-.15,.32,0],buff=.06,color=orange,stroke_width=2)
        current=jp('今回：データ全体への予測でモデルを比べる',23,GOLD).move_to([0,-1.53,0])
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('R1.3 three data roles',a*.48,lambda:FadeIn(rows)),
            ('R1.3 validation selects model',a*.52,lambda:AnimationGroup(GrowArrow(arrow),pulse(rows[1],orange))),
            ('R1.3 map to evidence comparison',b*.65,lambda:FadeIn(current)),
            ('R1.3 compare models',b*.35,lambda:pulse(current)),
        ])
        self.restore_recap(saved)

    def bayes_recap(self):
        saved=self.recap_card('復習: 1.2 ベイズの定理')
        # 1.2 bayes(): red/blue boxes, masses .3*.75 and .7*.2.
        prior=np.array([.3,.7]);evidence=np.array([.75,.2]);joint=prior*evidence
        def strip(values):
            return VGroup(*[Rectangle(width=7*v,height=1.15,stroke_color=c,fill_color=c,fill_opacity=.65)
                .move_to([-3.5+7*sum(values[:i])+3.5*v,.05,0])
                for i,(v,c) in enumerate(zip(values,[RED,BLUE]))])
        band=strip(joint)
        full=strip(joint/joint.sum())
        boxes=VGroup(jp('赤い箱',23,RED),jp('青い箱',23,BLUE)).arrange(RIGHT,buff=2).move_to([0,1.2,0])
        models=VGroup(tex('M_1',29,RED),tex('M_2',29,BLUE)).arrange(RIGHT,buff=2.4).move_to(boxes)
        factors=VGroup(tex(r'0.3\times0.75=0.225',26,RED),tex(r'0.7\times0.2=0.140',26,BLUE)).arrange(RIGHT,buff=.7).move_to([0,-.92,0])
        meaning=jp('事前確率 × モデル証拠',23,GOLD).move_to([0,-1.65,0])
        total=jp('合計 1',23,GREEN).move_to([3.8,1.2,0])
        self.add(jp('説明用の例：観測は2種類',18,MUTED).move_to([2.65,2.02,0]))
        self.add(boxes,band)
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        def model_labels():
            # Avoid morphing Japanese glyphs into overlapping mathematical glyphs.
            self.remove(boxes)
            return AnimationGroup(FadeIn(models),FadeOut(total))
        self.beat(phases=[
            ('R1.2 observed areas',a*.60,lambda:pulse(band)),
            ('R1.2 normalize box areas',a*.40,lambda:AnimationGroup(Transform(band,full),FadeIn(total))),
            ('R1.2 boxes become models',b*.52,model_labels),
            ('R1.2 prior times evidence',b*.48,lambda:AnimationGroup(Transform(band,strip(joint)),FadeIn(factors),FadeIn(meaning))),
            ('R1.2 normalize model weights',c,lambda:AnimationGroup(Transform(band,full),FadeIn(total))),
        ])
        self.restore_recap(saved)

    def kl_recap(self):
        saved=self.recap_card('復習: 1.6 KLダイバージェンス')
        orange=ManimColor('#FFB45B')
        # Reuse the four-result example from 1.6 divergence(), in natural logs.
        p=np.array([.5,.25,.125,.125]);q=np.array([.1,.2,.3,.4])
        excess=float(p@np.log(p/q))
        def bars(values,color,outline=False):
            return VGroup(*[Rectangle(width=.62,height=3*v,stroke_color=color,stroke_width=2,
                fill_color=color,fill_opacity=0 if outline else .7).move_to([-3.9+i*.92,-.8+1.5*v,0])
                for i,v in enumerate(values)])
        pb=bars(p,BLUE);qb=bars(q,orange,True)
        xl=VGroup(*[tex(f'x_{i+1}',24).move_to([-3.9+i*.92,-1.12,0]) for i in range(4)])
        dl=VGroup(*[tex(f'D_{i+1}',24).move_to(xl[i]) for i in range(4)])
        legend=VGroup(jp('真の p',21,BLUE),jp('想定 q',21,orange)).arrange(RIGHT,buff=.4).move_to([-2.5,1.24,0])
        mapping=VGroup(tex(r'p(D)=p(D\mid M_1)',25,BLUE),tex(r'q(D)=p(D\mid M_2)',25,orange)).arrange(DOWN,buff=.2).move_to([2.1,1.03,0])
        formula=tex(r'\sum_D p(D)\ln\frac{p(D)}{q(D)}',29,GOLD).move_to([1.5,.05,0])
        bar=Rectangle(width=excess*3,height=.38,stroke_width=0,fill_color=GOLD,fill_opacity=.85).move_to([.6+excess*1.5,-.82,0])
        result=tex(r'\mathrm{KL}=%.3f'%excess,26,PURPLE).move_to([2.05,-1.3,0])
        meaning=jp('余分な平均コスト',22,PURPLE).move_to([2,1.18,0])
        self.add(jp('説明用の例：4通り・確率はすべて正',17,MUTED).move_to([2.5,2.02,0]))
        self.add(pb,qb,xl,legend,meaning)
        a,b,c=[self.sentence_duration(i) for i in range(3)]
        def dataset_labels():
            # The caption is already explaining the new meaning at this boundary.
            self.remove(meaning)
            return AnimationGroup(Transform(xl,dl),FadeIn(mapping))
        self.beat(phases=[
            ('R1.6 true and assumed distributions',a*.55,lambda:pulse(qb,orange)),
            ('R1.6 extra average cost',a*.45,lambda:AnimationGroup(FadeIn(bar),FadeIn(result))),
            ('R1.6 outcomes become datasets',b,dataset_labels),
            ('R1.6 log ratio averaged under truth',c*.65,lambda:AnimationGroup(Write(formula),pulse(pb,BLUE))),
            ('R1.6 connect average to KL',c*.35,lambda:pulse(bar,GOLD)),
        ])
        self.restore_recap(saved)

    def question(self):
        ax=self.ax([-1,1,.5],[0,1.8,.5])
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.055,color=BLUE) for x,t in zip(bm.X,bm.T)])
        labels=self.legend([('同じ12点',BLUE),('最良の1本',GREEN)])
        self.add(ax,labels,tex('x',25).next_to(ax.x_axis,RIGHT,buff=.2),tex('t',25).move_to(ax.c2p(0,1.8)+LEFT*.28))
        self.beat(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=.12))
        degree=ValueTracker(1)
        fit=always_redraw(lambda:curve(ax,bm.GRID,bm.interpolate_rows(bm.ML_CURVES,degree.get_value()),GREEN))
        label=readout('d=',degree.get_value,[4.8,1.65,0],GREEN,0)
        self.add(fit,label)
        self.beat(degree.animate.set_value(2),end_sentence=1)
        self.roles_recap()
        residual=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,float(np.interp(x,bm.GRID,bm.interpolate_rows(bm.ML_CURVES,degree.get_value())))),color=GOLD,stroke_width=2)
                                                for x,t in zip(bm.X,bm.T)]))
        self.add(residual)
        def score_row(name, numbers, color, y):
            return VGroup(jp(name,21,color),tex(numbers,26,color)).arrange(RIGHT,buff=.25).move_to([0,y,0])
        ml_score=score_row('最良の1本  対数尤度',
            rf'd=2:{bm.LOG_ML[2]:.2f}\quad d=7:{bm.LOG_ML[7]:.2f}',GOLD,-2.02)
        evidence_score=score_row('モデル全体  対数エビデンス',
            rf'd=2:{bm.LOG_EVIDENCE[2]:.2f}\quad d=7:{bm.LOG_EVIDENCE[7]:.2f}',GREEN,-2.48)
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[('increase degree',a,lambda:degree.animate.set_value(7)),
                          ('best curve wins',b,lambda:FadeIn(ml_score))])
        rng=np.random.default_rng(34)
        fit7=bm.FITS[7]
        alternatives=VGroup(*[
            curve(ax,bm.GRID,bm.design(bm.GRID,7)@(
                fit7['mean']+.42*np.linalg.cholesky(fit7['cov'])@rng.normal(size=8)),PURPLE,2)
            .set_stroke(opacity=.48) for _ in range(4)])
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[('ranking reverses',a,lambda:FadeIn(evidence_score)),
                          ('other coefficient choices',b,lambda:AnimationGroup(
                              FadeOut(residual),LaggedStart(*[Create(c) for c in alternatives],lag_ratio=.15)))])
        f=self.formula(r'M_i:\quad p(D\mid\mathbf w,M_i)',r'p(\mathbf w\mid M_i)',colors=[BLUE,PURPLE])
        self.remove(ml_score,evidence_score)
        self.beat(Write(f))
        self.beat(pulse(alternatives),pulse(f))

    def integral(self):
        ax=self.ax([-2,2,.5],[0,1.3,.5])
        like=lambda w:bm.normal(w,1,.35)
        prior=curve(ax,[-2,-2,2,2],[0,.25,.25,0],PURPLE)
        l=graph(ax,like,-2,2,BLUE)
        self.add(ax,tex('w',26).next_to(ax.x_axis,RIGHT,buff=.2))
        obs=jp('観測  t = 1   ／   ノイズ標準偏差  0.35',23).move_to([0,2.25,0])
        self.beat(FadeIn(obs),Create(l))
        tr=ValueTracker(-1.5)
        dot=always_redraw(lambda:Dot(ax.c2p(tr.get_value(),like(tr.get_value())),color=BLUE,radius=.08))
        v=always_redraw(lambda:Line(ax.c2p(tr.get_value(),0),dot.get_center(),color=BLUE))
        self.add(dot,v)
        self.beat(tr.animate.set_value(1.5))
        legend=self.legend([('尤度',BLUE),('事前',PURPLE),('積',GOLD)])
        self.beat(FadeOut(obs),FadeIn(legend),Create(prior))
        sweep=ValueTracker(-2)
        product=lambda w:like(w)/4
        shade=always_redraw(lambda:area(ax,product,-2,sweep.get_value()))
        p=graph(ax,product,-2,2,GOLD)
        self.add(shade)
        f=self.formula(r'p(D\mid M_i)=',r'\int',r'p(D\mid w,M_i)',r'p(w\mid M_i)',r'\,dw',colors=[GOLD,WHITE,BLUE,PURPLE,WHITE],size=30)
        self.beat(Create(p),sweep.animate.set_value(2),Write(f))
        rects=VGroup(*[Rectangle(width=(ax.c2p(-2+.1,0)-ax.c2p(-2,0))[0]*.96,height=max(.001,(ax.c2p(w,product(w))-ax.c2p(w,0))[1]),fill_color=GOLD,fill_opacity=.5,stroke_color=GOLD,stroke_width=.4).move_to((ax.c2p(w,0)+ax.c2p(w,product(w)))/2) for w in np.arange(-1.95,2,.1)])
        self.beat(LaggedStart(*[FadeIn(r) for r in rects],lag_ratio=.03),pulse(f[1]))
        self.remove(shade,rects,*rects,dot,v)
        z=bm.uniform_evidence(4)
        posterior=graph(ax,lambda w:product(w)/z,-2,2,GREEN)
        f2=self.formula(r'p(w\mid D,M_i)=',r'\frac{p(D\mid w,M_i)p(w\mid M_i)}{p(D\mid M_i)}',colors=[GREEN,GOLD],size=31)
        posterior_legend=self.legend([('尤度',BLUE),('事前',PURPLE),('事後',GREEN)])
        self.beat(Transform(p,posterior),ReplacementTransform(f,f2),l.animate.set_stroke(opacity=.2),
                  ReplacementTransform(legend,posterior_legend))
        flow=jp('事前から係数を選ぶ → 観測を生成する',24,GREEN).move_to([0,2.25,0])
        self.remove(legend,*legend,posterior_legend,*posterior_legend)
        self.beat(FadeIn(flow),pulse(f2))

    def models(self):
        self.bayes_recap()
        ax=self.ax([.5,3.5,1],[0,1,.25],width=7.8)
        # Explicit labels avoid suggesting model identifiers are continuous.
        ax.x_axis.numbers.set_opacity(0)
        labels=VGroup(*[tex(f'M_{i+1}',27,COLORS[i]).move_to(ax.c2p(i+1,0)+DOWN*.35) for i in range(3)])
        values=np.ones(3)/3
        def bars(v):
            return VGroup(*[Rectangle(width=1.15,height=max(.008,3.25*x),fill_color=COLORS[i],fill_opacity=.8,stroke_width=0).move_to(ax.c2p(i+1,0)+UP*max(.008,3.25*x)/2) for i,x in enumerate(v)])
        b=bars(values)
        state=jp('モデル事前確率：合計 1',25,PURPLE).move_to([0,2.25,0])
        self.add(ax,labels)
        self.beat(FadeIn(b),FadeIn(state))
        evidence=VGroup(*[tex(f'{e:.5f}',25,GOLD).move_to(ax.c2p(i+1,.82)) for i,e in enumerate(bm.EVIDENCES)])
        self.beat(FadeIn(evidence))
        f=self.formula(r'p(M_i\mid D)\propto',r'p(M_i)',r'p(D\mid M_i)',colors=[GREEN,PURPLE,GOLD])
        new=jp('事前 × エビデンス：まだ合計 1 ではない',24,GOLD).move_to(state)
        self.remove(state)
        self.add(new)
        self.beat(Transform(b,bars(values*bm.EVIDENCES)),Write(f))
        new2=jp('モデル事後確率：合計 1',25,GREEN).move_to(new)
        self.remove(new)
        self.add(new2)
        self.beat(Transform(b,bars(bm.posterior())))
        odds=self.formula(r'\frac{p(M_2\mid D)}{p(M_3\mid D)}=',r'\frac{p(M_2)}{p(M_3)}',r'\frac{p(D\mid M_2)}{p(D\mid M_3)}',colors=[GREEN,PURPLE,GOLD],size=30)
        self.beat(ReplacementTransform(f,odds),pulse(evidence[1:]))
        self.remove(b)
        preference=ValueTracker(1/3)
        dynamic=always_redraw(lambda:bars(bm.posterior(np.array([(1-preference.get_value())/2,preference.get_value(),(1-preference.get_value())/2]))))
        slider=self.slider(preference,.1,.8,r'p(M_2)=',pos=(0,1.9,0),width=3)
        self.add(dynamic)
        self.remove(new2,evidence)
        self.add(slider)
        self.beat(preference.animate.set_value(.75),start_sentence=1)

    def width_plot(self):
        width=ValueTracker(4)
        ax=self.ax([-4,4,2],[0,1.3,.5])
        like=lambda w:bm.normal(w,1,.35)
        prior=always_redraw(lambda:curve(ax,[-width.get_value()/2,-width.get_value()/2,width.get_value()/2,width.get_value()/2],[0,1/width.get_value(),1/width.get_value(),0],PURPLE))
        product=always_redraw(lambda:graph(ax,lambda w:like(w)/width.get_value(),-width.get_value()/2,width.get_value()/2,GOLD))
        shade=always_redraw(lambda:area(ax,lambda w:like(w)/width.get_value(),-width.get_value()/2,width.get_value()/2))
        l=graph(ax,like,-4,4,BLUE)
        z=readout('p(D)=',lambda:bm.uniform_evidence(width.get_value()),[3.1,1.45,0],GOLD)
        slider=self.slider(width,4,8,r'\Delta w_{\rm prior}=',pos=(-1.1,2.25,0),width=3.5)
        self.add(ax,l,prior,shade,product,z,slider,tex('w',25).next_to(ax.x_axis,RIGHT,buff=.2))
        return width,VGroup(ax,l,prior,shade,product,z,slider)

    def occam(self):
        width,visual=self.width_plot()
        self.beat(pulse(visual[1]))
        self.beat(width.animate.set_value(8))
        ax=visual[0]
        effective=np.sqrt(2*np.pi)*.35
        rect=Polygon(ax.c2p(1-effective/2,0),ax.c2p(1-effective/2,bm.normal(1,1,.35)/8),ax.c2p(1+effective/2,bm.normal(1,1,.35)/8),ax.c2p(1+effective/2,0),fill_color=GREEN,fill_opacity=.35,stroke_color=GREEN)
        label=tex(r'w_{\rm MAP}=1',27,GREEN).move_to([-2,1.5,0])
        width_label=tex(r'\Delta w_{\rm posterior}',26,GREEN).move_to([-2,.5,0])
        pointer=Arrow(width_label.get_right(),rect.get_top(),color=GREEN,buff=.15,stroke_width=2,tip_length=.15)
        self.beat(Create(rect),FadeIn(label),FadeIn(width_label),Create(pointer))
        f=self.formula(r'p(D)\approx',r'p(D\mid w_{\rm MAP})',r'\frac{\Delta w_{\rm posterior}}{\Delta w_{\rm prior}}',colors=[GOLD,BLUE,PURPLE],size=31)
        self.beat(Write(f),pulse(rect))
        f2=self.formula(r'\ln p(D)\approx',r'\ln p(D\mid w_{\rm MAP})',r'+\ln\frac{\Delta w_{\rm posterior}}{\Delta w_{\rm prior}}',colors=[GOLD,BLUE,PURPLE],size=30)
        self.beat(ReplacementTransform(f,f2))
        # Same probability budget, now in two parameter directions.
        self.remove(*visual,visual,rect,label,width_label,pointer)
        for mob in list(self.mobjects):
            if mob not in [f2,self.subtitle] and mob.get_center()[1]<2.65:self.remove(mob)
        square=Square(side_length=3.2,color=PURPLE,fill_opacity=.08).move_to([-2.4,.1,0])
        strip=Rectangle(width=1.6,height=3.2,color=GREEN,fill_opacity=.45).move_to(square)
        small=Square(side_length=1.6,color=GREEN,fill_opacity=.5).move_to(square)
        ratio=tex(r'\frac12\times\frac12=\frac14',37,GREEN).move_to([2.4,.25,0])
        self.add(square,strip)
        self.beat(Transform(strip,small),Write(ratio))
        f3=self.formula(r'\ln p(D)\approx',r'\ln p(D\mid w_{\rm MAP})',r'+M\ln\frac{\Delta w_{\rm posterior}}{\Delta w_{\rm prior}}',colors=[GOLD,BLUE,PURPLE],size=29)
        note=jp('M：係数の個数   ／   同じ幅比 r の近似',24).move_to([0,2.2,0])
        self.beat(ReplacementTransform(f2,f3),FadeIn(note),pulse(strip))

    def data_space(self):
        ax=self.ax([-6,6,2],[0,1.2,.4],height=3.25)
        sd=ValueTracker(.35);datum=ValueTracker(1.4)
        c=always_redraw(lambda:graph(ax,lambda x:bm.normal(x,std=sd.get_value()),-6,6,GREEN))
        marker=always_redraw(lambda:DashedLine(ax.c2p(datum.get_value(),0),ax.c2p(datum.get_value(),1.16),color=GOLD))
        self.add(ax,tex('D',26).next_to(ax.x_axis,RIGHT,buff=.2))
        foot=self.formula(r'\int p(D\mid M_i)\,dD=1',size=32)
        self.beat(Write(foot))
        slider=self.slider(sd,.35,3,r'\sigma_D=',pos=(-1.2,2.25,0),width=3.5)
        self.add(slider,c,marker)
        dot=always_redraw(lambda:Dot(ax.c2p(datum.get_value(),bm.normal(datum.get_value(),std=sd.get_value())),color=GOLD,radius=.075))
        z=readout('p(D_0)=',lambda:bm.normal(datum.get_value(),std=sd.get_value()),[3.4,1.7,0],GOLD,5)
        self.add(dot,z)
        self.beat(pulse(c))
        self.beat(sd.animate.set_value(1.3),end_sentence=1)
        self.beat(sd.animate.set_value(3))
        self.remove(c,dot,z,slider)
        curves=VGroup(*[graph(ax,lambda x,s=s:bm.normal(x,std=s),-6,6,col) for s,col in zip(bm.MODEL_STDS,COLORS)])
        dots=always_redraw(lambda:VGroup(*[Dot(ax.c2p(datum.get_value(),bm.normal(datum.get_value(),std=s)),color=col,radius=.065) for s,col in zip(bm.MODEL_STDS,COLORS)]))
        legend=self.legend([('狭い 0.35',BLUE),('中 1.30',GREEN),('広い 3.00',PURPLE)])
        self.add(dots)
        self.beat(FadeIn(curves),FadeIn(legend))
        self.beat(datum.animate.set_value(.1),end_sentence=1)
        self.beat(datum.animate.set_value(1.4),pulse(foot))

    def regression(self):
        ax=self.ax([-1,1,1],[0,1.8,.5],center=(-3.15,.15,0),width=5.5,height=3.15)
        score=self.ax([0,7,1],[-22,10,10],center=(3.2,.15,0),width=5.15,height=3.15)
        d=ValueTracker(0)
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.045,color=BLUE) for x,t in zip(bm.X,bm.T)])
        fit=always_redraw(lambda:curve(ax,bm.GRID,bm.interpolate_rows(bm.CURVES,d.get_value()),GREEN))
        ev=curve(score,bm.DEGREES,bm.LOG_EVIDENCE,GREEN)
        ml=curve(score,bm.DEGREES,bm.LOG_ML,GOLD)
        marker=always_redraw(lambda:Dot(score.c2p(d.get_value(),float(np.interp(d.get_value(),bm.DEGREES,bm.LOG_EVIDENCE))),color=GREEN,radius=.075))
        slider=self.slider(d,0,7,'d=',pos=(-1,2.25,0),width=4)
        legend=self.legend([('対数エビデンス',GREEN),('最大対数尤度',GOLD)],y=-1.97)
        self.add(ax,score,dots,fit,slider,tex('x',24).next_to(ax.x_axis,RIGHT,buff=.1),tex('d',24).next_to(score.x_axis,RIGHT,buff=.1))
        self.beat(Create(ev),FadeIn(marker))
        self.beat(d.animate.set_value(2),end_sentence=1)
        self.beat(d.animate.set_value(7))
        self.beat(Create(ml),FadeIn(legend))
        f=self.formula(r'\mathbf w\sim\mathcal N(0,I),\quad',r'C=0.18^2 I+\Phi\Phi^T,\quad',r'p(\mathbf t)=\mathcal N(\mathbf t\mid0,C)',colors=[PURPLE,BLUE,GOLD],size=25)
        self.beat(Write(f))
        self.beat(d.animate.set_value(2),pulse(ev))

    def mixture(self):
        ax=self.ax([-3.5,3.5,1],[0,1.3,.5])
        p=ValueTracker(.5);scale=ValueTracker(0)
        a=lambda x:bm.normal(x,-1.35,.32)
        b=lambda x:bm.normal(x,1.35,.32)
        c1=always_redraw(lambda:graph(ax,lambda x:a(x)*(1-scale.get_value()+scale.get_value()*p.get_value()),-3.5,3.5,BLUE))
        c2=always_redraw(lambda:graph(ax,lambda x:b(x)*(1-scale.get_value()+scale.get_value()*(1-p.get_value())),-3.5,3.5,PURPLE))
        self.add(ax,tex('t',25).next_to(ax.x_axis,RIGHT,buff=.2))
        recap_label=jp('復習: 2.3 混合分布',23).move_to([0,2.25,0])
        self.add(recap_label,c1,c2)
        first_duration,second_duration=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[('show two predictions',first_duration,lambda:AnimationGroup(
                              Indicate(c1,color=BLUE),Indicate(c2,color=PURPLE))),
                          ('compare two peaks',second_duration,lambda:AnimationGroup(
                              Indicate(c1,color=BLUE),Indicate(c2,color=PURPLE)))])
        self.remove(recap_label)
        slider=self.slider(p,0,1,r'p(M_1\mid D)=',pos=(-1.5,2.25,0),width=3)
        self.add(slider)
        self.beat(scale.animate.set_value(1),start_sentence=1)
        mix=always_redraw(lambda:graph(ax,lambda x:p.get_value()*a(x)+(1-p.get_value())*b(x),-3.5,3.5,WHITE))
        self.beat(Create(mix))
        mean=always_redraw(lambda:DashedLine(ax.c2p(-1.35*p.get_value()+1.35*(1-p.get_value()),0),ax.c2p(-1.35*p.get_value()+1.35*(1-p.get_value()),.45),color=GOLD))
        label=jp('平均',24,GOLD).move_to([.6,-.2,0])
        self.beat(Create(mean),FadeIn(label))
        self.remove(label)
        self.beat(p.animate.set_value(.85))
        f=self.formula(r'p(t\mid x,D)=',r'\sum_i',r'p(t\mid x,M_i,D)',r'p(M_i\mid D)',colors=[WHITE,WHITE,BLUE,PURPLE],size=29)
        self.beat(Write(f),p.animate.set_value(.5))

    def expected(self):
        ax=self.ax([-.65,6.5,1],[-4,4,2],width=8.4,height=3.2)
        ax.x_axis.numbers.set_opacity(0)
        ax.y_axis.set_opacity(0)
        def ticks(axis,ys):
            bottom=axis.y_range[0]
            return VGroup(Line(axis.c2p(-.6,bottom),axis.c2p(-.6,axis.y_range[1]),color=MUTED,stroke_width=1.2),
                *[tex(str(k),20,MUTED).move_to(axis.c2p(k,bottom)+DOWN*.18) for k in bm.K],
                *[tex(f'{y:g}',19,MUTED).next_to(axis.c2p(-.6,y),LEFT,buff=.1) for y in ys])
        ticks0=ticks(ax,[-4,-2,0,2,4])
        labels=self.legend([('真：表の確率 0.65',BLUE),('候補：表の確率 0.35',PURPLE)])
        def signed_bars(values,axis=ax):
            out=VGroup()
            for k,value in zip(bm.K,values):
                zero=axis.c2p(k,0);top=axis.c2p(k,value)
                out.add(Rectangle(width=.65,height=max(.005,abs(top[1]-zero[1])),stroke_width=0,fill_color=GREEN if value>=0 else RED,fill_opacity=.85).move_to((zero+top)/2))
            return out
        bars=signed_bars(bm.LOG_BF)
        self.add(ax,labels,ticks0,tex('k',26).next_to(ax.x_axis,RIGHT,buff=.15))
        self.beat(FadeIn(bars))
        self.beat(pulse(bars[:3],RED))
        f=self.formula(r'\ell(k)=\ln\frac{p(k\mid M_1)}{p(k\mid M_2)}',size=31)
        self.beat(Write(f),pulse(bars[4:],GREEN))
        weighted_ax=self.ax([-.65,6.5,1],[-.2,.8,.2],width=8.4,height=3.2)
        weighted_ax.x_axis.numbers.set_opacity(0)
        weighted_ax.y_axis.set_opacity(0)
        weighted=signed_bars(bm.COUNT_PROBS*bm.LOG_BF,weighted_ax)
        ticks1=ticks(weighted_ax,[-.2,0,.2,.4,.6,.8])
        average=readout(r'\mathbb E[\ell]=',lambda:bm.EXPECTED_LOG_BF,[3,1.45,0],GREEN)
        self.remove(ticks0,*ticks0,labels,f)
        weighted_label=jp('棒の高さ：確率 × 対数比（縦軸を拡大）',24).move_to([0,2.25,0])
        weighted_formula=self.formula(r'\mathbb E[\ell]=\sum_{k=0}^6 p(k\mid M_1)\ell(k)',size=31)
        self.add(weighted_label,weighted_formula)
        self.beat(Transform(bars,weighted),Transform(ax,weighted_ax),FadeIn(ticks1),FadeIn(average))
        self.kl_recap()
        kl=self.formula(r'\int',r'p(D\mid M_1)',r'\ln\frac{p(D\mid M_1)}{p(D\mid M_2)}',r'\,dD=D_{\rm KL}(p_1\Vert p_2)\geq0',colors=[WHITE,BLUE,GOLD,GREEN],size=28)
        self.beat(ReplacementTransform(weighted_formula,kl),pulse(average))
        note=jp('真の分布が候補に含まれるときの、対数比の平均',24).move_to([0,2.25,0])
        self.remove(labels,weighted_label)
        self.beat(FadeIn(note),pulse(kl))

    def caution(self):
        width,visual=self.width_plot()
        self.beat(width.animate.set_value(8))
        # Change parameter range and show the exact integral as width grows further.
        self.remove(*visual,visual)
        self.clear()
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]),jp(self.story['reference'],16,MUTED).move_to([0,2.83,0]))
        ax=self.ax([4,40,8],[0,.26,.1])
        width.set_value(8)
        c=graph(ax,lambda w:np.array([bm.uniform_evidence(float(v)) for v in np.atleast_1d(w)]),4,40,GOLD)
        point=always_redraw(lambda:Dot(ax.c2p(width.get_value(),bm.uniform_evidence(width.get_value())),color=GOLD,radius=.08))
        self.add(ax,c,point,tex(r'\Delta w_{\rm prior}',25).move_to([3.9,-1.98,0]))
        z=readout('p(D)=',lambda:bm.uniform_evidence(width.get_value()),[2.5,1.5,0],GOLD)
        self.add(z)
        self.beat(width.animate.set_value(40))
        f=self.formula(r'\int p(w)\,dw=1',r'\quad\Longrightarrow\quad p(D)\ \mathrm{defined}',colors=[PURPLE,GOLD],size=30)
        self.beat(Write(f))
        self.remove(ax,c,point,z)
        for mob in list(self.mobjects):
            if mob is not self.subtitle and mob is not f and mob.get_center()[1]<2.65:self.remove(mob)
        train=VGroup(*[Dot([-.15+(i%6)*.42-3.8,1.1-(i//6)*.42,0],color=BLUE,radius=.07) for i in range(18)])
        test=VGroup(*[Dot([1.7+(i%4)*.42,1.1-(i//4)*.42,0],color=GREEN,radius=.07) for i in range(12)])
        names=VGroup(jp('訓練：候補を比較',27,BLUE).move_to([-2.8,1.9,0]),jp('独立テスト：性能を確認',27,GREEN).move_to([2.4,1.9,0]))
        self.beat(FadeIn(train),FadeIn(test),FadeIn(names))
        self.remove(train,test,names,f)
        chain=VGroup(self.formula(r'\int p(D\mid w,M_i)p(w\mid M_i)\,dw',y=1.65,size=33),
                     self.formula(r'p(M_i\mid D)\propto p(M_i)p(D\mid M_i)',y=.2,size=33),
                     self.formula(r'p(t\mid x,D)=\sum_i p(t\mid x,M_i,D)p(M_i\mid D)',y=-1.25,size=30))
        chain[0].set_color(GOLD);chain[1].set_color(GREEN);chain[2].set_color(BLUE)
        self.beat(LaggedStart(*[Write(c) for c in chain],lag_ratio=.3),end_sentence=1)
        # Return to precisely the opening dataset and its two computed rankings.
        self.clear()
        self.add(jp(self.story['title'],34).move_to([0,3.35,0]),
                 jp(self.story['reference'],16,MUTED).move_to([0,2.83,0]))
        ax=self.ax([-1,1,.5],[0,1.8,.5],center=(0,.2,0),width=8.4,height=3.1)
        degree=ValueTracker(7)
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.055,color=BLUE) for x,t in zip(bm.X,bm.T)])
        fitted=always_redraw(lambda:curve(ax,bm.GRID,bm.interpolate_rows(
            bm.ML_CURVES,degree.get_value()),GREEN))
        degree_label=readout('d=',degree.get_value,[4.85,1.55,0],GREEN,0)
        best=VGroup(jp('最良の1本',22,GOLD),tex(
            rf'd=2:{bm.LOG_ML[2]:.2f}\quad d=7:{bm.LOG_ML[7]:.2f}',27,GOLD)
            ).arrange(RIGHT,buff=.3).move_to([0,-2.02,0])
        averaged=VGroup(jp('モデル全体',22,GREEN),tex(
            rf'd=2:{bm.LOG_EVIDENCE[2]:.2f}\quad d=7:{bm.LOG_EVIDENCE[7]:.2f}',27,GREEN)
            ).arrange(RIGHT,buff=.3).move_to([0,-2.48,0])
        returned=VGroup(ax,dots,fitted,degree_label,best,averaged)
        next_title=jp('次の問い：事前とノイズの強さは？  3.5',25,GOLD).move_to([0,2.25,0])
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[('return to the same twelve points',a*.48,lambda:FadeIn(returned)),
                          ('compare the same degrees',a*.52,lambda:degree.animate.set_value(2)),
                          ('question for section 3.5',b,lambda:FadeIn(next_title))])

    def begin(self, index):
        self.clear()
        self.story = SCENES[index]
        self.beat_index = 0
        self.subtitle = None
        self.next_section(self.story['id'])
        title = jp(self.story['title'], 34).move_to([0, 3.35, 0])
        self.add(title)
        self.add(jp(self.story['reference'], 16, MUTED).move_to([0, 2.83, 0]))
        entry = self.manifest.get(self.story['id'], {})
        audio_valid = valid_entry(self.story, entry)
        self.audio_entry = entry if audio_valid else None
        self.durations = entry['beat_durations'] if audio_valid else [estimated_duration(b) for b in self.story['beats']]
        # Frame quantization is applied to cumulative boundaries, so audio does not drift.
        fps = config.frame_rate
        self.boundaries = np.ceil(np.cumsum(self.durations) * fps - 1e-6) / fps
        self.scene_start = float(self.time)
        if audio_valid:
            self.add_sound(str(OUTPUT_DIR / f"{self.story['id']}.wav"))
        self.timeline.append({'id': self.story['id'], 'title': self.story['title'],
                              'start': self.scene_start, 'reference': self.story['reference'],
                              'audio': audio_valid, 'beats': []})

    def beat_cues(self):
        offset = sum(self.durations[:self.beat_index])
        if self.audio_entry:
            return [dict(id=c['id'], display=c['display'], start=c['start'] - offset, end=c['end'] - offset)
                    for c in self.audio_entry['subtitle_cues'] if c['beat_index'] == self.beat_index]
        segments = self.story['beats'][self.beat_index]['segments']
        lengths = np.array([len(s['speech']) for s in segments], dtype=float)
        boundaries = np.r_[0, np.cumsum(lengths / lengths.sum() * self.durations[self.beat_index])]
        return [dict(id=s['id'], display=s['display'], start=float(a), end=float(b))
                for s, a, b in zip(segments, boundaries, boundaries[1:])]

    def sentence_duration(self, index):
        cue = self.beat_cues()[index]
        return cue['end'] - cue['start']

    def beat(self, *animations, moving=True, start_sentence=0, end_sentence=None, actions=None, phases=None):
        # Captions use display notation and the PCM duration of the paired speech.
        # The visual action shares this clock; no minimum-duration silent padding.
        item = self.story['beats'][self.beat_index]
        if self.subtitle is not None:
            self.remove(self.subtitle)
        cues = self.beat_cues()
        captions = VGroup()
        for cue in cues:
            caption = caption_mobject(cue['display']).set_opacity(0)
            captions.add(caption)
        self.subtitle = captions
        self.add(captions)
        start = float(self.time)
        fps = config.frame_rate
        frames = round((self.scene_start + self.boundaries[self.beat_index] - start) * fps)
        duration = frames / fps
        action_start = cues[start_sentence]['start']
        action_end = cues[end_sentence - 1]['end'] if end_sentence is not None else cues[-1]['end']
        def caption_at(m, alpha):
            clock = alpha * duration
            index = max(i for i, c in enumerate(cues) if c['start'] <= clock + 1e-8)
            for i, caption in enumerate(m):
                caption.set_opacity(1 if i == index else 0)
        caption_at(captions, 0)
        record = {'start': start, 'end': start + duration, 'display': ''.join(s['display'] for s in item['segments']),
                  'action_start': start + action_start, 'action_end': start + action_end,
                  'cues': [dict(c, start=start+c['start'], end=start+c['end']) for c in cues]}
        if actions:
            record['actions'] = [dict(a, start=start+a['start'], end=start+a['end']) for a in actions]
        self.timeline[-1]['beats'].append(record)
        if phases:
            # Build later animations only after the previous phase has finished.
            # Otherwise Manim can suspend a future target's updater from frame 0,
            # or add a not-yet-stamped RMS marker to the scene prematurely.
            elapsed_frames = 0
            elapsed_seconds = 0.
            record['actions'] = []
            for name, seconds, factory in [*phases, ('breath', duration - sum(p[1] for p in phases), lambda: Wait())]:
                elapsed_seconds += seconds
                end_frame = min(frames, round(elapsed_seconds * fps))
                phase_frames = end_frame - elapsed_frames
                if phase_frames <= 0:
                    continue
                self.update_mobjects(0)
                animation = factory()
                phase_start = float(self.time)
                offset = elapsed_frames / fps
                phase_duration = phase_frames / fps
                self.play(animation, UpdateFromAlphaFunc(captions,
                          lambda m, alpha, offset=offset, d=phase_duration: caption_at(m, (offset + alpha*d) / duration),
                          rate_func=linear), run_time=(phase_frames-1e-5)/fps, rate_func=linear)
                record['actions'].append({'name': name, 'start': phase_start, 'end': float(self.time)})
                elapsed_frames = end_frame
            self.beat_index += 1
            return
        visual = []
        if animations:
            if action_start > 0:
                visual.append(Wait(action_start))
            visual.append(AnimationGroup(*animations, run_time=action_end-action_start))
            visual.append(Wait(max(.001, duration-action_end)))
        else:
            visual.append(Wait(duration))
        self.play(Succession(*visual), UpdateFromAlphaFunc(captions, caption_at, rate_func=linear),
                  run_time=(frames - 1e-5) / fps, rate_func=linear)
        self.beat_index += 1
