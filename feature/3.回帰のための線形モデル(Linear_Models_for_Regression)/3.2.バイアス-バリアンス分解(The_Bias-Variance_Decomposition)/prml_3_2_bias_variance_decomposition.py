"""PRML 3.2: resampling, geometric decomposition, and a live ridge experiment."""
from video_support import *
from bias_variance_model import *


def curve(ax, values, color=MODEL_RED, width=2.5, opacity=1):
    origin = ax.c2p(0, 0)
    points = origin + GRID[:, None]*(ax.c2p(1, 0)-origin) + np.asarray(values)[:, None]*(ax.c2p(0,1)-origin)
    return polyline(points, color, width, opacity)


def dots(ax, index):
    return VGroup(*[Dot(ax.c2p(x,t), radius=.043, color=BLUE_DATA) for x,t in zip(X[index],T[index])])


class PRML32BiasVarianceDecomposition(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.timeline = []
        self.manifest = {e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        for i, method in enumerate([self.question,self.noise,self.ensemble,self.geometry,self.algebra,self.regularization,self.integrated,self.limits]):
            self.begin(i)
            method()
            assert self.beat_index == len(self.story['beats'])
            self.timeline[-1]['end'] = float(self.time)
        (Path(config.media_dir)/'prml32_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def axes(self, width=10, height=3.5, center=(0,.25,0)):
        ax=Axes(x_range=[0,1,.25],y_range=[-1.75,1.75,1],x_length=width,y_length=height,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.3,'include_ticks':False}).move_to(center)
        labels=VGroup()
        for x in [0,.5,1]: labels.add(tex(str(x),19,MUTED).move_to(ax.c2p(x,-1.75)+DOWN*.19))
        for y in [-1,0,1]: labels.add(tex(str(y),19,MUTED).next_to(ax.c2p(0,y),LEFT,buff=.15))
        labels.add(tex('x',24).next_to(ax.c2p(1,0),RIGHT,buff=.18))
        self.add(ax,labels)
        return ax

    def formula(self, *parts, pos=(0,-2.48,0), size=31):
        f=MathTex(*parts,font_size=size).move_to(pos)
        assert f.width < 12.9
        self.add(f)
        return f

    def flash(self, mob, color=MEAN):
        return ShowPassingFlash(mob.copy().clear_updaters().set_stroke(color,4),time_width=.25)

    def slider(self,tr):
        rail=NumberLine(x_range=[-3,3,1],length=7,include_ticks=True,color=MUTED).move_to([0,-2.35,0])
        g=VGroup(rail,tex(r'\ln\lambda',26).next_to(rail,LEFT,buff=.35))
        for v in [-3,0,3]: g.add(tex(str(v),19,MUTED).next_to(rail.n2p(v),DOWN,buff=.16))
        knob=Dot(color=MEAN,radius=.09).add_updater(lambda m:m.move_to(rail.n2p(tr.get_value())))
        g.add(knob,readout('=',tr.get_value,[5,-2.35,0],MEAN,1,25))
        self.add(g)
        return g

    def live_cloud(self,ax,tr,count=20):
        cloud=always_redraw(lambda: VGroup(*[curve(ax,v,MODEL_RED,1.25,.30) for v in experiment(float(tr.get_value()))['curves'][:count]]))
        mean=always_redraw(lambda:curve(ax,experiment(float(tr.get_value()))['mean'],MEAN,4))
        truth_line=curve(ax,truth(GRID),TRUE_GREEN,3)
        self.add(cloud,truth_line,mean)
        return cloud,mean,truth_line

    def question(self):
        ax=self.axes()
        tr=ValueTracker(-3)
        sample=dots(ax,0)
        line=curve(ax,experiment(-3)['curves'][0])
        self.add(sample)
        self.beat(Create(line))
        self.beat(Transform(sample,dots(ax,1)),Transform(line,curve(ax,experiment(-3)['curves'][1])))
        self.beat(Transform(sample,dots(ax,2)),Transform(line,curve(ax,experiment(-3)['curves'][2])))
        self.remove(line)
        line=always_redraw(lambda:curve(ax,experiment(float(tr.get_value()))['curves'][2]))
        self.add(line)
        self.beat(tr.animate.set_value(3))
        true=curve(ax,truth(GRID),TRUE_GREEN,4)
        self.beat(Create(true),self.flash(line))
        self.formula(r'\text{bias}^2', '+',r'\text{variance}',size=38)[0].set_color(BIAS)
        self.beat(Transform(sample,dots(ax,3)),FadeOut(line),Create(curve(ax,experiment(3)['curves'][3])))

    def noise(self):
        ax=self.axes(width=7,height=3.25,center=(-2,.35,0))
        x0=.25; h0=float(truth(x0)); y=ValueTracker(1.6)
        guide=DashedLine(ax.c2p(x0,-1.6),ax.c2p(x0,1.65),color=MUTED)
        self.beat(Create(guide))
        eps=np.random.default_rng(32).normal(0,SIGMA,12); eps-=eps.mean()
        obs=h0+eps
        points=VGroup(*[Dot(ax.c2p(x0,t),radius=.045,color=NOISE) for t in obs])
        self.beat(LaggedStart(*[FadeIn(d) for d in points],lag_ratio=.12))
        mark=always_redraw(lambda:Dot(ax.c2p(x0,y.get_value()),color=MODEL_RED,radius=.08))
        residual=always_redraw(lambda:VGroup(*[Line(ax.c2p(x0-.013*i,y.get_value()),ax.c2p(x0-.013*i,t),color=NOISE,stroke_width=1) for i,t in enumerate(obs)]))
        def squares():
            return VGroup(*[Square(side_length=max(.009,abs(y.get_value()-t)*.78),color=NOISE,fill_opacity=.3,stroke_width=1)
                .move_to([2.3+(i%4)*.9,1.25-(i//4)*.9,0]) for i,t in enumerate(obs)])
        tiles=always_redraw(squares)
        label=jp('距離の二乗 → 平均',22,NOISE).move_to([3.65,2.05,0])
        metric=readout(r'\frac1{12}\sum_i(y-t_i)^2=',lambda:float(np.mean((y.get_value()-obs)**2)),[3.55,-1.6,0],NOISE,3,24)
        self.add(mark,residual,tiles,label,metric)
        self.beat(y.animate.set_value(.4))
        self.beat(y.animate.set_value(h0))
        true=curve(ax,truth(GRID),TRUE_GREEN,3)
        f=self.formula(r'h(x)=\mathbb E[t\mid x]=\int t\,p(t\mid x)\,dt',size=32)
        self.beat(Create(true),Indicate(f,scale_factor=1.02))
        self.beat(LaggedStart(*[Indicate(d,color=NOISE,scale_factor=1.5) for d in points],lag_ratio=.15))
        self.remove(f,tiles,label,metric)
        f=self.formula(r'\mathbb E[L]=',r'\int (y-h)^2p(x)\,dx','+',r'\iint(h-t)^2p(x,t)\,dx\,dt',size=28)
        f[1].set_color(MODEL_RED);f[3].set_color(NOISE)
        self.beat(y.animate.set_value(.65),Indicate(f[1],scale_factor=1.02))
        self.beat(y.animate.set_value(h0),Indicate(f[3],scale_factor=1.02))

    def ensemble(self):
        ax=self.axes()
        pred=experiment(BEST)['curves']
        sample=dots(ax,0); line=curve(ax,pred[0]); true=curve(ax,truth(GRID),TRUE_GREEN)
        self.add(true,sample)
        badge=self.formula('N=25',r'\qquad L=100')
        self.beat(Create(line))
        self.beat(Transform(sample,dots(ax,1)),Transform(line,curve(ax,pred[1])),start_sentence=1)
        cloud=VGroup(*[curve(ax,v,MODEL_RED,1.25,.28) for v in pred[:20]])
        self.beat(FadeOut(sample),FadeOut(line),LaggedStart(*[Create(c) for c in cloud],lag_ratio=.06))
        bases=VGroup(*[curve(ax,design(GRID)[:,j],VAR,1.2,.55) for j in range(1,25)])
        self.beat(FadeIn(bases),cloud.animate.set_opacity(.12))
        self.remove(bases)
        mean=curve(ax,pred.mean(axis=0),MEAN,4)
        self.beat(Create(mean),cloud.animate.set_opacity(.28))
        self.remove(badge)
        formula=self.formula(r'\bar y(x)=\frac1L\sum_{l=1}^{L}y(x;\mathcal D^{(l)})',size=34)
        self.beat(Indicate(formula,scale_factor=1.02),self.flash(mean))
        theoretical=tex(r'm(x)=\mathbb E_{\mathcal D}[y(x;\mathcal D)]',31,MEAN).move_to(formula)
        self.beat(ReplacementTransform(formula,theoretical),LaggedStart(*[self.flash(c) for c in cloud[:5]],lag_ratio=.15))
        self.beat(LaggedStart(*[self.flash(c) for c in cloud[5:10]],lag_ratio=.15),self.flash(mean))

    def geometry(self):
        ax=self.axes()
        pred=experiment(.7)['curves']; idx=50
        values=pred[:,idx]; m=values.mean(); h=truth(GRID[idx])
        cloud=VGroup(*[curve(ax,v,MODEL_RED,1,.25) for v in pred[:20]])
        true=curve(ax,truth(GRID),TRUE_GREEN);mean=curve(ax,pred.mean(axis=0),MEAN)
        points=VGroup(*[Dot(ax.c2p(GRID[idx],v),radius=.026,color=MODEL_RED) for v in values])
        rail=NumberLine(x_range=[.0,1.8,.2],length=10,include_ticks=True,color=MUTED).move_to([0,1.45,0])
        target=VGroup(*[Dot(rail.n2p(v)+UP*((i%7)-3)*.055,radius=.029,color=MODEL_RED) for i,v in enumerate(values)])
        self.add(cloud,true,mean,points)
        self.beat(Transform(points,target),FadeOut(cloud),FadeOut(true),FadeOut(mean),FadeOut(ax),FadeIn(rail))
        # Remove labels of the old graph after the coordinate change.
        for mob in list(self.mobjects):
            if mob not in [points,rail,self.subtitle] and mob.get_center()[1]<2.5: self.remove(mob)
        shift=ValueTracker(0);spread=ValueTracker(1)
        meanpos=lambda:m+shift.get_value()
        v=lambda:meanpos()+spread.get_value()*(values-m)
        for i,d in enumerate(points): d.add_updater(lambda d,i=i:d.move_to(rail.n2p(v()[i])+UP*((i%7)-3)*.055))
        hm=Line(rail.n2p(h)+DOWN*.35,rail.n2p(h)+UP*.35,color=TRUE_GREEN,stroke_width=4)
        mm=always_redraw(lambda:Line(rail.n2p(meanpos())+DOWN*.35,rail.n2p(meanpos())+UP*.35,color=MEAN,stroke_width=4))
        self.add(hm,mm,tex('h',28,TRUE_GREEN).next_to(hm,UP,buff=.12))
        ml=tex('m',28,MEAN).add_updater(lambda a:a.next_to(mm,DOWN,buff=.12));self.add(ml)
        self.beat(Indicate(points,scale_factor=1.015))
        bias_square=always_redraw(lambda:Square(side_length=max(.008,3*abs(meanpos()-h)),color=BIAS,fill_opacity=.25).move_to([-3.5,-.7,0]))
        bias_label=jp('平均のずれ²',24,BIAS).move_to([-3.5,-1.9,0])
        self.add(bias_label)
        self.beat(FadeIn(bias_square))
        variance_label=jp('各予測の揺れ² → 平均',24,VAR).move_to([2.2,-1.9,0]);self.add(variance_label)
        selected=ValueTracker(0)
        def selected_square():
            k=min(99,int(selected.get_value()));d=spread.get_value()*(values[k]-m)
            return Square(side_length=max(.008,3*abs(d)),color=VAR,fill_opacity=.25).move_to([2.2,-.7,0])
        sq=always_redraw(selected_square)
        link=always_redraw(lambda:Line(rail.n2p(meanpos())+DOWN*.45,rail.n2p(v()[min(99,int(selected.get_value()))])+DOWN*.45,color=VAR,stroke_width=3))
        self.add(link,sq)
        self.beat(selected.animate.set_value(10))
        self.add(readout(r'B_x^2=',lambda:(meanpos()-h)**2,[-3.5,-2.5,0],BIAS),readout(r'V_x=',lambda:float(np.mean((v()-meanpos())**2)),[2.2,-2.5,0],VAR))
        self.beat(selected.animate.set_value(30))
        note=jp('模式実験：移動と拡大を別々に操作',21,MUTED).move_to([0,2.35,0]);self.add(note)
        self.beat(shift.animate.set_value(-.28))
        self.beat(spread.animate.set_value(1.65))
        self.beat(selected.animate.set_value(80))

    def algebra(self):
        note=tex(r'x\ \mathrm{fixed},\quad y=y(x;\mathcal D),\quad m=\mathbb E_{\mathcal D}[y]',29,MEAN).move_to([0,2.05,0])
        self.add(note)
        f=tex('(y-h)^2',43).move_to([0,.8,0])
        self.beat(FadeIn(f),Indicate(note,scale_factor=1.02))
        split=MathTex(r'(y-h)^2=',r'\big[(y-m)', '+',r'(m-h)\big]^2',font_size=40).move_to(f)
        split[1].set_color(VAR);split[3].set_color(BIAS)
        self.beat(ReplacementTransform(f,split))
        expansion=MathTex(r'(y-m)^2','+',r'(m-h)^2','+',r'2(y-m)(m-h)',font_size=36).move_to([0,-.15,0])
        expansion[0].set_color(VAR);expansion[2].set_color(BIAS);expansion[4].set_color(MEAN)
        self.beat(FadeIn(expansion),Indicate(expansion[4],scale_factor=1.04))
        deviations=np.array([-.4,-.2,.1,.5])
        bars=VGroup(*[Line([(-1.5+i)*1.4,-1.5,0],[(-1.5+i)*1.4,-1.5+d*1.3,0],color=VAR,stroke_width=9) for i,d in enumerate(deviations)])
        self.beat(LaggedStart(*[Create(b) for b in bars],lag_ratio=.15))
        zero=self.formula(r'\mathbb E_{\mathcal D}[y-m]=\mathbb E_{\mathcal D}[y]-m=0',size=32)
        self.beat(Transform(bars,VGroup(*[Dot([(-1.5+i)*1.4,-1.5,0],radius=.035,color=VAR) for i in range(4)])),Indicate(zero,scale_factor=1.02))
        cross=tex(r'2(m-h)\,\mathbb E_{\mathcal D}[y-m]=0',32,MEAN).move_to(zero)
        self.beat(ReplacementTransform(zero,cross),expansion[4].animate.set_opacity(.22))
        self.remove(split,expansion,bars,cross)
        final=MathTex(r'\mathbb E_{\mathcal D}[(y-h)^2]=',r'(m-h)^2','+',r'\mathbb E_{\mathcal D}[(y-m)^2]',font_size=35).move_to([0,.6,0])
        final[1].set_color(BIAS);final[3].set_color(VAR)
        labels=VGroup(jp('バイアス二乗',28,BIAS).move_to([-2,-.7,0]),jp('バリアンス',28,VAR).move_to([3,-.7,0]))
        self.beat(FadeIn(final),FadeIn(labels))
        self.formula(r'\frac1L\sum_l(y_l-h)^2=(\bar y-h)^2+\frac1L\sum_l(y_l-\bar y)^2',size=30)
        self.beat(Indicate(final[1],scale_factor=1.04),Indicate(final[3],scale_factor=1.04))

    def regularization(self):
        ax=self.axes(height=2.9,center=(0,.5,0))
        tr=ValueTracker(BEST)
        cloud,mean,true=self.live_cloud(ax,tr)
        f=self.formula(r'\frac12\sum_n(y(x_n,\mathbf w)-t_n)^2','+',r'\frac\lambda2\|\mathbf w\|^2',size=32)
        f[2].set_color(VAR)
        self.beat(Indicate(f[0],scale_factor=1.03),Indicate(f[2],scale_factor=1.03))
        self.remove(f);self.slider(tr)
        bias=readout('B^2=',lambda:experiment(float(tr.get_value()))['bias2'],[-2.6,-1.4,0],BIAS)
        variance=readout('V=',lambda:experiment(float(tr.get_value()))['variance'],[2.6,-1.4,0],VAR)
        self.add(bias,variance)
        self.beat(self.flash(mean))
        self.beat(tr.animate.set_value(3))
        self.beat(self.flash(mean,BIAS),self.flash(true,TRUE_GREEN))
        self.beat(tr.animate.set_value(-3))
        self.beat(LaggedStart(*[self.flash(c,VAR) for c in cloud[:6]],lag_ratio=.15))
        self.beat(tr.animate.set_value(BEST))
        warning=tex(r'\min_{\lambda\geq0,\mathbf w}\left[E_D(\mathbf w)+\frac\lambda2\|\mathbf w\|^2\right]\ \Rightarrow\ \lambda=0\ \mathrm{is\ optimal}',28).move_to([0,2.18,0])
        self.beat(FadeIn(warning),Indicate(warning,scale_factor=1.01))

    def integrated(self):
        x=ValueTracker(.05); ax=self.axes(height=2.4,center=(0,.8,0))
        d=experiment(BEST);mean=curve(ax,d['mean'],MEAN);true=curve(ax,truth(GRID),TRUE_GREEN)
        scan=always_redraw(lambda:Line(ax.c2p(x.get_value(),-1.6),ax.c2p(x.get_value(),1.6),color=VAR,stroke_width=2))
        self.add(mean,true,scan)
        f=self.formula(r'B^2=\int(m-h)^2p(x)\,dx',r'\quad V=\int\mathbb E_{\mathcal D}[(y-m)^2]p(x)\,dx',size=28)
        f[0].set_color(BIAS);f[1].set_color(VAR)
        self.beat(x.animate.set_value(.95))
        self.remove(f)
        loss=self.formula(r'\mathbb E_{\mathcal D}[\mathbb E[L]]=',r'B^2','+','V','+',r'\underbrace{\iint(h-t)^2p(x,t)\,dx\,dt}_{\mathrm{noise}}',size=28)
        loss[1].set_color(BIAS);loss[3].set_color(VAR);loss[5].set_color(NOISE)
        self.beat(x.animate.set_value(.15),Indicate(loss,scale_factor=1.01))
        self.remove(loss)
        approx=self.formula(r'\widehat B^2=\frac1K\sum_{k=1}^K[\bar y(x_k)-h(x_k)]^2',size=31);approx.set_color(BIAS)
        sample=VGroup(*[Dot(ax.c2p(u,0),color=BIAS,radius=.026) for u in EVAL_X[:60]])
        self.beat(LaggedStart(*[FadeIn(t) for t in sample],lag_ratio=.03))
        self.remove(approx)
        approx=self.formula(r'\widehat V=\frac1K\sum_{k=1}^K\frac1L\sum_{l=1}^L[y_l(x_k)-\bar y(x_k)]^2',size=31);approx.set_color(VAR)
        self.beat(x.animate.set_value(.9),Indicate(approx,scale_factor=1.01))
        # Reuse the main stage for the quantitative trade-off.
        for mob in list(self.mobjects):
            if mob is not self.subtitle and mob.get_center()[1]<2.5: self.remove(mob)
        chart=Axes(x_range=[-3,3,1],y_range=[0,.26,.05],x_length=9.6,y_length=3.7,tips=False,
                   axis_config={'color':MUTED,'stroke_width':1.3}).move_to([0,.15,0])
        self.add(chart,tex(r'\ln\lambda',24).next_to(chart.x_axis,RIGHT,buff=.2))
        for u in [-3,0,3]:self.add(tex(str(u),19,MUTED).next_to(chart.c2p(u,0),DOWN,buff=.18))
        for v in [.05,.10,.15,.20,.25]:self.add(tex(f'{v:.2f}',18,MUTED).next_to(chart.c2p(-3,v),LEFT,buff=.15))
        paths=[polyline([chart.c2p(l,y) for l,y in zip(LOG_LAMBDAS,values)],color,2.7) for values,color in
               [(METRICS[:,0],BIAS),(METRICS[:,1],VAR),(METRICS[:,0]+METRICS[:,1],MEAN),(METRICS[:,2],WHITE)]]
        # Replace the generic legend with metric labels, well above the axis.
        for mob in list(self.mobjects):
            if 2.6<mob.get_center()[1]<3.0:self.remove(mob)
        legend=VGroup(*[VGroup(Line(ORIGIN,RIGHT*.35,color=c),jp(t,18,c)).arrange(RIGHT,buff=.1) for t,c in
                       [('バイアス²',BIAS),('分散',VAR),('合計',MEAN),('期待損失',WHITE),('テスト',NOISE)]]).arrange(RIGHT,buff=.3).move_to([0,2.75,0])
        self.add(legend)
        self.beat(Create(paths[0]),Create(paths[1]))
        best_y=float(np.min(METRICS[:,0]+METRICS[:,1]));best=Dot(chart.c2p(BEST,best_y),color=MEAN,radius=.08)
        best_note=tex(rf'\ln\lambda={BEST:.1f}',28,MEAN).move_to([0,-2.38,0])
        self.beat(Create(paths[2]),FadeIn(best),FadeIn(best_note))
        testing=VGroup(*[Dot(chart.c2p(l,y),radius=.028,color=NOISE) for l,y in zip(LOG_LAMBDAS[::3],METRICS[::3,3])])
        self.beat(Create(paths[3]),LaggedStart(*[FadeIn(t) for t in testing],lag_ratio=.06))
        tr=ValueTracker(-3)
        gap=always_redraw(lambda:Line(chart.c2p(tr.get_value(),np.interp(tr.get_value(),LOG_LAMBDAS,METRICS[:,0]+METRICS[:,1])),
              chart.c2p(tr.get_value(),np.interp(tr.get_value(),LOG_LAMBDAS,METRICS[:,2])),color=NOISE,stroke_width=5))
        self.add(gap)
        self.remove(best_note);self.formula(r'\mathrm{noise}=\sigma^2=0.25^2=0.0625',size=31)
        self.beat(tr.animate.set_value(3))

    def limits(self):
        ax=self.axes()
        d=experiment(BEST)
        cloud=VGroup(*[curve(ax,v,MODEL_RED,1.3,.28) for v in d['curves'][:20]])
        sample=dots(ax,0);one=curve(ax,d['curves'][0]);true=curve(ax,truth(GRID),TRUE_GREEN)
        self.add(cloud,true)
        self.beat(FadeOut(cloud),FadeIn(sample),Create(one))
        unknown=jp('本当の平均は、通常は未知',28,TRUE_GREEN).move_to([0,-2.4,0]);self.add(unknown)
        self.beat(true.animate.set_opacity(.1),Indicate(unknown,scale_factor=1.03))
        more=VGroup(*[Dot(ax.c2p(u,t),radius=.025,color=BLUE_DATA) for u,t in zip(X[1:8].ravel(),T[1:8].ravel())])
        self.beat(LaggedStart(*[FadeIn(t) for t in more],lag_ratio=.01))
        self.remove(more,unknown)
        f=self.formula(r'p(\mathbf w\mid\mathcal D)',size=38);f.set_color(VAR)
        self.beat(FadeOut(one),FadeIn(cloud),Indicate(f,scale_factor=1.03))
        # Schematic candidate curves illustrate posterior weighting, not posterior samples.
        note=jp('次節の概念図：係数の候補に重みを付ける',22,MUTED).move_to([0,2.16,0]);self.add(note)
        self.remove(f)
        f=self.formula(r'\int y(x,\mathbf w)\,p(\mathbf w\mid\mathcal D)\,d\mathbf w',size=34);f.set_color(MEAN)
        self.beat(LaggedStart(*[self.flash(c,VAR) for c in cloud[:6]],lag_ratio=.12),Indicate(f,scale_factor=1.02))
        self.remove(f)
        final=self.formula(r'\text{expected loss}=',r'\text{bias}^2','+',r'\text{variance}','+',r'\text{noise}',size=36)
        final[1].set_color(BIAS);final[3].set_color(VAR);final[5].set_color(NOISE)
        self.beat(FadeOut(cloud),Create(curve(ax,d['mean'],MEAN,4)),true.animate.set_opacity(1),Indicate(final,scale_factor=1.02))
