"""PRML 3.2: resampling, geometric decomposition, and a live ridge experiment."""
from video_support import *
from bias_variance_model import *


def pulse(mobject, scale_factor=1.03, color=None):
    """Emphasize size while retaining each term's semantic color."""
    target=mobject.copy().scale(scale_factor)
    if color is not None: target.set_color(color)
    return Transform(mobject,target,rate_func=there_and_back)


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

    def drop(self, *mobjects):
        # Indicating a formula part can promote it to a top-level scene object.
        # Remove the complete family before replacing formulas or plot stages.
        self.remove(*[part for mob in mobjects for part in mob.get_family()])

    def axes(self, width=10, height=3.5, center=(0,.25,0)):
        ax=Axes(x_range=[0,1,.25],y_range=[-1.75,1.75,1],x_length=width,y_length=height,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.3,'include_ticks':False}).move_to(center)
        labels=VGroup()
        for x in [0,.5,1]: labels.add(tex(str(x),19,MUTED).move_to(ax.c2p(x,-1.75)+DOWN*.19))
        for y in [-1,0,1]: labels.add(tex(str(y),19,MUTED).next_to(ax.c2p(0,y),LEFT,buff=.15))
        labels.add(tex('t,\\ y',23).next_to(ax.c2p(0,1.75),UP,buff=.12))
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
        self.add(jp('表示20本・平均100本',19,MUTED).move_to([3.8,2.18,0]))
        return cloud,mean,truth_line

    def aid_card(self, label):
        """Temporarily replace the body; preserve the original scene and its clock."""
        saved = [m for m in self.mobjects if m is not self.subtitle]
        header = [m for m in saved if m.get_center()[1] > 3]
        self.clear()
        self.add(*header, jp(label,23).move_to([-4.85,2.02,0],aligned_edge=LEFT))
        return saved

    def restore_body(self, saved):
        self.clear()
        self.add(*saved)

    def conditional_mean_recap(self):
        saved = self.aid_card('復習: 1.5 条件付き平均')
        # 1.5 regression(): the same N(0, .6^2), blue density, yellow prediction.
        density_ax = Axes(x_range=[-2,2,1],y_range=[0,.8,.4],x_length=3.7,y_length=2,
                         tips=False,axis_config={'color':MUTED,'include_ticks':False}).move_to([-2.5,.1,0])
        loss_ax = Axes(x_range=[-2,2,1],y_range=[0,4.5,1],x_length=3.7,y_length=2,
                      tips=False,axis_config={'color':MUTED,'include_ticks':False}).move_to([2.5,.1,0])
        ts = np.linspace(-2,2,161)
        density = polyline([density_ax.c2p(t,np.exp(-.5*(t/.6)**2)/(.6*np.sqrt(2*np.pi))) for t in ts],BLUE_DATA)
        risk = polyline([loss_ax.c2p(t,t*t+.36) for t in ts],MEAN)
        pred = ValueTracker(1.6)
        marker = always_redraw(lambda:DashedLine(density_ax.c2p(pred.get_value(),0),density_ax.c2p(pred.get_value(),.8),color=MEAN))
        point = always_redraw(lambda:Dot(loss_ax.c2p(pred.get_value(),pred.get_value()**2+.36),color=MEAN,radius=.07))
        mean_dot = Dot(density_ax.c2p(0,0),color=TRUE_GREEN,radius=.075)
        bridge = tex(r'\mathbb E[t\mid x]\ \longrightarrow\ h(x)',30,TRUE_GREEN).move_to([0,-1.65,0])
        self.add(density_ax,loss_ax,density,risk,marker,point,
                 jp('入力を固定した分布',20,BLUE_DATA).move_to([-2.5,1.4,0]),
                 jp('期待二乗損失',20,MEAN).move_to([2.5,1.4,0]),
                 tex('t',23).next_to(density_ax.x_axis,RIGHT,buff=.12),
                 tex('y',23).next_to(loss_ax.x_axis,RIGHT,buff=.12),
                 jp('説明用の例',18,MUTED).move_to([3.9,2.02,0]),
                 tex('0',20,MUTED).next_to(density_ax.c2p(0,0),DOWN,buff=.12),
                 tex('0',20,MUTED).next_to(loss_ax.c2p(0,0),DOWN,buff=.12))
        a,b = [self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('R1.5 recall density',a*.24,lambda:Wait()),
            ('R1.5 minimize squared loss',a*.61,lambda:pred.animate.set_value(0)),
            ('R1.5 mark conditional mean',a*.15,lambda:FadeIn(mean_dot)),
            ('R1.5 map mean to h(x)',b*.60,lambda:FadeIn(bridge)),
            ('R1.5 retain ideal prediction',b*.40,lambda:pulse(mean_dot,scale_factor=1.35)),
        ])
        self.restore_body(saved)

    def two_averages_aid(self):
        saved = self.aid_card('補足: 2段階の平均')
        blue, yellow, green = '#58C4DD', '#FFFF00', '#83C167'
        predictions = np.array([[0,1],[2,2],[4,3]])
        means = predictions.mean(axis=0)
        deviations = (predictions-means)**2
        self.add(jp('説明用の例：3組の訓練集合・等確率の2入力',20,MUTED).move_to([0,1.45,0]))
        xs = [-1.25,2.05]; ys = [.55,0,-.55]
        cells = VGroup(*[tex(str(predictions[i,j]),30,blue).move_to([xs[j],ys[i],0])
                         for i in range(3) for j in range(2)])
        self.add(cells)
        for j in range(2): self.add(tex(rf'x_{j+1}',27,blue).move_to([xs[j],1.,0]))
        for i in range(3): self.add(tex(rf'\mathcal D^{{({i+1})}}',25,blue).move_to([-4.2,ys[i],0]))
        columns = [VGroup(*[cells[2*i+j] for i in range(3)]) for j in range(2)]
        boxes = VGroup(*[SurroundingRectangle(c,color=yellow,buff=.16) for c in columns])
        mean_label = jp('予測の平均',20,green).move_to([-3.8,-1.12,0])
        mean_values = VGroup(*[tex(rf'\bar y={v:g}',27,green).move_to([x,-1.12,0]) for x,v in zip(xs,means)])
        square_cells = VGroup(*[tex(rf'({predictions[i,j]}-2)^2={int(deviations[i,j])}',25,yellow)
                               .move_to([xs[j],ys[i],0]) for i in range(3) for j in range(2)])
        # The yellow areas encode the same squared deviations at a common scale.
        tiles = VGroup(*[Square(side_length=max(.012,.15*np.sqrt(deviations[i,j])),color=yellow,
                                fill_opacity=.3,stroke_width=1).move_to([xs[j]+1.12,ys[i],0])
                         for i in range(3) for j in range(2)])
        variances = VGroup(tex(r'V_1=\frac{4+0+4}{3}=\frac83',25,green).move_to([xs[0],-1.7,0]),
                          tex(r'V_2=\frac{1+0+1}{3}=\frac23',25,green).move_to([xs[1],-1.7,0]))
        variance_label = jp('二乗の平均',20,green).move_to([-3.8,-1.7,0])
        total = tex(r'\widehat V=\frac12\left(\frac83+\frac23\right)=\frac53',30,green).move_to([-2.35,-2.58,0])
        general = VGroup(jp('一般には',19),tex(r'p(x)',25,yellow),jp('で重み付け',19)).arrange(RIGHT,buff=.12).move_to([2.6,-2.58,0])
        a,b,c = [self.sentence_duration(i) for i in range(3)]
        self.beat(phases=[
            ('V07b select columns',a*.40,lambda:Create(boxes)),
            ('V07b mean predictions',a*.60,lambda:AnimationGroup(FadeIn(mean_label),FadeIn(mean_values,shift=UP*.18))),
            ('V07b square deviations',b*.56,lambda:AnimationGroup(
                FadeOut(boxes),Succession(FadeOut(cells),FadeIn(square_cells)),FadeIn(tiles))),
            ('V07b average over datasets',b*.44,lambda:AnimationGroup(FadeIn(variance_label),FadeIn(variances))),
            ('V07b average over inputs',c*.58,lambda:FadeIn(total,shift=UP*.18)),
            ('V07b general input weights',c*.42,lambda:FadeIn(general)),
        ])
        self.restore_body(saved)

    def question(self):
        ax=self.axes()
        x0=STORY_X
        tr=ValueTracker(-3)
        sample=dots(ax,0)
        line=curve(ax,experiment(-3)['curves'][0])
        guide=DashedLine(ax.c2p(x0,-1.65),ax.c2p(x0,1.65),color=MUTED,stroke_width=2)
        value=lambda i,l:float(story_prediction(i,l))
        marker=Dot(ax.c2p(x0,value(0,-3)),radius=.085,color=MEAN)
        readout=self.formula(r'\ln\lambda=-3,\quad y(0.25)=0.830',size=32)
        readout.set_opacity(0)
        self.add(sample)
        self.beat(Create(line),Create(guide),FadeIn(marker),FadeIn(readout))
        next_label=tex(r'\ln\lambda=-3,\quad y(0.25)=1.457',32).move_to(readout)
        self.beat(Transform(sample,dots(ax,2)),Transform(line,curve(ax,experiment(-3)['curves'][2])),
                  marker.animate.move_to(ax.c2p(x0,value(2,-3))),
                  Succession(FadeOut(readout),FadeIn(next_label)))
        readout=next_label
        first=Dot(ax.c2p(x0,value(0,-3)),radius=.06,color=MODEL_RED)
        gap=Line(first.get_center(),marker.get_center(),color=VAR,stroke_width=5)
        self.beat(FadeIn(first),Create(gap))
        self.drop(line)
        line=always_redraw(lambda:curve(ax,experiment(float(tr.get_value()))['curves'][2]))
        moving_dot=always_redraw(lambda:Dot(ax.c2p(x0,value(2,float(tr.get_value()))),radius=.085,color=MEAN))
        self.add(line,moving_dot)
        self.drop(marker,first,gap)
        strong_label=tex(r'\ln\lambda=3,\quad y(0.25)=0.299',32).move_to(readout)
        self.beat(tr.animate.set_value(3),Succession(FadeOut(readout),FadeIn(strong_label)))
        readout=strong_label
        true=curve(ax,truth(GRID),TRUE_GREEN,4)
        strong0=curve(ax,experiment(3)['curves'][0])
        true_dot=Dot(ax.c2p(x0,truth(x0)),radius=.09,color=TRUE_GREEN)
        final_label=tex(r'\ln\lambda=3,\quad y(0.25)=0.402,\quad h(0.25)=1',30).move_to(readout)
        self.beat(Transform(sample,dots(ax,0)),FadeOut(line),FadeOut(moving_dot),
                  Create(strong0),Create(true),FadeIn(true_dot),
                  Succession(FadeOut(readout),FadeIn(final_label)))
        self.beat(self.flash(strong0),pulse(true_dot,scale_factor=1.35))

    def noise(self):
        ax=self.axes(width=7,height=3.25,center=(-2,.35,0))
        x0=.25; h0=float(truth(x0)); y=ValueTracker(1.6)
        guide=DashedLine(ax.c2p(x0,-1.6),ax.c2p(x0,1.65),color=MUTED)
        self.beat(Create(guide))
        eps=np.random.default_rng(32).normal(0,SIGMA,12); eps-=eps.mean()
        obs=h0+eps
        points=VGroup(*[Dot(ax.c2p(x0,t),radius=.045,color=NOISE) for t in obs])
        self.beat(LaggedStart(*[FadeIn(d) for d in points],lag_ratio=.12))
        self.conditional_mean_recap()
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
        self.beat(Create(true),pulse(f,scale_factor=1.02))
        self.beat(LaggedStart(*[pulse(d,color=NOISE,scale_factor=1.5) for d in points],lag_ratio=.15))
        self.drop(f,tiles,label,metric)
        f=self.formula(r'\mathbb E[L]=',r'\int (y-h)^2p(x)\,dx','+',r'\iint(h-t)^2p(x,t)\,dx\,dt',size=28)
        f[1].set_color(MODEL_RED);f[3].set_color(NOISE)
        self.beat(y.animate.set_value(.65),pulse(f[1],scale_factor=1.02))
        self.beat(y.animate.set_value(h0),pulse(f[3],scale_factor=1.02))

    def ensemble(self):
        ax=self.axes()
        pred=experiment(BEST)['curves']
        sample=dots(ax,0); line=curve(ax,pred[0]); true=curve(ax,truth(GRID),TRUE_GREEN)
        self.add(true,sample)
        badge=self.formula('N=25',r'\qquad L=100',r'\qquad\ln\lambda=0.1')
        self.add(jp('表示20本・平均100本',19,MUTED).move_to([3.8,2.18,0]))
        self.beat(Create(line))
        self.beat(Transform(sample,dots(ax,1)),Transform(line,curve(ax,pred[1])),start_sentence=1)
        cloud=VGroup(*[curve(ax,v,MODEL_RED,1.25,.28) for v in pred[:20]])
        self.beat(FadeOut(sample),FadeOut(line),LaggedStart(*[Create(c) for c in cloud],lag_ratio=.06))
        bases=VGroup(*[curve(ax,design(GRID)[:,j],VAR,1.2,.55) for j in range(1,25)])
        self.beat(FadeIn(bases),cloud.animate.set_stroke(opacity=.12))
        self.drop(bases)
        mean=curve(ax,pred.mean(axis=0),MEAN,4)
        self.beat(Create(mean),cloud.animate.set_stroke(opacity=.28))
        self.drop(badge)
        formula=self.formula(r'\bar y(x)=\frac1L\sum_{l=1}^{L}y(x;\mathcal D^{(l)})',size=34)
        self.beat(pulse(formula,scale_factor=1.02),self.flash(mean))
        theoretical=tex(r'm(x)=\mathbb E_{\mathcal D}[y(x;\mathcal D)]',31,MEAN).move_to(formula)
        self.drop(formula);self.add(theoretical)
        self.beat(pulse(theoretical,scale_factor=1.02),LaggedStart(*[self.flash(c) for c in cloud[:5]],lag_ratio=.15))
        self.beat(LaggedStart(*[self.flash(c) for c in cloud[5:10]],lag_ratio=.15),self.flash(mean))

    def geometry(self):
        ax=self.axes()
        pred=experiment(.7)['curves']; idx=50
        values=pred[:,idx]; m=values.mean(); h=truth(GRID[idx])
        cloud=VGroup(*[curve(ax,v,MODEL_RED,1,.25) for v in pred[:20]])
        true=curve(ax,truth(GRID),TRUE_GREEN);mean=curve(ax,pred.mean(axis=0),MEAN)
        points=VGroup(*[Dot(ax.c2p(GRID[idx],v),radius=.026,color=MODEL_RED) for v in values])
        rail=NumberLine(x_range=[.0,1.4,.2],length=10,include_ticks=True,color=MUTED).move_to([0,1.45,0])
        target=VGroup(*[Dot(rail.n2p(v)+UP*((i%7)-3)*.055,radius=.029,color=MODEL_RED) for i,v in enumerate(values)])
        self.add(cloud,true,mean,points)
        self.beat(Transform(points,target),FadeOut(cloud),FadeOut(true),FadeOut(mean),FadeOut(ax),FadeIn(rail))
        # Remove labels of the old graph after the coordinate change.
        for mob in list(self.mobjects):
            if mob not in [points,rail,self.subtitle] and mob.get_center()[1]<2.5: self.drop(mob)
        for tick in [0,.4,.8,1.2]: self.add(tex(str(tick),18,MUTED).next_to(rail.n2p(tick),DOWN,buff=.35))
        shift=ValueTracker(0);spread=ValueTracker(1)
        meanpos=lambda:m+shift.get_value()
        v=lambda:meanpos()+spread.get_value()*(values-m)
        for i,d in enumerate(points): d.add_updater(lambda d,i=i:d.move_to(rail.n2p(v()[i])+UP*((i%7)-3)*.055))
        hm=Line(rail.n2p(h)+DOWN*.35,rail.n2p(h)+UP*.35,color=TRUE_GREEN,stroke_width=4)
        mm=always_redraw(lambda:Line(rail.n2p(meanpos())+DOWN*.35,rail.n2p(meanpos())+UP*.35,color=MEAN,stroke_width=4))
        self.add(hm,mm,tex('h',28,TRUE_GREEN).next_to(hm,UP,buff=.12))
        ml=tex('m',28,MEAN).add_updater(lambda a:a.next_to(mm,UP,buff=.12));self.add(ml)
        self.beat(pulse(points,scale_factor=1.015))
        bias_square=always_redraw(lambda:Square(side_length=max(.008,4*abs(meanpos()-h)),color=BIAS,fill_opacity=.25).move_to([-3.5,-.4,0]))
        bias_label=jp('平均のずれ²',24,BIAS).move_to([-3.5,-1.9,0])
        self.add(bias_label)
        bias_distance=always_redraw(lambda:Line(rail.n2p(h)+UP*.28,rail.n2p(meanpos())+UP*.28,color=BIAS,stroke_width=3))
        self.add(bias_distance)
        self.beat(FadeIn(bias_square))
        variance_label=jp('各予測の揺れ² → 平均',24,VAR).move_to([2.2,-1.9,0]);self.add(variance_label)
        selected=ValueTracker(0)
        def selected_square():
            k=min(99,int(selected.get_value()));d=spread.get_value()*(values[k]-m)
            return Square(side_length=max(.008,4*abs(d)),color=VAR,fill_opacity=.25).move_to([2.2,-.4,0])
        sq=always_redraw(selected_square)
        link=always_redraw(lambda:Line(rail.n2p(meanpos())+DOWN*.65,rail.n2p(v()[min(99,int(selected.get_value()))])+DOWN*.65,color=VAR,stroke_width=3))
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
        self.beat(FadeIn(f),pulse(note,scale_factor=1.02))
        split=MathTex(r'(y-h)^2=',r'\big[(y-m)', '+',r'(m-h)\big]^2',font_size=40).move_to(f)
        split[1].set_color(VAR);split[3].set_color(BIAS)
        self.drop(f);self.add(split)
        self.beat(pulse(split,scale_factor=1.02))
        expansion=MathTex(r'(y-m)^2','+',r'(m-h)^2','+',r'2(y-m)(m-h)',font_size=36).move_to([0,-.15,0])
        expansion[0].set_color(VAR);expansion[2].set_color(BIAS);expansion[4].set_color(MEAN)
        self.add(expansion)
        self.beat(pulse(expansion[4],scale_factor=1.04))
        deviations=np.array([-.4,-.2,.1,.5])
        bars=VGroup(*[Line([(-1.5+i)*1.4,-1.5,0],[(-1.5+i)*1.4,-1.5+d*1.3,0],color=VAR,stroke_width=9) for i,d in enumerate(deviations)])
        self.beat(LaggedStart(*[Create(b) for b in bars],lag_ratio=.15))
        zero=self.formula(r'\mathbb E_{\mathcal D}[y-m]=\mathbb E_{\mathcal D}[y]-m=0',size=32)
        self.beat(Transform(bars,VGroup(*[Dot([(-1.5+i)*1.4,-1.5,0],radius=.035,color=VAR) for i in range(4)])),pulse(zero,scale_factor=1.02))
        cross=tex(r'2(m-h)\,\mathbb E_{\mathcal D}[y-m]=0',32,MEAN).move_to(zero)
        self.drop(zero);self.add(cross)
        self.beat(pulse(cross,scale_factor=1.02),expansion[4].animate.set_opacity(.22))
        self.drop(split,expansion,bars,cross)
        final=MathTex(r'\mathbb E_{\mathcal D}[(y-h)^2]=',r'(m-h)^2','+',r'\mathbb E_{\mathcal D}[(y-m)^2]',font_size=35).move_to([0,.6,0])
        final[1].set_color(BIAS);final[3].set_color(VAR)
        labels=VGroup(jp('バイアス二乗',28,BIAS).move_to([-2,-.7,0]),jp('バリアンス',28,VAR).move_to([3,-.7,0]))
        self.beat(FadeIn(final),FadeIn(labels))
        self.formula(r'\frac1L\sum_l(y_l-h)^2=(\bar y-h)^2+\frac1L\sum_l(y_l-\bar y)^2',size=30)
        self.beat(pulse(final[1],scale_factor=1.04),pulse(final[3],scale_factor=1.04))

    def regularization(self):
        ax=self.axes(height=2.9,center=(0,.5,0))
        tr=ValueTracker(BEST)
        cloud,mean,true=self.live_cloud(ax,tr)
        f=self.formula(r'\frac12\sum_n(y(x_n,\mathbf w)-t_n)^2','+',r'\frac\lambda2\|\mathbf w\|^2',size=32)
        f[2].set_color(VAR)
        self.beat(pulse(f[0],scale_factor=1.03),pulse(f[2],scale_factor=1.03))
        self.drop(f);self.slider(tr)
        bias=readout('B^2=',lambda:experiment(float(tr.get_value()))['bias2'],[-2.6,-1.4,0],BIAS)
        variance=readout('V=',lambda:experiment(float(tr.get_value()))['variance'],[2.6,-1.4,0],VAR)
        self.add(bias,variance)
        self.beat(self.flash(mean))
        self.beat(tr.animate.set_value(3))
        self.beat(self.flash(mean,BIAS),self.flash(true,TRUE_GREEN))
        self.beat(tr.animate.set_value(-3))
        self.beat(LaggedStart(*[self.flash(c,VAR) for c in cloud[:6]],lag_ratio=.15))
        self.beat(tr.animate.set_value(BEST))
        for mob in list(self.mobjects):
            if 2.05<mob.get_center()[1]<2.4: self.drop(mob)
        warning=tex(r'\min_{\lambda\geq0,\mathbf w}\left[E_D(\mathbf w)+\frac\lambda2\|\mathbf w\|^2\right]\ \Rightarrow\ \lambda=0\ \mathrm{is\ optimal}',28).move_to([0,2.18,0])
        self.add(warning)
        self.beat(pulse(warning,scale_factor=1.01))

    def integrated(self):
        x=ValueTracker(.05); ax=self.axes(height=2.4,center=(0,.8,0))
        d=experiment(BEST);mean=curve(ax,d['mean'],MEAN);true=curve(ax,truth(GRID),TRUE_GREEN)
        scan=always_redraw(lambda:Line(ax.c2p(x.get_value(),-1.6),ax.c2p(x.get_value(),1.6),color=VAR,stroke_width=2))
        self.add(mean,true,scan)
        f=self.formula(r'B^2=\int(m-h)^2p(x)\,dx',r'\quad V=\int\mathbb E_{\mathcal D}[(y-m)^2]p(x)\,dx',size=28)
        f[0].set_color(BIAS);f[1].set_color(VAR)
        self.beat(x.animate.set_value(.95))
        self.drop(f)
        loss=self.formula(r'\mathbb E_{\mathcal D}[\mathbb E[L]]=',r'B^2','+','V','+',r'\underbrace{\iint(h-t)^2p(x,t)\,dx\,dt}_{\mathrm{noise}}',size=28)
        loss[1].set_color(BIAS);loss[3].set_color(VAR);loss[5].set_color(NOISE)
        self.beat(x.animate.set_value(.15),pulse(loss,scale_factor=1.01))
        self.drop(loss)
        approx=self.formula(r'\widehat B^2=\frac1K\sum_{k=1}^K[\bar y(x_k)-h(x_k)]^2',size=31);approx.set_color(BIAS)
        sample=VGroup(*[Dot(ax.c2p(u,0),color=BIAS,radius=.026) for u in EVAL_X[:60]])
        self.beat(LaggedStart(*[FadeIn(t) for t in sample],lag_ratio=.03))
        self.drop(approx)
        approx=self.formula(r'\widehat V=\frac1K\sum_{k=1}^K\frac1L\sum_{l=1}^L[y_l(x_k)-\bar y(x_k)]^2',size=31);approx.set_color(VAR)
        self.beat(x.animate.set_value(.9),pulse(approx,scale_factor=1.01))
        self.two_averages_aid()
        # Reuse the main stage for the quantitative trade-off.
        for mob in list(self.mobjects):
            if mob is not self.subtitle and mob.get_center()[1]<2.5: self.drop(mob)
        chart=Axes(x_range=[-3,3,1],y_range=[0,.26,.05],x_length=9.6,y_length=3.7,tips=False,
                   axis_config={'color':MUTED,'stroke_width':1.3}).move_to([0,.15,0])
        chart.y_axis.set_opacity(0)
        self.add(Line(chart.c2p(-3,0),chart.c2p(-3,.26),color=MUTED,stroke_width=1.3),jp('二乗誤差',20,MUTED).move_to([-4.8,2.25,0]))
        self.add(chart,tex(r'\ln\lambda',24).next_to(chart.x_axis,RIGHT,buff=.2))
        for u in [-3,0,3]:self.add(tex(str(u),19,MUTED).next_to(chart.c2p(u,0),DOWN,buff=.18))
        for v in [.05,.10,.15,.20,.25]:self.add(tex(f'{v:.2f}',18,MUTED).next_to(chart.c2p(-3,v),LEFT,buff=.15))
        paths=[polyline([chart.c2p(l,y) for l,y in zip(LOG_LAMBDAS,values)],color,2.7) for values,color in
               [(METRICS[:,0],BIAS),(METRICS[:,1],VAR),(METRICS[:,0]+METRICS[:,1],MEAN),(METRICS[:,2],WHITE)]]
        # Replace the generic legend with metric labels, well above the axis.
        for mob in list(self.mobjects):
            if 2.6<mob.get_center()[1]<3.0:self.drop(mob)
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
        self.drop(best_note);self.formula(r'\mathrm{noise}=\sigma^2=0.25^2=0.0625',size=31)
        self.beat(tr.animate.set_value(3))

    def limits(self):
        ax=self.axes()
        x0=STORY_X
        def prediction_dot(index, log_lambda, color=MEAN):
            return Dot(ax.c2p(x0,story_prediction(index,log_lambda)),radius=.085,color=color)
        guide=DashedLine(ax.c2p(x0,-1.65),ax.c2p(x0,1.65),color=MUTED,stroke_width=2)
        sample=dots(ax,0)
        first_curve=curve(ax,experiment(-3)['curves'][0])
        first_dot=prediction_dot(0,-3)
        label=self.formula(r'\ln\lambda=-3,\quad y_0(0.25)=0.830',size=31)
        label.set_opacity(0)
        self.add(sample)
        self.beat(Create(first_curve),Create(guide),FadeIn(first_dot),FadeIn(label))
        second_curve=curve(ax,experiment(-3)['curves'][2],MODEL_RED,2.3,.65)
        second_dot=prediction_dot(2,-3,VAR)
        weak_label=tex(r'\ln\lambda=-3:\quad 0.830,\ 1.457',31).move_to(label)
        self.beat(Transform(sample,dots(ax,2)),Create(second_curve),FadeIn(second_dot),
                  Succession(FadeOut(label),FadeIn(weak_label)))
        label=weak_label
        strong0=curve(ax,experiment(3)['curves'][0])
        strong2=curve(ax,experiment(3)['curves'][2],MODEL_RED,2.3,.65)
        strong_label=tex(r'\ln\lambda=3:\quad 0.402,\ 0.299\quad (h=1)',30).move_to(label)
        truth_line=curve(ax,truth(GRID),TRUE_GREEN,3.5)
        truth_dot=Dot(ax.c2p(x0,truth(x0)),radius=.08,color=TRUE_GREEN)
        self.beat(Transform(first_curve,strong0),Transform(second_curve,strong2),
                  first_dot.animate.move_to(prediction_dot(0,3).get_center()),
                  second_dot.animate.move_to(prediction_dot(2,3).get_center()),
                  Succession(FadeOut(label),FadeIn(strong_label)),Create(truth_line),FadeIn(truth_dot))
        label=strong_label
        middle0=curve(ax,experiment(BEST)['curves'][0])
        middle2=curve(ax,experiment(BEST)['curves'][2],MODEL_RED,2.3,.65)
        middle_label=tex(r'\ln\lambda=0.1:\quad 0.817,\ 0.999\quad (h=1)',30).move_to(label)
        self.beat(Transform(first_curve,middle0),Transform(second_curve,middle2),
                  first_dot.animate.move_to(prediction_dot(0,BEST).get_center()),
                  second_dot.animate.move_to(prediction_dot(2,BEST).get_center()),
                  Succession(FadeOut(label),FadeIn(middle_label)),pulse(truth_dot,scale_factor=1.35))
        label=middle_label
        self.drop(label)
        scores=self.formula(r'\mathbb E[L]:\quad 0.101\ (\ln\lambda=-3),\quad '
                            r'\mathbf{0.084}\ (0.1),\quad 0.225\ (3)',size=30)
        scores.set_color(MEAN)
        self.beat(FadeOut(second_curve),FadeOut(second_dot),
                  truth_line.animate.set_stroke(opacity=.18),FadeOut(truth_dot),pulse(scores,scale_factor=1.02))
        caveat=jp('実際には訓練データは1組・真の平均は未知',23,MUTED).move_to([0,2.17,0])
        self.drop(scores)
        posterior=self.formula(r'p(\mathbf w\mid\mathcal D)\quad\longrightarrow\quad '
                               r'\int y(x,\mathbf w)p(\mathbf w\mid\mathcal D)\,d\mathbf w',size=31)
        posterior.set_color(VAR)
        self.beat(FadeIn(caveat),FadeIn(posterior),self.flash(first_curve,VAR))
