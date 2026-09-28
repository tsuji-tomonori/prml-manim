"""PRML 5.1 — linked visual experiments, implemented in Manim Community."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import *
from network_model import (sigmoid, softmax, bump, network, targets, fitted_weights,
                           class_hidden, class_score, decision_boundary, X)

C1=BLUE_CLASS
C2=PURPLE_ACC
C3=GREEN_CLASS
OUT=RED_CLASS
COLOR_TEMPLATE=TexTemplate()
COLOR_TEMPLATE.add_to_preamble(r"\usepackage{xcolor}")


def pulse(mobject, color=YELLOW_ACC, scale_factor=1):
    """Emphasize color while preserving plotted coordinates and edge endpoints."""
    return Indicate(mobject, color=color, scale_factor=1)


def nodespec(label, at, color, getter=None):
    circle=Circle(.31,color=color,stroke_width=2.5).move_to(at)
    if getter:
        circle.add_updater(lambda m:m.set_fill(color,opacity=.10+.30*min(1,abs(getter()))))
    text=tex(label,27,WHITE).move_to(at)
    g=VGroup(circle,text)
    if getter:
        anchor=np.array(at)+DOWN*.57
        value=DecimalNumber(getter(),num_decimal_places=2,font_size=22,color=color).move_to(anchor)
        value.add_updater(lambda m:m.set_value(getter()).move_to(anchor))
        g.add(value)
    return g


def edge(a,b,color=MUTED):
    return Arrow(a[0].get_center(),b[0].get_center(),buff=.35,color=color,stroke_width=2,
                 max_tip_length_to_length_ratio=.06)


class PRML51FeedForwardNetworkFunctions(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.entries={s['id']:s for s in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.ingredients,self.unit,self.combine,self.nonlinear,
                                    self.outputs,self.classification,self.topology,self.approximation,self.symmetry]):
            self.begin(i)
            method()
            assert self.bi==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml51_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def legend(self, entries):
        g=VGroup(*[VGroup(Line(LEFT*.16,RIGHT*.16,color=c),jp(s,19,c)).arrange(RIGHT,buff=.1)
                   for s,c in entries]).arrange(RIGHT,buff=.4).move_to([0,1.85,0])
        self.add(g)
        return g

    def ingredients(self):
        ax=self.plot_axes(x=(-1,1,.5),y=(-1,1.5,.5),height=3.15,center=(-.7,-.2,0))
        c=ValueTracker(.18); v=ValueTracker(.35)
        pts=VGroup(*[Dot(ax.c2p(x,y),color=C1,radius=.05) for x,y in zip(X[::2],bump(X[::2],.43)+.025*np.sin(37*X[::2]))])
        question=jp('材料の形も、変えられる？',29).move_to([0,2.48,0])
        self.add(question)
        self.legend([('観測',C1),('材料１',C1),('材料２',C2),('予測',OUT)])
        self.beat(LaggedStart(*[FadeIn(d) for d in pts],lag_ratio=.04))
        h1=always_redraw(lambda:curve(ax,lambda x:np.tanh(3*(x+c.get_value())),color=C1).set_stroke(width=2))
        h2=always_redraw(lambda:curve(ax,lambda x:np.tanh(3*(x-c.get_value())),color=C2).set_stroke(width=2))
        self.beat(Create(h1),Create(h2))
        total=always_redraw(lambda:curve(ax,lambda x:bump(x,c.get_value(),v.get_value()),color=OUT))
        self.add(total,knob('v=',v,0,1,color=OUT))
        self.beat(v.animate.set_value(.85))
        # Replace only the explicitly owned control, not the plotted data or title.
        for m in list(self.mobjects):
            if isinstance(m,VGroup) and len(m)==3 and any(isinstance(z,NumberLine) for z in m): self.remove(m)
        self.add(knob('c=',c,0,.65,color=C2))
        self.beat(c.animate.set_value(.43),v.animate.set_value(.65))
        self.remove(question)
        eq=self.equation(r'y(x,\mathbf w)=f\!\left(\sum_j',r'w_j',r'\phi_j(x;\theta_j)',r'\right)')
        eq[1].set_color(OUT); eq[2].set_color(C2)
        self.beat(c.animate.set_value(.30))
        self.beat(c.animate.set_value(.43),pulse(h1,color=C1))

    def unit(self):
        ax=self.plot_axes(x=(-1,1,.5),y=(-1.5,1.5,1),height=3.2,center=(-1,-.15,0),labels=('x',r'a,\ z'))
        w=ValueTracker(.7); b=ValueTracker(0); mix=ValueTracker(0); x=ValueTracker(-.7)
        fn=lambda u:(1-mix.get_value())*(w.get_value()*u+b.get_value())+mix.get_value()*np.tanh(w.get_value()*u+b.get_value())
        g=always_redraw(lambda:curve(ax,fn,color=C2))
        eq=self.equation('a=','w','x+','b'); eq[1].set_color(C1); eq[3].set_color(YELLOW_ACC)
        self.add(g,number('w=',w.get_value,[5,.9,0],C1),number('b=',b.get_value,[5,.2,0],YELLOW_ACC))
        self.beat(Create(g))
        control=knob('b=',b,-1,1,color=YELLOW_ACC); self.add(control)
        self.beat(b.animate.set_value(.4))
        eq=self.equation('z=',r'\tanh','(wx+b)'); eq[1].set_color(C2)
        self.beat(mix.animate.set_value(1))
        self.remove(control); control=knob('w=',w,.5,4,color=C1); self.add(control)
        self.beat(w.animate.set_value(3))
        self.remove(control); control=knob('b=',b,-1,1,color=YELLOW_ACC); self.add(control)
        zero=always_redraw(lambda:Dot(ax.c2p(-b.get_value()/w.get_value(),0),color=YELLOW_ACC,radius=.07))
        self.add(zero)
        self.beat(b.animate.set_value(.9))
        self.remove(control,zero)
        dot=always_redraw(lambda:Dot(ax.c2p(x.get_value(),fn(x.get_value())),radius=.08,color=OUT))
        self.add(dot,knob('x=',x,-1,1,color=OUT),number('a=',lambda:w.get_value()*x.get_value()+b.get_value(),[5,-.55,0],YELLOW_ACC),number('z=',lambda:fn(x.get_value()),[5,-1.2,0],C2))
        self.beat(x.animate.set_value(.7))
        eq=self.equation(r'a_j=\sum_{i=1}^{D}w_{ji}^{(1)}x_i+w_{j0}^{(1)},',r'\quad z_j=h(a_j)',size=30)
        eq[0].set_color(YELLOW_ACC); eq[1].set_color(C2)
        self.beat(x.animate.set_value(-.4))

    def combine(self):
        ax=self.plot_axes(x=(-1,1,.5),y=(-1.3,1.3,1),height=3.15,center=(-.6,-.15,0))
        x=ValueTracker(-.8)
        h=lambda u:np.tanh(3*(np.asarray(u)[...,None]+np.array([.4,-.4])))
        g1=curve(ax,lambda u:h(u)[:,0],color=C1)
        g2=curve(ax,lambda u:h(u)[:,1],color=C2)
        self.equation(r'z_1=\tanh(3x+1.2),\quad z_2=\tanh(3x-1.2)')
        self.beat(Create(g1),Create(g2))
        eq=self.equation('y=',r'0.65z_1',r'-0.65z_2'); eq[1].set_color(C1); eq[2].set_color(C2)
        g=curve(ax,lambda u:bump(u),color=OUT)
        self.beat(Transform(g1,curve(ax,lambda u:.65*h(u)[:,0],color=C1)),Transform(g2,curve(ax,lambda u:-.65*h(u)[:,1],color=C2)),Create(g))
        stack=always_redraw(lambda:VGroup(
            Line(ax.c2p(x.get_value(),0),ax.c2p(x.get_value(),.65*h(x.get_value())[0]),color=C1,stroke_width=6),
            Line(ax.c2p(x.get_value(),.65*h(x.get_value())[0]),ax.c2p(x.get_value(),bump(x.get_value())),color=C2,stroke_width=6),
            Dot(ax.c2p(x.get_value(),bump(x.get_value())),color=OUT)))
        self.add(stack)
        self.equation(r'a_k=\sum_{j=1}^{M}w_{kj}^{(2)}z_j+w_{k0}^{(2)},\quad y_k=f(a_k)\qquad(5.4)',size=30)
        self.beat(x.animate.set_value(.8))
        self.remove(*[m for m in self.mobjects if m not in [self.formula,self.subtitle] and m.get_center()[1]<2])
        x.set_value(.25)
        ins=nodespec('x',[-4.4,.1,0],C1,x.get_value)
        hs=[nodespec('z_1',[0,1,0],C1,lambda:h(x.get_value())[0]),nodespec('z_2',[0,-.8,0],C2,lambda:h(x.get_value())[1])]
        mode=ValueTracker(0)
        out=nodespec('y',[4.3,.1,0],OUT,lambda:(1-mode.get_value())*bump(x.get_value())+mode.get_value()*sigmoid(bump(x.get_value())))
        e1=VGroup(*[edge(ins,z) for z in hs]); e2=VGroup(*[edge(z,out) for z in hs])
        net=VGroup(e1,e2,ins,*hs,out)
        self.beat(actions=[lambda:FadeIn(net),lambda:LaggedStart(*[pulse(m,color=YELLOW_ACC) for m in [ins,e1,VGroup(*hs),e2,out]],lag_ratio=.35)])
        bias=nodespec('1',[-3.3,-1.65,0],YELLOW_ACC)
        be=VGroup(*[edge(bias,z,YELLOW_ACC) for z in hs])
        self.equation(r'x_0=1:\quad a_j=\sum_{i=0}^{D}w_{ji}^{(1)}x_i\qquad (5.8)')
        self.beat(FadeIn(bias),Create(be))
        self.remove(bias,be)
        # Use TeX colors without nested substring SVG groups (CE 0.20.1 drops ungrouped glyphs).
        self.equation(r'{\color[HTML]{FF7687}y_k}=\sigma\!\left(\sum_{j=1}^{M}{\color[HTML]{FF7687}w_{kj}^{(2)}}h\!\left(\sum_{i=1}^{D}{\color[HTML]{62B7EE}w_{ji}^{(1)}}x_i+{\color[HTML]{FFE184}w_{j0}^{(1)}}\right)+{\color[HTML]{FFE184}w_{k0}^{(2)}}\right)',size=28,tex_template=COLOR_TEMPLATE)
        compact=tex(r'x_0=z_0=1:\quad y_k=\sigma\!\left(\sum_{j=0}^{M}w_{kj}^{(2)}z_j\right),\quad z_j=h\!\left(\sum_{i=0}^{D}w_{ji}^{(1)}x_i\right)\ (j\geq1)',24).move_to([0,-2.2,0])
        self.add(compact)
        self.beat(actions=[lambda:pulse(e1,color=C1),lambda:AnimationGroup(pulse(e2,color=OUT),mode.animate.set_value(1))])
        self.remove(compact)
        self.add(note('重みの層を数える：第１層 → 第２層'))
        self.beat(actions=[lambda:pulse(VGroup(ins,*hs,out),color=C2),lambda:LaggedStart(pulse(e1,color=C1),pulse(e2,color=OUT),lag_ratio=.5)])

    def nonlinear(self):
        ax=self.plot_axes(x=(-1,1,.5),y=(-1.8,1.8,1),height=3.2,center=(-.7,-.15,0))
        w=ValueTracker(.65); v=ValueTracker(.8); t=ValueTracker(0)
        fn=lambda x:v.get_value()*((1-t.get_value())*(w.get_value()*x+.2)+t.get_value()*np.tanh(w.get_value()*x+.2))-.15
        g=always_redraw(lambda:curve(ax,fn,color=OUT))
        self.equation(r'x\ \longrightarrow\ wx+b\ \longrightarrow\ v(wx+b)+c')
        self.add(g,knob('w=',w,.5,2,color=C1),number('v=',v.get_value,[5,1,0],C2))
        self.beat(Create(g))
        self.beat(actions=[lambda:w.animate.set_value(1.5),lambda:v.animate.set_value(.6)])
        eq=self.equation('y=',r'(vw)x',r'+(vb+c)'); eq[1].set_color(C1); eq[2].set_color(YELLOW_ACC)
        self.beat(w.animate.set_value(.65))
        self.equation(r'y=v\,',r'\tanh(wx+b)',r'+c'); self.formula[1].set_color(C2)
        self.beat(actions=[lambda:t.animate.set_value(1),lambda:w.animate.set_value(2)])
        self.equation(r'y=v\,h(wx+b)+c,\qquad h(a):\ a\leftrightarrow\tanh(a)')
        self.beat(actions=[lambda:t.animate.set_value(0),lambda:t.animate.set_value(1)])
        self.equation(r'y=v\tanh(wx+b)+c')
        q=ValueTracker(-.8)
        tangent=always_redraw(lambda:Line(ax.c2p(q.get_value()-.15,fn(q.get_value())-.15*v.get_value()*w.get_value()*(1-np.tanh(w.get_value()*q.get_value()+.2)**2)),ax.c2p(q.get_value()+.15,fn(q.get_value())+.15*v.get_value()*w.get_value()*(1-np.tanh(w.get_value()*q.get_value()+.2)**2)),color=YELLOW_ACC,stroke_width=4))
        self.add(tangent)
        self.beat(q.animate.set_value(.8))

    def outputs(self):
        ax=self.plot_axes(x=(-3,3,1),y=(-3,3,1),height=3.2,center=(-.8,-.15,0),labels=('a','y'))
        axis_names=self.mobjects[-1]
        t=ValueTracker(0); a=ValueTracker(-2)
        fn=lambda x:(1-t.get_value())*x+t.get_value()*sigmoid(x)
        g=always_redraw(lambda:curve(ax,fn,color=OUT))
        dot=always_redraw(lambda:Dot(ax.c2p(a.get_value(),fn(a.get_value())),color=YELLOW_ACC,radius=.09))
        self.equation('y=a'); self.add(g,dot,number('a=',a.get_value,[5,.7,0],YELLOW_ACC),number('y=',lambda:fn(a.get_value()),[5,0,0],OUT))
        self.beat(a.animate.set_value(2))
        self.equation(r'y=\sigma(a)=\frac{1}{1+\exp(-a)}\qquad(5.5),(5.6)')
        self.beat(t.animate.set_value(1),a.animate.set_value(-2))
        self.remove(ax,axis_names)
        ax=self.plot_axes(x=(-3,3,1),y=(0,1,.5),height=3.2,center=(-.8,-.15,0),labels=('a','y'))
        self.beat(a.animate.set_value(0),pulse(dot,color=OUT))
        # Reuse a single bar coordinate system for independent binary outputs and softmax.
        self.remove(*[m for m in self.mobjects if m not in [self.formula,self.subtitle] and m.get_center()[1]<2])
        scores=[ValueTracker(1.3),ValueTracker(.7),ValueTracker(-.6)]; mode=ValueTracker(0)
        probs=lambda:(1-mode.get_value())*sigmoid([v.get_value() for v in scores])+mode.get_value()*softmax([v.get_value() for v in scores])
        bars=always_redraw(lambda:VGroup(*[Rectangle(width=1.35,height=max(.01,2.5*p),fill_color=c,fill_opacity=.8,stroke_width=0).move_to([-3.2+i*3.2,-1.65+1.25*p,0]) for i,(p,c) in enumerate(zip(probs(),[C1,C2,C3]))]))
        labels=VGroup(*[tex('C_'+str(i+1),25,c).move_to([-3.2+i*3.2,-1.98,0]) for i,c in enumerate([C1,C2,C3])])
        numbers=VGroup(*[number('y_'+str(i+1)+'=',lambda i=i:probs()[i],[-3.2+i*3.2,1.3,0],c) for i,c in enumerate([C1,C2,C3])])
        self.equation(r'y_k=\sigma(a_k)\qquad\sum_k y_k\ \text{need not equal}\ 1')
        self.add(labels,numbers,number(r'\sum_k y_k=',lambda:probs().sum(),[0,-2.55,0],YELLOW_ACC))
        self.beat(FadeIn(bars),scores[0].animate.set_value(2.2))
        self.equation(r'y_k=\frac{e^{a_k}}{\sum_l e^{a_l}},\qquad\sum_k y_k=1\quad(4.62)')
        self.beat(mode.animate.set_value(1))
        self.beat(scores[2].animate.set_value(3.2))

    def classification(self):
        ax=self.plot_axes(x=(-1.6,1.6,1),y=(-1.6,1.6,1),width=5.7,height=3.5,center=(-1.6,-.2,0),labels=('x_1','x_2'))
        w=ValueTracker(2.)
        rng=np.random.default_rng(512); xy=rng.uniform(-1.5,1.5,(64,2)); labels=class_score(xy)>0
        pts=VGroup(*[Dot(ax.c2p(*p),radius=.045,color=C1 if c else ORANGE) for p,c in zip(xy,labels)])
        self.equation(r'(x_1,x_2)\longrightarrow(z_1,z_2)\longrightarrow y')
        self.beat(LaggedStart(*[FadeIn(d) for d in pts],lag_ratio=.01))
        def iso(j):
            xx=np.linspace(-1.6,1.6,161)
            yy=(np.arctanh(.5)-.3-w.get_value()*xx) if j==0 else (np.arctanh(.5)+.4+xx)/2
            mask=abs(yy)<=1.6
            line=VMobject().set_points_as_corners([ax.c2p(a,b) for a,b in zip(xx[mask],yy[mask])]).set_stroke([C2,C3][j],2.2)
            return DashedVMobject(line,num_dashes=25)
        l1=always_redraw(lambda:iso(0)); l2=always_redraw(lambda:iso(1))
        self.add(tex(r'z_1=0.5',27,C2).move_to([4,1.15,0]),tex(r'z_2=0.5',27,C3).move_to([4,.55,0]),tex(r'y=0.5',27,OUT).move_to([4,-.05,0]))
        self.beat(Create(l1))
        self.beat(Create(l2))
        bound=always_redraw(lambda:VMobject().set_points_as_corners([ax.c2p(*p) for p in decision_boundary(w.get_value())]).set_stroke(OUT,4))
        self.equation(r'y=\sigma(2z_1+1.3z_2-0.35)',size=32)
        self.beat(Create(bound))
        self.add(knob(r'w_{11}^{(1)}=',w,.6,2.2,color=C2))
        self.beat(w.animate.set_value(.7))
        self.beat(w.animate.set_value(2.))

    def topology(self):
        x=ValueTracker(.4); skip=ValueTracker(0); dense=ValueTracker(1)
        z1=lambda:np.tanh(1.5*x.get_value()+dense.get_value()*(-.4)+.2)
        z2=lambda:np.tanh(-x.get_value()+.8*(-.4)-.1)
        y=lambda:sigmoid(1.2*z1()-.7*z2()+skip.get_value()*x.get_value())
        ns=[nodespec('x_1',[-4.5,.95,0],C1,x.get_value),nodespec('x_2',[-4.5,-.85,0],C1,lambda:-.4),
            nodespec('z_1',[-.3,.95,0],C2,z1),nodespec('z_2',[-.3,-.85,0],C2,z2),nodespec('y',[4.2,.05,0],OUT,y)]
        links=VGroup(*[edge(ns[i],ns[j]) for i,j in [(0,2),(1,2),(0,3),(1,3),(2,4),(3,4)]])
        self.equation(r'\mathbf x\longrightarrow\mathbf z\longrightarrow y')
        self.add(links,*ns)
        self.beat(LaggedStart(*[pulse(n,color=YELLOW_ACC) for n in ns],lag_ratio=.3))
        se=CurvedArrow(ns[0][0].get_top()+UP*.08,ns[4][0].get_top()+UP*.08,angle=-.45,color=YELLOW_ACC,stroke_width=2)
        self.beat(Create(se),skip.animate.set_value(.8))
        self.beat(FadeOut(links[1]),dense.animate.set_value(0))
        self.equation(r'z_k=h\!\left(\sum_{j\to k}w_{kj}z_j\right)\qquad(5.10)')
        self.beat(pulse(VGroup(links[4],links[5],se),color=YELLOW_ACC))
        self.add(note('閉路なし → 必要な値がそろった順に計算'))
        self.beat(LaggedStart(*[pulse(n,color=YELLOW_ACC) for n in ns],lag_ratio=.4))
        self.beat(actions=[lambda:x.animate.set_value(-.6),lambda:x.animate.set_value(.4)])

    def approximation(self):
        weights=fitted_weights()
        ax=self.plot_axes(x=(-1,1,.5),y=(-1.3,1.3,1),height=3.15,center=(-.6,-.15,0))
        self.legend([('観測',C1),('隠れ出力',C2),('予測',OUT)])
        self.equation(r'y(x)=\sum_{j=1}^{3}v_j\tanh(w_jx+b_j)+c')
        def objects(index):
            w=weights[index]
            dots=VGroup(*[Dot(ax.c2p(x,y),color=C1,radius=.035) for x,y in zip(X,targets(index,X))])
            hs=VGroup(*[curve(ax,lambda x,j=j:np.tanh(w[j]*x+w[j+3]),color=c).set_stroke(width=1.4,opacity=.45) for j,c in enumerate([C1,C2,C3])])
            out=curve(ax,lambda x:network(x,w),color=OUT)
            return VGroup(dots,hs,out)
        group=objects(0)
        name=tex('t=x^2',29,YELLOW_ACC).move_to([5,.8,0])
        self.add(name)
        self.beat(FadeIn(group))
        nxt=objects(1)
        nxt_name=tex(r't=\sin(\pi x)',27,YELLOW_ACC).move_to([5,.8,0])
        self.beat(actions=[lambda:AnimationGroup(FadeOut(group),FadeOut(name)),lambda:AnimationGroup(FadeIn(nxt),FadeIn(nxt_name))])
        group=nxt; name=nxt_name
        nxt=objects(2)
        nxt_name=tex(r't=|x|',29,YELLOW_ACC).move_to([5,.8,0])
        self.beat(actions=[lambda:AnimationGroup(FadeOut(group),FadeOut(name)),lambda:AnimationGroup(FadeIn(nxt),FadeIn(nxt_name))])
        group=nxt; name=nxt_name
        self.remove(group)
        sharp=ValueTracker(2.)
        stepdots=VGroup(*[Dot(ax.c2p(x,y),color=C1,radius=.04) for x,y in zip(X,targets(3,X))])
        g=always_redraw(lambda:curve(ax,lambda x:.5+.5*np.tanh(sharp.get_value()*x),color=OUT))
        self.add(stepdots,g,knob('w=',sharp,2,18,color=C2))
        self.remove(name)
        self.add(tex('t=H(x)',29,YELLOW_ACC).move_to([5,.8,0]))
        self.beat(sharp.animate.set_value(18))
        err=Line(ax.c2p(0,.5),ax.c2p(0,1),color=YELLOW_ACC,stroke_width=6)
        self.add(tex(r'|y(0)-t(0)|=0.5',25,YELLOW_ACC).move_to([4.65,-.1,0]))
        self.beat(pulse(err,color=YELLOW_ACC),pulse(g,color=OUT))
        self.remove(*[m for m in self.mobjects if m not in [self.formula,self.subtitle] and m.get_center()[1]<2])
        self.equation(r'\sup_{x\in[a,b]}|y(x)-f(x)|<\varepsilon')
        self.add(jp('左辺：区間全体の最大誤差',22,MUTED).move_to([0,1.65,0]))
        conditions=VGroup(jp('閉じた有限区間',29,C1),jp('連続な目標関数',29,C2),jp('適切な活性化 ＋ 十分な隠れユニット',29,C3)).arrange(DOWN,buff=.5).move_to([0,.3,0])
        self.beat(LaggedStart(*[FadeIn(c,shift=UP*.15) for c in conditions],lag_ratio=.5))
        self.add(note('よい重みの存在 → データから探す学習へ',color=YELLOW_ACC))
        self.beat(pulse(conditions[2],color=YELLOW_ACC))

    def symmetry(self):
        ax=self.plot_axes(x=(-1,1,.5),y=(-1.4,1.4,1),height=2.4,center=(-.7,.35,0))
        ins=ValueTracker(1); outs=ValueTracker(1)
        h1=lambda x:np.tanh(ins.get_value()*(3*x+1.2))
        contribution=lambda x:.65*outs.get_value()*h1(x)
        y=lambda x:contribution(x)-.65*np.tanh(3*x-1.2)
        one=always_redraw(lambda:curve(ax,contribution,color=C1).set_stroke(width=2))
        two=curve(ax,lambda x:-.65*np.tanh(3*x-1.2),color=C2).set_stroke(width=2)
        total=always_redraw(lambda:curve(ax,y,color=OUT))
        ghost=curve(ax,bump,color=MUTED).set_stroke(width=5,opacity=.35)
        self.equation(r'y=0.65\tanh(3x+1.2)-0.65\tanh(3x-1.2)')
        self.add(ghost,one,two,total,number(r's_{\rm in}=',ins.get_value,[5,.75,0],C1,places=2),number(r's_{\rm out}=',outs.get_value,[5,0,0],C2,places=2))
        self.beat(Create(total))
        self.equation(r'\tanh(-a)=-\tanh(a)')
        self.beat(ins.animate.set_value(-1))
        self.equation(r'(-v)\tanh(-wx-b)=v\tanh(wx+b)')
        self.beat(outs.animate.set_value(-1))
        self.add(note('灰色の元の曲線と、赤い曲線が一致',color=YELLOW_ACC))
        self.beat(pulse(total,color=YELLOW_ACC))
        # Swap complete labeled summands; the actual output is permutation invariant.
        left=VGroup(tex(r'(-0.65)\tanh(-3x-1.2)',26,C1),jp('ユニット１',20,C1)).arrange(DOWN,buff=.15).move_to([-3,-2.1,0])
        right=VGroup(tex(r'(-0.65)\tanh(3x-1.2)',26,C2),jp('ユニット２',20,C2)).arrange(DOWN,buff=.15).move_to([3,-2.1,0])
        for m in list(self.mobjects):
            if isinstance(m,Text) and m.get_center()[1]<-2: self.remove(m)
        self.add(left,right)
        self.beat(actions=[lambda:Succession(
            AnimationGroup(left.animate.shift(UP*.42),right.animate.shift(DOWN*.42)),
            AnimationGroup(left.animate.shift(RIGHT*6),right.animate.shift(LEFT*6)),
            AnimationGroup(left.animate.shift(DOWN*.42),right.animate.shift(UP*.42))),
            lambda:pulse(total,color=YELLOW_ACC,scale_factor=1.02)])
        self.equation(r'2^M M!\qquad M=2:\quad 2^2\cdot2!=8')
        self.beat(pulse(self.formula,color=YELLOW_ACC))
        self.remove(left,right)
        self.equation(r'\mathbf x\ \xrightarrow{\ W^{(1)},\,h\ }\ \mathbf z\ \xrightarrow{\ W^{(2)},\,f\ }\ \mathbf y')
        self.beat(pulse(one,color=C1),pulse(two,color=C2))
