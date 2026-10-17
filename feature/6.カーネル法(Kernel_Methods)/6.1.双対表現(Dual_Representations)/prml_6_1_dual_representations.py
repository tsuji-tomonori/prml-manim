"""PRML §6.1 — recover one ridge prediction from training-point comparisons."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import NarratedScene, jp, tex
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from dual_model import X,T,LAMBDA,PROBE,K,W,A,K_PROBE,PREDICTION,CONTRIBUTIONS,phi,kernel,primal,alternative_kernel,ALT_K,ALT_A,ALT_PREDICTION

_MANIM_FADE_IN=FadeIn
_MANIM_FADE_OUT=FadeOut
def _quick(t):
    return smooth(min(1., 5*t))
def FadeIn(mobject, **kwargs):
    kwargs.setdefault('rate_func', _quick)
    return _MANIM_FADE_IN(mobject, **kwargs)
def FadeOut(mobject, **kwargs):
    kwargs.setdefault('rate_func', _quick)
    return _MANIM_FADE_OUT(mobject, **kwargs)

BG='#10141F'; BLUE=ManimColor('#58C4DD'); YELLOW=ManimColor('#FFFF00')
GREEN=ManimColor('#83C167'); PURPLE=ManimColor('#9A72AC')
RED=ManimColor('#FC6255'); MUTED=ManimColor('#A8B2C5')
RECAP_DATA=ManimColor('#58B5ED'); RECAP_MODEL=ManimColor('#FF6B77')
RECAP_BASIS=ManimColor('#77D49A'); RECAP_GOLD=ManimColor('#FFE079')

def line(ax, xs, ys, color=GREEN, width=3):
    return VMobject().set_points_as_corners([ax.c2p(float(x),float(y)) for x,y in zip(xs,ys)]).set_stroke(color,width)

def label(s, size=24, color=WHITE, pos=None):
    m=jp(s,size,color)
    if pos is not None:m.move_to(pos)
    return m

def formula(s, size=33, color=WHITE, pos=(0,-2.45,0)):
    m=tex(s,size,color).move_to(pos)
    if m.width>12.2:m.scale_to_fit_width(12.2)
    return m

class PRML61DualRepresentations(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.opening,self.weights,self.kernel,self.gram,self.coefficients,self.recovery,self.outlook]):
            self.begin(i)
            method()
            assert self.beat_index==len(SCENES[i]['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml61_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def plot_axes(self,center=(0,.15,0),width=8.6,height=3.75):
        ax=Axes(x_range=[-1.25,1.25,.5],y_range=[-.25,1.35,.25],x_length=width,y_length=height,tips=False,axis_config=dict(color=MUTED,stroke_width=1.3,include_ticks=False)).move_to(center)
        ticks=VGroup()
        for x in [-1,0,1]:ticks.add(formula(str(x),19,MUTED,ax.c2p(x,-.19)+DOWN*.15))
        ticks.add(formula('x',23,WHITE,ax.c2p(1.25,-.19)+RIGHT*.3))
        ticks.add(formula('t',23,WHITE,ax.c2p(-1.25,1.35)+UP*.18))
        return ax,ticks

    def training_points(self,ax):
        return VGroup(*[Dot(ax.c2p(x,t),radius=.065,color=BLUE) for x,t in zip(X,T)])

    def fit_curve(self,ax,lam=LAMBDA,color=GREEN):
        xs=np.linspace(-1.23,1.23,120)
        w=primal(lam)
        return line(ax,xs,phi(xs)@w,color)

    def topnote(self,s,color=MUTED):
        return label(s,22,color,(0,2.57,0))

    def pair_equation(self,left,right):
        l=formula(left,29,PURPLE,(-3.25,1.15,0))
        r=formula(right,29,GREEN,(3.2,1.15,0))
        return VGroup(l,r)

    def opening(self):
        header=VGroup(label('5章  重み',24,PURPLE,(-2.1,.8,0)),Arrow((-1.2,.8,0),(.6,.8,0),buff=.15,color=YELLOW),label('6章  比較',24,BLUE,(2,.8,0)))
        self.beat(FadeIn(header))
        self.remove(header)
        ax,ticks=self.plot_axes()
        pts=self.training_points(ax)
        marks=VGroup(*[formula(f'x={int(x)}',21,BLUE,ax.c2p(x,t)+UP*.32) for x,t in zip(X,T) if x!=0])
        self.beat(Create(ax),FadeIn(ticks),FadeIn(pts),FadeIn(marks))
        probe=DashedLine(ax.c2p(PROBE,-.1),ax.c2p(PROBE,1.08),color=YELLOW,dash_length=.08)
        guess=Dot(ax.c2p(PROBE,.5),radius=.075,color=PURPLE)
        guess_note=self.topnote('予想 0.5',PURPLE)
        self.beat(Create(probe),FadeIn(guess),FadeIn(guess_note))
        curve=self.fit_curve(ax)
        real=Dot(ax.c2p(PROBE,PREDICTION),radius=.08,color=GREEN)
        real_note=self.topnote('回帰の予測 0.368',GREEN)
        self.beat(Create(curve),FadeIn(real),FadeOut(guess_note),FadeIn(real_note),FadeOut(guess))
        eq=formula(r'y(x)=w^T\phi(x)=w_0+w_1x+w_2x^2')
        self.beat(FadeIn(eq),Indicate(real,color=GREEN,scale_factor=1.3))
        question=self.topnote('重みを直接使わず、同じ予測を出せる？',YELLOW)
        self.beat(FadeOut(real_note),FadeIn(question),Indicate(probe,color=YELLOW,scale_factor=1))
        kept=formula(r'\{(x_n,t_n)\}_{n=1}^{3}',34,BLUE,(0,-2.45,0))
        self.beat(FadeIn(kept),FadeOut(eq),Indicate(pts,color=BLUE,scale_factor=1.15))
        bridge=formula(r'w\;\longleftarrow\;\{(x_n,t_n)\}',32,YELLOW,(0,-2.45,0))
        self.beat(FadeOut(kept),FadeIn(bridge),Indicate(pts,color=YELLOW,scale_factor=1.02))

    def weights(self):
        # The recap is a body card; its thin border belongs to the recap convention.
        recap=VGroup(label('復習: 3.1 正則化付き最小二乗',25,YELLOW,(-2.75,2.15,0)),
                     formula(r'J(w)=\frac12\sum_n (w^T\phi(x_n)-t_n)^2+\frac\lambda2 w^Tw',34,WHITE,(0,.65,0)))
        # Recreate only the 3.1 diagram roles: blue observations, red fit,
        # green basis contribution, and yellow regularization control.
        mini=VGroup(*[Dot((x,-.95+t*.38,0),radius=.075,color=RECAP_DATA) for x,t in [(-2,1),(0,0),(2,1)]])
        mini_curve=VMobject().set_points_as_corners([(-2,-.7,0),(-1,-1.05,0),(0,-1.1,0),(1,-1.05,0),(2,-.7,0)]).set_stroke(RECAP_MODEL,3)
        basis=Line((-2,-1.65,0),(2,-1.65,0),color=RECAP_BASIS,stroke_width=4)
        lam_dot=Dot((-2,-1.65,0),radius=.08,color=RECAP_GOLD)
        border=SurroundingRectangle(VGroup(recap,mini,mini_curve,basis),buff=.3,color=YELLOW,stroke_width=1.2)
        self.beat(FadeIn(recap),Create(border),FadeIn(mini),Create(mini_curve),Create(basis),MoveAlongPath(lam_dot,basis))
        self.remove(recap,border,mini,mini_curve,basis,lam_dot)
        ax,ticks=self.plot_axes(center=(-1.25,.1,0),width=7.7)
        pts=self.training_points(ax);curve=self.fit_curve(ax)
        e=formula(r'J(w)=\frac12\sum_{n=1}^{3}(w^T\phi(x_n)-t_n)^2+\frac{0.5}{2}w^Tw',30)
        self.beat(Create(ax),FadeIn(ticks),FadeIn(pts),Create(curve),FadeIn(e))
        lam=ValueTracker(.5)
        moving=always_redraw(lambda:self.fit_curve(ax,lam.get_value()))
        slide=NumberLine(x_range=[.1,2,.5],length=3,include_numbers=False,include_ticks=False,color=MUTED).move_to((4.5,-.65,0))
        val=always_redraw(lambda:formula(r'\lambda='+f'{lam.get_value():.1f}',27,YELLOW,(4.5,-1.3,0)))
        knob=Dot(color=YELLOW,radius=.09).add_updater(lambda m:m.move_to(slide.n2p(lam.get_value())))
        self.remove(curve);self.add(moving,slide,knob,val)
        self.beat(lam.animate(rate_func=there_and_back).set_value(2.0),Indicate(moving,color=GREEN,scale_factor=1))
        self.remove(ax,ticks,pts,moving,slide,knob,val,e)
        gx=Axes(x_range=[-1.5,1.5,.5],y_range=[0,2,.5],x_length=7.5,y_length=3.6,tips=False,axis_config=dict(color=MUTED,include_ticks=False)).move_to((0,.1,0))
        graph=gx.plot(lambda x:x*x+.25,x_range=[-1.3,1.3],color=BLUE)
        q=ValueTracker(-1.0)
        dot=always_redraw(lambda:Dot(gx.c2p(q.get_value(),q.get_value()**2+.25),radius=.07,color=YELLOW))
        tangent=always_redraw(lambda:Line(gx.c2p(q.get_value()-.45,q.get_value()**2+.25-.9*q.get_value()),gx.c2p(q.get_value()+.45,q.get_value()**2+.25+.9*q.get_value()),color=YELLOW,stroke_width=2))
        zero=Dot(gx.c2p(0,.25),color=GREEN,radius=.08)
        aid=label('補足  傾き → 0 → 逆向き',23,YELLOW,(0,2.55,0))
        self.add(gx,graph,dot,tangent,aid)
        self.beat(q.animate.set_value(1.0),FadeIn(zero))
        self.remove(gx,graph,dot,tangent,zero,aid)
        grad=formula(r'\nabla_wJ=\sum_n(w^T\phi(x_n)-t_n)\phi(x_n)+\lambda w=0',31,WHITE,(0,1.05,0))
        arrow=VGroup(*[Arrow(((-2.6 if i==1 else -4.1)+i*2.5,-.65,0),((-4.1 if i==1 else -2.6)+i*2.5,-.65,0),buff=0,color=BLUE if i!=1 else RED,stroke_width=5) for i in range(3)])
        self.beat(FadeIn(grad),FadeIn(arrow))
        w=formula(r'w=\sum_{n=1}^{3}a_n\phi(x_n)=\Phi^Ta',38,GREEN,(0,1,0))
        aterms=VGroup(*[formula(v,30,c,(i*3-3,-.65,0)) for i,(v,c) in enumerate([(r'a_1\phi(x_1)',BLUE),(r'a_2\phi(x_2)',RED),(r'a_3\phi(x_3)',BLUE)])])
        self.beat(FadeOut(grad),FadeIn(w),FadeOut(arrow),FadeIn(aterms))
        co=formula(r'a_n=\frac{t_n-w^T\phi(x_n)}{\lambda}',35,YELLOW)
        self.beat(FadeIn(co),Indicate(aterms[1],color=RED,scale_factor=1.15))
        prompt=self.topnote('係数 a を直接決めればよい',YELLOW)
        self.beat(FadeIn(prompt),Indicate(w,color=GREEN,scale_factor=1.02))
        compare=formula(r'w_1,w_2,w_3\quad\longleftrightarrow\quad a_1,a_2,a_3',30,WHITE,(0,-1.25,0))
        self.beat(FadeOut(co),FadeIn(compare),Indicate(w,color=GREEN,scale_factor=1.02))
        bridge=formula(r'w^T\phi(x)=\sum_n a_n\,\phi(x_n)^T\phi(x)',32,YELLOW,(0,-2.35,0))
        self.beat(FadeOut(compare),FadeIn(bridge),Indicate(aterms,color=YELLOW,scale_factor=1.02))

    def kernel(self):
        xs=VGroup(*[formula(v,35,c,(i*2.9-2.9,.45,0)) for i,(v,c) in enumerate([(r'x',RECAP_DATA),(r'\phi(x)',RECAP_GOLD),(r'(1,x,x^2)',RECAP_BASIS)])])
        arrows=VGroup(Arrow(xs[0].get_right(),xs[1].get_left(),buff=.15,color=YELLOW),Arrow(xs[1].get_right(),xs[2].get_left(),buff=.15,color=YELLOW))
        recap=label('復習: 3.1 入力から特徴へ',25,YELLOW,(0,2.3,0))
        border=SurroundingRectangle(VGroup(xs,arrows,recap),buff=.4,color=YELLOW,stroke_width=1.2)
        self.beat(FadeIn(xs),FadeIn(arrows),FadeIn(recap),Create(border))
        self.remove(xs,arrows,recap,border)
        probe=formula(r'x_*=0.5\quad\longrightarrow\quad\phi(x_*)=(1,0.5,0.25)^T',37,BLUE,(0,.8,0))
        tr=formula(r'\phi(-1)=(1,-1,1)^T,\quad\phi(0)=(1,0,0)^T,\quad\phi(1)=(1,1,1)^T',29,PURPLE,(0,-.25,0))
        self.beat(FadeIn(probe),FadeIn(tr))
        mult=VGroup(*[formula(s,33,c,(i*2.8-2.8,.2,0)) for i,(s,c) in enumerate([(r'1\times1',BLUE),(r'x\times x^\prime',YELLOW),(r'x^2\times {x^\prime}^2',PURPLE)])])
        ke=formula(r'k(x,x^\prime)=\phi(x)^T\phi(x^\prime)=1+xx^\prime+x^2{x^\prime}^2',32,GREEN,(0,-1.35,0))
        self.beat(FadeOut(probe),FadeOut(tr),FadeIn(mult),FadeIn(ke))
        example=formula(r'k(1,0.5)=1+0.5+0.25=1.75',37,GREEN,(0,.85,0))
        self.beat(FadeOut(mult),FadeIn(example),Indicate(ke,color=GREEN,scale_factor=1.01))
        sym=formula(r'k(x,x^\prime)=k(x^\prime,x)',39,YELLOW,(0,.8,0))
        note=self.topnote('内積の値。距離や確率ではない',MUTED)
        self.beat(FadeOut(example),FadeIn(sym),FadeIn(note))
        self.beat(FadeOut(sym),FadeOut(note),ke.animate.move_to((0,.4,0)))
        left=formula(r'k(-1,0.5)=0.75',36,BLUE,(0,1.35,0))
        self.beat(FadeIn(left),Indicate(ke,color=YELLOW,scale_factor=1.02))
        center=formula(r'k(0,0.5)=1',36,PURPLE,(0,-.75,0))
        self.beat(FadeIn(center),Indicate(left,color=BLUE,scale_factor=1.02))

    def gram_table(self,origin=(0,.35,0),cell=.85):
        grid=VGroup(); nums=VGroup(); labels=VGroup()
        for i in range(3):
            for j in range(3):
                pos=np.array([origin[0]+(j-1)*cell,origin[1]+(1-i)*cell,0])
                grid.add(Square(side_length=cell,stroke_color=MUTED,stroke_width=1.2).move_to(pos))
                nums.add(formula(str(int(K[i,j])),29,GREEN,pos))
        for i,x in enumerate(X):
            labels.add(formula(f'{int(x)}',22,BLUE,(origin[0]-2*cell,origin[1]+(1-i)*cell,0)))
            labels.add(formula(f'{int(x)}',22,PURPLE,(origin[0]+(i-1)*cell,origin[1]+2*cell,0)))
        return grid,nums,labels

    def gram(self):
        grid,nums,labels=self.gram_table()
        one=SurroundingRectangle(grid[0],buff=.02,color=YELLOW,stroke_width=2)
        hint=self.topnote('補足  行の点 × 列の点',YELLOW)
        self.beat(FadeIn(grid),FadeIn(labels),FadeIn(nums[0]),Create(one),FadeIn(hint))
        self.remove(one,hint)
        diag=VGroup(nums[0],nums[4],nums[8])
        self.beat(FadeIn(nums[4]),FadeIn(nums[8]),Indicate(nums[0],color=YELLOW,scale_factor=1.02))
        off=VGroup(*[nums[i] for i in [1,2,3,5,6,7]])
        self.beat(LaggedStart(*[FadeIn(m) for m in off],lag_ratio=.14))
        eq=formula(r'K=\Phi\Phi^T=\begin{pmatrix}3&1&1\\1&1&1\\1&1&3\end{pmatrix}',32,GREEN,(4.3,.1,0))
        self.beat(FadeIn(eq),Indicate(grid,color=YELLOW,scale_factor=1))
        objective=formula(r'J(a)=\frac12a^TK^2a-a^TKt+\frac12t^Tt+\frac\lambda2a^TKa',28,WHITE,(0,-2.45,0))
        self.beat(FadeIn(objective),Indicate(eq,color=GREEN,scale_factor=1.01))
        targets=formula(r't=(1,0,1)^T',34,BLUE,(4.25,1.45,0))
        self.beat(FadeIn(targets),Indicate(labels,color=BLUE,scale_factor=1.03))
        mapped=formula(r'Ka=\begin{pmatrix}y(x_1)\\y(x_2)\\y(x_3)\end{pmatrix}',30,YELLOW,(4.25,.15,0))
        self.beat(FadeOut(eq),FadeIn(mapped))

    def coefficients(self):
        first=formula(r'a=\frac{t-Ka}{\lambda}',40,WHITE,(0,.65,0))
        prior=formula(r'a_n=\frac{t_n-w^T\phi(x_n)}{\lambda}',31,PURPLE,(0,1.75,0))
        self.beat(FadeIn(first),FadeIn(prior))
        next_eq=formula(r'(K+\lambda I)a=t',42,GREEN,(0,.45,0))
        note=label('正誤表: (6.3) を (6.4) へ代入',21,MUTED,(0,2.3,0))
        self.beat(FadeOut(first),FadeIn(next_eq),FadeIn(note))
        self.remove(prior,note,next_eq)
        grid,nums,labels=self.gram_table(origin=(-2.5,.3,0),cell=.8)
        plus=VGroup(*[formula(r'+0.5',21,YELLOW,grid[i].get_center()+DOWN*.25) for i in [0,4,8]])
        eq2=formula(r'K+0.5I',34,GREEN,(2.5,1.2,0))
        self.beat(FadeIn(grid),FadeIn(nums),FadeIn(labels),FadeIn(plus),FadeIn(eq2))
        solve=formula(r'a=(K+\lambda I)^{-1}t',38,YELLOW,(2.65,.3,0))
        solve_note=label('連立方程式を解く',24,YELLOW,(2.65,-.7,0))
        self.beat(FadeOut(eq2),FadeIn(solve),FadeIn(solve_note))
        self.remove(grid,nums,labels,plus,solve,solve_note)
        coeff=formula(r'a=(0.315789,-0.421053,0.315789)^T',33,GREEN,(0,1.15,0))
        bars=VGroup()
        for i,a in enumerate(A):
            color=RED if a<0 else BLUE
            start=np.array([i*2-2,-.4,0]);end=start+UP*float(a*3)
            bars.add(Line(start,end,stroke_width=17,color=color),formula(f'{a:+.3f}',22,color,end+UP*(.2 if a>=0 else -.3)))
        self.beat(FadeIn(coeff),FadeIn(bars))
        verify=formula(r'(K+0.5I)a=t',38,YELLOW,(0,-2.45,0))
        self.beat(FadeIn(verify),Indicate(coeff,color=GREEN,scale_factor=1.02))
        residual=formula(r't-Ka=(0.158,-0.211,0.158)^T=0.5a',29,WHITE,(0,-2.45,0))
        self.beat(FadeOut(verify),FadeIn(residual),Indicate(bars,color=YELLOW,scale_factor=1.02))
        center_note=label('中央: 目標 0、予測 0.211',23,RED,(0,2.4,0))
        self.beat(FadeIn(center_note),Indicate(bars[1],color=RED,scale_factor=1.15))

    def recovery(self):
        ax,ticks=self.plot_axes(center=(-1.15,.15,0),width=8)
        pts=self.training_points(ax);probe=DashedLine(ax.c2p(PROBE,-.12),ax.c2p(PROBE,1.08),color=YELLOW,dash_length=.08)
        self.beat(Create(ax),FadeIn(ticks),FadeIn(pts),Create(probe))
        kval=formula(r'k(x_*)=(0.75,1,1.75)^T',37,PURPLE,(4.45,1.15,0))
        compare=VGroup(*[Line((2.45+i*.85,-.45,0),(2.45+i*.85,-.45+.6*float(v),0),stroke_width=15,color=PURPLE) for i,v in enumerate(K_PROBE)])
        self.beat(FadeIn(kval),FadeIn(compare))
        contrib=formula(r'k_n(x_*)a_n=(0.237,-0.421,0.553)',29,GREEN,(3.35,2.15,0))
        cs=VGroup(*[Line((2.45+i*.85,-.35,0),(2.45+i*.85,-.35+float(v)*1.8,0),stroke_width=16,color=RED if v<0 else GREEN) for i,v in enumerate(CONTRIBUTIONS)])
        self.beat(ReplacementTransform(compare,cs),FadeIn(contrib))
        result=formula(r'y(0.5)=\sum_n a_n k(x_n,0.5)=0.368421',32,GREEN)
        curve=self.fit_curve(ax);dot=Dot(ax.c2p(PROBE,PREDICTION),radius=.09,color=GREEN)
        self.beat(FadeIn(result),Create(curve),FadeIn(dot))
        eqs=self.pair_equation(r'w^T\phi(x_*)',r'k(x_*)^Ta')
        self.beat(FadeOut(kval),FadeOut(contrib),FadeOut(cs),FadeIn(eqs),Indicate(dot,color=GREEN,scale_factor=1.25))
        guess=Dot(ax.c2p(PROBE,.5),radius=.07,color=PURPLE)
        naive=label('予想 0.5',23,PURPLE,(4.4,1.3,0))
        found=label('予測 0.368',23,GREEN,(4.4,.7,0))
        self.beat(FadeOut(eqs),FadeIn(guess),FadeIn(naive),FadeIn(found),Indicate(dot,color=YELLOW,scale_factor=1.2))
        exact=formula(r'0.368421\ldots=\frac{7}{19}',32,YELLOW,(4.2,2.15,0))
        self.beat(FadeOut(naive),FadeOut(found),FadeIn(exact),Indicate(dot,color=YELLOW,scale_factor=1.2))
        trail=Dot(ax.c2p(-.5,float(phi(-.5)@W)),radius=.075,color=YELLOW)
        path=ax.plot(lambda x:float(phi(x)@W),x_range=[-.5,.5],color=YELLOW,stroke_width=5)
        self.beat(MoveAlongPath(trail,path),Indicate(curve,color=GREEN,scale_factor=1.01))

    def outlook(self):
        eq=formula(r'y(x)=k(x)^T(K+\lambda I)^{-1}t',40,GREEN,(0,.7,0))
        phi_cross=formula(r'\phi(x)\;\;\longrightarrow\;\; k(x,x_n)',32,YELLOW,(0,-.8,0))
        self.beat(FadeIn(eq),FadeIn(phi_cross))
        xs=np.linspace(-1.15,1.15,100)
        ax,ticks=self.plot_axes(center=(-1,.1,0),width=7.7)
        original=self.fit_curve(ax)
        alt=line(ax,xs,alternative_kernel(xs[:,None],X[None,:])@ALT_A,YELLOW)
        kin=formula(r'k_1=1+xx^\prime+x^2{x^\prime}^2',25,GREEN,(4.5,1.15,0))
        kout=formula(r'k_2=(1+xx^\prime)^3',25,YELLOW,(4.5,.55,0))
        self.remove(eq,phi_cross)
        alt_points=self.training_points(ax)
        self.add(ax,ticks,alt_points,original,kin)
        changed=formula(r'y_2(0.5)=0.302',24,YELLOW,(4.5,-.1,0))
        self.beat(Transform(original,alt),FadeIn(kout),FadeIn(changed),FadeOut(kin))
        self.remove(ax,ticks,alt_points,original,kout,changed)
        sizing=VGroup(formula(r'K:\ N\times N',34,PURPLE,(-3.3,1.3,0)),formula(r'\Phi^T\Phi:\ M\times M',34,BLUE,(3.3,1.3,0)))
        n_grid=VGroup(*[Square(.2,stroke_color=PURPLE,stroke_width=1.3).move_to((-4.0+j*.25,.55-i*.25,0)) for i in range(8) for j in range(8)])
        m_grid=VGroup(*[Square(.4,stroke_color=BLUE,stroke_width=1.5).move_to((3.3+j*.48,.1-i*.48,0)) for i in range(3) for j in range(3)])
        ex=label('説明用: N=8、M=3',22,MUTED,(0,2.35,0))
        self.beat(FadeIn(sizing),FadeIn(n_grid),FadeIn(m_grid),FadeIn(ex))
        n_grid2=n_grid.copy().scale(1.12).move_to(n_grid)
        self.beat(Transform(n_grid,n_grid2),Indicate(sizing[0],color=PURPLE,scale_factor=1.1))
        cloud=VGroup(*[formula(v,33,c,(x,.2,0)) for v,c,x in [(r'\phi_1',BLUE,-3),(r'\phi_2',BLUE,-1.7),(r'\cdots',MUTED,0),(r'\phi_M',BLUE,1.8),(r'k(x,x^\prime)',GREEN,4.15)]])
        self.beat(FadeOut(sizing),FadeOut(n_grid),FadeOut(m_grid),FadeOut(ex),FadeIn(cloud))
        final=label('次の問い: どんな比較がカーネルになる？',29,YELLOW,(0,.3,0))
        self.beat(FadeOut(cloud),FadeIn(final))
