"""PRML 3.6: linked experiments in Manim CE, narrated sentence by sentence."""
import numpy as np
from manim import *
from scene_support import NarratedScene
from caption_layout import jp, tex
from basis_model import (CENTERS, WIDTH, X, T, BEST, design, manifold, CLEAN,
                         NOISE, DATA, GRID, NEAR, LOCAL, SQUARE, TARGET,
                         sigmoid, direction, response, rmse, radial)

BLUE_DATA=ManimColor('#58B5ED')
RED_MODEL=ManimColor('#FF6B77')
YELLOW_BASIS=ManimColor('#FFE079')
PURPLE_BASIS=ManimColor('#C29AFF')
GREEN_BASIS=ManimColor('#77D49A')
MUTED=ManimColor('#A8B2C5')
COLORS=[YELLOW_BASIS,PURPLE_BASIS,GREEN_BASIS]
AID_INPUT=ManimColor('#58C4DD')
AID_OPERATION=ManimColor('#FFFF00')
AID_RESULT=ManimColor('#83C167')

def line(points,color=WHITE,width=3,opacity=1):
    return VMobject().set_points_as_corners(points).set_stroke(color,width,opacity)

def curve(ax,x,y,color=RED_MODEL,width=3):
    return line([ax.c2p(a,b) for a,b in zip(x,y)],color,width)

def readout(label,getter,position,color=WHITE,places=2,size=27):
    prefix=tex(label,size,color)
    number=DecimalNumber(getter(),num_decimal_places=places,font_size=size,color=color)
    group=VGroup(prefix,number).arrange(RIGHT,buff=.12).move_to(position)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group

def slider(tracker,lo,hi,pos,label,color=YELLOW_BASIS,width=2.4):
    rail=Line(LEFT*width/2,RIGHT*width/2,color=MUTED,stroke_width=2).move_to(pos)
    dot=Dot(radius=.065,color=color)
    dot.add_updater(lambda m:m.move_to(rail.get_start()+(tracker.get_value()-lo)/(hi-lo)*rail.get_vector()))
    lab=tex(label,24,color).next_to(rail,LEFT,buff=.18)
    return VGroup(rail,dot,lab)

class PRML36FixedBasisLimitations(NarratedScene):
    def scenes(self):
        return [self.fixed,self.growth,self.manifold_scene,self.local,
                self.relevant,self.adaptive,self.conclusion]

    def hint(self,text,color=MUTED):
        obj=jp(text,21,color).move_to([0,2.74,0]); self.add(obj); return obj

    def axes(self,xr=(-1,1,.5),yr=(0,1.5,.5),center=(0,.25,0),width=9,height=3.6,labels=('x','y')):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.3,'include_ticks':False}).move_to(center)
        self.add(ax)
        for x in [xr[0],xr[1]]:
            self.add(tex(str(x),19,MUTED).next_to(ax.c2p(x,yr[0]),DOWN,buff=.12))
        for y in [yr[0],yr[1]]:
            self.add(tex(str(y),19,MUTED).next_to(ax.c2p(xr[0],y),LEFT,buff=.12))
        self.add(tex(labels[0],25).next_to(ax.c2p(xr[1],yr[0]),RIGHT,buff=.2),
                 tex(labels[1],25).next_to(ax.c2p(xr[0],yr[1]),UP,buff=.13))
        return ax

    def plane(self,center=(-1.9,.05,0),size=4.1):
        ax=self.axes(yr=(-1,1,.5),center=center,width=size,height=size,labels=('x_1','x_2'))
        for v in [-1,-.5,.5,1]:
            self.add(line([ax.c2p(v,-1),ax.c2p(v,1)],MUTED,1,.15),
                     line([ax.c2p(-1,v),ax.c2p(1,v)],MUTED,1,.15))
        return ax

    def fixed(self):
        self.hint('青：観測　赤：予測　黄・紫・緑：基底の寄与')
        ax=self.axes(width=9,height=3.25,center=(0,.35,0))
        u=np.linspace(-1,1,201)
        ws=[ValueTracker(v) for v in [0,.3,.2,.3]]
        weights=lambda:np.array([w.get_value() for w in ws])
        parts=VGroup(*[always_redraw(lambda j=j:curve(ax,u,design(u)[:,j+1]*ws[j+1].get_value(),COLORS[j],2)) for j in range(3)])
        model=always_redraw(lambda:curve(ax,u,design(u)@weights()))
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.045,color=BLUE_DATA) for x,t in zip(X,T)])
        self.add(parts,model)
        self.beat(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=.04))
        knobs=VGroup(*[slider(ws[j+1],0,1.3,[-3.7+3.7*j,-1.8,0],f'w_{j+1}',COLORS[j],2.3) for j in range(3)])
        self.add(knobs)
        self.beat(ws[1].animate.set_value(1.2))
        self.beat(ws[2].animate.set_value(.8))
        formula=MathTex('y=',r'\sum_{j=0}^{M-1}',r'w_j',r'\phi_j(x)',font_size=34).move_to([0,-2.53,0])
        formula[2].set_color(RED_MODEL);formula[3].set_color(YELLOW_BASIS)
        self.beat(Write(formula),ShowPassingFlash(model.copy().clear_updaters(),time_width=.35))
        residuals=always_redraw(lambda:VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,y)+UP*1e-8,color=BLUE_DATA,stroke_opacity=.5) for x,t,y in zip(X,T,design(X)@weights())]))
        self.add(residuals)
        self.beat(*[w.animate.set_value(v) for w,v in zip(ws,BEST)])
        normal=tex(r'\mathbf w_{\rm ML}=(\Phi^T\Phi)^{-1}\Phi^T\mathbf t',32).move_to(formula)
        self.beat(ReplacementTransform(formula,normal),ShowPassingFlash(model.copy().clear_updaters(),time_width=.35))
        self.beat(Circumscribe(parts,color=YELLOW_BASIS),Circumscribe(knobs,color=YELLOW_BASIS))
        question=jp('入力が増えたら、基底は何個？',30,YELLOW_BASIS).move_to(normal)
        self.beat(ReplacementTransform(normal,question),ShowPassingFlash(model.copy().clear_updaters(),time_width=.7))

    def growth(self):
        self.grid_recap()
        hint=self.hint('各方向に 5 個の局所基底を置く例')
        def grid(dim):
            pts=[]
            for z in range(5 if dim>=3 else 1):
                for y in range(5 if dim>=2 else 1):
                    for x in range(5):
                        pts.append([(x-2)*.85+(z-2 if dim>=3 else 0)*.23,
                                    (y-2)*.62+(z-2 if dim>=3 else 0)*.24 if dim>=2 else .1,0])
            return VGroup(*[Dot(a,radius=.065,color=YELLOW_BASIS) for a in pts])
        dots=grid(1); label=tex(r'D=1\qquad B=5',36).move_to([0,-2.2,0])
        self.add(label);self.beat(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=.1))
        nxt=grid(2);lab2=tex(r'D=2\qquad B=25',36).move_to(label)
        self.beat(ReplacementTransform(dots,nxt,run_time=5),ReplacementTransform(label,lab2,run_time=.7));dots=nxt;label=lab2
        nxt=grid(3);lab3=tex(r'D=3\qquad B=125',36).move_to(label)
        self.beat(ReplacementTransform(dots,nxt,run_time=5),ReplacementTransform(label,lab3,run_time=.7));dots=nxt;label=lab3
        formula=MathTex('B=', 'K^D',font_size=44).move_to([0,-2.2,0]);formula[1].set_color(YELLOW_BASIS)
        self.beat(ReplacementTransform(label,formula,run_time=.7),Indicate(dots,scale_factor=1.03,run_time=5))
        self.remove(dots)
        ax=self.axes((1,10,1),(0,7,1),center=(-.6,.15,0),width=7.8,height=3.35,labels=('D',r'\log_{10}B'))
        for y in range(1,7):
            self.add(tex(str(y),18,MUTED).next_to(ax.c2p(1,y),LEFT,buff=.12),line([ax.c2p(1,y),ax.c2p(10,y)],MUTED,1,.18))
        for x in [3,6]: self.add(tex(str(x),19,MUTED).next_to(ax.c2p(x,0),DOWN,buff=.12))
        dim=ValueTracker(3)
        path=always_redraw(lambda:curve(ax,np.linspace(1,dim.get_value(),100),np.linspace(1,dim.get_value(),100)*np.log10(5),YELLOW_BASIS))
        dot=always_redraw(lambda:Dot(ax.c2p(dim.get_value(),dim.get_value()*np.log10(5)),color=YELLOW_BASIS))
        counter=readout('B=',lambda:5**round(dim.get_value()),[4.5,1.35,0],YELLOW_BASIS,0,25)
        self.add(path,dot,counter,tex('K=5',26,MUTED).move_to([4.5,2,0]))
        self.beat(dim.animate.set_value(6))
        self.beat(dim.animate.set_value(10))
        integers=VGroup(*[Dot(ax.c2p(d,d*np.log10(5)),radius=.04,color=WHITE) for d in range(1,11)])
        self.beat(LaggedStart(*[FadeIn(d) for d in integers],lag_ratio=.1),Indicate(formula))
        note=jp('全空間を覆う前に、データを見る',27,GREEN_BASIS).move_to([0,-2.2,0])
        self.beat(ReplacementTransform(formula,note),Indicate(path,scale_factor=1.015))

    def manifold_scene(self):
        self.hint('独自データ：曲線＋小さな測定の揺らぎ')
        recap=jp('復習: 1.4 多様体',23).move_to([3.2,2.05,0])
        self.add(recap)
        ax=self.plane(center=(-1.7,.05,0),size=4.2)
        flatten=ValueTracker(0);noise=ValueTracker(1);s=ValueTracker(-.9)
        coords=lambda: np.column_stack([CLEAN[:,0],CLEAN[:,1]*(1-flatten.get_value())])+NOISE*noise.get_value()
        dots=VGroup(*[Dot(radius=.035,color=BLUE_DATA) for _ in DATA])
        for i,d in enumerate(dots):d.add_updater(lambda m,i=i:m.move_to(ax.c2p(*coords()[i])))
        self.beat(FadeIn(dots))
        self.remove(recap)
        u=np.linspace(-1,1,180)
        path=always_redraw(lambda:curve(ax,u,.68*np.sin(2.4*u)*(1-flatten.get_value()),GREEN_BASIS,3))
        self.beat(Create(path))
        probe=always_redraw(lambda:Dot(ax.c2p(s.get_value(),.68*np.sin(2.4*s.get_value())*(1-flatten.get_value())),color=YELLOW_BASIS,radius=.075))
        self.add(probe,slider(s,-1,1,[3.25,-1.45,0],'s',YELLOW_BASIS,2.5))
        self.beat(s.animate.set_value(.9))
        label=VGroup(jp('記録する数：2 個',28,BLUE_DATA),jp('曲線上の自由度：1',28,GREEN_BASIS),jp('多様体',35,GREEN_BASIS)).arrange(DOWN,buff=.43).move_to([3.2,.9,0])
        self.beat(Write(label),s.animate.set_value(-.5))
        self.beat(flatten.animate.set_value(1),noise.animate.set_value(0))
        equation=tex(r'(x_1,x_2)=(s,\ 0.68\sin(2.4s))',28,GREEN_BASIS).move_to([0,-2.6,0])
        self.beat(Write(equation),flatten.animate.set_value(0),s.animate.set_value(.5))
        self.beat(noise.animate.set_value(1),s.animate.set_value(-.7))
        self.beat(s.animate.set_value(.9),Indicate(label[1],scale_factor=1.04))

    def local(self):
        self.hint('青：データ　橙：基底の中心　黄：動かす入力')
        ax=self.plane(center=(-2,.03,0),size=4.15)
        dots=VGroup(*[Dot(ax.c2p(*v),color=BLUE_DATA,radius=.032) for v in DATA]);self.add(dots)
        centers=VGroup(*[Dot(ax.c2p(*v),color=ORANGE,radius=.05) for v in GRID])
        count=jp('均等配置：81 個',29,ORANGE).move_to([3,.9,0]);self.add(count)
        self.beat(LaggedStart(*[FadeIn(d) for d in centers],lag_ratio=.008))
        self.beat(*[d.animate.set_opacity(1 if n else .1) for d,n in zip(centers,NEAR)])
        selected=jp(f'距離 < 0.18：{int(NEAR.sum())} 個',26,GREEN_BASIS).move_to([3,.2,0]);self.add(selected)
        self.beat(Indicate(VGroup(*[d for d,n in zip(centers,NEAR) if n]),color=GREEN_BASIS,scale_factor=1.12))
        initial=GRID[np.round(np.linspace(0,80,12)).astype(int)]
        movers=VGroup(*[Dot(ax.c2p(*v),color=ORANGE,radius=.07) for v in initial]);self.remove(centers,*centers);self.add(movers)
        twelve=jp('データから選ぶ：12 個',28,GREEN_BASIS).move_to(count)
        self.beat(*[d.animate.move_to(ax.c2p(*v)) for d,v in zip(movers,LOCAL)],ReplacementTransform(count,twelve),FadeOut(selected))
        s=ValueTracker(-.92);h=ValueTracker(.22)
        probe=always_redraw(lambda:Dot(ax.c2p(*manifold(s.get_value())),color=YELLOW_BASIS,radius=.075))
        rings=VGroup(*[Circle(radius=.19,color=ORANGE).move_to(ax.c2p(*v)) for v in LOCAL])
        for j,r in enumerate(rings):
            r.add_updater(lambda m,j=j:m.set_stroke(opacity=.08+.92*radial(manifold(s.get_value()),h=h.get_value())[j],width=3))
        self.add(rings,probe,slider(s,-1,1,[3,-1.15,0],'s',YELLOW_BASIS,2.6))
        self.beat(s.animate.set_value(.92))
        formula=MathTex(r'\phi_j(\mathbf x)=\exp\!\left(-\frac{\|\mathbf x-\mu_j\|^2}{2h^2}\right)',font_size=29).move_to([0,-2.62,0])
        formula.set_color(ORANGE)
        self.beat(Write(formula),s.animate.set_value(-.65))
        methods=VGroup(jp('動径基底関数ネットワーク',22),jp('サポートベクトルマシン',22),jp('関連ベクトルマシン',22)).arrange(DOWN,buff=.25).move_to([3,1.4,0])
        self.remove(twelve);self.beat(FadeIn(methods),s.animate.set_value(.65))
        self.beat(s.animate.set_value(-.3),Circumscribe(movers,color=ORANGE))

    def heatmap(self,ax,angle,slope,bias):
        n=96;v=np.linspace(-1,1,n);xx,yy=np.meshgrid(v,v[::-1]);points=np.stack([xx,yy],axis=-1)
        def pixels():
            values=response(points,angle.get_value(),slope.get_value(),bias.get_value())
            low=np.array([34,70,117]);high=np.array([186,56,76])
            rgb=(low[None,None,:]+values[:,:,None]*(high-low)).astype(np.uint8)
            return np.concatenate([rgb,np.full((n,n,1),255,dtype=np.uint8)],axis=2)
        heat=ImageMobject(pixels()).set_resampling_algorithm(RESAMPLING_ALGORITHMS['bilinear'])
        heat.stretch_to_fit_width(np.linalg.norm(ax.c2p(1,0)-ax.c2p(-1,0)))
        heat.stretch_to_fit_height(np.linalg.norm(ax.c2p(0,1)-ax.c2p(0,-1)))
        heat.move_to(ax.c2p(0,0)).set_z_index(-3)
        heat.add_updater(lambda m:setattr(m,'pixel_array',pixels()))
        self.add(heat)
        arrow=always_redraw(lambda:Arrow(ax.c2p(0,0),ax.c2p(*(.82*direction(angle.get_value()))),buff=0,color=YELLOW_BASIS,stroke_width=5))
        self.add(arrow)
        return heat,arrow

    def colored_data(self,ax):
        return VGroup(*[Dot(ax.c2p(*v),radius=.049,color=interpolate_color(BLUE_DATA,RED_MODEL,float(t))).set_stroke(WHITE,.6) for v,t in zip(SQUARE,TARGET)])

    def relevant(self):
        self.hint('点の色：目標　背景：方向から計算した予測　青 0 → 赤 1')
        ax=self.plane(center=(-2.6,.1,0),size=4.1)
        dots=self.colored_data(ax);self.beat(FadeIn(dots))
        angle=ValueTracker(0);slope=ValueTracker(3);bias=ValueTracker(0)
        heat,arrow=self.heatmap(ax,angle,slope,bias)
        theta=readout(r'\theta=',lambda:angle.get_value()*180/PI,[3.2,1.6,0],YELLOW_BASIS,0)
        error=readout(r'\mathrm{RMSE}=',lambda:rmse(angle.get_value()),[3.2,-1.7,0],RED_MODEL,3)
        self.add(theta,error,jp('誤差：二乗平均の平方根',19,MUTED).move_to([3.2,-2.2,0]),slider(angle,0,PI/2,[3.2,.85,0],r'\theta',YELLOW_BASIS,2.5))
        self.beat(Indicate(arrow,scale_factor=1.025))
        self.beat(angle.animate.set_value(PI/4))
        formula=tex(r'z=\frac{x_1+x_2}{\sqrt2}\qquad t=\frac{1}{1+e^{-3z}}',29).move_to([0,-2.66,0])
        self.beat(Write(formula),Indicate(arrow,scale_factor=1.03))
        self.projection_aid()
        self.beat(Indicate(arrow,scale_factor=1.03))
        progress=ValueTracker(-.8);mode=ValueTracker(0)
        pos=lambda: np.array([progress.get_value(),-progress.get_value()]) if mode.get_value()<.5 else np.array([progress.get_value(),progress.get_value()])
        probe=always_redraw(lambda:Dot(ax.c2p(*pos()),radius=.1,color=interpolate_color(BLUE_DATA,RED_MODEL,float(response(pos())))).set_stroke(WHITE,2))
        value=readout('t=',lambda:float(response(pos())),[3.2,-.25,0],WHITE,3,33)
        self.add(probe,value)
        self.beat(progress.animate.set_value(.8))
        mode.set_value(1);progress.set_value(-.8)
        self.beat(progress.animate.set_value(.8))
        text=jp('広がり：2 次元\n目標に効く方向：1',25,GREEN_BASIS).move_to([3.2,.1,0])
        self.remove(value);self.beat(FadeIn(text),Circumscribe(dots,color=GREEN_BASIS))
        self.beat(angle.animate.set_value(.15))

    def body_card(self,title):
        saved=[m for m in self.mobjects if m is not self.subtitle]
        header=[m for m in saved if m.get_center()[1]>3]
        self.clear()
        frame=RoundedRectangle(width=10.4,height=4.45,corner_radius=.12,
                               color=AID_OPERATION,stroke_width=1.2).move_to([0,.1,0])
        label=jp(title,23).move_to([-4.85,2.02,0],aligned_edge=LEFT)
        self.add(*header,frame,label)
        return saved

    def restore_body(self,saved):
        self.clear()
        self.add(*saved)

    def grid_recap(self):
        saved=self.body_card('復習: 1.4 次元の呪い')
        self.add(jp('説明用の例：各方向を5分割',19,MUTED).move_to([2.65,2.02,0]))
        # Preserve 1.4 grid(): blue squares, blue-to-purple depth layers.
        row=VGroup(*[Square(.66,color=BLUE_DATA,fill_opacity=.16)
                     .move_to([(i-2)*.68,0,0]) for i in range(5)]).move_to([-2.2,0,0])
        plane=VGroup(*[Square(.66,color=BLUE_DATA,fill_opacity=.13)
                       .move_to([(i-2)*.68,(j-2)*.68,0]) for i in range(5) for j in range(5)]).move_to(row)
        cube=VGroup(*[Square(.45,color=interpolate_color(BLUE_DATA,PURPLE_BASIS,k/4),fill_opacity=.03)
                      .move_to([(i-2)*.47+.27*k,(j-2)*.47+.17*k,0])
                      for k in range(5) for i in range(5) for j in range(5)]).move_to(row)
        count=tex('5',40,BLUE_DATA).move_to([2.4,.65,0])
        count25=tex(r'5\times5=25',38,BLUE_DATA).move_to(count)
        count125=tex(r'5\times5\times5=125',34,BLUE_DATA).move_to(count)
        centers=VGroup(*[Dot(c.get_center(),radius=.035,color=YELLOW_BASIS) for c in cube])
        mapping=VGroup(jp('箱ひとつ → 基底ひとつ',24,YELLOW_BASIS),
                       tex('B=125',34,YELLOW_BASIS)).arrange(DOWN,buff=.3).move_to([2.4,-.65,0])
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('R1.4 five boxes',a*.25,lambda:AnimationGroup(FadeIn(row),FadeIn(count))),
            ('R1.4 twenty-five boxes',a*.35,lambda:AnimationGroup(ReplacementTransform(row,plane),Succession(FadeOut(count),FadeIn(count25)))),
            ('R1.4 125 boxes',a*.40,lambda:AnimationGroup(ReplacementTransform(plane,cube),Succession(FadeOut(count25),FadeIn(count125)))),
            ('R1.4 centers are bases',b*.65,lambda:AnimationGroup(LaggedStart(*[FadeIn(d) for d in centers],lag_ratio=.006),FadeIn(mapping))),
            ('R1.4 basis count',b*.35,lambda:Indicate(centers,scale_factor=1.025)),
        ])
        self.restore_body(saved)

    def projection_aid(self):
        saved=self.body_card('補足: 方向への影と内積')
        self.add(jp('説明用の例・矢印の長さは1',19,MUTED).move_to([2.5,2.02,0]))
        ax=Axes(x_range=[-.15,1.2,.5],y_range=[-.15,1.2,.5],x_length=3.2,y_length=3.2,
                tips=False,axis_config={'color':MUTED,'include_ticks':False}).move_to([-2.5,-.15,0])
        u=np.ones(2)/np.sqrt(2);x=np.array([1.,0.]);p=(x@u)*u
        arrow=Arrow(ax.c2p(0,0),ax.c2p(*u),buff=0,color=AID_OPERATION)
        point=Dot(ax.c2p(*x),color=AID_INPUT,radius=.065)
        drop=DashedLine(ax.c2p(*x),ax.c2p(*p),color=AID_INPUT,dash_length=.08)
        shadow=Line(ax.c2p(0,0),ax.c2p(*p),color=AID_RESULT,stroke_width=7)
        foot=Dot(ax.c2p(*p),color=AID_RESULT,radius=.055)
        xlab=tex(r'\mathbf x=(1,0)',25,AID_INPUT).next_to(point,DOWN,buff=.2)
        ulab=tex(r'\mathbf u=(1,1)/\sqrt2',25,AID_OPERATION).move_to([-2.55,1.43,0])
        numberline=NumberLine(x_range=[0,1,.5],length=3.1,include_numbers=True,
                              font_size=22,color=MUTED).move_to([2.55,-.7,0])
        mark=Dot(numberline.n2p(1/np.sqrt(2)),radius=.07,color=AID_RESULT)
        result=tex(r'z=\mathbf u^T\mathbf x=\frac{1+0}{\sqrt2}=\frac1{\sqrt2}',29,AID_RESULT).move_to([2.45,.65,0])
        note=jp('矢印と同じ向きが ＋',22).move_to([2.45,-1.45,0])
        self.add(ax,tex('x_1',23).next_to(ax.x_axis,RIGHT,buff=.1),
                 tex('x_2',23).next_to(ax.y_axis,UP,buff=.1),point,xlab,ulab)
        a,b=[self.sentence_duration(i) for i in range(2)]
        self.beat(phases=[
            ('V08a unit direction',a*.45,lambda:GrowArrow(arrow)),
            ('V08a perpendicular projection',a*.55,lambda:Create(drop)),
            ('V08a signed shadow',b*.32,lambda:AnimationGroup(Create(shadow),FadeIn(foot),FadeIn(numberline),FadeIn(note))),
            ('V08a shadow becomes feature',b*.48,lambda:AnimationGroup(TransformFromCopy(foot,mark),FadeIn(result))),
            ('V08a scalar result',b*.20,lambda:Indicate(mark,scale_factor=1.4)),
        ])
        self.restore_body(saved)

    def adaptive(self):
        self.hint('内部の向き・位置・幅と、外側の重みを分けて動かす')
        ax=self.plane(center=(-3,.1,0),size=3.85)
        angle=ValueTracker(0);slope=ValueTracker(3);bias=ValueTracker(0);weight=ValueTracker(1)
        heat,arrow=self.heatmap(ax,angle,slope,bias)
        sx=self.axes((-1.4,1.4,.7),(0,1.1,.5),center=(3,.5,0),width=4.5,height=2.8,labels=('z',r'w_j\phi_j'))
        u=np.linspace(-1.4,1.4,160)
        response_curve=always_redraw(lambda:curve(sx,u,weight.get_value()*sigmoid(slope.get_value()*u+bias.get_value()),GREEN_BASIS))
        self.add(response_curve)
        knobs=VGroup(slider(angle,0,PI/2,[-3.2,-2.25,0],r'\theta',YELLOW_BASIS,1.8),slider(bias,-2,2,[.5,-2.25,0],'b_j',PURPLE_BASIS,1.8),slider(slope,1,8,[4.1,-2.25,0],r'\|a_j\|',GREEN_BASIS,1.7))
        self.add(knobs)
        self.beat(Create(response_curve))
        self.beat(angle.animate.set_value(PI/4))
        cues=self.beat_cues()
        self.beat(phases=[('shift band',cues[0]['end'],lambda:bias.animate.set_value(-1.5)),('sharpen band',cues[1]['end']-cues[1]['start'],lambda:slope.animate.set_value(7))])
        formula=MathTex(r'\phi_j(\mathbf x)=\sigma(',r'\mathbf a_j^T\mathbf x', '+', 'b_j', ')',font_size=30).move_to([0,-2.77,0]);formula[1].set_color(YELLOW_BASIS);formula[3].set_color(PURPLE_BASIS)
        self.beat(Write(formula),Circumscribe(knobs,color=YELLOW_BASIS))
        wnum=readout('w_j=',weight.get_value,[3,.0-1.35,0],GREEN_BASIS,2,27);self.add(wnum)
        self.beat(weight.animate.set_value(.35),start_sentence=1)
        self.beat(bias.animate.set_value(0),slope.animate.set_value(3),weight.animate.set_value(1),angle.animate.set_value(PI/4))
        note=jp('内部まで学ぶと、非線形の最適化',25,RED_MODEL).move_to([0,2.7,0])
        # Replace the top explanatory line without duplicating it.
        for m in list(self.mobjects):
            if isinstance(m,Text) and abs(m.get_center()[1]-2.74)<.02:self.remove(m)
        self.beat(FadeIn(note),Indicate(formula,scale_factor=1.025))
        self.beat(angle.animate.set_value(.9),bias.animate.set_value(.5),slope.animate.set_value(4))

    def conclusion(self):
        hint=self.hint('三つの実験を、同じ舞台で振り返る')
        ax=self.plane(center=(-2.2,0,0),size=3.95)
        u=np.linspace(-1,1,160)
        function=curve(ax,u,(design(u)@BEST)-.5,RED_MODEL)
        formula=MathTex('y=',r'\sum_j',r'w_j',r'\phi_j(\mathbf x)',font_size=40).move_to([3,1.3,0]);formula[2].set_color(RED_MODEL);formula[3].set_color(YELLOW_BASIS)
        self.add(formula);self.beat(Create(function))
        grid=VGroup(*[Dot(ax.c2p(*v),radius=.04,color=ORANGE) for v in GRID])
        growth=tex(r'5\ \to\ 25\ \to\ 125',32,ORANGE).move_to([3,.1,0])
        self.beat(FadeOut(function),FadeIn(grid),Write(growth))
        data=VGroup(*[Dot(ax.c2p(*v),radius=.032,color=BLUE_DATA) for v in DATA])
        local=VGroup(*[Circle(radius=.15,color=GREEN_BASIS).move_to(ax.c2p(*v)) for v in LOCAL])
        self.beat(FadeOut(grid),FadeIn(data),Create(local),FadeOut(growth))
        colored=self.colored_data(ax);angle=ValueTracker(0);slope=ValueTracker(3);bias=ValueTracker(0)
        self.remove(data,local);self.add(colored)
        heat,arrow=self.heatmap(ax,angle,slope,bias)
        self.beat(angle.animate.set_value(PI/4))
        reminder=VGroup(jp('データがある場所',28,GREEN_BASIS),jp('目標に効く方向',28,YELLOW_BASIS)).arrange(DOWN,buff=.45).move_to([3,-.2,0])
        self.beat(Write(reminder),Circumscribe(colored,color=GREEN_BASIS))
        models=jp('基底を選ぶ・基底を調整する',25).move_to([0,-2.55,0])
        self.beat(Write(models),angle.animate.set_value(.4))
        cues=self.beat_cues()
        self.beat(phases=[('outer weights',cues[0]['end']/2,lambda:Indicate(formula[2],scale_factor=1.2)),('inner basis',cues[0]['end']/2,lambda:Indicate(formula[3],scale_factor=1.1)),('data decides',cues[1]['end']-cues[1]['start'],lambda:angle.animate.set_value(PI/4))])
        next_text=jp('回帰から分類へ：入力を見る道具を作る',27).move_to(models)
        self.beat(ReplacementTransform(models,next_text),Indicate(formula[3],scale_factor=1.04))
