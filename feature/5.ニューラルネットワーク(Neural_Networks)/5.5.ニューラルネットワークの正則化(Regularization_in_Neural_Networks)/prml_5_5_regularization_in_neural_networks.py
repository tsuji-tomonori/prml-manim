"""PRML 5.5: linked numerical experiments in Manim Community."""
from pathlib import Path
import json
import numpy as np
from manim import *
import regularization_model as model
from video_support import jp,tex,caption_mobject
from narration_content import SCENES
from make_voicevox_narration import MANIFEST,OUTPUT_DIR,valid_entry

BLUE_DATA=ManimColor('#58B5ED'); RED_MODEL=ManimColor('#FF6B77')
GREEN_TRUE=ManimColor('#77D49A'); PURPLE_REG=ManimColor('#C29AFF')
ORANGE_VAL=ManimColor('#FFB45B'); YELLOW_TERM=ManimColor('#FFE079')
MUTED=ManimColor('#A8B2C5');BG='#10141F'


def line(points,color=RED_MODEL,width=3):
    return VMobject().set_points_as_corners(points).set_stroke(color,width)


def curve(ax,x,y,color=RED_MODEL):
    origin=ax.c2p(0,0);dx=ax.c2p(1,0)-origin;dy=ax.c2p(0,1)-origin
    return line(origin+np.asarray(x)[:,None]*dx+np.asarray(y)[:,None]*dy,color)


def axes(xrange=(0,1,.25),yrange=(-1.5,1.5,1),center=(-1.1,.0,0),width=8,height=3.65,xlabel='x',ylabel='y'):
    ax=Axes(x_range=xrange,y_range=yrange,x_length=width,y_length=height,tips=False,
            axis_config={'color':MUTED,'stroke_width':1.3,'include_numbers':True,'font_size':18})
    ax.move_to(center)
    labels=VGroup(tex(xlabel,24,MUTED).next_to(ax.x_axis.get_right(),RIGHT,buff=.15),
                  tex(ylabel,24,MUTED).next_to(ax.y_axis.get_top(),UP,buff=.13))
    return ax,VGroup(ax,labels)


def readout(label,getter,pos,color=WHITE,places=3):
    prefix=tex(label,26,color);num=DecimalNumber(getter(),num_decimal_places=places,font_size=26,color=color)
    group=VGroup(prefix,num).arrange(RIGHT,buff=.13).move_to(pos)
    anchor=num.get_left().copy()
    num.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group


def pixels(data,cell=.30,center=(0,0,0),signed=False):
    rows,cols=data.shape
    squares=VGroup()
    for i in range(rows):
        for j in range(cols):
            value=float(data[i,j]);color=BLUE_DATA if value>=0 else ORANGE_VAL
            sq=Square(cell,stroke_width=.7,stroke_color='#424B5C')
            sq.set_fill(color,opacity=min(abs(value)/(3 if signed else 1),1))
            sq.move_to([(j-(cols-1)/2)*cell,((rows-1)/2-i)*cell,0]);squares.add(sq)
    return squares.move_to(center)


class PRML55RegularizationInNeuralNetworks(Scene):
    def construct(self):
        self.camera.background_color=BG
        self.timeline=[]
        entries=json.loads(MANIFEST.read_text())['scenes']
        self.entries={e['id']:e for e in entries}
        for i,method in enumerate([self.question,self.decay,self.priors,self.stopping,self.invariance,
                                   self.tangent,self.augmentation,self.convolution,self.soft_sharing]):
            self.begin(i);method()
            assert self.beat_index==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path('scene_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def begin(self,i):
        self.clear();self.story=SCENES[i];self.beat_index=0;self.caption=None;self.formula=None;self.note=None
        self.entry=self.entries[self.story['id']]
        if not valid_entry(self.story,self.entry):raise ValueError('Audio and script mismatch')
        self.scene_start=float(self.time)
        self.add(jp(self.story['title'],32).move_to([0,3.45,0]))
        self.add(Line([-6.7,-2.91,0],[6.7,-2.91,0],color=MUTED,stroke_opacity=.22))
        self.add_sound(str(OUTPUT_DIR/f"{self.story['id']}.wav"))
        self.timeline.append(dict(id=self.story['id'],title=self.story['title'],reference=self.story['reference'],start=self.scene_start,beats=[]))

    def equation(self,expression,size=31):
        if self.formula is not None:self.remove(self.formula)
        self.formula=MathTex(expression,font_size=size,substrings_to_isolate=[r'\lambda',r'\alpha',r'\Omega',r'\tau']).move_to([0,2.53,0])
        self.formula.set_color_by_tex_to_color_map({r'\lambda':PURPLE_REG,r'\alpha':PURPLE_REG,r'\Omega':PURPLE_REG,r'\tau':YELLOW_TERM})
        if self.formula.width>12.4:raise ValueError('Equation too wide: '+expression)
        self.add(self.formula)
        return self.formula

    def label(self,message,color=MUTED):
        if self.note is not None:self.remove(self.note)
        self.note=jp(message,21,color).move_to([0,-2.55,0]);self.add(self.note)

    def beat(self,*factories):
        b=self.story['beats'][self.beat_index]
        cues=[c for c in self.entry['subtitle_cues'] if c['beat_index']==self.beat_index]
        record=dict(start=float(self.time),visual=b['visual_note'],cues=[],actions=[])
        for i,c in enumerate(cues):
            if self.caption is not None:self.remove(self.caption)
            self.caption=caption_mobject(c['display']);self.add(self.caption)
            end=cues[i+1]['start'] if i+1<len(cues) else sum(self.entry['beat_durations'][:self.beat_index+1])
            absolute_end=self.scene_start+end
            frames=round((absolute_end-float(self.time))*config.frame_rate)
            factory=factories[min(i,len(factories)-1)]
            action=factory()
            start=float(self.time)
            self.play(action,run_time=(frames-1e-5)/config.frame_rate,rate_func=linear)
            record['cues'].append(dict(id=c['id'],display=c['display'],start=self.scene_start+c['start'],end=self.scene_start+c['end']))
            record['actions'].append(dict(name=b['visual_note'],start=start,end=float(self.time)))
        record['end']=float(self.time);self.timeline[-1]['beats'].append(record);self.beat_index+=1

    def regression(self,getter):
        ax,group=axes();self.add(group)
        data=VGroup(*[Dot(ax.c2p(x,t),radius=.055,color=BLUE_DATA) for x,t in zip(model.X,model.T)])
        truth=curve(ax,model.GRID,np.sin(2*np.pi*model.GRID),GREEN_TRUE).set_stroke(opacity=.35)
        pred=always_redraw(lambda:curve(ax,model.GRID,model.predict(getter(),model.GRID)))
        self.add(truth,data,pred)
        return ax,group,data,pred

    def question(self):
        w=model.fit(1);ax,group,data,pred=self.regression(lambda:w)
        self.remove(pred);self.label('青：訓練データ　緑：生成関数（答え合わせ用）　赤：予測')
        self.beat(lambda:LaggedStart(*[Indicate(d,color=BLUE_DATA) for d in data],lag_ratio=.15),lambda:Create(pred))
        pred.clear_updaters();self.remove(pred);pred=curve(ax,model.GRID,model.predict(w,model.GRID));self.add(pred)
        self.equation(r'y(x)=\sum_{j=1}^{M}v_j\tanh(a_jx+b_j)+c')
        self.label('同じ座標系で比較：隠れユニット数 1 → 3 → 10')
        self.beat(lambda:Transform(pred,curve(ax,model.GRID,model.predict(model.fit(3),model.GRID))),lambda:Transform(pred,curve(ax,model.GRID,model.predict(model.fit(10),model.GRID))))
        validation=VGroup(*[Dot(ax.c2p(x,t),radius=.025,color=ORANGE_VAL) for x,t in zip(model.XV,model.TV)])
        self.beat(lambda:Create(validation),lambda:Indicate(pred,color=ORANGE_VAL,scale_factor=1))
        self.label('M = 10 を固定。初期値だけを変更')
        variants=[curve(ax,model.GRID,model.predict(model.fit(10,s),model.GRID)) for s in [551,553,554,550]]
        self.beat(lambda:Succession(Transform(pred,variants[0]),Transform(pred,variants[1])),lambda:Succession(Transform(pred,variants[2]),Transform(pred,variants[3])))
        residual=VGroup(*[Line(ax.c2p(x,t),ax.c2p(x,float(model.predict(model.fit(10),x))),color=ORANGE_VAL,stroke_width=2) for x,t in zip(model.XV[::4],model.TV[::4])])
        self.beat(lambda:Create(residual),lambda:Indicate(validation,color=ORANGE_VAL,scale_factor=1))
        self.label('正則化：予測に必要な自由度を選ぶ',PURPLE_REG)
        self.equation(r'\widetilde E(\mathbf w)=E(\mathbf w)+\frac{\lambda}{2}\|\mathbf w\|^2')
        self.beat(lambda:FadeOut(residual),lambda:Indicate(self.formula,color=PURPLE_REG))

    def decay(self):
        knob=ValueTracker(-5);get=lambda:model.ridge_weights(knob.get_value())
        ax,group,data,pred=self.regression(get)
        self.equation(r'\widetilde E=\underbrace{E}_{\text{data}}+\underbrace{\frac{\lambda}{2}\|\mathbf w\|^2}_{\text{weights}}')
        track=Line([3.5,-1.6,0],[6.2,-1.6,0],color=PURPLE_REG)
        dot=always_redraw(lambda:Dot(track.point_from_proportion((knob.get_value()+5)/6),color=PURPLE_REG))
        self.add(track,dot,readout(r'\log_{10}\lambda=',knob.get_value,[4.8,-1.1,0],PURPLE_REG,2),
                 readout(r'\|w\|=',lambda:np.linalg.norm(get()),[4.8,1.15,0],PURPLE_REG),
                 readout(r'E_{\rm train}/N=',lambda:.5*model.mse(get()),[4.8,.5,0],BLUE_DATA),
                 readout(r'E_{\rm val}/N_v=',lambda:.5*model.mse(get(),model.XV,model.TV),[4.8,-.15,0],ORANGE_VAL))
        self.label('隠れユニット数10。全重み・バイアスに二乗正則化')
        self.beat(lambda:knob.animate.set_value(-4),lambda:knob.animate.set_value(-3))
        self.beat(lambda:Indicate(data,color=BLUE_DATA,scale_factor=1),lambda:knob.animate.set_value(-2.5))
        self.beat(lambda:knob.animate.set_value(-4),lambda:knob.animate.set_value(-5))
        self.beat(lambda:knob.animate.set_value(1),lambda:knob.animate.set_value(-3))
        self.equation(r'\mathbf w^+=(1-\eta\lambda)\mathbf w-\eta\nabla E(\mathbf w)',32)
        self.label('η：学習率　　紫のつまみと、曲線・誤差・重みが連動',PURPLE_REG)
        self.beat(lambda:knob.animate.set_value(-2),lambda:knob.animate.set_value(-1))
        self.beat(lambda:knob.animate.set_value(-3),lambda:Indicate(self.formula,color=PURPLE_REG))

    def priors(self):
        scale=ValueTracker(1)
        self.equation(r'\widetilde x=ax,\quad \widetilde w=w/a,\quad \widetilde w\widetilde x=wx')
        self.add(readout('a=',scale.get_value,[-4,1,0],BLUE_DATA),readout(r'\widetilde x=',lambda:2*scale.get_value(),[-1.3,1,0],BLUE_DATA),
                 readout(r'\widetilde w=',lambda:3/scale.get_value(),[1.6,1,0],PURPLE_REG),readout(r'\widetilde w\widetilde x=',lambda:6,[4.9,1,0],GREEN_TRUE))
        bar=always_redraw(lambda:Rectangle(width=5/scale.get_value()**2,height=.42,color=PURPLE_REG,fill_opacity=.75).move_to([-4.5,-.3,0],aligned_edge=LEFT))
        self.add(bar,readout(r'\widetilde w^2=',lambda:9/scale.get_value()**2,[2.8,-.3,0],PURPLE_REG))
        self.label('同じ入力の意味、同じ予測。重みの数値だけが変わる')
        self.beat(lambda:scale.animate.set_value(1.5),lambda:scale.animate.set_value(2))
        self.beat(lambda:scale.animate.set_value(1),lambda:scale.animate.set_value(2))
        self.equation(r'\widetilde x=ax+b,\quad \widetilde w=w/a,\quad \widetilde b_0=b_0-wb/a',29)
        self.beat(lambda:Indicate(self.formula,color=BLUE_DATA),lambda:Indicate(bar,color=PURPLE_REG))
        self.remove(*[m for m in self.mobjects if m not in [self.caption,self.formula,self.note] and m.get_center()[1]<2.9])
        alpha=ValueTracker(1);ax,g=axes((-3,3,1),(0,1.3,.5),center=(-3,.1,0),width=5,height=3)
        gaussian=always_redraw(lambda:curve(ax,np.linspace(-3,3,161),np.sqrt(alpha.get_value()/(2*np.pi))*np.exp(-.5*alpha.get_value()*np.linspace(-3,3,161)**2),PURPLE_REG))
        self.add(g,gaussian,readout(r'\alpha=',alpha.get_value,[-3,-1.85,0],PURPLE_REG,2))
        self.equation(r'p(w)\propto e^{-\alpha w^2/2},\qquad \operatorname{Var}(w)=\alpha^{-1}')
        self.beat(lambda:alpha.animate.set_value(2),lambda:alpha.animate.set_value(8))
        a1=ValueTracker(1);a2=ValueTracker(1);rng=np.random.default_rng(5511)
        samples=rng.normal(size=(3,3*12+1));fx=np.linspace(-1,1,121)
        bx,bg=axes((-1,1,1),(-6,6,3),center=(3.35,.15,0),width=5.1,height=3,xlabel='x',ylabel='y')
        colors=[BLUE_DATA,GREEN_TRUE,ORANGE_VAL]
        def sample(j):
            w=samples[j].copy();w[:12]/=np.sqrt(a1.get_value());w[24:36]/=np.sqrt(a2.get_value());return model.predict(w,fx)
        curves=always_redraw(lambda:VGroup(*[curve(bx,fx,sample(j),colors[j]) for j in range(3)]))
        self.add(bg,curves)
        self.label('同じ標準正規乱数を使用。入力側と出力側の精度を別々に操作')
        self.beat(lambda:a2.animate.set_value(16),lambda:a1.animate.set_value(16))
        self.equation(r'p(\mathbf w)\propto\exp\!\left(-\frac12\sum_k\alpha_k\|\mathbf w\|_k^2\right)',31)
        self.beat(lambda:a2.animate.set_value(1),lambda:a1.animate.set_value(1))

    def stopping(self):
        weights,tr,va=model.training_history();tick=ValueTracker(0);best=int(np.argmin(va))
        ax,g=axes((0,4000,1000),(0,.25,.05),center=(0,.15,0),width=10.5,height=3.6,xlabel='t',ylabel='E/N')
        self.add(g)
        def history(vals,color):
            q=tick.get_value();n=max(1,int(q));xx=np.r_[np.arange(n),q]*20;yy=np.r_[vals[:n],np.interp(q,np.arange(len(vals)),vals)]*.5
            return curve(ax,xx,yy,color)
        paths=always_redraw(lambda:VGroup(history(tr,BLUE_DATA),history(va,ORANGE_VAL)))
        marker=always_redraw(lambda:Line(ax.c2p(tick.get_value()*20,0),ax.c2p(tick.get_value()*20,.25),color=YELLOW_TERM))
        self.add(paths,marker);self.equation(r't_* = \arg\min_t E_{\rm validation}(\mathbf w_t)')
        self.label('青：訓練誤差　橙：検証誤差　　自作データで実際に学習')
        self.beat(lambda:tick.animate.set_value(25),lambda:tick.animate.set_value(200))
        bestdot=Dot(ax.c2p(best*20,.5*va[best]),color=ORANGE_VAL,radius=.08)
        self.beat(lambda:FadeIn(bestdot),lambda:tick.animate.set_value(best))
        self.remove(g,paths,marker,bestdot)
        ax,g=axes((0,3.4,1),(0,2.5,1),center=(0,.1,0),width=8,height=3.5,xlabel='w_1',ylabel='w_2')
        self.add(g);theta=np.linspace(0,2*np.pi,181)
        contours=VGroup(*[curve(ax,model.WML[0]+r/np.sqrt(model.H[0])*np.cos(theta),model.WML[1]+r/np.sqrt(model.H[1])*np.sin(theta),MUTED).set_stroke(opacity=.4) for r in [.15,.3,.45]])
        self.equation(r'E=\frac12\sum_i h_i(w_i-w_{{\rm ML},i})^2,\quad (h_1,h_2)=(0.35,3)')
        self.label('ここからは仕組みを見る二次誤差の例。原点から勾配降下')
        self.beat(lambda:Create(contours),lambda:Indicate(contours,scale_factor=1))
        time=ValueTracker(0);point=always_redraw(lambda:Dot(ax.c2p(*model.early_point(time.get_value())),color=YELLOW_TERM))
        trace=always_redraw(lambda:curve(ax,*np.array([model.early_point(t) for t in np.linspace(0,max(.001,time.get_value()),80)]).T,YELLOW_TERM))
        self.add(point,trace)
        self.beat(lambda:time.animate.set_value(2),lambda:time.animate.set_value(7))
        lam=ValueTracker(20);ridge=always_redraw(lambda:Dot(ax.c2p(*model.ridge_point(lam.get_value())),color=PURPLE_REG))
        ridgepath=curve(ax,*np.array([model.ridge_point(l) for l in np.geomspace(40,.01,160)]).T,PURPLE_REG)
        self.add(ridge);self.label('黄：早期終了の軌跡　紫：重み減衰の解　　似ているが一致しない')
        self.beat(lambda:Create(ridgepath),lambda:lam.animate.set_value(.5))
        self.equation(r'\frac{w_i(t)}{w_{{\rm ML},i}}=1-(1-\eta h_i)^t,\quad \frac{w_i(\lambda)}{w_{{\rm ML},i}}=\frac{h_i}{h_i+\lambda}',29)
        self.beat(lambda:time.animate.set_value(15),lambda:lam.animate.set_value(.05))

    def invariance(self):
        digit=pixels(model.DIGIT,.38,(-3,.1,0));self.add(digit)
        output=tex(r'\text{class}=2',40,GREEN_TRUE).move_to([3,.1,0]);self.add(output)
        arrow=Arrow([-1.4,.1,0],[1.7,.1,0],color=MUTED);self.add(arrow)
        self.equation(r'y(s(\mathbf x,\xi))\approx y(\mathbf x)')
        self.label('意味を保つ変換を選ぶ：移動、小さな回転、拡縮')
        self.beat(lambda:digit.animate.shift(RIGHT*.7),lambda:Indicate(output,color=GREEN_TRUE))
        self.beat(lambda:Rotate(digit,.15),lambda:digit.animate.scale(.85))
        copies=VGroup(*[digit.copy().scale(.55).move_to([x,-.1,0]) for x in [-4.5,-2.7,-.9]])
        self.beat(lambda:Transform(digit,copies),lambda:Indicate(output,color=GREEN_TRUE))
        self.remove(digit,arrow,output)
        ax,g=axes((-1,1,1),(0,1,.5),center=(0,.1,0),width=8,height=3.2,xlabel=r'\xi',ylabel='y')
        response=ValueTracker(.38);u=np.linspace(-1,1,121)
        graph=always_redraw(lambda:curve(ax,u,.5+response.get_value()*np.sin(2*u)))
        self.add(g,graph);self.label('二つ目：変換量に対する出力の変化を小さくする',PURPLE_REG)
        self.beat(lambda:response.animate.set_value(.2),lambda:response.animate.set_value(.015))
        self.remove(g,graph);digit=pixels(model.DIGIT,.3,(-3,.1,0));self.add(digit)
        total=tex(r'\sum_i x_i='+str(int(model.DIGIT.sum())),40,GREEN_TRUE).move_to([2.4,.1,0]);self.add(total)
        shuffled=model.DIGIT.ravel()[np.random.default_rng(5).permutation(model.DIGIT.size)].reshape(model.DIGIT.shape)
        self.label('三つ目：総画素値は位置に依存しないが、形の情報を失う')
        self.beat(lambda:digit.animate.shift(RIGHT*.5),lambda:Transform(digit,pixels(shuffled,.3,(-2.5,.1,0))))
        patch=Square(.9,color=YELLOW_TERM).move_to([-2.5,.1,0]);self.add(patch)
        self.label('四つ目：同じ局所検出器を各位置で使う構造')
        self.beat(lambda:patch.animate.shift(UP*.6),lambda:patch.animate.shift(DOWN*1.2))

    def tangent(self):
        angle=ValueTracker(.45);c=ValueTracker(1)
        ax,g=axes((-1.5,1.5,1),(-1.5,1.5,1),center=(-2.65,.0,0),width=4.5,height=3.7,xlabel='x_1',ylabel='x_2');self.add(g)
        th=np.linspace(0,2*np.pi,181);circle=curve(ax,np.cos(th),np.sin(th),BLUE_DATA);self.add(circle)
        x=lambda:np.array([np.cos(angle.get_value()),np.sin(angle.get_value())])
        dot=always_redraw(lambda:Dot(ax.c2p(*x()),color=BLUE_DATA))
        tau=always_redraw(lambda:Arrow(ax.c2p(*x()),ax.c2p(*(x()+.55*model.tangent(x()))),buff=0,color=YELLOW_TERM))
        self.add(dot);self.equation(r'\mathbf s(\mathbf x,\xi)=R(\xi)\mathbf x')
        self.label('回転の軌道を、2次元の入力空間で見る')
        self.beat(lambda:angle.animate.set_value(1.2),lambda:angle.animate.set_value(3.7))
        self.equation(r'\tau=\left.\frac{\partial s}{\partial\xi}\right|_0,\qquad \tau=(-x_2,x_1)^T')
        self.beat(lambda:Create(tau),lambda:angle.animate.set_value(.8))
        self.equation(r'\frac{\partial y}{\partial\xi}=J\tau,\qquad y=x_1^2+x_2^2+c x_1')
        sens=readout(r'J\tau=',lambda:model.sensitivity(x(),c.get_value()),[3.5,.8,0],YELLOW_TERM)
        value=readout('y=',lambda:model.radial(x(),c.get_value()),[3.5,.0,0],RED_MODEL)
        coef=readout('c=',c.get_value,[3.5,-.8,0],PURPLE_REG)
        self.add(sens,value,coef)
        self.beat(lambda:angle.animate.set_value(2.2),lambda:angle.animate.set_value(.8))
        self.beat(lambda:angle.animate.set_value(2.2),lambda:AnimationGroup(c.animate.set_value(0),angle.animate.set_value(4)))
        self.equation(r'\widetilde E=E+\lambda\Omega,\qquad \Omega=\frac12\sum_{n,k}(J_{nk}\tau_n)^2')
        self.label('c = 0：円周上で y = 1、回転方向の微分は0',GREEN_TRUE)
        self.beat(lambda:angle.animate.set_value(5.4),lambda:angle.animate.set_value(.45))
        self.remove(sens,value,coef);eps=ValueTracker(.02);base=np.array([1.,0.])
        approx=always_redraw(lambda:Dot(ax.c2p(*(base+eps.get_value()*model.tangent(base))),color=PURPLE_REG))
        actual=always_redraw(lambda:Dot(ax.c2p(*model.rotate(base,eps.get_value())),color=ORANGE_VAL))
        self.add(approx,actual);self.equation(r'\tau\approx\frac{s(x,\epsilon)-x}{\epsilon},\quad x+\epsilon\tau\approx s(x,\epsilon)')
        self.label('紫：接線近似　橙：真の回転　　局所的な一致')
        self.beat(lambda:eps.animate.set_value(.8),lambda:eps.animate.set_value(.03))

    def augmentation(self):
        eps=ValueTracker(.12);slope=ValueTracker(1.2);swing=ValueTracker(-1)
        ax,g=axes((-1,1,.5),(-.7,2.7,1),center=(-2,.05,0),width=6.5,height=3.5);self.add(g)
        u=np.linspace(-1,1,181)
        graph=always_redraw(lambda:curve(ax,u,slope.get_value()*u+u*u))
        dot=always_redraw(lambda:Dot(ax.c2p(swing.get_value()*eps.get_value(),slope.get_value()*swing.get_value()*eps.get_value()+(swing.get_value()*eps.get_value())**2),color=BLUE_DATA))
        self.add(graph,dot);self.equation(r'\xi\in\{-\epsilon,+\epsilon\},\quad \mathbb E[\xi]=0,\quad\mathbb E[\xi^2]=\epsilon^2')
        self.label('自作例：y(x) = a x + x²、中心 x = 0、目標 t = −0.3')
        self.beat(lambda:swing.animate.set_value(1),lambda:swing.animate.set_value(-1))
        exact=readout(r'\mathbb E[E_\xi]=',lambda:model.noise_example(eps.get_value(),slope.get_value())[0],[3.9,.8,0],ORANGE_VAL,4)
        approx=readout(r'E_{\rm approx}=',lambda:model.noise_example(eps.get_value(),slope.get_value())[1],[3.9,.0,0],PURPLE_REG,4)
        self.add(exact,approx)
        self.beat(lambda:slope.animate.set_value(1.8),lambda:slope.animate.set_value(.3))
        self.equation(r'\mathbb E[E_\xi]\approx\frac{r^2}{2}+\frac{\epsilon^2}{2}\left[(y\prime)^2+r\,y\prime\prime\right]',30)
        self.beat(lambda:slope.animate.set_value(1.2),lambda:eps.animate.set_value(.5))
        self.equation(r'\Omega\simeq\frac12\int(\tau^T\nabla y)^2p(x)\,dx\quad\left[y\approx\mathbb E[t|x]\right]',29)
        self.label('無限データ極限・小さい平均0の変換・条件付き平均に近い解')
        self.beat(lambda:eps.animate.set_value(.2),lambda:eps.animate.set_value(.08))
        self.equation(r'\operatorname{Cov}(\xi)=\lambda I\quad\Rightarrow\quad\Omega=\frac12\int\|\nabla y\|^2p(x)\,dx',29)
        self.label('ティホノフ正則化：全方向への小さな変化を抑える')
        self.beat(lambda:swing.animate.set_value(1),lambda:swing.animate.set_value(-1))
        self.equation(r'\mathbb E[E_\xi]-E_{\rm approx}=\frac12\epsilon^4\quad\text{(this example)}',31)
        self.label('橙：対称な2点での正確な平均　紫：二次近似')
        self.beat(lambda:eps.animate.set_value(.6),lambda:eps.animate.set_value(.04))

    def convolution(self):
        img=model.IMAGE;feat=model.convolution(img);activation=model.sigmoid(feat)
        image=pixels(img,.31,(-4.5,.25,0));feature=pixels(activation,.31,(.7,.25,0));kernel=pixels(model.KERNEL,.33,(-1.8,.25,0),True)
        self.add(image,kernel)
        labels=VGroup(jp('入力 10×10',20).move_to([-4.5,2,0]),jp('共有する重み',20,YELLOW_TERM).move_to([-1.8,1.2,0]),jp('応答 8×8',20).move_to([.7,2,0]))
        self.add(labels);self.label('橙：−1　青：+1　黒：0　　縦の変化を検出する自作フィルタ')
        self.equation(r'z_{r,c}=\sigma\!\left(\sum_{i,j}K_{ij}x_{r+i,c+j}+b\right)',30)
        scan=ValueTracker(0)
        def patchpos():
            k=min(63,int(scan.get_value()));r,c=divmod(k,8)
            return image[r*10+c].get_center()+[.31,-.31,0]
        patch=always_redraw(lambda:Square(.93,color=YELLOW_TERM,stroke_width=3).move_to(patchpos()))
        self.add(patch)
        self.beat(lambda:scan.animate.set_value(7),lambda:scan.animate.set_value(16))
        self.add(feature)
        pointer=always_redraw(lambda:Square(.31,color=YELLOW_TERM,stroke_width=3).move_to(feature[min(63,int(scan.get_value()))]))
        value=readout(r'\sum Kx=',lambda:feat.ravel()[min(63,int(scan.get_value()))],[4.25,.8,0],YELLOW_TERM,1)
        self.add(pointer,value)
        self.beat(lambda:scan.animate.set_value(32),lambda:scan.animate.set_value(63))
        self.label('8×8 箇所で同じ9個の重みと1個のバイアスを使用')
        self.beat(lambda:scan.animate.set_value(40),lambda:scan.animate.set_value(5))
        self.remove(patch,pointer,value)
        shifted=np.roll(img,1,axis=1);sf=model.sigmoid(model.convolution(shifted))
        self.label('入力を右へ1画素 → 応答も右へ1画素（境界の影響を除く）')
        self.beat(lambda:Transform(image,pixels(shifted,.31,(-4.5,.25,0))),lambda:Transform(feature,pixels(sf,.31,(.7,.25,0))))
        pooled=model.subsample(sf);pool=pixels(pooled,.45,(4.7,.25,0))
        self.equation(r'u=\sigma\!\left(a\cdot\frac14\sum_{i,j\in 2\times2}z_{ij}+b\right),\quad a=1,\ b=0',29)
        window=Square(.62,color=GREEN_TRUE).move_to(feature[0].get_center()+[.155,-.155,0]);self.add(window)
        self.label('原文のサブサンプリング：平均 → 学習可能な重みとバイアス → 非線形')
        self.beat(lambda:Create(pool),lambda:window.animate.shift(RIGHT*.62+DOWN*.62))
        self.equation(r'\frac{\partial E}{\partial K_{ij}}=\sum_{r,c}\frac{\partial E}{\partial a_{r,c}}x_{r+i,c+j}',31)
        self.label('平均しても境界をまたぐ移動で値は変わる。重みへの勾配は全位置から合計')
        sf2=model.sigmoid(model.convolution(np.roll(img,2,axis=1)))
        self.beat(lambda:Transform(pool,pixels(model.subsample(sf2),.45,(4.7,.25,0))),lambda:LaggedStart(*[Indicate(kernel[i],color=YELLOW_TERM) for i in range(9)],lag_ratio=.12))

    def soft_sharing(self):
        ax,g=axes((-3,3,1),(0,.55,.2),center=(0,.15,0),width=10.5,height=3.1,xlabel='w',ylabel='p(w)');self.add(g)
        u=np.linspace(-3,3,241);mu=np.array([-1.2,1.2]);sigma=np.array([.6,.6]);pi=np.array([.5,.5])
        comps=model.mixture_components(u);graphs=VGroup(curve(ax,u,comps[:,0],BLUE_DATA),curve(ax,u,comps[:,1],ORANGE_VAL),curve(ax,u,comps.sum(1),PURPLE_REG))
        weights=model.soft_history()[0][0];dots=VGroup(*[Dot(ax.c2p(w,.015),radius=.055,color=YELLOW_TERM) for w in weights]);self.add(dots)
        self.equation(r'p(w_i)=\sum_j\pi_j\mathcal N(w_i|\mu_j,\sigma_j^2)')
        self.label('黄：重み　青・橙：混合成分　紫：合計の密度')
        self.beat(lambda:Create(graphs),lambda:Indicate(dots,color=YELLOW_TERM,scale_factor=1))
        self.equation(r'\Omega=-\sum_i\ln\!\left[\sum_j\pi_j\mathcal N(w_i|\mu_j,\sigma_j^2)\right]',31)
        self.beat(lambda:Indicate(graphs[0],color=BLUE_DATA),lambda:Indicate(graphs[2],color=PURPLE_REG))
        selected=ValueTracker(-.2);point=always_redraw(lambda:Dot(ax.c2p(selected.get_value(),.10),radius=.085,color=RED_MODEL))
        self.add(point);self.equation(r'\gamma_j(w)=\frac{\pi_j\mathcal N(w|\mu_j,\sigma_j^2)}{\sum_k\pi_k\mathcal N(w|\mu_k,\sigma_k^2)}',31)
        self.label('同じ幅・同じ混合比の例。左右の山へ確率的に所属する')
        bars=always_redraw(lambda:VGroup(*[Rectangle(width=max(.006,2*model.responsibilities(selected.get_value())[j]),height=.17,color=color,fill_opacity=.8).move_to([x,1.75,0],aligned_edge=LEFT) for j,x,color in [(0,-5,BLUE_DATA),(1,2,ORANGE_VAL)]]))
        self.add(bars)
        self.beat(lambda:selected.animate.set_value(-.65),lambda:selected.animate.set_value(0))
        self.equation(r'\frac{\partial\Omega}{\partial w_i}=\sum_j\gamma_j(w_i)\frac{w_i-\mu_j}{\sigma_j^2}',32)
        force=always_redraw(lambda:Arrow(ax.c2p(selected.get_value(),.10),ax.c2p(selected.get_value()-.18*model.soft_gradient(selected.get_value()),.10),buff=0,color=RED_MODEL))
        self.add(force)
        self.beat(lambda:selected.animate.set_value(.7),lambda:selected.animate.set_value(-.7))
        self.remove(point,force,bars);rows=model.soft_history();step=ValueTracker(0)
        def update_group():
            w,m,s,p=rows[min(100,int(step.get_value()))];v=model.mixture_components(u,m,s,p)
            return VGroup(curve(ax,u,v[:,0],BLUE_DATA),curve(ax,u,v[:,1],ORANGE_VAL),curve(ax,u,v.sum(1),PURPLE_REG))
        graphs.add_updater(lambda mob:mob.become(update_group()))
        dots.add_updater(lambda mob:[d.move_to(ax.c2p(w,.015)) for d,w in zip(mob,rows[min(100,int(step.get_value()))][0])])
        self.equation(r'\sigma_j^2=e^{\rho_j}>0,\qquad\pi_j=\frac{e^{\eta_j}}{\sum_k e^{\eta_k}}',31)
        self.label('自作の二次データ誤差 + Ω を共同最適化。分散の崩壊は別途注意')
        self.beat(lambda:step.animate.set_value(50),lambda:step.animate.set_value(100))
        graphs.clear_updaters();dots.clear_updaters()
        self.label('重みの大きさ → 学習時刻 → 変換への感度 → 重みのまとまり',GREEN_TRUE)
        self.equation(r'\text{data fit}\quad+\quad\text{prior knowledge}\quad\longrightarrow\quad\text{generalization}',30)
        self.beat(lambda:Indicate(dots,color=YELLOW_TERM,scale_factor=1),lambda:Indicate(graphs,color=GREEN_TRUE,scale_factor=1))
