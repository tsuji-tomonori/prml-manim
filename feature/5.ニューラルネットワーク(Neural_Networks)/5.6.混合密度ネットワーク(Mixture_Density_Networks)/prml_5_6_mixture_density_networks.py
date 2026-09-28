"""PRML 5.6: linked visual experiments in Manim Community, with PCM cue timing."""
from pathlib import Path
import json
import numpy as np
from manim import *
from manim.animation.animation import prepare_animation
from video_support import jp, tex, caption_mobject
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry
from mdn_model import (load_model, parameters, raw_outputs, density, normal,
                       responsibility, robot, forward_function, mean_prediction)

BG='#10141F'
C=[ManimColor('#58B5ED'),ManimColor('#FFB45B'),ManimColor('#B793F5')]
RED=ManimColor('#FF6B77')
GREEN=ManimColor('#7ED5A0')
YELLOW=ManimColor('#FFE079')
MUTED=ManimColor('#A8B2C5')


def curve(ax,x,y,color=WHITE,width=3):
    o=ax.c2p(0,0)
    points=o+np.asarray(x)[:,None]*(ax.c2p(1,0)-o)+np.asarray(y)[:,None]*(ax.c2p(0,1)-o)
    return VMobject().set_points_as_corners(points).set_stroke(color,width)


def axes(xrange=(0,1,.25),yrange=(0,1,.25),width=8,height=3.5,center=(0,-.25,0),xlabel='x',ylabel='t'):
    ax=Axes(x_range=xrange,y_range=yrange,x_length=width,y_length=height,tips=False,
            axis_config={'color':MUTED,'stroke_width':1.4,'include_numbers':True,'font_size':19},
            x_axis_config={'decimal_number_config':{'num_decimal_places':1}},
            y_axis_config={'decimal_number_config':{'num_decimal_places':1}}).move_to(center)
    labels=VGroup(tex(xlabel,24).next_to(ax.x_axis,RIGHT,buff=.13),
                  tex(ylabel,24).next_to(ax.y_axis,UP,buff=.12))
    return ax,labels


def density_axes(getter,width=5.2,height=3.0,center=(3.25,-.25,0)):
    """Keep the density's peak readable; tick values track the changing vertical scale."""
    ax=Axes(x_range=(-.15,1.15,.25),y_range=(0,1,.5),x_length=width,y_length=height,tips=False,
            axis_config={'color':MUTED,'stroke_width':1.4},
            x_axis_config={'include_numbers':True,'font_size':19,'decimal_number_config':{'num_decimal_places':1}},
            y_axis_config={'include_numbers':False}).move_to(center)
    native=ax.c2p
    grid=np.linspace(-.15,1.15,241)
    ymax=lambda:1.2*float(np.max(density(grid,*getter())))
    ax.c2p=lambda x,y=0,z=0:native(x,y/ymax(),z)
    labels=VGroup(tex('t',24).next_to(ax.x_axis,RIGHT,buff=.13),tex('p',24).next_to(ax.y_axis,UP,buff=.12))
    for ratio in [.5,1.]:
        number=DecimalNumber(ratio*ymax(),num_decimal_places=1,font_size=18).next_to(native(0,ratio),LEFT,buff=.1)
        number.add_updater(lambda m,r=ratio:m.set_value(r*ymax()).next_to(native(0,r),LEFT,buff=.1))
        labels.add(number)
    labels.add(jp('縦軸自動',15,MUTED).next_to(labels[1],RIGHT,buff=.25))
    return ax,labels


def readout(label,getter,pos,color=WHITE,places=2,size=25):
    a=tex(label,size,color); b=DecimalNumber(getter(),num_decimal_places=places,font_size=size,color=color)
    g=VGroup(a,b).arrange(RIGHT,buff=.13).move_to(pos)
    anchor=b.get_left().copy()
    b.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return g


def slider(tracker,low,high,pos,width=3.4,label='x',color=YELLOW):
    line=Line(LEFT*width/2,RIGHT*width/2,color=MUTED).move_to(pos)
    dot=Dot(radius=.065,color=color)
    dot.add_updater(lambda m:m.move_to(line.point_from_proportion(np.clip((tracker.get_value()-low)/(high-low),0,1))))
    name=tex(label,26,color).next_to(line,LEFT,buff=.25)
    return VGroup(line,dot,name)


def mix_graph(ax,getter,fill=False):
    ts=np.linspace(ax.x_range[0],ax.x_range[1],241)
    def build():
        p,m,s=getter()
        ys=p[None,:]*normal(ts[:,None],m,s)
        group=VGroup()
        for k in range(len(p)):
            path=curve(ax,ts,ys[:,k],C[k],2.3)
            if fill:
                points=[ax.c2p(ts[0],0),*path.get_points()[::3],ax.c2p(ts[-1],0)]
                area=VMobject().set_points_as_corners(points)
                area.close_path()
                group.add(area.set_fill(C[k],.16).set_stroke(width=0))
            group.add(path)
        group.add(curve(ax,ts,ys.sum(1),WHITE,3.2))
        return group
    return always_redraw(build)


def bars(getter,pos,width=3.6,labels=True):
    def build():
        values=getter(); left=np.array(pos)-RIGHT*width/2
        g=VGroup()
        for k,v in enumerate(values):
            r=Rectangle(width=max(.001,width*float(v)),height=.24,stroke_width=0,fill_color=C[k],fill_opacity=.9)
            r.move_to(left+RIGHT*r.width/2); left+=RIGHT*r.width; g.add(r)
        return g
    g=always_redraw(build)
    if not labels:return g
    nums=VGroup(*[readout(rf'\pi_{k+1}=',lambda k=k:float(getter()[k]),
                        np.array(pos)+DOWN*.45+RIGHT*((k-1)*1.45),C[k],size=21) for k in range(3)])
    return VGroup(g,nums)


class PRML56MixtureDensityNetworks(Scene):
    def construct(self):
        self.camera.background_color=BG
        self.model=load_model(); self.w=self.model['weights'][-1]
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        methods=[self.robot_scene,self.inverse_scene,self.mixture_scene,self.network_scene,
                 self.training_scene,self.responsibility_scene,self.map_scene,self.summaries_scene,self.recap_scene]
        for i,method in enumerate(methods):
            self.begin(i); method()
            assert self.beat_index==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml56_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def begin(self,index):
        self.clear(); self.story=SCENES[index]; self.beat_index=0; self.subtitle=None
        self.add(jp(self.story['title'],32).move_to([0,3.35,0]))
        self.add(jp(self.story['reference'],15,MUTED).move_to([0,2.83,0]))
        entry=self.manifest[self.story['id']]
        if not valid_entry(self.story,entry):raise RuntimeError('Missing or stale narration: '+self.story['id'])
        self.audio_entry=entry; self.durations=entry['beat_durations']
        self.boundaries=np.ceil(np.cumsum(self.durations)*config.frame_rate-1e-6)/config.frame_rate
        self.scene_start=float(self.time)
        self.add_sound(str(OUTPUT_DIR/f"{self.story['id']}.wav"))
        self.timeline.append(dict(id=self.story['id'],title=self.story['title'],reference=self.story['reference'],
                                  start=self.scene_start,audio=True,beats=[]))

    def beat_cues(self):
        offset=sum(self.durations[:self.beat_index])
        return [dict(id=c['id'],display=c['display'],start=c['start']-offset,end=c['end']-offset)
                for c in self.audio_entry['subtitle_cues'] if c['beat_index']==self.beat_index]

    def beat(self,*animations,start_sentence=0):
        if self.subtitle is not None:self.remove(self.subtitle)
        cues=self.beat_cues(); captions=VGroup(*[caption_mobject(c['display']).set_opacity(0) for c in cues])
        self.subtitle=captions; self.add(captions)
        start=float(self.time); fps=config.frame_rate
        frames=round((self.scene_start+self.boundaries[self.beat_index]-start)*fps); duration=frames/fps
        a=cues[start_sentence]['start']; b=cues[-1]['end']
        def caption_at(m,alpha):
            idx=max(i for i,c in enumerate(cues) if c['start']<=alpha*duration+1e-8)
            for i,cap in enumerate(m):cap.set_opacity(1 if i==idx else 0)
        caption_at(captions,0)
        self.timeline[-1]['beats'].append(dict(start=start,end=start+duration,visual=self.story['beats'][self.beat_index]['visual_note'],
            action_start=start+a,action_end=start+b,cues=[dict(c,start=start+c['start'],end=start+c['end']) for c in cues]))
        visual=[]
        if a>0:visual.append(Wait(a))
        prepared=[prepare_animation(a) for a in animations]
        for anim in prepared:
            if isinstance(anim,(Write,FadeIn,FadeOut,Create)) or (isinstance(anim,Transform) and isinstance(anim.mobject,MathTex)):
                anim.set_run_time(min(1.8,(b-a)*.25))
            else:
                anim.set_run_time(b-a)
        visual.append(AnimationGroup(*prepared,Wait(b-a)) if prepared else Wait(b-a))
        visual.append(Wait(max(.001,duration-b)))
        self.play(Succession(*visual),UpdateFromAlphaFunc(captions,caption_at,rate_func=linear),
                  run_time=(frames-1e-5)/fps,rate_func=linear)
        self.beat_index+=1

    def formula(self,s,pos=(0,2.2,0),size=31):
        m=tex(s,size).move_to(pos)
        if m.width>12:m.scale_to_fit_width(12)
        return m

    def note(self,s,color=MUTED):
        return jp(s,22,color).move_to([0,-2.65,0])

    def p(self,x,w=None):
        return tuple(a[0] for a in parameters(self.w if w is None else w,[x]))

    def arm(self,a,b,color,origin=(-3,-.4,0),scale=2):
        e,t=robot(a,b); o=np.array(origin)
        e=o+scale*np.r_[e,0];t=o+scale*np.r_[t,0]
        return VGroup(Line(o,e,color=color,stroke_width=7),Line(e,t,color=color,stroke_width=7),
                      Dot(o,color=WHITE),Dot(e,color=color),Dot(t,color=color))

    def robot_scene(self):
        q=ValueTracker(-.7); r=ValueTracker(.2)
        arm=always_redraw(lambda:self.arm(q.get_value(),r.get_value(),C[0]))
        target=Dot([-.6,-.4,0],color=YELLOW,radius=.11)
        angle=np.arccos(.6)
        self.add(target,arm,self.note('棒の長さ：1 ＋ 1　　目標までの距離：1.2'))
        self.beat(q.animate.set_value(angle),r.animate.set_value(-2*angle))
        ghost=self.arm(angle,-2*angle,C[0]).set_opacity(.3)
        self.add(ghost)
        self.beat(q.animate.set_value(-angle),r.animate.set_value(2*angle))
        f=self.formula(r'(\theta_1,\theta_2)\ \longrightarrow\ (x_1,x_2)')
        self.beat(Write(f),Indicate(target))
        red=always_redraw(lambda:self.arm(q.get_value(),r.get_value(),RED))
        self.remove(arm);self.add(red)
        inv=self.formula(r'(x_1,x_2)\ \longrightarrow\ \{\mathrm{A},\mathrm{B}\}')
        self.beat(Transform(f,inv),q.animate.set_value(0),r.animate.set_value(0))
        error=Line(target.get_center(),[1,-.4,0],color=RED,stroke_width=5)
        label=readout(r'\|\mathrm{error}\|=',lambda:.8,[3,1,0],RED)
        self.beat(Create(error),FadeIn(label),Indicate(red))
        other=self.arm(-angle,2*angle,C[1]);self.remove(red)
        self.beat(FadeOut(error),FadeOut(label),FadeIn(other),ghost.animate.set_opacity(1))

    def inverse_scene(self):
        ax,lab=axes(width=8.6,height=3.8,xlabel='u',ylabel='v')
        self.add(ax,lab)
        x,t=self.model['x'][::3],self.model['t'][::3]
        dots=VGroup(*[Dot(ax.c2p(u,v),radius=.032,color=C[0]) for u,v in zip(t,x)])
        u=np.linspace(0,1,241);f=curve(ax,u,forward_function(u),GREEN)
        self.add(self.note('自作データ：480点のうち160点を表示'))
        self.beat(FadeIn(dots),Create(f))
        newdots=VGroup(*[Dot(ax.c2p(v,u),radius=.032,color=C[0]) for u,v in zip(t,x)])
        newlab=VGroup(tex('x',24).move_to(lab[0]),tex('t',24).move_to(lab[1]))
        self.beat(Transform(dots,newdots),FadeOut(f),Transform(lab,newlab))
        xv=ValueTracker(.14)
        scan=always_redraw(lambda:Line(ax.c2p(xv.get_value(),0),ax.c2p(xv.get_value(),1),color=YELLOW))
        self.add(scan,readout('x=',xv.get_value,[4.9,2,0],YELLOW))
        self.beat(xv.animate.set_value(.5))
        mean=curve(ax,u,mean_prediction(self.model['mean_weights'],u),RED,4)
        formula=self.formula(r'\min_y\ \mathbb{E}[(t-y)^2\mid x]\quad\Rightarrow\quad y=\mathbb{E}[t\mid x]',size=28)
        self.beat(Create(mean),Write(formula))
        _,mus,_=self.p(.5)
        marks=VGroup(*[Circle(radius=.18,color=c).move_to(ax.c2p(.5,mu)) for c,mu in zip(C,mus)])
        self.beat(Create(marks),mean.animate.set_stroke(opacity=.3))
        eq=self.formula(r'p(t\mid x)\quad\text{?}',size=38)
        self.beat(Transform(formula,eq),xv.animate.set_value(.62))

    def mixture_scene(self):
        ax,lab=axes(xrange=(-.2,1.2,.2),yrange=(0,5,1),width=9,height=2.9,center=(0,-.1,0),xlabel='t',ylabel='p')
        mu=ValueTracker(.45); sig=ValueTracker(.09); blend=ValueTracker(0); gate=ValueTracker(0)
        def pars():
            b=blend.get_value(); g=gate.get_value()
            return np.array([1-b+b*(.25+.45*g),b*.4*(1-g),b*(.35-.05*g)]),np.array([mu.get_value(),.5,.82]),np.array([sig.get_value(),.10,.075])
        # First single Gaussian peak needs a 5.0-high axis (peak 4.43).
        graph=mix_graph(ax,pars,True)
        self.add(ax,lab)
        self.beat(FadeIn(graph),Write(self.note('確率 ＝ 区間に対応する面積',GREEN)))
        self.add(slider(mu,.15,.85,(-3,-2.3,0),label=r'\mu',color=C[1]),readout(r'\mu=',mu.get_value,[3,-2.3,0],C[1]))
        self.beat(mu.animate.set_value(.22))
        self.beat(sig.animate.set_value(.16))
        self.beat(blend.animate.set_value(1),sig.animate.set_value(.095))
        prob=bars(lambda:pars()[0],[0,2.15,0],width=5)
        self.add(prob)
        self.beat(gate.animate.set_value(1))
        self.remove(prob)
        eq=self.formula(r'p(t\mid x)=\sum_{k=1}^{K}\pi_k(x)\,\mathcal{N}\!\left(t\mid\mu_k(x),\sigma_k^2(x)\right)',size=34)
        eq.set_color_by_tex('p',WHITE)
        self.beat(Write(eq),gate.animate.set_value(0))

    def network(self,xv,origin=(-4,0,0)):
        ox,oy,_=origin
        inp=Circle(radius=.19,color=YELLOW).move_to([ox-1.5,oy,0])
        hs=VGroup(*[Circle(radius=.13,color=C[0]).move_to([ox,oy+1.4-j*.4,0]) for j in range(8)])
        outs=VGroup(*[Circle(radius=.105,color=C[j//3]).move_to([ox+1.65,oy+1.6-j*.4,0]) for j in range(9)])
        edges=VGroup(*[Line(inp.get_right(),h.get_left(),color=MUTED,stroke_width=.7) for h in hs],
                     *[Line(h.get_right(),o.get_left(),color=C[k//3],stroke_width=.6,stroke_opacity=.28) for h in hs for k,o in enumerate(outs)])
        for j,h in enumerate(hs):
            h.add_updater(lambda m,j=j:m.set_fill(C[0],float((raw_outputs(self.w,[xv.get_value()])[1][0,j]+1)/2)))
        for j,o in enumerate(outs):
            o.add_updater(lambda m,j=j:m.set_fill(C[j//3],float(.2+.65/(1+np.exp(-raw_outputs(self.w,[xv.get_value()])[0][0,j])))))
        labels=VGroup(*[tex(s,23,c).move_to([ox+2.25,oy+y,0]) for s,c,y in [(r'a^\pi',C[0],1.2),(r'a^\mu',C[1],0),(r'a^\sigma',C[2],-1.2)]])
        return VGroup(edges,inp,hs,outs,labels)

    def network_scene(self):
        xv=ValueTracker(.15); net=self.network(xv,origin=(-3.6,0,0))
        ax,lab=density_axes(lambda:self.p(xv.get_value()))
        graph=mix_graph(ax,lambda:self.p(xv.get_value()),True)
        self.add(slider(xv,.12,.88,(-3.7,-2.2,0),label='x'),ax,lab,graph)
        self.add(self.note('隠れ層16個のうち8個を表示　／　学習例の幅には下限0.012を加算'))
        self.add(net)
        self.beat(xv.animate.set_value(.4))
        count=self.formula(r'K=3,\quad 3K=9',size=32)
        self.beat(Write(count),Indicate(net[3]))
        soft=self.formula(r'\pi_k={e^{a_k^\pi}\over\sum_l e^{a_l^\pi}},\quad\sum_k\pi_k=1',size=30).set_color(C[0])
        self.beat(Transform(count,soft),xv.animate.set_value(.6))
        sigma=self.formula(r'\sigma_k=e^{a_k^\sigma}>0,\qquad \mathrm{Var}_k=\sigma_k^2',size=31).set_color(C[2])
        self.beat(Transform(count,sigma),Indicate(net[3][6:]))
        mean=self.formula(r'\mu_k=a_k^\mu',size=36).set_color(C[1])
        self.beat(Transform(count,mean),xv.animate.set_value(.85))
        self.beat(xv.animate.set_value(.15),Transform(count,self.formula(r'x\ \longrightarrow\ (\pi_k,\mu_k,\sigma_k)\ \longrightarrow\ p(t\mid x)')))

    def training_scene(self):
        ax,lab=axes(xrange=(-.15,1.15,.25),yrange=(0,7,2),width=8.7,height=3.0,center=(0,-.2,0),xlabel='t',ylabel='p')
        mu=ValueTracker(.30);obs=.62
        pars=lambda:(np.array([.45,.35,.2]),np.array([mu.get_value(),.83,.13]),np.array([.085,.1,.09]))
        graph=mix_graph(ax,pars)
        guide=always_redraw(lambda:Line(ax.c2p(obs,0),ax.c2p(obs,float(density(obs,*pars()))),color=YELLOW,stroke_width=4))
        self.add(ax,lab,graph,Dot(ax.c2p(obs,0),color=YELLOW))
        self.beat(Create(guide))
        val=readout(r'p(t_n\mid x_n)=',lambda:float(density(obs,*pars())),[0,2.08,0],YELLOW)
        self.add(val)
        self.beat(mu.animate.set_value(.59))
        self.remove(val)
        nll=readout(r'-\ln p(t_n\mid x_n)=',lambda:-float(np.log(density(obs,*pars()))),[0,2.08,0],RED)
        self.add(nll)
        self.beat(mu.animate.set_value(.38))
        self.remove(nll)
        eq=self.formula(r'E(w)=-\sum_{n=1}^{N}\ln\left[\sum_{k=1}^{K}\pi_k(x_n,w)\mathcal{N}(t_n\mid\mu_k,\sigma_k^2)\right]',size=30)
        self.beat(Write(eq),mu.animate.set_value(.62))
        self.remove(graph,guide)
        step=ValueTracker(0)
        def w_now():
            z=np.clip(step.get_value(),0,100);i=min(int(z),99)
            return (1-(z-i))*self.model['weights'][i]+(z-i)*self.model['weights'][i+1]
        graph=mix_graph(ax,lambda:self.p(.5,w_now()))
        self.add(graph,self.note('実測した50更新ごとの重みを補間　／　断面 x = 0.5'))
        counter=readout(r'\mathrm{step}=',lambda:50*step.get_value(),[-3,-2.15,0],YELLOW,0)
        loss=readout(r'E/N=',lambda:float(np.interp(step.get_value(),np.arange(101),self.model['losses'])),[3,-2.15,0],RED)
        self.add(counter,loss)
        self.beat(step.animate.set_value(100))
        floor=self.formula(r'\sigma_k=0.012+e^{a_k^\sigma}\quad\text{(experiment)}',size=31)
        self.beat(Transform(eq,floor),Indicate(graph))

    def responsibility_scene(self):
        p=np.array([.28,.42,.30]); mu=np.array([.2,.5,.8]);sig=np.array([.13,.11,.12])
        tv=ValueTracker(.18)
        ax,lab=axes(xrange=(-.15,1.15,.25),yrange=(0,2.1,.5),width=7,height=3.0,center=(-2,-.15,0),xlabel='t',ylabel='p')
        graph=mix_graph(ax,lambda:(p,mu,sig))
        guide=always_redraw(lambda:Line(ax.c2p(tv.get_value(),0),ax.c2p(tv.get_value(),2),color=YELLOW))
        self.add(ax,lab,graph,guide)
        self.beat(tv.animate.set_value(.75))
        gamma=lambda:responsibility(tv.get_value(),p,mu,sig)
        bs=bars(gamma,[3.8,.6,0],width=3.1,labels=False)
        nums=VGroup(*[readout(rf'\gamma_{k+1}=',lambda k=k:float(gamma()[k]),[3.8,-k*.45,0],C[k]) for k in range(3)])
        eq=self.formula(r'\gamma_k={\pi_k\mathcal{N}_k\over\sum_l\pi_l\mathcal{N}_l}',size=32)
        self.add(bs,nums)
        self.beat(Write(eq),tv.animate.set_value(.46))
        prior=jp('観測前',21,MUTED).move_to([3.8,1.65,0]);post=jp('観測後',21,YELLOW).move_to([3.8,1,0])
        priorbar=bars(lambda:p,[3.8,1.35,0],width=3.1,labels=False)
        self.beat(FadeIn(priorbar),Write(prior),Write(post),tv.animate.set_value(.3))
        self.beat(Transform(eq,self.formula(r'{\partial E_n\over\partial a_k^\pi}=\pi_k-\gamma_k')),tv.animate.set_value(.78))
        derivatives=VGroup(tex(r'{\partial E_n\over\partial a_k^\mu}=\gamma_k{\mu_k-t_n\over\sigma_k^2}',28,C[1]),
                           tex(r'{\partial E_n\over\partial a_k^\sigma}=\gamma_k\left(1-{(t_n-\mu_k)^2\over\sigma_k^2}\right)',28,C[2])).arrange(RIGHT,buff=.8).move_to([0,2.08,0])
        self.beat(Transform(eq,derivatives),tv.animate.set_value(.57))
        arrow=Arrow([4,-2.2,0],[-4,-2.2,0],color=YELLOW)
        self.beat(GrowArrow(arrow),Write(self.note('出力の勾配 → 共有する隠れ層 → 重みの更新',YELLOW)))

    def map_scene(self):
        ax,lab=axes(width=6,height=3.4,center=(-3,-.2,0))
        xs=np.linspace(0,1,260); ts=np.linspace(1,0,220)
        p,m,s=parameters(self.w,xs)
        z=np.sum(p[None,:,:]*normal(ts[:,None,None],m[None,:,:],s[None,:,:]),axis=2)
        # Fixed linear color scale; top saturated at density 14, legend explicit.
        strength=np.clip(z/14,0,1)
        bg=np.array([16,20,31]); fg=np.array([110,218,190])
        pixels=(bg+(fg-bg)*strength[:,:,None]).astype(np.uint8)
        im=ImageMobject(pixels).stretch_to_fit_width(ax.x_length).stretch_to_fit_height(ax.y_length).move_to(ax.c2p(.5,.5))
        dots=VGroup(*[Dot(ax.c2p(x,t),radius=.021,color=C[0]) for x,t in zip(self.model['x'][::4],self.model['t'][::4])])
        self.add(im,ax,lab)
        self.beat(FadeIn(dots),Write(self.note('密度：暗 0 → 明 14以上　／　学習例')))
        xv=ValueTracker(.12)
        scan=always_redraw(lambda:Line(ax.c2p(xv.get_value(),0),ax.c2p(xv.get_value(),1),color=YELLOW))
        da,dl=density_axes(lambda:self.p(xv.get_value()),width=4.7,height=3.0,center=(3.8,-.25,0))
        graph=mix_graph(da,lambda:self.p(xv.get_value()),True)
        self.add(scan,da,dl,graph,slider(xv,.12,.88,(-3,-2.3,0),width=4,label='x'),readout('x=',xv.get_value,[3.7,-2.35,0],YELLOW))
        self.beat(xv.animate.set_value(.18))
        self.beat(xv.animate.set_value(.5))
        self.beat(xv.animate.set_value(.88))
        means=VGroup(*[curve(ax,xs[(p[:,k]>.02)&(m[:,k]>=0)&(m[:,k]<=1)],m[:,k][(p[:,k]>.02)&(m[:,k]>=0)&(m[:,k]<=1)],C[k],2) for k in range(3)])
        self.beat(Create(means),xv.animate.set_value(.5))
        self.beat(xv.animate.set_value(.25),Write(self.formula(r'\pi_k(x),\quad\mu_k(x),\quad\sigma_k(x)',size=31)))

    def summaries_scene(self):
        ax,lab=axes(xrange=(-.5,1.5,.5),yrange=(0,4,1),width=9,height=2.9,center=(0,-.1,0),xlabel='t',ylabel='p')
        sep=ValueTracker(.18);width=ValueTracker(.12);weight=ValueTracker(.5);second=ValueTracker(.12)
        def pars():return np.array([weight.get_value(),1-weight.get_value()]),np.array([.5-sep.get_value(),.5+sep.get_value()]),np.array([width.get_value(),second.get_value()])
        def mean():p,m,s=pars();return float(p@m)
        def within():p,m,s=pars();return float(p@(s*s))
        def between():p,m,s=pars();return float(p@((m-mean())**2))
        graph=mix_graph(ax,pars)
        meanline=always_redraw(lambda:Line(ax.c2p(mean(),0),ax.c2p(mean(),3.4),color=RED))
        eq=self.formula(r'm=\mathbb{E}[t\mid x]=\sum_k\pi_k\mu_k',size=34)
        self.add(ax,lab,graph)
        self.beat(Write(eq),Create(meanline))
        self.beat(sep.animate.set_value(.4))
        var=VGroup(tex(r's^2=',30),tex(r'\sum_k\pi_k\underbrace{\sigma_k^2}_{\text{within}}',30,C[0]),tex('+',30),tex(r'\sum_k\pi_k\underbrace{(\mu_k-m)^2}_{\text{between}}',30,C[1])).arrange(RIGHT,buff=.15).move_to([0,2.2,0])
        nums=VGroup(readout(r'\mathrm{within}=',within,[-3,-2.2,0],C[0],3),readout(r'\mathrm{between}=',between,[3,-2.3,0],C[1],3))
        self.beat(Transform(eq,var),FadeIn(nums))
        self.beat(sep.animate.set_value(.46),width.animate.set_value(.2),second.animate.set_value(.2))
        self.remove(nums,meanline)
        approx=always_redraw(lambda:Line(ax.c2p(pars()[1][np.argmax(pars()[0])],0),ax.c2p(pars()[1][np.argmax(pars()[0])],3.2),color=C[1]))
        grid=np.linspace(-.5,1.5,4001)
        mode=lambda:float(grid[np.argmax(density(grid,*pars()))])
        modepoint=always_redraw(lambda:Dot(ax.c2p(mode(),float(density(mode(),*pars()))),color=YELLOW,radius=.08))
        self.add(approx,modepoint,self.note('橙の線：最大の混合係数の中心　／　黄色の点：数値的なモード',YELLOW))
        self.beat(Transform(eq,self.formula(r'\mathrm{mode}=\arg\max_t p(t\mid x),\qquad \widetilde t=\mu_{\arg\max_k\pi_k}',size=30)),weight.animate.set_value(.65))
        self.beat(second.animate.set_value(.055),width.animate.set_value(.22))

    def recap_scene(self):
        a=np.arccos(.6)
        upper=self.arm(a,-2*a,C[0],origin=(-4.4,0,0),scale=1.5)
        lower=self.arm(-a,2*a,C[1],origin=(-4.4,0,0),scale=1.5)
        self.add(Dot([-2.6,0,0],color=YELLOW))
        self.beat(FadeIn(upper),FadeIn(lower))
        xv=ValueTracker(.5)
        ax,lab=density_axes(lambda:self.p(xv.get_value()),width=5.4,height=3.0,center=(3,-.25,0))
        graph=mix_graph(ax,lambda:self.p(xv.get_value()),True)
        eq=self.formula(r'x\longrightarrow (\pi,\mu,\sigma)\longrightarrow p(t\mid x)')
        self.beat(Create(ax),FadeIn(lab),FadeIn(graph),Write(eq))
        self.beat(Transform(eq,self.formula(r'E=-\sum_n\ln p(t_n\mid x_n),\qquad\gamma_k\longrightarrow\nabla_w E')),xv.animate.set_value(.6))
        dimensions=self.formula(r't\in\mathbb{R}^{D}:\quad K(D+2)\ \text{outputs (isotropic)}',size=30)
        self.beat(Transform(eq,dimensions),xv.animate.set_value(.3))
        distinction=self.formula(r'p(t\mid x,w_{\mathrm{ML}})\qquad\qquad p(w\mid\mathcal{D})',size=34)
        self.beat(Transform(eq,distinction),xv.animate.set_value(.85))
        self.beat(Transform(eq,self.formula(r'x\quad\longrightarrow\quad p(t\mid x)',size=36)),xv.animate.set_value(.5),Indicate(upper),Indicate(lower))
