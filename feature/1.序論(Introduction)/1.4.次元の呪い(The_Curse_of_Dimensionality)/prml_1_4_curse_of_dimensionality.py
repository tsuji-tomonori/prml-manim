"""PRML 1.4 — linked visual experiments in Manim Community."""
from pathlib import Path
import json
import numpy as np
from manim import *
from manim.animation.animation import prepare_animation
from caption_layout import jp, tex, caption_mobject
from dimension_model import *
from narration_content import SCENES, estimated_duration
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry

BG = "#10141F"
BLUE = ManimColor("#58B5ED")
GREEN = ManimColor("#77D49A")
RED = ManimColor("#FF6B77")
YELLOW = ManimColor("#FFE079")
PURPLE = ManimColor("#C29AFF")
MUTED = ManimColor("#A8B2C5")


def line_graph(ax, x, y, color=BLUE, fill=False):
    points = [ax.c2p(a,b) for a,b in zip(x,y)]
    if fill:
        points = [ax.c2p(x[0],0), *points, ax.c2p(x[-1],0), ax.c2p(x[0],0)]
    m=VMobject().set_points_as_corners(points).set_stroke(color, 3 if not fill else 0)
    if fill: m.set_fill(color,.35)
    return m


def meter(label, getter, pos, color=YELLOW, places=2):
    prefix=tex(label,29,color)
    number=DecimalNumber(getter(),num_decimal_places=places,font_size=29,color=color,group_with_commas=True)
    g=VGroup(prefix,number).arrange(RIGHT,buff=.15).move_to(pos)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return g


def pace_visual(animation, duration):
    """Finish text reveals promptly while tracker changes use the speech clock."""
    animation = prepare_animation(animation)
    if isinstance(animation, (FadeIn, FadeOut, Create, Write, ReplacementTransform,
                              TransformMatchingTex, TransformFromCopy, LaggedStart)):
        short = min(1.0, duration * .3)
        animation.set_run_time(short)
        return Succession(animation, Wait(max(.001, duration-short)))
    if isinstance(animation, AnimationGroup):
        return AnimationGroup(*[pace_visual(a, duration) for a in animation.animations],
                              run_time=duration)
    animation.set_run_time(duration)
    return animation


class PRML14CurseOfDimensionality(Scene):
    def construct(self):
        self.camera.background_color=BG
        self.timeline=[]
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        for i,method in enumerate([self.classify,self.grid,self.polynomial,self.sphere,self.shells,self.gaussian,self.concentration,self.manifold]):
            self.begin(i)
            method()
            assert self.beat_index==len(self.story['beats']), self.story['id']
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml14_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def begin(self, index):
        self.clear()
        self.story = SCENES[index]
        self.beat_index = 0
        self.subtitle = None
        self.next_section(self.story['id'])
        title = jp(self.story['title'], 34).move_to([0, 3.35, 0])
        self.add(title)
        entry = self.manifest.get(self.story['id'], {})
        audio_valid = valid_entry(self.story, entry)
        if not audio_valid:
            raise RuntimeError("Generate matching narration before rendering")
        self.audio_entry = entry
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
                self.play(pace_visual(animation, phase_duration), UpdateFromAlphaFunc(captions,
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
            visual.append(AnimationGroup(*[pace_visual(a, action_end-action_start) for a in animations], run_time=action_end-action_start))
            visual.append(Wait(max(.001, duration-action_end)))
        else:
            visual.append(Wait(duration))
        self.play(Succession(*visual), UpdateFromAlphaFunc(captions, caption_at, rate_func=linear),
                  run_time=(frames - 1e-5) / fps, rate_func=linear)
        self.beat_index += 1

    def graph_axes(self, xr, yr, center=(0,0,0), width=9, height=3.5, xlabel='r', ylabel='p(r)'):
        ax=Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
                axis_config={'color':MUTED,'stroke_width':1.3,'include_numbers':True,'font_size':18})
        ax.move_to(center)
        labs=VGroup(tex(xlabel,24).next_to(ax.x_axis,RIGHT,buff=.15),tex(ylabel,24).next_to(ax.y_axis,UP,buff=.16))
        return ax,VGroup(ax,labs)

    def slider(self,t,low,high,pos,width=5,label='D',ticks=None,color=PURPLE):
        line=NumberLine(x_range=[low,high,1],length=width,include_ticks=False,color=MUTED).move_to(pos)
        group=VGroup(line,tex(label,26,color).next_to(line,LEFT,buff=.3))
        for v in ticks or [low,high]:
            pt=line.n2p(v)
            group.add(Line(pt+UP*.06,pt+DOWN*.06,color=MUTED),tex(str(v),18,MUTED).next_to(pt,DOWN,buff=.13))
        dot=Dot(radius=.075,color=color)
        dot.add_updater(lambda m:m.move_to(line.n2p(t.get_value())))
        group.add(dot)
        return group

    def two(self,a,b):
        """One visual action per sentence, from measured PCM boundaries."""
        self.beat(phases=[('sentence 1',self.sentence_duration(0),a),('sentence 2',self.sentence_duration(1),b)])

    def classify(self):
        ax,axes=self.graph_axes([0,1,.25],[0,1,.25],center=(-1.7,0,0),width=5,height=3.8,xlabel='x_6',ylabel='x_7')
        pts,labels=class_data(); colors=[RED,GREEN,BLUE]
        dots=VGroup(*[Dot(ax.c2p(*xy),radius=.045,color=colors[c]) for xy,c in zip(pts,labels)])
        query=Cross(stroke_color=WHITE,stroke_width=3).scale(.07).move_to(ax.c2p(.43,.56))
        question=jp('?',62).move_to([3.4,.5,0])
        self.add(axes,dots,query)
        self.beat(FadeIn(question),Indicate(query,scale_factor=1.8))
        legend=VGroup(*[VGroup(Dot(color=c,radius=.07),jp(n,24,c)).arrange(RIGHT,buff=.2) for n,c in zip(['種類 A','種類 B','種類 C'],colors)]).arrange(DOWN,buff=.3).move_to([3.5,.2,0])
        self.beat(FadeOut(question),FadeIn(legend),Indicate(axes, scale_factor=1))
        note=jp('分類の仕組みを見る自作データ',23,MUTED).move_to([0,2.65,0])
        self.beat(FadeIn(note),LaggedStart(*[Indicate(d, scale_factor=1) for d in dots[::10]],lag_ratio=.12))
        cell=Polygon(ax.c2p(.2,.4),ax.c2p(.6,.4),ax.c2p(.6,.8),ax.c2p(.2,.8),color=YELLOW,fill_opacity=.07)
        inside=np.all((pts>=[.2,.4])&(pts<[.6,.8]),axis=1)
        counts=np.bincount(labels[inside],minlength=3)
        count=jp('箱の中  '+ ' / '.join(f'{n}: {v}' for n,v in zip('ABC',counts)),23,YELLOW).move_to([0,-2.4,0])
        self.two(lambda:Create(cell),lambda:AnimationGroup(FadeIn(count),*[Indicate(d, scale_factor=1) for d,v in zip(dots,inside) if v]))
        answer=jp('予測：種類 A',29,RED).move_to([3.5,-1.45,0])
        self.beat(FadeIn(answer),query.animate.set_color(RED))
        vec=tex(r'\mathbf{x}=(x_1,x_2,\ldots,x_{12})',34,PURPLE).move_to([0,2.6,0])
        self.remove(note)
        self.beat(FadeIn(vec))
        hidden=VGroup(*[Line([2.1+i*.27,-2.2,0],[2.1+i*.27,-.8-.8*(i%3)/2,0],color=PURPLE,stroke_width=5) for i in range(10)])
        self.remove(answer)
        self.beat(FadeIn(hidden),Indicate(vec, scale_factor=1))

    def grid(self):
        row=VGroup(*[Square(.66,color=BLUE,fill_opacity=.16).move_to([(i-2)*.68,.5,0]) for i in range(5)])
        equation=tex('5',44,BLUE).move_to([0,2.3,0])
        self.beat(LaggedStart(*[Create(c) for c in row],lag_ratio=.12),FadeIn(equation))
        plane=VGroup(*[Square(.66,color=BLUE,fill_opacity=.13).move_to([(i-2)*.68,(j-2)*.68,0]) for i in range(5) for j in range(5)])
        eq2=tex(r'5\times5=25',44,BLUE).move_to(equation)
        self.beat(ReplacementTransform(row,plane),TransformMatchingTex(equation,eq2))
        cube=VGroup(*[Square(.45,color=interpolate_color(BLUE,PURPLE,k/4),fill_opacity=.03).move_to([(i-2)*.47+.27*k,(j-2)*.47+.17*k-.3,0]) for k in range(5) for i in range(5) for j in range(5)])
        eq3=tex(r'5\times5\times5=125',42,BLUE).move_to(eq2)
        self.beat(ReplacementTransform(plane,cube),TransformMatchingTex(eq2,eq3))
        self.remove(cube,eq3)
        d=ValueTracker(3); val=lambda:int(np.clip(np.round(d.get_value()),1,10))
        ax,axes=self.graph_axes([1,10,1],[0,7,1],center=(-.6,.1,0),width=8.3,height=3.5,xlabel='D',ylabel=r'\log_{10}(5^D)')
        curve=line_graph(ax,np.arange(1,11),np.log10(5)*np.arange(1,11),BLUE)
        dot=always_redraw(lambda:Dot(ax.c2p(val(),np.log10(cell_count(val()))),color=YELLOW,radius=.09))
        counter=meter(r'5^D=',lambda:cell_count(val()),[2.8,2.5,0],BLUE,0)
        dlabel=meter('D=',val,[-3.5,2.5,0],PURPLE,0)
        slider=self.slider(d,1,10,[0,-2.4,0],ticks=[1,2,3,6,10])
        self.add(axes,curve,dot,counter,dlabel,slider)
        self.two(lambda:d.animate.set_value(6),lambda:d.animate.set_value(10))
        upper=tex(r'\text{occupied fraction}\leq\frac{1000}{5^D}',29,YELLOW).move_to([0,.8,0])
        self.remove(curve,dot,axes)
        bar=Rectangle(width=9,height=.32,color=MUTED).move_to([0,-.3,0])
        fill=Rectangle(width=9*1000/cell_count(10),height=.32,color=BLUE,fill_opacity=1).align_to(bar,LEFT).set_y(-.3)
        info=jp('1000点でも、埋められる箱は最大1000個',26).move_to([0,-1.1,0])
        self.beat(FadeIn(upper),Create(bar),FadeIn(fill),FadeIn(info))
        exact=tex(r'\mathbb{E}[\text{empty fraction}]=\left(1-\frac1{5^D}\right)^{1000}',34,YELLOW).move_to([0,.8,0])
        empty=meter(r'\text{empty }(\%)=',lambda:100*empty_fraction(val()),[0,-.25,0],YELLOW,4)
        self.beat(ReplacementTransform(upper,exact),FadeOut(bar),FadeOut(fill),FadeIn(empty))
        self.remove(exact,empty,info)
        self.add(axes,curve,dot)
        self.two(lambda:d.animate.set_value(2),lambda:d.animate.set_value(10))

    def polynomial(self):
        single=MathTex('w_0','+w_1x','+w_2x^2','+w_3x^3',font_size=43).move_to([0,1.7,0])
        for m,c in zip(single,[WHITE,BLUE,YELLOW,PURPLE]):m.set_color(c)
        knobs=VGroup(*[Line([-3+i*2,-1.7,0],[-3+i*2,.1,0],color=MUTED) for i in range(4)])
        knobs.add(*[Dot([-3+i*2,-.8,0],color=c) for i,c in enumerate([WHITE,BLUE,YELLOW,PURPLE])])
        self.beat(FadeIn(single),Create(knobs))
        terms=VGroup(*[tex(s,30,c) for s,c in [('1',WHITE),('x_1',BLUE),('x_2',BLUE),('x_1^2',YELLOW),('x_1x_2',YELLOW),('x_2^2',YELLOW),('x_1^3',PURPLE),('x_1^2x_2',PURPLE),('x_1x_2^2',PURPLE),('x_2^3',PURPLE)]]).arrange_in_grid(rows=2,cols=5,buff=(.7,.8)).move_to([0,.7,0])
        self.beat(FadeOut(knobs),ReplacementTransform(single,terms))
        self.remove(terms)
        c=ValueTracker(-.7)
        def point(u,v):return np.array([2.4*u+.85*v,-.15+.75*v+1.3*c.get_value()*u*v,0])
        def surface():
            lines=VGroup()
            for u in np.linspace(-1,1,9):
                lines.add(VMobject().set_points_as_corners([point(u,v) for v in np.linspace(-1,1,25)]).set_stroke(YELLOW,1.6))
            for v in np.linspace(-1,1,9):
                lines.add(VMobject().set_points_as_corners([point(u,v) for u in np.linspace(-1,1,25)]).set_stroke(BLUE,1.6))
            return lines
        mesh=always_redraw(surface)
        interaction=tex(r'y=c\,x_1x_2',40,YELLOW).move_to([0,2.3,0])
        sl=self.slider(c,-1,1,[0,-2.3,0],label='c',color=YELLOW)
        self.add(mesh,interaction,sl)
        self.two(lambda:c.animate.set_value(.7),lambda:c.animate.set_value(-.7))
        self.remove(mesh,interaction,sl)
        eq=MathTex(r'y(\mathbf{x},\mathbf{w})=',r'w_0',r'+\sum_{i=1}^Dw_ix_i',r'+\sum_{i,j=1}^Dw_{ij}x_ix_j',r'+\sum_{i,j,k=1}^Dw_{ijk}x_ix_jx_k',font_size=29).move_to([0,1,0])
        for m,col in zip(eq,[WHITE,WHITE,BLUE,YELLOW,PURPLE]):m.set_color(col)
        label=VGroup(*[jp(t,22,c).next_to(eq[i],DOWN,buff=.4) for i,t,c in [(1,'定数',WHITE),(2,'一次',BLUE),(3,'二次',YELLOW),(4,'三次',PURPLE)]])
        ref=jp('式 (1.74)',21,MUTED).move_to([0,2.5,0])
        self.beat(FadeIn(eq),FadeIn(label),FadeIn(ref))
        symmetry=tex(r'x_1x_2=x_2x_1',39,YELLOW).move_to([0,-1.4,0])
        self.two(lambda:FadeIn(symmetry),lambda:Indicate(eq[3], scale_factor=1))
        self.remove(eq,label,symmetry,ref)
        d=ValueTracker(2); dv=lambda:int(round(d.get_value()))
        formula=tex(r'\binom{D+3}{3}=\frac{(D+1)(D+2)(D+3)}6',40,PURPLE).move_to([0,2.15,0])
        counter=meter('=',lambda:coefficient_count(dv()),[.8,.1,0],PURPLE,0)
        count_label=jp('係数の数',26,PURPLE).move_to([-2.2,.1,0])
        dcount=meter('D=',dv,[0,-.55,0],BLUE,0)
        slide=self.slider(d,1,100,[0,-2.2,0],width=8,ticks=[1,10,50,100])
        self.add(formula,counter,count_label,dcount,slide)
        self.two(lambda:d.animate.set_value(10),lambda:d.animate.set_value(100))
        asym=tex(r'\binom{D+3}{3}\sim\frac{D^3}{6}\qquad(D\to\infty)',34,YELLOW).move_to([0,1.03,0])
        note=jp('固定した次数 M：係数数は D の M 乗のオーダー',24).move_to([0,-1.25,0])
        self.beat(FadeIn(asym),FadeIn(note),Indicate(counter, scale_factor=1))

    def sphere(self):
        center=np.array([-3.6,.3,0]); radius=1.65
        outer=Circle(radius,color=BLUE,fill_opacity=.15).move_to(center)
        q=jp('外側の薄い層に、何％？',30,YELLOW).move_to([2.3,.4,0])
        self.beat(Create(outer),FadeIn(q))
        label=jp('半径と厚さの模式図',22,MUTED).move_to([-3.6,-1.65,0])
        line=Line(center,center+RIGHT*radius,color=WHITE)
        rlabel=tex('1',26).next_to(line,DOWN,buff=.12)
        self.beat(FadeIn(label),Create(line),FadeIn(rlabel))
        eps=ValueTracker(.001)
        ring=always_redraw(lambda:Annulus(inner_radius=radius*(1-eps.get_value()),outer_radius=radius,color=YELLOW,fill_opacity=.65,stroke_width=0).move_to(center))
        inner=always_redraw(lambda:Circle(radius*(1-eps.get_value()),color=GREEN,stroke_width=2).move_to(center))
        self.add(ring,inner)
        self.beat(eps.animate.set_value(.1),FadeOut(q))
        scale=ValueTracker(1.)
        sq=always_redraw(lambda:Square(1.6*scale.get_value(),color=GREEN,fill_opacity=.2).move_to([2,.4,0]))
        scale_formula=tex(r's\times s=s^2\quad;\quad s^D',34,GREEN).move_to([2,-1.2,0])
        self.add(sq,scale_formula)
        self.beat(scale.animate.set_value(.5))
        self.remove(sq,scale_formula)
        vol=MathTex(r'V_D(r)=',r'K_D',r'r^D',font_size=43).move_to([2.4,1,0]);vol[1].set_color(PURPLE);vol[2].set_color(GREEN)
        volnote=jp('式 (1.75)　相似な球の体積',22,MUTED).move_to([2.4,2,0])
        self.beat(FadeIn(vol),FadeIn(volnote),eps.animate.set_value(.2))
        ratio=tex(r'\frac{K_D-K_D(1-\varepsilon)^D}{K_D}',40).move_to([2.4,-.25,0])
        ratio.set_color_by_tex('K_D',PURPLE)
        self.beat(FadeIn(ratio),Indicate(vol[1], scale_factor=1))
        final=MathTex(r'1-',r'(1-\varepsilon)^D',font_size=44).move_to([2.4,-.25,0]);final[0].set_color(YELLOW);final[1].set_color(GREEN)
        result=tex(r'D=2,\ \varepsilon=0.1\quad\Rightarrow\quad19\%',32,YELLOW).move_to([1.6,-1.65,0])
        self.beat(ReplacementTransform(ratio,final),FadeIn(result),eps.animate.set_value(.1))

    def shells(self):
        d=ValueTracker(2);eps=ValueTracker(.1)
        ax,axes=self.graph_axes([0,.3,.05],[0,1,.2],center=(-.3,.1,0),width=8.5,height=3.4,xlabel=r'\varepsilon',ylabel='f')
        x=np.linspace(0,.3,181)
        curve=always_redraw(lambda:line_graph(ax,x,shell_fraction(d.get_value(),x),YELLOW))
        marker=always_redraw(lambda:Dot(ax.c2p(eps.get_value(),shell_fraction(d.get_value(),eps.get_value())),color=RED,radius=.075))
        cross=always_redraw(lambda:DashedLine(ax.c2p(eps.get_value(),0),marker.get_center(),color=RED))
        formula=tex(r'f=1-(1-\varepsilon)^D',33,YELLOW).move_to([-2.3,2.55,0])
        num=meter(r'f(\%)=',lambda:100*shell_fraction(d.get_value(),eps.get_value()),[2.8,2.55,0],YELLOW,2)
        sl=self.slider(d,1,50,[-2.6,-2.3,0],width=4.3,ticks=[1,20,50])
        se=self.slider(eps,0,.3,[3.2,-2.3,0],width=3.6,label=r'\varepsilon',ticks=[0,.1,.3],color=RED)
        dc=meter('D=',d.get_value,[5.4,1,0],PURPLE,1)
        self.add(axes,formula,curve,marker,cross,num,sl,se,dc)
        self.beat(d.animate.set_value(3))
        self.two(lambda:Indicate(num, scale_factor=1),lambda:d.animate.set_value(20))
        self.two(lambda:d.animate.set_value(50),lambda:Indicate(num,scale_factor=1))
        self.beat(d.animate.set_value(20))
        self.two(lambda:d.animate.set_value(20),lambda:eps.animate.set_value(.01))
        self.two(lambda:AnimationGroup(d.animate.set_value(50),eps.animate.set_value(1-2**(-1/50))),lambda:Indicate(num, scale_factor=1))
        self.beat(eps.animate.set_value(.1),Indicate(formula, scale_factor=1))

    def gaussian(self):
        ax,axes=self.graph_axes([-4,4,1],[-4,4,1],center=(-2.8,0,0),width=4.1,height=4.1,xlabel='x_1',ylabel='x_2')
        dots=VGroup(*[Dot(ax.c2p(*xy),radius=.023,color=BLUE) for xy in GAUSSIAN_POINTS])
        formula=tex(r'p(\mathbf{x})=(2\pi)^{-D/2}e^{-\|\mathbf{x}\|^2/2}',32,BLUE).move_to([0,2.6,0])
        assumption=tex(r'\mu=0,\quad\sigma=1',33).move_to([3.2,1.25,0])
        self.add(axes)
        self.beat(FadeIn(dots),FadeIn(formula),FadeIn(assumption))
        center=Dot(ax.c2p(0,0),color=YELLOW,radius=.08)
        density=jp('一点での密度は中心が最大',25,BLUE).move_to([3.1,.2,0])
        self.beat(FadeIn(density),Indicate(center,scale_factor=2))
        rad=ValueTracker(.4); unit=np.linalg.norm(ax.c2p(1,0)-ax.c2p(0,0))
        ring=always_redraw(lambda:Annulus(inner_radius=unit*rad.get_value(),outer_radius=unit*(rad.get_value()+.3),color=YELLOW,fill_opacity=.4,stroke_width=0).move_to(ax.c2p(0,0)))
        self.add(ring)
        self.beat(rad.animate.set_value(2))
        self.remove(assumption,density,center,formula)
        rax,ra=self.graph_axes([0,4,.5],[0,.8,.2],center=(2.5,-.1,0),width=5,height=3.2,xlabel='r',ylabel='p(r)')
        counts,bins=np.histogram(GAUSSIAN_RADII,bins=np.linspace(0,4,21));dens=counts/(len(GAUSSIAN_RADII)*.2)
        bars=VGroup(*[Rectangle(width=.235,height=max(.001,h*4),color=BLUE,fill_opacity=.55,stroke_width=.5).move_to(rax.c2p((a+b)/2,h/2)) for a,b,h in zip(bins[:-1],bins[1:],dens)])
        histnote=jp('600点の半径 → 確率密度',23,BLUE).move_to([2.6,2.15,0])
        bins_used=np.clip(np.searchsorted(bins,GAUSSIAN_RADII,side='right')-1,0,19)
        levels=np.zeros(20,dtype=int)
        histdots=VGroup()
        for j in bins_used:
            levels[j]+=1
            histdots.add(Dot(rax.c2p((bins[j]+bins[j+1])/2,levels[j]/120),radius=.023,color=BLUE))
        self.two(lambda:AnimationGroup(FadeIn(ra),TransformFromCopy(dots,histdots)),
                 lambda:AnimationGroup(FadeOut(histdots),FadeIn(bars),FadeIn(histnote)))
        self.remove(histdots)
        rr=np.linspace(0,4,181);curve=line_graph(rax,rr,radial_pdf(rr,2),GREEN)
        area=always_redraw(lambda:line_graph(rax,np.linspace(rad.get_value(),rad.get_value()+.3,35),radial_pdf(np.linspace(rad.get_value(),rad.get_value()+.3,35),2),YELLOW,True))
        self.add(area)
        self.beat(FadeIn(curve),rad.animate.set_value(.6),bars.animate.set_opacity(.25))
        factors=MathTex('p(r)',r'\propto',r'r^{D-1}',r'e^{-r^2/2}',font_size=36).move_to([0,2.65,0]);factors[0].set_color(GREEN);factors[2].set_color(YELLOW);factors[3].set_color(BLUE)
        self.remove(histnote)
        self.beat(FadeIn(factors),rad.animate.set_value(1.4))
        peak=Dot(rax.c2p(1,radial_pdf(1,2)),color=GREEN,radius=.07)
        peaklabel=tex(r'r_{\rm mode}=1',27,GREEN).move_to([3.8,1.5,0])
        self.beat(FadeIn(peak),FadeIn(peaklabel),rad.animate.set_value(.85))
        self.beat(rad.animate.set_value(2.5))

    def concentration(self):
        d=ValueTracker(1)
        ax,axes=self.graph_axes([0,7,1],[0,.85,.2],center=(-.5,0,0),width=9.6,height=3.6,xlabel='r',ylabel='p(r)')
        r=np.linspace(0,7,301)
        curve=always_redraw(lambda:line_graph(ax,r,radial_pdf(r,d.get_value()),GREEN))
        mode=always_redraw(lambda:Dot(ax.c2p(np.sqrt(max(d.get_value()-1,0)),radial_pdf(np.sqrt(max(d.get_value()-1,0)),d.get_value())),radius=.07,color=YELLOW))
        sl=self.slider(d,1,100,[0,-2.4,0],width=8,ticks=[1,20,50,100])
        dn=meter('D=',d.get_value,[3.7,2.5,0],PURPLE,1)
        note=jp('各座標：独立・平均0・標準偏差1',23,MUTED).move_to([-1.6,2.5,0])
        self.add(axes,curve,mode,sl,dn,note)
        self.two(lambda:d.animate.set_value(2),lambda:d.animate.set_value(20))
        formula=tex(r'r_{\rm mode}=\sqrt{D-1}\simeq4.36',32,YELLOW).move_to([.6,1.4,0])
        self.beat(FadeIn(formula),Indicate(mode, scale_factor=1))
        tail=line_graph(ax,np.linspace(5.4,7,80),radial_pdf(np.linspace(5.4,7,80),20),YELLOW,True)
        self.beat(FadeIn(tail),Indicate(formula, scale_factor=1))
        self.remove(axes,curve,mode,formula,tail,note)
        uax,ua=self.graph_axes([0,2,.25],[0,6,1],center=(-.5,0,0),width=9.6,height=3.6,xlabel='u',ylabel='q(u)')
        u=np.linspace(0,2,301)
        uc=always_redraw(lambda:line_graph(uax,u,normalized_pdf(u,d.get_value()),GREEN))
        label=tex(r'u=\frac{r}{\sqrt D},\quad q(u)=\sqrt D\,p(\sqrt D\,u)',30,GREEN).move_to([-1.3,2.55,0])
        one=DashedLine(uax.c2p(1,0),uax.c2p(1,6),color=YELLOW)
        self.add(ua,uc,label,one)
        self.beat(d.animate.set_value(30))
        sums=VGroup(tex(r'r^2=\sum_{i=1}^D x_i^2',30,BLUE),VGroup(jp('標準偏差 ÷ 平均',18,BLUE),tex(r'=\sqrt{2/D}',27,BLUE)).arrange(RIGHT,buff=.15)).arrange(DOWN,buff=.3).move_to([-3.5,.9,0])
        self.beat(FadeIn(sums),d.animate.set_value(20))
        self.remove(sums)
        self.two(lambda:d.animate.set_value(100),lambda:Indicate(label, scale_factor=1))
        self.beat(d.animate.set_value(20))

    def manifold(self):
        x=ValueTracker(0);y=ValueTracker(0);theta=ValueTracker(0)
        image=VGroup(*[Square(.205,stroke_width=.3,stroke_color=MUTED).move_to([-3.4+(i%16-7.5)*.208,.15+(7.5-i//16)*.208,0]) for i in range(256)])
        def update_pixels(g):
            for m,v in zip(g,object_pixels(x.get_value(),y.get_value(),theta.get_value()).ravel()):
                m.set_fill(interpolate_color(ManimColor('#151923'),WHITE,float(v)),1)
        image.add_updater(update_pixels)
        border=SurroundingRectangle(image,buff=.04,color=BLUE)
        self.add(image,border)
        question=jp('256個の数字を全部覚える？',28,YELLOW).move_to([2.7,.5,0])
        self.beat(FadeIn(question),theta.animate.set_value(.25))
        self.beat(theta.animate.set_value(0),Indicate(border, scale_factor=1))
        self.remove(question)
        vector=tex(r'\mathbf{x}=(x_1,\ldots,x_{256})',36,BLUE).move_to([2.7,1.8,0])
        number=jp('16 × 16 = 256 画素',25,BLUE).move_to([-3.4,-1.95,0])
        strip=VGroup(*[Rectangle(width=.11,height=.7,stroke_width=.2) for i in range(32)]).arrange(RIGHT,buff=.025).move_to([2.7,.7,0])
        strip.add_updater(lambda g:[m.set_fill(interpolate_color(ManimColor('#151923'),WHITE,float(v)),1) for m,v in zip(g,object_pixels(x.get_value(),y.get_value(),theta.get_value()).ravel()[::8])])
        stripnote=jp('ベクトルの一部を表示',19,MUTED).move_to([2.7,-.05,0])
        self.beat(FadeIn(vector),FadeIn(number),TransformFromCopy(VGroup(*image[::8]),strip),FadeIn(stripnote))
        sx=self.slider(x,-.5,.5,[1.8,-.9,0],width=2.6,label='a',color=BLUE)
        sy=self.slider(y,-.5,.5,[1.8,-1.7,0],width=2.6,label='b',color=GREEN)
        st=self.slider(theta,-1,1,[1.8,-2.5,0],width=2.6,label=r'\theta',color=YELLOW)
        self.add(sx,sy,st)
        self.two(lambda:AnimationGroup(x.animate.set_value(.4),y.animate.set_value(.3)),lambda:theta.animate.set_value(.9))
        dim=jp('入力の次元 256　／　動きの自由度 3',27,PURPLE).move_to([0,2.6,0])
        self.beat(FadeIn(dim),x.animate.set_value(-.3),y.animate.set_value(-.2))
        angle=meter(r'\theta=',theta.get_value,[5,-.9,0],YELLOW,2)
        self.add(angle)
        self.beat(x.animate.set_value(.3),y.animate.set_value(.2),Indicate(angle, scale_factor=1))
        self.beat(theta.animate.set_value(.25))
        target=jp('向きの予測に効く自由度 1',27,YELLOW).move_to([2.5,1.8,0])
        self.beat(ReplacementTransform(vector,target),x.animate.set_value(0),y.animate.set_value(0),theta.animate.set_value(.5))
