"""PRML 1.6: original linked experiments, rendered in Manim Community."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from manim import *
from caption_layout import jp, tex, caption_mobject
from information_model import *
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry
from narration_content import SCENES, estimated_duration

BG = '#10141F'
BLUE_DATA = ManimColor('#58B5ED')
YELLOW_INFO = ManimColor('#FFE079')
ORANGE_MODEL = ManimColor('#FFB45B')
GREEN_TRUE = ManimColor('#77D49A')
PURPLE_KL = ManimColor('#C29AFF')
MUTED = ManimColor('#A8B2C5')
COLORS = [BLUE_DATA, GREEN_TRUE, YELLOW_INFO, PURPLE_KL]

def pulse(m, **kwargs):
    """Highlight a snapshot without freezing its source or leaving a ghost."""
    return Circumscribe(m, color=kwargs.get('color',YELLOW_INFO), buff=.10)

def readout(label, getter, pos, color=WHITE, places=3):
    prefix = tex(label, 27, color)
    number = DecimalNumber(getter(), num_decimal_places=places, font_size=27, color=color)
    group = VGroup(prefix, number).arrange(RIGHT, buff=.13).move_to(pos)
    anchor = number.get_left().copy()
    number.add_updater(lambda m: m.set_value(getter()).move_to(anchor, aligned_edge=LEFT))
    return group

def curve(ax, fn, lo, hi, color=BLUE_DATA):
    xs = np.linspace(lo,hi,240)
    return VMobject().set_points_as_corners([ax.c2p(x,fn(x)) for x in xs]).set_stroke(color,3)

def bars(values, width=8, height=3, bottom=-1.7, color=None):
    step=width/len(values)
    return VGroup(*[Rectangle(width=step*.76,height=max(.003,height*v),
        fill_color=color or COLORS[i%4],fill_opacity=.8,stroke_width=0)
        .move_to([-width/2+step*(i+.5),bottom+max(.003,height*v)/2,0]) for i,v in enumerate(values)])

class PRML16InformationTheory(Scene):
    def construct(self):
        self.camera.background_color=BG
        self.timeline=[]
        self.manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        for i,method in enumerate([self.surprise,self.coding,self.counting,self.bins,
            self.gaussian,self.conditioning,self.divergence,self.jensen,self.learning,self.shared]):
            self.begin(i)
            if not self.audio_entry:
                raise RuntimeError('Generate matching narration before rendering')
            method()
            assert self.beat_index == len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path('media/prml16_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def formula(self, *parts, colors=None, y=-2.45, size=32):
        m=MathTex(*parts,font_size=size).move_to([0,y,0])
        if colors:
            for item,c in zip(m,colors): item.set_color(c)
        if m.width>12.6: raise ValueError('Formula too wide')
        return m

    def ax(self, xr, yr, center=(0,.1,0), width=8.5, height=3.2):
        return Axes(x_range=xr,y_range=yr,x_length=width,y_length=height,tips=False,
            axis_config={'color':MUTED,'stroke_width':1.3,'include_numbers':True,
                         'font_size':19}).move_to(center)

    def slider(self, tracker, lo, hi, label, pos=(0,2.1,0), width=5):
        line=Line(LEFT*width/2,RIGHT*width/2,color=MUTED).move_to(pos)
        dot=Dot(color=YELLOW_INFO,radius=.075)
        dot.add_updater(lambda m:m.move_to(line.point_from_proportion((tracker.get_value()-lo)/(hi-lo))))
        val=readout(label,tracker.get_value,[pos[0]+width/2+1.1,pos[1],0],YELLOW_INFO,2)
        return VGroup(line,dot,val)
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

    def surprise(self):
        cards=VGroup(*[VGroup(RoundedRectangle(width=.85,height=1.25,corner_radius=.08,color=BLUE_DATA),
            tex(str(i+1),32)).arrange(ORIGIN) for i in range(8)]).arrange(RIGHT,buff=.2).move_to([0,.6,0])
        self.beat(LaggedStart(*[FadeIn(c,shift=UP*.2) for c in cards],lag_ratio=.12))
        self.beat(phases=[('half',self.sentence_duration(0),lambda:cards[4:].animate.set_opacity(.15)),
            ('quarter',self.sentence_duration(1)*.5,lambda:cards[2:4].animate.set_opacity(.15)),
            ('eighth',self.sentence_duration(1)*.5,lambda:cards[1:2].animate.set_opacity(.15)),
            ('identified',self.sentence_duration(2),lambda:pulse(cards[0],color=YELLOW_INFO))])
        self.remove(*cards.get_family())
        p=ValueTracker(.5)
        ax=self.ax([0,1,.25],[0,4,1],center=(0,.05,0))
        graph=curve(ax,lambda p:-np.log2(p),1/16,1,YELLOW_INFO)
        dot=always_redraw(lambda:Dot(ax.c2p(p.get_value(),-np.log2(p.get_value())),color=BLUE_DATA))
        guides=always_redraw(lambda:VGroup(DashedLine(ax.c2p(p.get_value(),0),dot.get_center(),color=BLUE_DATA),
            DashedLine(ax.c2p(0,-np.log2(p.get_value())),dot.get_center(),color=YELLOW_INFO)))
        labels=VGroup(tex('p',25,BLUE_DATA).next_to(ax.x_axis,RIGHT),jp('情報量 [bit]',21,YELLOW_INFO).move_to([-4,2.1,0]))
        counter=readout('h=',lambda:-np.log2(p.get_value()),[3,2.1,0],YELLOW_INFO)
        self.add(ax,graph,dot,guides,labels,counter)
        self.beat(p.animate.set_value(.25))
        self.beat(p.animate.set_value(1/16))
        f=self.formula(r'h(x)',r'=-\log_2',r'p(x)',colors=[YELLOW_INFO,WHITE,BLUE_DATA])
        self.add(f)
        relation=tex(r'-\log_2(ab)=-\log_2a-\log_2b',25).move_to([0,-2,0])
        self.beat(FadeIn(relation),pulse(f))
        self.beat(p.animate.set_value(1),end_sentence=1)

    def coding(self):
        mix=ValueTracker(0)
        probs=lambda:(1-mix.get_value())*np.full(4,.25)+mix.get_value()*P
        chart=always_redraw(lambda:bars(probs(),height=5,bottom=-.7))
        labels=VGroup(*[tex(c,27,COLORS[i]).move_to([-3+2*i,-1,0]) for i,c in enumerate('ABCD')])
        codes=VGroup(*[tex(c,29).move_to([-3+2*i,-1.65,0]) for i,c in enumerate(['00','01','10','11'])])
        prob_labels=VGroup()
        for i in range(4):
            n=DecimalNumber(probs()[i],num_decimal_places=3,font_size=23)
            n.add_updater(lambda m,i=i:m.set_value(probs()[i]).next_to(chart[i],UP,buff=.16))
            prob_labels.add(n)
        note=jp('高さ：確率',21,BLUE_DATA).move_to([-5.1,1.4,0])
        self.add(labels,chart,codes,note,prob_labels)
        self.beat(LaggedStart(*[pulse(c) for c in codes],lag_ratio=.2))
        self.beat(mix.animate.set_value(1),end_sentence=1)
        new=VGroup(*[tex(c,29).move_to([-3+2*i,-1.65,0]) for i,c in enumerate(['0','10','110','111'])])
        self.beat(Transform(codes,new),end_sentence=1)
        f=self.formula(r'\bar L=\tfrac12\cdot1+\tfrac14\cdot2+\tfrac18\cdot3+\tfrac18\cdot3=1.75\ \mathrm{bit}',size=29)
        self.add(f)
        self.remove(prob_labels,note)
        self.add(jp('幅：確率　高さ：符号長　面積：平均への寄与',21).move_to([0,2.2,0]))
        areas=VGroup();left=-4
        targets=[]
        for i,(p,length) in enumerate(zip(P,[1,2,3,3])):
            w=8*p;h=.7*length;center=left+w/2;left+=w
            areas.add(Rectangle(width=w,height=h,fill_color=COLORS[i],fill_opacity=.8,stroke_color=BG,stroke_width=2).move_to([center,-.7+h/2,0]))
            targets.extend([labels[i].animate.set_x(center),codes[i].animate.set_x(center)])
        chart.clear_updaters()
        self.beat(Transform(chart,areas),*targets)
        newf=self.formula(r'H[x]=',r'\sum_x p(x)',r'[-\log_2p(x)]',r'=1.75\ \mathrm{bit}',colors=[WHITE,BLUE_DATA,YELLOW_INFO,WHITE])
        self.remove(f); self.add(newf)
        self.beat(pulse(newf))
        nat=self.formula(r'H_{\rm nat}=H_{\rm bit}\ln2=1.213\ \mathrm{nat}')
        self.remove(newf); self.add(nat)
        self.beat(pulse(nat))

    def counting(self):
        def row(indices):
            return VGroup(*[Dot(radius=.085,color=BLUE_DATA if i in indices else YELLOW_INFO) for i in range(6)]).arrange(RIGHT,buff=.12)
        first=row(range(6)).scale(2.2).move_to([0,.6,0])
        self.add(first)
        self.beat(pulse(first))
        rows=VGroup(*[row(c) for c in itertools.combinations(range(6),3)]).arrange_in_grid(rows=5,cols=4,buff=(.55,.32)).move_to([0,.3,0])
        self.remove(first)
        self.beat(LaggedStart(*[FadeIn(r) for r in rows],lag_ratio=.07))
        f=self.formula(r'W=\frac{N!}{\prod_i n_i!}',r',\quad\frac{\ln W}{N}\longrightarrow-\sum_i p_i\ln p_i',size=29)
        self.add(f)
        self.beat(pulse(f))
        self.remove(*rows.get_family(),f)
        sigma=ValueTracker(1.3); uniform=ValueTracker(0); concentrate=ValueTracker(0)
        one=np.eye(30)[14]
        probs=lambda:(1-concentrate.get_value())*((1-uniform.get_value())*spread(sigma.get_value())+uniform.get_value()/30)+concentrate.get_value()*one
        chart=always_redraw(lambda:bars(probs(),width=10,height=3/max(.001,float(max(probs()))),color=BLUE_DATA))
        # Explicit adaptive vertical scale keeps both the broad and point-mass views readable.
        note=jp('30状態 ／ 棒の高さは確率（縦軸は拡大縮小）',20,MUTED).move_to([0,2.15,0])
        counter=readout('H=',lambda:entropy(probs()),[0,-2.35,0],YELLOW_INFO)
        self.add(chart,note,counter,readout(r'p_{\max}=',lambda:float(max(probs())),[4,-2.35,0],BLUE_DATA))
        self.beat(sigma.animate.set_value(6))
        self.beat(uniform.animate.set_value(1))
        self.beat(concentrate.animate.set_value(1))

    def bins(self):
        ax=self.ax([0,1,.25],[0,2,1],center=(0,.1,0),height=3)
        width=ValueTracker(1)
        density=always_redraw(lambda:Polygon(ax.c2p(0,0),ax.c2p(width.get_value(),0),
            ax.c2p(width.get_value(),1/width.get_value()),ax.c2p(0,1/width.get_value()),
            color=BLUE_DATA,fill_opacity=.3))
        self.add(ax,density,jp('密度',21,BLUE_DATA).move_to([-5.1,1.7,0]))
        self.beat(pulse(density))
        def divisions(n):
            return VGroup(*[Line(ax.c2p(i/n,0),ax.c2p(i/n,1),color=YELLOW_INFO,stroke_width=2) for i in range(1,n)])
        div=divisions(4); f=self.formula(r'\Delta=1/4,\quad H_\Delta=\log_2 4=2\ \mathrm{bit}',size=31)
        self.add(f)
        self.beat(Create(div))
        def split_bins(n):
            self.remove(f)
            f.become(self.formula(rf'\Delta=1/{n},\quad H_\Delta={int(np.log2(n))}\ \mathrm{{bit}}'))
            self.add(f)
            return Transform(div,divisions(n))
        self.beat(phases=[('eight bins',self.sentence_duration(0),lambda:split_bins(8)),
            ('sixteen bins',self.sentence_duration(1),lambda:split_bins(16))])
        self.remove(f)
        f=self.formula(r'P_i\approx',r'p(x_i)',r'\Delta',colors=[WHITE,BLUE_DATA,YELLOW_INFO])
        self.add(f)
        box=Polygon(ax.c2p(.5,0),ax.c2p(.5625,0),ax.c2p(.5625,1),ax.c2p(.5,1),fill_color=YELLOW_INFO,fill_opacity=.6,stroke_width=0)
        self.beat(FadeIn(box))
        new=self.formula(r'H_\Delta\approx',r'h[p]',r'-\ln\Delta',colors=[WHITE,BLUE_DATA,YELLOW_INFO])
        self.remove(f); self.add(new)
        integral=tex(r'h[p]=-\int p(x)\ln p(x)\,dx',29,BLUE_DATA).move_to([0,2.15,0])
        self.beat(FadeIn(integral),FadeOut(box))
        self.remove(div,new)
        counter=readout(r'h=\ln w=',lambda:np.log(width.get_value()),[0,-2.4,0],BLUE_DATA)
        self.add(counter)
        self.beat(width.animate.set_value(.5),start_sentence=1)

    def gaussian(self):
        ax=self.ax([-4,4,2],[0,1,.25],center=(0,.1,0))
        ax.y_axis.set_opacity(0)
        self.add(Line(ax.c2p(-4,0),ax.c2p(-4,1),color=MUTED,stroke_width=1.3))
        span=ValueTracker(.5)
        plot=SimpleNamespace(c2p=lambda x,y:ax.c2p(x,y/span.get_value()))
        ticks=VGroup(*[readout('',lambda f=f:span.get_value()*f,ax.c2p(-4,f)+LEFT*.42,MUTED,2) for f in [.25,.5,.75,1]])
        self.add(ticks)
        sigma=ValueTracker(1); mix=ValueTracker(0)
        uniform=lambda x:1/(2*np.sqrt(3)) if abs(x)<=np.sqrt(3) else 0
        flat=curve(plot,uniform,-4,4,MUTED)
        g=curve(plot,lambda x:gaussian_pdf(x),-4,4,GREEN_TRUE)
        legend=jp('灰：一様分布　緑：ガウス分布',20).move_to([0,2.15,0])
        self.add(ax,flat,g,legend)
        self.beat(Create(g))
        dynamic=always_redraw(lambda:curve(plot,lambda x:(1-mix.get_value())*uniform(x)+mix.get_value()*gaussian_pdf(x,sigma.get_value()),-4,4,BLUE_DATA))
        self.remove(flat); self.add(dynamic)
        pdf=self.formula(r'p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)',size=29)
        self.add(pdf)
        hcomp=tex(r'h_{\rm uniform}=1.242 < h_{\rm Gaussian}=1.419\ \mathrm{nat}',26).move_to([0,-1.95,0])
        self.add(hcomp)
        self.beat(mix.animate.set_value(1))
        self.remove(pdf,hcomp)
        f=self.formula(r'\int p=1,\quad E[x]=\mu,\quad E[(x-\mu)^2]=\sigma^2',size=31)
        self.add(f)
        self.beat(pulse(g),pulse(f))
        self.remove(g,f,legend)
        self.add(self.slider(sigma,.3,1.3,r'\sigma=',pos=(-.8,2.15,0),width=4))
        formula=self.formula(r'h[\mathcal N]=\tfrac12\{1+\ln(2\pi',r'\sigma^2',r')\}',colors=[WHITE,YELLOW_INFO,WHITE])
        sread=readout(r'\sigma=',sigma.get_value,[-3,-1.95,0],YELLOW_INFO)
        hread=readout('h=',lambda:gaussian_entropy(sigma.get_value()),[3,-1.95,0],BLUE_DATA)
        markers=always_redraw(lambda:VGroup(*[DashedLine(plot.c2p(sign*sigma.get_value(),0),plot.c2p(sign*sigma.get_value(),.25/sigma.get_value()),color=YELLOW_INFO) for sign in [-1,1]]))
        self.add(formula,sread,hread,markers)
        self.beat(sigma.animate.set_value(1.3))
        self.beat(sigma.animate.set_value(.3),span.animate.set_value(1.4))
        self.beat(sigma.animate.set_value(.6),span.animate.set_value(.75),start_sentence=1)

    def mosaic(self,r,width=4.5,center=(-1.5,.2,0)):
        # Columns have P(x)=1/2, blue/yellow are y=0/1; area equals joint probability.
        group=VGroup(); a=(1+r)/2
        for j in range(2):
            prob=a if j==0 else 1-a
            x=center[0]-width/4+j*width/2
            for h,c,cy in [(prob,BLUE_DATA,center[1]-1.5+1.5*prob),
                           (1-prob,YELLOW_INFO,center[1]+1.5-1.5*(1-prob))]:
                group.add(Rectangle(width=width/2,height=max(.002,3*h),color=BG,stroke_width=2,
                    fill_color=c,fill_opacity=.85).move_to([x,cy,0]))
        return group

    def conditioning(self):
        r=ValueTracker(.6)
        mosaic=always_redraw(lambda:self.mosaic(r.get_value()))
        labels=VGroup(tex('x=0',26).move_to([-2.625,2,0]),tex('x=1',26).move_to([-.375,2,0]),
            jp('青：y = 0　黄：y = 1',21).move_to([-1.5,-1.65,0]))
        self.add(mosaic,labels)
        self.beat(pulse(mosaic))
        selected=self.mosaic(.6)[0:2].copy()
        self.beat(selected.animate.stretch(2,0).move_to([3.5,.2,0]))
        cond=tex(r'p(y\mid x=0)=(0.8,0.2)',27).move_to([0,-2.4,0]);self.add(cond)
        defn=tex(r'H[y\mid x]=-\sum_{x,y}p(x,y)\ln p(y\mid x)',26).move_to([3.2,2.1,0])
        self.add(defn)
        self.beat(pulse(selected),pulse(cond))
        self.remove(defn)
        other=self.mosaic(.6)[2:4].copy().stretch(2,0).move_to([3.5,.2,0])
        self.remove(cond)
        cond=tex(r'p(y\mid x=1)=(0.2,0.8)',27).move_to([0,-2.4,0]);self.add(cond)
        self.beat(Transform(selected,other),end_sentence=1)
        self.remove(cond,selected)
        f=self.formula(r'H[x,y]=',r'H[x]',r'+H[y\mid x]',colors=[WHITE,BLUE_DATA,YELLOW_INFO])
        self.add(f)
        h=readout(r'H[y\mid x]=',lambda:conditional(r.get_value()),[3.5,.2,0],YELLOW_INFO)
        self.add(h,jp('nat',20,YELLOW_INFO).move_to([3.5,-.4,0]))
        self.beat(pulse(f))
        self.beat(r.animate.set_value(1),end_sentence=1)

    def divergence(self):
        t=ValueTracker(0)
        q=lambda:(1-t.get_value())*P+t.get_value()*Q
        chart=always_redraw(lambda:bars(P,width=8,height=4.5,bottom=-.8,color=BLUE_DATA))
        outlines=always_redraw(lambda:bars(q(),width=8,height=4.5,bottom=-.8,color=ORANGE_MODEL).set_fill(opacity=0).set_stroke(ORANGE_MODEL,3))
        legend=jp('青：本当の確率 p　橙の枠：想定 q',22).move_to([0,2.15,0])
        self.add(chart,outlines,legend)
        self.beat(t.animate.set_value(.25))
        self.beat(t.animate.set_value(1))
        self.remove(chart,outlines,legend)
        self.add(jp('幅：本当の確率 p　高さ：−ln q　面積：平均コスト',21).move_to([0,2.15,0]))
        def areas():
            group=VGroup();left=-4
            for p,qi in zip(P,q()):
                w=8*p;h=-np.log(qi)
                rect=Rectangle(width=w,height=h,stroke_color=BLUE_DATA,stroke_width=3,
                    fill_color=ORANGE_MODEL,fill_opacity=.55).move_to([left+w/2,-1+h/2,0]);group.add(rect);left+=w
            return group
        areas_m=always_redraw(areas); self.add(areas_m)
        f=self.formula(r'C(p,q)=-\sum_x',r'p(x)',r'\ln q(x)',colors=[WHITE,BLUE_DATA,ORANGE_MODEL])
        self.add(f)
        self.beat(LaggedStart(*[pulse(a) for a in areas_m],lag_ratio=.15))
        self.remove(f)
        f=self.formula(r'\mathrm{KL}(p\Vert q)=',r'C(p,q)',r'-H[p]',colors=[PURPLE_KL,ORANGE_MODEL,BLUE_DATA])
        counter=readout(r'\mathrm{KL}=',lambda:kl(P,q()),[0,-1.6,0],PURPLE_KL)
        self.add(f,counter)
        self.beat(pulse(f))
        self.beat(t.animate.set_value(0))
        reverse=readout(r'\mathrm{KL}(q\Vert p)=',lambda:kl(q(),P),[0,-1.95,0],ORANGE_MODEL)
        self.add(reverse)
        self.beat(t.animate.set_value(1),end_sentence=1)

    def jensen(self):
        ax=self.ax([.2,2.2,.5],[-.9,1.8,.5],center=(-1.2,.2,0),width=7,height=3.1)
        a=.25;b=2.;x=ValueTracker(.6)
        fn=lambda z:-np.log(z)
        chord=lambda z:fn(a)+(z-a)/(b-a)*(fn(b)-fn(a))
        graph=curve(ax,fn,.2,2.2,BLUE_DATA)
        line=Line(ax.c2p(a,fn(a)),ax.c2p(b,fn(b)),color=YELLOW_INFO)
        dots=always_redraw(lambda:VGroup(Dot(ax.c2p(x.get_value(),fn(x.get_value())),color=BLUE_DATA),
            Dot(ax.c2p(x.get_value(),chord(x.get_value())),color=YELLOW_INFO),
            Line(ax.c2p(x.get_value(),fn(x.get_value())),ax.c2p(x.get_value(),chord(x.get_value())),color=PURPLE_KL,stroke_width=5)))
        self.add(ax,graph,tex(r'f(z)=-\ln z',28).move_to([0,2.15,0]))
        self.beat(Create(line))
        self.add(dots)
        self.beat(x.animate.set_value(1.6))
        labels=VGroup(tex('E[f(z)]',26,YELLOW_INFO),tex('f(E[z])',26,BLUE_DATA)).arrange(DOWN,buff=.6).move_to([4,.5,0])
        self.add(labels)
        self.beat(x.animate.set_value(.7))
        f=self.formula(r'f(E[z])',r'\le',r'E[f(z)]',colors=[BLUE_DATA,WHITE,YELLOW_INFO]);self.add(f)
        self.beat(x.animate.set_value(1.3))
        self.remove(f)
        f=self.formula(r'E_p[q/p]=\sum_xq(x)=1',size=31);self.add(f)
        self.beat(x.animate.set_value(1))
        self.remove(f)
        f=self.formula(r'\mathrm{KL}(p\Vert q)=E_p[-\ln(q/p)]\ge-\ln1=0',size=30);self.add(f)
        self.beat(pulse(f),pulse(dots))

    def learning(self):
        dots=VGroup(*[Dot(radius=.1,color=BLUE_DATA if d else YELLOW_INFO) for d in DATA]).arrange_in_grid(rows=2,cols=10,buff=.24).move_to([0,.5,0])
        self.add(jp('自作データ：1 が14回、0 が6回',24).move_to([0,2.1,0]))
        self.beat(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=.06))
        self.remove(*dots.get_family())
        theta=ValueTracker(.3)
        ax=self.ax([0,1,.2],[.5,2.2,.5],center=(0,.05,0))
        g=curve(ax,nll,.05,.95,ORANGE_MODEL)
        dot=always_redraw(lambda:Dot(ax.c2p(theta.get_value(),nll(theta.get_value())),color=YELLOW_INFO))
        self.add(ax,g,dot,tex(r'\theta',24,YELLOW_INFO).next_to(ax.x_axis,RIGHT))
        self.add(self.slider(theta,.05,.95,r'\theta=',pos=(-2,2.1,0),width=3.6),readout(r'\overline{L}=',lambda:nll(theta.get_value()),[3,2.1,0],ORANGE_MODEL))
        # Replace original heading before readouts occupy the same row.
        for m in list(self.mobjects):
            if isinstance(m,Text) and '自作データ' in m.text:self.remove(m)
        self.beat(theta.animate.set_value(.9))
        f=self.formula(r'\bar L(\theta)=-\frac1N\sum_n\ln q(x_n\mid\theta)',size=32);self.add(f)
        self.beat(theta.animate.set_value(.25))
        self.remove(f)
        f=self.formula(r'\mathrm{KL}(p\Vert q_\theta)\approx',r'\bar L(\theta)',r'+\mathrm{const.}',colors=[WHITE,ORANGE_MODEL,MUTED],size=31);self.add(f)
        self.beat(pulse(f[2]))
        self.beat(theta.animate.set_value(.7),end_sentence=1)
        self.beat(pulse(dot),pulse(f))

    def shared(self):
        r=ValueTracker(0)
        mosaic=always_redraw(lambda:self.mosaic(r.get_value(),center=(-2,.3,0)))
        self.add(mosaic,jp('x と y：それぞれ半分ずつ',22).move_to([-2,2.1,0]))
        self.beat(pulse(mosaic))
        self.beat(r.animate.set_value(.6))
        f=self.formula(r'I[x,y]=\mathrm{KL}\big(p(x,y)\Vert p(x)p(y)\big)',size=31);self.add(f)
        independent=self.mosaic(0,center=(-2,.3,0)).set_fill(opacity=0).set_stroke(WHITE,2)
        self.beat(FadeIn(independent))
        self.remove(independent,f)
        f=self.formula(r'I[x,y]=',r'H[y]',r'-H[y\mid x]',colors=[PURPLE_KL,BLUE_DATA,YELLOW_INFO]);self.add(f)
        def stack():
            c=conditional(r.get_value())/np.log(2);i=mutual(r.get_value())/np.log(2)
            return VGroup(Rectangle(width=1.2,height=max(.002,3*c),fill_color=YELLOW_INFO,fill_opacity=.8,stroke_width=0).move_to([3.1,-1.2+1.5*c,0]),
                Rectangle(width=1.2,height=max(.002,3*i),fill_color=PURPLE_KL,fill_opacity=.8,stroke_width=0).move_to([3.1,1.8-1.5*i,0]))
        stack_m=always_redraw(stack)
        self.add(stack_m,readout('I=',lambda:mutual(r.get_value())/np.log(2),[4.9,.9,0],PURPLE_KL),
            jp('bit',20).move_to([4.9,.4,0]),jp('黄：残り　紫：共有',19).move_to([3.7,2.2,0]))
        self.beat(r.animate.set_value(.8))
        self.beat(r.animate.set_value(1),end_sentence=1)
        self.remove(mosaic,stack_m,f)
        # Clear data readouts for the final visual recap, preserving title and subtitles.
        keep=[m for m in self.mobjects if m is self.subtitle or (isinstance(m,Text) and m.get_y()>2.7)]
        self.remove(*[m for m in list(self.mobjects) if m not in keep])
        chain=VGroup(*[VGroup(tex(t,38,c),jp(label,22,c)).arrange(DOWN,buff=.3) for t,label,c in [
            ('h(x)','一回の驚き',YELLOW_INFO),('H[x]','平均の情報',BLUE_DATA),
            (r'\mathrm{KL}','想定のずれ',ORANGE_MODEL),('I[x,y]','共有する情報',PURPLE_KL)]]).arrange(RIGHT,buff=.8).move_to([0,.3,0])
        self.add(chain)
        self.beat(LaggedStart(*[pulse(m) for m in chain],lag_ratio=.5))
