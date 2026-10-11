"""PCM-derived clock; same typography and frame quantization as PRML 1.1."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry
from video_support import jp, tex, caption_mobject

BG='#10141F'
RED_CLASS=ManimColor('#FF7687')
BLUE_CLASS=ManimColor('#62B7EE')
GREEN_CLASS=ManimColor('#80D4A2')
YELLOW_ACC=ManimColor('#FFE184')
PURPLE_ACC=ManimColor('#C5A1FF')
MUTED=ManimColor('#AAB6CA')

class NarratedScene(Scene):
    def begin(self,index):
        self.clear()
        self.story=SCENES[index]; self.bi=0; self.subtitle=None
        self.entry=self.entries[self.story['id']]
        if not valid_entry(self.story,self.entry):
            raise RuntimeError('Missing or stale audio: '+self.story['id'])
        self.start=float(self.time)
        self.title=jp(self.story['title'],32).move_to([0,3.35,0])
        self.add(self.title)
        self.formula=None
        self.add_sound(str(OUTPUT_DIR/(self.story['id']+'.wav')))
        self.timeline.append(dict(id=self.story['id'],start=self.start,reference=self.story['reference'],beats=[]))

    def equation(self,*parts,size=31):
        if self.formula is not None: self.remove(*self.formula.get_family())
        self.formula=MathTex(*parts,font_size=size).move_to([0,2.48,0])
        if self.formula.width>12.5: self.formula.scale_to_fit_width(12.5)
        self.add(self.formula)
        return self.formula

    def body_card(self, label):
        saved = [m for m in self.mobjects if m is not self.title and m is not self.subtitle]
        self.remove(*saved)
        frame = VGroup()
        heading = jp(label, 25).move_to([-5.35,2.05,0], aligned_edge=LEFT)
        self.add(heading)
        return saved, frame, heading

    def restore_body(self, saved):
        self.remove(*[m for m in self.mobjects if m is not self.title and m is not self.subtitle])
        self.add(*saved)

    def sentence_duration(self, index):
        cues=[c for c in self.entry['subtitle_cues'] if c['beat_index']==self.bi]
        return cues[index]['end']-cues[index]['start']

    def recap_label(self, label):
        # Measure at the origin before positioning: shifted empty SVG parents
        # can enlarge Cairo's family bounds in this Manim version.
        text=jp(label,21)
        return text.move_to([6.05,1.9,0],aligned_edge=RIGHT)

    def beat(self,*animations,actions=None,phases=None):
        beat=self.story['beats'][self.bi]
        cues=[c for c in self.entry['subtitle_cues'] if c['beat_index']==self.bi]
        offset=sum(self.entry['beat_durations'][:self.bi])
        duration=self.entry['beat_durations'][self.bi]
        if self.subtitle is not None: self.remove(self.subtitle)
        captions=VGroup(*[caption_mobject(c['display']).set_opacity(0) for c in cues])
        self.subtitle=captions; self.add(captions)
        start=float(self.time)
        def cap(m,alpha):
            now=alpha*duration+offset
            i=max([j for j,c in enumerate(cues) if c['start']<=now+1e-6] or [0])
            for j,x in enumerate(m): x.set_opacity(j==i)
        record=dict(note=beat['visual_note'],start=start,end=start+duration,
                    cues=[dict(c,start=self.start+c['start'],end=self.start+c['end']) for c in cues],actions=[])
        cap(captions,0)
        if phases:
            # All phase boundaries share the PCM clock, rounded cumulatively to frames.
            elapsed=0
            requested=0.
            assert sum(p[1] for p in phases) <= duration + 1e-6
            for name, seconds, factory in [*phases, ('breath', duration-sum(p[1] for p in phases), lambda:Wait())]:
                requested += seconds
                end=min(duration, round(requested*15)/15)
                dt=end-elapsed
                if dt<=1e-6:
                    continue
                self.update_mobjects(0)
                animation=factory()
                self.play(animation,UpdateFromAlphaFunc(captions,
                    lambda m,a,e=elapsed,d=dt:cap(m,(e+a*d)/duration)),
                    run_time=dt-1e-7,rate_func=linear)
                record['actions'].append(dict(name=name,start=start+elapsed,end=start+end))
                elapsed=end
        elif actions:
            # Factories defer target creation until the corresponding spoken sentence.
            elapsed=0
            for i,factory in enumerate(actions):
                end=duration if i==len(actions)-1 else round((cues[i+1]['start']-offset)*15)/15
                dt=end-elapsed
                anim=factory()
                record['actions'].append(dict(sentence=cues[i]['id'],start=start+elapsed,end=start+end))
                self.play(anim,UpdateFromAlphaFunc(captions,lambda m,a,e=elapsed,d=dt: cap(m,(e+a*d)/duration)),run_time=dt-1e-7,rate_func=linear)
                elapsed=end
        else:
            self.play(*animations,UpdateFromAlphaFunc(captions,cap),run_time=duration-1e-7,rate_func=linear)
            record['actions'].append(dict(sentence='beat',start=start,end=start+duration))
        self.timeline[-1]['beats'].append(record); self.bi+=1

    def plot_axes(self,x=(-3,3,1),y=(0,1,.5),width=8.7,height=3.5,center=(-.5,-.15,0),labels=('x','y')):
        ax=Axes(x_range=x,y_range=y,x_length=width,y_length=height,tips=False,
                axis_config=dict(color=MUTED,stroke_width=1.5,include_numbers=True,font_size=19))
        ax.move_to(center)
        names=VGroup(tex(labels[0],25).next_to(ax.x_axis,RIGHT,buff=.12),tex(labels[1],25).move_to(ax.get_corner(UL)+UP*.15+RIGHT*.35))
        self.add(ax,names)
        return ax


def curve(ax,fn,lo=None,hi=None,color=YELLOW_ACC):
    u=np.linspace(ax.x_range[0] if lo is None else lo,ax.x_range[1] if hi is None else hi,161)
    v=fn(u)
    origin=ax.c2p(0,0)
    pts=origin+u[:,None]*(ax.c2p(1,0)-origin)+np.asarray(v)[:,None]*(ax.c2p(0,1)-origin)
    return VMobject().set_points_as_corners(pts).set_stroke(color,3)


def number(label,fn,at,color=WHITE,places=2,size=25):
    t=tex(label,size,color); n=DecimalNumber(fn(),num_decimal_places=places,font_size=size,color=color)
    g=VGroup(t,n).arrange(RIGHT,buff=.14).move_to(at)
    anchor=n.get_left().copy()
    n.add_updater(lambda m:m.set_value(fn()).next_to(t,RIGHT,buff=.14))
    return g


def knob(label,tracker,lo,hi,at=(-.5,-2.5,0),color=YELLOW_ACC):
    line=NumberLine(x_range=[lo,hi,(hi-lo)/4],length=4,include_numbers=False,color=MUTED)
    line.move_to(at)
    dot=Dot(radius=.085,color=color).add_updater(lambda m:m.move_to(line.n2p(tracker.get_value())))
    n=number(label,tracker.get_value,np.array(at)+RIGHT*3.5,color)
    return VGroup(line,dot,n)


def note(text,at=(0,-2.45,0),color=MUTED):
    return jp(text,23,color).move_to(at)
