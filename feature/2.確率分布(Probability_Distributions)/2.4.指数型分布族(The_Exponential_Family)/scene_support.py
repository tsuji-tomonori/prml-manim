"""Measured sentence audio drives both captions and visual actions."""
import json
import inspect
from pathlib import Path
import numpy as np
from manim import *
from caption_layout import jp, tex, caption_mobject
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry

BG='#10141F'
BLUE=ManimColor('#58B5ED')
RED=ManimColor('#FF7986')
GREEN=ManimColor('#77D49A')
YELLOW=ManimColor('#FFE079')
PURPLE=ManimColor('#C29AFF')
MUTED=ManimColor('#A8B2C5')


def formula(s, pos=(0,2.25,0), size=32):
    m=tex(s,size).move_to(pos)
    if m.width>12.7:
        raise ValueError(f'Formula too wide: {s}')
    return m

def curve(ax, f, lo, hi, color=BLUE, fill=False):
    xs=np.linspace(lo,hi,201)
    points=[ax.c2p(x,y) for x,y in zip(xs,f(xs))]
    m=VMobject().set_points_as_corners(points).set_stroke(color,3)
    if fill:
        m.set_points_as_corners([ax.c2p(lo,0), *points, ax.c2p(hi,0), ax.c2p(lo,0)])
        m.set_fill(color,.22)
    return m

def readout(label, getter, pos, color=YELLOW, places=2):
    prefix=tex(label,27,color)
    number=DecimalNumber(getter(),num_decimal_places=places,font_size=27,color=color)
    group=VGroup(prefix,number).arrange(RIGHT,buff=.12).move_to(pos)
    anchor=number.get_left().copy()
    number.add_updater(lambda m:m.set_value(getter()).move_to(anchor,aligned_edge=LEFT))
    return group

def slider(tracker, low, high, pos=(0,-2.3,0), label=r'\eta', width=7, color=YELLOW):
    line=NumberLine(x_range=[low,high,(high-low)/4],length=width,include_numbers=True,
                    font_size=18,color=MUTED).move_to(pos)
    knob=Dot(radius=.08,color=color).add_updater(lambda m:m.move_to(line.n2p(tracker.get_value())))
    value=readout(label+'=',tracker.get_value,(pos[0]+width/2+1.1,pos[1],0),color)
    return VGroup(line,knob,value)

def axes(xrange, yrange, center=(0,.05,0), width=8, height=3.35, xlabel='x', ylabel='p'):
    ax=Axes(x_range=xrange,y_range=yrange,x_length=width,y_length=height,tips=False,
            axis_config=dict(color=MUTED,include_numbers=True,font_size=19,
                             decimal_number_config=dict(num_decimal_places=2 if xrange[2]==.25 else 1)))
    ax.move_to(center)
    labels=VGroup(tex(xlabel,24).next_to(ax.x_axis.get_right(),RIGHT,buff=.18),
                  tex(ylabel,24).next_to(ax.y_axis.get_top(),UP,buff=.15))
    return ax,labels

class NarratedScene(Scene):
    def run_scenes(self, methods):
        self.camera.background_color=BG
        manifest={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for index,method in enumerate(methods):
            self.clear()
            self.story=SCENES[index]
            self.entry=manifest[self.story['id']]
            if not valid_entry(self.story,self.entry):
                raise RuntimeError('Missing or stale narration: '+self.story['id'])
            self.start=float(self.time)
            self.bi=0
            self.subtitle=None
            self.header=jp(self.story['title'],33).move_to([0,3.35,0])
            self.add(self.header)
            self.add_sound(str(OUTPUT_DIR/(self.story['id']+'.wav')))
            self.timeline.append(dict(id=self.story['id'],title=self.story['title'],reference=self.story['reference'],start=self.start,beats=[]))
            method()
            assert self.bi==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        path=Path(config.media_dir)/'prml24_timeline.json'
        path.write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def beat(self, *actions):
        cues=[c for c in self.entry['subtitle_cues'] if c['beat_index']==self.bi]
        assert len(actions)==len(cues)
        record=dict(note=self.story['beats'][self.bi]['visual_note'],start=float(self.time),cues=[])
        for c,action in zip(cues,actions):
            if self.subtitle is not None:
                self.remove(self.subtitle)
            self.subtitle=caption_mobject(c['display'])
            self.add(self.subtitle)
            self.update_mobjects(0)
            target_frame=round((self.start+c['end'])*config.frame_rate)
            frames=target_frame-round(self.time*config.frame_rate)
            start=float(self.time)
            animation=action() if inspect.isfunction(action) else action
            self.play(animation,run_time=(frames-1e-5)/config.frame_rate,rate_func=linear)
            record['cues'].append(dict(id=c['id'],display=c['display'],audio_start=self.start+c['start'],audio_end=self.start+c['end'],start=start,end=float(self.time)))
        end=self.start+sum(self.entry['beat_durations'][:self.bi+1])
        frames=round((end-self.time)*config.frame_rate)
        if frames:
            self.wait(frames/config.frame_rate+1e-8, frozen_frame=True)
        record['end']=float(self.time)
        self.timeline[-1]['beats'].append(record)
        self.bi+=1

    def highlight(self,m,color=YELLOW):
        return lambda: ShowPassingFlash(SurroundingRectangle(m,buff=.12,color=color),time_width=.7)

    def change(self, current, target):
        """Finish symbol changes early, then trace the readable completed formula."""
        return AnimationGroup(
            Transform(current, target, rate_func=lambda a:smooth(min(1,10*a))),
            ShowPassingFlash(SurroundingRectangle(target,buff=.1,color=YELLOW),time_width=.5))

    def match(self, current, target):
        return AnimationGroup(
            TransformMatchingTex(current,target,rate_func=lambda a:smooth(min(1,10*a))),
            ShowPassingFlash(SurroundingRectangle(target,buff=.1,color=YELLOW),time_width=.5))

    def replace(self, current, target):
        return AnimationGroup(
            ReplacementTransform(current,target,rate_func=lambda a:smooth(min(1,10*a))),
            ShowPassingFlash(SurroundingRectangle(target,buff=.1,color=YELLOW),time_width=.5))
