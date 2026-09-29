"""One PCM narration beat = one continuously animated visual action."""
import json
from pathlib import Path
import numpy as np
from manim import *
from caption_layout import jp, tex, caption_mobject, polyline
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry

BG="#10141F"
BLUE=ManimColor("#58B5ED")
RED=ManimColor("#FF6B77")
GOLD=ManimColor("#FFE079")
GREEN=ManimColor("#77D49A")
PURPLE=ManimColor("#C29AFF")
MUTED=ManimColor("#A8B2C5")

class NarratedScene(Scene):
    def setup(self):
        self.camera.background_color=BG
        self.entries={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        for scene in SCENES:
            if not valid_entry(scene,self.entries.get(scene['id'],{})):
                raise RuntimeError('Generate current narration first: '+scene['id'])
        self.timeline=[]

    def begin(self,i):
        self.clear();self.story=SCENES[i];self.entry=self.entries[self.story['id']]
        self.index=0;self.caption=None;self.formula=None;self.start=float(self.time)
        self.add(jp(self.story['title'],32).move_to([0,3.45,0]))
        self.add_sound(str(OUTPUT_DIR/(self.story['id']+'.wav')))
        self.timeline.append(dict(id=self.story['id'],title=self.story['title'],start=self.start,reference=self.story['reference'],beats=[]))

    def beat(self,*animations):
        b=self.story['beats'][self.index];cue=self.entry['subtitle_cues'][self.index]
        if self.caption is not None:self.remove(self.caption)
        self.caption=caption_mobject(cue['display']);self.add(self.caption)
        duration=self.entry['beat_durations'][self.index]
        start=float(self.time)
        # Every beat is frame-aligned in the generated PCM. No accumulated drift.
        prepared=[]
        for animation in animations:
            if getattr(animation,'quick_formula',False) or (isinstance(animation,ReplacementTransform) and isinstance(animation.mobject,Text)):
                animation.set_run_time(.85)
                if len(animations)==1:
                    rest=ShowPassingFlash(SurroundingRectangle(self.formula,color=GOLD,buff=.13),time_width=.7,run_time=duration-.85)
                else:
                    rest=Wait(duration-.85)
                prepared.append(Succession(animation,rest))
            else:
                prepared.append(animation)
        self.play(*prepared,run_time=duration-1e-6,rate_func=linear)
        self.timeline[-1]['beats'].append(dict(id=cue['id'],start=start,end=float(self.time),speech_end=self.start+cue['end'],display=cue['display'],action=b['visual_note']))
        self.index+=1

    def finish(self):
        assert self.index==len(self.story['beats'])
        self.timeline[-1]['end']=float(self.time)

    def save_timeline(self):
        out=Path(config.media_dir)/'prml44_timeline.json'
        out.write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def equation(self,*parts,colors=None,size=31):
        target=MathTex(*parts,font_size=size).move_to([0,-2.55,0])
        if target.width>12.5:raise ValueError('Formula too wide')
        if colors:
            for i,c in colors.items():target[i].set_color(c)
        old=self.formula;self.formula=target
        animation=FadeIn(target) if old is None else AnimationGroup(FadeOut(old),FadeIn(target))
        animation.quick_formula=True
        return animation

    def legend(self,*items):
        group=VGroup(*[VGroup(Line(LEFT*.16,RIGHT*.16,color=c,stroke_width=4),jp(s,19,c)).arrange(RIGHT,buff=.12) for s,c in items]).arrange(RIGHT,buff=.55)
        group.move_to([0,2.83,0]);self.add(group);return group

    def axes(self,x=(-2,3,1),y=(0,1.2,.4),width=9,height=3.5,center=(0,.45,0),xlabel='z',ylabel=None):
        ax=Axes(x_range=x,y_range=y,x_length=width,y_length=height,tips=False,axis_config={'color':MUTED,'stroke_width':1.4,'include_ticks':False}).move_to(center)
        labels=VGroup()
        for value in np.arange(x[0],x[1]+.01,x[2]):
            labels.add(tex(f'{value:g}',18,MUTED).move_to(ax.c2p(value,y[0])+DOWN*.22))
        for value in np.arange(y[0],y[1]+.01,y[2]):
            labels.add(tex(f'{value:g}',18,MUTED).next_to(ax.c2p(x[0],value),LEFT,buff=.13))
        labels.add(tex(xlabel,24).move_to([0,-1.85,0]) if len(xlabel)>15 else tex(xlabel,24).next_to(ax.c2p(x[1],y[0]),RIGHT,buff=.2))
        if ylabel:labels.add(tex(ylabel,23).next_to(ax.c2p(x[0],y[1]),UP,buff=.1))
        self.add(ax,labels);return ax

    def changing_axes(self,x,limits,clip,height=3.35,center=(0,.5,0)):
        """Animate the labelled y window with the curve, retaining screen bounds."""
        ax=VGroup();ax.x_range=x;ax.y_range=clip
        left=center[0]-4.5;bottom=center[1]-height/2
        def point(xx,yy):
            lo,hi=limits()
            return np.array([left+9*(xx-x[0])/(x[1]-x[0]),bottom+height*(yy-lo)/(hi-lo),0.])
        ax.c2p=point
        rails=always_redraw(lambda:VGroup(Line(point(x[0],0),point(x[1],0),color=MUTED,stroke_width=1.4),Line(point(0,limits()[0]),point(0,limits()[1]),color=MUTED,stroke_width=1.4)))
        labels=VGroup()
        for v in np.arange(x[0],x[1]+.01,x[2]):
            labels.add(tex(f'{v:g}',18,MUTED).move_to([point(v,limits()[0])[0],bottom-.22,0]))
        for fraction in np.linspace(0,1,6):
            label=DecimalNumber(0,num_decimal_places=1,font_size=18,color=MUTED)
            label.add_updater(lambda mob,f=fraction:mob.set_value(limits()[0]+f*(limits()[1]-limits()[0])).move_to([left-.32,bottom+f*height,0]))
            label.update(0);labels.add(label)
        labels.add(tex('z',24).move_to([left+9.25,bottom,0]))
        ax.add(rails,labels);self.add(ax)
        return ax

    def curve(self,ax,fn,color=BLUE,lo=None,hi=None,width=3):
        x=np.linspace(ax.x_range[0] if lo is None else lo,ax.x_range[1] if hi is None else hi,301)
        y=np.asarray(fn(x))+np.zeros_like(x)
        # Visible portion only, never flatten an out-of-range curve at the border.
        mask=(y>=ax.y_range[0])&(y<=ax.y_range[1])
        ids=np.where(mask)[0];groups=np.split(ids,np.where(np.diff(ids)>1)[0]+1)
        origin=ax.c2p(0,0);ux=ax.c2p(1,0)-origin;uy=ax.c2p(0,1)-origin
        points=origin+x[:,None]*ux+y[:,None]*uy
        return VGroup(*[polyline(points[g],color,width) for g in groups if len(g)>1])

    def area(self,ax,fn,color=BLUE,lo=None,hi=None):
        x=np.linspace(ax.x_range[0] if lo is None else lo,ax.x_range[1] if hi is None else hi,160)
        origin=ax.c2p(0,0);ux=ax.c2p(1,0)-origin;uy=ax.c2p(0,1)-origin
        points=[ax.c2p(x[0],0),*(origin+x[:,None]*ux+np.asarray(fn(x))[:,None]*uy),ax.c2p(x[-1],0)]
        return Polygon(*points,stroke_width=0,fill_color=color,fill_opacity=.22)

    def slider(self,tr,low,high,label,color=GOLD,y=-1.92):
        rail=Line([-3,y,0],[3,y,0],color=MUTED,stroke_width=2)
        dot=Dot(radius=.07,color=color)
        dot.add_updater(lambda m:m.move_to(rail.point_from_proportion((tr.get_value()-low)/(high-low))))
        group=VGroup(rail,dot,tex(label,24,color).move_to([-4.1,y,0]),tex(f'{low:g}',17,MUTED).move_to([-3,y-.25,0]),tex(f'{high:g}',17,MUTED).move_to([3,y-.25,0]))
        self.add(group);return group

    def number(self,label,fn,pos,color=GOLD):
        prefix=tex(label,25,color);n=DecimalNumber(fn(),num_decimal_places=3,font_size=25,color=color)
        g=VGroup(prefix,n).arrange(RIGHT,buff=.12).move_to(pos);left=n.get_left().copy()
        n.add_updater(lambda m:m.set_value(fn()).move_to(left,aligned_edge=LEFT))
        self.add(g);return g

    def body_card(self, heading):
        """Temporarily replace the body, retaining all original object identities."""
        saved = (list(self.mobjects), self.formula, self.caption)
        title = self.mobjects[0]
        self.clear()
        self.add(title)
        self.formula = None
        self.caption = None
        frame = RoundedRectangle(width=11.6, height=4.8, corner_radius=.12,
                                 stroke_color='#FFFF00', stroke_width=1.5).move_to([0,.2,0])
        label = jp(heading,25).move_to([-5.35,2.05,0],aligned_edge=LEFT)
        self.add(frame,label)
        return saved

    def restore_body(self, saved):
        self.clear()
        objects, self.formula, self.caption = saved
        self.add(*objects)

    def brief(self, animation, seconds=.7):
        """Complete text transitions promptly while the numerical action continues."""
        duration=self.entry['beat_durations'][self.index]
        return Succession(animation.set_run_time(seconds),Wait(duration-seconds))
