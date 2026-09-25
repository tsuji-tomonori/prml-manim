"""Local caption metrics and PCM-synchronised beats, following PRML 1.1."""
import json
import re
import numpy as np
from manim import *
from narration_content import SCENES, estimated_duration
from make_voicevox_narration import valid_entry, OUTPUT_DIR

def jp(text, size=25, color=WHITE):
    return Text(text, font="Noto Sans CJK JP", font_size=size, color=color)


def tex(text, size=30, color=WHITE):
    return MathTex(text, font_size=size, color=color)


def caption_math(formula, kana_height):
    """Scale to kana x-height and center the unscripted body on the text axis."""
    body = re.sub(r"[_^](?:\{[^}]*\}|\\[A-Za-z]+|.)", "", formula)
    reference = MathTex(formula, body, 'x', font_size=26)
    # Measure actual glyphs: font_size has different meanings in Pango and TeX.
    scale = .67 * kana_height / reference[2].height
    # Large operators and tall delimiters should remain inline-sized.
    scale = min(scale, 1.12 * kana_height / reference[1].height)
    reference.scale(scale)
    mob = reference[0].copy()
    mob.shift(LEFT * mob.get_center()[0] + DOWN * reference[1].get_center()[1])
    return mob


def caption_mobject(display):
    """Wrap Japanese and atomic inline math with a shared body centerline."""
    kana_height = jp('あ', 22).height
    gap = .35 * kana_height
    tokens = re.findall(r"\$[^$]+\$|.", display)
    # Include the enlarged formula and both side bearings when choosing a wrap.
    weights = [max(1, (caption_math(t[1:-1], kana_height).width + 2*gap) / .30)
               if t.startswith('$') else (.55 if t.isascii() else 1) for t in tokens]
    if sum(weights) > 32:
        cumulative = np.cumsum(weights)
        split = min(range(1, len(tokens)), key=lambda i:
                    abs(cumulative[i-1] - sum(weights)/2) + (0 if tokens[i-1] in '、。' else 5))
        lines = [tokens[:split], tokens[split:]]
    else:
        lines = [tokens]
    def line_mobject(parts):
        runs = re.split(r"(\$[^$]+\$)", ''.join(parts))
        mobs = []
        cursor = 0.
        for run in (r for r in runs if r.strip()):
            if run.startswith('$'):
                mob = caption_math(run[1:-1], kana_height)
            else:
                # Reference kana keep standalone punctuation in its natural place.
                reference = jp('あ' + run.strip() + 'あ', 22)
                mob = VGroup(*reference[1:-1])
                mob.shift(DOWN * reference[0].get_center()[1])
            # Horizontal placement must not recenter superscripts/subscripts.
            mob.shift(RIGHT * (cursor - mob.get_left()[0]))
            cursor = mob.get_right()[0] + gap
            mobs.append(mob)
        return VGroup(*mobs)
    caption = VGroup(*[line_mobject(line) for line in lines]).arrange(DOWN, buff=.13)
    caption.move_to([0, -3.4, 0])
    if caption.width > 12.9 or caption.height > .9:
        raise ValueError(f'Caption outside safe area ({caption.width}, {caption.height}): {display}')
    return caption



class NarratedScene(Scene):
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

