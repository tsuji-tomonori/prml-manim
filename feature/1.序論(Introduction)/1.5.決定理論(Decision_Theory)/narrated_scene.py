"""PCM-clock narration and inline captions, adapted from PRML 1.1."""
import json
from pathlib import Path
import numpy as np
from manim import *
from visual_support import caption_mobject, jp
from make_voicevox_narration import MANIFEST, OUTPUT_DIR, valid_entry
from narration_content import SCENES

class NarratedScene(Scene):
    def begin(self, index):
        self.clear()
        self.story = SCENES[index]
        self.beat_index = 0
        self.subtitle = None
        self.formula = None
        self.next_section(self.story['id'])
        self.header=VGroup(jp(self.story['title'], 33).move_to([0, 3.48, 0]),
                           jp('PRML 1.5  /  ' + str(index+1) + '  /  9', 15, '#A8B2C5').move_to([0, 3.03, 0]))
        self.add(self.header)
        entry = self.manifest[self.story['id']]
        if not valid_entry(self.story, entry):
            raise RuntimeError('Missing or stale narration: ' + self.story['id'])
        self.audio_entry = entry
        self.durations = entry['beat_durations']
        self.boundaries = np.ceil(np.cumsum(self.durations)*config.frame_rate-1e-6)/config.frame_rate
        self.scene_start = float(self.time)
        self.add_sound(str(OUTPUT_DIR / f"{self.story['id']}.wav"))
        self.timeline.append(dict(id=self.story['id'], title=self.story['title'], start=self.scene_start,
                                  reference=self.story['reference'], audio=True, beats=[]))

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

