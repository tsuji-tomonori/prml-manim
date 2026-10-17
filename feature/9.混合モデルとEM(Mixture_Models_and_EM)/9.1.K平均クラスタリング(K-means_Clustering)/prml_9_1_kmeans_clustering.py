"""PRML 9.1: K-means on one point set, then RGB vector quantization."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from manim import *

from kmeans_model import POINTS, INITIAL, ALTERNATE, assign, run, toy_rgb_image, quantize
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import NarratedScene, jp, tex

BLUE = ManimColor("#58C4DD")
YELLOW = ManimColor("#FFFF00")
GREEN = ManimColor("#83C167")
PURPLE = ManimColor("#9A72AC")
RED = ManimColor("#FC6255")
MUTED = ManimColor("#A8B2C5")
BG = "#10141F"
GROUP_COLORS = (RED, BLUE)
MAIN = run()
OTHER = run(initial=ALTERNATE)


def early(t):
    return smooth(min(1, 5 * t))


def snap(t):
    return 0 if t < .12 else 1


def J(message, size=23, color=WHITE):
    return jp(str(message), size, color)


def map_point(point):
    return np.array([-1.95 + 1.05 * point[0], .12 + point[1], 0.])


def marker(point, index):
    center = map_point(point)
    cross = VGroup(
        Line(center + LEFT * .12 + DOWN * .12, center + RIGHT * .12 + UP * .12,
             color=GROUP_COLORS[index], stroke_width=5),
        Line(center + LEFT * .12 + UP * .12, center + RIGHT * .12 + DOWN * .12,
             color=GROUP_COLORS[index], stroke_width=5),
    )
    label = tex(r"\mu_" + str(index + 1), 26, GROUP_COLORS[index]).next_to(cross, UP, buff=.16)
    return VGroup(cross, label).set_z_index(6)


def bisector(centres):
    a, b = centres
    mid = (a + b) / 2
    normal = b - a
    candidates = []
    for x in (-3.05, 3.05):
        if abs(normal[1]) > 1e-8:
            y = (np.dot(normal, mid) - normal[0] * x) / normal[1]
            if -1.95 <= y <= 2.25:
                candidates.append(np.array([x, y]))
    for y in (-1.95, 2.25):
        if abs(normal[0]) > 1e-8:
            x = (np.dot(normal, mid) - normal[1] * y) / normal[0]
            if -3.05 <= x <= 3.05:
                candidates.append(np.array([x, y]))
    if len(candidates) < 2:
        return VGroup()
    return Line(map_point(candidates[0]), map_point(candidates[-1]),
                color=PURPLE, stroke_width=2.4).set_opacity(.8).set_z_index(-1)


def links(centres, labels):
    return VGroup(*[
        Line(map_point(point), map_point(centres[label]), color=GROUP_COLORS[label],
             stroke_width=1.5).set_opacity(.32).set_z_index(-2)
        for point, label in zip(POINTS, labels)
    ])


def cost_panel(value, counts=None):
    title = J("ひずみ", 23, MUTED).move_to([4.10, 1.85, 0])
    number = tex(f"J={value:.3f}", 31, GREEN).move_to([4.1, 1.36, 0])
    width = max(.05, 3.65 * value / 100)
    bar = Rectangle(width=width, height=.19, stroke_width=0,
                    fill_color=GREEN, fill_opacity=.85).move_to([2.36 + width / 2, .92, 0])
    group = VGroup(title, number, bar)
    if counts is not None:
        line = J(f"赤 {counts[0]}点   青 {counts[1]}点", 22).move_to([4.1, .30, 0])
        group.add(line)
    return group


def cost_trace(count):
    values = [91.4, 28.588, 15.4196, 11.2641666667]
    points = [np.array([2.48 + i * 1.02, -.52 - 1.05 * (91.4 - v) / 80.2, 0])
              for i, v in enumerate(values[:count])]
    strokes = VGroup(*[Line(a, b, color=GREEN, stroke_width=2.5)
                       for a, b in zip(points, points[1:])])
    dots = VGroup(*[Dot(point, radius=.07, color=BLUE if i % 2 == 0 else RED)
                    for i, point in enumerate(points)])
    labels = VGroup(*[J("E" if i % 2 == 0 else "M", 18, MUTED)
                      .move_to([point[0], -1.79, 0])
                      for i, point in enumerate(points)])
    return VGroup(strokes, dots, labels)


class PRML91KMeansClustering(NarratedScene):
    def construct(self):
        self.camera.background_color = BG
        self.manifest = {e["id"]: e for e in json.loads(MANIFEST.read_text())["scenes"]}
        self.timeline = []
        methods = (self.opening, self.distance, self.score, self.e_step,
                   self.m_step, self.iterate, self.local_minimum, self.pixels,
                   self.return_to_points)
        for i, method in enumerate(methods):
            self.begin(i)
            self.note = None
            method()
            assert self.beat_index == len(SCENES[i]["beats"])
            self.timeline[-1]["end"] = float(self.time)
        (Path(config.media_dir) / "prml91_timeline.json").write_text(
            json.dumps(self.timeline, ensure_ascii=False, indent=2) + "\n")

    def note_anim(self, message, color=MUTED, math=False):
        note = tex(message, 26, color) if math else J(message, 22, color)
        if note.width > 4.5:
            note.scale_to_fit_width(4.5)
        note.move_to([4.10, -2.22, 0])
        if self.note is not None:
            self.remove(self.note)
        self.note = note
        return FadeIn(note, rate_func=lambda t: smooth(min(1, 6 * t)))

    def say(self, message, *animations, color=MUTED, math=False):
        self.beat(self.note_anim(message, color, math), *animations)

    def plot(self, centres=INITIAL, labels=None, boundary=False):
        x_axis = Line([-5.28, -2.04, 0], [1.44, -2.04, 0], color=MUTED, stroke_width=2)
        y_axis = Line([-5.28, -2.04, 0], [-5.28, 2.36, 0], color=MUTED, stroke_width=2)
        ticks = VGroup()
        for x in (-2, 0, 2):
            px = map_point([x, 0])[0]
            ticks.add(Line([px, -2.04, 0], [px, -2.12, 0], color=MUTED, stroke_width=1.5))
            ticks.add(J(str(x), 16, MUTED).move_to([px, -2.30, 0]))
        for y in (-1, 0, 1, 2):
            py = map_point([0, y])[1]
            ticks.add(Line([-5.28, py, 0], [-5.36, py, 0], color=MUTED, stroke_width=1.5))
            ticks.add(J(str(y), 16, MUTED).move_to([-5.58, py, 0]))
        self.dots = VGroup(*[
            Dot(map_point(point), radius=.072, color=GREEN if labels is None else GROUP_COLORS[labels[i]])
            .set_z_index(4)
            for i, point in enumerate(POINTS)
        ])
        self.centres = VGroup(*[marker(c, k) for k, c in enumerate(centres)])
        self.cpos = np.asarray(centres, dtype=float).copy()
        self.border = bisector(centres) if boundary else VGroup()
        self.link_group = VGroup()
        self.add(x_axis, y_axis, ticks, self.border, self.dots, self.centres)
        return VGroup(x_axis, y_axis, ticks, self.border, self.dots, self.centres)

    def recolor(self, labels):
        return AnimationGroup(*[
            dot.animate.set_color(GROUP_COLORS[int(label)])
            for dot, label in zip(self.dots, labels)
        ], lag_ratio=.035)

    def move_centres(self, new):
        new = np.asarray(new)
        animations = [Transform(self.centres[k], marker(new[k], k)) for k in range(2)]
        if len(self.border.submobjects) or isinstance(self.border, Line):
            animations.append(Transform(self.border, bisector(new)))
        self.cpos = new.copy()
        return AnimationGroup(*animations, lag_ratio=0)

    def show_links(self, centres, labels):
        fresh = links(centres, labels)
        if len(self.link_group) == 0:
            self.link_group = fresh
            self.add(fresh)
            return FadeIn(fresh)
        return Transform(self.link_group, fresh)

    def show_cost(self, value, counts=None):
        fresh = cost_panel(value, counts)
        if not hasattr(self, "cost") or self.cost is None:
            self.cost = fresh
            self.add(fresh)
            return FadeIn(fresh, rate_func=early)
        return Transform(self.cost, fresh, rate_func=snap)

    def opening(self):
        self.plot(centres=INITIAL)
        self.cost = None
        self.say("9章：ラベルのない点を分ける", Indicate(self.dots[7], color=YELLOW))
        self.say("14点を2群へ", LaggedStart(*[Indicate(d, color=YELLOW) for d in self.dots], lag_ratio=.05))
        self.say("中心2つは左に置く", Indicate(self.centres[0], color=RED),
                 Indicate(self.centres[1], color=BLUE))
        self.say("右の点はどちらへ？", Indicate(self.dots[10], color=YELLOW))
        assignment = MAIN[0]["labels"]
        self.say("赤4点、青10点", self.recolor(assignment),
                 self.show_cost(MAIN[0]["J"], (4, 10)), color=GREEN)
        arrow = Arrow(map_point(INITIAL[1]) + DOWN * .15, map_point(MAIN[1]["centres"][1]) + DOWN * .15,
                      buff=.22, color=YELLOW, stroke_width=3)
        self.say("青い中心が右へ動く予告", GrowArrow(arrow), color=YELLOW)
        self.say("割当てと移動の謎", Indicate(self.centres[1], color=YELLOW))

    def distance(self):
        self.plot(labels=MAIN[0]["labels"], boundary=True)
        self.cost = None
        point = self.dots[6]
        red_line = Line(point.get_center(), map_point(INITIAL[0]), color=RED, stroke_width=2)
        blue_line = Line(point.get_center(), map_point(INITIAL[1]), color=BLUE, stroke_width=2)
        self.say("同じ点から2本の線", Create(red_line), Create(blue_line))
        horizontal = DashedLine(point.get_center(), [map_point(INITIAL[1])[0], point.get_y(), 0],
                                color=YELLOW, stroke_width=2)
        vertical = DashedLine(horizontal.get_end(), map_point(INITIAL[1]), color=YELLOW, stroke_width=2)
        self.say("横と縦のずれを測る", Create(horizontal), Create(vertical))
        formula = tex(r"\|x_n-\mu_k\|^2=\Delta x^2+\Delta y^2", 30, YELLOW)
        formula.scale_to_fit_width(4.3).move_to([4.10, .85, 0])
        self.say("二乗距離＝横²＋縦²", FadeIn(formula, rate_func=early), color=YELLOW)
        self.say("等距離の境界", Indicate(self.border, color=PURPLE))
        self.say("境界で色が替わる", Indicate(self.dots[6], color=YELLOW),
                 Indicate(self.dots[7], color=YELLOW))
        self.say("近い方へ必ず1票", Indicate(self.dots[6], color=BLUE))
        self.say("色だけでは改善を測れない", FadeOut(red_line), FadeOut(blue_line),
                 FadeOut(horizontal), FadeOut(vertical))

    def score(self):
        self.plot(labels=MAIN[0]["labels"])
        self.cost = None
        recap = J("復習: 1.1 二乗誤差", 21).move_to([4.08, 2.36, 0])
        frame = SurroundingRectangle(recap, color=YELLOW, stroke_width=1.2, buff=.13)
        areas = VGroup(
            Square(side_length=.42, stroke_color=BLUE, fill_color=BLUE,
                   fill_opacity=.28).move_to([3.28, .48, 0]),
            Square(side_length=.84, stroke_color=YELLOW, fill_color=YELLOW,
                   fill_opacity=.28).move_to([4.78, .48, 0]),
            J("短い", 18, BLUE).move_to([3.28, -.22, 0]),
            J("長い", 18, YELLOW).move_to([4.78, -.22, 0]),
        )
        self.say("復習 1.1：ずれを二乗", FadeIn(recap, rate_func=early),
                 Create(frame, rate_func=early), FadeIn(areas, rate_func=early), color=YELLOW)
        self.say("長い線ほど大きな面積", self.show_links(INITIAL, MAIN[0]["labels"]))
        self.remove(areas)
        self.say("14個の面積を足す", self.show_cost(91.4, (4, 10)), color=GREEN)
        self.remove(recap, frame)
        self.remove(self.cost)
        self.cost = None
        table = VGroup(J("担当の印", 22, MUTED).move_to([4.10, 2.30, 0]),
                       tex(r"r_{nk}\in\{0,1\}", 29, YELLOW).move_to([4.10, 1.82, 0]))
        self.say("担当だけ1、他は0", FadeIn(table, rate_func=early), color=YELLOW)
        formula = tex(r"J=\sum_n\sum_k r_{nk}\|x_n-\mu_k\|^2", 30, GREEN)
        formula.scale_to_fit_width(4.5).move_to([4.10, -.68, 0])
        self.say("担当した距離だけ足す", FadeIn(formula, rate_func=early), color=GREEN)
        self.remove(table)
        self.cost = None
        self.add(cost_panel(91.4, (4, 10)))
        self.cost = self.mobjects[-1]
        self.say("最初のひずみ J=91.400", Indicate(self.cost[1], color=YELLOW))
        self.say("次は色と中心を交互に直す", Indicate(self.dots[7], color=YELLOW),
                 Indicate(self.centres[1], color=YELLOW))

    def e_step(self):
        self.plot(centres=INITIAL, boundary=True)
        self.cost = None
        self.say("中心は固定", Indicate(self.centres[0], color=RED), Indicate(self.centres[1], color=BLUE))
        red_line = Line(map_point(POINTS[3]), map_point(INITIAL[0]), color=RED, stroke_width=2)
        blue_line = Line(map_point(POINTS[3]), map_point(INITIAL[1]), color=BLUE, stroke_width=2)
        self.say("この点の距離を比べる", Create(red_line), Create(blue_line))
        self.say("短い方へ色を付ける", self.recolor(MAIN[0]["labels"]), color=YELLOW)
        self.say("Eステップ：近い中心へ", Indicate(self.border, color=PURPLE), color=GREEN)
        self.say("境界を越えると担当が替わる", Indicate(self.dots[6], color=YELLOW))
        self.say("4対10、J=91.400", self.show_cost(91.4, (4, 10)), color=GREEN)
        self.say("次は担当を固定して中心へ", FadeOut(red_line), FadeOut(blue_line),
                 Indicate(self.centres[1], color=YELLOW))

    def m_step(self):
        self.plot(centres=INITIAL, labels=MAIN[0]["labels"], boundary=True)
        self.cost = None
        self.say("担当は4点と10点のまま", self.show_cost(91.4, (4, 10)))
        red_mean = MAIN[1]["centres"][0]
        guides = VGroup(
            DashedLine(map_point([red_mean[0], -1.90]), map_point([red_mean[0], red_mean[1]]),
                       color=YELLOW, stroke_width=2),
            DashedLine(map_point([-3.0, red_mean[1]]), map_point(red_mean),
                       color=YELLOW, stroke_width=2))
        self.say("赤4点の横と縦を平均", FadeIn(guides), color=YELLOW)
        self.say("青10点もそれぞれ平均", self.show_links(INITIAL, MAIN[0]["labels"]))
        formula = tex(r"\mu_k=\frac{\sum_n r_{nk}x_n}{\sum_n r_{nk}}", 29, YELLOW)
        formula.move_to([4.1, -.50, 0])
        self.say("分母は担当した点数", FadeIn(formula, rate_func=early), color=YELLOW)
        self.say("Mステップ：平均へ移動", self.move_centres(MAIN[1]["centres"]),
                 self.show_links(MAIN[1]["centres"], MAIN[1]["labels"]),
                 FadeOut(self.cost, rate_func=early), color=GREEN)
        self.cost = None
        self.say("J: 91.400 → 28.588", self.show_cost(MAIN[1]["J"], (4, 10)), color=GREEN)
        self.say("境界が動いたので、もう1度Eへ", Indicate(self.border, color=PURPLE))

    def iterate(self):
        self.plot(centres=MAIN[1]["centres"], labels=MAIN[1]["labels"], boundary=True)
        self.cost = None
        self.trace = cost_trace(2)
        self.say("中心が動き、境界も回った", self.show_cost(MAIN[1]["J"], (4, 10)),
                 FadeIn(self.trace, rate_func=early))
        self.say("2回目のEで赤6、青8", self.recolor(MAIN[2]["labels"]),
                 self.show_cost(MAIN[2]["J"], (6, 8)),
                 Transform(self.trace, cost_trace(3), rate_func=snap), color=YELLOW)
        self.say("Jは15.420まで下がる", Indicate(self.cost[1], color=GREEN), color=GREEN)
        self.say("2回目のMで中心を平均へ", self.move_centres(MAIN[3]["centres"]), color=GREEN)
        self.say("Jは11.264まで下がる", self.show_cost(MAIN[3]["J"], (6, 8)),
                 Transform(self.trace, cost_trace(4), rate_func=snap), color=GREEN)
        self.say("もう1度Eでも色は同じ", Indicate(self.dots[6], color=YELLOW))
        self.say("中心も動かず、ここで停止", Indicate(self.centres[0], color=RED),
                 Indicate(self.centres[1], color=BLUE))

    def local_minimum(self):
        self.plot(centres=MAIN[3]["centres"], labels=MAIN[3]["labels"], boundary=True)
        self.cost = None
        self.say("同じ14点で初期値だけ変更", self.show_cost(MAIN[3]["J"], (6, 8)))
        neutral = AnimationGroup(*[d.animate.set_color(GREEN) for d in self.dots], lag_ratio=.03)
        self.say("片方を右から始める", self.move_centres(ALTERNATE), neutral,
                 FadeOut(self.cost, rate_func=early), color=YELLOW)
        self.cost = None
        self.say("もう1度EとMを繰り返す", self.recolor(OTHER[-1]["labels"]),
                 self.move_centres(OTHER[-1]["centres"]), color=GREEN)
        self.say("今度は J=10.560", self.show_cost(OTHER[-1]["J"], (7, 7)), color=GREEN)
        near = self.dots[6]
        self.say("境界近くの点が別の群に", Indicate(near, color=YELLOW), color=YELLOW)
        comparison = VGroup(tex("11.264", 28, PURPLE).move_to([3.25, -.6, 0]),
                            tex("10.560", 28, GREEN).move_to([4.95, -.6, 0]))
        self.say("停止しても最良とは限らない", FadeIn(comparison, rate_func=early), color=YELLOW)
        self.say("初期値を変えて比べる", Indicate(self.centres[0], color=RED),
                 Indicate(self.centres[1], color=BLUE))

    def pixel_mobject(self, array):
        big = Image.fromarray(array).resize((640, 480), Image.Resampling.NEAREST)
        return ImageMobject(np.array(big)).scale_to_fit_width(5.2).move_to([-2.28, .15, 0])

    def pixels(self):
        image = toy_rgb_image()
        k2, c2, _, e2 = quantize(image, 2)
        k4, c4, _, e4 = quantize(image, 4)
        picture = self.pixel_mobject(image)
        self.add(picture)
        pixel_mark = Square(side_length=.325, stroke_color=YELLOW, stroke_width=2.5,
                            fill_opacity=0).move_to([-4.39, 1.61, 0])
        self.say("自作の16×12画素の絵", FadeIn(pixel_mark, rate_func=early))
        legend = VGroup(J("1画素＝赤・緑・青", 23, MUTED).move_to([4.0, 1.80, 0]),
                        tex(r"x=(R,G,B)", 29, BLUE).move_to([4.0, 1.22, 0]))
        self.say("色を3つの数で表す", FadeIn(legend, rate_func=early), color=BLUE)
        self.remove(pixel_mark)
        knob = J("K = 2", 30, YELLOW).move_to([4.0, .32, 0])
        self.say("2色にすると？", FadeIn(knob, rate_func=early), color=YELLOW)
        picture2 = self.pixel_mobject(k2)
        self.say("K = 2：色を置き換える",
                 FadeOut(picture, rate_func=early),
                 FadeIn(picture2, rate_func=early), color=GREEN)
        picture = picture2
        bits2 = J("240ビット", 25, GREEN).move_to([4.0, -.55, 0])
        self.say("少ない色は画質も変える", FadeIn(bits2, rate_func=early), color=GREEN)
        picture4 = self.pixel_mobject(k4)
        knob4 = J("K = 4", 30, YELLOW).move_to(knob)
        bits4 = J("480ビット", 25, GREEN).move_to(bits2)
        bit_note = J("元 4608 → 2色 240 → 4色 480", 20, MUTED).move_to([4.0, -1.35, 0])
        bit_note.scale_to_fit_width(4.5)
        self.say("K = 4：色が増える",
                 FadeOut(picture, rate_func=early),
                 FadeIn(picture4, rate_func=early),
                 Transform(knob, knob4, rate_func=snap),
                 Transform(bits2, bits4, rate_func=snap),
                 FadeIn(bit_note, rate_func=early), color=GREEN)
        picture = picture4
        separate_pixels = VGroup(*[
            Square(side_length=.325, stroke_color=YELLOW, stroke_width=2.5,
                   fill_opacity=0).move_to(point)
            for point in ([-4.39, 1.61, 0], [-2.76, 1.61, 0])
        ])
        self.say("画素の位置関係は見ていない",
                 FadeIn(separate_pixels, rate_func=early))

    def return_to_points(self):
        self.plot(centres=INITIAL)
        self.cost = None
        self.say("冒頭と同じ14点・同じ初期値", Indicate(self.centres[1], color=BLUE))
        self.say("E：赤4点、青10点", self.recolor(MAIN[0]["labels"]),
                 self.show_cost(91.4, (4, 10)), color=YELLOW)
        self.say("M：青は10点の平均へ", self.move_centres(MAIN[1]["centres"]),
                 self.show_cost(MAIN[1]["J"], (4, 10)), color=GREEN)
        self.say("またE、Mで J=11.264", self.recolor(MAIN[2]["labels"]),
                 self.move_centres(MAIN[3]["centres"]),
                 self.show_cost(MAIN[3]["J"], (6, 8)), color=GREEN)
        alternate_result = VGroup(
            J("別の初期値", 20, PURPLE),
            tex("J=10.560", 27, PURPLE)
        ).arrange(DOWN, buff=.12).move_to([4.1, -.73, 0])
        self.say("別の初期値では J=10.560",
                 FadeIn(alternate_result, rate_func=early), color=PURPLE)
        self.border = bisector(MAIN[3]["centres"])
        self.say("境界の点も1群へ確定", FadeIn(self.border, rate_func=early),
                 Indicate(self.dots[6], color=YELLOW))
        self.say("次は迷いを確率で表す", Indicate(self.dots[6], color=PURPLE), color=PURPLE)
