from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
TARGET_GREEN = GREEN_C
ONLINE_ORANGE = ORANGE
GRAD_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def target_function(x: float) -> float:
    return 0.58 + 0.24 * math.sin(1.8 * math.pi * x) + 0.12 * math.cos(3.2 * math.pi * x)


def network_prediction(x: float, a: float, b: float) -> float:
    return 0.55 + 0.22 * math.tanh(a * (x - 0.35)) - 0.17 * math.tanh(b * (x - 0.75))


def make_training_data() -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.08, 0.94, 12)
    offsets = np.array([0.02, -0.035, 0.018, 0.04, -0.025, 0.01, -0.018, 0.03, -0.02, 0.016, -0.026, 0.018])
    ys = np.array([target_function(float(x)) for x in xs]) + offsets
    return xs, ys


def error_value(point: np.ndarray) -> float:
    x, y = point
    return (
        0.22 * (x + 2.2) ** 2
        + 0.95 * (y + 1.05) ** 2
        + 0.16 * math.sin(2.1 * x + 0.4)
        + 0.1 * math.cos(2.4 * y)
    )


class PRML52NetworkTraining(Scene):
    """PRML 5.2 network training overview.

    Render example:
        uv run manim -pql prml_5_2_network_training.py PRML52NetworkTraining
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.training_x, self.training_y = make_training_data()

        self.learning_as_error_minimization()
        self.output_error_pairings()
        self.weight_space_descent()
        self.local_quadratic_approximation()
        self.why_gradient_information()
        self.batch_and_online_updates()

    def start_narration(self, scene_id: str) -> tuple[float, float | None]:
        audio_path = VOICEOVER_DIR / f"{scene_id}.wav"
        start_time = float(getattr(self, "time", 0.0))
        if not audio_path.exists():
            return start_time, None
        self.add_sound(str(audio_path))
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        return start_time, duration

    def finish_narration(self, narration: tuple[float, float | None], pad: float = 0.2) -> None:
        start_time, duration = narration
        if duration is None:
            return
        elapsed = float(getattr(self, "time", 0.0)) - start_time
        remaining = duration - elapsed + pad
        if remaining > 0:
            self.wait(remaining)

    def section_label(self, text: str) -> Text:
        label = Text(text, font_size=18, color=TEXT_GREY)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 34) -> Text:
        title = Text(text, font_size=font_size)
        title.to_edge(UP).shift(DOWN * 0.34)
        return title

    def make_network_diagram(self) -> VGroup:
        layer_x = [-1.8, 0.0, 1.8]
        layer_counts = [2, 4, 1]
        nodes = []
        edges = VGroup()
        for x, count in zip(layer_x, layer_counts):
            ys = np.linspace(-0.75, 0.75, count)
            layer = VGroup(*[Circle(radius=0.16, color=BLUE_D, fill_color=BLUE_E, fill_opacity=0.55).move_to([x, y, 0]) for y in ys])
            nodes.append(layer)
        for left, right in zip(nodes[:-1], nodes[1:]):
            for a in left:
                for b in right:
                    edges.add(Line(a.get_center(), b.get_center(), color=GREY_D, stroke_width=1.6))
        labels = VGroup(
            Text("x", font_size=22, color=BLUE_DATA).next_to(nodes[0], DOWN, buff=0.12),
            Text("hidden", font_size=18, color=TEXT_GREY).next_to(nodes[1], DOWN, buff=0.12),
            Text("y(x,w)", font_size=22, color=MODEL_RED).next_to(nodes[2], DOWN, buff=0.12),
        )
        return VGroup(edges, *nodes, labels)

    def make_curve_axes(self, width: float = 5.8, height: float = 3.5) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 1, 0.25],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_curve(self, axes: Axes, a: float, b: float, color: ManimColor = MODEL_RED) -> VMobject:
        return axes.plot(lambda x: network_prediction(x, a, b), x_range=[0, 1], color=color, stroke_width=4)

    def make_training_points(self, axes: Axes) -> VGroup:
        return VGroup(*[Dot(axes.c2p(float(x), float(y)), radius=0.055, color=BLUE_DATA) for x, y in zip(self.training_x, self.training_y)])

    def make_residuals(self, axes: Axes, a: float, b: float) -> VGroup:
        lines = VGroup()
        for x, target in zip(self.training_x, self.training_y):
            pred = network_prediction(float(x), a, b)
            lines.add(Line(axes.c2p(float(x), float(target)), axes.c2p(float(x), pred), color=YELLOW, stroke_width=2.2))
        return lines

    def make_contours(self, center: np.ndarray = np.array([0.0, 0.0, 0.0]), color: ManimColor = BLUE_D) -> VGroup:
        contours = VGroup()
        for width, height, opacity in [(1.2, 0.55, 0.9), (2.0, 0.9, 0.75), (2.9, 1.35, 0.58), (3.9, 1.85, 0.42), (5.1, 2.45, 0.3)]:
            contours.add(Ellipse(width=width, height=height, color=color, stroke_width=2.4, stroke_opacity=opacity).rotate(0.45).move_to(center))
        return contours

    def path_from_points(self, points: list[np.ndarray], color: ManimColor, width: float = 4.0) -> VGroup:
        dots = VGroup(*[Dot(p, radius=0.065, color=color) for p in points])
        arrows = VGroup(*[Arrow(a, b, buff=0.08, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.18) for a, b in zip(points[:-1], points[1:])])
        return VGroup(arrows, dots)

    def learning_as_error_minimization(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 5.2 Network Training")
        title = self.scene_title("学習: 重み w を動かして誤差 E(w) を下げる", font_size=32)
        network = self.make_network_diagram().scale(0.92).to_edge(LEFT).shift(RIGHT * 0.45 + DOWN * 0.15)
        axes = self.make_curve_axes().to_edge(RIGHT).shift(LEFT * 0.35 + DOWN * 0.2)
        true_curve = axes.plot(target_function, x_range=[0, 1], color=TARGET_GREEN, stroke_width=3.2)
        points = self.make_training_points(axes)
        rough_curve = self.make_curve(axes, 1.0, 1.0, MODEL_RED)
        tuned_curve = self.make_curve(axes, 3.8, 3.0, MODEL_RED)
        residuals = self.make_residuals(axes, 1.0, 1.0)
        tuned_residuals = self.make_residuals(axes, 3.8, 3.0)
        formula = MathTex(r"E(w)=\frac{1}{2}\sum_n\|y(x_n,w)-t_n\|^2", font_size=34).next_to(axes, DOWN, buff=0.25)
        knob = VGroup(
            Text("w", font_size=26, color=MODEL_RED),
            Line(LEFT * 0.9, RIGHT * 0.9, color=GREY_B, stroke_width=4),
            Dot(LEFT * 0.55, color=MODEL_RED, radius=0.09),
        ).arrange(DOWN, buff=0.14).next_to(network, DOWN, buff=0.35)
        knob_tuned = Dot(RIGHT * 0.55, color=MODEL_RED, radius=0.09).move_to(knob[2])
        knob_tuned.shift(RIGHT * 1.1)

        self.play(FadeIn(label), Write(title), FadeIn(network), run_time=1.6)
        self.play(Create(axes), Create(true_curve), FadeIn(points), Write(formula), FadeIn(knob), run_time=2.0)
        self.play(Create(rough_curve), Create(residuals), run_time=1.8)
        self.play(
            ReplacementTransform(rough_curve, tuned_curve),
            ReplacementTransform(residuals, tuned_residuals),
            knob[2].animate.move_to(knob_tuned),
            run_time=2.4,
        )
        note = Text("最大尤度  ->  二乗和誤差の最小化", font_size=24, color=TARGET_GREEN).next_to(formula, DOWN, buff=0.12)
        self.play(Write(note), run_time=1.4)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def output_error_pairings(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Probabilistic output view")
        title = self.scene_title("出力ユニットと誤差関数は、確率モデルで組になる", font_size=31)
        headers = ["問題", "出力", "誤差関数"]
        rows = [
            ("回帰", "linear", r"\frac{1}{2}\sum (y-t)^2"),
            ("二値分類", "sigmoid", r"-t\log y-(1-t)\log(1-y)"),
            ("多クラス分類", "softmax", r"-\sum_k t_k\log y_k"),
        ]
        table = VGroup()
        for i, header in enumerate(headers):
            cell = Text(header, font_size=24, color=TEXT_GREY)
            cell.move_to([-3.8 + i * 3.75, 1.45, 0])
            table.add(cell)
        for r, row in enumerate(rows):
            y = 0.65 - r * 0.95
            bg = Rectangle(width=10.8, height=0.72, stroke_color=GREY_D, stroke_width=1.2, fill_color="#1a1a1a", fill_opacity=0.9)
            bg.move_to([0, y, 0])
            table.add(bg)
            table.add(Text(row[0], font_size=25, color=WHITE).move_to([-3.8, y, 0]))
            table.add(Text(row[1], font_size=25, color=TARGET_GREEN).move_to([-0.05, y, 0]))
            table.add(MathTex(row[2], font_size=28, color=MODEL_RED).move_to([3.8, y, 0]))
        derivative = VGroup(
            Text("共通して扱いやすい出力層の形", font_size=24, color=TEXT_GREY),
            MathTex(r"\frac{\partial E}{\partial a_k}=y_k-t_k", font_size=42, color=YELLOW),
        ).arrange(DOWN, buff=0.18).to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(LaggedStart(*[FadeIn(mob) for mob in table], lag_ratio=0.08), run_time=3.0)
        self.play(Write(derivative), run_time=1.7)
        self.play(Indicate(derivative[1], color=YELLOW), run_time=1.0)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def weight_space_descent(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Parameter optimization")
        title = self.scene_title("勾配は上り坂、マイナス勾配は下り坂", font_size=32)
        center_global = np.array([-2.2, -0.75, 0.0])
        center_local = np.array([2.4, 0.85, 0.0])
        contours = VGroup(
            self.make_contours(center_global, BLUE_D),
            self.make_contours(center_local, PURPLE_D).scale(0.75),
        )
        local_label = Text("local minimum", font_size=22, color=PURPLE_B).next_to(center_local, UP, buff=0.9)
        global_label = Text("global minimum", font_size=22, color=BLUE_B).next_to(center_global, DOWN, buff=0.85)
        start = np.array([1.25, -2.15, 0.0])
        p1 = np.array([0.45, -1.55, 0.0])
        p2 = np.array([-0.55, -1.25, 0.0])
        p3 = np.array([-1.35, -0.95, 0.0])
        p4 = center_global + np.array([0.18, 0.12, 0.0])
        path = self.path_from_points([start, p1, p2, p3, p4], TARGET_GREEN)
        grad_arrow = Arrow(start, start + np.array([0.68, 0.72, 0.0]), buff=0, color=MODEL_RED, stroke_width=5)
        neg_grad_arrow = Arrow(start, start + np.array([-0.68, 0.52, 0.0]), buff=0, color=TARGET_GREEN, stroke_width=5)
        grad_labels = VGroup(
            MathTex(r"\nabla E", font_size=30, color=MODEL_RED).next_to(grad_arrow, RIGHT, buff=0.08),
            MathTex(r"-\nabla E", font_size=30, color=TARGET_GREEN).next_to(neg_grad_arrow, LEFT, buff=0.08),
        )
        update = MathTex(r"w^{(\tau+1)}=w^{(\tau)}+\Delta w^{(\tau)}", font_size=35).to_edge(DOWN).shift(UP * 0.2)
        stationary = MathTex(r"\nabla E(w)=0", font_size=32, color=YELLOW).next_to(center_global, RIGHT, buff=0.45)

        self.play(FadeIn(label), Write(title), Create(contours), run_time=2.0)
        self.play(FadeIn(local_label), FadeIn(global_label), Create(grad_arrow), Create(neg_grad_arrow), Write(grad_labels), run_time=2.0)
        self.play(Create(path), Write(update), run_time=2.4)
        self.play(Write(stationary), Indicate(p4 := path[1][-1], color=YELLOW), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def local_quadratic_approximation(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Local quadratic approximation")
        title = self.scene_title("最小点の近くでは、誤差面は楕円の等高線に見える", font_size=31)
        center = np.array([-1.25, -0.1, 0.0])
        ellipses = self.make_contours(center, BLUE_D).scale(1.12)
        minimum = Dot(center, color=YELLOW, radius=0.08)
        min_label = MathTex(r"w^\ast", font_size=32, color=YELLOW).next_to(minimum, DOWN, buff=0.12)
        u1 = Arrow(center, center + np.array([1.8, 0.86, 0.0]), buff=0.1, color=TARGET_GREEN, stroke_width=5)
        u2 = Arrow(center, center + np.array([-0.52, 1.08, 0.0]), buff=0.1, color=ONLINE_ORANGE, stroke_width=5)
        u_labels = VGroup(
            MathTex(r"u_1\;(\lambda_1\ \mathrm{small})", font_size=26, color=TARGET_GREEN).next_to(u1, RIGHT, buff=0.1),
            MathTex(r"u_2\;(\lambda_2\ \mathrm{large})", font_size=26, color=ONLINE_ORANGE).next_to(u2, UP, buff=0.1),
        )
        formula = MathTex(
            r"E(w)\simeq E(w^\ast)+\frac{1}{2}\sum_i\lambda_i\alpha_i^2",
            font_size=35,
            color=WHITE,
        ).to_edge(RIGHT).shift(LEFT * 0.25 + UP * 0.9)
        notes = VGroup(
            Text("Hessian H が曲がり方を表す", font_size=25, color=TEXT_GREY),
            Text("急な方向とゆるい方向が混ざる", font_size=25, color=TEXT_GREY),
            Text("単純な勾配降下はジグザグしやすい", font_size=25, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(formula, DOWN, buff=0.45).align_to(formula, LEFT)
        zigzag_points = [
            center + np.array([2.1, -1.35, 0.0]),
            center + np.array([0.9, 0.75, 0.0]),
            center + np.array([0.18, -0.52, 0.0]),
            center + np.array([-0.08, 0.22, 0.0]),
        ]
        zigzag = self.path_from_points(zigzag_points, MODEL_RED, width=3.4)

        self.play(FadeIn(label), Write(title), Create(ellipses), FadeIn(minimum), Write(min_label), run_time=2.0)
        self.play(Create(u1), Create(u2), Write(u_labels), run_time=1.8)
        self.play(Write(formula), FadeIn(notes[0]), FadeIn(notes[1]), run_time=2.0)
        self.play(Create(zigzag), FadeIn(notes[2]), run_time=1.8)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def why_gradient_information(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("Use of gradient information")
        title = self.scene_title("勾配は、進む向きの情報をまとめてくれる", font_size=32)
        left_box = Rectangle(width=5.2, height=3.7, color=GREY_C, stroke_width=1.5).shift(LEFT * 3.05 + DOWN * 0.15)
        right_box = Rectangle(width=5.2, height=3.7, color=GREY_C, stroke_width=1.5).shift(RIGHT * 3.05 + DOWN * 0.15)
        left_title = Text("関数値だけ", font_size=27, color=TEXT_GREY).next_to(left_box, UP, buff=0.2)
        right_title = Text("勾配あり", font_size=27, color=TARGET_GREEN).next_to(right_box, UP, buff=0.2)
        probe_points = VGroup(
            Dot(left_box.get_center() + LEFT * 1.4 + UP * 0.6, color=BLUE_DATA),
            Dot(left_box.get_center() + RIGHT * 0.2 + UP * 1.0, color=BLUE_DATA),
            Dot(left_box.get_center() + LEFT * 0.35 + DOWN * 0.55, color=BLUE_DATA),
            Dot(left_box.get_center() + RIGHT * 1.35 + DOWN * 0.85, color=BLUE_DATA),
        )
        question_marks = VGroup(*[Text("?", font_size=28, color=YELLOW).next_to(dot, UP, buff=0.08) for dot in probe_points])
        gradient_point = Dot(right_box.get_center() + LEFT * 0.2 + DOWN * 0.2, color=YELLOW, radius=0.08)
        gradient_vector = Arrow(gradient_point.get_center(), gradient_point.get_center() + np.array([-1.15, 0.85, 0.0]), buff=0.04, color=MODEL_RED, stroke_width=5)
        descent_vector = Arrow(gradient_point.get_center(), gradient_point.get_center() + np.array([1.15, -0.85, 0.0]), buff=0.04, color=TARGET_GREEN, stroke_width=5)
        components = VGroup(
            MathTex(r"\nabla E=(g_1,g_2,\ldots,g_W)", font_size=32, color=TARGET_GREEN),
            Text("1 回で W 成分の情報", font_size=25, color=TARGET_GREEN),
            Text("backprop: O(W) で勾配評価", font_size=24, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.22).next_to(right_box, DOWN, buff=0.28)
        left_note = VGroup(
            MathTex(r"E(w)", font_size=32, color=BLUE_DATA),
            Text("1 回で 1 つの値", font_size=24, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.18).next_to(left_box, DOWN, buff=0.28)

        self.play(FadeIn(label), Write(title), Create(left_box), Create(right_box), Write(left_title), Write(right_title), run_time=1.7)
        self.play(FadeIn(probe_points), FadeIn(question_marks), Write(left_note), run_time=1.7)
        self.play(FadeIn(gradient_point), Create(gradient_vector), Create(descent_vector), Write(components), run_time=2.2)
        self.play(Indicate(components[0], color=TARGET_GREEN), run_time=1.0)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def batch_and_online_updates(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Gradient descent optimization")
        title = self.scene_title("データをまとめて使うか、小刻みに使うか", font_size=32)
        left_center = np.array([-3.0, 0.25, 0.0])
        right_center = np.array([3.0, 0.25, 0.0])
        batch_contours = self.make_contours(left_center, BLUE_D).scale(0.78)
        online_contours = self.make_contours(right_center, BLUE_D).scale(0.78)
        batch_points = [
            left_center + np.array([1.95, -1.35, 0.0]),
            left_center + np.array([0.95, -0.75, 0.0]),
            left_center + np.array([0.25, -0.28, 0.0]),
            left_center + np.array([-0.08, -0.05, 0.0]),
        ]
        online_points = [
            right_center + np.array([1.95, -1.35, 0.0]),
            right_center + np.array([1.18, -0.35, 0.0]),
            right_center + np.array([0.55, -0.75, 0.0]),
            right_center + np.array([0.12, 0.18, 0.0]),
            right_center + np.array([-0.28, -0.12, 0.0]),
            right_center + np.array([-0.05, 0.05, 0.0]),
        ]
        batch_path = self.path_from_points(batch_points, TARGET_GREEN, width=4.0)
        online_path = self.path_from_points(online_points, ONLINE_ORANGE, width=3.5)
        batch_label = Text("batch", font_size=29, color=TARGET_GREEN).next_to(batch_contours, UP, buff=0.22)
        online_label = Text("online / stochastic", font_size=29, color=ONLINE_ORANGE).next_to(online_contours, UP, buff=0.22)
        formulas = VGroup(
            MathTex(r"w^{(\tau+1)}=w^{(\tau)}-\eta\nabla E(w^{(\tau)})", font_size=31, color=TARGET_GREEN),
            MathTex(r"w^{(\tau+1)}=w^{(\tau)}-\eta\nabla E_n(w^{(\tau)})", font_size=31, color=ONLINE_ORANGE),
        ).arrange(DOWN, buff=0.22).to_edge(DOWN).shift(UP * 0.15)
        data_batch = VGroup(*[Dot(left_center + np.array([-1.7 + i * 0.28, -1.7 + 0.08 * math.sin(i), 0]), color=BLUE_DATA, radius=0.045) for i in range(13)])
        data_online = VGroup(*[Dot(right_center + np.array([-1.7 + i * 0.28, -1.7 + 0.08 * math.sin(i), 0]), color=BLUE_DATA if i != 5 else YELLOW, radius=0.045) for i in range(13)])
        notes = VGroup(
            Text("全データで安定した一歩", font_size=22, color=TEXT_GREY).next_to(batch_label, DOWN, buff=0.1),
            Text("データ点ごとに揺らぎながら進む", font_size=22, color=TEXT_GREY).next_to(online_label, DOWN, buff=0.1),
        )
        summary = Text("誤差関数 + 勾配 + 学習率 eta + データの使い方", font_size=27, color=YELLOW).to_edge(DOWN).shift(UP * 0.02)

        self.play(FadeIn(label), Write(title), Create(batch_contours), Create(online_contours), Write(batch_label), Write(online_label), run_time=2.0)
        self.play(FadeIn(data_batch), FadeIn(data_online), FadeIn(notes), run_time=1.3)
        self.play(Create(batch_path), Write(formulas[0]), run_time=1.8)
        self.play(Create(online_path), Write(formulas[1]), run_time=2.0)
        self.play(FadeOut(formulas), Write(summary), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
