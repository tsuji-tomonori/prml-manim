from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


CLASS_RED = RED_C
CLASS_BLUE = BLUE_C
CLASS_GREEN = GREEN_C
ACCENT = ORANGE
MODEL_YELLOW = YELLOW
PROBIT_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def gaussian_pdf(x: np.ndarray | float, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray | float:
    return np.exp(-0.5 * ((np.asarray(x) - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def probit(x: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in x])


class PRML43ProbabilisticDiscriminativeModels(Scene):
    """PRML 4.3 probabilistic discriminative models overview.

    Render example:
        uv run manim -pql prml_4_3_probabilistic_discriminative_models.py PRML43ProbabilisticDiscriminativeModels
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.opening_direct_posterior()
        self.fixed_basis_functions()
        self.logistic_regression()
        self.cross_entropy_gradient()
        self.irls_weighted_least_squares()
        self.multiclass_softmax()
        self.probit_regression()
        self.canonical_link_summary()

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
        title.to_edge(UP).shift(DOWN * 0.35)
        return title

    def make_box(self, width: float, height: float, stroke_color: ManimColor, fill_color: ManimColor | None = None) -> Rectangle:
        return Rectangle(
            width=width,
            height=height,
            stroke_color=stroke_color,
            stroke_width=2.2,
            fill_color=fill_color or stroke_color,
            fill_opacity=0.14,
        )

    def fade_all(self) -> None:
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def make_path_step(self, title: str, formula: str, color: ManimColor) -> VGroup:
        box = self.make_box(2.35, 0.82, color)
        title_text = Text(title, font_size=18, color=color)
        formula_text = MathTex(formula, font_size=27)
        group = VGroup(box, title_text, formula_text)
        title_text.next_to(box.get_top(), DOWN, buff=0.1)
        formula_text.next_to(title_text, DOWN, buff=0.08)
        return group

    def opening_direct_posterior(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 4.3 Probabilistic Discriminative Models")
        title = self.scene_title("分類に必要な確率を、直接合わせる", font_size=34)

        generative_inputs = VGroup(
            self.make_path_step("class density", r"p(x|C_k)", CLASS_BLUE),
            self.make_path_step("class prior", r"p(C_k)", CLASS_GREEN),
        ).arrange(DOWN, buff=0.28)
        bayes_step = self.make_path_step("Bayes", r"p(C_k|x)", ACCENT).next_to(generative_inputs, RIGHT, buff=0.75)
        generative_steps = VGroup(generative_inputs, bayes_step)
        gen_arrows = VGroup(
            Arrow(generative_inputs[0].get_right(), bayes_step.get_left() + UP * 0.14, buff=0.08, color=TEXT_GREY),
            Arrow(generative_inputs[1].get_right(), bayes_step.get_left() + DOWN * 0.14, buff=0.08, color=TEXT_GREY),
        )
        gen_title = Text("generative route", font_size=24, color=CLASS_BLUE).next_to(generative_steps, UP, buff=0.25)
        generative = VGroup(gen_title, generative_steps, gen_arrows).to_edge(LEFT).shift(UP * 0.8 + RIGHT * 0.35)

        discriminative_steps = VGroup(
            self.make_path_step("parameters", r"w", PROBIT_PURPLE),
            self.make_path_step("direct posterior", r"p(C_k|x)", ACCENT),
        ).arrange(RIGHT, buff=0.4)
        disc_arrow = Arrow(discriminative_steps[0].get_right(), discriminative_steps[1].get_left(), buff=0.08, color=TEXT_GREY)
        disc_title = Text("discriminative route", font_size=24, color=ACCENT).next_to(discriminative_steps, UP, buff=0.25)
        discriminative = VGroup(disc_title, discriminative_steps, disc_arrow).next_to(generative, DOWN, buff=0.9).align_to(generative, LEFT)

        m_value = 8
        gen_params = m_value * (m_value + 5) / 2 + 1
        disc_params = m_value
        chart_axis = Line(LEFT * 2.65, RIGHT * 2.65, color=TEXT_GREY, stroke_width=2)
        chart_axis.to_corner(DR).shift(UP * 1.2 + LEFT * 0.45)
        gen_bar = Rectangle(width=4.1, height=0.28, fill_color=CLASS_BLUE, fill_opacity=0.8, stroke_width=0)
        disc_bar = Rectangle(width=0.72, height=0.28, fill_color=ACCENT, fill_opacity=0.85, stroke_width=0)
        gen_bar.next_to(chart_axis.get_left(), UP, buff=0.55).align_to(chart_axis, LEFT)
        disc_bar.next_to(chart_axis.get_left(), DOWN, buff=0.35).align_to(chart_axis, LEFT)
        bar_labels = VGroup(
            Text(f"generative Gaussian: {gen_params:.0f} params", font_size=22, color=CLASS_BLUE).next_to(gen_bar, UP, buff=0.1).align_to(gen_bar, LEFT),
            Text(f"logistic: {disc_params} params", font_size=22, color=ACCENT).next_to(disc_bar, DOWN, buff=0.1).align_to(disc_bar, LEFT),
            Text("example: feature dimension M = 8", font_size=19, color=TEXT_GREY).next_to(chart_axis, DOWN, buff=0.75),
        )
        parameter_chart = VGroup(chart_axis, gen_bar, disc_bar, bar_labels)

        note = Text("density assumptions が外れると、直接学習が有利になることがある", font_size=23, color=MODEL_YELLOW)
        note.to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title), run_time=1.3)
        self.play(FadeIn(generative, shift=RIGHT * 0.2), run_time=1.8)
        self.play(FadeIn(discriminative, shift=RIGHT * 0.2), run_time=1.8)
        self.play(FadeIn(parameter_chart), run_time=1.7)
        self.play(Write(note), Indicate(discriminative_steps[1], color=MODEL_YELLOW), run_time=1.7)
        self.finish_narration(narration)
        self.fade_all()

    def input_axes(self) -> Axes:
        return Axes(
            x_range=[-2.4, 2.4, 1.2],
            y_range=[-1.8, 1.8, 0.9],
            x_length=5.0,
            y_length=3.8,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        )

    def feature_axes(self) -> Axes:
        return Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.5],
            x_length=4.6,
            y_length=3.8,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        )

    def rbf_features(self, points: np.ndarray) -> np.ndarray:
        centers = np.array([[-1.05, 0.2], [1.05, -0.1]])
        sigma = 0.95
        features = []
        for point in points:
            distances = np.sum((centers - point) ** 2, axis=1)
            features.append(np.exp(-distances / (2.0 * sigma**2)))
        return np.array(features)

    def fixed_basis_data(self) -> tuple[np.ndarray, np.ndarray]:
        red = np.array([
            [-1.75, 0.55], [-1.35, 0.85], [-1.05, 0.2], [-0.7, 0.65],
            [0.65, -0.55], [0.95, -0.15], [1.25, -0.45], [1.55, 0.1],
        ])
        blue = np.array([
            [-1.6, -0.85], [-1.05, -0.55], [-0.45, -0.2], [-0.1, 0.25],
            [0.35, 0.55], [0.8, 0.75], [1.25, 0.55], [1.65, 0.85],
        ])
        return red, blue

    def make_dots(self, axes: Axes, points: np.ndarray, color: ManimColor, radius: float = 0.06) -> VGroup:
        return VGroup(*[Dot(axes.c2p(float(x), float(y)), radius=radius, color=color) for x, y in points])

    def fixed_basis_functions(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("4.3.1 Fixed basis functions")
        title = self.scene_title("特徴空間では直線、入力空間では曲線", font_size=34)

        left_axes = self.input_axes().to_edge(LEFT).shift(RIGHT * 0.55 + DOWN * 0.2)
        right_axes = self.feature_axes().to_edge(RIGHT).shift(LEFT * 0.65 + DOWN * 0.2)
        left_title = Text("input x-space", font_size=23, color=TEXT_GREY).next_to(left_axes, UP, buff=0.15)
        right_title = Text("feature phi-space", font_size=23, color=TEXT_GREY).next_to(right_axes, UP, buff=0.15)
        red, blue = self.fixed_basis_data()
        all_points = np.vstack([red, blue])
        features = self.rbf_features(all_points)
        left_red = self.make_dots(left_axes, red, CLASS_RED)
        left_blue = self.make_dots(left_axes, blue, CLASS_BLUE)
        right_red = self.make_dots(right_axes, features[: len(red)], CLASS_RED)
        right_blue = self.make_dots(right_axes, features[len(red) :], CLASS_BLUE)

        centers = [(-1.05, 0.2), (1.05, -0.1)]
        basis_marks = VGroup()
        for i, center in enumerate(centers, start=1):
            cross = Cross(scale_factor=0.16, stroke_color=CLASS_GREEN).move_to(left_axes.c2p(*center))
            contour = Circle(radius=0.62, color=CLASS_GREEN, stroke_width=2).move_to(left_axes.c2p(*center))
            text = MathTex(rf"\phi_{i}", font_size=28, color=CLASS_GREEN).next_to(contour, UP, buff=0.05)
            basis_marks.add(VGroup(cross, contour, text))

        boundary = ParametricFunction(
            lambda t: left_axes.c2p(0.9 * math.cos(t) + 0.35 * math.cos(2.0 * t), 0.95 * math.sin(t) - 0.1),
            t_range=[0, TAU],
            color=WHITE,
            stroke_width=4,
        )
        feature_line = Line(right_axes.c2p(0.12, 0.76), right_axes.c2p(0.82, 0.18), color=WHITE, stroke_width=4)
        transform_arrow = Arrow(left_axes.get_right(), right_axes.get_left(), buff=0.25, color=ACCENT)
        transform_label = MathTex(r"\phi(x)", font_size=36, color=ACCENT).next_to(transform_arrow, UP, buff=0.08)
        note = Text("固定基底は境界表現を楽にするが、重なり自体は消さない", font_size=24, color=MODEL_YELLOW)
        note.to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title), Create(left_axes), Write(left_title), run_time=1.5)
        self.play(FadeIn(left_red), FadeIn(left_blue), FadeIn(basis_marks), run_time=1.8)
        self.play(Create(boundary), run_time=1.5)
        self.play(Create(transform_arrow), Write(transform_label), Create(right_axes), Write(right_title), run_time=1.5)
        self.play(FadeIn(right_red, shift=RIGHT * 0.2), FadeIn(right_blue, shift=RIGHT * 0.2), Create(feature_line), run_time=2.0)
        self.play(Write(note), Indicate(feature_line, color=MODEL_YELLOW), run_time=1.5)
        self.finish_narration(narration)
        self.fade_all()

    def make_sigmoid_axes(self, width: float = 5.6, height: float = 3.2) -> Axes:
        return Axes(
            x_range=[-6, 6, 3],
            y_range=[0, 1, 0.5],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        )

    def make_curve(self, axes: Axes, x_values: np.ndarray, y_values: np.ndarray, color: ManimColor, width: float = 4.0, opacity: float = 1.0) -> VMobject:
        curve = VMobject(color=color)
        curve.set_points_smoothly([axes.c2p(float(x), float(y)) for x, y in zip(x_values, y_values)])
        curve.set_stroke(width=width, opacity=opacity)
        return curve

    def logistic_regression(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("4.3.2 Logistic regression")
        title = self.scene_title("線形スコアを、確率へ押し込む", font_size=34)

        sigmoid_axes = self.make_sigmoid_axes().to_edge(LEFT).shift(RIGHT * 0.55 + DOWN * 0.4)
        x_grid = np.linspace(-6.0, 6.0, 400)
        sigmoid_curve = self.make_curve(sigmoid_axes, x_grid, sigmoid(x_grid), ACCENT, width=4.3)
        half_line = DashedLine(sigmoid_axes.c2p(-6, 0.5), sigmoid_axes.c2p(6, 0.5), color=TEXT_GREY)
        zero_line = DashedLine(sigmoid_axes.c2p(0, 0), sigmoid_axes.c2p(0, 1), color=WHITE)
        sigmoid_label = MathTex(r"\sigma(a)=\frac{1}{1+\exp(-a)}", font_size=33).next_to(sigmoid_axes, UP, buff=0.15)
        boundary_label = Text("a = 0 で p = 0.5", font_size=22, color=WHITE).next_to(zero_line, DOWN, buff=0.18)

        plane = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=5.3,
            y_length=3.7,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        ).to_edge(RIGHT).shift(LEFT * 0.55 + DOWN * 0.2)
        strips = VGroup()
        for i in range(18):
            x0 = -3 + i * (6 / 18)
            x1 = -3 + (i + 1) * (6 / 18)
            probability = float(sigmoid(1.45 * ((x0 + x1) / 2.0) - 0.25))
            color = interpolate_color(CLASS_BLUE, CLASS_RED, probability)
            rect = Rectangle(
                width=abs(plane.c2p(x1, 0)[0] - plane.c2p(x0, 0)[0]),
                height=abs(plane.c2p(0, 2)[1] - plane.c2p(0, -2)[1]),
                fill_color=color,
                fill_opacity=0.18,
                stroke_width=0,
            ).move_to(plane.c2p((x0 + x1) / 2.0, 0.0))
            strips.add(rect)
        red = np.array([[0.7, 0.8], [1.1, -0.3], [1.6, 0.35], [2.0, -0.75], [2.35, 0.75]])
        blue = np.array([[-2.1, -0.65], [-1.6, 0.55], [-1.2, -0.15], [-0.75, 0.7], [-0.35, -0.55]])
        red_dots = self.make_dots(plane, red, CLASS_RED, radius=0.075)
        blue_dots = self.make_dots(plane, blue, CLASS_BLUE, radius=0.075)
        decision = Line(plane.c2p(0.17, -2), plane.c2p(0.17, 2), color=WHITE, stroke_width=4)
        equation = MathTex(r"p(C_1|\phi)=\sigma(w^T\phi)", font_size=38).to_corner(UR).shift(DOWN * 0.55 + LEFT * 0.15)
        class_two = MathTex(r"p(C_2|\phi)=1-p(C_1|\phi)", font_size=31, color=TEXT_GREY).next_to(equation, DOWN, buff=0.18)
        note = Text("確率を返すので、境界付近の迷いも表せる", font_size=24, color=MODEL_YELLOW)
        note.to_edge(DOWN).shift(UP * 0.28)

        self.play(FadeIn(label), Write(title), Create(sigmoid_axes), Write(sigmoid_label), run_time=1.5)
        self.play(Create(sigmoid_curve), Create(half_line), Create(zero_line), Write(boundary_label), run_time=2.0)
        self.play(Create(plane), FadeIn(strips), FadeIn(red_dots), FadeIn(blue_dots), Create(decision), run_time=2.0)
        self.play(Write(equation), Write(class_two), run_time=1.4)
        self.play(Write(note), Indicate(decision, color=MODEL_YELLOW), run_time=1.4)
        self.finish_narration(narration)
        self.fade_all()

    def cross_entropy_gradient(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Likelihood, cross-entropy, and over-fitting")
        title = self.scene_title("予測と目標の差が、勾配になる", font_size=34)

        formula_block = VGroup(
            MathTex(r"p(t|w)=\prod_{n=1}^{N}y_n^{t_n}(1-y_n)^{1-t_n}", font_size=31),
            MathTex(r"E(w)=-\sum_{n=1}^{N}\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\}", font_size=30),
            MathTex(r"\nabla E(w)=\sum_{n=1}^{N}(y_n-t_n)\phi_n", font_size=34, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        formula_block.to_edge(LEFT).shift(RIGHT * 0.5 + UP * 0.45)

        axis = self.make_sigmoid_axes(width=5.7, height=3.0).to_edge(RIGHT).shift(LEFT * 0.45 + DOWN * 0.2)
        x_grid = np.linspace(-5.0, 5.0, 400)
        curves = [
            self.make_curve(axis, x_grid, sigmoid(scale * x_grid), color, width=3.6, opacity=0.9)
            for scale, color in [(0.75, CLASS_BLUE), (1.8, ACCENT), (4.2, CLASS_RED)]
        ]
        curve_labels = VGroup(
            Text("|w| small", font_size=20, color=CLASS_BLUE),
            Text("|w| grows", font_size=20, color=ACCENT),
            Text("step-like", font_size=20, color=CLASS_RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        curve_labels.next_to(axis, UP, buff=0.2).align_to(axis, LEFT)
        separable_note = Text("linearly separable: ML drives |w| upward", font_size=21, color=MODEL_YELLOW)
        separable_note.next_to(axis, DOWN, buff=0.18)

        error_points = VGroup()
        arrows = VGroup()
        examples = [(-3.0, 0, CLASS_BLUE), (-1.0, 0, CLASS_BLUE), (0.9, 1, CLASS_RED), (2.6, 1, CLASS_RED)]
        for x, target, color in examples:
            y = float(sigmoid(1.25 * x))
            dot = Dot(axis.c2p(x, y), color=color, radius=0.075)
            target_y = 0.0 if target == 0 else 1.0
            arrow = Arrow(axis.c2p(x, y), axis.c2p(x, target_y), buff=0.02, color=MODEL_YELLOW, stroke_width=3, max_tip_length_to_length_ratio=0.2)
            error_points.add(dot)
            arrows.add(arrow)
        error_label = MathTex(r"y_n-t_n", font_size=32, color=MODEL_YELLOW).next_to(arrows, RIGHT, buff=0.15)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(LaggedStart(*[Write(item) for item in formula_block], lag_ratio=0.22), run_time=2.4)
        self.play(Create(axis), Create(curves[0]), Write(curve_labels[0]), run_time=1.4)
        self.play(FadeIn(error_points), Create(arrows), Write(error_label), run_time=1.6)
        self.play(ReplacementTransform(curves[0], curves[1]), Write(curve_labels[1]), run_time=1.4)
        self.play(ReplacementTransform(curves[1], curves[2]), Write(curve_labels[2]), Write(separable_note), run_time=1.8)
        self.finish_narration(narration)
        self.fade_all()

    def irls_weighted_least_squares(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("4.3.3 Iterative reweighted least squares")
        title = self.scene_title("IRLS: 境界付近を重く見て解き直す", font_size=33)

        left_axis = Axes(
            x_range=[-3, 3, 1.5],
            y_range=[0, 0.28, 0.14],
            x_length=5.2,
            y_length=2.8,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        ).to_edge(LEFT).shift(RIGHT * 0.6 + UP * 0.15)
        a_grid = np.linspace(-3, 3, 300)
        weights = sigmoid(a_grid) * (1.0 - sigmoid(a_grid))
        weight_curve = self.make_curve(left_axis, a_grid, weights, ACCENT, width=4.2)
        weight_formula = MathTex(r"R_{nn}=y_n(1-y_n)", font_size=35, color=ACCENT).next_to(left_axis, UP, buff=0.18)
        half_note = Text("p = 0.5 付近が最大", font_size=22, color=MODEL_YELLOW).next_to(left_axis, DOWN, buff=0.2)

        plane = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=5.0,
            y_length=3.35,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        ).to_edge(RIGHT).shift(LEFT * 0.55 + UP * 0.1)
        points = np.array([
            [-2.35, -0.6], [-1.8, 0.4], [-1.15, -0.2], [-0.35, 0.2],
            [0.15, -0.3], [0.7, 0.45], [1.35, -0.15], [2.0, 0.65],
        ])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = 1.55 * points[:, 0] - 0.25
        probabilities = sigmoid(scores)
        r_values = probabilities * (1.0 - probabilities)
        weighted_dots = VGroup()
        for point, label_value, r_value in zip(points, labels, r_values):
            color = CLASS_RED if label_value == 1 else CLASS_BLUE
            radius = 0.055 + 0.22 * float(r_value / np.max(r_values))
            weighted_dots.add(Dot(plane.c2p(float(point[0]), float(point[1])), radius=radius, color=color, fill_opacity=0.78))
        boundary_old = Line(plane.c2p(-0.15, -2), plane.c2p(-0.15, 2), color=TEXT_GREY, stroke_width=3)
        boundary_new = Line(plane.c2p(0.15, -2), plane.c2p(0.15, 2), color=WHITE, stroke_width=4)
        plane_note = Text("dot size = IRLS weight", font_size=21, color=TEXT_GREY).next_to(plane, DOWN, buff=0.18)

        update_equations = VGroup(
            MathTex(r"w_{\mathrm{new}}=w_{\mathrm{old}}-H^{-1}\nabla E", font_size=32),
            MathTex(r"H=\Phi^T R\Phi", font_size=34, color=MODEL_YELLOW),
            MathTex(r"w_{\mathrm{new}}=(\Phi^T R\Phi)^{-1}\Phi^T Rz", font_size=30, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        update_equations.to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(Create(left_axis), Write(weight_formula), Create(weight_curve), Write(half_note), run_time=2.0)
        self.play(Create(plane), FadeIn(weighted_dots), Create(boundary_old), Write(plane_note), run_time=1.8)
        self.play(Transform(boundary_old, boundary_new), run_time=1.3)
        self.play(LaggedStart(*[Write(item) for item in update_equations], lag_ratio=0.2), run_time=2.3)
        self.play(Indicate(weighted_dots[3], color=MODEL_YELLOW), Indicate(weighted_dots[4], color=MODEL_YELLOW), run_time=1.4)
        self.finish_narration(narration)
        self.fade_all()

    def make_softmax_bar(self, probabilities: np.ndarray, labels: list[str]) -> VGroup:
        bars = VGroup()
        max_height = 1.65
        colors = [CLASS_RED, CLASS_BLUE, CLASS_GREEN]
        for i, (probability, label, color) in enumerate(zip(probabilities, labels, colors)):
            height = max_height * float(probability)
            bar = Rectangle(width=0.48, height=max(height, 0.03), fill_color=color, fill_opacity=0.82, stroke_width=0)
            bar.move_to(RIGHT * (i * 0.7))
            bar.align_to(ORIGIN, DOWN)
            value = Text(f"{probability:.2f}", font_size=18, color=color).next_to(bar, UP, buff=0.08)
            name = Text(label, font_size=18, color=TEXT_GREY).next_to(bar, DOWN, buff=0.08)
            bars.add(VGroup(bar, value, name))
        bars.arrange(RIGHT, buff=0.24, aligned_edge=DOWN)
        return bars

    def multiclass_softmax(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("4.3.4 Multiclass logistic regression")
        title = self.scene_title("三つ以上の確率は softmax で正規化する", font_size=32)

        plane = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=5.3,
            y_length=3.7,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        ).to_edge(LEFT).shift(RIGHT * 0.45 + DOWN * 0.1)
        class_points = [
            (np.array([[-2.2, 0.8], [-1.7, 0.35], [-1.25, 1.05], [-0.9, 0.55]]), CLASS_RED),
            (np.array([[0.55, -1.0], [1.1, -0.55], [1.6, -1.15], [2.05, -0.45]]), CLASS_BLUE),
            (np.array([[0.2, 1.0], [0.75, 0.55], [1.35, 0.92], [1.9, 0.62]]), CLASS_GREEN),
        ]
        dots = VGroup(*[self.make_dots(plane, points, color, radius=0.075) for points, color in class_points])
        separators = VGroup(
            Line(plane.c2p(-0.65, -2), plane.c2p(-0.15, 2), color=WHITE, stroke_width=3),
            Line(plane.c2p(-0.2, 0.05), plane.c2p(2.55, 1.75), color=WHITE, stroke_width=3),
            Line(plane.c2p(-0.1, 0.05), plane.c2p(2.45, -1.7), color=WHITE, stroke_width=3),
        )
        query = Dot(plane.c2p(0.45, 0.15), color=MODEL_YELLOW, radius=0.1)
        scores = np.array([0.3, 1.15, 0.75])
        probabilities = softmax(scores)
        bars = self.make_softmax_bar(probabilities, ["C1", "C2", "C3"]).to_edge(RIGHT).shift(LEFT * 0.85 + DOWN * 0.4)
        softmax_formula = MathTex(r"y_k=\frac{\exp(a_k)}{\sum_j\exp(a_j)}", font_size=36).to_corner(UR).shift(DOWN * 0.55 + LEFT * 0.15)
        activation = MathTex(r"a_k=w_k^T\phi", font_size=32, color=TEXT_GREY).next_to(softmax_formula, DOWN, buff=0.14)
        target = VGroup(
            Text("1-of-K target", font_size=22, color=TEXT_GREY),
            MathTex(r"t=(0,1,0)", font_size=34, color=MODEL_YELLOW),
            MathTex(r"E=-\sum_{n,k}t_{nk}\ln y_{nk}", font_size=31),
            MathTex(r"\nabla_{w_j}E=\sum_n(y_{nj}-t_{nj})\phi_n", font_size=29, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        target.to_edge(DOWN).shift(UP * 0.15)

        self.play(FadeIn(label), Write(title), Create(plane), FadeIn(dots), run_time=1.8)
        self.play(Create(separators), FadeIn(query), run_time=1.5)
        self.play(Write(softmax_formula), Write(activation), FadeIn(bars), run_time=2.0)
        self.play(LaggedStart(*[Write(item) for item in target], lag_ratio=0.2), run_time=2.3)
        self.play(Indicate(bars[1], color=MODEL_YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.fade_all()

    def probit_regression(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("4.3.5 Probit regression")
        title = self.scene_title("しきい値が揺れると、CDF が活性化関数になる", font_size=31)

        axis = Axes(
            x_range=[-4, 4, 2],
            y_range=[0, 1.05, 0.5],
            x_length=7.0,
            y_length=3.6,
            tips=False,
            axis_config={"color": TEXT_GREY, "stroke_width": 2},
        ).to_edge(LEFT).shift(RIGHT * 0.75 + DOWN * 0.2)
        x_grid = np.linspace(-4, 4, 500)
        pdf_scaled = gaussian_pdf(x_grid) * 1.95
        cdf = probit(x_grid)
        logistic = sigmoid(1.7 * x_grid)
        pdf_curve = self.make_curve(axis, x_grid, pdf_scaled, CLASS_BLUE, width=3.4, opacity=0.85)
        probit_curve = self.make_curve(axis, x_grid, cdf, PROBIT_PURPLE, width=4.0)
        logistic_curve = self.make_curve(axis, x_grid, logistic, ACCENT, width=3.4, opacity=0.85)
        threshold = DashedLine(axis.c2p(0.75, 0), axis.c2p(0.75, 1), color=MODEL_YELLOW)
        shaded = Polygon(axis.c2p(-4, 0), *[axis.c2p(float(x), float(y)) for x, y in zip(x_grid[x_grid <= 0.75], pdf_scaled[x_grid <= 0.75])], axis.c2p(0.75, 0), color=CLASS_BLUE, fill_opacity=0.2, stroke_width=0)
        legend = VGroup(
            VGroup(Line(LEFT * 0.25, RIGHT * 0.25, color=CLASS_BLUE, stroke_width=4), Text("threshold density", font_size=18, color=CLASS_BLUE)).arrange(RIGHT, buff=0.1),
            VGroup(Line(LEFT * 0.25, RIGHT * 0.25, color=PROBIT_PURPLE, stroke_width=4), Text("probit CDF", font_size=18, color=PROBIT_PURPLE)).arrange(RIGHT, buff=0.1),
            VGroup(Line(LEFT * 0.25, RIGHT * 0.25, color=ACCENT, stroke_width=4), Text("logistic", font_size=18, color=ACCENT)).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        legend.next_to(axis, UP, buff=0.15).align_to(axis, LEFT)

        formulas = VGroup(
            MathTex(r"t=1\quad \mathrm{if}\quad a\ge \theta", font_size=31),
            MathTex(r"f(a)=\int_{-\infty}^{a}p(\theta)\,d\theta", font_size=31, color=PROBIT_PURPLE),
            MathTex(r"\Phi(a)=\int_{-\infty}^{a}\mathcal{N}(\theta|0,1)\,d\theta", font_size=27),
            MathTex(r"p(t|x)=\epsilon+(1-2\epsilon)\sigma(x)", font_size=28, color=MODEL_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        formulas.to_edge(RIGHT).shift(LEFT * 0.35 + DOWN * 0.1)
        outlier_axis = Line(LEFT * 1.4, RIGHT * 1.4, color=TEXT_GREY)
        outlier_axis.next_to(formulas, DOWN, buff=0.45)
        boundary = Line(outlier_axis.get_center() + DOWN * 0.35, outlier_axis.get_center() + UP * 0.35, color=WHITE)
        outlier = Dot(outlier_axis.get_right() + LEFT * 0.25 + UP * 0.1, color=CLASS_BLUE, radius=0.09)
        outlier_label = Text("mislabeled outlier", font_size=19, color=MODEL_YELLOW).next_to(outlier_axis, DOWN, buff=0.12)

        self.play(FadeIn(label), Write(title), Create(axis), run_time=1.4)
        self.play(Create(pdf_curve), FadeIn(shaded), Create(threshold), FadeIn(legend[0]), Write(formulas[0]), run_time=1.8)
        self.play(Create(probit_curve), FadeIn(legend[1]), Write(formulas[1]), Write(formulas[2]), run_time=2.1)
        self.play(Create(logistic_curve), FadeIn(legend[2]), run_time=1.4)
        self.play(Create(outlier_axis), Create(boundary), FadeIn(outlier), Write(outlier_label), Write(formulas[3]), run_time=2.0)
        self.finish_narration(narration)
        self.fade_all()

    def canonical_link_summary(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("4.3.6 Canonical link functions")
        title = self.scene_title("対応する link を選ぶと、勾配がそろう", font_size=33)

        headers = ["model", "activation / link", "loss", "gradient"]
        rows = [
            ["linear", r"y=w^T\phi", "squared", r"(y-t)\phi"],
            ["logistic", r"y=\sigma(w^T\phi)", "cross entropy", r"(y-t)\phi"],
            ["softmax", r"y_k=\mathrm{softmax}(a)_k", "multi CE", r"(y_j-t_j)\phi"],
        ]
        table = VGroup()
        cell_widths = [2.0, 3.1, 2.1, 2.55]
        cell_height = 0.62
        for row_index, row in enumerate([headers, *rows]):
            row_group = VGroup()
            for col_index, value in enumerate(row):
                box = Rectangle(
                    width=cell_widths[col_index],
                    height=cell_height,
                    stroke_color=TEXT_GREY,
                    stroke_width=1.5,
                    fill_color=GREY_E if row_index == 0 else BLACK,
                    fill_opacity=0.24 if row_index == 0 else 0.05,
                )
                if row_index == 0 or col_index in (0, 2):
                    content = Text(value, font_size=19 if row_index else 18, color=TEXT_GREY if row_index == 0 else WHITE)
                else:
                    content = MathTex(value, font_size=26 if col_index == 3 else 24, color=ACCENT if col_index == 3 else WHITE)
                row_group.add(VGroup(box, content))
            row_group.arrange(RIGHT, buff=0)
            table.add(row_group)
        table.arrange(DOWN, buff=0)
        table.move_to(UP * 0.72)

        explanation = VGroup(
            MathTex(r"f^{-1}(y)=\psi(y)", font_size=36, color=MODEL_YELLOW),
            Text("exp. family / canonical link", font_size=22, color=TEXT_GREY),
            Text("余分な微分項が打ち消される", font_size=22, color=ACCENT),
        ).arrange(DOWN, buff=0.18)

        summary = VGroup(
            Text("1. 事後確率を直接モデル化", font_size=20, color=WHITE),
            Text("2. 交差エントロピーで確率を合わせる", font_size=20, color=WHITE),
            Text("3. IRLS / 勾配法で最適化", font_size=20, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        lower_panel = VGroup(summary, explanation).arrange(RIGHT, buff=0.65, aligned_edge=UP)
        lower_panel.next_to(table, DOWN, buff=0.32)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(table, shift=UP * 0.15), run_time=2.0)
        self.play(FadeIn(explanation), run_time=1.7)
        self.play(LaggedStart(*[Write(item) for item in summary], lag_ratio=0.2), run_time=2.0)
        self.play(Indicate(table[2][3], color=MODEL_YELLOW), Indicate(table[3][3], color=MODEL_YELLOW), run_time=1.5)
        self.finish_narration(narration)
        self.fade_all()
