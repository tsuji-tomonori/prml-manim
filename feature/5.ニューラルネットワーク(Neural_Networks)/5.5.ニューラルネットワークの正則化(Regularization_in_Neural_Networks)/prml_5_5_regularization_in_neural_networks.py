from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
REG_PURPLE = PURPLE_C
TRUE_GREEN = GREEN_C
VALID_ORANGE = ORANGE
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def make_sine_data(n: int = 12, seed: int = 55) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.05, 0.95, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.22, size=n)
    return x, y


def true_function(x: float | np.ndarray) -> float | np.ndarray:
    return np.sin(2.0 * np.pi * x)


def overfit_function(x: float | np.ndarray, amplitude: float = 0.34) -> float | np.ndarray:
    return np.sin(2.0 * np.pi * x) + amplitude * np.sin(14.0 * np.pi * x)


def gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


class PRML55RegularizationInNeuralNetworks(Scene):
    """PRML 5.5 regularization in neural networks overview.

    Render example:
        uv run manim -pql prml_5_5_regularization_in_neural_networks.py PRML55RegularizationInNeuralNetworks
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.x_train, self.y_train = make_sine_data()

        self.opening_capacity_control()
        self.weight_decay()
        self.gaussian_priors()
        self.early_stopping()
        self.invariances()
        self.tangent_propagation()
        self.transformed_data()
        self.convolutional_networks()
        self.soft_weight_sharing()

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

    def make_regression_axes(self, width: float = 6.8, height: float = 3.9) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.5, 1.5, 0.5],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def plot_function(self, axes: Axes, func, color: ManimColor, width: float = 4.0, opacity: float = 1.0) -> VMobject:
        curve = axes.plot(lambda u: float(func(u)), x_range=[0, 1], color=color, use_smoothing=False)
        curve.set_stroke(width=width, opacity=opacity)
        return curve

    def make_data_dots(self, axes: Axes) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(x), float(y)), radius=0.06, color=BLUE_DATA)
                for x, y in zip(self.x_train, self.y_train)
            ]
        )

    def make_lambda_slider(self, values: list[str], index: int) -> tuple[VGroup, Dot]:
        line = Line(LEFT * 2.25, RIGHT * 2.25, color=GREY_B, stroke_width=4)
        ticks = VGroup()
        labels = VGroup()
        for i, value in enumerate(values):
            point = line.point_from_proportion(i / (len(values) - 1))
            ticks.add(Line(point + DOWN * 0.08, point + UP * 0.08, color=GREY_B, stroke_width=3))
            labels.add(Text(value, font_size=16, color=TEXT_GREY).next_to(point, DOWN, buff=0.12))
        knob = Dot(line.point_from_proportion(index / (len(values) - 1)), color=REG_PURPLE, radius=0.09)
        title = MathTex(r"\lambda", font_size=34, color=REG_PURPLE).next_to(line, UP, buff=0.12)
        return VGroup(line, ticks, labels, title), knob

    def make_weight_bars(self, magnitudes: list[float], color: ManimColor) -> VGroup:
        bars = VGroup()
        for i, magnitude in enumerate(magnitudes):
            height = max(0.08, 0.44 * magnitude)
            bar = Rectangle(width=0.22, height=height, stroke_width=1.5, fill_color=color, fill_opacity=0.75)
            bar.move_to(RIGHT * (i - (len(magnitudes) - 1) / 2.0) * 0.34 + UP * height / 2)
            bars.add(bar)
        baseline = Line(LEFT * 1.35, RIGHT * 1.35, color=GREY_B, stroke_width=2)
        title = Text("重みの大きさ", font_size=20, color=TEXT_GREY).next_to(bars, UP, buff=0.18)
        group = VGroup(baseline, bars, title)
        return group

    def make_layer_network(self, hidden_count: int, scale: float = 0.8) -> VGroup:
        input_points = [LEFT * 2.2 + UP * y for y in (-0.8, 0.0, 0.8)]
        hidden_y = np.linspace(-1.25, 1.25, hidden_count)
        hidden_points = [ORIGIN + UP * float(y) for y in hidden_y]
        output_points = [RIGHT * 2.2 + UP * y for y in (-0.45, 0.45)]
        edges = VGroup()
        for source in input_points:
            for target in hidden_points:
                edges.add(Line(source, target, color=GREY_D, stroke_width=1.2, stroke_opacity=0.55))
        for source in hidden_points:
            for target in output_points:
                edges.add(Line(source, target, color=GREY_D, stroke_width=1.2, stroke_opacity=0.55))
        nodes = VGroup(
            *[Circle(radius=0.11, color=BLUE_DATA, fill_opacity=0.85).move_to(p) for p in input_points],
            *[Circle(radius=0.09, color=REG_PURPLE, fill_opacity=0.85).move_to(p) for p in hidden_points],
            *[Circle(radius=0.11, color=TRUE_GREEN, fill_opacity=0.85).move_to(p) for p in output_points],
        )
        label = Text(f"M = {hidden_count}", font_size=24, color=REG_PURPLE).next_to(nodes, DOWN, buff=0.16)
        return VGroup(edges, nodes, label).scale(scale)

    def make_digit(self, cell_size: float = 0.18, color: ManimColor = BLUE_DATA) -> VGroup:
        pixels = [
            "01110",
            "10001",
            "00110",
            "01000",
            "11111",
        ]
        cells = VGroup()
        for row, line in enumerate(pixels):
            for col, value in enumerate(line):
                square = Square(side_length=cell_size, stroke_width=0.8, stroke_color=GREY_C)
                square.set_fill(color if value == "1" else "#1c1c1c", opacity=0.9 if value == "1" else 0.35)
                square.move_to(RIGHT * (col - 2) * cell_size + DOWN * (row - 2) * cell_size)
                cells.add(square)
        return cells

    def opening_capacity_control(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 5.5 Regularization in Neural Networks")
        title = self.scene_title("大きなネットワークを、実効的に小さく使う", font_size=34)
        axes = self.make_regression_axes(width=5.7, height=3.2).to_edge(LEFT).shift(DOWN * 0.45 + RIGHT * 0.35)
        dots = self.make_data_dots(axes)
        under_curve = self.plot_function(axes, lambda x: 0.25 * math.sin(2 * math.pi * x), VALID_ORANGE, width=3.5)
        over_curve = self.plot_function(axes, lambda x: overfit_function(x, 0.34), MODEL_RED, width=3.5)
        smooth_curve = self.plot_function(axes, true_function, TRUE_GREEN, width=3.8)
        network_small = self.make_layer_network(2, 0.88).to_edge(RIGHT).shift(UP * 0.95 + LEFT * 0.15)
        network_large = self.make_layer_network(10, 0.72).to_edge(RIGHT).shift(DOWN * 1.25 + LEFT * 0.15)
        under_label = Text("underfit", font_size=20, color=VALID_ORANGE).next_to(under_curve, UP, buff=0.2)
        over_label = Text("overfit", font_size=20, color=MODEL_RED).next_to(axes, DOWN, buff=0.15).shift(LEFT * 1.1)
        regularized = Text("regularized", font_size=20, color=TRUE_GREEN).next_to(over_label, RIGHT, buff=0.45)
        note = Text("M だけでなく、解の選び方も複雑さを決める", font_size=25, color=WHITE).to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(Create(axes), FadeIn(dots), FadeIn(network_small), run_time=1.8)
        self.play(Create(under_curve), FadeIn(under_label), run_time=1.5)
        self.play(ReplacementTransform(network_small.copy(), network_large), Create(over_curve), FadeIn(over_label), run_time=1.9)
        self.play(Transform(over_curve, smooth_curve), FadeIn(regularized), Write(note), run_time=2.0)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, dots, under_curve, under_label, network_small, network_large, over_label, regularized, note)), run_time=0.8)
        self.clear()

    def weight_decay(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Weight decay")
        title = self.scene_title("lambda を上げると、重みと曲線の暴れが小さくなる", font_size=32)
        formula = MathTex(r"\tilde{E}(w)=E(w)+\frac{\lambda}{2}w^T w", font_size=38)
        formula.to_corner(UR).shift(DOWN * 0.42 + LEFT * 0.2)
        axes = self.make_regression_axes(width=6.4, height=3.55).shift(LEFT * 2.45 + DOWN * 0.2)
        dots = self.make_data_dots(axes)
        true_curve = self.plot_function(axes, true_function, TRUE_GREEN, width=3.0, opacity=0.55)
        low_curve = self.plot_function(axes, lambda x: overfit_function(x, 0.42), MODEL_RED, width=4.0)
        mid_curve = self.plot_function(axes, lambda x: overfit_function(x, 0.20), REG_PURPLE, width=4.0)
        high_curve = self.plot_function(axes, lambda x: overfit_function(x, 0.06), TRUE_GREEN, width=4.0)
        slider, knob = self.make_lambda_slider(["小", "中", "大"], 0)
        slider_group = VGroup(slider, knob).next_to(axes, DOWN, buff=0.24)
        bars_low = self.make_weight_bars([1.7, 1.2, 1.55, 0.9, 1.45, 1.1, 1.35], MODEL_RED).shift(RIGHT * 3.75 + DOWN * 0.9)
        bars_mid = self.make_weight_bars([1.0, 0.75, 0.95, 0.62, 0.85, 0.65, 0.8], REG_PURPLE).move_to(bars_low)
        bars_high = self.make_weight_bars([0.55, 0.42, 0.5, 0.35, 0.46, 0.32, 0.4], TRUE_GREEN).move_to(bars_low)
        note = Text("fit だけを追うほど、重みは大きくなりやすい", font_size=22, color=TEXT_GREY).next_to(bars_low, DOWN, buff=0.22)

        self.play(FadeIn(label), Write(title), Write(formula), run_time=1.6)
        self.play(Create(axes), FadeIn(dots), Create(true_curve), FadeIn(slider_group), FadeIn(bars_low), Write(note), run_time=2.0)
        self.play(Create(low_curve), run_time=1.4)
        self.play(
            Transform(low_curve, mid_curve),
            knob.animate.move_to(slider[0].point_from_proportion(0.5)),
            Transform(bars_low, bars_mid),
            run_time=2.0,
        )
        self.play(
            Transform(low_curve, high_curve),
            knob.animate.move_to(slider[0].point_from_proportion(1.0)),
            Transform(bars_low, bars_high),
            run_time=2.0,
        )
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, formula, axes, dots, true_curve, low_curve, slider_group, bars_low, note)), run_time=0.8)
        self.clear()

    def gaussian_priors(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Consistent Gaussian priors")
        title = self.scene_title("重み群ごとに、抑える強さを分ける", font_size=34)
        network = self.make_layer_network(6, 1.0).shift(LEFT * 3.25 + DOWN * 0.1)
        groups = VGroup(
            VGroup(Text("1層目 bias", font_size=21), MathTex(r"\alpha_1^b", font_size=32, color=VALID_ORANGE)).arrange(DOWN, buff=0.08),
            VGroup(Text("1層目 weight", font_size=21), MathTex(r"\alpha_1^w", font_size=32, color=REG_PURPLE)).arrange(DOWN, buff=0.08),
            VGroup(Text("2層目 bias", font_size=21), MathTex(r"\alpha_2^b", font_size=32, color=TRUE_GREEN)).arrange(DOWN, buff=0.08),
            VGroup(Text("2層目 weight", font_size=21), MathTex(r"\alpha_2^w", font_size=32, color=MODEL_RED)).arrange(DOWN, buff=0.08),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.45)).to_edge(RIGHT).shift(LEFT * 0.3 + UP * 0.25)
        prior_formula = MathTex(r"p(w_i)\propto \exp\{-\alpha_i w_i^2/2\}", font_size=36).to_edge(DOWN).shift(UP * 0.45)
        arrows = VGroup(
            Arrow(groups[0].get_left(), network[1].get_center() + LEFT * 0.05, color=VALID_ORANGE, buff=0.15),
            Arrow(groups[1].get_left(), network[0].get_center() + LEFT * 0.2, color=REG_PURPLE, buff=0.15),
            Arrow(groups[2].get_left(), network[1].get_right(), color=TRUE_GREEN, buff=0.15),
            Arrow(groups[3].get_left(), network[0].get_right(), color=MODEL_RED, buff=0.15),
        )
        note = Text("入力・出力のスケールに合わせて prior を設計する", font_size=24, color=WHITE)
        note.next_to(prior_formula, UP, buff=0.25)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(FadeIn(network), run_time=1.4)
        self.play(LaggedStart(*[FadeIn(group) for group in groups], lag_ratio=0.15), run_time=1.8)
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.15), Write(prior_formula), Write(note), run_time=2.2)
        self.play(Indicate(groups[1], color=REG_PURPLE), Indicate(groups[3], color=MODEL_RED), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, network, groups, arrows, prior_formula, note)), run_time=0.8)
        self.clear()

    def early_stopping(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Early stopping")
        title = self.scene_title("検証誤差が上がり始める前に止める", font_size=34)
        axes = Axes(
            x_range=[0, 40, 10],
            y_range=[0, 1.2, 0.3],
            x_length=5.9,
            y_length=3.35,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).to_edge(LEFT).shift(RIGHT * 0.55 + DOWN * 0.2)
        train = axes.plot(lambda t: 0.95 * math.exp(-0.055 * t) + 0.08, x_range=[0, 40], color=BLUE_DATA)
        valid = axes.plot(lambda t: 0.62 * math.exp(-0.075 * t) + 0.12 + 0.00095 * (t - 17) ** 2, x_range=[0, 40], color=VALID_ORANGE)
        labels = VGroup(
            Text("training", font_size=19, color=BLUE_DATA).move_to(axes.c2p(32, 0.22)),
            Text("validation", font_size=19, color=VALID_ORANGE).move_to(axes.c2p(31, 0.62)),
            Text("iteration", font_size=19, color=TEXT_GREY).next_to(axes, DOWN, buff=0.16),
            Text("error", font_size=19, color=TEXT_GREY).next_to(axes, LEFT, buff=0.12).rotate(PI / 2),
        )
        stop_x = 18.0
        stop_dot = Dot(axes.c2p(stop_x, 0.62 * math.exp(-0.075 * stop_x) + 0.12 + 0.00095 * (stop_x - 17) ** 2), color=YELLOW, radius=0.08)
        stop_line = DashedLine(axes.c2p(stop_x, 0), axes.c2p(stop_x, 0.52), color=YELLOW, stroke_width=3)
        stop_label = Text("stop", font_size=24, color=YELLOW).next_to(stop_line, UP, buff=0.12)

        plane = NumberPlane(
            x_range=[-0.1, 2.8, 1],
            y_range=[-0.1, 2.2, 1],
            x_length=4.0,
            y_length=3.0,
            background_line_style={"stroke_color": GREY_E, "stroke_width": 1, "stroke_opacity": 0.35},
            axis_config={"stroke_color": GREY_B, "stroke_width": 2},
        ).to_edge(RIGHT).shift(LEFT * 0.45 + DOWN * 0.1)
        ellipses = VGroup(
            Ellipse(width=3.0, height=1.15, color=GREY_C, stroke_width=2).rotate(0.42).move_to(plane.c2p(1.55, 1.05)),
            Ellipse(width=2.2, height=0.82, color=GREY_C, stroke_width=2).rotate(0.42).move_to(plane.c2p(1.55, 1.05)),
            Ellipse(width=1.35, height=0.5, color=GREY_C, stroke_width=2).rotate(0.42).move_to(plane.c2p(1.55, 1.05)),
        )
        path = VMobject(color=YELLOW).set_points_smoothly(
            [plane.c2p(0.1, 0.15), plane.c2p(0.45, 0.85), plane.c2p(0.9, 1.15), plane.c2p(1.55, 1.05)]
        )
        early_point = Dot(plane.c2p(0.9, 1.15), color=YELLOW, radius=0.07)
        ml_point = Dot(plane.c2p(1.55, 1.05), color=MODEL_RED, radius=0.07)
        w_labels = VGroup(
            Text("早く止めた解", font_size=18, color=YELLOW).next_to(early_point, UP, buff=0.12),
            Text("訓練誤差の最小", font_size=18, color=MODEL_RED).next_to(ml_point, DOWN, buff=0.12),
        )

        self.play(FadeIn(label), Write(title), run_time=1.3)
        self.play(Create(axes), Create(train), Create(valid), FadeIn(labels), run_time=2.0)
        self.play(Create(stop_line), FadeIn(stop_dot), Write(stop_label), run_time=1.4)
        self.play(Create(plane), Create(ellipses), Create(path), FadeIn(early_point), FadeIn(ml_point), FadeIn(w_labels), run_time=2.4)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, train, valid, labels, stop_dot, stop_line, stop_label, plane, ellipses, path, early_point, ml_point, w_labels)), run_time=0.8)
        self.clear()

    def invariances(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("Invariances")
        title = self.scene_title("同じ数字なら、少し動いても同じ答え", font_size=34)
        digit = self.make_digit(cell_size=0.22).shift(LEFT * 4.4 + UP * 0.55)
        shifted_digit = self.make_digit(cell_size=0.22, color=TRUE_GREEN).move_to(digit).shift(RIGHT * 1.0 + DOWN * 0.2)
        arrows = VGroup(
            Arrow(digit.get_right(), shifted_digit.get_left(), color=YELLOW, buff=0.15),
            Text("translation", font_size=20, color=YELLOW).next_to(shifted_digit, DOWN, buff=0.18),
        )
        classifier_box = RoundedRectangle(width=2.2, height=0.9, corner_radius=0.08, color=GREY_B, fill_color="#202020", fill_opacity=0.8)
        classifier_text = Text("network", font_size=25).move_to(classifier_box)
        classifier = VGroup(classifier_box, classifier_text).to_edge(RIGHT).shift(UP * 0.55 + LEFT * 0.55)
        output = Text("class = 2", font_size=28, color=TRUE_GREEN).next_to(classifier, DOWN, buff=0.3)
        map_arrows = VGroup(
            Arrow(shifted_digit.get_right(), classifier.get_left(), color=GREY_C, buff=0.2),
            Arrow(classifier.get_bottom(), output.get_top(), color=TRUE_GREEN, buff=0.12),
        )
        options = VGroup(
            Text("1. 変換データを増やす", font_size=24),
            Text("2. 出力変化を抑える正則化", font_size=24),
            Text("3. 不変な特徴を作る", font_size=24),
            Text("4. 構造に不変性を入れる", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title), FadeIn(digit), run_time=1.5)
        self.play(TransformFromCopy(digit, shifted_digit), GrowArrow(arrows[0]), Write(arrows[1]), run_time=1.8)
        self.play(FadeIn(classifier), LaggedStart(*[GrowArrow(a) for a in map_arrows], lag_ratio=0.25), Write(output), run_time=1.7)
        self.play(LaggedStart(*[FadeIn(option, shift=UP * 0.12) for option in options], lag_ratio=0.15), run_time=2.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, digit, shifted_digit, arrows, classifier, output, map_arrows, options)), run_time=0.8)
        self.clear()

    def tangent_propagation(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Tangent propagation")
        title = self.scene_title("変換方向へ出力が変わらないようにする", font_size=34)
        plane = NumberPlane(
            x_range=[-0.2, 4, 1],
            y_range=[-0.2, 3, 1],
            x_length=5.9,
            y_length=4.0,
            background_line_style={"stroke_color": GREY_E, "stroke_width": 1, "stroke_opacity": 0.35},
            axis_config={"stroke_color": GREY_B, "stroke_width": 2},
        ).to_edge(LEFT).shift(RIGHT * 0.45 + DOWN * 0.3)
        manifold = VMobject(color=BLUE_DATA).set_points_smoothly(
            [plane.c2p(0.7, 1.15), plane.c2p(1.35, 1.55), plane.c2p(2.05, 1.7), plane.c2p(2.9, 1.35), plane.c2p(3.45, 0.8)]
        )
        x_point = Dot(plane.c2p(1.95, 1.68), color=YELLOW, radius=0.08)
        tangent = Arrow(plane.c2p(1.65, 1.63), plane.c2p(2.45, 1.58), color=YELLOW, buff=0.0)
        labels = VGroup(
            MathTex(r"x_n", font_size=32, color=YELLOW).next_to(x_point, UP, buff=0.12),
            MathTex(r"\tau_n=\left.\frac{\partial s(x_n,\xi)}{\partial \xi}\right|_{\xi=0}", font_size=31, color=YELLOW).next_to(tangent, DOWN, buff=0.25),
            Text("入力変換が作る局所的な曲線", font_size=22, color=BLUE_DATA).next_to(manifold, UP, buff=0.2),
        )
        network = self.make_layer_network(5, 0.74).to_edge(RIGHT).shift(UP * 0.75 + LEFT * 0.2)
        formula = VGroup(
            MathTex(r"\frac{\partial y_k}{\partial \xi}=\sum_i J_{ki}\tau_i", font_size=34),
            MathTex(r"\Omega=\frac{1}{2}\sum_{n,k}\left(\sum_i J_{nki}\tau_{ni}\right)^2", font_size=31, color=REG_PURPLE),
        ).arrange(DOWN, buff=0.28).to_edge(RIGHT).shift(DOWN * 1.35 + LEFT * 0.15)
        arrow = Arrow(plane.get_right(), network.get_left(), color=GREY_C, buff=0.2)
        invariant = Text("変換しても y が動かない", font_size=24, color=REG_PURPLE).next_to(network, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), Create(plane), run_time=1.5)
        self.play(Create(manifold), FadeIn(x_point), GrowArrow(tangent), FadeIn(labels), run_time=2.1)
        self.play(GrowArrow(arrow), FadeIn(network), Write(invariant), run_time=1.6)
        self.play(Write(formula), run_time=2.0)
        self.play(Indicate(formula[1], color=REG_PURPLE), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, plane, manifold, x_point, tangent, labels, network, formula, arrow, invariant)), run_time=0.8)
        self.clear()

    def transformed_data(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Training with transformed data")
        title = self.scene_title("データ拡張は、微分を抑える正則化になる", font_size=33)
        base = self.make_digit(cell_size=0.18).shift(LEFT * 4.9 + UP * 0.75)
        copies = VGroup()
        shifts = [RIGHT * 0.85, RIGHT * 1.7 + UP * 0.18, RIGHT * 2.55 + DOWN * 0.18, RIGHT * 3.4]
        colors = [BLUE_DATA, TRUE_GREEN, VALID_ORANGE, REG_PURPLE]
        for shift, color in zip(shifts, colors):
            copies.add(self.make_digit(cell_size=0.18, color=color).move_to(base).shift(shift))
        braces = VGroup(
            Brace(VGroup(base, copies), DOWN, color=TEXT_GREY),
            Text("small transformed copies", font_size=21, color=TEXT_GREY),
        )
        braces[1].next_to(braces[0], DOWN, buff=0.12)
        formula = VGroup(
            MathTex(r"\tilde{E}=E+\lambda\Omega", font_size=40),
            MathTex(r"\Omega \simeq \frac{1}{2}\int (\tau^T\nabla y(x))^2p(x)\,dx", font_size=31, color=REG_PURPLE),
            MathTex(r"x\rightarrow x+\epsilon \quad \Rightarrow \quad \mathrm{Tikhonov}", font_size=30, color=TRUE_GREEN),
        ).arrange(DOWN, buff=0.28).to_edge(RIGHT).shift(LEFT * 0.2 + UP * 0.1)
        arrow = Arrow(copies.get_right(), formula.get_left(), color=YELLOW, buff=0.25)
        caption = Text("平均ゼロ・小さい変換でテイラー展開", font_size=24, color=YELLOW).next_to(arrow, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), FadeIn(base), run_time=1.4)
        self.play(LaggedStart(*[TransformFromCopy(base, copy) for copy in copies], lag_ratio=0.15), FadeIn(braces), run_time=2.1)
        self.play(GrowArrow(arrow), Write(caption), run_time=1.2)
        self.play(Write(formula), run_time=2.4)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, base, copies, braces, formula, arrow, caption)), run_time=0.8)
        self.clear()

    def convolutional_networks(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("Convolutional networks")
        title = self.scene_title("局所受容野、重み共有、サブサンプリング", font_size=34)
        image = self.make_digit(cell_size=0.22).to_edge(LEFT).shift(RIGHT * 0.75 + UP * 0.45)
        patch = Rectangle(width=0.66, height=0.66, color=YELLOW, stroke_width=4).move_to(image[6:9].get_center())
        kernel = VGroup(
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
            Square(side_length=0.18, color=YELLOW, fill_color="#303030", fill_opacity=0.8),
        ).arrange_in_grid(rows=3, cols=3, buff=0.0).next_to(image, RIGHT, buff=0.9)
        kernel_label = Text("shared weights", font_size=17, color=YELLOW).next_to(kernel, UP, buff=0.12)
        feature_map = VGroup()
        for row in range(4):
            for col in range(4):
                square = Square(side_length=0.18, stroke_width=1.0, stroke_color=GREY_C)
                square.set_fill(REG_PURPLE if (row + col) % 2 == 0 else BLUE_D, opacity=0.75)
                square.move_to(RIGHT * col * 0.18 + DOWN * row * 0.18)
                feature_map.add(square)
        feature_map.next_to(kernel, RIGHT, buff=1.05)
        feature_label = Text("feature map", font_size=17, color=REG_PURPLE).next_to(feature_map, DOWN, buff=0.14)
        pool = VGroup()
        for row in range(2):
            for col in range(2):
                square = Square(side_length=0.28, stroke_width=1.0, stroke_color=GREY_C)
                square.set_fill(TRUE_GREEN, opacity=0.75 if (row + col) % 2 == 0 else 0.45)
                square.move_to(RIGHT * col * 0.28 + DOWN * row * 0.28)
                pool.add(square)
        pool.next_to(feature_map, RIGHT, buff=1.05)
        pool_label = Text("subsample", font_size=17, color=TRUE_GREEN).next_to(pool, UP, buff=0.12)
        arrows = VGroup(
            Arrow(image.get_right(), kernel.get_left(), color=GREY_C, buff=0.2),
            Arrow(kernel.get_right(), feature_map.get_left(), color=GREY_C, buff=0.2),
            Arrow(feature_map.get_right(), pool.get_left(), color=GREY_C, buff=0.2),
        )
        bullets = VGroup(
            Text("local receptive fields: 近い画素だけを見る", font_size=22),
            Text("weight sharing: 同じ検出器を各位置で使う", font_size=22),
            Text("subsampling: 小さな位置ずれに鈍感にする", font_size=22),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title), FadeIn(image), run_time=1.4)
        self.play(Create(patch), FadeIn(kernel), Write(kernel_label), GrowArrow(arrows[0]), run_time=1.7)
        self.play(patch.animate.shift(RIGHT * 0.44), run_time=0.7)
        self.play(patch.animate.shift(DOWN * 0.44), run_time=0.7)
        self.play(FadeIn(feature_map), Write(feature_label), GrowArrow(arrows[1]), run_time=1.3)
        self.play(FadeIn(pool), Write(pool_label), GrowArrow(arrows[2]), run_time=1.3)
        self.play(LaggedStart(*[FadeIn(bullet, shift=UP * 0.1) for bullet in bullets], lag_ratio=0.15), run_time=1.8)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, image, patch, kernel, kernel_label, feature_map, feature_label, pool, pool_label, arrows, bullets)), run_time=0.8)
        self.clear()

    def soft_weight_sharing(self) -> None:
        narration = self.start_narration("scene09")
        label = self.section_label("Soft weight sharing")
        title = self.scene_title("重みを、いくつかの中心へやわらかく集める", font_size=33)
        number_line = NumberLine(
            x_range=[-3, 3, 1],
            length=7.0,
            include_numbers=True,
            color=GREY_B,
            font_size=20,
        ).shift(LEFT * 1.95 + DOWN * 0.25)
        weights = [-2.55, -1.8, -1.15, -0.35, 0.2, 0.65, 1.2, 1.85, 2.55]
        centers = [-1.25, 1.35]
        dots = VGroup(*[Dot(number_line.n2p(w), radius=0.07, color=BLUE_DATA) for w in weights])
        centers_dots = VGroup(*[Dot(number_line.n2p(c), radius=0.1, color=YELLOW) for c in centers])
        centers_labels = VGroup(
            MathTex(r"\mu_1", font_size=28, color=YELLOW).next_to(centers_dots[0], UP, buff=0.12),
            MathTex(r"\mu_2", font_size=28, color=YELLOW).next_to(centers_dots[1], UP, buff=0.12),
        )
        pulls = VGroup()
        pulled_targets = []
        for w in weights:
            nearest = centers[0] if abs(w - centers[0]) < abs(w - centers[1]) else centers[1]
            target = 0.55 * w + 0.45 * nearest
            pulled_targets.append(target)
            pulls.add(Arrow(number_line.n2p(w), number_line.n2p(target), color=REG_PURPLE, buff=0.08, max_tip_length_to_length_ratio=0.22))
        pulled_dots = VGroup(*[Dot(number_line.n2p(t), radius=0.07, color=REG_PURPLE) for t in pulled_targets])
        formula = VGroup(
            MathTex(r"p(w_i)=\sum_j \pi_j\,\mathcal{N}(w_i|\mu_j,\sigma_j^2)", font_size=34),
            MathTex(r"\tilde{E}=E+\lambda\Omega(w)", font_size=34, color=REG_PURPLE),
        ).arrange(DOWN, buff=0.28).to_edge(RIGHT).shift(LEFT * 0.3 + UP * 0.25)
        summary = VGroup(
            Text("explicit: weight decay", font_size=21, color=REG_PURPLE),
            Text("implicit: early stopping", font_size=21, color=VALID_ORANGE),
            Text("invariance: tangent / CNN", font_size=21, color=TRUE_GREEN),
            Text("grouping: soft sharing", font_size=21, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title), Create(number_line), FadeIn(dots), run_time=1.6)
        self.play(FadeIn(centers_dots), FadeIn(centers_labels), Write(formula[0]), run_time=1.6)
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in pulls], lag_ratio=0.06), Transform(dots, pulled_dots), Write(formula[1]), run_time=2.2)
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.1) for item in summary], lag_ratio=0.13), run_time=1.8)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, number_line, dots, centers_dots, centers_labels, pulls, formula, summary)), run_time=0.8)
        self.clear()
