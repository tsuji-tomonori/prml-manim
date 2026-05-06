from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_CLASS = BLUE_C
ORANGE_CLASS = ORANGE
GREEN_CLASS = GREEN_C
RED_BOUNDARY = RED_C
FISHER_YELLOW = YELLOW
MODEL_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def make_two_class_data(seed: int = 41) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_one = rng.multivariate_normal(
        mean=[1.1, 0.85],
        cov=[[0.13, 0.03], [0.03, 0.15]],
        size=16,
    )
    class_two = rng.multivariate_normal(
        mean=[-1.0, -0.55],
        cov=[[0.16, -0.02], [-0.02, 0.13]],
        size=16,
    )
    return class_one, class_two


def make_multiclass_data(seed: int = 42) -> list[tuple[np.ndarray, ManimColor, str]]:
    rng = np.random.default_rng(seed)
    specs = [
        ([1.15, 0.95], [[0.11, 0.02], [0.02, 0.11]], BLUE_CLASS, "C1"),
        ([-1.25, 0.65], [[0.13, -0.01], [-0.01, 0.12]], ORANGE_CLASS, "C2"),
        ([0.05, -1.25], [[0.13, 0.0], [0.0, 0.12]], GREEN_CLASS, "C3"),
    ]
    return [(rng.multivariate_normal(mean, cov, size=12), color, label) for mean, cov, color, label in specs]


def fit_least_squares_boundary(class_one: np.ndarray, class_two: np.ndarray) -> np.ndarray:
    points = np.vstack([class_one, class_two])
    targets = np.r_[np.ones(len(class_one)), -np.ones(len(class_two))]
    design = np.c_[np.ones(len(points)), points]
    return np.linalg.lstsq(design, targets, rcond=None)[0]


def fisher_direction(class_one: np.ndarray, class_two: np.ndarray) -> np.ndarray:
    mean_one = class_one.mean(axis=0)
    mean_two = class_two.mean(axis=0)
    centered_one = class_one - mean_one
    centered_two = class_two - mean_two
    sw = centered_one.T @ centered_one + centered_two.T @ centered_two
    direction = np.linalg.solve(sw + 0.02 * np.eye(2), mean_two - mean_one)
    return direction / np.linalg.norm(direction)


class PRML41DiscriminantFunctions(Scene):
    """PRML 4.1 discriminant functions overview.

    Render example:
        uv run manim -pql prml_4_1_discriminant_functions.py PRML41DiscriminantFunctions
    """

    x_range = (-3.0, 3.0)
    y_range = (-2.5, 2.5)

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.opening_direct_decision()
        self.two_class_geometry()
        self.multiclass_argmax()
        self.least_squares_classifier()
        self.fisher_projection()
        self.fisher_threshold_and_multiclass()
        self.perceptron_learning()
        self.summary_limitations()

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

    def make_axes(self, width: float = 6.2, height: float = 4.6) -> Axes:
        return Axes(
            x_range=[self.x_range[0], self.x_range[1], 1],
            y_range=[self.y_range[0], self.y_range[1], 1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_points(self, axes: Axes, points: np.ndarray, color: ManimColor, radius: float = 0.055) -> VGroup:
        return VGroup(*[Dot(axes.c2p(float(x), float(y)), color=color, radius=radius) for x, y in points])

    def boundary_line(
        self,
        axes: Axes,
        coef: np.ndarray,
        color: ManimColor = RED_BOUNDARY,
        stroke_width: float = 4.0,
        opacity: float = 1.0,
    ) -> Line:
        b, w1, w2 = [float(v) for v in coef]
        xmin, xmax = self.x_range
        ymin, ymax = self.y_range
        points: list[tuple[float, float]] = []
        if abs(w2) > 1e-8:
            for x in (xmin, xmax):
                y = -(b + w1 * x) / w2
                if ymin - 1e-6 <= y <= ymax + 1e-6:
                    points.append((x, y))
        if abs(w1) > 1e-8:
            for y in (ymin, ymax):
                x = -(b + w2 * y) / w1
                if xmin - 1e-6 <= x <= xmax + 1e-6:
                    points.append((x, y))
        unique: list[tuple[float, float]] = []
        for point in points:
            if not any(np.linalg.norm(np.array(point) - np.array(existing)) < 1e-5 for existing in unique):
                unique.append(point)
        if len(unique) < 2:
            unique = [(xmin, 0.0), (xmax, 0.0)]
        line = Line(
            axes.c2p(unique[0][0], unique[0][1]),
            axes.c2p(unique[1][0], unique[1][1]),
            color=color,
            stroke_width=stroke_width,
        )
        line.set_opacity(opacity)
        return line

    def make_score_box(self, label: str, expression: str, color: ManimColor) -> VGroup:
        tag = Text(label, font_size=26, color=color)
        formula = MathTex(expression, font_size=34, color=WHITE)
        group = VGroup(tag, formula).arrange(DOWN, buff=0.18)
        box = SurroundingRectangle(group, color=color, buff=0.16, corner_radius=0.04)
        return VGroup(box, group)

    def opening_direct_decision(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 4.1 Discriminant Functions")
        title = self.scene_title("識別関数は、入力から直接クラスを選ぶ")

        axes = self.make_axes(width=5.9, height=4.1).shift(LEFT * 2.4 + DOWN * 0.25)
        data = make_multiclass_data()
        clouds = VGroup(*[self.make_points(axes, points, color) for points, color, _ in data])
        boundaries = VGroup(
            self.boundary_line(axes, np.array([0.0, 1.0, -0.78]), RED_BOUNDARY, 3.2, 0.9),
            self.boundary_line(axes, np.array([0.15, -1.0, -0.95]), RED_BOUNDARY, 3.2, 0.9),
            self.boundary_line(axes, np.array([-0.1, 0.2, 1.0]), RED_BOUNDARY, 3.2, 0.9),
        )
        region_labels = VGroup(
            Text("R1", font_size=26, color=BLUE_CLASS).move_to(axes.c2p(1.95, 1.45)),
            Text("R2", font_size=26, color=ORANGE_CLASS).move_to(axes.c2p(-2.15, 1.15)),
            Text("R3", font_size=26, color=GREEN_CLASS).move_to(axes.c2p(-0.2, -1.9)),
        )

        arrow_one = Arrow(LEFT * 0.9, RIGHT * 0.9, color=TEXT_GREY, buff=0.05)
        input_text = Text("入力 x", font_size=27)
        scores = VGroup(
            self.make_score_box("C1", r"y_1(x)", BLUE_CLASS),
            self.make_score_box("C2", r"y_2(x)", ORANGE_CLASS),
            self.make_score_box("C3", r"y_3(x)", GREEN_CLASS),
        ).arrange(DOWN, buff=0.25)
        decision = MathTex(r"k^*=\operatorname*{arg\,max}_k\, y_k(x)", font_size=36)
        class_text = Text("最大スコアのクラス", font_size=27, color=FISHER_YELLOW)
        flow = VGroup(input_text, arrow_one, scores, decision, class_text).arrange(RIGHT, buff=0.42)
        flow.scale(0.72).to_edge(RIGHT).shift(LEFT * 0.05 + DOWN * 0.1)
        decision.next_to(scores, RIGHT, buff=0.42)
        class_text.next_to(decision, DOWN, buff=0.28)

        approaches = VGroup(
            Text("直接: 識別関数", font_size=24, color=FISHER_YELLOW),
            Text("確率: p(C_k | x)", font_size=22, color=TEXT_GREY),
            Text("生成: p(x | C_k)p(C_k)", font_size=22, color=TEXT_GREY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        approaches.to_corner(DR).shift(UP * 0.35 + LEFT * 0.15)

        self.play(FadeIn(label), Write(title), run_time=1.5)
        self.play(Create(axes), FadeIn(clouds, lag_ratio=0.05), run_time=1.7)
        self.play(Create(boundaries), FadeIn(region_labels), run_time=1.8)
        self.play(FadeIn(flow), run_time=1.5)
        self.play(Indicate(scores[0], color=BLUE_CLASS), Indicate(decision, color=FISHER_YELLOW), run_time=1.5)
        self.play(FadeIn(approaches), run_time=1.4)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def two_class_geometry(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("4.1.1 Two classes")
        title = self.scene_title("w は向き、w0 は境界の位置を決める")
        axes = self.make_axes(width=6.6, height=4.7).shift(LEFT * 1.9 + DOWN * 0.2)
        class_one, class_two = make_two_class_data()
        points_one = self.make_points(axes, class_one, BLUE_CLASS)
        points_two = self.make_points(axes, class_two, ORANGE_CLASS)
        coef = np.array([-0.18, 1.0, 0.82])
        boundary = self.boundary_line(axes, coef, RED_BOUNDARY)
        boundary_label = MathTex(r"y(x)=0", font_size=30, color=RED_BOUNDARY).next_to(boundary, UP, buff=0.12)

        b, w1, w2 = coef
        normal = np.array([w1, w2]) / np.linalg.norm([w1, w2])
        base = np.array([0.15, -(b + w1 * 0.15) / w2])
        w_arrow = Arrow(
            axes.c2p(base[0], base[1]),
            axes.c2p(base[0] + 0.75 * normal[0], base[1] + 0.75 * normal[1]),
            color=FISHER_YELLOW,
            buff=0.0,
            stroke_width=6,
        )
        w_label = MathTex(r"w", font_size=34, color=FISHER_YELLOW).next_to(w_arrow, RIGHT, buff=0.1)

        x_point = np.array([1.35, 1.35])
        signed_distance = (b + np.dot(np.array([w1, w2]), x_point)) / np.linalg.norm([w1, w2])
        projection = x_point - signed_distance * normal
        point = Dot(axes.c2p(x_point[0], x_point[1]), color=WHITE, radius=0.07)
        projection_dot = Dot(axes.c2p(projection[0], projection[1]), color=TEXT_GREY, radius=0.05)
        distance_line = DashedLine(
            axes.c2p(x_point[0], x_point[1]),
            axes.c2p(projection[0], projection[1]),
            color=FISHER_YELLOW,
            stroke_width=3,
        )
        distance_label = MathTex(r"r=\frac{y(x)}{\|w\|}", font_size=30, color=FISHER_YELLOW)
        distance_label.next_to(distance_line, RIGHT, buff=0.15)

        shifted_plus = self.boundary_line(axes, np.array([0.62, 1.0, 0.82]), GREY_B, 2.4, 0.55)
        shifted_minus = self.boundary_line(axes, np.array([-0.95, 1.0, 0.82]), GREY_B, 2.4, 0.55)
        bias_label = MathTex(r"-\frac{w_0}{\|w\|}", font_size=30, color=TEXT_GREY).to_corner(DR).shift(UP * 1.0 + LEFT * 0.35)

        formula = VGroup(
            MathTex(r"y(x)=w^Tx+w_0", font_size=38),
            MathTex(r"y(x)\ge 0 \Rightarrow C_1,\quad y(x)<0 \Rightarrow C_2", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        formula.to_edge(RIGHT).shift(LEFT * 0.25 + UP * 0.75)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(points_one), FadeIn(points_two), run_time=2.0)
        self.play(Create(boundary), Write(boundary_label), Write(formula[0]), run_time=1.6)
        self.play(Create(w_arrow), Write(w_label), Write(formula[1]), run_time=1.5)
        self.play(Create(shifted_plus), Create(shifted_minus), Write(bias_label), run_time=1.5)
        self.play(FadeIn(point), FadeIn(projection_dot), Create(distance_line), Write(distance_label), run_time=1.6)
        self.play(Indicate(point, color=FISHER_YELLOW), Indicate(distance_label, color=FISHER_YELLOW), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def multiclass_argmax(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("4.1.2 Multiple classes")
        title = self.scene_title("多クラスは、K 本のスコアを同時に比べる")

        left_box = RoundedRectangle(width=4.9, height=3.9, corner_radius=0.06, color=GREY_B, stroke_width=2)
        left_box.shift(LEFT * 3.25 + DOWN * 0.15)
        left_title = Text("二値識別器を貼り合わせる", font_size=24, color=TEXT_GREY).next_to(left_box, UP, buff=0.12)
        left_lines = VGroup(
            Line(LEFT * 1.6, RIGHT * 1.6, color=RED_BOUNDARY, stroke_width=3).rotate(0.25),
            Line(LEFT * 1.6, RIGHT * 1.6, color=RED_BOUNDARY, stroke_width=3).rotate(2.05),
            Line(LEFT * 1.6, RIGHT * 1.6, color=RED_BOUNDARY, stroke_width=3).rotate(-1.0),
        ).move_to(left_box.get_center())
        question_region = VGroup(
            RegularPolygon(n=6, radius=0.48, color=GREEN_B, fill_color=GREEN_E, fill_opacity=0.35),
            Text("?", font_size=36, color=FISHER_YELLOW),
        ).move_to(left_box.get_center())
        ambiguous = Text("あいまい領域", font_size=23, color=FISHER_YELLOW).next_to(question_region, DOWN, buff=0.25)

        axes = self.make_axes(width=5.35, height=3.9).shift(RIGHT * 2.85 + DOWN * 0.15)
        data = make_multiclass_data(seed=45)
        clouds = VGroup(*[self.make_points(axes, points, color, radius=0.047) for points, color, _ in data])
        boundaries = VGroup(
            self.boundary_line(axes, np.array([0.1, 1.0, -0.85]), RED_BOUNDARY, 3.0),
            self.boundary_line(axes, np.array([0.2, -1.0, -0.9]), RED_BOUNDARY, 3.0),
            self.boundary_line(axes, np.array([-0.15, 0.08, 1.0]), RED_BOUNDARY, 3.0),
        )
        argmax_title = Text("単一の K クラス識別器", font_size=24, color=FISHER_YELLOW).next_to(axes, UP, buff=0.2)
        equations = VGroup(
            MathTex(r"y_k(x)=w_k^Tx+w_{k0}", font_size=32),
            MathTex(r"C_k:\ y_k(x)>y_j(x)\quad(j\ne k)", font_size=30),
            MathTex(r"y_k(x)=y_j(x)\Rightarrow\text{linear boundary}", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        equations.to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(FadeIn(left_box), Write(left_title), Create(left_lines), run_time=1.5)
        self.play(FadeIn(question_region), Write(ambiguous), run_time=1.2)
        self.play(Create(axes), FadeIn(clouds), Write(argmax_title), run_time=1.7)
        self.play(Create(boundaries), Write(equations[0]), run_time=1.5)
        self.play(Write(equations[1]), Write(equations[2]), run_time=1.4)
        self.play(Indicate(boundaries, color=FISHER_YELLOW), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def least_squares_classifier(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("4.1.3 Least squares for classification")
        title = self.scene_title("最小二乗は閉形式で解けるが、分類には弱い")
        axes = self.make_axes(width=6.2, height=4.45).shift(LEFT * 2.1 + DOWN * 0.2)
        class_one, class_two = make_two_class_data(seed=53)
        extra = np.array([[-2.35, -1.75], [-2.2, -1.95], [-2.55, -1.35], [-2.05, -2.05], [-2.7, -1.6]])
        coef_base = fit_least_squares_boundary(class_one, class_two)
        coef_extra = fit_least_squares_boundary(class_one, np.vstack([class_two, extra]))

        points_one = self.make_points(axes, class_one, BLUE_CLASS)
        points_two = self.make_points(axes, class_two, ORANGE_CLASS)
        extra_points = self.make_points(axes, extra, ORANGE_CLASS, radius=0.065)
        base_line = self.boundary_line(axes, coef_base, MODEL_PURPLE, 3.4)
        extra_line = self.boundary_line(axes, coef_extra, RED_BOUNDARY, 4.2)
        base_label = Text("外れ値なし", font_size=21, color=MODEL_PURPLE).next_to(base_line, UP, buff=0.1)
        extra_label = Text("正しく分類されても境界を引っぱる", font_size=21, color=RED_BOUNDARY).to_edge(DOWN).shift(LEFT * 2.1 + UP * 0.15)

        formula = VGroup(
            MathTex(r"E_D(W)=\frac12\operatorname{Tr}\{(\widetilde XW-T)^T(\widetilde XW-T)\}", font_size=28),
            MathTex(r"W=\widetilde X^\dagger T", font_size=34, color=FISHER_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        formula.to_edge(RIGHT).shift(LEFT * 0.2 + UP * 1.05)

        probability_note = VGroup(
            Text("1-of-K なら出力の和は 1", font_size=24),
            Text("でも各値は 0 から 1 に収まらない", font_size=24, color=FISHER_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        probability_note.next_to(formula, DOWN, buff=0.5).align_to(formula, LEFT)
        output_axis = Line(LEFT * 1.35, RIGHT * 1.35, color=GREY_B, stroke_width=4)
        tick_zero = Line(DOWN * 0.08, UP * 0.08, color=GREY_B, stroke_width=3).move_to(output_axis.point_from_proportion(0.2))
        tick_one = Line(DOWN * 0.08, UP * 0.08, color=GREY_B, stroke_width=3).move_to(output_axis.point_from_proportion(0.72))
        low_dot = Dot(output_axis.point_from_proportion(0.05), color=RED_BOUNDARY, radius=0.07)
        high_dot = Dot(output_axis.point_from_proportion(0.92), color=RED_BOUNDARY, radius=0.07)
        range_labels = VGroup(
            Text("0", font_size=18, color=TEXT_GREY).next_to(tick_zero, DOWN, buff=0.08),
            Text("1", font_size=18, color=TEXT_GREY).next_to(tick_one, DOWN, buff=0.08),
            Text("-0.3", font_size=18, color=RED_BOUNDARY).next_to(low_dot, UP, buff=0.08),
            Text("1.2", font_size=18, color=RED_BOUNDARY).next_to(high_dot, UP, buff=0.08),
        )
        score_bar = VGroup(output_axis, tick_zero, tick_one, low_dot, high_dot, range_labels)
        score_bar.next_to(probability_note, DOWN, buff=0.35).align_to(probability_note, LEFT)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(points_one), FadeIn(points_two), run_time=2.0)
        self.play(Create(base_line), Write(base_label), Write(formula), run_time=1.8)
        self.play(FadeIn(extra_points, lag_ratio=0.12), run_time=1.2)
        self.play(ReplacementTransform(base_line.copy(), extra_line), FadeIn(extra_label), run_time=1.6)
        self.play(Write(probability_note), FadeIn(score_bar), run_time=1.7)
        self.play(Indicate(extra_points, color=FISHER_YELLOW), Indicate(extra_line, color=FISHER_YELLOW), run_time=1.4)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def make_fisher_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(61)
        cov = np.array([[0.9, -0.72], [-0.72, 0.82]])
        class_one = rng.multivariate_normal([-1.0, -0.55], cov, size=28)
        class_two = rng.multivariate_normal([1.05, 0.6], cov, size=28)
        return class_one, class_two

    def projection_strip(self, class_one: np.ndarray, class_two: np.ndarray, direction: np.ndarray, color: ManimColor) -> VGroup:
        direction = direction / np.linalg.norm(direction)
        values_one = class_one @ direction
        values_two = class_two @ direction
        all_values = np.r_[values_one, values_two]
        min_value, max_value = float(all_values.min()), float(all_values.max())
        width = 3.7

        def map_x(value: float) -> float:
            return -width / 2 + width * (value - min_value) / max(max_value - min_value, 1e-6)

        line = Line(LEFT * width / 2, RIGHT * width / 2, color=GREY_B, stroke_width=3)
        dots_one = VGroup(*[Dot([map_x(float(v)), 0.14, 0], color=BLUE_CLASS, radius=0.035) for v in values_one])
        dots_two = VGroup(*[Dot([map_x(float(v)), -0.14, 0], color=ORANGE_CLASS, radius=0.035) for v in values_two])
        marker = Triangle(color=color, fill_color=color, fill_opacity=0.9).scale(0.12).rotate(PI).next_to(line, UP, buff=0.08)
        return VGroup(line, dots_one, dots_two, marker)

    def fisher_projection(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("4.1.4 Fisher's linear discriminant")
        title = self.scene_title("平均の差だけでなく、クラス内の広がりも見る")
        axes = self.make_axes(width=5.9, height=4.4).shift(LEFT * 2.55 + DOWN * 0.1)
        class_one, class_two = self.make_fisher_data()
        points_one = self.make_points(axes, class_one, BLUE_CLASS, radius=0.045)
        points_two = self.make_points(axes, class_two, ORANGE_CLASS, radius=0.045)
        mean_one = class_one.mean(axis=0)
        mean_two = class_two.mean(axis=0)
        mean_direction = (mean_two - mean_one) / np.linalg.norm(mean_two - mean_one)
        fisher = fisher_direction(class_one, class_two)

        mean_arrow = Arrow(
            axes.c2p(-1.0 * mean_direction[0], -1.0 * mean_direction[1]),
            axes.c2p(1.25 * mean_direction[0], 1.25 * mean_direction[1]),
            color=TEXT_GREY,
            buff=0.0,
            stroke_width=5,
        )
        fisher_arrow = Arrow(
            axes.c2p(-1.2 * fisher[0], -1.2 * fisher[1]),
            axes.c2p(1.45 * fisher[0], 1.45 * fisher[1]),
            color=FISHER_YELLOW,
            buff=0.0,
            stroke_width=6,
        )
        mean_label = Text("平均差だけ", font_size=22, color=TEXT_GREY).next_to(mean_arrow, UP, buff=0.12)
        fisher_label = Text("Fisher 方向", font_size=23, color=FISHER_YELLOW).next_to(fisher_arrow, RIGHT, buff=0.12)

        strip_mean = self.projection_strip(class_one, class_two, mean_direction, TEXT_GREY)
        strip_fisher = self.projection_strip(class_one, class_two, fisher, FISHER_YELLOW)
        strip_mean.to_edge(RIGHT).shift(LEFT * 0.85 + UP * 0.85)
        strip_fisher.to_edge(RIGHT).shift(LEFT * 0.85 + DOWN * 0.65)
        strip_labels = VGroup(
            Text("射影後: 重なりが残る", font_size=22, color=TEXT_GREY).next_to(strip_mean, UP, buff=0.15),
            Text("射影後: 重なりを小さくする", font_size=22, color=FISHER_YELLOW).next_to(strip_fisher, UP, buff=0.15),
        )
        formula = VGroup(
            MathTex(r"J(w)=\frac{(m_2-m_1)^2}{s_1^2+s_2^2}", font_size=32),
            MathTex(r"=\frac{w^TS_Bw}{w^TS_Ww}", font_size=32, color=FISHER_YELLOW),
        ).arrange(RIGHT, buff=0.22)
        formula.to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(points_one), FadeIn(points_two), run_time=2.0)
        self.play(Create(mean_arrow), Write(mean_label), FadeIn(strip_mean), Write(strip_labels[0]), run_time=1.8)
        self.play(Write(formula[0]), run_time=1.1)
        self.play(ReplacementTransform(mean_arrow.copy(), fisher_arrow), Write(fisher_label), FadeIn(strip_fisher), Write(strip_labels[1]), Write(formula[1]), run_time=2.0)
        self.play(Indicate(strip_fisher, color=FISHER_YELLOW), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def fisher_threshold_and_multiclass(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("4.1.5-4.1.6 Fisher and least squares")
        title = self.scene_title("Fisher は射影方向、判定にはしきい値を置く")
        class_one, class_two = self.make_fisher_data()
        fisher = fisher_direction(class_one, class_two)
        values_one = class_one @ fisher
        values_two = class_two @ fisher
        threshold = 0.5 * (values_one.mean() + values_two.mean())
        strip = self.projection_strip(class_one, class_two, fisher, FISHER_YELLOW).scale(1.35)
        strip.to_edge(LEFT).shift(RIGHT * 2.55 + UP * 0.75)
        thresh_x = strip[0].point_from_proportion(
            float((threshold - min(values_one.min(), values_two.min())) / (max(values_one.max(), values_two.max()) - min(values_one.min(), values_two.min())))
        )
        threshold_line = DashedLine(thresh_x + DOWN * 0.65, thresh_x + UP * 0.65, color=RED_BOUNDARY, stroke_width=4)
        threshold_label = MathTex(r"y_0", font_size=32, color=RED_BOUNDARY).next_to(threshold_line, UP, buff=0.1)
        classify_text = VGroup(
            MathTex(r"y=w^Tx", font_size=34),
            MathTex(r"y\ge y_0\Rightarrow C_2,\quad y<y_0\Rightarrow C_1", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        classify_text.next_to(strip, DOWN, buff=0.35)

        relation = VGroup(
            Text("target coding 最小二乗", font_size=22, color=TEXT_GREY),
            MathTex(r"t_{C_1}=N/N_1,\quad t_{C_2}=-N/N_2", font_size=29),
            Text("Fisher と同じ向きが出る", font_size=22, color=FISHER_YELLOW),
            MathTex(r"w\propto S_W^{-1}(m_2-m_1)", font_size=33, color=FISHER_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        relation.to_edge(RIGHT).shift(LEFT * 0.45 + UP * 1.45)

        multiclass = VGroup(
            Text("K クラス Fisher", font_size=27, color=GREEN_CLASS),
            MathTex(r"y=W^Tx", font_size=32),
            VGroup(
                Text("有効な軸は最大", font_size=23, color=GREEN_CLASS),
                MathTex(r"K-1", font_size=31, color=GREEN_CLASS),
            ).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, buff=0.16)
        box = SurroundingRectangle(multiclass, color=GREEN_CLASS, buff=0.2, corner_radius=0.05)
        multiclass_box = VGroup(box, multiclass).to_edge(DOWN).shift(UP * 0.25 + RIGHT * 2.5)

        self.play(FadeIn(label), Write(title), FadeIn(strip), run_time=1.7)
        self.play(Create(threshold_line), Write(threshold_label), Write(classify_text), run_time=1.6)
        self.play(FadeIn(relation[0]), Write(relation[1]), run_time=1.4)
        self.play(Write(relation[2]), Write(relation[3]), run_time=1.4)
        self.play(FadeIn(multiclass_box), run_time=1.4)
        self.play(Indicate(relation[3], color=FISHER_YELLOW), Indicate(multiclass_box, color=GREEN_CLASS), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def perceptron_boundary(self, axes: Axes, weights: np.ndarray, color: ManimColor = RED_BOUNDARY) -> Line:
        return self.boundary_line(axes, weights, color=color, stroke_width=4.0)

    def perceptron_learning(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("4.1.7 The perceptron algorithm")
        title = self.scene_title("間違えた点だけで、境界を更新する")
        axes = self.make_axes(width=6.3, height=4.55).shift(LEFT * 2.15 + DOWN * 0.15)
        positives = np.array([[1.05, 1.15], [1.55, 0.55], [0.65, 1.55], [1.75, 1.25]])
        negatives = np.array([[-1.2, -0.65], [-1.65, -0.15], [-0.55, -1.45], [-1.75, -1.1]])
        pos_points = self.make_points(axes, positives, BLUE_CLASS, radius=0.06)
        neg_points = self.make_points(axes, negatives, ORANGE_CLASS, radius=0.06)

        weights = np.array([-0.12, -0.45, 0.68])
        line_one = self.perceptron_boundary(axes, weights, TEXT_GREY)
        step_one = positives[0]
        weights_two = weights + np.r_[1.0, step_one]
        line_two = self.perceptron_boundary(axes, weights_two, MODEL_PURPLE)
        step_two = negatives[1]
        weights_three = weights_two - np.r_[1.0, step_two]
        line_three = self.perceptron_boundary(axes, weights_three, RED_BOUNDARY)

        circle_one = Circle(radius=0.18, color=FISHER_YELLOW, stroke_width=4).move_to(axes.c2p(step_one[0], step_one[1]))
        circle_two = Circle(radius=0.18, color=FISHER_YELLOW, stroke_width=4).move_to(axes.c2p(step_two[0], step_two[1]))
        update_formula = VGroup(
            MathTex(r"f(a)=\begin{cases}+1,&a\ge0\\-1,&a<0\end{cases}", font_size=31),
            MathTex(r"a=w^T\phi(x)", font_size=32),
            MathTex(r"\text{misclassified: } w\leftarrow w+\phi(x_n)t_n", font_size=31, color=FISHER_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        update_formula.to_edge(RIGHT).shift(LEFT * 0.25 + UP * 0.65)

        rule_notes = VGroup(
            Text("正解なら変更しない", font_size=23, color=TEXT_GREY),
            Text("線形分離なら有限回で解を見つける", font_size=23, color=GREEN_CLASS),
            Text("分離不能なら収束しない", font_size=23, color=ORANGE_CLASS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        rule_notes.next_to(update_formula, DOWN, buff=0.55).align_to(update_formula, LEFT)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(pos_points), FadeIn(neg_points), run_time=2.0)
        self.play(Create(line_one), Write(update_formula), run_time=1.7)
        self.play(Create(circle_one), Indicate(pos_points[0], color=FISHER_YELLOW), run_time=1.0)
        self.play(ReplacementTransform(line_one, line_two), FadeOut(circle_one), run_time=1.4)
        self.play(Create(circle_two), Indicate(neg_points[1], color=FISHER_YELLOW), run_time=1.0)
        self.play(ReplacementTransform(line_two, line_three), FadeOut(circle_two), run_time=1.4)
        self.play(Write(rule_notes), Indicate(line_three, color=FISHER_YELLOW), run_time=1.7)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def summary_limitations(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("PRML 4.1 Summary")
        title = self.scene_title("識別関数は速い。ただし確率はまだ出てこない")

        columns = VGroup()
        items = [
            ("最小二乗", "閉形式で速い", "外れ値と確率解釈に弱い", MODEL_PURPLE),
            ("Fisher", "重なりの少ない射影", "しきい値は別に決める", FISHER_YELLOW),
            ("Perceptron", "誤分類だけで更新", "分離不能では収束しない", RED_BOUNDARY),
        ]
        for heading, strength, caution, color in items:
            heading_text = Text(heading, font_size=29, color=color)
            strength_text = Text(strength, font_size=23)
            caution_text = Text(caution, font_size=22, color=TEXT_GREY)
            formula_hint = Line(LEFT * 0.9, RIGHT * 0.9, color=color, stroke_width=5)
            col = VGroup(heading_text, formula_hint, strength_text, caution_text).arrange(DOWN, buff=0.22)
            box = SurroundingRectangle(col, color=color, buff=0.22, corner_radius=0.06)
            columns.add(VGroup(box, col))
        columns.arrange(RIGHT, buff=0.5).shift(UP * 0.35)

        bottom = VGroup(
            MathTex(r"\text{decision: } x\mapsto C_k", font_size=36, color=FISHER_YELLOW),
            Text("次は p(x|Ck) と p(Ck|x) を使う確率的な線形分類へ", font_size=26, color=GREEN_CLASS),
        ).arrange(DOWN, buff=0.25)
        bottom.to_edge(DOWN).shift(UP * 0.45)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(FadeIn(columns, lag_ratio=0.12), run_time=1.8)
        self.play(Write(bottom[0]), run_time=1.2)
        self.play(Write(bottom[1]), run_time=1.2)
        self.play(Indicate(bottom, color=FISHER_YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
