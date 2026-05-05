from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
TRUE_GREEN = GREEN_C
MODEL_RED = RED_C
TEST_ORANGE = ORANGE
RESIDUAL_YELLOW = YELLOW_C
REG_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def make_sine_data(
    n: int = 10,
    noise_std: float = 0.25,
    seed: int = 3,
    endpoint: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n, endpoint=endpoint)
    t = np.sin(2.0 * np.pi * x) + rng.normal(0.0, noise_std, size=n)
    return x, t


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(np.asarray(x), N=degree + 1, increasing=True)


def fit_polynomial(
    x: np.ndarray,
    t: np.ndarray,
    degree: int,
    lam: float = 0.0,
    regularize_bias: bool = False,
) -> np.ndarray:
    phi = design_matrix(x, degree)
    if lam == 0:
        return np.linalg.lstsq(phi, t, rcond=None)[0]
    penalty = lam * np.eye(degree + 1)
    if not regularize_bias:
        penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ t)


def eval_poly(w: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    return design_matrix(np.asarray(x), len(w) - 1) @ w


def rms_error(w: np.ndarray, x: np.ndarray, t: np.ndarray) -> float:
    residual = eval_poly(w, x) - t
    return float(np.sqrt(np.mean(residual**2)))


def sine(x: float | np.ndarray) -> float | np.ndarray:
    return np.sin(2.0 * np.pi * x)


class PRML11PolynomialCurveFitting(Scene):
    """PRML 1.1 overview for a high-school math audience.

    Render example:
        uv run manim -pql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.x_train, self.t_train = make_sine_data(n=10, seed=7)
        self.x_test, self.t_test = make_sine_data(n=100, seed=17)

        self.opening_pattern_recognition()
        self.handwritten_digit_vector()
        self.learning_generalization()
        self.task_types()
        self.curve_fitting_problem()
        self.polynomial_model()
        self.least_squares()
        self.compare_degrees()
        self.train_vs_test_error()
        self.coefficient_growth()
        self.more_data()
        self.regularization()
        self.bridge_to_probability()

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

    def make_axes(self, width: float = 7.0, height: float = 4.0) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.5, 1.5, 0.5],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def plot_true_curve(self, axes: Axes, opacity: float = 1.0) -> VMobject:
        curve = axes.plot(lambda u: np.sin(2 * np.pi * u), x_range=[0, 1], color=TRUE_GREEN)
        curve.set_stroke(width=4, opacity=opacity)
        return curve

    def plot_model_curve(
        self,
        axes: Axes,
        degree: int,
        x: np.ndarray | None = None,
        t: np.ndarray | None = None,
        lam: float = 0.0,
        color: ManimColor = MODEL_RED,
    ) -> VMobject:
        if x is None:
            x = self.x_train
        if t is None:
            t = self.t_train
        w = fit_polynomial(x, t, degree, lam=lam)

        def model(u: float) -> float:
            return float(eval_poly(w, np.array([u]))[0])

        curve = axes.plot(model, x_range=[0, 1], color=color, use_smoothing=False)
        curve.set_stroke(width=4)
        return curve

    def make_data_dots(
        self,
        axes: Axes,
        x: np.ndarray,
        t: np.ndarray,
        color: ManimColor = BLUE_DATA,
        radius: float = 0.065,
    ) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(xi), float(ti)), radius=radius, color=color)
                for xi, ti in zip(x, t)
            ]
        )

    def make_residuals(
        self,
        axes: Axes,
        degree: int,
        x: np.ndarray | None = None,
        t: np.ndarray | None = None,
        color: ManimColor = RESIDUAL_YELLOW,
        stroke_width: float = 4,
    ) -> VGroup:
        if x is None:
            x = self.x_train
        if t is None:
            t = self.t_train
        w = fit_polynomial(x, t, degree)
        lines = VGroup()
        for xi, ti in zip(x, t):
            yi = float(eval_poly(w, np.array([xi]))[0])
            line = Line(
                axes.c2p(float(xi), float(ti)),
                axes.c2p(float(xi), yi),
                color=color,
                stroke_width=stroke_width,
            )
            lines.add(line)
        return lines

    def make_degree_slider(
        self,
        values: list[int],
        current_index: int,
        width: float = 3.6,
        color: ManimColor = MODEL_RED,
    ) -> tuple[VGroup, Dot, Text]:
        line = Line(LEFT * width / 2, RIGHT * width / 2, color=GREY_B, stroke_width=4)
        label_size = 17 if len(values) > 6 else 22
        ticks = VGroup()
        labels = VGroup()
        for i, value in enumerate(values):
            proportion = i / (len(values) - 1)
            point = line.point_from_proportion(proportion)
            ticks.add(Line(point + DOWN * 0.08, point + UP * 0.08, color=GREY_B, stroke_width=3))
            labels.add(MathTex(str(value), font_size=label_size).next_to(point, DOWN, buff=0.14))
        knob = Dot(line.point_from_proportion(current_index / (len(values) - 1)), color=color, radius=0.09)
        title = Text("次数 M", font_size=22, color=color).next_to(line, UP, buff=0.16)
        slider = VGroup(line, ticks, labels, title)
        return slider, knob, title

    def opening_pattern_recognition(self) -> None:
        narration = self.start_narration("scene01")
        title = Text(
            "PRML Chapter 1: データの点から“見えない関数”を探す",
            font_size=36,
            color=WHITE,
        )
        subtitle = Text(
            "1.1 Example: Polynomial Curve Fitting",
            font_size=26,
            color=TEXT_GREY,
        ).next_to(title, DOWN, buff=0.25)

        rng = np.random.default_rng(10)
        random_points = VGroup(
            *[
                Dot(
                    np.array(
                        [
                            rng.uniform(-5.5, 5.5),
                            rng.uniform(-2.8, 2.8),
                            0.0,
                        ]
                    ),
                    radius=0.028,
                    color=BLUE_E,
                )
                for _ in range(120)
            ]
        )

        curve_points = VGroup(
            *[
                Dot(
                    np.array(
                        [
                            -5.5 + 11.0 * u,
                            1.3 * np.sin(2 * np.pi * u),
                            0.0,
                        ]
                    ),
                    radius=0.032,
                    color=BLUE_DATA,
                )
                for u in np.linspace(0, 1, 120)
            ]
        )

        key_sentence = Text(
            "データの中の規則性を見つけ、まだ見ていない入力に使う",
            font_size=30,
            color=WHITE,
        ).to_edge(DOWN)

        self.play(FadeIn(random_points, lag_ratio=0.01), run_time=1.5)
        self.play(Transform(random_points, curve_points), run_time=2.0)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.2)
        self.play(Write(key_sentence), run_time=1.5)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(random_points), FadeOut(title), FadeOut(subtitle), FadeOut(key_sentence))

    def handwritten_digit_vector(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("PRML 1章冒頭 / Fig. 1.1")
        heading = self.scene_title("手書き数字は、機械には数値のリスト", font_size=34)

        grid_size = 14
        cell = 0.16
        pattern = np.zeros((grid_size, grid_size))
        for r in range(2, 12):
            pattern[r, 9] = 1
        for c in range(4, 10):
            pattern[2, c] = 1
            pattern[6, c] = 1
            pattern[11, c] = 1
        pattern[3:6, 10] = 1
        pattern[7:11, 10] = 1
        pattern += np.random.default_rng(0).normal(0, 0.08, pattern.shape)
        pattern = np.clip(pattern, 0, 1)

        pixels = VGroup()
        for r in range(grid_size):
            for c in range(grid_size):
                value = pattern[r, c]
                square = Square(side_length=cell)
                square.set_fill(WHITE, opacity=0.12 + 0.78 * value)
                square.set_stroke(GREY_D, width=0.35)
                square.move_to(np.array([c * cell, -r * cell, 0]))
                pixels.add(square)
        pixels.center().shift(LEFT * 3.6)

        calculation = MathTex(r"28\times 28 = 784", font_size=42)
        calculation.next_to(pixels, DOWN, buff=0.45)

        vector_entries = VGroup(
            *[Text(f"{v:.1f}", font_size=18, color=GREY_A) for v in [0.0, 0.1, 0.8, 1.0, 0.7, 0.2]]
        ).arrange(DOWN, buff=0.08)
        left_bracket = Text("[", font_size=78, color=GREY_A)
        right_bracket = Text("]", font_size=78, color=GREY_A)
        vector = VGroup(left_bracket, vector_entries, right_bracket).arrange(RIGHT, buff=0.05)
        vector.shift(RIGHT * 3.6)
        ellipsis = Text("...", font_size=24, color=TEXT_GREY).next_to(vector_entries, DOWN, buff=0.03)

        arrow = Arrow(pixels.get_right() + RIGHT * 0.25, vector.get_left() + LEFT * 0.25, buff=0.1)
        note = Text("この例では: 白黒 28×28 ピクセル", font_size=25, color=BLUE_DATA)
        note.next_to(arrow, UP)
        bw_note = Text("白黒なので 1ピクセル = 明るさ1つ", font_size=23, color=TEXT_GREY)
        bw_note.next_to(arrow, DOWN, buff=0.28)

        self.play(FadeIn(label), Write(heading))
        self.play(FadeIn(pixels, lag_ratio=0.005), run_time=1.2)
        self.play(Write(calculation), run_time=0.8)
        self.play(GrowArrow(arrow), Write(note), FadeIn(vector), FadeIn(ellipsis), run_time=1.4)
        self.play(Write(bw_note), run_time=0.8)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(*[FadeOut(m) for m in [label, heading, pixels, calculation, arrow, note, bw_note, vector, ellipsis]])

    def learning_generalization(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("PRML 1章冒頭: training set / test set / generalization")
        title = self.scene_title("学習とは、練習問題から初見問題に強くなること", font_size=34)

        train_box = RoundedRectangle(width=3.0, height=1.4, corner_radius=0.12, color=BLUE_DATA)
        train_text = Text("training set\n練習問題", font_size=28).move_to(train_box)
        model_box = RoundedRectangle(width=2.6, height=1.4, corner_radius=0.12, color=MODEL_RED)
        model_text = Text("model\ny(x)", font_size=30).move_to(model_box)
        test_box = RoundedRectangle(width=3.0, height=1.4, corner_radius=0.12, color=TEST_ORANGE)
        test_text = Text("test input\n初見問題", font_size=28).move_to(test_box)
        pred_box = RoundedRectangle(width=2.5, height=1.4, corner_radius=0.12, color=GREEN_C)
        pred_text = Text("prediction\n予測", font_size=28).move_to(pred_box)

        flow = VGroup(
            VGroup(train_box, train_text),
            Arrow(RIGHT, RIGHT * 1.7, buff=0),
            VGroup(model_box, model_text),
            Arrow(RIGHT, RIGHT * 1.7, buff=0),
            VGroup(test_box, test_text),
            Arrow(RIGHT, RIGHT * 1.7, buff=0),
            VGroup(pred_box, pred_text),
        ).arrange(RIGHT, buff=0.35)
        flow.scale(0.78).move_to(ORIGIN)

        core = Text("目的: 訓練データの暗記ではなく generalization", font_size=30, color=WHITE)
        core.to_edge(DOWN)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(flow[0]), GrowArrow(flow[1]), FadeIn(flow[2]))
        self.play(GrowArrow(flow[3]), FadeIn(flow[4]), GrowArrow(flow[5]), FadeIn(flow[6]))
        self.play(Write(core))
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(flow), FadeOut(core))

    def task_types(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("PRML 1章冒頭: supervised / classification / regression")
        title = self.scene_title("今回は、連続値を予測する regression", font_size=34)

        panels = VGroup()
        contents = [
            ("classification", "0,1,2,... のどれ？", BLUE_DATA),
            ("regression", "x から t を予測", MODEL_RED),
            ("unsupervised", "似たものを探す", REG_PURPLE),
        ]
        for name, desc, color in contents:
            box = RoundedRectangle(width=3.5, height=2.1, corner_radius=0.12, color=color)
            name_text = Text(name, font_size=28, color=color).move_to(box.get_top() + DOWN * 0.45)
            desc_text = Text(desc, font_size=24).move_to(box.get_center() + DOWN * 0.25)
            panels.add(VGroup(box, name_text, desc_text))
        panels.arrange(RIGHT, buff=0.35).scale(0.95)

        arrow = MathTex(r"x \longrightarrow t", font_size=58, color=MODEL_RED)
        arrow.next_to(panels[1], DOWN, buff=0.4)
        function_box = RoundedRectangle(width=4.7, height=0.72, corner_radius=0.08, color=MODEL_RED)
        function_text = Text("回帰では  x を入れると t を返す関数  y(x)  を作る", font_size=22)
        function_text.move_to(function_box)
        function_group = VGroup(function_box, function_text).to_edge(DOWN, buff=0.45)
        candidate = Text("今回は、その候補として多項式を使う", font_size=25, color=MODEL_RED)
        candidate.next_to(function_group, UP, buff=0.22)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(panels, lag_ratio=0.25))
        self.play(Write(arrow))
        self.play(FadeIn(function_group), Write(candidate), run_time=1.1)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(panels), FadeOut(arrow), FadeOut(function_group), FadeOut(candidate))

    def curve_fitting_problem(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("PRML 1.1 / Fig. 1.2")
        title = self.scene_title("10個の点から、見えない曲線を想像する", font_size=34)
        axes = self.make_axes()
        axes.shift(DOWN * 0.25)
        x_label = MathTex("x", font_size=30).next_to(axes.x_axis.get_end(), RIGHT)
        t_label = MathTex("t", font_size=30).next_to(axes.y_axis.get_end(), UP)
        dots = self.make_data_dots(axes, self.x_train, self.t_train)
        true_curve = self.plot_true_curve(axes)
        true_curve.set_opacity(0.0)

        note = Text("学習アルゴリズムに見えるのは青い点だけ", font_size=28, color=BLUE_DATA)
        note.to_edge(DOWN)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(x_label), FadeIn(t_label))
        self.play(FadeIn(dots, lag_ratio=0.12), run_time=1.4)
        self.play(Write(note))
        self.wait(0.7)
        reveal = Text("今回は比較用に、背後の sin(2πx) を薄く表示", font_size=26, color=TRUE_GREEN)
        reveal.to_edge(DOWN)
        self.play(Transform(note, reveal), true_curve.animate.set_opacity(0.35), run_time=1.2)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(x_label), FadeOut(t_label), FadeOut(dots), FadeOut(true_curve), FadeOut(note))

    def polynomial_model(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("PRML 1.1 / 式 (1.1)")
        title = self.scene_title("次数 M を動かすと、曲線の自由さが変わる", font_size=32)

        axes = self.make_axes(width=6.0, height=3.65).shift(LEFT * 0.55 + DOWN * 0.2)
        dots = self.make_data_dots(axes, self.x_train, self.t_train)
        true_curve = self.plot_true_curve(axes, opacity=0.22)
        degrees = list(range(1, 10))
        curves = [self.plot_model_curve(axes, degree=degree) for degree in degrees]

        slider, knob, _ = self.make_degree_slider(degrees, 0, width=3.05)
        slider.next_to(axes, RIGHT, buff=0.28).shift(UP * 0.75)
        knob.move_to(slider[0].point_from_proportion(0))
        degree_text = MathTex("M=1", font_size=34, color=MODEL_RED)
        degree_text.next_to(slider, DOWN, buff=0.35)
        comment = Text("直線から始める", font_size=24, color=TEXT_GREY)
        comment.next_to(degree_text, DOWN, buff=0.18)

        formula = MathTex(r"y(x,\mathbf{w})=\sum_{j=0}^{M}w_jx^j", font_size=36, color=WHITE)
        formula.to_edge(DOWN, buff=0.45)
        ingredients = Text("材料: 1, x, x², x³, ... をどこまで使うか", font_size=24, color=TEXT_GREY)
        ingredients.next_to(formula, UP, buff=0.18)

        comments = {
            1: "直線から始める",
            2: "少し曲がる",
            3: "波に近づく",
            4: "さらに合わせる",
            5: "細部も追う",
            6: "曲がりが増える",
            7: "揺れも拾う",
            8: "かなり自由",
            9: "自由すぎる",
        }

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(dots), Create(true_curve))
        self.play(Create(curves[0]), FadeIn(slider), FadeIn(knob), Write(degree_text), Write(comment))
        self.play(Write(ingredients), Write(formula), run_time=1.0)
        current_curve = curves[0]
        for i, degree in enumerate(degrees[1:], start=1):
            next_text = MathTex(f"M={degree}", font_size=34, color=MODEL_RED).move_to(degree_text)
            next_comment = Text(comments[degree], font_size=23, color=TEXT_GREY).move_to(comment)
            self.play(
                Transform(current_curve, curves[i]),
                knob.animate.move_to(slider[0].point_from_proportion(i / (len(degrees) - 1))),
                Transform(degree_text, next_text),
                Transform(comment, next_comment),
                run_time=0.65,
            )
            self.wait(0.08)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(true_curve), FadeOut(current_curve), FadeOut(slider), FadeOut(knob), FadeOut(degree_text), FadeOut(comment), FadeOut(ingredients), FadeOut(formula))

    def least_squares(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("PRML 1.1 / Fig. 1.3 / 式 (1.2)")
        title = self.scene_title("最小二乗: ズレを二乗して足す", font_size=34)
        axes = self.make_axes(width=5.5, height=3.35).shift(LEFT * 1.7 + DOWN * 0.15)
        dots = self.make_data_dots(axes, self.x_train, self.t_train)
        curve = self.plot_model_curve(axes, degree=3)
        residuals = self.make_residuals(axes, degree=3)

        signed_values = [0.45, -0.40, 0.22, -0.23]
        signed_rows = VGroup(
            *[
                MathTex(
                    f"{value:+.2f}",
                    font_size=26,
                    color=BLUE_DATA if value > 0 else TEST_ORANGE,
                )
                for value in signed_values
            ]
        ).arrange(DOWN, buff=0.16, aligned_edge=RIGHT)
        signed_title = Text("足す", font_size=22, color=TEXT_GREY)
        signed_sum = MathTex(r"\approx 0.04", font_size=30, color=YELLOW)
        signed_panel = VGroup(signed_title, signed_rows, Line(LEFT * 0.75, RIGHT * 0.75, color=GREY_B), signed_sum).arrange(DOWN, buff=0.16)
        signed_panel.to_edge(RIGHT, buff=2.0).shift(UP * 0.15)

        cancel_note = Text("上と下が打ち消す", font_size=20, color=YELLOW)
        cancel_note.next_to(signed_panel, DOWN, buff=0.28)

        square_rows = VGroup()
        for value in signed_values:
            side = 0.22 + abs(value) * 0.62
            square = Square(side_length=side, color=RESIDUAL_YELLOW)
            square.set_fill(RESIDUAL_YELLOW, opacity=0.25)
            label = MathTex(f"({value:+.2f})^2", font_size=20, color=WHITE)
            row = VGroup(square, label).arrange(RIGHT, buff=0.16)
            square_rows.add(row)
        square_rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        square_title = Text("二乗して足す", font_size=20, color=RESIDUAL_YELLOW)
        square_sum = MathTex(r">0", font_size=30, color=RESIDUAL_YELLOW)
        square_note = Text("大きいズレほど点数が増える", font_size=18, color=RESIDUAL_YELLOW)
        square_panel = VGroup(square_title, square_rows, Line(LEFT * 0.72, RIGHT * 0.72, color=GREY_B), square_sum, square_note)
        square_panel.arrange(DOWN, buff=0.13, aligned_edge=LEFT)
        square_panel.to_edge(RIGHT, buff=1.25).shift(UP * 0.05)

        formula_words = Text("外し具合 = ズレ₁² + ズレ₂² + ... + ズレₙ²", font_size=25)
        formula_words.to_edge(DOWN, buff=0.35)
        formula = MathTex(
            r"E(\mathbf{w})=\frac12\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2",
            font_size=31,
            color=WHITE,
        )
        formula.next_to(formula_words, UP, buff=0.25)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(dots), Create(curve))
        self.wait(1.4)
        self.play(Create(residuals), run_time=2.8)
        self.wait(1.5)
        self.play(FadeIn(signed_panel), Write(cancel_note), run_time=2.3)
        for item in signed_rows:
            self.play(Indicate(item, scale_factor=1.12), run_time=0.65)
        self.wait(2.2)
        self.play(FadeOut(signed_panel), FadeOut(cancel_note), FadeIn(square_panel), run_time=2.2)
        for row in square_rows:
            self.play(Indicate(row, scale_factor=1.08), run_time=0.7)
        self.wait(2.0)
        self.play(Write(formula_words), run_time=1.8)
        self.wait(1.4)
        self.play(Write(formula), run_time=2.0)
        self.wait(3.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(curve), FadeOut(residuals), FadeOut(square_panel), FadeOut(formula_words), FadeOut(formula))
        self.remove(label, title, axes, dots, curve, residuals, signed_panel, cancel_note, square_panel, formula_words, formula)

    def compare_degrees(self) -> None:
        self.clear()
        narration = self.start_narration("scene08")
        label = self.section_label("PRML 1.1 / Fig. 1.4")
        title = self.scene_title("同じ点で M=1 から 9 まで動かす", font_size=32)
        axes = self.make_axes(width=6.65, height=3.75).shift(LEFT * 0.9 + DOWN * 0.1)
        dots = self.make_data_dots(axes, self.x_train, self.t_train)
        true_curve = self.plot_true_curve(axes, opacity=0.3)
        degrees = list(range(1, 10))
        curves = [self.plot_model_curve(axes, degree=degree) for degree in degrees]

        slider, knob, _ = self.make_degree_slider(degrees, 0, width=3.05)
        slider.next_to(axes, RIGHT, buff=0.35).shift(UP * 0.9)
        knob.move_to(slider[0].point_from_proportion(0))
        degree_label = MathTex("M=1", font_size=36, color=MODEL_RED).next_to(slider, DOWN, buff=0.32)
        comment = Text("直線ではまだ硬い", font_size=23, color=TEXT_GREY).next_to(degree_label, DOWN, buff=0.18)
        comments = {
            1: ("直線ではまだ硬い", TEXT_GREY),
            2: ("曲がれるが浅い", TEXT_GREY),
            3: ("全体の流れをつかむ", TRUE_GREEN),
            4: ("点に寄り始める", TRUE_GREEN),
            5: ("細部も拾い始める", TEXT_GREY),
            6: ("揺れが増える", TEXT_GREY),
            7: ("点を追い込み始める", MODEL_RED),
            8: ("かなり細かい", MODEL_RED),
            9: ("点には合うが暴れる", MODEL_RED),
        }
        definition = Text("過学習 = 練習問題に合わせすぎて、初見問題に弱くなること", font_size=28, color=YELLOW)
        definition.to_edge(DOWN)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(dots), Create(true_curve))
        self.play(Create(curves[0]), FadeIn(slider), FadeIn(knob), Write(degree_label), Write(comment))
        current_curve = curves[0]
        for i, degree in enumerate(degrees[1:], start=1):
            text, color = comments[degree]
            next_degree = MathTex(f"M={degree}", font_size=36, color=MODEL_RED).move_to(degree_label)
            next_comment = Text(text, font_size=23, color=color).move_to(comment)
            self.play(
                Transform(current_curve, curves[i]),
                knob.animate.move_to(slider[0].point_from_proportion(i / (len(degrees) - 1))),
                Transform(degree_label, next_degree),
                Transform(comment, next_comment),
                run_time=0.85 if degree < 7 else 1.05,
            )
            self.wait(0.12)
        self.play(Write(definition), run_time=1.2)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(true_curve), FadeOut(current_curve), FadeOut(slider), FadeOut(knob), FadeOut(degree_label), FadeOut(comment), FadeOut(definition))

    def train_vs_test_error(self) -> None:
        self.clear()
        narration = self.start_narration("scene09")
        label = self.section_label("PRML 1.1 / Fig. 1.5 / 式 (1.3)")
        title = self.scene_title("曲線のズレを測り、Mごとの誤差として記録する", font_size=31)
        degrees = np.arange(0, 10)
        selected = list(range(1, 10))
        train_errors = []
        test_errors = []
        for degree in degrees:
            w = fit_polynomial(self.x_train, self.t_train, int(degree))
            train_errors.append(rms_error(w, self.x_train, self.t_train))
            test_errors.append(rms_error(w, self.x_test, self.t_test))

        model_axes = self.make_axes(width=4.2, height=2.85).shift(LEFT * 3.35 + DOWN * 0.12)
        train_dots = self.make_data_dots(model_axes, self.x_train, self.t_train, radius=0.052)
        test_sample_x = self.x_test[::10]
        test_sample_t = self.t_test[::10]
        test_dots = self.make_data_dots(model_axes, test_sample_x, test_sample_t, color=TEST_ORANGE, radius=0.032)
        curves = [self.plot_model_curve(model_axes, degree=int(degree)) for degree in selected]
        residual_sets = [
            self.make_residuals(model_axes, degree=int(degree), color=RESIDUAL_YELLOW, stroke_width=3)
            for degree in selected
        ]

        error_axes = Axes(
            x_range=[1, 9, 1],
            y_range=[0, max(test_errors) * 1.15, 0.5],
            x_length=4.25,
            y_length=3.0,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 2.25 + DOWN * 0.2)
        x_label = MathTex("M", font_size=28).next_to(error_axes.x_axis.get_end(), RIGHT)
        y_label = Text("RMS", font_size=20).next_to(error_axes.y_axis.get_end(), LEFT, buff=0.12)

        legend = VGroup(
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=BLUE_DATA, stroke_width=5), Text("training", font_size=20)).arrange(RIGHT, buff=0.12),
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=TEST_ORANGE, stroke_width=5), Text("test", font_size=20)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        legend.next_to(error_axes, RIGHT, buff=0.22).shift(UP * 0.85)

        model_title = Text("左: いまの曲線とズレ", font_size=22, color=TEXT_GREY).next_to(model_axes, UP, buff=0.15)
        error_title = Text("右: ズレの記録", font_size=22, color=TEXT_GREY).next_to(error_axes, UP, buff=0.3)
        degree_label = MathTex("M=1", font_size=30, color=MODEL_RED).next_to(model_axes, DOWN, buff=0.16)

        count_marks = VGroup(*[Dot(radius=0.035, color=RESIDUAL_YELLOW) for _ in self.x_train])
        count_marks.arrange(RIGHT, buff=0.06)
        count_label = Text("training: ズレを 10 本数える / test: 100 本も同じ計算", font_size=18, color=TEXT_GREY)

        def rms_formula_for(degree: int) -> MathTex:
            w = fit_polynomial(self.x_train, self.t_train, degree)
            residual = eval_poly(w, self.x_train) - self.t_train
            sum_sq = float(np.sum(residual**2))
            return MathTex(
                rf"\sum r_n^2={sum_sq:.2f},\quad E_{{RMS}}=\sqrt{{{sum_sq:.2f}/10}}={train_errors[degree]:.2f}",
                font_size=24,
                color=WHITE,
            )

        calc_formula = rms_formula_for(1)
        calc_panel = VGroup(count_label, count_marks, calc_formula).arrange(DOWN, buff=0.11)
        calc_panel.to_edge(DOWN, buff=0.22)

        train_points = VGroup()
        test_points = VGroup()
        train_segments = VGroup()
        test_segments = VGroup()
        for i, degree in enumerate(selected):
            train_point = Dot(error_axes.c2p(degree, train_errors[degree]), color=BLUE_DATA, radius=0.055)
            test_point = Dot(error_axes.c2p(degree, test_errors[degree]), color=TEST_ORANGE, radius=0.055)
            train_points.add(train_point)
            test_points.add(test_point)
            if i > 0:
                prev_degree = selected[i - 1]
                train_segments.add(Line(error_axes.c2p(prev_degree, train_errors[prev_degree]), error_axes.c2p(degree, train_errors[degree]), color=BLUE_DATA, stroke_width=4))
                test_segments.add(Line(error_axes.c2p(prev_degree, test_errors[prev_degree]), error_axes.c2p(degree, test_errors[degree]), color=TEST_ORANGE, stroke_width=4))

        self.play(FadeIn(label), Write(title), Create(model_axes), Create(error_axes), FadeIn(x_label), FadeIn(y_label))
        self.play(FadeIn(model_title), FadeIn(error_title), FadeIn(train_dots), Create(curves[0]), Create(residual_sets[0]), Write(degree_label))
        self.play(FadeIn(calc_panel[0]), FadeIn(count_marks, lag_ratio=0.08), Write(calc_formula), run_time=1.7)
        self.play(FadeIn(legend), FadeIn(train_points[0]), run_time=0.7)
        self.play(FadeIn(test_dots), FadeIn(test_points[0]), run_time=0.8)
        current_curve = curves[0]
        current_residuals = residual_sets[0]
        for i, degree in enumerate(selected[1:], start=1):
            next_label = MathTex(f"M={degree}", font_size=30, color=MODEL_RED).move_to(degree_label)
            next_formula = rms_formula_for(degree).move_to(calc_formula)
            self.play(
                Transform(current_curve, curves[i]),
                Transform(current_residuals, residual_sets[i]),
                Transform(degree_label, next_label),
                Transform(calc_formula, next_formula),
                Create(train_segments[i - 1]),
                Create(test_segments[i - 1]),
                FadeIn(train_points[i]),
                FadeIn(test_points[i]),
                run_time=0.95,
            )
            self.wait(0.12)
        self.play(Indicate(calc_panel, scale_factor=1.03), run_time=1.2)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(model_axes), FadeOut(error_axes), FadeOut(x_label), FadeOut(y_label), FadeOut(model_title), FadeOut(error_title), FadeOut(train_dots), FadeOut(test_dots), FadeOut(current_curve), FadeOut(current_residuals), FadeOut(degree_label), FadeOut(train_points), FadeOut(test_points), FadeOut(train_segments), FadeOut(test_segments), FadeOut(legend), FadeOut(calc_panel))

    def coefficient_growth(self) -> None:
        narration = self.start_narration("scene10")
        label = self.section_label("PRML 1.1 / Table 1.1")
        title = self.scene_title("曲線の裏側では、係数つまみが大きく動いている", font_size=31)
        w3 = fit_polynomial(self.x_train, self.t_train, 3)
        w9 = fit_polynomial(self.x_train, self.t_train, 9)
        w3_padded = np.zeros(10)
        w3_padded[: len(w3)] = w3

        model_axes = self.make_axes(width=4.7, height=3.15).shift(LEFT * 3.0 + DOWN * 0.18)
        dots = self.make_data_dots(model_axes, self.x_train, self.t_train, radius=0.052)
        curve3 = self.plot_model_curve(model_axes, degree=3)
        curve9 = self.plot_model_curve(model_axes, degree=9)

        bar_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-1, 1, 0.5],
            x_length=4.7,
            y_length=3.15,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 2.5 + DOWN * 0.18)

        max_abs = max(float(np.max(np.abs(w9))), 1e-6)

        def normalized_bars(weights: np.ndarray, color: ManimColor) -> VGroup:
            bars = VGroup()
            for i, weight in enumerate(weights):
                scaled = np.sign(weight) * np.log10(1.0 + abs(float(weight))) / np.log10(1.0 + max_abs)
                start = bar_axes.c2p(i + 0.55, 0)
                end = bar_axes.c2p(i + 0.55, scaled)
                bar = Line(start, end, color=color, stroke_width=10)
                bars.add(bar)
            return bars

        bars3 = normalized_bars(w3_padded, TRUE_GREEN)
        bars9 = normalized_bars(w9, MODEL_RED)
        labels = VGroup(*[MathTex(f"w_{i}", font_size=20) for i in range(10)])
        for i, item in enumerate(labels):
            item.move_to(bar_axes.c2p(i + 0.55, -0.14))
        model_caption = Text("曲線", font_size=23, color=TEXT_GREY).next_to(model_axes, UP, buff=0.15)
        bar_caption = Text("係数つまみ", font_size=23, color=TEXT_GREY).next_to(bar_axes, UP, buff=0.15)
        m3_text = Text("M=3: 少ない係数で全体を見る", font_size=23, color=TRUE_GREEN).to_edge(DOWN, buff=0.5)
        m9_text = Text("M=9: 係数を大きく振って点を追う", font_size=23, color=MODEL_RED).move_to(m3_text)
        note = Text("棒は係数の大きさを log 表示。極端な係数が曲線の暴れ方につながる", font_size=21, color=TEXT_GREY)
        note.to_edge(DOWN)

        self.play(FadeIn(label), Write(title), Create(model_axes), Create(bar_axes), FadeIn(model_caption), FadeIn(bar_caption), FadeIn(dots), FadeIn(labels))
        self.play(Create(curve3), Create(bars3), Write(m3_text), run_time=1.0)
        self.wait(0.5)
        self.play(Transform(curve3, curve9), Transform(bars3, bars9), Transform(m3_text, m9_text), run_time=1.5)
        self.play(Write(note))
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(model_axes), FadeOut(bar_axes), FadeOut(model_caption), FadeOut(bar_caption), FadeOut(dots), FadeOut(curve3), FadeOut(labels), FadeOut(bars3), FadeOut(m3_text), FadeOut(note))

    def more_data(self) -> None:
        narration = self.start_narration("scene11")
        label = self.section_label("PRML 1.1 / Fig. 1.6")
        title = self.scene_title("同じ M=9 でも、点が増えると曲線が支えられる", font_size=32)
        axes = self.make_axes(width=7.0, height=3.75).shift(DOWN * 0.05)
        x15, t15 = make_sine_data(n=15, seed=5)
        x100, t100 = make_sine_data(n=100, seed=6)
        dots15 = self.make_data_dots(axes, x15, t15, radius=0.055)
        dots100 = self.make_data_dots(axes, x100, t100, radius=0.026)
        true_curve = self.plot_true_curve(axes, opacity=0.25)
        model15 = self.plot_model_curve(axes, degree=9, x=x15, t=t15)
        model100 = self.plot_model_curve(axes, degree=9, x=x100, t=t100)
        caption15 = MathTex(r"N=15,\ M=9", font_size=31, color=MODEL_RED).next_to(axes, UP, buff=0.1)
        caption100 = MathTex(r"N=100,\ M=9", font_size=31, color=TRUE_GREEN).move_to(caption15)
        flow = VGroup(
            Text("少ない点", font_size=22, color=MODEL_RED),
            Arrow(LEFT * 0.6, RIGHT * 0.6, buff=0),
            Text("多い点", font_size=22, color=TRUE_GREEN),
        ).arrange(RIGHT, buff=0.18).to_edge(RIGHT, buff=0.9).shift(UP * 1.55)
        note = Text("複雑なモデルを使うには、それを支えるデータも必要", font_size=28, color=WHITE).to_edge(DOWN)

        self.play(FadeIn(label), Write(title), Create(axes), Create(true_curve))
        self.play(FadeIn(dots15, lag_ratio=0.05), Create(model15), Write(caption15), run_time=1.2)
        self.play(FadeIn(flow), run_time=0.7)
        self.play(FadeOut(dots15), FadeIn(dots100, lag_ratio=0.01), Transform(model15, model100), Transform(caption15, caption100), run_time=2.0)
        self.play(Write(note), run_time=0.9)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(true_curve), FadeOut(dots100), FadeOut(model15), FadeOut(caption15), FadeOut(flow), FadeOut(note))

    def regularization(self) -> None:
        narration = self.start_narration("scene12")
        label = self.section_label("PRML 1.1 / Fig. 1.7, 1.8 / 式 (1.4)")
        title = self.scene_title("正則化: 係数の大きさも外し具合の点数に足す", font_size=30)
        axes = self.make_axes(width=6.2, height=3.6).shift(LEFT * 1.0 + DOWN * 0.25)
        dots = self.make_data_dots(axes, self.x_train, self.t_train)
        true_curve = self.plot_true_curve(axes, opacity=0.35)

        lambda_values = [0.0, np.exp(-18), np.exp(-6), np.exp(-18), np.exp(-6), 1.0]
        labels = [r"\lambda=0", r"\ln\lambda=-18", r"\ln\lambda=-6", r"\ln\lambda=-18", r"\ln\lambda=-6", r"\ln\lambda=0"]
        slider_positions = [0, 1, 2, 1, 2, 3]
        curves = [self.plot_model_curve(axes, degree=9, lam=lam) for lam in lambda_values]

        slider_line = Line(LEFT * 1.35, RIGHT * 1.35, color=GREY_B).to_edge(RIGHT, buff=0.85).shift(UP * 0.85)
        slider_title = Text("なめらかさ λ", font_size=22, color=REG_PURPLE).next_to(slider_line, UP, buff=0.18)
        knob = Dot(slider_line.get_start(), color=REG_PURPLE)
        lambda_text = MathTex(labels[0], font_size=24, color=REG_PURPLE).next_to(slider_line, DOWN, buff=0.18)
        formula = MathTex(
            r"\tilde{E}=\frac12\sum_n\{y(x_n)-t_n\}^2+\frac{\lambda}{2}\sum_j w_j^2",
            font_size=27,
        )
        formula.to_edge(DOWN)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(dots), Create(true_curve))
        self.play(Create(curves[0]), Create(slider_line), FadeIn(knob), Write(slider_title), Write(lambda_text))
        self.play(Write(formula))
        current_curve = curves[0]
        for i in range(1, len(curves)):
            target_pos = slider_line.point_from_proportion(slider_positions[i] / 3)
            new_text = MathTex(labels[i], font_size=24, color=REG_PURPLE).next_to(slider_line, DOWN, buff=0.18)
            self.play(
                Transform(current_curve, curves[i]),
                knob.animate.move_to(target_pos),
                Transform(lambda_text, new_text),
                run_time=1.45,
            )
            self.wait(0.35)

        note = Text("λ は、点への合い方と曲線の暴れにくさのバランスを取るつまみ", font_size=22, color=WHITE)
        note.next_to(formula, UP, buff=0.18)
        self.play(Write(note))
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(true_curve), FadeOut(current_curve), FadeOut(slider_line), FadeOut(slider_title), FadeOut(knob), FadeOut(lambda_text), FadeOut(formula), FadeOut(note))

    def bridge_to_probability(self) -> None:
        narration = self.start_narration("scene13")
        title = self.scene_title("次回: 不確かさをどう扱うか", font_size=36)
        axes = self.make_axes(width=7.2, height=4.0).shift(DOWN * 0.25)
        dots = self.make_data_dots(axes, self.x_train, self.t_train)
        model = self.plot_model_curve(axes, degree=3)
        band_upper = axes.plot(lambda u: np.sin(2 * np.pi * u) + 0.35, x_range=[0, 1], color=GREY_C)
        band_lower = axes.plot(lambda u: np.sin(2 * np.pi * u) - 0.35, x_range=[0, 1], color=GREY_C)
        band_upper.set_stroke(opacity=0.45)
        band_lower.set_stroke(opacity=0.45)
        noise_clouds = VGroup(
            *[
                Circle(radius=0.18, color=GREY_B, stroke_opacity=0.4)
                .set_fill(GREY_B, opacity=0.08)
                .move_to(dot.get_center())
                for dot in dots
            ]
        )
        message = Text(
            "一本の曲線だけでなく、「どれくらい不確かか」を表すために確率論へ進む",
            font_size=28,
            color=WHITE,
        ).to_edge(DOWN)
        self.play(Write(title), Create(axes), FadeIn(dots), Create(model))
        self.play(FadeIn(noise_clouds), Create(band_upper), Create(band_lower), run_time=1.2)
        self.play(Write(message), run_time=1.2)
        self.wait(1.5)
        self.finish_narration(narration)
        self.play(FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(model), FadeOut(noise_clouds), FadeOut(band_upper), FadeOut(band_lower), FadeOut(message))
