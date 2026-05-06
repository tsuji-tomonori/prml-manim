from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
BASIS_GREEN = GREEN_C
GAUSS_ORANGE = ORANGE
SIGMOID_TEAL = TEAL_C
REG_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
PANEL_BG = "#1B1B1B"
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def target_function(x: np.ndarray | float) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    return np.sin(2.0 * np.pi * x_array) + 0.35 * np.cos(5.0 * np.pi * x_array)


def make_data(n: int = 12, seed: int = 31) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.04, 0.96, n)
    t = target_function(x) + rng.normal(0.0, 0.22, size=n)
    return x, t


def gaussian_basis(x: np.ndarray | float, centers: np.ndarray, scale: float = 0.16) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    return np.exp(-0.5 * ((x_array[:, None] - centers[None, :]) / scale) ** 2)


def sigmoid_basis(x: np.ndarray | float, centers: np.ndarray, scale: float = 0.08) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    return 1.0 / (1.0 + np.exp(-(x_array[:, None] - centers[None, :]) / scale))


def design_matrix(x: np.ndarray, centers: np.ndarray, scale: float = 0.16) -> np.ndarray:
    return np.column_stack([np.ones_like(x), gaussian_basis(x, centers, scale=scale)])


def fit_ridge(x: np.ndarray, t: np.ndarray, centers: np.ndarray, lam: float = 0.0) -> np.ndarray:
    phi = design_matrix(x, centers)
    penalty = lam * np.eye(phi.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ t)


def predict_gaussian(w: np.ndarray, x: np.ndarray | float, centers: np.ndarray) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    return design_matrix(x_array, centers) @ w


class PRML31LinearBasisFunctionModels(Scene):
    """PRML 3.1 linear basis function models overview.

    Render example:
        uv run manim -pql prml_3_1_linear_basis_function_models.py PRML31LinearBasisFunctionModels
    """

    def construct(self) -> None:
        self.camera.background_color = "#111111"
        self.x_train, self.t_train = make_data()
        self.centers = np.linspace(0.08, 0.92, 8)

        self.introduce_linear_basis_model()
        self.combine_basis_functions()
        self.compare_basis_choices()
        self.gaussian_noise_to_least_squares()
        self.design_matrix_and_normal_equation()
        self.sequential_learning()
        self.regularized_least_squares()
        self.q_regularization_summary()

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
            self.wait(0.6)
            return
        elapsed = float(getattr(self, "time", 0.0)) - start_time
        remaining = duration - elapsed + pad
        if remaining > 0:
            self.wait(remaining)

    def section_label(self) -> Text:
        label = Text("3.1 Linear Basis Function Models", font_size=18, color=TEXT_GREY)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 34) -> Text:
        title = Text(text, font_size=font_size)
        title.to_edge(UP).shift(DOWN * 0.28)
        return title

    def make_axes(self, width: float = 6.6, height: float = 3.8, y_range: list[float] | None = None) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=y_range or [-1.6, 1.6, 0.8],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def data_dots(self, axes: Axes) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(x), float(t)), color=BLUE_DATA, radius=0.055)
                for x, t in zip(self.x_train, self.t_train)
            ]
        )

    def curve_from_values(self, axes: Axes, values_fn, color: ManimColor, width: int = 4) -> VMobject:
        curve = axes.plot(lambda u: float(np.clip(values_fn(u), -1.7, 1.7)), x_range=[0, 1], color=color, use_smoothing=False)
        curve.set_stroke(width=width)
        return curve

    def introduce_linear_basis_model(self) -> None:
        self.clear()
        narration = self.start_narration("scene01")
        label = self.section_label()
        title = self.scene_title("入力を変換してから線形に足す")

        left_formula = MathTex(r"y(x,\mathbf{w})=w_0+w_1x_1+\cdots+w_Dx_D", font_size=36)
        left_formula.shift(UP * 1.35)
        transform_arrow = Arrow(LEFT * 2.8 + UP * 0.05, LEFT * 0.4 + UP * 0.05, buff=0.1, color=GAUSS_ORANGE)
        feature_box = VGroup(
            RoundedRectangle(width=2.4, height=1.35, corner_radius=0.08, color=GAUSS_ORANGE, stroke_width=3),
            MathTex(r"\boldsymbol{\phi}(x)", font_size=42, color=GAUSS_ORANGE),
            Text("特徴量", font_size=22, color=TEXT_GREY),
        )
        feature_box[1].move_to(feature_box[0].get_center() + UP * 0.18)
        feature_box[2].next_to(feature_box[1], DOWN, buff=0.18)
        weighted_sum = MathTex(r"y(x,\mathbf{w})=\mathbf{w}^{T}\boldsymbol{\phi}(x)", font_size=42, color=WHITE)
        weighted_sum.shift(DOWN * 1.0)

        nonlinear = Text("入力 x には非線形", font_size=25, color=GAUSS_ORANGE).shift(LEFT * 2.7 + DOWN * 2.15)
        linear = Text("重み w には線形", font_size=25, color=BASIS_GREEN).shift(RIGHT * 2.65 + DOWN * 2.15)
        brace = Brace(weighted_sum, DOWN, color=BASIS_GREEN)

        self.play(FadeIn(label), Write(title))
        self.play(Write(left_formula))
        self.play(GrowArrow(transform_arrow), FadeIn(feature_box))
        self.play(TransformMatchingTex(left_formula.copy(), weighted_sum))
        self.play(FadeIn(nonlinear), FadeIn(linear), FadeIn(brace))
        self.finish_narration(narration)

    def combine_basis_functions(self) -> None:
        self.clear()
        narration = self.start_narration("scene02")
        label = self.section_label()
        title = self.scene_title("基底関数の山を重み付きで足す")

        axes = self.make_axes(width=7.4, height=3.9, y_range=[-1.5, 2.0, 0.5]).shift(LEFT * 1.1 + DOWN * 0.25)
        axis_labels = VGroup(
            MathTex("x", font_size=26).next_to(axes.x_axis, RIGHT, buff=0.12),
            MathTex(r"\phi_j(x), y", font_size=26).next_to(axes.y_axis, UP, buff=0.12),
        )
        weights = np.array([0.25, 0.9, 0.2, -0.55, -0.95, -0.15, 0.7, 0.35])

        basis_curves = VGroup()
        for i, center in enumerate(self.centers):
            curve = self.curve_from_values(
                axes,
                lambda u, c=center: 0.9 * float(gaussian_basis(u, np.array([c]))[0, 0]),
                color=interpolate_color(GAUSS_ORANGE, GREY_B, i / 10),
                width=2,
            )
            curve.set_opacity(0.65)
            basis_curves.add(curve)

        def combined(u: float) -> float:
            return float(np.r_[1.0, gaussian_basis(u, self.centers)[0]] @ np.r_[0.02, weights])

        combined_curve = self.curve_from_values(axes, combined, MODEL_RED, width=5)
        bars = self.weight_bars(weights).shift(RIGHT * 4.6 + DOWN * 0.1)
        bars_title = MathTex(r"w_j", font_size=34, color=BASIS_GREEN).next_to(bars, UP, buff=0.2)
        formula = MathTex(r"y(x)=\sum_j w_j\phi_j(x)", font_size=38, color=WHITE)
        formula.to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(axis_labels))
        self.play(LaggedStart(*[Create(curve) for curve in basis_curves], lag_ratio=0.08), run_time=1.6)
        self.play(FadeIn(bars), Write(bars_title))
        self.play(Create(combined_curve), Write(formula), run_time=1.2)
        self.finish_narration(narration)

    def weight_bars(self, weights: np.ndarray, scale: float = 0.75) -> VGroup:
        group = VGroup()
        base = Line(LEFT * 1.1, RIGHT * 1.1, color=GREY_C, stroke_width=2)
        group.add(base)
        for i, weight in enumerate(weights):
            height = abs(float(weight)) * scale
            bar = Rectangle(width=0.16, height=max(height, 0.04), stroke_width=0)
            bar.set_fill(BASIS_GREEN if weight >= 0 else MODEL_RED, opacity=0.9)
            bar.move_to(np.array([(i - (len(weights) - 1) / 2) * 0.26, np.sign(weight) * height / 2, 0.0]))
            group.add(bar)
        return group

    def compare_basis_choices(self) -> None:
        self.clear()
        narration = self.start_narration("scene03")
        label = self.section_label()
        title = self.scene_title("基底関数の代表例")

        panels = VGroup(
            self.basis_panel("polynomial", BLUE_DATA, self.polynomial_curves),
            self.basis_panel("Gaussian", GAUSS_ORANGE, self.gaussian_curves),
            self.basis_panel("sigmoid", SIGMOID_TEAL, self.sigmoid_curves),
        ).arrange(RIGHT, buff=0.42).shift(DOWN * 0.05)
        notes = VGroup(
            Text("全体に広がる", font_size=21, color=BLUE_DATA).next_to(panels[0], DOWN, buff=0.25),
            Text("場所ごとに反応", font_size=21, color=GAUSS_ORANGE).next_to(panels[1], DOWN, buff=0.25),
            Text("なだらかに切替", font_size=21, color=SIGMOID_TEAL).next_to(panels[2], DOWN, buff=0.25),
        )
        message = Text("以降の議論は φ(x) の具体形に強く依存しない", font_size=26, color=WHITE)
        message.to_edge(DOWN).shift(UP * 0.22)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(panels, lag_ratio=0.18), run_time=1.4)
        self.play(FadeIn(notes, lag_ratio=0.2))
        self.play(Write(message))
        self.finish_narration(narration)

    def basis_panel(self, title_text: str, color: ManimColor, curve_builder) -> VGroup:
        rect = RoundedRectangle(width=3.55, height=3.15, corner_radius=0.08, color=GREY_C, stroke_width=2)
        rect.set_fill(PANEL_BG, opacity=0.9)
        title = Text(title_text, font_size=23, color=color).next_to(rect, UP, buff=0.18)
        axes = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1.15, 1.15, 0.5],
            x_length=2.95,
            y_length=2.25,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 1.4},
        ).move_to(rect)
        curves = curve_builder(axes, color)
        return VGroup(rect, title, axes, curves)

    def polynomial_curves(self, axes: Axes, color: ManimColor) -> VGroup:
        return VGroup(
            axes.plot(lambda u: u, x_range=[-1, 1], color=color, use_smoothing=False),
            axes.plot(lambda u: u**2, x_range=[-1, 1], color=interpolate_color(color, WHITE, 0.35), use_smoothing=False),
            axes.plot(lambda u: u**3, x_range=[-1, 1], color=interpolate_color(color, WHITE, 0.65), use_smoothing=False),
        ).set_stroke(width=3)

    def gaussian_curves(self, axes: Axes, color: ManimColor) -> VGroup:
        centers = [-0.55, 0.0, 0.55]
        return VGroup(
            *[
                axes.plot(lambda u, c=c: np.exp(-0.5 * ((u - c) / 0.26) ** 2), x_range=[-1, 1], color=color, use_smoothing=False)
                for c in centers
            ]
        ).set_stroke(width=3)

    def sigmoid_curves(self, axes: Axes, color: ManimColor) -> VGroup:
        centers = [-0.45, 0.0, 0.45]
        return VGroup(
            *[
                axes.plot(lambda u, c=c: 1.0 / (1.0 + np.exp(-(u - c) / 0.16)), x_range=[-1, 1], color=color, use_smoothing=False)
                for c in centers
            ]
        ).set_stroke(width=3)

    def gaussian_noise_to_least_squares(self) -> None:
        self.clear()
        narration = self.start_narration("scene04")
        label = self.section_label()
        title = self.scene_title("ガウスノイズの最尤推定は最小二乗")

        axes = self.make_axes(width=6.3, height=3.65).shift(LEFT * 2.35 + DOWN * 0.35)
        dots = self.data_dots(axes)
        w = fit_ridge(self.x_train, self.t_train, self.centers, lam=0.25)
        curve = self.curve_from_values(axes, lambda u: float(predict_gaussian(w, u, self.centers)[0]), MODEL_RED, width=4)
        residuals = VGroup()
        squares = VGroup()
        for x, t in zip(self.x_train[::2], self.t_train[::2]):
            y = float(predict_gaussian(w, x, self.centers)[0])
            residuals.add(Line(axes.c2p(x, y), axes.c2p(x, t), color=GAUSS_ORANGE, stroke_width=3))
            side = min(abs(t - y) * 0.23, 0.28)
            square = Square(side_length=max(side, 0.05), stroke_color=GAUSS_ORANGE, fill_color=GAUSS_ORANGE, fill_opacity=0.25, stroke_width=2)
            square.move_to(axes.c2p(x, (t + y) / 2))
            squares.add(square)

        formulas = VGroup(
            MathTex(r"t=y(x,\mathbf{w})+\epsilon", font_size=34),
            MathTex(r"p(t|x,\mathbf{w},\beta)=\mathcal{N}(t|y(x,\mathbf{w}),\beta^{-1})", font_size=31),
            MathTex(r"E_D(\mathbf{w})={1\over2}\sum_n\{t_n-\mathbf{w}^T\boldsymbol{\phi}(x_n)\}^2", font_size=30, color=GAUSS_ORANGE),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).shift(RIGHT * 3.1 + DOWN * 0.15)
        equivalence = Text("log likelihood 最大化  =  二乗和最小化", font_size=25, color=WHITE)
        equivalence.to_edge(DOWN).shift(UP * 0.22)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(dots), Create(curve))
        self.play(FadeIn(residuals, lag_ratio=0.08), FadeIn(squares, lag_ratio=0.08))
        self.play(Write(formulas[0]), Write(formulas[1]))
        self.play(Write(formulas[2]), Write(equivalence))
        self.finish_narration(narration)

    def design_matrix_and_normal_equation(self) -> None:
        self.clear()
        narration = self.start_narration("scene05")
        label = self.section_label()
        title = self.scene_title("デザイン行列にまとめる")

        matrix = MathTex(
            r"\Phi=",
            r"\begin{pmatrix}"
            r"\phi_0(x_1)&\phi_1(x_1)&\cdots&\phi_{M-1}(x_1)\\"
            r"\phi_0(x_2)&\phi_1(x_2)&\cdots&\phi_{M-1}(x_2)\\"
            r"\vdots&\vdots&\ddots&\vdots\\"
            r"\phi_0(x_N)&\phi_1(x_N)&\cdots&\phi_{M-1}(x_N)"
            r"\end{pmatrix}",
            font_size=31,
        ).shift(UP * 0.95)
        row_note = Text("行: データ点", font_size=23, color=GAUSS_ORANGE).next_to(matrix, LEFT, buff=0.35)
        col_note = Text("列: 基底関数", font_size=23, color=BASIS_GREEN).next_to(matrix, RIGHT, buff=0.35)
        vector_form = MathTex(r"\mathbf{y}=\Phi\mathbf{w}", font_size=40, color=WHITE).shift(DOWN * 1.0 + LEFT * 2.15)
        normal_eq = MathTex(r"\mathbf{w}_{ML}=(\Phi^T\Phi)^{-1}\Phi^T\mathbf{t}", font_size=38, color=WHITE)
        normal_eq.shift(DOWN * 1.0 + RIGHT * 2.25)
        arrow = Arrow(vector_form.get_right(), normal_eq.get_left(), buff=0.25, color=TEXT_GREY)
        note = Text("重みに対して線形なので閉形式解を持つ", font_size=26, color=WHITE).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Write(matrix))
        self.play(FadeIn(row_note), FadeIn(col_note))
        self.play(Write(vector_form), GrowArrow(arrow), Write(normal_eq))
        self.play(Write(note))
        self.finish_narration(narration)

    def sequential_learning(self) -> None:
        self.clear()
        narration = self.start_narration("scene06")
        label = self.section_label()
        title = self.scene_title("一点ずつ重みを更新する")

        axes = self.make_axes(width=6.7, height=3.8).shift(LEFT * 2.25 + DOWN * 0.2)
        formula = MathTex(
            r"\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}+\eta(t_n-\mathbf{w}^{(\tau)T}\boldsymbol{\phi}_n)\boldsymbol{\phi}_n",
            font_size=30,
            color=WHITE,
        ).shift(RIGHT * 2.2 + UP * 1.45)
        status = VGroup(
            Text("stream", font_size=24, color=TEXT_GREY),
            Text("error > 0", font_size=24, color=GAUSS_ORANGE),
            Text("update w", font_size=24, color=BASIS_GREEN),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT).shift(RIGHT * 3.1 + DOWN * 0.2)

        w = np.zeros(len(self.centers) + 1)
        curve = self.curve_from_values(axes, lambda u: float(predict_gaussian(w, u, self.centers)[0]), MODEL_RED, width=4)
        shown_points = VGroup()
        eta = 0.45

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), Write(formula), FadeIn(status[0]))
        self.play(Create(curve))
        for i in [0, 2, 4, 6, 8, 10]:
            x = self.x_train[i]
            t = self.t_train[i]
            phi = np.r_[1.0, gaussian_basis(x, self.centers)[0]]
            prediction = float(w @ phi)
            w = w + eta * (t - prediction) * phi
            point = Dot(axes.c2p(float(x), float(t)), color=BLUE_DATA, radius=0.065)
            residual = Line(axes.c2p(float(x), prediction), axes.c2p(float(x), float(t)), color=GAUSS_ORANGE, stroke_width=3)
            new_curve = self.curve_from_values(axes, lambda u, w=w.copy(): float(predict_gaussian(w, u, self.centers)[0]), MODEL_RED, width=4)
            shown_points.add(point)
            self.play(FadeIn(point), Create(residual), FadeIn(status[1]), run_time=0.45)
            self.play(Transform(curve, new_curve), FadeOut(residual), FadeIn(status[2]), run_time=0.55)
        self.finish_narration(narration)

    def regularized_least_squares(self) -> None:
        self.clear()
        narration = self.start_narration("scene07")
        label = self.section_label()
        title = self.scene_title("重みの大きさを抑える")

        axes = self.make_axes(width=6.5, height=3.7).shift(LEFT * 2.1 + DOWN * 0.3)
        dots = self.data_dots(axes)
        w_free = fit_ridge(self.x_train, self.t_train, self.centers, lam=0.0)
        w_reg = fit_ridge(self.x_train, self.t_train, self.centers, lam=1.6)
        curve_free = self.curve_from_values(axes, lambda u: float(predict_gaussian(w_free, u, self.centers)[0]), MODEL_RED, width=4)
        curve_reg = self.curve_from_values(axes, lambda u: float(predict_gaussian(w_reg, u, self.centers)[0]), BASIS_GREEN, width=4)

        formula = MathTex(
            r"E(\mathbf{w})={1\over2}\sum_n(t_n-\mathbf{w}^T\boldsymbol{\phi}_n)^2+{\lambda\over2}\mathbf{w}^T\mathbf{w}",
            font_size=29,
            color=WHITE,
        ).shift(RIGHT * 2.45 + UP * 1.45)
        lambda_label = MathTex(r"\lambda: 0 \rightarrow 1.6", font_size=34, color=REG_PURPLE).shift(RIGHT * 3.05 + UP * 0.35)
        bars_free = self.weight_bars(w_free[1:], scale=0.24).shift(RIGHT * 3.1 + DOWN * 1.0)
        bars_reg = self.weight_bars(w_reg[1:], scale=0.45).shift(RIGHT * 3.1 + DOWN * 1.0)
        bars_title = Text("重みが縮む", font_size=24, color=BASIS_GREEN).next_to(bars_free, DOWN, buff=0.28)
        note = Text("fit だけでなく係数の大きさも見る", font_size=25, color=WHITE).to_edge(DOWN).shift(UP * 0.22)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(dots), Create(curve_free))
        self.play(Write(formula), FadeIn(bars_free), Write(lambda_label))
        self.play(Transform(curve_free, curve_reg), Transform(bars_free, bars_reg), run_time=1.3)
        self.play(Write(bars_title), Write(note))
        self.finish_narration(narration)

    def q_regularization_summary(self) -> None:
        self.clear()
        narration = self.start_narration("scene08")
        label = self.section_label()
        title = self.scene_title("正則化の形とまとめ")

        left = self.constraint_panel("q=2", "全体を縮める", Circle(radius=1.05, color=REG_PURPLE, stroke_width=4))
        right_shape = Polygon(UP * 1.08, RIGHT * 1.08, DOWN * 1.08, LEFT * 1.08, color=GAUSS_ORANGE, stroke_width=4)
        right = self.constraint_panel("q=1", "ゼロになりやすい", right_shape)
        panels = VGroup(left, right).arrange(RIGHT, buff=1.1).shift(UP * 0.15)

        tangent = Line(LEFT * 0.95 + UP * 0.7, RIGHT * 0.95 + DOWN * 0.25, color=BLUE_DATA, stroke_width=3).move_to(right[2].get_center() + UP * 0.18)
        sparse_dot = Dot(right[2].get_center() + LEFT * 1.08, color=MODEL_RED, radius=0.08)
        summary = VGroup(
            self.summary_item("1. phi(x) で特徴量を作る", GAUSS_ORANGE),
            self.summary_item("2. w には線形なので解きやすい", BASIS_GREEN),
            self.summary_item("3. 正則化で有効な複雑さを抑える", REG_PURPLE),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(panels, lag_ratio=0.2))
        self.play(Create(tangent), FadeIn(sparse_dot))
        self.play(FadeIn(summary, lag_ratio=0.15))
        self.finish_narration(narration)

    def constraint_panel(self, header: str, caption: str, shape: VMobject) -> VGroup:
        rect = RoundedRectangle(width=4.1, height=3.0, corner_radius=0.08, color=GREY_C, stroke_width=2)
        rect.set_fill(PANEL_BG, opacity=0.9)
        axes = VGroup(
            Line(LEFT * 1.35, RIGHT * 1.35, color=GREY_B, stroke_width=2),
            Line(DOWN * 1.35, UP * 1.35, color=GREY_B, stroke_width=2),
        ).move_to(rect)
        shape.move_to(rect)
        title = MathTex(header, font_size=36, color=shape.get_color()).next_to(rect, UP, buff=0.15)
        note = Text(caption, font_size=23, color=WHITE).next_to(rect, DOWN, buff=0.16)
        return VGroup(rect, title, shape, axes, note)

    def summary_item(self, text: str, color: ManimColor) -> VGroup:
        bullet = Dot(color=color, radius=0.075)
        label = Text(text, font_size=24, color=WHITE)
        return VGroup(bullet, label).arrange(RIGHT, buff=0.18)
