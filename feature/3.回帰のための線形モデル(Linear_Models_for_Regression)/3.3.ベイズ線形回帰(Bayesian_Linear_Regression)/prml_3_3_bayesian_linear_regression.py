from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


DATA_BLUE = BLUE_C
POSTERIOR_TEAL = TEAL_C
PRIOR_PURPLE = PURPLE_C
LIKELIHOOD_ORANGE = ORANGE
MEAN_RED = RED_C
BAND_GREEN = GREEN_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([-0.9, -0.55, -0.2, 0.05, 0.35, 0.65, 0.9])
    noise = np.array([-0.18, 0.12, -0.04, 0.08, -0.10, 0.06, 0.14])
    t = -0.25 + 1.25 * x + noise
    return x, t


def design_matrix(x: np.ndarray | float) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    return np.column_stack([np.ones_like(x_array), x_array])


def posterior(x: np.ndarray, t: np.ndarray, alpha: float = 2.0, beta: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    phi = design_matrix(x)
    precision = alpha * np.eye(2) + beta * phi.T @ phi
    cov = np.linalg.inv(precision)
    mean = beta * cov @ phi.T @ t
    return mean, cov


def predictive(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, beta: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    phi = design_matrix(x)
    y_mean = phi @ mean
    y_var = 1.0 / beta + np.sum((phi @ cov) * phi, axis=1)
    return y_mean, y_var


def sample_weights(mean: np.ndarray, cov: np.ndarray, scale: float = 1.35) -> list[np.ndarray]:
    angles = np.array([0.15, 1.2, 2.35, 3.6, 4.7])
    chol = np.linalg.cholesky(cov)
    return [mean + scale * chol @ np.array([np.cos(angle), np.sin(angle)]) for angle in angles]


class PRML33BayesianLinearRegression(Scene):
    """PRML 3.3 Bayesian linear regression overview.

    Render example:
        uv run manim -pql prml_3_3_bayesian_linear_regression.py PRML33BayesianLinearRegression
    """

    def construct(self) -> None:
        self.camera.background_color = "#111111"
        self.x_all, self.t_all = make_regression_data()

        self.opening_distribution_view()
        self.prior_in_weight_space()
        self.likelihood_and_posterior()
        self.sequential_update()
        self.predictive_distribution()
        self.alpha_beta_controls()
        self.bridge_to_model_comparison()

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
        label = Text("3.3 Bayesian Linear Regression", font_size=18, color=TEXT_GREY)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 34) -> Text:
        title = Text(text, font_size=font_size)
        title.to_edge(UP).shift(DOWN * 0.28)
        return title

    def data_axes(self, width: float = 5.8, height: float = 3.8) -> Axes:
        return Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-1.8, 1.6, 0.8],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def weight_axes(self, width: float = 4.9, height: float = 3.8) -> Axes:
        return Axes(
            x_range=[-1.35, 0.85, 0.5],
            y_range=[-1.0, 2.35, 0.8],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def dots_for(self, axes: Axes, x: np.ndarray, t: np.ndarray, radius: float = 0.06) -> VGroup:
        return VGroup(*[Dot(axes.c2p(float(xi), float(ti)), color=DATA_BLUE, radius=radius) for xi, ti in zip(x, t)])

    def line_for_weights(self, axes: Axes, weights: np.ndarray, color: ManimColor, width: float = 3.0, opacity: float = 1.0) -> VMobject:
        line = axes.plot(lambda u: float(weights[0] + weights[1] * u), x_range=[-1, 1], color=color, use_smoothing=False)
        line.set_stroke(width=width, opacity=opacity)
        return line

    def sample_lines(self, axes: Axes, mean: np.ndarray, cov: np.ndarray, color: ManimColor = PRIOR_PURPLE) -> VGroup:
        return VGroup(
            *[
                self.line_for_weights(axes, weights, color=color, width=2.2, opacity=0.62)
                for weights in sample_weights(mean, cov)
            ]
        )

    def ellipse_for(self, axes: Axes, mean: np.ndarray, cov: np.ndarray, color: ManimColor, n_std: float = 2.0) -> VMobject:
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, 1e-6)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        points = []
        for angle in np.linspace(0, TAU, 121):
            unit = np.array([np.cos(angle), np.sin(angle)])
            coord = mean + n_std * vectors @ (np.sqrt(values) * unit)
            points.append(axes.c2p(float(coord[0]), float(coord[1])))
        ellipse = VMobject(color=color, stroke_width=4)
        ellipse.set_points_smoothly(points)
        return ellipse

    def predictive_band(self, axes: Axes, x_grid: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray, color: ManimColor) -> Polygon:
        sigma = np.sqrt(y_var)
        upper = y_mean + 1.6 * sigma
        lower = y_mean - 1.6 * sigma
        upper_points = [axes.c2p(float(x), float(y)) for x, y in zip(x_grid, upper)]
        lower_points = [axes.c2p(float(x), float(y)) for x, y in zip(x_grid[::-1], lower[::-1])]
        band = Polygon(*upper_points, *lower_points, stroke_width=0)
        band.set_fill(color, opacity=0.22)
        return band

    def axis_labels(self, axes: Axes, x_label: str, y_label: str) -> VGroup:
        x_text = MathTex(x_label, font_size=28).next_to(axes.x_axis, RIGHT, buff=0.12)
        y_text = MathTex(y_label, font_size=28).next_to(axes.y_axis, UP, buff=0.12)
        return VGroup(x_text, y_text)

    def opening_distribution_view(self) -> None:
        self.clear()
        narration = self.start_narration("scene01")
        label = self.section_label()
        title = self.scene_title("点推定から、重みの分布へ")

        axes = self.data_axes(width=6.1, height=3.7).shift(LEFT * 2.35 + DOWN * 0.15)
        dots = self.dots_for(axes, self.x_all[:5], self.t_all[:5])
        mean, cov = posterior(self.x_all[:5], self.t_all[:5])
        single = self.line_for_weights(axes, mean, color=MEAN_RED, width=4.2)
        samples = self.sample_lines(axes, mean, cov, color=POSTERIOR_TEAL)
        labels = self.axis_labels(axes, "x", "t")

        point_panel = VGroup(
            Text("点推定", font_size=28, color=MEAN_RED),
            MathTex(r"\mathbf{w}_{\mathrm{best}}", font_size=34, color=MEAN_RED),
        ).arrange(DOWN, buff=0.3).move_to(RIGHT * 3.6 + UP * 0.9)

        dist_panel = VGroup(
            Text("分布推定", font_size=28, color=POSTERIOR_TEAL),
            MathTex(r"p(\mathbf{w}\mid \mathbf{t})", font_size=34, color=POSTERIOR_TEAL),
            Text("ありそうな直線を残す", font_size=22, color=WHITE),
        ).arrange(DOWN, buff=0.24).move_to(RIGHT * 3.6 + DOWN * 0.8)

        arrow = Arrow(point_panel.get_bottom(), dist_panel.get_top(), buff=0.18, color=TEXT_GREY)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(labels), FadeIn(dots), run_time=1.0)
        self.play(Create(single), FadeIn(point_panel))
        self.play(GrowArrow(arrow), run_time=0.6)
        self.play(FadeOut(single), FadeIn(samples), FadeIn(dist_panel), run_time=1.2)
        self.finish_narration(narration)

    def prior_in_weight_space(self) -> None:
        self.clear()
        narration = self.start_narration("scene02")
        label = self.section_label()
        title = self.scene_title("事前分布を重み空間に置く")

        weight_axes = self.weight_axes().shift(LEFT * 2.75 + DOWN * 0.1)
        data_axes = self.data_axes(width=4.8, height=3.25).shift(RIGHT * 3.0 + DOWN * 0.15)
        axis_labels = VGroup(self.axis_labels(weight_axes, "w_0", "w_1"), self.axis_labels(data_axes, "x", "t"))

        alpha_lo = 1.0
        alpha_hi = 6.0
        prior_mean = np.zeros(2)
        ellipse_lo = self.ellipse_for(weight_axes, prior_mean, np.eye(2) / alpha_lo, PRIOR_PURPLE, n_std=1.8)
        ellipse_hi = self.ellipse_for(weight_axes, prior_mean, np.eye(2) / alpha_hi, PRIOR_PURPLE, n_std=1.8)
        dot = Dot(weight_axes.c2p(0, 0), color=WHITE, radius=0.05)
        samples_lo = self.sample_lines(data_axes, prior_mean, np.eye(2) / alpha_lo, color=PRIOR_PURPLE)
        samples_hi = self.sample_lines(data_axes, prior_mean, np.eye(2) / alpha_hi, color=PRIOR_PURPLE)

        prior_formula = MathTex(
            r"p(\mathbf{w})=\mathcal{N}(\mathbf{w}\mid \mathbf{0}, \alpha^{-1}\mathbf{I})",
            font_size=34,
            color=WHITE,
        ).to_edge(DOWN).shift(UP * 0.18)
        alpha_text = Text("alpha を上げる", font_size=24, color=PRIOR_PURPLE).next_to(weight_axes, UP, buff=0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Create(weight_axes), Create(data_axes), FadeIn(axis_labels), Write(prior_formula))
        self.play(Create(ellipse_lo), FadeIn(dot), FadeIn(samples_lo), run_time=1.2)
        self.play(Write(alpha_text))
        self.play(Transform(ellipse_lo, ellipse_hi), Transform(samples_lo, samples_hi), run_time=1.4)
        self.finish_narration(narration)

    def likelihood_band(self, axes: Axes, x_value: float, t_value: float) -> Line:
        w0_values = np.array([-1.35, 0.85])
        w1_values = (t_value - w0_values) / x_value
        band = Line(
            axes.c2p(float(w0_values[0]), float(w1_values[0])),
            axes.c2p(float(w0_values[1]), float(w1_values[1])),
            color=LIKELIHOOD_ORANGE,
            stroke_width=22,
            stroke_opacity=0.22,
        )
        center = Line(
            axes.c2p(float(w0_values[0]), float(w1_values[0])),
            axes.c2p(float(w0_values[1]), float(w1_values[1])),
            color=LIKELIHOOD_ORANGE,
            stroke_width=4,
            stroke_opacity=0.9,
        )
        return VGroup(band, center)

    def likelihood_and_posterior(self) -> None:
        self.clear()
        narration = self.start_narration("scene03")
        label = self.section_label()
        title = self.scene_title("尤度を掛けると事後分布になる")

        weight_axes = self.weight_axes().shift(LEFT * 2.65 + DOWN * 0.05)
        data_axes = self.data_axes(width=4.9, height=3.25).shift(RIGHT * 3.0 + DOWN * 0.05)
        axis_labels = VGroup(self.axis_labels(weight_axes, "w_0", "w_1"), self.axis_labels(data_axes, "x", "t"))

        x_one = self.x_all[:1]
        t_one = self.t_all[:1]
        prior_cov = np.eye(2) / 2.0
        post_mean, post_cov = posterior(x_one, t_one)
        prior_ellipse = self.ellipse_for(weight_axes, np.zeros(2), prior_cov, PRIOR_PURPLE, n_std=1.8)
        post_ellipse = self.ellipse_for(weight_axes, post_mean, post_cov, POSTERIOR_TEAL, n_std=2.0)
        band = self.likelihood_band(weight_axes, float(x_one[0]), float(t_one[0]))
        point = self.dots_for(data_axes, x_one, t_one, radius=0.075)
        point_line = data_axes.get_vertical_line(data_axes.c2p(float(x_one[0]), float(t_one[0])), color=LIKELIHOOD_ORANGE)

        formula = MathTex(
            r"p(\mathbf{w}\mid \mathbf{t}) \propto p(\mathbf{t}\mid \mathbf{w})\,p(\mathbf{w})",
            font_size=36,
        ).to_edge(DOWN).shift(UP * 0.2)
        captions = VGroup(
            Text("事前分布", font_size=22, color=PRIOR_PURPLE),
            Text("尤度", font_size=22, color=LIKELIHOOD_ORANGE),
            Text("事後分布", font_size=22, color=POSTERIOR_TEAL),
        ).arrange(RIGHT, buff=0.75).next_to(formula, UP, buff=0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Create(weight_axes), Create(data_axes), FadeIn(axis_labels), Write(formula))
        self.play(Create(prior_ellipse), FadeIn(captions[0]))
        self.play(FadeIn(point), Create(point_line), FadeIn(band), FadeIn(captions[1]), run_time=1.1)
        self.play(Create(post_ellipse), FadeIn(captions[2]), run_time=1.2)
        self.finish_narration(narration)

    def sequential_update(self) -> None:
        self.clear()
        narration = self.start_narration("scene04")
        label = self.section_label()
        title = self.scene_title("データを増やすと不確実性が縮む")

        weight_axes = self.weight_axes(width=4.7, height=3.5).shift(LEFT * 3.05 + DOWN * 0.05)
        data_axes = self.data_axes(width=5.2, height=3.5).shift(RIGHT * 2.7 + DOWN * 0.05)
        axis_labels = VGroup(self.axis_labels(weight_axes, "w_0", "w_1"), self.axis_labels(data_axes, "x", "t"))

        self.play(FadeIn(label), Write(title))
        self.play(Create(weight_axes), Create(data_axes), FadeIn(axis_labels))

        current_ellipse: VMobject | None = None
        current_lines: VGroup | None = None
        current_dots = VGroup()
        count_text = Text("N = 0", font_size=26, color=WHITE).next_to(data_axes, UP, buff=0.18)
        self.play(FadeIn(count_text))

        for n in [1, 2, 4, 7]:
            x_n = self.x_all[:n]
            t_n = self.t_all[:n]
            mean, cov = posterior(x_n, t_n)
            next_ellipse = self.ellipse_for(weight_axes, mean, cov, POSTERIOR_TEAL, n_std=2.0)
            next_lines = self.sample_lines(data_axes, mean, cov, color=POSTERIOR_TEAL)
            next_dots = self.dots_for(data_axes, x_n, t_n)
            next_count = Text(f"N = {n}", font_size=26, color=WHITE).next_to(data_axes, UP, buff=0.18)

            if current_ellipse is None or current_lines is None:
                self.play(FadeIn(next_dots), Create(next_ellipse), FadeIn(next_lines), Transform(count_text, next_count), run_time=1.0)
            else:
                self.play(
                    Transform(current_dots, next_dots),
                    Transform(current_ellipse, next_ellipse),
                    Transform(current_lines, next_lines),
                    Transform(count_text, next_count),
                    run_time=1.15,
                )
                continue
            current_dots = next_dots
            current_ellipse = next_ellipse
            current_lines = next_lines

        note = Text("候補の直線が分布ごと絞られる", font_size=26, color=WHITE).to_edge(DOWN).shift(UP * 0.2)
        self.play(Write(note))
        self.finish_narration(narration)

    def predictive_distribution(self) -> None:
        self.clear()
        narration = self.start_narration("scene05")
        label = self.section_label()
        title = self.scene_title("予測は平均と幅を持つ")

        axes = self.data_axes(width=7.4, height=4.0).shift(DOWN * 0.05)
        labels = self.axis_labels(axes, "x", "t")
        x_grid = np.linspace(-1, 1, 121)
        mean, cov = posterior(self.x_all, self.t_all)
        y_mean, y_var = predictive(x_grid, mean, cov)
        band = self.predictive_band(axes, x_grid, y_mean, y_var, BAND_GREEN)
        mean_curve = axes.plot(lambda u: float(mean[0] + mean[1] * u), x_range=[-1, 1], color=MEAN_RED, use_smoothing=False)
        mean_curve.set_stroke(width=4)
        dots = self.dots_for(axes, self.x_all, self.t_all)

        formula = MathTex(
            r"p(t_\ast\mid x_\ast,\mathbf{t})=\mathcal{N}(m_N^T\phi(x_\ast),\ \sigma_N^2(x_\ast))",
            font_size=31,
        ).to_edge(DOWN).shift(UP * 0.15)
        components = VGroup(
            Text("平均", font_size=22, color=MEAN_RED),
            Text("予測の幅", font_size=22, color=BAND_GREEN),
            Text("観測ノイズ + 重みの不確実性", font_size=22, color=WHITE),
        ).arrange(RIGHT, buff=0.35).next_to(axes, UP, buff=0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(labels), FadeIn(dots), run_time=1.0)
        self.play(FadeIn(band), Create(mean_curve), FadeIn(components), run_time=1.3)
        self.play(Write(formula))
        self.finish_narration(narration)

    def alpha_beta_controls(self) -> None:
        self.clear()
        narration = self.start_narration("scene06")
        label = self.section_label()
        title = self.scene_title("alpha と beta は不確実性のつまみ")

        axes = self.data_axes(width=6.1, height=3.5).shift(LEFT * 2.1 + DOWN * 0.15)
        dots = self.dots_for(axes, self.x_all, self.t_all)
        labels = self.axis_labels(axes, "x", "t")
        x_grid = np.linspace(-1, 1, 121)

        mean_lo, cov_lo = posterior(self.x_all, self.t_all, alpha=0.6, beta=8.0)
        y_lo, var_lo = predictive(x_grid, mean_lo, cov_lo, beta=8.0)
        band_lo = self.predictive_band(axes, x_grid, y_lo, var_lo, BAND_GREEN)
        curve_lo = self.line_for_weights(axes, mean_lo, MEAN_RED, width=4)

        mean_hi, cov_hi = posterior(self.x_all, self.t_all, alpha=4.0, beta=45.0)
        y_hi, var_hi = predictive(x_grid, mean_hi, cov_hi, beta=45.0)
        band_hi = self.predictive_band(axes, x_grid, y_hi, var_hi, BAND_GREEN)
        curve_hi = self.line_for_weights(axes, mean_hi, MEAN_RED, width=4)

        alpha_slider = self.slider("alpha", 0.22, PRIOR_PURPLE).move_to(RIGHT * 3.55 + UP * 0.9)
        beta_slider = self.slider("beta", 0.30, LIKELIHOOD_ORANGE).move_to(RIGHT * 3.55 + DOWN * 0.2)
        alpha_hi_slider = self.slider("alpha", 0.72, PRIOR_PURPLE).move_to(RIGHT * 3.55 + UP * 0.9)
        beta_hi_slider = self.slider("beta", 0.82, LIKELIHOOD_ORANGE).move_to(RIGHT * 3.55 + DOWN * 0.2)
        note = Text("alpha: 事前分布の精度\nbeta: 観測ノイズの精度", font_size=25, color=WHITE, line_spacing=0.9)
        note.move_to(RIGHT * 3.55 + DOWN * 1.25)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(labels), FadeIn(dots), FadeIn(alpha_slider), FadeIn(beta_slider), Write(note))
        self.play(FadeIn(band_lo), Create(curve_lo), run_time=1.0)
        self.play(
            Transform(alpha_slider, alpha_hi_slider),
            Transform(beta_slider, beta_hi_slider),
            Transform(band_lo, band_hi),
            Transform(curve_lo, curve_hi),
            run_time=1.4,
        )
        self.finish_narration(narration)

    def slider(self, label: str, value: float, color: ManimColor) -> VGroup:
        track = Line(LEFT * 1.0, RIGHT * 1.0, color=GREY_B, stroke_width=6)
        knob = Dot(track.point_from_proportion(value), color=color, radius=0.11)
        text = Text(label, font_size=24, color=color).next_to(track, UP, buff=0.18)
        return VGroup(track, knob, text)

    def bridge_to_model_comparison(self) -> None:
        self.clear()
        narration = self.start_narration("scene07")
        label = self.section_label()
        title = self.scene_title("重みの積分からモデル比較へ")

        left = VGroup(
            Text("重みの分布", font_size=28, color=POSTERIOR_TEAL),
            MathTex(r"p(\mathbf{w}\mid \mathbf{t})", font_size=40, color=POSTERIOR_TEAL),
        ).arrange(DOWN, buff=0.28).move_to(LEFT * 4.0)
        middle = VGroup(
            Text("予測分布", font_size=28, color=BAND_GREEN),
            MathTex(r"p(t_\ast\mid x_\ast,\mathbf{t})", font_size=38, color=BAND_GREEN),
        ).arrange(DOWN, buff=0.28).move_to(ORIGIN)
        right = VGroup(
            Text("モデルの証拠", font_size=28, color=LIKELIHOOD_ORANGE),
            MathTex(r"p(\mathbf{t}\mid M)", font_size=40, color=LIKELIHOOD_ORANGE),
        ).arrange(DOWN, buff=0.28).move_to(RIGHT * 4.0)
        arrow1 = Arrow(left.get_right(), middle.get_left(), buff=0.25, color=TEXT_GREY)
        arrow2 = Arrow(middle.get_right(), right.get_left(), buff=0.25, color=TEXT_GREY)
        integral = MathTex(
            r"p(\mathbf{t}\mid M)=\int p(\mathbf{t}\mid\mathbf{w},M)p(\mathbf{w}\mid M)d\mathbf{w}",
            font_size=34,
        ).to_edge(DOWN).shift(UP * 0.3)
        summary = Text("分布を残すから、モデルの比較にも進める", font_size=28, color=WHITE)
        summary.next_to(integral, UP, buff=0.35)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(left))
        self.play(GrowArrow(arrow1), FadeIn(middle))
        self.play(GrowArrow(arrow2), FadeIn(right))
        self.play(Write(summary), Write(integral))
        self.finish_narration(narration)
