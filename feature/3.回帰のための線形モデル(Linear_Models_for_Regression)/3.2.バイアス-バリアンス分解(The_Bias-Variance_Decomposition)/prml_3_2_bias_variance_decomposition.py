from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


JAPANESE_FONT = "Noto Sans CJK JP"
TRUE_GREEN = GREEN_C
FIT_RED = RED_C
MEAN_ORANGE = ORANGE
BIAS_BLUE = BLUE_C
VAR_PURPLE = PURPLE_C
NOISE_GREY = GREY_B
TEXT_GREY = GREY_B

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def h_function(x: np.ndarray | float) -> np.ndarray:
    return np.sin(2.0 * np.pi * np.asarray(x))


def make_datasets(count: int = 36, n: int = 25, noise_std: float = 0.28, seed: int = 32) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    datasets = []
    for _ in range(count):
        x = np.sort(rng.uniform(0.0, 1.0, size=n))
        t = h_function(x) + rng.normal(0.0, noise_std, size=n)
        datasets.append((x, t))
    return datasets


def design_matrix(x: np.ndarray | float, centers: np.ndarray, scale: float = 0.08) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    gaussians = np.exp(-0.5 * ((x_array[:, None] - centers[None, :]) / scale) ** 2)
    return np.column_stack([np.ones(len(x_array)), gaussians])


def fit_ridge(x: np.ndarray, t: np.ndarray, centers: np.ndarray, lam: float) -> np.ndarray:
    phi = design_matrix(x, centers)
    penalty = lam * np.eye(phi.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ t)


def predict(w: np.ndarray, x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return design_matrix(x, centers) @ w


def clip_y(y: np.ndarray) -> np.ndarray:
    return np.clip(y, -1.45, 1.45)


class PRML32BiasVarianceDecomposition(Scene):
    """PRML 3.2 bias-variance decomposition.

    Render example:
        uv run manim -pql prml_3_2_bias_variance_decomposition.py PRML32BiasVarianceDecomposition
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.centers = np.linspace(0.0, 1.0, 24)
        self.x_grid = np.linspace(0.0, 1.0, 160)
        self.true_grid = h_function(self.x_grid)
        self.datasets = make_datasets()

        self.opening_tradeoff()
        self.squared_loss_start()
        self.dataset_ensemble()
        self.algebra_decomposition()
        self.large_lambda_case()
        self.small_lambda_case()
        self.bias_variance_curves()
        self.bridge_to_bayesian()

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

    def section_label(self, text: str) -> Text:
        label = Text(text, font_size=18, color=TEXT_GREY)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 34) -> Text:
        title = Text(text, font_size=font_size, color=WHITE)
        title.to_edge(UP).shift(DOWN * 0.32)
        return title

    def make_axes(self, width: float = 5.8, height: float = 3.3) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.5, 1.5, 0.75],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def curve_from_values(self, axes: Axes, y_values: np.ndarray, color: ManimColor, width: float = 4, opacity: float = 1.0) -> VMobject:
        curve = VMobject(color=color)
        points = [axes.c2p(float(x), float(y)) for x, y in zip(self.x_grid, clip_y(y_values))]
        curve.set_points_as_corners(points)
        curve.set_stroke(width=width, opacity=opacity)
        return curve

    def fit_predictions(self, ln_lambda: float) -> np.ndarray:
        lam = float(np.exp(ln_lambda))
        predictions = []
        for x, t in self.datasets:
            w = fit_ridge(x, t, self.centers, lam)
            predictions.append(predict(w, self.x_grid, self.centers))
        return np.asarray(predictions)

    def model_panel(self, ln_lambda: float, title: str, subtitle: str, shift: np.ndarray, width: float = 5.0) -> VGroup:
        axes = self.make_axes(width=width, height=2.9).shift(shift)
        predictions = self.fit_predictions(ln_lambda)
        true_curve = self.curve_from_values(axes, self.true_grid, TRUE_GREEN, width=4)
        fit_curves = VGroup(
            *[
                self.curve_from_values(axes, prediction, FIT_RED, width=2, opacity=0.24)
                for prediction in predictions[:20]
            ]
        )
        mean_curve = self.curve_from_values(axes, predictions.mean(axis=0), MEAN_ORANGE, width=5)
        title_text = Text(title, font_size=24, color=WHITE).next_to(axes, UP, buff=0.18)
        subtitle_text = Text(subtitle, font_size=20, color=TEXT_GREY).next_to(axes, DOWN, buff=0.18)
        legend = VGroup(
            Line(ORIGIN, RIGHT * 0.35, color=TRUE_GREEN, stroke_width=4),
            Text("h(x)", font_size=18),
            Line(ORIGIN, RIGHT * 0.35, color=MEAN_ORANGE, stroke_width=5),
            Text("mean", font_size=18),
        ).arrange(RIGHT, buff=0.12)
        legend.next_to(axes, UP, buff=0.05).align_to(axes, RIGHT)
        return VGroup(axes, fit_curves, true_curve, mean_curve, title_text, subtitle_text, legend)

    def opening_tradeoff(self) -> None:
        self.clear()
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 3.2")
        title = self.scene_title("バイアスとバリアンスで失敗を分ける", font_size=33)

        high_bias = self.model_panel(2.6, "強い正則化", "high bias / low variance", LEFT * 3.05 + DOWN * 0.1, width=4.65)
        high_variance = self.model_panel(-2.4, "弱い正則化", "low bias / high variance", RIGHT * 3.05 + DOWN * 0.1, width=4.65)
        bias_tag = self.term_tag("bias", "平均的なズレ", BIAS_BLUE).to_edge(DOWN).shift(LEFT * 2.2 + UP * 0.25)
        variance_tag = self.term_tag("variance", "データによる揺れ", VAR_PURPLE).to_edge(DOWN).shift(RIGHT * 2.2 + UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(high_bias[0]), Create(high_bias[2]), FadeIn(high_bias[4:]), run_time=1.0)
        self.play(LaggedStart(*[Create(curve) for curve in high_bias[1]], lag_ratio=0.04), Create(high_bias[3]), run_time=1.4)
        self.play(FadeIn(high_variance[0]), Create(high_variance[2]), FadeIn(high_variance[4:]), run_time=1.0)
        self.play(LaggedStart(*[Create(curve) for curve in high_variance[1]], lag_ratio=0.035), Create(high_variance[3]), run_time=1.4)
        self.play(FadeIn(bias_tag), FadeIn(variance_tag))
        self.finish_narration(narration)

    def term_tag(self, main: str, sub: str, color: ManimColor) -> VGroup:
        box = RoundedRectangle(width=2.6, height=0.92, corner_radius=0.08, color=color, stroke_width=3)
        box.set_fill("#1B1B1B", opacity=0.92)
        main_text = Text(main, font_size=26, color=color)
        sub_text = Text(sub, font_size=18, color=WHITE).next_to(main_text, DOWN, buff=0.08)
        content = VGroup(main_text, sub_text).move_to(box)
        return VGroup(box, content)

    def squared_loss_start(self) -> None:
        self.clear()
        narration = self.start_narration("scene02")
        label = self.section_label("Squared loss")
        title = self.scene_title("理想予測 h(x) と残る noise")

        axes = self.make_axes(width=6.8, height=3.45).shift(LEFT * 2.2 + DOWN * 0.15)
        true_curve = self.curve_from_values(axes, self.true_grid, TRUE_GREEN, width=5)
        model_y = h_function(self.x_grid) * 0.62 + 0.24
        model_curve = self.curve_from_values(axes, model_y, FIT_RED, width=5)
        x0 = 0.68
        h0 = float(h_function(x0))
        y0 = float(h_function(x0) * 0.62 + 0.24)
        t0 = h0 - 0.46
        gap_model = DashedLine(axes.c2p(x0, h0), axes.c2p(x0, y0), color=BIAS_BLUE, stroke_width=4)
        gap_noise = DashedLine(axes.c2p(x0 + 0.04, h0), axes.c2p(x0 + 0.04, t0), color=NOISE_GREY, stroke_width=4)
        dot_t = Dot(axes.c2p(x0 + 0.04, t0), color=WHITE, radius=0.075)
        labels = VGroup(
            MathTex("h(x)", font_size=32, color=TRUE_GREEN).next_to(true_curve, LEFT, buff=0.08),
            MathTex("y(x)", font_size=32, color=FIT_RED).next_to(model_curve, RIGHT, buff=0.08),
            MathTex("t", font_size=30, color=WHITE).next_to(dot_t, DOWN, buff=0.1),
            Text("model gap", font_size=20, color=BIAS_BLUE).next_to(gap_model, RIGHT, buff=0.12),
            Text("noise", font_size=20, color=NOISE_GREY).next_to(gap_noise, RIGHT, buff=0.12),
        )

        formula = VGroup(
            MathTex(r"h(x)=\mathbb{E}[t\mid x]", font_size=36, color=TRUE_GREEN),
            MathTex(r"\mathbb{E}[L]", r"=", r"\{y(x)-h(x)\}^2", r"+", r"\mathrm{noise}", font_size=34),
        ).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        formula[1][2].set_color(BIAS_BLUE)
        formula[1][4].set_color(NOISE_GREY)
        formula.move_to(RIGHT * 3.35 + DOWN * 0.2)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), Create(true_curve), Create(model_curve))
        self.play(FadeIn(dot_t), Create(gap_model), Create(gap_noise), FadeIn(labels))
        self.play(Write(formula))
        self.finish_narration(narration)

    def dataset_ensemble(self) -> None:
        self.clear()
        narration = self.start_narration("scene03")
        label = self.section_label("Ensemble of data sets")
        title = self.scene_title("訓練データ集合を何度も取り直す")

        axes = self.make_axes(width=7.2, height=3.6).shift(LEFT * 1.65 + DOWN * 0.1)
        predictions = self.fit_predictions(-0.31)
        true_curve = self.curve_from_values(axes, self.true_grid, TRUE_GREEN, width=5)
        fit_curves = VGroup(
            *[
                self.curve_from_values(axes, prediction, FIT_RED, width=2, opacity=0.28)
                for prediction in predictions[:18]
            ]
        )
        mean_curve = self.curve_from_values(axes, predictions.mean(axis=0), MEAN_ORANGE, width=5)
        sample_dots = VGroup(
            *[Dot(axes.c2p(float(x), float(t)), radius=0.045, color=BLUE_C) for x, t in zip(*self.datasets[0])]
        )
        equation = MathTex(r"\bar{y}(x)=\mathbb{E}_{\mathcal{D}}[y(x;\mathcal{D})]", font_size=38, color=MEAN_ORANGE)
        equation.move_to(RIGHT * 3.35 + UP * 0.65)
        note = Text("同じ N=25 のデータ集合を想像上で何度も引く", font_size=23, color=WHITE)
        note.move_to(RIGHT * 3.35 + DOWN * 0.35)
        variance_note = Text("散らばり = variance", font_size=27, color=VAR_PURPLE)
        variance_note.next_to(note, DOWN, buff=0.38)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), Create(true_curve), FadeIn(sample_dots), run_time=1.0)
        self.play(LaggedStart(*[Create(curve) for curve in fit_curves], lag_ratio=0.06), run_time=2.0)
        self.play(Create(mean_curve), Write(equation), FadeIn(note))
        self.play(Write(variance_note))
        self.finish_narration(narration)

    def algebra_decomposition(self) -> None:
        self.clear()
        narration = self.start_narration("scene04")
        label = self.section_label("Algebra")
        title = self.scene_title("平均予測を足して引く")

        line1 = MathTex(r"\{y(x;\mathcal{D})-h(x)\}^2", font_size=44)
        line2 = MathTex(
            r"=\{y(x;\mathcal{D})-\bar{y}(x)+\bar{y}(x)-h(x)\}^2",
            font_size=40,
        )
        line3 = MathTex(
            r"\mathbb{E}_{\mathcal{D}}\{y(x;\mathcal{D})-h(x)\}^2",
            r"=",
            r"\{\bar{y}(x)-h(x)\}^2",
            r"+",
            r"\mathbb{E}_{\mathcal{D}}\{y(x;\mathcal{D})-\bar{y}(x)\}^2",
            font_size=34,
        )
        line3[2].set_color(BIAS_BLUE)
        line3[4].set_color(VAR_PURPLE)
        line4 = MathTex(r"\mathrm{expected\ loss}", "=", r"(\mathrm{bias})^2", "+", r"\mathrm{variance}", "+", r"\mathrm{noise}", font_size=40)
        line4[2].set_color(BIAS_BLUE)
        line4[4].set_color(VAR_PURPLE)
        line4[6].set_color(NOISE_GREY)
        formulas = VGroup(line1, line2, line3, line4).arrange(DOWN, buff=0.55)
        formulas.shift(DOWN * 0.15)

        bias_card = self.term_tag("bias^2", "平均予測のズレ", BIAS_BLUE).to_edge(DOWN).shift(LEFT * 2.6 + UP * 0.2)
        var_card = self.term_tag("variance", "平均の周りの揺れ", VAR_PURPLE).to_edge(DOWN).shift(RIGHT * 2.6 + UP * 0.2)
        cross = Text("交差項は平均を取ると 0", font_size=24, color=TEXT_GREY)
        cross.next_to(line2, DOWN, buff=0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Write(line1))
        self.play(TransformMatchingTex(line1.copy(), line2), FadeIn(cross))
        self.play(Write(line3))
        self.play(FadeOut(cross), FadeIn(bias_card), FadeIn(var_card))
        self.play(Write(line4))
        self.finish_narration(narration)

    def lambda_case_scene(self, scene_id: str, ln_lambda: float, title_text: str, summary: str, main_color: ManimColor) -> None:
        self.clear()
        narration = self.start_narration(scene_id)
        label = self.section_label("Regularized Gaussian basis model")
        title = self.scene_title(title_text)

        panel = self.model_panel(ln_lambda, f"ln lambda = {ln_lambda:.1f}", summary, LEFT * 2.25 + DOWN * 0.12, width=6.15)
        predictions = self.fit_predictions(ln_lambda)
        mean = predictions.mean(axis=0)
        bias2 = float(np.mean((mean - self.true_grid) ** 2))
        variance = float(np.mean(np.mean((predictions - mean[None, :]) ** 2, axis=0)))

        meter = self.metric_meters(bias2, variance).move_to(RIGHT * 3.65 + DOWN * 0.25)
        headline = Text(summary, font_size=28, color=main_color).next_to(meter, UP, buff=0.55)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(panel[0]), Create(panel[2]), FadeIn(panel[4:]), run_time=1.0)
        self.play(LaggedStart(*[Create(curve) for curve in panel[1]], lag_ratio=0.04), run_time=1.6)
        self.play(Create(panel[3]), FadeIn(headline), FadeIn(meter))
        self.finish_narration(narration)

    def metric_meters(self, bias2: float, variance: float) -> VGroup:
        max_value = 0.55
        rows = VGroup()
        for label, value, color in [("bias^2", bias2, BIAS_BLUE), ("variance", variance, VAR_PURPLE)]:
            base = Rectangle(width=2.4, height=0.28, color=GREY_D, fill_color="#222222", fill_opacity=1)
            fill_width = max(0.08, 2.4 * min(value / max_value, 1.0))
            fill = Rectangle(width=fill_width, height=0.28, color=color, fill_color=color, fill_opacity=0.9)
            fill.align_to(base, LEFT).move_to(base.get_center() + LEFT * (2.4 - fill_width) / 2)
            name = Text(label, font_size=22, color=color).next_to(base, LEFT, buff=0.22)
            number = Text(f"{value:.3f}", font_size=20, color=WHITE).next_to(base, RIGHT, buff=0.18)
            rows.add(VGroup(name, base, fill, number))
        rows.arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        return rows

    def large_lambda_case(self) -> None:
        self.lambda_case_scene("scene05", 2.6, "正則化が強い: 高 bias / 低 variance", "high bias / low variance", BIAS_BLUE)

    def small_lambda_case(self) -> None:
        self.lambda_case_scene("scene06", -2.4, "正則化が弱い: 低 bias / 高 variance", "low bias / high variance", VAR_PURPLE)

    def compute_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ln_lambdas = np.linspace(-3.0, 2.8, 16)
        bias2_values = []
        variance_values = []
        for ln_lambda in ln_lambdas:
            predictions = self.fit_predictions(float(ln_lambda))
            mean = predictions.mean(axis=0)
            bias2_values.append(np.mean((mean - self.true_grid) ** 2))
            variance_values.append(np.mean(np.mean((predictions - mean[None, :]) ** 2, axis=0)))
        return ln_lambdas, np.asarray(bias2_values), np.asarray(variance_values)

    def bias_variance_curves(self) -> None:
        self.clear()
        narration = self.start_narration("scene07")
        label = self.section_label("Trade-off")
        title = self.scene_title("bias^2 と variance の合計を小さくする", font_size=31)

        ln_lambdas, bias2, variance = self.compute_metrics()
        total = bias2 + variance
        y_max = float(max(total) * 1.12)
        axes = Axes(
            x_range=[-3.0, 2.8, 1.0],
            y_range=[0.0, y_max, y_max / 4],
            x_length=8.4,
            y_length=4.25,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.15)
        x_label = MathTex(r"\ln \lambda", font_size=30).next_to(axes.x_axis, RIGHT, buff=0.12)
        y_label = Text("error contribution", font_size=22, color=TEXT_GREY).next_to(axes.y_axis, UP, buff=0.18)

        bias_curve = axes.plot_line_graph(ln_lambdas, bias2, line_color=BIAS_BLUE, add_vertex_dots=False, stroke_width=4)
        var_curve = axes.plot_line_graph(ln_lambdas, variance, line_color=VAR_PURPLE, add_vertex_dots=False, stroke_width=4)
        total_curve = axes.plot_line_graph(ln_lambdas, total, line_color=MEAN_ORANGE, add_vertex_dots=False, stroke_width=5)
        min_index = int(np.argmin(total))
        min_dot = Dot(axes.c2p(float(ln_lambdas[min_index]), float(total[min_index])), color=MEAN_ORANGE, radius=0.09)
        min_line = DashedLine(
            axes.c2p(float(ln_lambdas[min_index]), 0.0),
            axes.c2p(float(ln_lambdas[min_index]), float(total[min_index])),
            color=MEAN_ORANGE,
            stroke_width=3,
        )
        min_text = Text("中間で最小", font_size=24, color=MEAN_ORANGE).next_to(min_dot, UP, buff=0.18)
        legend = VGroup(
            Line(ORIGIN, RIGHT * 0.45, color=BIAS_BLUE, stroke_width=4),
            Text("bias^2", font_size=21),
            Line(ORIGIN, RIGHT * 0.45, color=VAR_PURPLE, stroke_width=4),
            Text("variance", font_size=21),
            Line(ORIGIN, RIGHT * 0.45, color=MEAN_ORANGE, stroke_width=5),
            Text("sum", font_size=21),
        ).arrange(RIGHT, buff=0.15).to_corner(UR).shift(DOWN * 0.82)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), Write(x_label), Write(y_label), FadeIn(legend))
        self.play(Create(bias_curve), run_time=1.1)
        self.play(Create(var_curve), run_time=1.1)
        self.play(Create(total_curve), FadeIn(min_dot), Create(min_line), Write(min_text), run_time=1.2)
        self.finish_narration(narration)

    def bridge_to_bayesian(self) -> None:
        self.clear()
        narration = self.start_narration("scene08")
        label = self.section_label("From 3.2 to 3.3")
        title = self.scene_title("分解は洞察、次はベイズ的な平均へ")

        many = self.dataset_stack(5, "many independent data sets", FIT_RED).move_to(LEFT * 3.7 + UP * 0.55)
        single = self.dataset_stack(1, "single observed data set", BLUE_C).move_to(ORIGIN + UP * 0.55)
        bayes = self.bayesian_box().move_to(RIGHT * 3.65 + UP * 0.55)
        arrow1 = Arrow(many.get_right(), single.get_left(), buff=0.28, color=GREY_B)
        arrow2 = Arrow(single.get_right(), bayes.get_left(), buff=0.28, color=GREY_B)

        limitation = Text("実際に手元にあるのは、たいてい 1 つの訓練データだけ", font_size=27, color=WHITE)
        limitation.to_edge(DOWN).shift(UP * 1.05)
        conclusion = MathTex(r"\int y(x,w)\,p(w\mid \mathcal{D})\,dw", font_size=42, color=MEAN_ORANGE)
        conclusion.next_to(limitation, DOWN, buff=0.35)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(many), GrowArrow(arrow1), FadeIn(single))
        self.play(GrowArrow(arrow2), FadeIn(bayes))
        self.play(Write(limitation))
        self.play(Write(conclusion), bayes.animate.set_stroke(MEAN_ORANGE, width=4))
        self.finish_narration(narration)

    def dataset_stack(self, count: int, label: str, color: ManimColor) -> VGroup:
        pages = VGroup()
        for i in range(count):
            rect = RoundedRectangle(width=2.05, height=1.25, corner_radius=0.08, color=color, stroke_width=2)
            rect.set_fill("#1D1D1D", opacity=1.0)
            rect.shift(RIGHT * 0.12 * i + DOWN * 0.08 * i)
            pages.add(rect)
        dots = VGroup(*[Dot(radius=0.04, color=color) for _ in range(12)]).arrange_in_grid(rows=3, cols=4, buff=0.18)
        dots.move_to(pages[-1])
        text = Text(label, font_size=20, color=WHITE).next_to(pages, DOWN, buff=0.25)
        return VGroup(pages, dots, text)

    def bayesian_box(self) -> VGroup:
        box = RoundedRectangle(width=2.45, height=1.55, corner_radius=0.08, color=MEAN_ORANGE, stroke_width=3)
        box.set_fill("#1D1D1D", opacity=1.0)
        text = VGroup(
            Text("Bayesian", font_size=25, color=MEAN_ORANGE),
            Text("average over", font_size=19, color=WHITE),
            Text("parameters", font_size=19, color=WHITE),
        ).arrange(DOWN, buff=0.07).move_to(box)
        return VGroup(box, text)
