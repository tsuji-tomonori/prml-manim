from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


RED_CLASS = RED_C
BLUE_CLASS = BLUE_C
GREEN_CLASS = GREEN_C
POSTERIOR_YELLOW = YELLOW
ACCENT_ORANGE = ORANGE
ACCENT_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(value)))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def gaussian_pdf(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    return np.exp(-0.5 * ((np.asarray(x) - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def make_gaussian_points(mean: np.ndarray, cov: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, n)


class PRML42ProbabilisticGenerativeModels(Scene):
    """PRML 4.2 probabilistic generative models overview.

    Render example:
        uv run manim -pql prml_4_2_probabilistic_generative_models.py PRML42ProbabilisticGenerativeModels
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.shared_cov = np.array([[0.72, 0.22], [0.22, 0.42]])
        self.mu1 = np.array([-1.35, 0.52])
        self.mu2 = np.array([1.25, -0.38])
        self.mu3 = np.array([0.15, 1.45])
        self.red_points = make_gaussian_points(self.mu1, self.shared_cov, 28, 42)
        self.blue_points = make_gaussian_points(self.mu2, self.shared_cov, 30, 43)
        self.green_points = make_gaussian_points(self.mu3, self.shared_cov, 24, 44)

        self.opening_generative_route()
        self.log_odds_sigmoid()
        self.shared_covariance_linear_boundary()
        self.prior_shift()
        self.softmax_multiclass()
        self.maximum_likelihood_solution()
        self.discrete_naive_bayes()
        self.exponential_family_summary()

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

    def make_class_axes(self, width: float = 6.6, height: float = 4.8) -> Axes:
        return Axes(
            x_range=[-3.3, 3.3, 1],
            y_range=[-2.4, 2.6, 1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_points(self, axes: Axes, points: np.ndarray, color: ManimColor, radius: float = 0.055) -> VGroup:
        return VGroup(*[Dot(axes.c2p(float(x), float(y)), radius=radius, color=color) for x, y in points])

    def gaussian_contour(self, axes: Axes, mean: np.ndarray, cov: np.ndarray, color: ManimColor, scale: float = 1.55) -> VMobject:
        vals, vecs = np.linalg.eigh(cov)
        transform = vecs @ np.diag(np.sqrt(vals))
        points = []
        for theta in np.linspace(0, TAU, 120):
            offset = scale * transform @ np.array([math.cos(theta), math.sin(theta)])
            points.append(axes.c2p(float(mean[0] + offset[0]), float(mean[1] + offset[1])))
        curve = VMobject(color=color)
        curve.set_points_smoothly(points)
        curve.set_stroke(width=3.2, opacity=0.9)
        return curve

    def linear_params(self, mu_a: np.ndarray, mu_b: np.ndarray, prior_a: float = 0.5, prior_b: float = 0.5) -> tuple[np.ndarray, float]:
        inv_cov = np.linalg.inv(self.shared_cov)
        w = inv_cov @ (mu_a - mu_b)
        w0 = -0.5 * float(mu_a.T @ inv_cov @ mu_a) + 0.5 * float(mu_b.T @ inv_cov @ mu_b) + math.log(prior_a / prior_b)
        return w, w0

    def class_score_params(self, mean: np.ndarray, prior: float) -> tuple[np.ndarray, float]:
        inv_cov = np.linalg.inv(self.shared_cov)
        w = inv_cov @ mean
        w0 = -0.5 * float(mean.T @ inv_cov @ mean) + math.log(prior)
        return w, w0

    def boundary_line(self, axes: Axes, w: np.ndarray, w0: float, color: ManimColor, stroke_width: float = 4.0) -> Line:
        x_min, x_max = float(axes.x_range[0]), float(axes.x_range[1])
        y_min, y_max = float(axes.y_range[0]), float(axes.y_range[1])
        if abs(float(w[1])) > 1e-6:
            y_left = -(float(w[0]) * x_min + w0) / float(w[1])
            y_right = -(float(w[0]) * x_max + w0) / float(w[1])
            line = Line(axes.c2p(x_min, y_left), axes.c2p(x_max, y_right), color=color, stroke_width=stroke_width)
        else:
            x_value = -w0 / float(w[0])
            line = Line(axes.c2p(x_value, y_min), axes.c2p(x_value, y_max), color=color, stroke_width=stroke_width)
        return line

    def named_box(
        self,
        title: str,
        body: str,
        color: ManimColor,
        width: float = 3.0,
        height: float = 0.9,
        title_size: int = 23,
        body_size: int = 17,
    ) -> VGroup:
        box = RoundedRectangle(width=width, height=height, corner_radius=0.1, color=color, stroke_width=2.4)
        box.set_fill(color, opacity=0.13)
        title_text = Text(title, font_size=title_size, color=color)
        body_text = Text(body, font_size=body_size, color=WHITE)
        text = VGroup(title_text, body_text).arrange(DOWN, buff=0.08)
        text.move_to(box.get_center())
        return VGroup(box, text)

    def formula_panel(self, formulas: list[Mobject], width: float = 5.1, height: float = 2.1, color: ManimColor = GREY_B) -> VGroup:
        box = RoundedRectangle(width=width, height=height, corner_radius=0.1, color=color, stroke_width=2)
        box.set_fill(BLACK, opacity=0.2)
        content = VGroup(*formulas).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        content.move_to(box.get_center())
        return VGroup(box, content)

    def make_slider(self, label: str, values: list[str], index: int, color: ManimColor = ACCENT_ORANGE) -> tuple[VGroup, Dot]:
        line = Line(LEFT * 2.0, RIGHT * 2.0, color=GREY_B, stroke_width=4)
        ticks = VGroup()
        labels = VGroup()
        for i, value in enumerate(values):
            point = line.point_from_proportion(i / (len(values) - 1))
            ticks.add(Line(point + DOWN * 0.07, point + UP * 0.07, color=GREY_B, stroke_width=3))
            labels.add(Text(value, font_size=17, color=TEXT_GREY).next_to(point, DOWN, buff=0.12))
        knob = Dot(line.point_from_proportion(index / (len(values) - 1)), color=color, radius=0.09)
        title = Text(label, font_size=21, color=color).next_to(line, UP, buff=0.15)
        return VGroup(line, ticks, labels, title), knob

    def opening_generative_route(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 4.2 Probabilistic Generative Models")
        title = self.scene_title("クラスごとの生成過程から、分類確率を作る", font_size=33)
        axes = self.make_class_axes(width=5.7, height=4.2).shift(LEFT * 2.9 + DOWN * 0.2)
        red_dots = self.make_points(axes, self.red_points, RED_CLASS)
        blue_dots = self.make_points(axes, self.blue_points, BLUE_CLASS)
        red_contour = self.gaussian_contour(axes, self.mu1, self.shared_cov, RED_CLASS)
        blue_contour = self.gaussian_contour(axes, self.mu2, self.shared_cov, BLUE_CLASS)
        x_star = RegularPolygon(n=4, radius=0.12, color=WHITE, fill_opacity=1).rotate(PI / 4)
        x_star.move_to(axes.c2p(0.05, 0.05))
        x_label = MathTex(r"x", font_size=32, color=WHITE).next_to(x_star, UP, buff=0.08)

        likelihood = self.named_box(r"p(x|C_k)", "クラスごとの密度", BLUE_CLASS, width=3.25)
        prior = self.named_box(r"p(C_k)", "クラスの出やすさ", ACCENT_ORANGE, width=3.25)
        bayes = self.named_box(r"Bayes", "事後確率へ変換", GREEN_CLASS, width=3.25)
        posterior = self.named_box(r"p(C_k|x)", "分類に使う確率", POSTERIOR_YELLOW, width=3.25)
        arrows = VGroup(*[Arrow(DOWN * 0.18, DOWN * 0.55, color=TEXT_GREY, buff=0) for _ in range(3)])
        route = VGroup(likelihood, arrows[0], prior, arrows[1], bayes, arrows[2], posterior).arrange(DOWN, buff=0.12)
        route.shift(RIGHT * 3.25 + DOWN * 0.05)
        formula = MathTex(
            r"p(C_k|x)=\frac{p(x|C_k)p(C_k)}{\sum_j p(x|C_j)p(C_j)}",
            font_size=32,
        )
        formula.to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(Create(axes), FadeIn(red_dots), FadeIn(blue_dots), run_time=1.7)
        self.play(Create(red_contour), Create(blue_contour), FadeIn(x_star, scale=0.8), Write(x_label), run_time=1.8)
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.12) for item in route], lag_ratio=0.12), run_time=2.4)
        self.play(Write(formula), run_time=1.4)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def log_odds_sigmoid(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Two classes: log odds and sigmoid")
        title = self.scene_title("二クラスのベイズ則は、シグモイドで書ける", font_size=33)
        density_axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.55, 0.2],
            x_length=6.1,
            y_length=3.35,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 3.0 + DOWN * 0.45)
        c1_curve = density_axes.plot(lambda x: gaussian_pdf(x, -1.05, 0.95), x_range=[-4, 4], color=RED_CLASS, stroke_width=4)
        c2_curve = density_axes.plot(lambda x: gaussian_pdf(x, 1.05, 0.95), x_range=[-4, 4], color=BLUE_CLASS, stroke_width=4)
        legend = VGroup(
            Line(LEFT * 0.32, RIGHT * 0.32, color=RED_CLASS, stroke_width=5),
            MathTex(r"p(x|C_1)p(C_1)", font_size=25),
            Line(LEFT * 0.32, RIGHT * 0.32, color=BLUE_CLASS, stroke_width=5),
            MathTex(r"p(x|C_2)p(C_2)", font_size=25),
        ).arrange_in_grid(rows=2, cols=2, col_alignments="lr", buff=(0.2, 0.12))
        legend.next_to(density_axes, UP, buff=0.2).align_to(density_axes, LEFT)

        marker_x = 0.0
        marker = DashedLine(density_axes.c2p(marker_x, 0), density_axes.c2p(marker_x, 0.5), color=WHITE, stroke_width=3)
        equal_note = Text("a = 0 なら p = 0.5", font_size=22, color=POSTERIOR_YELLOW).next_to(marker, UP, buff=0.12)

        formulas = [
            MathTex(r"a=\ln\frac{p(x|C_1)p(C_1)}{p(x|C_2)p(C_2)}", font_size=34),
            MathTex(r"p(C_1|x)=\sigma(a)", font_size=36, color=POSTERIOR_YELLOW),
            MathTex(r"\sigma(a)=\frac{1}{1+\exp(-a)}", font_size=32),
        ]
        panel = self.formula_panel(formulas, width=5.4, height=2.2, color=POSTERIOR_YELLOW)
        panel.shift(RIGHT * 3.05 + UP * 1.15)

        sigmoid_axes = Axes(
            x_range=[-5, 5, 2.5],
            y_range=[0, 1, 0.5],
            x_length=4.6,
            y_length=2.35,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 3.1 + DOWN * 1.55)
        sigmoid_curve = sigmoid_axes.plot(lambda x: float(sigmoid(x)), x_range=[-5, 5], color=POSTERIOR_YELLOW, stroke_width=4)
        mid_dot = Dot(sigmoid_axes.c2p(0, 0.5), color=WHITE, radius=0.07)
        sigmoid_label = Text("logistic sigmoid", font_size=21, color=POSTERIOR_YELLOW).next_to(sigmoid_axes, UP, buff=0.12)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(Create(density_axes), Create(c1_curve), Create(c2_curve), FadeIn(legend), run_time=2.0)
        self.play(Create(marker), Write(equal_note), run_time=1.4)
        self.play(FadeIn(panel), run_time=1.8)
        self.play(Create(sigmoid_axes), Create(sigmoid_curve), FadeIn(mid_dot), Write(sigmoid_label), run_time=1.8)
        self.play(marker.animate.shift(RIGHT * 1.1), equal_note.animate.shift(RIGHT * 1.1), run_time=1.1)
        self.play(marker.animate.shift(LEFT * 2.2), equal_note.animate.shift(LEFT * 2.2), run_time=1.1)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def shared_covariance_linear_boundary(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Gaussian class-conditionals")
        title = self.scene_title("同じ共分散を仮定すると、二次項が消える", font_size=33)
        axes = self.make_class_axes(width=6.7, height=4.9).shift(LEFT * 2.25 + DOWN * 0.15)
        red_dots = self.make_points(axes, self.red_points, RED_CLASS)
        blue_dots = self.make_points(axes, self.blue_points, BLUE_CLASS)
        red_contour = self.gaussian_contour(axes, self.mu1, self.shared_cov, RED_CLASS)
        blue_contour = self.gaussian_contour(axes, self.mu2, self.shared_cov, BLUE_CLASS)
        w, w0 = self.linear_params(self.mu1, self.mu2)
        boundary = self.boundary_line(axes, w, w0, POSTERIOR_YELLOW)
        boundary_label = MathTex(r"w^T x+w_0=0", font_size=30, color=POSTERIOR_YELLOW)
        boundary_label.next_to(boundary, UP, buff=0.15).shift(LEFT * 0.7)

        formulas = [
            MathTex(r"p(x|C_k)=\mathcal{N}(x|\mu_k,\Sigma)", font_size=30),
            MathTex(r"p(C_1|x)=\sigma(w^T x+w_0)", font_size=31, color=POSTERIOR_YELLOW),
            MathTex(r"w=\Sigma^{-1}(\mu_1-\mu_2)", font_size=30),
        ]
        panel = self.formula_panel(formulas, width=5.0, height=2.25, color=GREEN_CLASS)
        panel.shift(RIGHT * 3.65 + UP * 1.05)
        cancel = VGroup(
            MathTex(r"x^T\Sigma^{-1}x", font_size=34, color=RED_CLASS),
            Text("共通なので相殺", font_size=24, color=WHITE),
            MathTex(r"\Rightarrow\ \mathrm{linear}", font_size=34, color=POSTERIOR_YELLOW),
        ).arrange(DOWN, buff=0.15)
        cancel.shift(RIGHT * 3.65 + DOWN * 1.3)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(Create(axes), FadeIn(red_dots), FadeIn(blue_dots), run_time=1.8)
        self.play(Create(red_contour), Create(blue_contour), run_time=1.5)
        self.play(FadeIn(panel), run_time=1.6)
        self.play(Write(cancel[0]), ReplacementTransform(cancel[0].copy(), cancel[1]), run_time=1.5)
        self.play(Create(boundary), Write(boundary_label), Write(cancel[2]), run_time=1.8)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def prior_shift(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Class priors")
        title = self.scene_title("事前確率は、線形境界のバイアスを動かす", font_size=33)
        axes = self.make_class_axes(width=6.6, height=4.75).shift(LEFT * 2.35 + DOWN * 0.1)
        red_dots = self.make_points(axes, self.red_points, RED_CLASS)
        blue_dots = self.make_points(axes, self.blue_points, BLUE_CLASS)
        red_contour = self.gaussian_contour(axes, self.mu1, self.shared_cov, RED_CLASS, scale=1.35)
        blue_contour = self.gaussian_contour(axes, self.mu2, self.shared_cov, BLUE_CLASS, scale=1.35)
        prior_values = [0.25, 0.50, 0.75]
        boundaries = [self.boundary_line(axes, *self.linear_params(self.mu1, self.mu2, p, 1 - p), POSTERIOR_YELLOW) for p in prior_values]
        ghost_lines = VGroup(
            boundaries[0].copy().set_opacity(0.32),
            boundaries[1].copy().set_opacity(0.45),
            boundaries[2].copy().set_opacity(0.32),
        )
        formulas = [
            MathTex(r"w_0=\cdots+\ln\frac{p(C_1)}{p(C_2)}", font_size=33),
            Text("prior は bias だけを変える", font_size=25, color=POSTERIOR_YELLOW),
            Text("境界は平行移動", font_size=25, color=WHITE),
        ]
        panel = self.formula_panel(formulas, width=5.25, height=2.15, color=ACCENT_ORANGE)
        panel.shift(RIGHT * 3.55 + UP * 1.05)
        slider, knob = self.make_slider("p(C1)", ["0.25", "0.50", "0.75"], 1, ACCENT_ORANGE)
        slider_group = VGroup(slider, knob).shift(RIGHT * 3.55 + DOWN * 1.55)
        caption = Text("C1 が出やすいほど、赤の領域が広がる", font_size=22, color=POSTERIOR_YELLOW)
        caption.next_to(slider_group, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(Create(axes), FadeIn(red_dots), FadeIn(blue_dots), Create(red_contour), Create(blue_contour), run_time=2.0)
        self.play(Create(boundaries[1]), FadeIn(panel), FadeIn(slider_group), Write(caption), run_time=1.9)
        self.play(
            ReplacementTransform(boundaries[1], boundaries[0]),
            knob.animate.move_to(slider[0].point_from_proportion(0.0)),
            run_time=1.6,
        )
        self.play(
            ReplacementTransform(boundaries[0], boundaries[2]),
            knob.animate.move_to(slider[0].point_from_proportion(1.0)),
            run_time=1.8,
        )
        self.play(FadeIn(ghost_lines), run_time=1.0)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def posterior_bars(self, values: np.ndarray, labels: list[str], colors: list[ManimColor]) -> VGroup:
        bars = VGroup()
        baseline = Line(LEFT * 1.55, RIGHT * 1.55, color=GREY_B, stroke_width=2)
        for i, (value, label, color) in enumerate(zip(values, labels, colors)):
            bar = Rectangle(width=0.55, height=max(0.08, float(value) * 1.75), stroke_color=color, fill_color=color, fill_opacity=0.75)
            bar.next_to(baseline, UP, buff=0).shift(RIGHT * (i - 1) * 0.95)
            text = Text(label, font_size=18, color=color).next_to(bar, DOWN, buff=0.16)
            pct = Text(f"{value:.2f}", font_size=17, color=WHITE).next_to(bar, UP, buff=0.08)
            bars.add(VGroup(bar, text, pct))
        title = Text("posterior", font_size=21, color=TEXT_GREY).next_to(baseline, DOWN, buff=0.45)
        return VGroup(baseline, bars, title)

    def softmax_multiclass(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("K classes: softmax")
        title = self.scene_title("多クラスの正規化は softmax になる", font_size=34)
        axes = self.make_class_axes(width=6.6, height=4.7).shift(LEFT * 2.55 + DOWN * 0.1)
        red_dots = self.make_points(axes, self.red_points, RED_CLASS, radius=0.052)
        blue_dots = self.make_points(axes, self.blue_points, BLUE_CLASS, radius=0.052)
        green_dots = self.make_points(axes, self.green_points, GREEN_CLASS, radius=0.052)
        contours = VGroup(
            self.gaussian_contour(axes, self.mu1, self.shared_cov, RED_CLASS, scale=1.25),
            self.gaussian_contour(axes, self.mu2, self.shared_cov, BLUE_CLASS, scale=1.25),
            self.gaussian_contour(axes, self.mu3, self.shared_cov, GREEN_CLASS, scale=1.25),
        )
        priors = [1 / 3, 1 / 3, 1 / 3]
        params = [self.class_score_params(mu, prior) for mu, prior in zip([self.mu1, self.mu2, self.mu3], priors)]
        boundaries = VGroup()
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            w = params[i][0] - params[j][0]
            w0 = params[i][1] - params[j][1]
            boundaries.add(self.boundary_line(axes, w, w0, POSTERIOR_YELLOW, stroke_width=3.2).set_opacity(0.85))

        test = np.array([0.0, 0.25])
        scores = np.array([float(w @ test + w0) for w, w0 in params])
        posterior = softmax(scores)
        test_dot = RegularPolygon(n=4, radius=0.12, color=WHITE, fill_opacity=1).rotate(PI / 4).move_to(axes.c2p(*test))
        bars = self.posterior_bars(posterior, ["C1", "C2", "C3"], [RED_CLASS, BLUE_CLASS, GREEN_CLASS])
        bars.shift(RIGHT * 3.65 + DOWN * 1.25)
        formulas = [
            MathTex(r"p(C_k|x)=\frac{\exp(a_k)}{\sum_j\exp(a_j)}", font_size=31),
            MathTex(r"a_k=w_k^T x+w_{k0}", font_size=34, color=POSTERIOR_YELLOW),
        ]
        panel = self.formula_panel(formulas, width=5.15, height=1.65, color=ACCENT_PURPLE)
        panel.shift(RIGHT * 3.55 + UP * 1.35)
        note = Text("共有 Σ なら各 a_k は線形", font_size=24, color=WHITE).next_to(panel, DOWN, buff=0.25)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(Create(axes), FadeIn(red_dots), FadeIn(blue_dots), FadeIn(green_dots), run_time=2.1)
        self.play(Create(contours), run_time=1.4)
        self.play(FadeIn(panel), Write(note), run_time=1.4)
        self.play(Create(boundaries), FadeIn(test_dot, scale=0.8), run_time=1.8)
        self.play(FadeIn(bars), run_time=1.5)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def maximum_likelihood_solution(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Maximum likelihood")
        title = self.scene_title("生成モデルの部品は、ラベル付きデータから推定する", font_size=32)
        axes = self.make_class_axes(width=6.1, height=4.45).shift(LEFT * 3.0 + DOWN * 0.12)
        red_dots = self.make_points(axes, self.red_points, RED_CLASS)
        blue_dots = self.make_points(axes, self.blue_points, BLUE_CLASS)
        mu1_hat = self.red_points.mean(axis=0)
        mu2_hat = self.blue_points.mean(axis=0)
        mu1_dot = Dot(axes.c2p(*mu1_hat), radius=0.12, color=RED_CLASS)
        mu2_dot = Dot(axes.c2p(*mu2_hat), radius=0.12, color=BLUE_CLASS)
        mu_labels = VGroup(
            MathTex(r"\mu_1", font_size=30, color=RED_CLASS).next_to(mu1_dot, UP, buff=0.1),
            MathTex(r"\mu_2", font_size=30, color=BLUE_CLASS).next_to(mu2_dot, DOWN, buff=0.1),
        )
        contour1 = self.gaussian_contour(axes, mu1_hat, np.cov(self.red_points.T), RED_CLASS, scale=1.35)
        contour2 = self.gaussian_contour(axes, mu2_hat, np.cov(self.blue_points.T), BLUE_CLASS, scale=1.35)
        shared = self.gaussian_contour(axes, (mu1_hat + mu2_hat) / 2, self.shared_cov, POSTERIOR_YELLOW, scale=1.55).set_opacity(0.65)

        n1 = len(self.red_points)
        n2 = len(self.blue_points)
        formulas = [
            MathTex(r"\pi=\frac{N_1}{N_1+N_2}", font_size=34, color=ACCENT_ORANGE),
            MathTex(r"\mu_1=\frac{1}{N_1}\sum_{n\in C_1}x_n,\quad \mu_2=\frac{1}{N_2}\sum_{n\in C_2}x_n", font_size=27),
            MathTex(r"\Sigma=\frac{N_1}{N}S_1+\frac{N_2}{N}S_2", font_size=32, color=POSTERIOR_YELLOW),
        ]
        panel = self.formula_panel(formulas, width=6.0, height=2.45, color=GREEN_CLASS)
        panel.shift(RIGHT * 2.95 + UP * 0.85)
        counts = VGroup(
            Text(f"N1 = {n1}", font_size=24, color=RED_CLASS),
            Text(f"N2 = {n2}", font_size=24, color=BLUE_CLASS),
            Text(f"pi = {n1 / (n1 + n2):.2f}", font_size=24, color=ACCENT_ORANGE),
            Text("外れ値には敏感", font_size=24, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        counts.shift(RIGHT * 3.0 + DOWN * 1.45)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(Create(axes), FadeIn(red_dots), FadeIn(blue_dots), run_time=1.8)
        self.play(FadeIn(mu1_dot, scale=0.8), FadeIn(mu2_dot, scale=0.8), Write(mu_labels), run_time=1.5)
        self.play(Create(contour1), Create(contour2), run_time=1.4)
        self.play(FadeIn(panel), run_time=1.6)
        self.play(Create(shared), LaggedStart(*[Write(item) for item in counts], lag_ratio=0.2), run_time=2.2)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def discrete_naive_bayes(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Discrete features")
        title = self.scene_title("条件付き独立を置くと、離散特徴でも線形になる", font_size=32)
        full_title = Text("一般の表", font_size=26, color=YELLOW)
        full_grid = VGroup(*[Square(side_length=0.28, fill_color=YELLOW, fill_opacity=0.45, stroke_color=GREY_B, stroke_width=1) for _ in range(32)])
        full_grid.arrange_in_grid(rows=4, cols=8, buff=0.04)
        full_count = MathTex(r"2^D-1", font_size=36, color=YELLOW).next_to(full_grid, DOWN, buff=0.2)
        full_group = VGroup(full_title, full_grid, full_count).arrange(DOWN, buff=0.18)
        full_group.shift(LEFT * 3.9 + UP * 0.65)

        class_node = Circle(radius=0.38, color=ACCENT_PURPLE, fill_opacity=0.18)
        class_text = MathTex(r"C_k", font_size=31, color=ACCENT_PURPLE).move_to(class_node)
        features = VGroup()
        arrows = VGroup()
        for i in range(5):
            square = Square(side_length=0.58, color=BLUE_CLASS, fill_opacity=0.16)
            label_i = MathTex(fr"x_{i + 1}", font_size=28, color=BLUE_CLASS).move_to(square)
            node = VGroup(square, label_i)
            node.shift(RIGHT * (i - 2) * 0.8 + DOWN * 1.0)
            features.add(node)
            arrows.add(Arrow(class_node.get_bottom(), node[0].get_top(), color=TEXT_GREY, buff=0.05, stroke_width=2.2))
        bayes_group = VGroup(class_node, class_text, arrows, features)
        bayes_group.shift(RIGHT * 1.15 + UP * 0.9)
        bayes_title = Text("naive Bayes", font_size=26, color=BLUE_CLASS).next_to(class_node, UP, buff=0.25)
        param_count = MathTex(r"D\ \mathrm{parameters/class}", font_size=31, color=BLUE_CLASS).next_to(features, DOWN, buff=0.28)

        formulas = [
            MathTex(r"p(x|C_k)=\prod_i \mu_{ki}^{x_i}(1-\mu_{ki})^{1-x_i}", font_size=30),
            MathTex(r"a_k(x)=\sum_i\{x_i\ln\mu_{ki}+(1-x_i)\ln(1-\mu_{ki})\}+\ln p(C_k)", font_size=24),
            Text("log を取ると x_i の線形関数", font_size=24, color=POSTERIOR_YELLOW),
        ]
        panel = self.formula_panel(formulas, width=11.5, height=2.05, color=POSTERIOR_YELLOW)
        panel.to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(full_group, shift=UP * 0.15), run_time=1.5)
        self.play(FadeIn(bayes_group), Write(bayes_title), Write(param_count), run_time=2.1)
        self.play(TransformFromCopy(full_grid, features), run_time=1.1)
        self.play(FadeIn(panel), run_time=1.9)
        self.play(Indicate(formulas[2], color=POSTERIOR_YELLOW), run_time=1.1)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def exponential_family_summary(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("Summary and bridge")
        title = self.scene_title("生成モデルの仮定が、識別モデルの形を決める", font_size=32)
        columns = VGroup(
            self.named_box("Gaussian", "共有 Σ なら線形", RED_CLASS, width=3.25, height=1.05),
            self.named_box("Naive Bayes", "条件付き独立で線形", BLUE_CLASS, width=3.25, height=1.05),
            self.named_box("Exponential family", "共有 scale で線形", GREEN_CLASS, width=3.25, height=1.05),
        ).arrange(RIGHT, buff=0.35)
        columns.shift(UP * 1.65)
        arrow = Arrow(UP * 0.9, DOWN * 0.2, color=TEXT_GREY, buff=0)
        arrow.next_to(columns, DOWN, buff=0.22)
        core = VGroup(
            MathTex(r"a_k(x)=w_k^T x+w_{k0}", font_size=42, color=POSTERIOR_YELLOW),
            MathTex(r"p(C_k|x)=\mathrm{softmax}(a_k)", font_size=38),
            Text("K=2 なら logistic sigmoid", font_size=25, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.18)
        core.next_to(arrow, DOWN, buff=0.2)
        core.shift(UP * 0.35)

        warnings = VGroup(
            Text("別々の共分散: quadratic boundary", font_size=22, color=YELLOW),
            Text("密度仮定が外れると、事後確率もずれる", font_size=22, color=YELLOW),
            Text("次: p(C|x) を直接学習する", font_size=24, color=ACCENT_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        warnings.to_edge(DOWN).shift(UP * 0.05 + LEFT * 0.1)

        mini_axes = self.make_class_axes(width=3.1, height=2.15).shift(RIGHT * 4.35 + DOWN * 0.12)
        line = Line(mini_axes.c2p(-2.4, -1.7), mini_axes.c2p(2.4, 1.7), color=POSTERIOR_YELLOW, stroke_width=3)
        curve = VMobject(color=YELLOW)
        curve.set_points_smoothly([mini_axes.c2p(x, 0.28 * (x**2) - 1.1) for x in np.linspace(-2.4, 2.4, 80)])
        curve.set_stroke(width=3)
        mini_label = VGroup(
            Text("linear", font_size=18, color=POSTERIOR_YELLOW).next_to(line, UP, buff=0.08),
            Text("quadratic", font_size=18, color=YELLOW).next_to(curve, DOWN, buff=0.05),
        )

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.16) for item in columns], lag_ratio=0.15), run_time=1.7)
        self.play(GrowArrow(arrow), FadeIn(core), run_time=1.9)
        self.play(Create(mini_axes), Create(line), Create(curve), FadeIn(mini_label), run_time=1.5)
        self.play(LaggedStart(*[Write(item) for item in warnings], lag_ratio=0.2), run_time=2.2)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
