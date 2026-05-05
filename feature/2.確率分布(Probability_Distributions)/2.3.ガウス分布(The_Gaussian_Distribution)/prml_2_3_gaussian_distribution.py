from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
GREEN_CURVE = GREEN_C
RED_MODEL = RED_C
ORANGE_ALT = ORANGE
YELLOW_NOTE = YELLOW_C
PURPLE_BAYES = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def normal_pdf(x: np.ndarray | float, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray | float:
    x_array = np.asarray(x)
    return np.exp(-0.5 * ((x_array - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def student_t_shape(x: np.ndarray | float, nu: float = 2.0) -> np.ndarray | float:
    x_array = np.asarray(x)
    coefficient = math.gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * math.gamma(nu / 2))
    return coefficient * (1 + x_array**2 / nu) ** (-(nu + 1) / 2)


class PRML23GaussianDistribution(Scene):
    """PRML 2.3 overview for a high-school math audience.

    Render example:
        uv run manim -pql prml_2_3_gaussian_distribution.py PRML23GaussianDistribution
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.gaussian_shape()
        self.central_limit_theorem()
        self.multivariate_geometry()
        self.covariance_restrictions()
        self.conditional_and_marginal()
        self.maximum_likelihood()
        self.bayesian_mean()
        self.robust_and_mixture_bridge()

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

    def density_axes(self, width: float = 7.0, height: float = 3.5, y_max: float = 0.55) -> Axes:
        return Axes(
            x_range=[-4, 4, 1],
            y_range=[0, y_max, 0.1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def plane_axes(self, width: float = 4.3, height: float = 3.6) -> Axes:
        return Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_histogram(self, axes: Axes, samples: np.ndarray, bins: np.ndarray, color: ManimColor) -> VGroup:
        counts, edges = np.histogram(samples, bins=bins, density=True)
        bars = VGroup()
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            width = axes.c2p(float(right), 0)[0] - axes.c2p(float(left), 0)[0]
            height = axes.c2p(0, float(count))[1] - axes.c2p(0, 0)[1]
            rect = Rectangle(width=width * 0.92, height=max(height, 0.001), stroke_width=0)
            rect.set_fill(color, opacity=0.72)
            rect.move_to(axes.c2p(float((left + right) / 2), 0) + UP * height / 2)
            bars.add(rect)
        return bars

    def gaussian_shape(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 2.3 / Eq. (2.42)")
        title = self.scene_title("ガウス分布: 中心と広がりで連続値を表す", font_size=32)
        axes = self.density_axes(width=7.2, height=3.4).shift(DOWN * 0.1)
        curve_mu0 = axes.plot(lambda x: normal_pdf(x, 0.0, 0.8), x_range=[-4, 4], color=GREEN_CURVE)
        curve_mu1 = axes.plot(lambda x: normal_pdf(x, 1.0, 0.8), x_range=[-4, 4], color=GREEN_CURVE)
        curve_wide = axes.plot(lambda x: normal_pdf(x, 1.0, 1.35), x_range=[-4, 4], color=ORANGE_ALT)
        for curve in [curve_mu0, curve_mu1, curve_wide]:
            curve.set_stroke(width=4)

        formula = MathTex(
            r"\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{1/2}}"
            r"\exp\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}",
            font_size=33,
        ).to_edge(DOWN, buff=0.45)
        mu_note = MathTex(r"\mu:\ center", font_size=31, color=GREEN_CURVE).next_to(axes, RIGHT, buff=0.35).shift(UP * 0.7)
        sigma_note = MathTex(r"\sigma^2:\ spread", font_size=31, color=ORANGE_ALT).next_to(mu_note, DOWN, buff=0.25)
        center_line = DashedLine(axes.c2p(0, 0), axes.c2p(0, 0.52), color=GREEN_CURVE, dash_length=0.1)
        center_line2 = DashedLine(axes.c2p(1, 0), axes.c2p(1, 0.52), color=GREEN_CURVE, dash_length=0.1)

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(Create(curve_mu0), FadeIn(center_line), Write(formula))
        self.play(Transform(curve_mu0, curve_mu1), Transform(center_line, center_line2), FadeIn(mu_note), run_time=1.4)
        self.play(Transform(curve_mu0, curve_wide), FadeIn(sigma_note), run_time=1.4)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(curve_mu0), FadeOut(center_line), FadeOut(formula), FadeOut(mu_note), FadeOut(sigma_note))

    def central_limit_theorem(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("PRML 2.3 / Fig. 2.6")
        title = self.scene_title("平均を取るほど、ベル型へ近づく", font_size=34)
        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 4.2, 1],
            x_length=7.0,
            y_length=3.55,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.1)
        rng = np.random.default_rng(23)
        bins = np.linspace(0, 1, 21)
        samples = {
            1: rng.uniform(0, 1, size=(9000, 1)).mean(axis=1),
            2: rng.uniform(0, 1, size=(9000, 2)).mean(axis=1),
            10: rng.uniform(0, 1, size=(9000, 10)).mean(axis=1),
        }
        bars = {n: self.make_histogram(axes, values, bins, BLUE_DATA) for n, values in samples.items()}
        gaussian_curve = axes.plot(lambda x: normal_pdf(x, 0.5, math.sqrt(1 / 120)), x_range=[0.08, 0.92], color=GREEN_CURVE)
        gaussian_curve.set_stroke(width=4)
        n_text = MathTex(r"N=1", font_size=38, color=YELLOW_NOTE).next_to(axes, RIGHT, buff=0.35).shift(UP * 0.75)
        equation = MathTex(r"\bar{x}=\frac{x_1+\cdots+x_N}{N}", font_size=38).to_edge(DOWN, buff=0.55)
        note = Text("小さな偶然の平均", font_size=25, color=TEXT_GREY).next_to(equation, UP, buff=0.15)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(bars[1]), Write(n_text), Write(equation))
        self.play(Write(note))
        for n in [2, 10]:
            next_text = MathTex(f"N={n}", font_size=38, color=YELLOW_NOTE).move_to(n_text)
            self.play(Transform(bars[1], bars[n]), Transform(n_text, next_text), run_time=1.2)
        self.play(Create(gaussian_curve), run_time=0.9)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(bars[1]), FadeOut(n_text), FadeOut(equation), FadeOut(note), FadeOut(gaussian_curve))

    def multivariate_geometry(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("PRML 2.3 / Fig. 2.7 / Eq. (2.44)-(2.52)")
        title = self.scene_title("多変量ガウス: 共分散が楕円の向きと幅を決める", font_size=31)
        axes = self.plane_axes(width=5.0, height=3.9).shift(LEFT * 2.1 + DOWN * 0.15)
        rng = np.random.default_rng(8)
        cov = np.array([[1.45, 0.85], [0.85, 0.8]])
        points = rng.multivariate_normal([0.15, -0.05], cov, size=120)
        dots = VGroup(*[Dot(axes.c2p(float(x), float(y)), radius=0.025, color=BLUE_DATA) for x, y in points])
        ellipse = Ellipse(width=3.2, height=1.25, color=GREEN_CURVE, stroke_width=4).rotate(0.58).move_to(axes.c2p(0.15, -0.05))
        center = Dot(axes.c2p(0.15, -0.05), radius=0.07, color=YELLOW_NOTE)
        u1 = Arrow(axes.c2p(0.15, -0.05), axes.c2p(1.45, 0.78), buff=0, color=GREEN_CURVE)
        u2 = Arrow(axes.c2p(0.15, -0.05), axes.c2p(-0.38, 0.92), buff=0, color=ORANGE_ALT)
        u1_label = MathTex(r"u_1,\lambda_1", font_size=28, color=GREEN_CURVE).next_to(u1.get_end(), RIGHT, buff=0.1)
        u2_label = MathTex(r"u_2,\lambda_2", font_size=28, color=ORANGE_ALT).next_to(u2.get_end(), UP, buff=0.1)
        formula = MathTex(r"\Delta^2=(x-\mu)^T\Sigma^{-1}(x-\mu)", font_size=36, color=WHITE)
        formula.to_edge(DOWN, buff=0.5)
        matrix = MathTex(
            r"\mu=\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix}",
            r"\quad",
            r"\Sigma=\begin{bmatrix}\sigma_1^2&\sigma_{12}\\\sigma_{21}&\sigma_2^2\end{bmatrix}",
            font_size=34,
        ).next_to(axes, RIGHT, buff=0.45).shift(UP * 0.65)
        note = Text("同じ密度 = 楕円上", font_size=27, color=GREEN_CURVE).next_to(matrix, DOWN, buff=0.35)

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(FadeIn(dots, lag_ratio=0.01), FadeIn(center), run_time=1.0)
        self.play(Create(ellipse), Write(matrix), Write(note))
        self.play(GrowArrow(u1), GrowArrow(u2), FadeIn(u1_label), FadeIn(u2_label))
        self.play(Write(formula))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(ellipse), FadeOut(center), FadeOut(u1), FadeOut(u2), FadeOut(u1_label), FadeOut(u2_label), FadeOut(matrix), FadeOut(note), FadeOut(formula))

    def covariance_restrictions(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("PRML 2.3 / Fig. 2.8")
        title = self.scene_title("共分散行列を制限すると、表現力と計算量が変わる", font_size=31)
        panels = VGroup()
        configs = [
            ("general", r"D(D+3)/2", 0.55, 3.0, 1.25, GREEN_CURVE),
            ("diagonal", r"2D", 0.0, 2.35, 1.15, ORANGE_ALT),
            ("isotropic", r"D+1", 0.0, 1.65, 1.65, BLUE_DATA),
        ]
        for name, params, angle, width, height, color in configs:
            box = RoundedRectangle(width=3.45, height=3.35, corner_radius=0.1, color=color)
            head = Text(name, font_size=27, color=color).move_to(box.get_top() + DOWN * 0.35)
            ellipse = Ellipse(width=width, height=height, color=color, stroke_width=4).rotate(angle).move_to(box.get_center() + DOWN * 0.1)
            par = MathTex(params, font_size=31, color=YELLOW_NOTE).move_to(box.get_bottom() + UP * 0.35)
            panels.add(VGroup(box, head, ellipse, par))
        panels.arrange(RIGHT, buff=0.35).shift(DOWN * 0.05)
        captions = Text("相関を表せるほど、必要なパラメータも増える", font_size=28, color=YELLOW_NOTE)
        captions.to_edge(DOWN, buff=0.55)
        limitation = Text("単峰: 山は一つだけ", font_size=25, color=RED_MODEL).next_to(captions, UP, buff=0.2)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(panels[0]), run_time=0.8)
        self.play(FadeIn(panels[1]), run_time=0.8)
        self.play(FadeIn(panels[2]), run_time=0.8)
        self.play(Write(captions), Write(limitation))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(panels), FadeOut(captions), FadeOut(limitation))

    def conditional_and_marginal(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("PRML 2.3.1-2.3.3")
        title = self.scene_title("条件付けても、周辺化しても、ガウスのまま", font_size=32)
        axes = self.plane_axes(width=4.5, height=3.75).shift(LEFT * 2.3 + DOWN * 0.05)
        ellipse = Ellipse(width=3.3, height=1.25, color=GREEN_CURVE, stroke_width=4).rotate(0.52).move_to(axes.c2p(0.0, 0.0))
        slice_line = DashedLine(axes.c2p(0.9, -2.1), axes.c2p(0.9, 2.1), color=YELLOW_NOTE, dash_length=0.1)
        slice_label = MathTex(r"x_b=fixed", font_size=28, color=YELLOW_NOTE).next_to(slice_line, UP, buff=0.1)
        cond_axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[0, 0.6, 0.2],
            x_length=3.7,
            y_length=1.55,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 2.7 + UP * 0.75)
        cond_curve = cond_axes.plot(lambda x: normal_pdf(x, 0.65, 0.65), x_range=[-2.5, 2.5], color=YELLOW_NOTE)
        marginal_axes = cond_axes.copy().shift(DOWN * 2.15)
        marginal_curve = marginal_axes.plot(lambda x: normal_pdf(x, 0.0, 1.1), x_range=[-2.5, 2.5], color=BLUE_DATA)
        cond_title = MathTex(r"p(x_a|x_b)=Gaussian", font_size=30, color=YELLOW_NOTE).next_to(cond_axes, UP, buff=0.13)
        marg_title = MathTex(r"p(x_a)=\int p(x_a,x_b)\,dx_b", font_size=30, color=BLUE_DATA).next_to(marginal_axes, UP, buff=0.13)
        note = Text("平均は固定した値に合わせて動く", font_size=26, color=TEXT_GREY).to_edge(DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title), Create(axes), Create(ellipse))
        self.play(Create(slice_line), FadeIn(slice_label))
        self.play(Create(cond_axes), Create(cond_curve), Write(cond_title))
        self.play(Create(marginal_axes), Create(marginal_curve), Write(marg_title), Write(note))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(ellipse), FadeOut(slice_line), FadeOut(slice_label), FadeOut(cond_axes), FadeOut(cond_curve), FadeOut(cond_title), FadeOut(marginal_axes), FadeOut(marginal_curve), FadeOut(marg_title), FadeOut(note))

    def maximum_likelihood(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("PRML 2.3.4 / Eq. (2.118)-(2.124)")
        title = self.scene_title("最尤推定: データに一番合う平均と共分散", font_size=32)
        axes = self.density_axes(width=6.4, height=3.2, y_max=0.72).shift(LEFT * 1.35 + DOWN * 0.1)
        rng = np.random.default_rng(11)
        data = rng.normal(0.55, 0.75, size=16)
        dots = VGroup(*[Dot(axes.c2p(float(x), 0.02), radius=0.055, color=BLUE_DATA) for x in data])
        sample_mean = float(data.mean())
        sample_std = float(data.std())
        curve = axes.plot(lambda x: normal_pdf(x, sample_mean, sample_std), x_range=[-4, 4], color=GREEN_CURVE)
        curve.set_stroke(width=4)
        mean_line = DashedLine(axes.c2p(sample_mean, 0), axes.c2p(sample_mean, 0.68), color=YELLOW_NOTE, dash_length=0.09)
        mu_formula = MathTex(r"\mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}x_n", font_size=34, color=YELLOW_NOTE)
        sigma_formula = MathTex(r"\Sigma_{ML}=\frac{1}{N}\sum_n(x_n-\mu_{ML})(x_n-\mu_{ML})^T", font_size=24, color=GREEN_CURVE)
        formulas = VGroup(mu_formula, sigma_formula).arrange(DOWN, buff=0.28, aligned_edge=LEFT).next_to(axes, RIGHT, buff=0.35)
        warning = Text("有限データでは共分散が小さめ", font_size=24, color=RED_MODEL).to_edge(DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(FadeIn(dots, lag_ratio=0.08), run_time=1.0)
        self.play(Create(mean_line), Write(mu_formula))
        self.play(Create(curve), Write(sigma_formula))
        self.play(Write(warning))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(curve), FadeOut(mean_line), FadeOut(formulas), FadeOut(warning))

    def bayesian_mean(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("PRML 2.3.6 / Fig. 2.12")
        title = self.scene_title("ベイズ推定: 平均そのものの不確かさを更新する", font_size=31)
        axes = self.density_axes(width=7.0, height=3.35, y_max=2.5).shift(DOWN * 0.05)
        posterior_specs = [
            (0, 0.0, 0.85, BLUE_DATA),
            (1, 0.38, 0.55, ORANGE_ALT),
            (2, 0.55, 0.38, YELLOW_NOTE),
            (10, 0.75, 0.18, GREEN_CURVE),
        ]
        curves = []
        for _, mu, sigma, color in posterior_specs:
            curve = axes.plot(lambda x, m=mu, s=sigma: normal_pdf(x, m, s), x_range=[-2.2, 2.2], color=color)
            curve.set_stroke(width=4)
            curves.append(curve)
        current = curves[0]
        n_text = MathTex(r"N=0", font_size=37, color=BLUE_DATA).next_to(axes, RIGHT, buff=0.35).shift(UP * 0.65)
        formula = MathTex(r"p(\mu|X)\propto p(X|\mu)p(\mu)", font_size=38, color=PURPLE_BAYES).to_edge(DOWN, buff=0.55)
        data_marks = VGroup(
            *[Triangle(color=WHITE, fill_opacity=1).scale(0.08).rotate(PI).move_to(axes.c2p(x, 0.02)) for x in [0.4, 0.7, 0.85, 0.95]]
        )
        note = Text("事後分布が狭くなる = 推定の不確かさが減る", font_size=25, color=TEXT_GREY).next_to(formula, UP, buff=0.18)

        self.play(FadeIn(label), Write(title), Create(axes), Create(current), Write(n_text), Write(formula))
        self.play(FadeIn(data_marks, lag_ratio=0.18), Write(note), run_time=1.0)
        for (n, _, _, color), curve in zip(posterior_specs[1:], curves[1:]):
            next_text = MathTex(f"N={n}", font_size=37, color=color).move_to(n_text)
            self.play(Transform(current, curve), Transform(n_text, next_text), run_time=1.05)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(current), FadeOut(n_text), FadeOut(formula), FadeOut(data_marks), FadeOut(note))

    def robust_and_mixture_bridge(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("PRML 2.3.7 / 2.3.9")
        title = self.scene_title("ガウスの限界から、t分布と混合ガウスへ", font_size=33)
        left_axes = self.density_axes(width=4.6, height=2.7, y_max=0.55).shift(LEFT * 3.0 + UP * 0.15)
        right_axes = self.density_axes(width=4.6, height=2.7, y_max=0.55).shift(RIGHT * 3.0 + UP * 0.15)
        gaussian = left_axes.plot(lambda x: normal_pdf(x, 0, 1), x_range=[-4, 4], color=GREEN_CURVE)
        t_curve = left_axes.plot(lambda x: student_t_shape(x, 2), x_range=[-4, 4], color=ORANGE_ALT)
        mixture = right_axes.plot(
            lambda x: 0.48 * normal_pdf(x, -1.25, 0.55) + 0.52 * normal_pdf(x, 1.15, 0.65),
            x_range=[-4, 4],
            color=PURPLE_BAYES,
        )
        for curve in [gaussian, t_curve, mixture]:
            curve.set_stroke(width=4)
        outliers = VGroup(*[Dot(left_axes.c2p(x, 0.03), radius=0.055, color=RED_MODEL) for x in [2.8, 3.2, 3.55]])
        left_label = Text("外れ値には厚い裾", font_size=25, color=ORANGE_ALT).next_to(left_axes, DOWN, buff=0.18)
        right_label = Text("複数の山には混合", font_size=25, color=PURPLE_BAYES).next_to(right_axes, DOWN, buff=0.18)
        legend = VGroup(
            MathTex(r"Gaussian", font_size=27, color=GREEN_CURVE),
            MathTex(r"Student\ t", font_size=27, color=ORANGE_ALT),
            MathTex(r"Mixture\ of\ Gaussians", font_size=27, color=PURPLE_BAYES),
        ).arrange(RIGHT, buff=0.45).to_edge(DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title), Create(left_axes), Create(right_axes))
        self.play(Create(gaussian), Create(t_curve), FadeIn(outliers), Write(left_label), FadeIn(legend[0]), FadeIn(legend[1]))
        self.play(Create(mixture), Write(right_label), FadeIn(legend[2]))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(left_axes), FadeOut(right_axes), FadeOut(gaussian), FadeOut(t_curve), FadeOut(mixture), FadeOut(outliers), FadeOut(left_label), FadeOut(right_label), FadeOut(legend))
