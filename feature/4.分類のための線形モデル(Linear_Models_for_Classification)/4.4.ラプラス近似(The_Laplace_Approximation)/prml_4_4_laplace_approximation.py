from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
GREEN_TRUE = GREEN_C
RED_LAPLACE = RED_C
ORANGE_CURVE = ORANGE
YELLOW_NOTE = YELLOW_C
PURPLE_EVIDENCE = PURPLE_C
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


def log_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return -np.logaddexp(0.0, -np.asarray(x))


def log_unnormalized(z: np.ndarray | float) -> np.ndarray | float:
    z_array = np.asarray(z)
    return -0.5 * z_array**2 + log_sigmoid(20.0 * z_array + 4.0)


def unnormalized_density(z: np.ndarray | float) -> np.ndarray | float:
    return np.exp(log_unnormalized(z))


GRID = np.linspace(-2.0, 4.0, 1400)
NORMALIZER = float(np.trapezoid(unnormalized_density(GRID), GRID))
MODE_Z = float(GRID[np.argmax(unnormalized_density(GRID))])
MODE_SIGMOID = float(sigmoid(20.0 * MODE_Z + 4.0))
CURVATURE_A = 1.0 + 400.0 * MODE_SIGMOID * (1.0 - MODE_SIGMOID)
MODE_LOG_VALUE = float(log_unnormalized(MODE_Z))


def target_pdf(z: np.ndarray | float) -> np.ndarray | float:
    return unnormalized_density(z) / NORMALIZER


def laplace_pdf(z: np.ndarray | float, scale: float = 1.0) -> np.ndarray | float:
    precision = CURVATURE_A * scale
    z_array = np.asarray(z)
    return np.sqrt(precision / (2.0 * np.pi)) * np.exp(-0.5 * precision * (z_array - MODE_Z) ** 2)


def negative_log_density(z: np.ndarray | float) -> np.ndarray | float:
    return -(log_unnormalized(z) - MODE_LOG_VALUE)


def quadratic_bowl(z: np.ndarray | float, scale: float = 1.0) -> np.ndarray | float:
    z_array = np.asarray(z)
    return 0.5 * CURVATURE_A * scale * (z_array - MODE_Z) ** 2


class PRML44LaplaceApproximation(Scene):
    """PRML 4.4 Laplace approximation overview.

    Render example:
        uv run manim -pql prml_4_4_laplace_approximation.py PRML44LaplaceApproximation
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.why_laplace()
        self.one_dimensional_laplace()
        self.curvature_as_uncertainty()
        self.multivariate_extension()
        self.evidence_and_occam()
        self.bic_bridge()
        self.strengths_and_limits()

    def start_narration(self, scene_id: str) -> tuple[float, float | None]:
        audio_path = VOICEOVER_DIR / f"{scene_id}.wav"
        start_time = float(getattr(self, "time", 0.0))
        if not audio_path.exists():
            return start_time, None
        self.add_sound(str(audio_path))
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        return start_time, duration

    def finish_narration(self, narration: tuple[float, float | None], pad: float = 0.25) -> None:
        start_time, duration = narration
        if duration is None:
            return
        elapsed = float(getattr(self, "time", 0.0)) - start_time
        remaining = duration - elapsed + pad
        if remaining > 0:
            self.wait(remaining)

    def clear_scene(self, *mobjects: Mobject) -> None:
        group = VGroup(*[m for m in mobjects if m is not None])
        if len(group) > 0:
            self.play(FadeOut(group), run_time=0.8)
        self.clear()

    def section_label(self, text: str) -> Text:
        label = Text(text, font_size=18, color=TEXT_GREY)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 34) -> Text:
        title = Text(text, font_size=font_size)
        title.to_edge(UP).shift(DOWN * 0.35)
        return title

    def density_axes(self, width: float = 7.4, height: float = 3.7, y_max: float = 0.9) -> Axes:
        return Axes(
            x_range=[-2, 4, 1],
            y_range=[0, y_max, 0.2],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def negative_log_axes(self, width: float = 7.2, height: float = 3.7) -> Axes:
        return Axes(
            x_range=[-2, 4, 1],
            y_range=[0, 7, 1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_curve_from_values(
        self,
        axes: Axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        color: ManimColor,
        width: float = 4.0,
        opacity: float = 1.0,
    ) -> VMobject:
        points = [axes.c2p(float(x), float(y)) for x, y in zip(x_values, y_values)]
        curve = VMobject(color=color)
        curve.set_points_smoothly(points)
        curve.set_stroke(width=width, opacity=opacity)
        return curve

    def formula_box(self, tex: str, color: ManimColor = WHITE, font_size: int = 34) -> VGroup:
        formula = MathTex(tex, font_size=font_size, color=color)
        box = SurroundingRectangle(formula, color=color, buff=0.18)
        box.set_stroke(width=1.5, opacity=0.7)
        return VGroup(box, formula)

    def why_laplace(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 4.4 / bridge to Bayesian logistic regression")
        title = self.scene_title("難しい事後分布を、モード周りのガウスで読む", font_size=32)

        axes = self.density_axes(width=7.0, height=3.55, y_max=0.86).shift(LEFT * 1.55 + DOWN * 0.15)
        true_curve = axes.plot(lambda z: target_pdf(z), x_range=[-2, 4], color=GREEN_TRUE)
        approx_curve = axes.plot(lambda z: laplace_pdf(z), x_range=[-2, 4], color=RED_LAPLACE)
        true_curve.set_stroke(width=4.5)
        approx_curve.set_stroke(width=4.0)
        mode_line = DashedLine(axes.c2p(MODE_Z, 0), axes.c2p(MODE_Z, 0.8), color=YELLOW_NOTE, dash_length=0.1)
        mode_dot = Dot(axes.c2p(MODE_Z, float(target_pdf(MODE_Z))), color=YELLOW_NOTE, radius=0.07)

        posterior = self.formula_box(r"p(w|D)\propto p(D|w)p(w)", color=WHITE, font_size=33)
        posterior.to_corner(UR).shift(DOWN * 0.8 + LEFT * 0.15)
        not_gaussian = Text("非ガウス: 解析積分が難しい", font_size=24, color=ORANGE_CURVE)
        not_gaussian.next_to(posterior, DOWN, buff=0.35)
        zoom = Arrow(LEFT * 0.45, RIGHT * 0.45, color=YELLOW_NOTE, stroke_width=5)
        local = Text("mode 近傍を拡大", font_size=22, color=YELLOW_NOTE)
        local_group = VGroup(zoom, local).arrange(RIGHT, buff=0.18)
        local_group.next_to(not_gaussian, DOWN, buff=0.35).align_to(not_gaussian, LEFT)
        gaussian = self.formula_box(r"q(w)=\mathcal{N}(w|w_0,\Sigma)", color=RED_LAPLACE, font_size=33)
        gaussian.next_to(local_group, DOWN, buff=0.35).align_to(posterior, LEFT)

        legend = VGroup(
            Line(LEFT * 0.35, RIGHT * 0.35, color=GREEN_TRUE, stroke_width=5),
            Text("true posterior shape", font_size=19),
            Line(LEFT * 0.35, RIGHT * 0.35, color=RED_LAPLACE, stroke_width=5),
            Text("Laplace Gaussian", font_size=19),
        ).arrange_in_grid(rows=2, cols=2, col_alignments="lr", buff=(0.2, 0.12))
        legend.next_to(axes, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(Create(axes), Create(true_curve), FadeIn(legend[0:2]), run_time=1.8)
        self.play(Write(posterior), Write(not_gaussian), run_time=1.4)
        self.play(Create(mode_line), FadeIn(mode_dot), FadeIn(local_group), run_time=1.2)
        self.play(Create(approx_curve), FadeIn(legend[2:4]), Write(gaussian), run_time=1.8)
        self.play(Indicate(mode_dot, color=YELLOW_NOTE), Indicate(approx_curve, color=RED_LAPLACE), run_time=1.1)
        self.finish_narration(narration)
        self.clear_scene(label, title, axes, true_curve, approx_curve, mode_line, mode_dot, posterior, not_gaussian, local_group, gaussian, legend)

    def one_dimensional_laplace(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("PRML 4.4 / Eq. (4.125)-(4.130)")
        title = self.scene_title("1変数: mode を探し、log f を二次で近似する", font_size=32)

        axes = self.density_axes(width=6.6, height=3.4, y_max=0.85).shift(LEFT * 2.1 + DOWN * 0.15)
        true_curve = axes.plot(lambda z: target_pdf(z), x_range=[-2, 4], color=GREEN_TRUE).set_stroke(width=4.2)
        approx_curve = axes.plot(lambda z: laplace_pdf(z), x_range=[-2, 4], color=RED_LAPLACE).set_stroke(width=4.0)
        mode_line = DashedLine(axes.c2p(MODE_Z, 0), axes.c2p(MODE_Z, 0.78), color=YELLOW_NOTE, dash_length=0.1)
        mode_label = MathTex(r"z_0", font_size=32, color=YELLOW_NOTE).next_to(mode_line, DOWN, buff=0.1)
        mode_dot = Dot(axes.c2p(MODE_Z, float(target_pdf(MODE_Z))), color=YELLOW_NOTE, radius=0.065)

        step1 = self.formula_box(r"p(z)=\frac{1}{Z}f(z)", font_size=32)
        step2 = self.formula_box(r"f'(z_0)=0", color=YELLOW_NOTE, font_size=32)
        step3 = self.formula_box(r"\ln f(z)\simeq \ln f(z_0)-\frac{A}{2}(z-z_0)^2", color=ORANGE_CURVE, font_size=30)
        step4 = self.formula_box(r"q(z)=\mathcal{N}(z|z_0,A^{-1})", color=RED_LAPLACE, font_size=31)
        steps = VGroup(step1, step2, step3, step4).arrange(DOWN, buff=0.23, aligned_edge=LEFT)
        steps.next_to(axes, RIGHT, buff=0.35).shift(UP * 0.05)
        arrow1 = Arrow(step1.get_bottom(), step2.get_top(), buff=0.08, color=TEXT_GREY)
        arrow2 = Arrow(step2.get_bottom(), step3.get_top(), buff=0.08, color=TEXT_GREY)
        arrow3 = Arrow(step3.get_bottom(), step4.get_top(), buff=0.08, color=TEXT_GREY)

        self.play(FadeIn(label), Write(title), Create(axes), run_time=1.3)
        self.play(Write(step1), Create(true_curve), run_time=1.4)
        self.play(GrowArrow(arrow1), Write(step2), Create(mode_line), FadeIn(mode_dot), FadeIn(mode_label), run_time=1.4)
        self.play(GrowArrow(arrow2), Write(step3), run_time=1.4)
        self.play(GrowArrow(arrow3), Write(step4), Create(approx_curve), run_time=1.7)
        self.wait(0.5)
        self.finish_narration(narration)
        self.clear_scene(label, title, axes, true_curve, approx_curve, mode_line, mode_label, mode_dot, steps, arrow1, arrow2, arrow3)

    def curvature_as_uncertainty(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("PRML 4.4 / negative log view")
        title = self.scene_title("曲率 A が大きいほど、近似分布は細くなる", font_size=33)

        axes = self.negative_log_axes(width=6.9, height=3.5).shift(LEFT * 1.75 + DOWN * 0.05)
        x_values = np.linspace(-1.25, 2.15, 500)
        true_values = np.minimum(negative_log_density(x_values), 7.0)
        quad_values = np.minimum(quadratic_bowl(x_values), 7.0)
        true_curve = self.make_curve_from_values(axes, x_values, true_values, GREEN_TRUE, width=4.2)
        quad_curve = self.make_curve_from_values(axes, x_values, quad_values, RED_LAPLACE, width=4.2)
        mode_dot = Dot(axes.c2p(MODE_Z, 0), color=YELLOW_NOTE, radius=0.07)
        tangent = Line(axes.c2p(MODE_Z - 0.45, 0), axes.c2p(MODE_Z + 0.45, 0), color=YELLOW_NOTE, stroke_width=5)

        equation = self.formula_box(r"A=-\frac{d^2}{dz^2}\ln f(z)\bigg|_{z_0}", color=ORANGE_CURVE, font_size=34)
        equation.to_corner(UR).shift(DOWN * 0.65 + LEFT * 0.1)

        mini_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.1, 0.5],
            x_length=3.3,
            y_length=1.75,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 1.5},
        ).next_to(equation, DOWN, buff=0.45)
        wide = mini_axes.plot(lambda x: math.exp(-0.5 * 0.55 * x * x) * math.sqrt(0.55 / (2 * math.pi)) * 2.3, x_range=[-3, 3], color=BLUE_DATA)
        narrow = mini_axes.plot(lambda x: math.exp(-0.5 * 2.6 * x * x) * math.sqrt(2.6 / (2 * math.pi)) * 2.3, x_range=[-3, 3], color=RED_LAPLACE)
        wide.set_stroke(width=3.5)
        narrow.set_stroke(width=3.5)
        wide_label = Text("小さい A: 広い", font_size=20, color=BLUE_DATA).next_to(mini_axes, DOWN, buff=0.12).align_to(mini_axes, LEFT)
        narrow_label = Text("大きい A: 細い", font_size=20, color=RED_LAPLACE).next_to(wide_label, RIGHT, buff=0.35)

        note = Text("log の山 -> 負の log の谷", font_size=26, color=YELLOW_NOTE).next_to(axes, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), Create(axes), run_time=1.3)
        self.play(Create(true_curve), Write(note), run_time=1.4)
        self.play(FadeIn(mode_dot), Create(tangent), Write(equation), run_time=1.3)
        self.play(Create(quad_curve), run_time=1.5)
        self.play(Create(mini_axes), Create(wide), Write(wide_label), run_time=1.2)
        self.play(Create(narrow), Write(narrow_label), run_time=1.2)
        self.play(Indicate(equation, color=ORANGE_CURVE), run_time=0.9)
        self.finish_narration(narration)
        self.clear_scene(label, title, axes, true_curve, quad_curve, mode_dot, tangent, equation, mini_axes, wide, narrow, wide_label, narrow_label, note)

    def multivariate_extension(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("PRML 4.4 / Eq. (4.131)-(4.134)")
        title = self.scene_title("多変数: Hessian がガウスの精度行列になる", font_size=33)

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5.0,
            y_length=4.0,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 2.25 + DOWN * 0.05)
        center = Dot(axes.c2p(0.15, -0.05), color=YELLOW_NOTE, radius=0.075)
        ellipses = VGroup()
        for scale, opacity in [(1.0, 1.0), (1.55, 0.65), (2.1, 0.38)]:
            ellipse = Ellipse(width=2.6 * scale, height=1.05 * scale, color=RED_LAPLACE, stroke_width=4 * opacity)
            ellipse.rotate(0.48)
            ellipse.move_to(axes.c2p(0.15, -0.05))
            ellipse.set_opacity(opacity)
            ellipses.add(ellipse)
        steep_arrow = Arrow(axes.c2p(0.15, -0.05), axes.c2p(-0.45, 0.75), buff=0, color=ORANGE_CURVE)
        flat_arrow = Arrow(axes.c2p(0.15, -0.05), axes.c2p(1.45, 0.68), buff=0, color=BLUE_DATA)
        steep_label = Text("曲率大", font_size=22, color=ORANGE_CURVE).next_to(steep_arrow.get_end(), UP, buff=0.08)
        flat_label = Text("曲率小", font_size=22, color=BLUE_DATA).next_to(flat_arrow.get_end(), RIGHT, buff=0.08)

        eq1 = self.formula_box(r"\ln f(z)\simeq \ln f(z_0)-\frac{1}{2}(z-z_0)^T A (z-z_0)", color=WHITE, font_size=28)
        eq2 = self.formula_box(r"A=-\nabla\nabla\ln f(z)\big|_{z_0}", color=ORANGE_CURVE, font_size=31)
        eq3 = self.formula_box(r"q(z)=\mathcal{N}(z|z_0,A^{-1})", color=RED_LAPLACE, font_size=31)
        equations = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        equations.next_to(axes, RIGHT, buff=0.35).shift(UP * 0.25)

        valid = VGroup(
            Text("local maximum", font_size=24, color=GREEN_TRUE),
            MathTex(r"A\succ 0", font_size=34, color=GREEN_TRUE),
        ).arrange(RIGHT, buff=0.25)
        invalid = VGroup(
            Text("minimum / saddle", font_size=24, color=RED_LAPLACE),
            Text("A は正定値でない", font_size=22, color=RED_LAPLACE),
        ).arrange(RIGHT, buff=0.25)
        signs = VGroup(valid, invalid).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        signs.next_to(equations, DOWN, buff=0.25)

        self.play(FadeIn(label), Write(title), Create(axes), run_time=1.3)
        self.play(FadeIn(center), Create(ellipses[2]), Create(ellipses[1]), Create(ellipses[0]), run_time=1.5)
        self.play(GrowArrow(steep_arrow), GrowArrow(flat_arrow), Write(steep_label), Write(flat_label), run_time=1.2)
        self.play(Write(eq1), run_time=1.1)
        self.play(Write(eq2), Write(eq3), run_time=1.4)
        self.play(FadeIn(valid), run_time=0.9)
        self.play(FadeIn(invalid), run_time=0.9)
        self.finish_narration(narration)
        self.clear_scene(label, title, axes, center, ellipses, steep_arrow, flat_arrow, steep_label, flat_label, equations, signs)

    def evidence_and_occam(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("PRML 4.4.1 / Eq. (4.135)-(4.137)")
        title = self.scene_title("積分 Z は、山の高さと幅の両方で決まる", font_size=33)

        formula = self.formula_box(r"Z\simeq f(z_0)\frac{(2\pi)^{M/2}}{|A|^{1/2}}", color=PURPLE_EVIDENCE, font_size=36)
        formula.to_edge(UP).shift(DOWN * 1.18)

        panel_group = VGroup()
        specs = [
            ("高いが狭い", RED_LAPLACE, 2.35, 0.23, 1.2),
            ("少し低いが広い", BLUE_DATA, 1.65, 0.65, 4.0),
        ]
        for heading, color, height, sigma, area_value in specs:
            axes = Axes(
                x_range=[-2.2, 2.2, 1],
                y_range=[0, 2.7, 1],
                x_length=4.3,
                y_length=2.65,
                tips=False,
                axis_config={"color": GREY_B, "stroke_width": 1.8},
            )
            curve = axes.plot(
                lambda x, h=height, s=sigma: h * math.exp(-0.5 * (x / s) ** 2),
                x_range=[-2.2, 2.2],
                color=color,
            ).set_stroke(width=4)
            peak = Dot(axes.c2p(0, height), color=YELLOW_NOTE, radius=0.06)
            width_line = Line(axes.c2p(-sigma, 0.18), axes.c2p(sigma, 0.18), color=YELLOW_NOTE, stroke_width=4)
            title_text = Text(heading, font_size=25, color=color).next_to(axes, UP, buff=0.16)
            area = MathTex(r"\mathrm{area}\propto", f"{area_value:.1f}", font_size=28, color=YELLOW_NOTE)
            area.next_to(axes, DOWN, buff=0.12)
            panel = VGroup(axes, curve, peak, width_line, title_text, area)
            panel_group.add(panel)
        panel_group.arrange(RIGHT, buff=0.55).shift(DOWN * 0.55)

        evidence = self.formula_box(r"p(D)=\int p(D|\theta)p(\theta)\,d\theta", color=WHITE, font_size=31)
        evidence.next_to(formula, DOWN, buff=0.25)
        occam = Text("Occam factor: 複雑さを「幅」で調整", font_size=28, color=YELLOW_NOTE)
        occam.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(label), Write(title), Write(formula), run_time=1.4)
        self.play(Write(evidence), run_time=1.1)
        self.play(Create(panel_group[0][0]), Create(panel_group[0][1]), FadeIn(panel_group[0][2:]), run_time=1.4)
        self.play(Create(panel_group[1][0]), Create(panel_group[1][1]), FadeIn(panel_group[1][2:]), run_time=1.4)
        self.play(Write(occam), run_time=1.1)
        self.play(Indicate(panel_group[0][3], color=YELLOW_NOTE), Indicate(panel_group[1][3], color=YELLOW_NOTE), run_time=1.0)
        self.finish_narration(narration)
        self.clear_scene(label, title, formula, evidence, panel_group, occam)

    def bic_bridge(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("PRML 4.4.1 / Eq. (4.138)-(4.139)")
        title = self.scene_title("BIC: 当てはまりから、パラメータ数の罰則を引く", font_size=32)

        lhs = MathTex(r"\ln p(D)\simeq", font_size=35, color=YELLOW_NOTE)
        fit_part = MathTex(r"\ln p(D|\theta_{\mathrm{MAP}})", font_size=35, color=GREEN_TRUE)
        minus = MathTex(r"-", font_size=35, color=YELLOW_NOTE)
        penalty_part = MathTex(r"\frac{M}{2}\ln N", font_size=35, color=RED_LAPLACE)
        formula_terms = VGroup(lhs, fit_part, minus, penalty_part).arrange(RIGHT, buff=0.16)
        formula_terms.to_edge(UP).shift(DOWN * 1.0)
        formula_box = SurroundingRectangle(formula_terms, color=YELLOW_NOTE, buff=0.18)
        formula_box.set_stroke(width=1.5, opacity=0.7)
        formula = VGroup(formula_box, formula_terms)
        under_fit = Brace(fit_part, DOWN, color=GREEN_TRUE)
        fit_label = Text("最適化した尤度", font_size=22, color=GREEN_TRUE).next_to(under_fit, DOWN, buff=0.1)
        under_penalty = Brace(penalty_part, DOWN, color=RED_LAPLACE)
        penalty_label = Text("複雑さへの罰則", font_size=22, color=RED_LAPLACE).next_to(under_penalty, DOWN, buff=0.1)

        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 36, 6],
            x_length=7.5,
            y_length=3.3,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.8)
        y_label = Text("penalty", font_size=20, color=TEXT_GREY).next_to(axes, UP, buff=0.05).align_to(axes, LEFT)
        m_values = [2, 6, 10]

        def make_bars(n_value: int, color: ManimColor) -> VGroup:
            bars = VGroup()
            for i, m_value in enumerate(m_values, start=1):
                penalty = 0.5 * m_value * math.log(n_value)
                width = 0.55
                height = axes.c2p(0, penalty)[1] - axes.c2p(0, 0)[1]
                bar = Rectangle(width=width, height=max(height, 0.001), stroke_width=0)
                bar.set_fill(color, opacity=0.75)
                bar.move_to(axes.c2p(i, penalty / 2.0))
                label_m = MathTex(f"M={m_value}", font_size=24, color=WHITE).next_to(axes.c2p(i, 0), DOWN, buff=0.15)
                value = MathTex(f"{penalty:.1f}", font_size=22, color=color).next_to(bar, UP, buff=0.08)
                bars.add(VGroup(bar, label_m, value))
            return bars

        bars_small = make_bars(50, BLUE_DATA)
        bars_large = make_bars(500, RED_LAPLACE)
        n_label = MathTex(r"N=50", font_size=34, color=BLUE_DATA).next_to(axes, RIGHT, buff=0.35).shift(UP * 0.7)
        n_label_large = MathTex(r"N=500", font_size=34, color=RED_LAPLACE).move_to(n_label)
        note = Text("M が多いほど、N が大きいほど、罰則は重い", font_size=27, color=YELLOW_NOTE)
        note.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(label), Write(title), Write(formula), run_time=1.4)
        self.play(GrowFromCenter(under_fit), Write(fit_label), GrowFromCenter(under_penalty), Write(penalty_label), run_time=1.2)
        self.play(Create(axes), Write(y_label), FadeIn(bars_small), Write(n_label), run_time=1.5)
        self.play(Transform(bars_small, bars_large), Transform(n_label, n_label_large), run_time=1.5)
        self.play(Write(note), run_time=1.0)
        self.finish_narration(narration)
        self.clear_scene(label, title, formula, under_fit, fit_label, under_penalty, penalty_label, axes, y_label, bars_small, n_label, note)

    def strengths_and_limits(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("PRML 4.4 / practical notes")
        title = self.scene_title("得意なのは、一つの鋭い山を局所的に読むこと", font_size=32)

        def item_box(text: str, color: ManimColor, width: float = 3.25) -> VGroup:
            rect = RoundedRectangle(width=width, height=1.0, corner_radius=0.08, color=color)
            rect.set_fill(color, opacity=0.12)
            label_text = Text(text, font_size=24, color=color)
            label_text.move_to(rect)
            return VGroup(rect, label_text)

        strengths = VGroup(
            item_box("Z を知らなくてよい", GREEN_TRUE),
            item_box("大標本で効きやすい", GREEN_TRUE),
            item_box("次節 4.5 の道具", GREEN_TRUE),
        ).arrange(RIGHT, buff=0.35).shift(UP * 1.25)

        limits = VGroup(
            item_box("多峰性は苦手", RED_LAPLACE),
            item_box("強い歪みは苦手", RED_LAPLACE),
            item_box("境界付き変数は変換", RED_LAPLACE),
        ).arrange(RIGHT, buff=0.35).shift(DOWN * 0.25)

        curve_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.1, 0.5],
            x_length=5.8,
            y_length=1.6,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 1.5},
        ).to_edge(DOWN, buff=0.4)
        multimodal = curve_axes.plot(
            lambda x: 0.85 * math.exp(-0.5 * ((x + 1.2) / 0.45) ** 2) + 0.65 * math.exp(-0.5 * ((x - 1.1) / 0.6) ** 2),
            x_range=[-3, 3],
            color=GREEN_TRUE,
        ).set_stroke(width=3.5)
        local_gaussian = curve_axes.plot(
            lambda x: 0.9 * math.exp(-0.5 * ((x + 1.2) / 0.45) ** 2),
            x_range=[-3, 0.2],
            color=RED_LAPLACE,
        ).set_stroke(width=3.5)
        curve_note = Text("一つの mode だけを見ると、もう一つの山を落とす", font_size=23, color=TEXT_GREY)
        curve_note.next_to(curve_axes, UP, buff=0.1)

        bridge = self.formula_box(r"p(w|D)\approx \mathcal{N}(w|w_{\mathrm{MAP}},S_N)", color=PURPLE_EVIDENCE, font_size=34)
        bridge.next_to(limits, DOWN, buff=0.35)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(strengths, lag_ratio=0.15), run_time=1.3)
        self.play(FadeIn(limits, lag_ratio=0.15), run_time=1.3)
        self.play(Write(bridge), run_time=1.1)
        self.play(Create(curve_axes), Create(multimodal), Create(local_gaussian), Write(curve_note), run_time=1.6)
        self.play(Indicate(limits[0], color=RED_LAPLACE), run_time=1.0)
        self.finish_narration(narration)
        self.clear_scene(label, title, strengths, limits, curve_axes, multimodal, local_gaussian, curve_note, bridge)
