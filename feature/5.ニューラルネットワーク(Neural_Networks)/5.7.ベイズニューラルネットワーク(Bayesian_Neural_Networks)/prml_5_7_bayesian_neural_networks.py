from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
GREEN_POSTERIOR = GREEN_C
RED_MAP = RED_C
ORANGE_LIKELIHOOD = ORANGE
YELLOW_EVIDENCE = YELLOW_C
PURPLE_PRIOR = PURPLE_C
TEAL_PREDICTIVE = TEAL_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def gaussian(x: np.ndarray | float, mean: float = 0.0, sigma: float = 1.0) -> np.ndarray | float:
    x_array = np.asarray(x)
    return np.exp(-0.5 * ((x_array - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def target_function(x: np.ndarray | float) -> np.ndarray | float:
    x_array = np.asarray(x)
    return np.sin(1.35 * x_array) + 0.18 * x_array


def prior_function(x: np.ndarray | float, w1: float, w2: float, w3: float, w4: float) -> np.ndarray | float:
    x_array = np.asarray(x)
    return 0.9 * np.tanh(w1 * x_array + w2) + 0.45 * np.tanh(w3 * x_array + w4)


def posterior_function(x: np.ndarray | float, phase: float, slope: float, offset: float) -> np.ndarray | float:
    x_array = np.asarray(x)
    return np.sin(1.35 * x_array + phase) + slope * x_array + offset


class PRML57BayesianNeuralNetworks(Scene):
    """PRML 5.7 Bayesian Neural Networks overview.

    Render example:
        uv run manim -ql prml_5_7_bayesian_neural_networks.py PRML57BayesianNeuralNetworks
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.x_data = np.array([-1.7, -1.25, -0.8, -0.35, 0.05, 0.45, 0.95, 1.35, 1.75])
        noise = np.array([0.09, -0.07, 0.05, -0.04, 0.02, 0.08, -0.05, 0.04, -0.07])
        self.t_data = target_function(self.x_data) + noise

        self.point_estimate_to_distribution()
        self.weight_prior()
        self.likelihood_reweights_candidates()
        self.posterior_distribution()
        self.predictive_distribution()
        self.laplace_approximation()
        self.evidence_and_complexity()
        self.summary()

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

    def function_axes(self, width: float = 7.4, height: float = 3.65) -> Axes:
        return Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.0, 2.0, 1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def density_axes(self, width: float = 4.4, height: float = 2.45) -> Axes:
        return Axes(
            x_range=[-3.0, 3.0, 1],
            y_range=[0.0, 0.55, 0.2],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def parameter_plane(self, width: float = 4.0, height: float = 3.3) -> Axes:
        return Axes(
            x_range=[-2.4, 2.4, 1],
            y_range=[-2.0, 2.0, 1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def data_dots(self, axes: Axes, radius: float = 0.055) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(x), float(t)), radius=radius, color=BLUE_DATA)
                for x, t in zip(self.x_data, self.t_data)
            ]
        )

    def make_network(self, weights: list[float] | None = None, scale: float = 1.0) -> VGroup:
        if weights is None:
            weights = [0.4, -0.8, 1.1, -0.55, 0.75, 0.25, -0.35, 0.9, -0.65]
        layers = [
            [LEFT * 2.3 + UP * 0.85, LEFT * 2.3 + DOWN * 0.85],
            [LEFT * 0.55 + UP * 1.15, LEFT * 0.55, LEFT * 0.55 + DOWN * 1.15],
            [RIGHT * 1.25 + UP * 0.75, RIGHT * 1.25 + DOWN * 0.75],
            [RIGHT * 2.75],
        ]
        nodes = VGroup()
        for layer in layers:
            for point in layer:
                nodes.add(Circle(radius=0.16 * scale, color=WHITE, fill_color="#202020", fill_opacity=1).move_to(point * scale))
        edges = VGroup()
        weight_index = 0
        for left_layer, right_layer in zip(layers, layers[1:]):
            for start in left_layer:
                for end in right_layer:
                    value = weights[weight_index % len(weights)]
                    color = GREEN_POSTERIOR if value >= 0 else RED_MAP
                    line = Line(start * scale, end * scale, color=color, stroke_width=1.2 + 2.2 * min(abs(value), 1.2))
                    line.set_opacity(0.72)
                    edges.add(line)
                    weight_index += 1
        return VGroup(edges, nodes)

    def candidate_curves(self, axes: Axes, kind: str) -> list[VMobject]:
        xs = np.linspace(-3, 3, 180)
        curves: list[VMobject] = []
        if kind == "prior":
            params = [
                (0.6, -1.0, 1.6, 0.5),
                (1.4, 0.25, -0.8, 1.1),
                (-0.9, 1.2, 1.0, -0.7),
                (1.9, -0.2, 0.55, -1.3),
                (-1.4, -0.6, -1.1, 0.8),
            ]
            for parameter in params:
                curve = axes.plot(lambda u, p=parameter: float(prior_function(u, *p)), x_range=[-3, 3], color=PURPLE_PRIOR, use_smoothing=False)
                curve.set_stroke(width=3, opacity=0.58)
                curves.append(curve)
        else:
            params = [
                (-0.16, 0.10, 0.06),
                (0.05, 0.18, -0.04),
                (0.16, 0.12, 0.03),
                (-0.04, 0.21, -0.08),
                (0.11, 0.14, 0.00),
            ]
            for parameter in params:
                curve = axes.plot(lambda u, p=parameter: float(posterior_function(u, *p)), x_range=[-3, 3], color=GREEN_POSTERIOR, use_smoothing=False)
                curve.set_stroke(width=3, opacity=0.6)
                curves.append(curve)
        return curves

    def point_estimate_to_distribution(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 5.7 Bayesian Neural Networks")
        title = self.scene_title("重みを一点ではなく、分布として扱う", font_size=33)
        axes = self.function_axes(width=7.4, height=3.55).shift(DOWN * 0.45)
        dots = self.data_dots(axes)
        map_curve = axes.plot(lambda u: float(target_function(u) + 0.02 * u), x_range=[-3, 3], color=RED_MAP)
        map_curve.set_stroke(width=4)
        posterior_curves = VGroup(*self.candidate_curves(axes, "posterior"))
        point_label = Text("従来: 一組の重み", font_size=24, color=RED_MAP).next_to(axes, RIGHT, buff=0.25).shift(UP * 0.65)
        dist_label = Text("ベイズ: 重みの分布", font_size=24, color=GREEN_POSTERIOR).next_to(point_label, DOWN, buff=0.28)
        equation = MathTex(r"w^\ast \quad \longrightarrow \quad p(w\mid D)", font_size=42).to_edge(DOWN, buff=0.36)

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(FadeIn(dots, shift=UP * 0.1), Create(map_curve), FadeIn(point_label))
        self.play(TransformFromCopy(map_curve, posterior_curves[0]), FadeIn(VGroup(*posterior_curves[1:]), lag_ratio=0.12), FadeIn(dist_label), Write(equation))
        self.play(Indicate(posterior_curves, color=GREEN_POSTERIOR))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, dots, map_curve, posterior_curves, point_label, dist_label, equation)))

    def weight_prior(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("重みの事前分布")
        title = self.scene_title("学習前に、重みのありやすさを決める", font_size=33)
        density = self.density_axes(width=4.4, height=2.35).shift(LEFT * 3.0 + UP * 0.1)
        prior = density.plot(lambda w: gaussian(w, 0, 0.9), x_range=[-3, 3], color=PURPLE_PRIOR)
        prior.set_stroke(width=4)
        formula = MathTex(r"p(w)=\mathcal{N}(w\mid 0,\alpha^{-1}I)", font_size=35, color=PURPLE_PRIOR).next_to(density, DOWN, buff=0.35)
        net_soft = self.make_network(scale=0.8).shift(RIGHT * 2.45 + UP * 0.85)
        net_strong = self.make_network([1.25, -1.15, 1.0, -0.9, 0.8, -1.25, 1.1], scale=0.8).move_to(net_soft)
        axes = self.function_axes(width=5.2, height=2.35).shift(RIGHT * 2.45 + DOWN * 1.55)
        curves = VGroup(*self.candidate_curves(axes, "prior"))
        note = Text("大きすぎる重みは起きにくい", font_size=23, color=TEXT_GREY).next_to(formula, DOWN, buff=0.2)

        self.play(FadeIn(label), Write(title))
        self.play(Create(density), Create(prior), Write(formula), FadeIn(note))
        self.play(FadeIn(net_soft), Create(axes))
        self.play(Transform(net_soft, net_strong), LaggedStart(*[Create(curve) for curve in curves], lag_ratio=0.12), run_time=1.6)
        self.play(Indicate(prior, color=PURPLE_PRIOR), Indicate(curves, color=PURPLE_PRIOR))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, density, prior, formula, note, net_soft, axes, curves)))

    def likelihood_reweights_candidates(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("尤度")
        title = self.scene_title("データは、重み候補の重み付けを変える", font_size=32)
        axes = self.function_axes(width=7.4, height=3.65).shift(LEFT * 1.0 + DOWN * 0.28)
        dots = self.data_dots(axes)
        bad = axes.plot(lambda u: float(-0.55 * np.sin(1.2 * u) + 0.55), x_range=[-3, 3], color=TEXT_GREY, use_smoothing=False)
        mid = axes.plot(lambda u: float(np.sin(1.15 * u - 0.35) + 0.12 * u - 0.18), x_range=[-3, 3], color=ORANGE_LIKELIHOOD, use_smoothing=False)
        good = axes.plot(lambda u: float(posterior_function(u, 0.03, 0.17, 0.01)), x_range=[-3, 3], color=GREEN_POSTERIOR, use_smoothing=False)
        for curve in [bad, mid, good]:
            curve.set_stroke(width=4)
        residuals = VGroup()
        for x, t in zip(self.x_data[::2], self.t_data[::2]):
            y_bad = -0.55 * math.sin(1.2 * float(x)) + 0.55
            residuals.add(Line(axes.c2p(float(x), float(t)), axes.c2p(float(x), y_bad), color=ORANGE_LIKELIHOOD, stroke_width=3))
        scores = VGroup(
            Text("低い尤度", font_size=22, color=TEXT_GREY),
            Text("中くらい", font_size=22, color=ORANGE_LIKELIHOOD),
            Text("高い尤度", font_size=22, color=GREEN_POSTERIOR),
        ).arrange(DOWN, buff=0.28).next_to(axes, RIGHT, buff=0.35)
        formula = MathTex(r"p(D\mid w)=\prod_n p(t_n\mid x_n,w)", font_size=35, color=ORANGE_LIKELIHOOD).to_edge(DOWN, buff=0.42)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(dots))
        self.play(Create(bad), FadeIn(scores[0]))
        self.play(Create(residuals), Write(formula))
        mid_trace = bad.copy()
        self.play(Transform(mid_trace, mid), FadeIn(scores[1]))
        self.play(Create(good), FadeIn(scores[2]), FadeOut(residuals))
        self.play(Indicate(good, color=GREEN_POSTERIOR), Indicate(scores[2], color=GREEN_POSTERIOR))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, dots, bad, mid_trace, good, residuals, scores, formula)))

    def posterior_distribution(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Bayes rule")
        title = self.scene_title("事前分布と尤度から、事後分布へ", font_size=33)
        formula = MathTex(
            r"p(w\mid D)=\frac{p(D\mid w)p(w)}{p(D)}",
            font_size=45,
        ).to_edge(UP).shift(DOWN * 1.05)
        prior_axes = self.density_axes(width=3.2, height=2.0).shift(LEFT * 4.0 + DOWN * 0.45)
        like_axes = self.density_axes(width=3.2, height=2.0).shift(DOWN * 0.45)
        post_axes = self.density_axes(width=3.2, height=2.0).shift(RIGHT * 4.0 + DOWN * 0.45)
        prior = prior_axes.plot(lambda w: gaussian(w, 0, 1.0), x_range=[-3, 3], color=PURPLE_PRIOR)
        likelihood = like_axes.plot(lambda w: 0.8 * gaussian(w, 0.75, 0.55), x_range=[-3, 3], color=ORANGE_LIKELIHOOD)
        posterior = post_axes.plot(lambda w: gaussian(w, 0.55, 0.42), x_range=[-3, 3], color=GREEN_POSTERIOR)
        for curve in [prior, likelihood, posterior]:
            curve.set_stroke(width=4)
        labels = VGroup(
            Text("事前分布", font_size=23, color=PURPLE_PRIOR).next_to(prior_axes, DOWN, buff=0.18),
            Text("尤度", font_size=23, color=ORANGE_LIKELIHOOD).next_to(like_axes, DOWN, buff=0.18),
            Text("事後分布", font_size=23, color=GREEN_POSTERIOR).next_to(post_axes, DOWN, buff=0.18),
        )
        arrows = VGroup(
            Arrow(prior_axes.get_right(), like_axes.get_left(), buff=0.25, color=TEXT_GREY),
            Arrow(like_axes.get_right(), post_axes.get_left(), buff=0.25, color=TEXT_GREY),
        )
        evidence = MathTex(r"p(D)=\int p(D\mid w)p(w)\,dw", font_size=33, color=YELLOW_EVIDENCE).to_edge(DOWN, buff=0.42)
        width_note = Text("幅が残る = 不確かさが残る", font_size=25, color=TEXT_GREY).next_to(posterior, UP, buff=0.16)

        self.play(FadeIn(label), Write(title), Write(formula))
        self.play(Create(prior_axes), Create(prior), FadeIn(labels[0]))
        self.play(Create(like_axes), Create(likelihood), FadeIn(labels[1]), GrowArrow(arrows[0]))
        self.play(Create(post_axes), Create(posterior), FadeIn(labels[2]), GrowArrow(arrows[1]))
        self.play(Write(evidence), FadeIn(width_note))
        self.play(Indicate(posterior, color=GREEN_POSTERIOR), Indicate(evidence, color=YELLOW_EVIDENCE))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, formula, prior_axes, like_axes, post_axes, prior, likelihood, posterior, labels, arrows, evidence, width_note)))

    def predictive_band(self, axes: Axes) -> tuple[Polygon, VMobject, VGroup]:
        xs = np.linspace(-3, 3, 160)
        mean = target_function(xs) + 0.02 * xs
        uncertainty = 0.12 + 0.10 * np.maximum(np.abs(xs) - 1.75, 0) ** 2
        upper = mean + uncertainty
        lower = mean - uncertainty
        points = [axes.c2p(float(x), float(y)) for x, y in zip(xs, upper)]
        points += [axes.c2p(float(x), float(y)) for x, y in zip(xs[::-1], lower[::-1])]
        band = Polygon(*points, color=TEAL_PREDICTIVE, fill_color=TEAL_PREDICTIVE, fill_opacity=0.18, stroke_width=0)
        mean_curve = axes.plot(lambda u: float(target_function(u) + 0.02 * u), x_range=[-3, 3], color=TEAL_PREDICTIVE)
        mean_curve.set_stroke(width=4)
        curves = VGroup(*self.candidate_curves(axes, "posterior"))
        return band, mean_curve, curves

    def predictive_distribution(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("予測分布")
        title = self.scene_title("重みの分布を平均して、新しい入力を予測する", font_size=31)
        axes = self.function_axes(width=7.5, height=3.6).shift(DOWN * 0.35)
        dots = self.data_dots(axes)
        band, mean_curve, curves = self.predictive_band(axes)
        formula = MathTex(r"p(t\mid x,D)=\int p(t\mid x,w)p(w\mid D)\,dw", font_size=36).to_edge(DOWN, buff=0.38)
        inside = Text("データ付近: 狭い", font_size=23, color=GREEN_POSTERIOR).next_to(axes.c2p(0.0, 1.45), UP, buff=0.1)
        outside = Text("外側: 広い", font_size=23, color=YELLOW_EVIDENCE).next_to(axes.c2p(2.35, 1.45), UP, buff=0.1)
        x_mark = DashedLine(axes.c2p(2.45, -1.75), axes.c2p(2.45, 1.65), color=YELLOW_EVIDENCE, dash_length=0.12)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(dots))
        self.play(LaggedStart(*[Create(curve) for curve in curves], lag_ratio=0.1), run_time=1.4)
        self.play(FadeIn(band), Create(mean_curve), Write(formula))
        self.play(FadeIn(inside), FadeIn(outside), Create(x_mark))
        self.play(Indicate(band, color=TEAL_PREDICTIVE), Indicate(outside, color=YELLOW_EVIDENCE))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, dots, band, mean_curve, curves, formula, inside, outside, x_mark)))

    def ellipse_on_axes(self, axes: Axes, center: tuple[float, float], width: float, height: float, angle: float, color: ManimColor) -> Ellipse:
        x0, y0 = center
        p_center = axes.c2p(x0, y0)
        p_width = axes.c2p(x0 + width, y0)
        p_height = axes.c2p(x0, y0 + height)
        ellipse = Ellipse(
            width=abs(p_width[0] - p_center[0]) * 2,
            height=abs(p_height[1] - p_center[1]) * 2,
            color=color,
            stroke_width=4,
        )
        ellipse.move_to(p_center).rotate(angle)
        return ellipse

    def laplace_approximation(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Laplace approximation")
        title = self.scene_title("MAP 解の近くをガウス分布で近似する", font_size=32)
        axes = self.parameter_plane(width=4.2, height=3.4).shift(LEFT * 3.0 + DOWN * 0.05)
        ellipses = VGroup(
            self.ellipse_on_axes(axes, (0.45, -0.2), 1.45, 0.86, -0.45, PURPLE_PRIOR),
            self.ellipse_on_axes(axes, (0.45, -0.2), 0.95, 0.52, -0.45, ORANGE_LIKELIHOOD),
            self.ellipse_on_axes(axes, (0.45, -0.2), 0.48, 0.25, -0.45, GREEN_POSTERIOR),
        )
        map_dot = Dot(axes.c2p(0.45, -0.2), radius=0.08, color=RED_MAP)
        map_label = MathTex(r"w_{\mathrm{MAP}}", font_size=30, color=RED_MAP).next_to(map_dot, RIGHT, buff=0.12)
        surface = VGroup(
            MathTex(r"E(w)\simeq E(w_{\mathrm{MAP}})", font_size=34),
            MathTex(r"+\frac{1}{2}(w-w_{\mathrm{MAP}})^T H (w-w_{\mathrm{MAP}})", font_size=32),
            MathTex(r"p(w\mid D)\approx \mathcal{N}(w_{\mathrm{MAP}},H^{-1})", font_size=34, color=GREEN_POSTERIOR),
        ).arrange(DOWN, buff=0.32).to_edge(RIGHT).shift(LEFT * 0.45 + DOWN * 0.05)
        notes = VGroup(
            Text("中心: MAP", font_size=23, color=RED_MAP),
            Text("幅: Hessian の逆行列", font_size=23, color=GREEN_POSTERIOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(surface, DOWN, buff=0.35)

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(Create(ellipses[0]), Create(ellipses[1]))
        self.play(FadeIn(map_dot), FadeIn(map_label), Create(ellipses[2]))
        self.play(Write(surface[0]), Write(surface[1]))
        self.play(Write(surface[2]), FadeIn(notes))
        self.play(Indicate(ellipses[2], color=GREEN_POSTERIOR), Indicate(surface[2], color=GREEN_POSTERIOR))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, ellipses, map_dot, map_label, surface, notes)))

    def evidence_and_complexity(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Evidence")
        title = self.scene_title("当てはまりだけでなく、複雑さも見る", font_size=32)
        bars = VGroup()
        labels = VGroup()
        values = [0.58, 0.88, 0.62]
        names = ["単純すぎる", "ちょうどよい", "複雑すぎる"]
        colors = [TEXT_GREY, GREEN_POSTERIOR, ORANGE_LIKELIHOOD]
        for index, (value, name, color) in enumerate(zip(values, names, colors)):
            rect = Rectangle(width=1.3, height=3.0 * value, color=color, fill_color=color, fill_opacity=0.72)
            rect.move_to(LEFT * 2.6 + RIGHT * index * 2.6 + DOWN * (1.25 - 1.5 * value))
            bars.add(rect)
            labels.add(Text(name, font_size=22, color=color).next_to(rect, DOWN, buff=0.18))
        chart_title = MathTex(r"p(D\mid \mathcal{M})", font_size=38, color=YELLOW_EVIDENCE).next_to(bars, UP, buff=0.35)
        fit = Text("当てはまり", font_size=24, color=GREEN_POSTERIOR).to_edge(RIGHT).shift(UP * 0.85 + LEFT * 0.8)
        complexity = Text("複雑さの広がり", font_size=24, color=ORANGE_LIKELIHOOD).next_to(fit, DOWN, buff=0.28)
        arrow1 = Arrow(fit.get_left() + LEFT * 0.15, bars[1].get_top(), buff=0.1, color=GREEN_POSTERIOR)
        arrow2 = Arrow(complexity.get_left() + LEFT * 0.15, bars[2].get_top(), buff=0.1, color=ORANGE_LIKELIHOOD)
        formula = MathTex(r"p(D\mid \mathcal{M})=\int p(D\mid w,\mathcal{M})p(w\mid\mathcal{M})\,dw", font_size=33).to_edge(DOWN, buff=0.42)
        brace = Brace(bars[2], UP, color=ORANGE_LIKELIHOOD)
        narrow_note = Text("よく合う領域が狭い", font_size=22, color=ORANGE_LIKELIHOOD).next_to(brace, UP, buff=0.14)

        self.play(FadeIn(label), Write(title), Write(chart_title))
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.18), FadeIn(labels))
        self.play(FadeIn(fit), GrowArrow(arrow1), Write(formula))
        self.play(FadeIn(complexity), GrowArrow(arrow2), Create(brace), FadeIn(narrow_note))
        self.play(Indicate(bars[1], color=GREEN_POSTERIOR), Indicate(formula, color=YELLOW_EVIDENCE))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, bars, labels, chart_title, fit, complexity, arrow1, arrow2, formula, brace, narrow_note)))

    def summary(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("まとめ")
        title = self.scene_title("一本の答えではなく、ありそうな答えの分布を見る", font_size=30)
        left = self.make_network(scale=0.78).shift(LEFT * 3.2 + DOWN * 0.1)
        right_axes = self.function_axes(width=5.1, height=2.85).shift(RIGHT * 2.4 + DOWN * 0.25)
        dots = self.data_dots(right_axes, radius=0.045)
        band, mean_curve, curves = self.predictive_band(right_axes)
        bullets = VGroup(
            Text("1. 重みに事前分布を置く", font_size=24, color=PURPLE_PRIOR),
            Text("2. データで事後分布へ更新する", font_size=24, color=GREEN_POSTERIOR),
            Text("3. 予測では重みを平均する", font_size=24, color=TEAL_PREDICTIVE),
            Text("4. 近似推論で計算可能にする", font_size=24, color=YELLOW_EVIDENCE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).to_edge(DOWN, buff=0.45)
        equation = MathTex(r"p(w)\rightarrow p(w\mid D)\rightarrow p(t\mid x,D)", font_size=38).next_to(title, DOWN, buff=0.28)
        arrow = Arrow(left.get_right(), right_axes.get_left(), buff=0.3, color=TEXT_GREY)

        self.play(FadeIn(label), Write(title), Write(equation))
        self.play(FadeIn(left), Create(right_axes), GrowArrow(arrow))
        self.play(FadeIn(dots), LaggedStart(*[Create(curve) for curve in curves], lag_ratio=0.08), FadeIn(band), Create(mean_curve), run_time=1.6)
        self.play(FadeIn(bullets, shift=UP * 0.15))
        self.play(Indicate(equation, color=YELLOW_EVIDENCE), Indicate(band, color=TEAL_PREDICTIVE))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, equation, left, right_axes, dots, curves, band, mean_curve, arrow, bullets)))
