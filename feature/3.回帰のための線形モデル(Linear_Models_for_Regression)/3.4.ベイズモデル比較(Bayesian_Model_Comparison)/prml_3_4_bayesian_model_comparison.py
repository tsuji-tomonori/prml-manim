from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
GOOD_GREEN = GREEN_C
EVIDENCE_GOLD = GOLD_C
PRIOR_PURPLE = PURPLE_C
OCCAM_TEAL = TEAL_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def make_sine_data(n: int = 12, noise_std: float = 0.22, seed: int = 34) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    t = np.sin(2.0 * np.pi * x) + rng.normal(0.0, noise_std, size=n)
    return x, t


def design_matrix(x: np.ndarray | float, degree: int) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    return np.vander(x_array, N=degree + 1, increasing=True)


def fit_polynomial(x: np.ndarray, t: np.ndarray, degree: int, lam: float = 1e-4) -> np.ndarray:
    phi = design_matrix(x, degree)
    penalty = lam * np.eye(degree + 1)
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ t)


def eval_poly(w: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    return design_matrix(x, len(w) - 1) @ w


def clipped_model(w: np.ndarray, u: float) -> float:
    return float(np.clip(eval_poly(w, u)[0], -1.65, 1.65))


def evidence_profile() -> tuple[list[int], list[float], list[float], list[float]]:
    degrees = list(range(1, 10))
    fit = [0.28, 0.46, 0.74, 0.86, 0.93, 0.98, 1.02, 1.04, 1.05]
    occam = [0.10, 0.18, 0.26, 0.37, 0.49, 0.61, 0.73, 0.85, 0.97]
    evidence = [f - o for f, o in zip(fit, occam)]
    low = min(evidence)
    high = max(evidence)
    normalized = [(v - low) / (high - low) * 0.72 + 0.20 for v in evidence]
    return degrees, fit, occam, normalized


class PRML34BayesianModelComparison(Scene):
    """PRML 3.4 Bayesian model comparison overview.

    Render example:
        uv run manim -pql prml_3_4_bayesian_model_comparison.py PRML34BayesianModelComparison
    """

    def construct(self) -> None:
        self.camera.background_color = "#111111"
        self.x_train, self.t_train = make_sine_data()

        self.opening_bayesian_question()
        self.posterior_over_models()
        self.integrating_weights()
        self.occam_factor()
        self.polynomial_evidence()
        self.posterior_odds()
        self.bayesian_model_averaging()
        self.bridge_to_evidence_approximation()

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
            self.wait(0.8)
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
        title.to_edge(UP).shift(DOWN * 0.28)
        return title

    def make_axes(self, width: float = 4.2, height: float = 2.65) -> Axes:
        return Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1.55, 1.55, 1.0],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def data_dots(self, axes: Axes, radius: float = 0.045) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(x), float(t)), color=BLUE_DATA, radius=radius)
                for x, t in zip(self.x_train, self.t_train)
            ]
        )

    def model_curve(self, axes: Axes, degree: int, color: ManimColor, width: float = 4.0) -> VMobject:
        w = fit_polynomial(self.x_train, self.t_train, degree, lam=5e-4 if degree > 5 else 1e-4)
        curve = axes.plot(lambda u: clipped_model(w, u), x_range=[0, 1], color=color, use_smoothing=False)
        curve.set_stroke(width=width)
        return curve

    def model_card(self, degree: int, color: ManimColor) -> VGroup:
        axes = self.make_axes(width=3.15, height=2.0)
        dots = self.data_dots(axes, radius=0.035)
        curve = self.model_curve(axes, degree, color=color, width=3.4)
        label = Text(f"M={degree}", font_size=23, color=color).next_to(axes, UP, buff=0.12)
        card = VGroup(axes, dots, curve, label)
        return card

    def bar_group(
        self,
        values: list[float],
        labels: list[str],
        color: ManimColor,
        width: float = 0.48,
        max_height: float = 1.7,
    ) -> VGroup:
        group = VGroup()
        for value, label_text in zip(values, labels):
            bar = Rectangle(width=width, height=max_height * value, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.85)
            bar.align_to(ORIGIN, DOWN)
            label = Text(label_text, font_size=18, color=TEXT_GREY).next_to(bar, DOWN, buff=0.10)
            item = VGroup(bar, label)
            if group:
                item.next_to(group[-1], RIGHT, buff=0.32, aligned_edge=DOWN)
            group.add(item)
        return group

    def opening_bayesian_question(self) -> None:
        self.clear()
        narration = self.start_narration("scene01")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("モデルそのものを確率で比べる")

        cards = VGroup(
            self.model_card(1, MODEL_RED),
            self.model_card(3, GOOD_GREEN),
            self.model_card(9, EVIDENCE_GOLD),
        ).arrange(RIGHT, buff=0.45)
        cards.shift(UP * 0.35)

        uncertainty = VGroup(
            Text("重み w の不確かさ", font_size=24, color=TEXT_GREY),
            Arrow(LEFT * 1.0, RIGHT * 1.0, buff=0.05, color=TEXT_GREY),
            Text("モデル M の不確かさ", font_size=24, color=WHITE),
        ).arrange(RIGHT, buff=0.25)
        uncertainty.to_edge(DOWN).shift(UP * 0.75)

        evidence = Text("Evidence: p(D | M)", font_size=30, color=EVIDENCE_GOLD)
        evidence.next_to(cards, DOWN, buff=0.30)

        self.play(FadeIn(label), Write(title))
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.15) for card in cards], lag_ratio=0.18), run_time=1.8)
        self.play(Write(evidence), FadeIn(uncertainty))
        self.play(cards[1].animate.scale(1.08), evidence.animate.set_color(GOOD_GREEN), run_time=0.8)
        self.finish_narration(narration)

    def posterior_over_models(self) -> None:
        self.clear()
        narration = self.start_narration("scene02")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("モデルの事後確率")
        equation = MathTex(
            r"p(M_i\mid D)\ \propto\ p(M_i)\,p(D\mid M_i)",
            font_size=42,
            color=WHITE,
        ).shift(UP * 1.65)

        prior = self.bar_group([0.50, 0.50, 0.50], ["M1", "M2", "M3"], PRIOR_PURPLE).scale(0.85)
        evidence = self.bar_group([0.25, 0.88, 0.42], ["M1", "M2", "M3"], EVIDENCE_GOLD).scale(0.85)
        posterior = self.bar_group([0.18, 0.92, 0.33], ["M1", "M2", "M3"], GOOD_GREEN).scale(0.85)
        rows = VGroup(prior, evidence, posterior).arrange(DOWN, buff=0.34, aligned_edge=LEFT).shift(LEFT * 0.65 + DOWN * 0.45)

        row_labels = VGroup(
            Text("prior", font_size=22, color=PRIOR_PURPLE),
            Text("evidence", font_size=22, color=EVIDENCE_GOLD),
            Text("posterior", font_size=22, color=GOOD_GREEN),
        )
        for row_label, row in zip(row_labels, rows):
            row_label.next_to(row, LEFT, buff=0.35)

        flow = VGroup(
            Arrow(rows[0].get_right() + RIGHT * 0.15, rows[1].get_right() + RIGHT * 0.15, color=TEXT_GREY, buff=0.05),
            Arrow(rows[1].get_right() + RIGHT * 0.15, rows[2].get_right() + RIGHT * 0.15, color=TEXT_GREY, buff=0.05),
        )
        note = Text("データでモデル確率を更新する", font_size=25, color=WHITE).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title), Write(equation))
        self.play(FadeIn(row_labels[0]), FadeIn(prior))
        self.play(FadeIn(flow[0]), FadeIn(row_labels[1]), FadeIn(evidence))
        self.play(FadeIn(flow[1]), FadeIn(row_labels[2]), FadeIn(posterior), Write(note))
        self.finish_narration(narration)

    def integrating_weights(self) -> None:
        self.clear()
        narration = self.start_narration("scene03")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("重み w を積分して消す")
        equation = MathTex(
            r"p(D\mid M)=\int p(D\mid w,M)\,p(w\mid M)\,dw",
            font_size=38,
        ).to_edge(DOWN).shift(UP * 0.45)

        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=6.0,
            y_length=4.0,
            background_line_style={"stroke_color": GREY_E, "stroke_width": 1, "stroke_opacity": 0.45},
            axis_config={"stroke_color": GREY_B, "stroke_width": 2},
        ).shift(UP * 0.15)
        prior = Ellipse(width=5.0, height=3.1, color=PRIOR_PURPLE, fill_color=PRIOR_PURPLE, fill_opacity=0.12, stroke_width=4).move_to(plane)
        likelihood = Ellipse(width=1.45, height=0.85, color=EVIDENCE_GOLD, fill_color=EVIDENCE_GOLD, fill_opacity=0.32, stroke_width=4)
        likelihood.move_to(plane.c2p(0.7, 0.35)).rotate(0.3)
        product = Ellipse(width=1.10, height=0.63, color=GOOD_GREEN, fill_color=GOOD_GREEN, fill_opacity=0.5, stroke_width=4)
        product.move_to(likelihood).rotate(0.3)

        labels = VGroup(
            MathTex("w_1", font_size=28, color=TEXT_GREY).next_to(plane.x_axis, RIGHT, buff=0.1),
            MathTex("w_2", font_size=28, color=TEXT_GREY).next_to(plane.y_axis, UP, buff=0.1),
            Text("prior", font_size=23, color=PRIOR_PURPLE).next_to(prior, LEFT, buff=0.2),
            Text("likelihood", font_size=23, color=EVIDENCE_GOLD).next_to(likelihood, RIGHT, buff=0.18),
            Text("積分される重なり", font_size=24, color=GOOD_GREEN).next_to(product, DOWN, buff=0.22),
        )

        self.play(FadeIn(label), Write(title))
        self.play(Create(plane), FadeIn(labels[0]), FadeIn(labels[1]))
        self.play(Create(prior), Write(labels[2]))
        self.play(Create(likelihood), Write(labels[3]))
        self.play(FadeIn(product), Write(labels[4]), Write(equation))
        self.finish_narration(narration)

    def occam_panel(self, name: str, width: float, useful_width: float, color: ManimColor) -> VGroup:
        outer = Rectangle(width=width, height=1.55, stroke_color=color, stroke_width=4, fill_color=color, fill_opacity=0.10)
        useful = Rectangle(width=useful_width, height=1.2, stroke_color=GOOD_GREEN, stroke_width=3, fill_color=GOOD_GREEN, fill_opacity=0.45)
        useful.move_to(outer)
        peak = Line(useful.get_top(), useful.get_top() + UP * 0.55, color=EVIDENCE_GOLD, stroke_width=6)
        label = Text(name, font_size=24, color=color).next_to(outer, UP, buff=0.22)
        volume = Text("事前体積", font_size=19, color=TEXT_GREY).next_to(outer, DOWN, buff=0.16)
        active = Text("合う領域", font_size=19, color=GOOD_GREEN).move_to(useful)
        return VGroup(outer, useful, peak, label, volume, active)

    def occam_factor(self) -> None:
        self.clear()
        narration = self.start_narration("scene04")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("エビデンスに入るオッカム因子")

        simple = self.occam_panel("単純なモデル", 2.7, 1.25, OCCAM_TEAL).shift(LEFT * 3.0 + UP * 0.25)
        complex_model = self.occam_panel("複雑なモデル", 4.4, 0.95, MODEL_RED).shift(RIGHT * 2.25 + UP * 0.25)

        formula = MathTex(
            r"\mathrm{evidence}\approx \mathrm{best\ fit}\times \frac{\mathrm{useful\ volume}}{\mathrm{prior\ volume}}",
            font_size=31,
        ).to_edge(DOWN).shift(UP * 0.70)
        bars = VGroup(
            self.bar_group([0.75, 0.55], ["simple", "complex"], OCCAM_TEAL, width=0.55, max_height=1.65),
            Text("evidence", font_size=23, color=EVIDENCE_GOLD),
        ).arrange(DOWN, buff=0.18).shift(DOWN * 0.95)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(simple), FadeIn(complex_model), run_time=1.2)
        self.play(simple[1].animate.set_fill(GOOD_GREEN, opacity=0.60), complex_model[1].animate.set_fill(GOOD_GREEN, opacity=0.60))
        self.play(Write(formula))
        self.play(FadeIn(bars))
        self.finish_narration(narration)

    def polynomial_evidence(self) -> None:
        self.clear()
        narration = self.start_narration("scene05")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("次数 M をエビデンスで比べる")

        axes = self.make_axes(width=5.1, height=3.1).shift(LEFT * 3.05 + DOWN * 0.12)
        dots = self.data_dots(axes)
        curves = [
            self.model_curve(axes, 1, MODEL_RED),
            self.model_curve(axes, 3, GOOD_GREEN),
            self.model_curve(axes, 9, EVIDENCE_GOLD),
        ]
        curve_labels = [
            Text("M=1: 硬すぎる", font_size=22, color=MODEL_RED),
            Text("M=3: よく説明", font_size=22, color=GOOD_GREEN),
            Text("M=9: 広すぎる", font_size=22, color=EVIDENCE_GOLD),
        ]
        for curve_label in curve_labels:
            curve_label.next_to(axes, DOWN, buff=0.24)

        degrees, _, _, evidences = evidence_profile()
        bar_axes = Axes(
            x_range=[0.5, 9.5, 1],
            y_range=[0, 1.0, 0.25],
            x_length=4.6,
            y_length=3.1,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 3.05 + DOWN * 0.12)
        bars = VGroup()
        for degree, value in zip(degrees, evidences):
            color = GOOD_GREEN if degree == 3 else EVIDENCE_GOLD if degree < 7 else MODEL_RED
            bottom = bar_axes.c2p(degree, 0)
            top = bar_axes.c2p(degree, value)
            bar = Rectangle(width=0.34, height=max(top[1] - bottom[1], 0.02), stroke_color=color, fill_color=color, fill_opacity=0.82, stroke_width=2)
            bar.move_to((bottom + top) / 2)
            bars.add(bar)
        bar_title = Text("エビデンス", font_size=24, color=EVIDENCE_GOLD).next_to(bar_axes, UP, buff=0.12)
        x_label = Text("次数 M", font_size=20, color=TEXT_GREY).next_to(bar_axes.x_axis, DOWN, buff=0.22)
        best_marker = Text("高い", font_size=21, color=GOOD_GREEN).next_to(bars[2], UP, buff=0.14)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(dots), Create(bar_axes), Write(bar_title), Write(x_label))
        self.play(Create(curves[0]), Write(curve_labels[0]))
        self.play(Transform(curves[0], curves[1]), Transform(curve_labels[0], curve_labels[1]))
        self.play(Transform(curves[0], curves[2]), Transform(curve_labels[0], curve_labels[2]))
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.06), FadeIn(best_marker), run_time=1.4)
        self.finish_narration(narration)

    def posterior_odds(self) -> None:
        self.clear()
        narration = self.start_narration("scene06")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("モデル同士は比で比較できる")

        equation = MathTex(
            r"\frac{p(M_a\mid D)}{p(M_b\mid D)}="
            r"\frac{p(M_a)}{p(M_b)}"
            r"\frac{p(D\mid M_a)}{p(D\mid M_b)}",
            font_size=38,
        ).shift(UP * 1.8)

        beam = Line(LEFT * 2.7, RIGHT * 2.7, color=GREY_B, stroke_width=6).shift(UP * 0.15)
        pivot = Triangle(color=GREY_B, fill_color=GREY_B, fill_opacity=0.85).scale(0.35).next_to(beam, DOWN, buff=0.0)
        left_pan = Circle(radius=0.55, color=GOOD_GREEN, fill_color=GOOD_GREEN, fill_opacity=0.20).shift(LEFT * 2.1 + DOWN * 0.75)
        right_pan = Circle(radius=0.55, color=MODEL_RED, fill_color=MODEL_RED, fill_opacity=0.20).shift(RIGHT * 2.1 + DOWN * 0.75)
        left_line = Line(beam.get_start() + RIGHT * 0.6, left_pan.get_top(), color=GREY_B, stroke_width=2)
        right_line = Line(beam.get_end() + LEFT * 0.6, right_pan.get_top(), color=GREY_B, stroke_width=2)
        model_labels = VGroup(
            Text("M_a", font_size=26, color=GOOD_GREEN).move_to(left_pan),
            Text("M_b", font_size=26, color=MODEL_RED).move_to(right_pan),
        )
        ratio_labels = VGroup(
            Text("事前オッズ", font_size=22, color=PRIOR_PURPLE),
            Text("エビデンス比", font_size=22, color=EVIDENCE_GOLD),
        ).arrange(RIGHT, buff=0.8).to_edge(DOWN).shift(UP * 0.55)
        arrows = VGroup(
            Arrow(ratio_labels[0].get_top(), beam.get_center() + LEFT * 1.2 + DOWN * 0.15, color=PRIOR_PURPLE, buff=0.1),
            Arrow(ratio_labels[1].get_top(), left_pan.get_center() + UP * 0.2, color=EVIDENCE_GOLD, buff=0.1),
        )
        data_note = Text("データが増えるほど evidence ratio が効く", font_size=24, color=WHITE).next_to(beam, UP, buff=0.45)

        self.play(FadeIn(label), Write(title), Write(equation))
        self.play(Create(beam), FadeIn(pivot), Create(left_line), Create(right_line), FadeIn(left_pan), FadeIn(right_pan), FadeIn(model_labels))
        self.play(FadeIn(ratio_labels), Create(arrows))
        self.play(Rotate(VGroup(beam, left_line, right_line, left_pan, right_pan, model_labels), angle=-0.15, about_point=pivot.get_top()), Write(data_note), run_time=1.0)
        self.finish_narration(narration)

    def bayesian_model_averaging(self) -> None:
        self.clear()
        narration = self.start_narration("scene07")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("ひとつに決めず、予測を平均する")

        axes = self.make_axes(width=7.0, height=3.6).shift(DOWN * 0.15)
        dots = self.data_dots(axes, radius=0.05)
        c1 = self.model_curve(axes, 1, MODEL_RED, width=3.0).set_opacity(0.45)
        c3 = self.model_curve(axes, 3, GOOD_GREEN, width=4.0).set_opacity(0.75)
        c9 = self.model_curve(axes, 9, EVIDENCE_GOLD, width=3.0).set_opacity(0.45)

        xs = np.linspace(0, 1, 160)
        w1 = fit_polynomial(self.x_train, self.t_train, 1)
        w3 = fit_polynomial(self.x_train, self.t_train, 3)
        w9 = fit_polynomial(self.x_train, self.t_train, 9, lam=5e-4)
        weights = np.array([0.18, 0.62, 0.20])

        def averaged(u: float) -> float:
            y = weights[0] * clipped_model(w1, u) + weights[1] * clipped_model(w3, u) + weights[2] * clipped_model(w9, u)
            return float(np.clip(y, -1.65, 1.65))

        avg_curve = axes.plot(averaged, x_range=[0, 1], color=WHITE, use_smoothing=False).set_stroke(width=5.5)
        formula = MathTex(
            r"p(t\mid x,D)=\sum_i p(t\mid x,M_i,D)\,p(M_i\mid D)",
            font_size=34,
        ).to_edge(DOWN).shift(UP * 0.35)
        weight_labels = VGroup(
            Text("0.18", font_size=22, color=MODEL_RED).next_to(c1, LEFT, buff=0.10),
            Text("0.62", font_size=22, color=GOOD_GREEN).next_to(c3, UP, buff=0.12),
            Text("0.20", font_size=22, color=EVIDENCE_GOLD).next_to(c9, RIGHT, buff=0.10),
        )
        average_label = Text("model averaged prediction", font_size=24, color=WHITE).next_to(avg_curve, UP, buff=0.20)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(dots))
        self.play(Create(c1), Create(c3), Create(c9), FadeIn(weight_labels), run_time=1.3)
        self.play(Create(avg_curve), Write(average_label), Write(formula))
        self.finish_narration(narration)

    def bridge_to_evidence_approximation(self) -> None:
        self.clear()
        narration = self.start_narration("scene08")
        label = self.section_label("3.4 Bayesian Model Comparison")
        title = self.scene_title("3.5 エビデンス近似へ")

        items = VGroup(
            self.summary_item("1", "モデル事後確率", "p(M|D)"),
            self.summary_item("2", "エビデンス積分", "p(D|M)"),
            self.summary_item("3", "オッカム因子", "fit x volume ratio"),
            self.summary_item("4", "モデル平均", "sum over M"),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT).shift(LEFT * 2.25 + DOWN * 0.05)

        next_box = RoundedRectangle(corner_radius=0.12, width=4.65, height=1.35, stroke_color=EVIDENCE_GOLD, fill_color=EVIDENCE_GOLD, fill_opacity=0.16, stroke_width=3)
        next_text = VGroup(
            Text("次節", font_size=22, color=TEXT_GREY),
            Text("Evidence Approximation", font_size=25, color=EVIDENCE_GOLD),
            Text("積分を近似して使う", font_size=22, color=WHITE),
        ).arrange(DOWN, buff=0.10).move_to(next_box)
        next_group = VGroup(next_box, next_text).shift(RIGHT * 3.2 + DOWN * 0.05)
        arrow = Arrow(items.get_right() + RIGHT * 0.35, next_group.get_left() + LEFT * 0.1, color=EVIDENCE_GOLD, buff=0.1)

        self.play(FadeIn(label), Write(title))
        self.play(LaggedStart(*[FadeIn(item, shift=RIGHT * 0.15) for item in items], lag_ratio=0.16), run_time=1.6)
        self.play(Create(arrow), FadeIn(next_group))
        self.finish_narration(narration)

    def summary_item(self, number: str, main: str, sub: str) -> VGroup:
        badge = Circle(radius=0.22, color=EVIDENCE_GOLD, fill_color=EVIDENCE_GOLD, fill_opacity=0.85)
        number_text = Text(number, font_size=18, color=BLACK).move_to(badge)
        main_text = Text(main, font_size=25, color=WHITE)
        sub_text = Text(sub, font_size=20, color=TEXT_GREY)
        text = VGroup(main_text, sub_text).arrange(DOWN, buff=0.05, aligned_edge=LEFT)
        return VGroup(VGroup(badge, number_text), text).arrange(RIGHT, buff=0.22)
