from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


JAPANESE_FONT = "Noto Sans CJK JP"
BLUE_DATA = BLUE_C
SUCCESS_GREEN = GREEN_C
FAIL_RED = RED_C
MODEL_YELLOW = YELLOW_C
PRIOR_PURPLE = PURPLE_C
POST_ORANGE = ORANGE
TEXT_GREY = GREY_B

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def beta_pdf(mu: float, a: float, b: float) -> float:
    if mu <= 0.0 or mu >= 1.0:
        return 0.0
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return float(math.exp(log_norm + (a - 1.0) * math.log(mu) + (b - 1.0) * math.log(1.0 - mu)))


def bernoulli_likelihood(mu: float, successes: int, failures: int) -> float:
    if mu <= 0.0 or mu >= 1.0:
        if successes == 0 and mu == 0.0:
            return 1.0
        if failures == 0 and mu == 1.0:
            return 1.0
        return 0.0
    return float(mu**successes * (1.0 - mu) ** failures)


class PRML21BinaryVariables(Scene):
    """PRML 2.1 Binary Variables.

    Render example:
        uv run manim -pql prml_2_1_binary_variables.py PRML21BinaryVariables
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.observations = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

        self.binary_variable_intro()
        self.bernoulli_distribution()
        self.maximum_likelihood()
        self.small_data_instability()
        self.beta_prior_to_posterior()
        self.sequential_update()
        self.predictive_distribution()

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
        title = Text(text, font_size=font_size, color=WHITE)
        title.to_edge(UP).shift(DOWN * 0.35)
        return title

    def make_mu_axes(self, width: float = 6.8, height: float = 3.4, y_max: float = 5.0) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, y_max, y_max / 5],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def beta_curve(self, axes: Axes, a: float, b: float, color: ManimColor = POST_ORANGE) -> VMobject:
        ymax = axes.y_range[1]
        curve = axes.plot(
            lambda u: min(beta_pdf(u, a, b), ymax),
            x_range=[0.001, 0.999],
            color=color,
            use_smoothing=False,
        )
        curve.set_stroke(width=4)
        return curve

    def likelihood_curve(self, axes: Axes, successes: int, failures: int) -> VMobject:
        raw = np.array([bernoulli_likelihood(x, successes, failures) for x in np.linspace(0, 1, 300)])
        scale = 1.0 / raw.max()
        curve = axes.plot(
            lambda u: bernoulli_likelihood(u, successes, failures) * scale,
            x_range=[0, 1],
            color=MODEL_YELLOW,
            use_smoothing=False,
        )
        curve.set_stroke(width=4)
        return curve

    def observation_tokens(self, observations: list[int], radius: float = 0.19) -> VGroup:
        tokens = VGroup()
        for value in observations:
            token = Circle(radius=radius, color=SUCCESS_GREEN if value else FAIL_RED, fill_opacity=0.9)
            label = Text(str(value), font_size=22, color=BLACK)
            token.add(label.move_to(token.get_center()))
            tokens.add(token)
        tokens.arrange(RIGHT, buff=0.13)
        return tokens

    def binary_variable_intro(self) -> None:
        narration = self.start_narration("scene01")
        title = Text("PRML 2.1 Binary Variables", font_size=42)
        subtitle = Text("二値変数: 1 か 0 の確率モデル", font_size=30, color=TEXT_GREY).next_to(title, DOWN, buff=0.25)

        examples = VGroup(
            self.binary_card("coin", "表", "裏"),
            self.binary_card("test", "陽性", "陰性"),
            self.binary_card("click", "する", "しない"),
        ).arrange(RIGHT, buff=0.5)
        examples.next_to(subtitle, DOWN, buff=0.85)

        mapping = VGroup(
            MathTex(r"x=1", font_size=42, color=SUCCESS_GREEN),
            Text("成功 / yes / on", font_size=25),
            MathTex(r"x=0", font_size=42, color=FAIL_RED),
            Text("失敗 / no / off", font_size=25),
        ).arrange(RIGHT, buff=0.35)
        mapping.to_edge(DOWN).shift(UP * 0.75)

        self.play(FadeIn(title, shift=DOWN * 0.15), FadeIn(subtitle, shift=DOWN * 0.15))
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.2) for card in examples], lag_ratio=0.18), run_time=1.7)
        self.play(FadeIn(mapping, shift=UP * 0.2))
        self.wait(1.2)
        self.play(Indicate(mapping[0], color=SUCCESS_GREEN), Indicate(mapping[2], color=FAIL_RED))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, subtitle, examples, mapping)))

    def binary_card(self, title: str, yes: str, no: str) -> VGroup:
        box = RoundedRectangle(width=3.15, height=1.7, corner_radius=0.08, color=GREY_B, fill_color="#1f1f1f", fill_opacity=1)
        head = Text(title, font_size=24, color=WHITE).move_to(box.get_top() + DOWN * 0.38)
        yes_dot = Dot(color=SUCCESS_GREEN, radius=0.08)
        no_dot = Dot(color=FAIL_RED, radius=0.08)
        yes_text = Text(yes, font_size=24)
        no_text = Text(no, font_size=24)
        row = VGroup(
            VGroup(yes_dot, yes_text).arrange(RIGHT, buff=0.12),
            VGroup(no_dot, no_text).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.32)
        row.move_to(box.get_center() + DOWN * 0.22)
        return VGroup(box, head, row)

    def bernoulli_distribution(self) -> None:
        narration = self.start_narration("scene02")
        title = self.scene_title("Bernoulli 分布")
        label = self.section_label("PRML 2.1")
        formula = MathTex(r"p(x \mid \mu)=\mu^x(1-\mu)^{1-x}", font_size=44)
        formula.next_to(title, DOWN, buff=0.35)

        bars = self.bernoulli_bars(0.7).move_to(LEFT * 3.25 + DOWN * 0.35)
        slider, knob, mu_label = self.mu_slider(0.7)
        slider.move_to(RIGHT * 2.7 + DOWN * 1.75)
        knob.move_to(slider[0].point_from_proportion(0.7))
        mu_label.next_to(slider, UP, buff=0.32)

        case_one = VGroup(MathTex(r"x=1", font_size=36, color=SUCCESS_GREEN), MathTex(r"p(x=1\mid\mu)=\mu", font_size=36))
        case_zero = VGroup(MathTex(r"x=0", font_size=36, color=FAIL_RED), MathTex(r"p(x=0\mid\mu)=1-\mu", font_size=36))
        for case in (case_one, case_zero):
            case.arrange(RIGHT, buff=0.35)
        cases = VGroup(case_one, case_zero).arrange(DOWN, buff=0.35).move_to(RIGHT * 2.7 + DOWN * 0.25)

        self.play(FadeIn(label), FadeIn(title), Write(formula))
        self.play(FadeIn(bars), FadeIn(slider), FadeIn(knob), FadeIn(mu_label), FadeIn(cases))
        for mu in [0.25, 0.5, 0.85, 0.7]:
            new_bars = self.bernoulli_bars(mu).move_to(bars)
            new_knob = knob.copy().move_to(slider[0].point_from_proportion(mu))
            new_label = Text(f"mu = {mu:.2f}", font_size=24, color=MODEL_YELLOW).next_to(slider, UP, buff=0.32)
            self.play(Transform(bars, new_bars), Transform(knob, new_knob), Transform(mu_label, new_label), run_time=0.9)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, formula, bars, slider, knob, mu_label, cases)))

    def bernoulli_bars(self, mu: float) -> VGroup:
        chart = VGroup()
        baseline = Line(LEFT * 1.35, RIGHT * 1.35, color=GREY_B, stroke_width=2)
        chart.add(baseline)
        for index, (value, prob, color) in enumerate([(0, 1.0 - mu, FAIL_RED), (1, mu, SUCCESS_GREEN)]):
            x = -0.65 + index * 1.3
            height = 2.4 * prob
            bar = Rectangle(width=0.62, height=max(height, 0.04), color=color, fill_color=color, fill_opacity=0.85)
            bar.move_to(np.array([x, height / 2, 0.0]))
            value_label = MathTex(str(value), font_size=30).next_to(np.array([x, 0.0, 0.0]), DOWN, buff=0.22)
            prob_label = Text(f"{prob:.2f}", font_size=24, color=color).next_to(bar, UP, buff=0.12)
            chart.add(bar, value_label, prob_label)
        axis_label = Text("x", font_size=24, color=TEXT_GREY).next_to(baseline, RIGHT, buff=0.12)
        chart.add(axis_label)
        return chart

    def mu_slider(self, mu: float) -> tuple[VGroup, Dot, Text]:
        line = Line(LEFT * 1.7, RIGHT * 1.7, color=GREY_B, stroke_width=4)
        ticks = VGroup()
        for p in [0.0, 0.5, 1.0]:
            point = line.point_from_proportion(p)
            ticks.add(Line(point + DOWN * 0.09, point + UP * 0.09, color=GREY_B, stroke_width=3))
        labels = VGroup(MathTex("0", font_size=22), MathTex("0.5", font_size=22), MathTex("1", font_size=22))
        for label, p in zip(labels, [0.0, 0.5, 1.0]):
            label.next_to(line.point_from_proportion(p), DOWN, buff=0.16)
        slider = VGroup(line, ticks, labels)
        knob = Dot(line.point_from_proportion(mu), color=MODEL_YELLOW, radius=0.1)
        mu_label = Text(f"mu = {mu:.2f}", font_size=24, color=MODEL_YELLOW).next_to(slider, UP, buff=0.32)
        return slider, knob, mu_label

    def maximum_likelihood(self) -> None:
        narration = self.start_narration("scene03")
        title = self.scene_title("データから mu を推定する")
        label = self.section_label("Maximum likelihood")
        observations = self.observation_tokens(self.observations).next_to(title, DOWN, buff=0.6)
        counts = Text("N = 10,  m = 7", font_size=30, color=WHITE).next_to(observations, DOWN, buff=0.35)

        axes = self.make_mu_axes(width=6.8, height=3.0, y_max=1.05)
        axes.move_to(DOWN * 1.15)
        x_label = MathTex(r"\mu", font_size=28).next_to(axes.x_axis, RIGHT, buff=0.12)
        y_label = Text("normalized likelihood", font_size=20, color=TEXT_GREY).next_to(axes.y_axis, UP, buff=0.12)
        curve = self.likelihood_curve(axes, 7, 3)
        peak = Dot(axes.c2p(0.7, 1.0), color=MODEL_YELLOW, radius=0.08)
        peak_line = DashedLine(axes.c2p(0.7, 0.0), axes.c2p(0.7, 1.0), color=MODEL_YELLOW, stroke_width=3)
        mle = MathTex(r"\mu_{\mathrm{ML}}=\frac{m}{N}=\frac{7}{10}=0.7", font_size=38, color=MODEL_YELLOW)
        mle.to_edge(DOWN).shift(UP * 0.18)

        likelihood_formula = MathTex(r"p(\mathcal{D}\mid\mu)=\mu^m(1-\mu)^{N-m}", font_size=38)
        likelihood_formula.next_to(counts, DOWN, buff=0.28)

        self.play(FadeIn(label), FadeIn(title))
        self.play(LaggedStart(*[FadeIn(token, shift=UP * 0.15) for token in observations], lag_ratio=0.08), run_time=1.6)
        self.play(FadeIn(counts), Write(likelihood_formula))
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), Create(curve), run_time=1.7)
        self.play(Create(peak_line), FadeIn(peak), Write(mle))
        self.play(Indicate(peak, color=MODEL_YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, observations, counts, likelihood_formula, axes, x_label, y_label, curve, peak, peak_line, mle)))

    def small_data_instability(self) -> None:
        narration = self.start_narration("scene04")
        title = self.scene_title("データが少ないときの最尤推定")
        label = self.section_label("Why prior matters")

        few_tokens = self.observation_tokens([1, 1, 1], radius=0.23)
        few_tokens.move_to(LEFT * 3.1 + UP * 0.8)
        few_text = VGroup(
            Text("3 回中 3 回が 1", font_size=28),
            MathTex(r"\mu_{\mathrm{ML}}=1.0", font_size=36, color=MODEL_YELLOW),
        ).arrange(DOWN, buff=0.25).next_to(few_tokens, DOWN, buff=0.4)

        many_tokens = self.observation_tokens([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], radius=0.16)
        many_tokens.move_to(RIGHT * 3.1 + UP * 0.8)
        many_text = VGroup(
            Text("10 回中 7 回が 1", font_size=28),
            MathTex(r"\mu_{\mathrm{ML}}=0.7", font_size=36, color=MODEL_YELLOW),
        ).arrange(DOWN, buff=0.25).next_to(many_tokens, DOWN, buff=0.4)

        beta_title = Text("mu 自体の不確かさを分布で表す", font_size=28, color=WHITE).to_edge(DOWN).shift(UP * 1.35)
        beta_formula = MathTex(r"\mathrm{Beta}(\mu\mid a,b)", font_size=42, color=PRIOR_PURPLE).next_to(beta_title, DOWN, buff=0.24)

        self.play(FadeIn(label), FadeIn(title))
        self.play(FadeIn(few_tokens), FadeIn(few_text), run_time=1.1)
        self.play(FadeIn(many_tokens), FadeIn(many_text), run_time=1.1)
        self.play(Indicate(few_text[1], color=MODEL_YELLOW))
        self.play(FadeIn(beta_title, shift=UP * 0.15), Write(beta_formula))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, few_tokens, few_text, many_tokens, many_text, beta_title, beta_formula)))

    def beta_prior_to_posterior(self) -> None:
        narration = self.start_narration("scene05")
        title = self.scene_title("Beta 分布は Bernoulli の共役事前分布")
        label = self.section_label("Prior to posterior")

        axes = self.make_mu_axes(width=6.8, height=3.4, y_max=5.2)
        axes.move_to(LEFT * 2.6 + DOWN * 0.55)
        x_label = MathTex(r"\mu", font_size=28).next_to(axes.x_axis, RIGHT, buff=0.12)
        y_label = Text("density", font_size=20, color=TEXT_GREY).next_to(axes.y_axis, UP, buff=0.12)

        prior = self.beta_curve(axes, 2, 2, PRIOR_PURPLE)
        posterior = self.beta_curve(axes, 9, 5, POST_ORANGE)
        prior_label = Text("prior: Beta(2, 2)", font_size=24, color=PRIOR_PURPLE).next_to(axes, UP, buff=0.12).shift(LEFT * 0.8)
        posterior_label = Text("posterior: Beta(9, 5)", font_size=24, color=POST_ORANGE).next_to(prior_label, RIGHT, buff=0.35)

        update_formula = VGroup(
            MathTex(r"p(\mu)=\mathrm{Beta}(\mu\mid a,b)", font_size=33, color=PRIOR_PURPLE),
            MathTex(r"m=\sum_n x_n,\quad l=N-m", font_size=33),
            MathTex(r"p(\mu\mid\mathcal{D})=\mathrm{Beta}(\mu\mid a+m,b+l)", font_size=33, color=POST_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        update_formula.move_to(RIGHT * 3.0 + DOWN * 0.25)

        counts = VGroup(
            Text("a = 2", font_size=27, color=PRIOR_PURPLE),
            Text("b = 2", font_size=27, color=PRIOR_PURPLE),
            Text("m = 7", font_size=27, color=SUCCESS_GREEN),
            Text("l = 3", font_size=27, color=FAIL_RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        counts.next_to(update_formula, DOWN, buff=0.45).align_to(update_formula, LEFT)

        self.play(FadeIn(label), FadeIn(title), Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(Create(prior), FadeIn(prior_label), FadeIn(counts[0]), FadeIn(counts[1]))
        self.play(FadeIn(counts[2]), FadeIn(counts[3]), Write(update_formula), run_time=2.0)
        self.play(TransformFromCopy(prior, posterior), FadeIn(posterior_label), run_time=1.6)
        self.play(Indicate(update_formula[2], color=POST_ORANGE), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, x_label, y_label, prior, posterior, prior_label, posterior_label, update_formula, counts)))

    def sequential_update(self) -> None:
        narration = self.start_narration("scene06")
        title = self.scene_title("データを 1 個ずつ見ても更新できる")
        label = self.section_label("Sequential Bayesian update")

        axes = self.make_mu_axes(width=7.0, height=3.6, y_max=5.6)
        axes.move_to(DOWN * 0.55)
        x_label = MathTex(r"\mu", font_size=28).next_to(axes.x_axis, RIGHT, buff=0.12)
        y_label = Text("posterior density", font_size=20, color=TEXT_GREY).next_to(axes.y_axis, UP, buff=0.12)
        curve = self.beta_curve(axes, 2, 2, PRIOR_PURPLE)

        tokens = self.observation_tokens(self.observations, radius=0.17)
        tokens.next_to(title, DOWN, buff=0.5)
        counter = Text("a = 2, b = 2", font_size=30, color=WHITE).next_to(tokens, DOWN, buff=0.28)
        rule = MathTex(r"x=1:\ a\leftarrow a+1,\qquad x=0:\ b\leftarrow b+1", font_size=34)
        rule.to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(label), FadeIn(title), FadeIn(tokens), FadeIn(counter), Create(axes), FadeIn(x_label), FadeIn(y_label), Create(curve), Write(rule))
        a = 2
        b = 2
        current_curve = curve
        for index, value in enumerate(self.observations[:7]):
            if value:
                a += 1
            else:
                b += 1
            new_curve = self.beta_curve(axes, a, b, POST_ORANGE if index >= 5 else MODEL_YELLOW)
            new_counter = Text(f"a = {a}, b = {b}", font_size=30, color=WHITE).move_to(counter)
            self.play(
                tokens[index].animate.scale(1.18).set_stroke(WHITE, width=3),
                Transform(current_curve, new_curve),
                Transform(counter, new_counter),
                run_time=0.65,
            )
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, x_label, y_label, current_curve, tokens, counter, rule)))

    def predictive_distribution(self) -> None:
        narration = self.start_narration("scene07")
        title = self.scene_title("次の 1 回を予測する")
        label = self.section_label("Predictive distribution")

        formula = MathTex(
            r"p(x=1\mid\mathcal{D})=\mathbb{E}[\mu]=\frac{m+a}{N+a+b}",
            font_size=43,
            color=POST_ORANGE,
        )
        formula.next_to(title, DOWN, buff=0.55)

        observations = self.observation_tokens(self.observations)
        observations.next_to(formula, DOWN, buff=0.55)
        calc = MathTex(r"\frac{7+2}{10+2+2}=\frac{9}{14}\approx 0.64", font_size=42, color=MODEL_YELLOW)
        calc.next_to(observations, DOWN, buff=0.45)

        comparison = VGroup(
            VGroup(Text("最尤推定", font_size=27, color=TEXT_GREY), MathTex(r"7/10=0.70", font_size=36, color=MODEL_YELLOW)).arrange(DOWN, buff=0.2),
            VGroup(Text("ベイズ予測", font_size=27, color=TEXT_GREY), MathTex(r"9/14\approx0.64", font_size=36, color=POST_ORANGE)).arrange(DOWN, buff=0.2),
        ).arrange(RIGHT, buff=0.9)
        comparison.to_edge(DOWN).shift(UP * 0.45)

        bridge = VGroup(
            Text("Binary", font_size=28, color=SUCCESS_GREEN),
            MathTex(r"x\in\{0,1\}", font_size=30),
            Arrow(LEFT, RIGHT, color=TEXT_GREY),
            Text("Multinomial", font_size=28, color=BLUE_DATA),
            MathTex(r"x\in\{1,\ldots,K\}", font_size=30),
        ).arrange(RIGHT, buff=0.25)
        bridge.move_to(DOWN * 0.7)

        self.play(FadeIn(label), FadeIn(title), Write(formula))
        self.play(FadeIn(observations), Write(calc))
        self.play(FadeIn(comparison, shift=UP * 0.2))
        self.wait(0.7)
        self.play(FadeOut(observations), FadeOut(calc), FadeOut(comparison), FadeIn(bridge, shift=UP * 0.2))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, formula, bridge)))
