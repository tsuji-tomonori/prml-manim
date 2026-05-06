from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


JAPANESE_FONT = "Noto Sans CJK JP"
BG = "#101010"
TEXT_GREY = GREY_B
ACCENT_BLUE = BLUE_C
ACCENT_GREEN = GREEN_C
ACCENT_RED = RED_C
ACCENT_ORANGE = ORANGE
ACCENT_PURPLE = PURPLE_C
ACCENT_YELLOW = YELLOW_C

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def normal_pdf(x: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


class PRML24ExponentialFamily(Scene):
    """PRML 2.4 explanatory video: The Exponential Family.

    Render example:
        uv run manim --disable_caching --flush_cache -ql prml_2_4_exponential_family.py PRML24ExponentialFamily
    """

    def construct(self) -> None:
        self.camera.background_color = BG

        self.opening()
        self.standard_form()
        self.bernoulli_example()
        self.gaussian_example()
        self.sufficient_statistics()
        self.maximum_likelihood()
        self.conjugate_prior()
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
        title = Text(text, font_size=font_size, color=WHITE)
        title.to_edge(UP).shift(DOWN * 0.35)
        return title

    def clear_scene(self) -> None:
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.45)
        self.clear()

    def named_box(self, title: str, subtitle: str, color: ManimColor, width: float = 3.2, height: float = 1.55) -> VGroup:
        box = RoundedRectangle(width=width, height=height, corner_radius=0.12, color=color, stroke_width=2.5)
        title_mob = Text(title, font_size=26, color=color).move_to(box.get_center() + UP * 0.28)
        subtitle_mob = Text(subtitle, font_size=20, color=TEXT_GREY).move_to(box.get_center() + DOWN * 0.32)
        return VGroup(box, title_mob, subtitle_mob)

    def opening(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 2.4 The Exponential Family")
        title = Text("指数型分布族", font_size=44, color=WHITE).to_edge(UP, buff=0.75)
        subtitle = Text("違う分布を、同じ式の形で見る", font_size=28, color=TEXT_GREY)
        subtitle.next_to(title, DOWN, buff=0.22)

        distributions = VGroup(
            self.named_box("Bernoulli", "二値変数", ACCENT_BLUE),
            self.named_box("Multinomial", "カテゴリ変数", ACCENT_ORANGE),
            self.named_box("Gaussian", "連続値", ACCENT_GREEN),
        ).arrange(RIGHT, buff=0.35).shift(UP * 0.1)

        formula = MathTex(
            r"p(x|\eta)=h(x)g(\eta)\exp\{\eta^{\mathrm T}u(x)\}",
            font_size=40,
            color=WHITE,
        ).to_edge(DOWN, buff=1.05)
        brace = Brace(formula, UP, color=ACCENT_YELLOW)
        brace_text = Text("共通の型", font_size=24, color=ACCENT_YELLOW).next_to(brace, UP, buff=0.12)

        self.play(FadeIn(label), Write(title), FadeIn(subtitle), run_time=1.2)
        self.play(FadeIn(distributions, lag_ratio=0.2), run_time=1.6)
        self.play(*[box.animate.shift(DOWN * 0.45).scale(0.88) for box in distributions], run_time=1.0)
        self.play(Write(formula), GrowFromCenter(brace), FadeIn(brace_text), run_time=1.5)
        self.wait(0.8)
        self.finish_narration(narration)
        self.clear_scene()

    def standard_form(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("PRML 2.4 / Eq. (2.194)")
        title = self.scene_title("標準形は、三つの部品を見る", font_size=34)

        formula = MathTex(
            r"p(x|\eta)",
            r"=",
            r"h(x)",
            r"g(\eta)",
            r"\exp",
            r"\{",
            r"\eta^{\mathrm T}",
            r"u(x)",
            r"\}",
            font_size=46,
        ).shift(UP * 1.1)

        explanations = VGroup(
            self.named_box("h(x)", "データだけの土台", ACCENT_BLUE, width=2.85, height=1.35),
            self.named_box("u(x)", "十分統計量", ACCENT_GREEN, width=2.85, height=1.35),
            self.named_box("η", "自然パラメータ", ACCENT_RED, width=2.85, height=1.35),
            self.named_box("g(η)", "正規化係数", ACCENT_PURPLE, width=2.85, height=1.35),
        ).arrange(RIGHT, buff=0.22).shift(DOWN * 0.75)

        highlights = [
            (SurroundingRectangle(formula[2], color=ACCENT_BLUE, buff=0.08), explanations[0]),
            (SurroundingRectangle(formula[7], color=ACCENT_GREEN, buff=0.08), explanations[1]),
            (SurroundingRectangle(formula[6], color=ACCENT_RED, buff=0.08), explanations[2]),
            (SurroundingRectangle(formula[3], color=ACCENT_PURPLE, buff=0.08), explanations[3]),
        ]
        core = Text("指数の中: パラメータ × データの特徴量", font_size=29, color=ACCENT_YELLOW)
        core.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title), Write(formula), run_time=1.8)
        for rect, box in highlights:
            self.play(Create(rect), FadeIn(box), run_time=0.9)
            self.wait(0.25)
            self.play(FadeOut(rect), run_time=0.35)
        self.play(Write(core), run_time=1.0)
        self.wait(0.9)
        self.finish_narration(narration)
        self.clear_scene()

    def make_bernoulli_bars(self, mu: float) -> VGroup:
        base_y = -1.35
        max_height = 2.6
        bars = VGroup()
        for i, prob in enumerate([1.0 - mu, mu]):
            x = -1.0 + 2.0 * i
            rect = Rectangle(width=0.75, height=max_height * prob, color=ACCENT_BLUE if i == 0 else ACCENT_ORANGE)
            rect.set_fill(rect.get_color(), opacity=0.7)
            rect.move_to(np.array([x, base_y + rect.height / 2.0, 0.0]))
            label = MathTex(str(i), font_size=30).move_to(np.array([x, base_y - 0.32, 0.0]))
            value = MathTex(f"{prob:.1f}", font_size=26, color=rect.get_color()).next_to(rect, UP, buff=0.12)
            bars.add(VGroup(rect, label, value))
        axis = Line(np.array([-1.8, base_y, 0.0]), np.array([1.8, base_y, 0.0]), color=GREY_B)
        return VGroup(axis, bars)

    def bernoulli_example(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("PRML 2.4 / Bernoulli distribution")
        title = self.scene_title("ベルヌーイ分布は log-odds で直線化する", font_size=32)

        original = MathTex(r"p(x|\mu)=\mu^x(1-\mu)^{1-x}", font_size=42)
        rewritten = MathTex(
            r"p(x|\mu)=(1-\mu)\exp\left\{x\ln\frac{\mu}{1-\mu}\right\}",
            font_size=38,
        )
        eta = MathTex(r"\eta=\ln\frac{\mu}{1-\mu}", r",\quad u(x)=x", font_size=38, color=ACCENT_YELLOW)
        formulas = VGroup(original, rewritten, eta).arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        formulas.to_edge(LEFT, buff=0.65).shift(UP * 0.25)

        bars = self.make_bernoulli_bars(0.2).shift(RIGHT * 3.45 + UP * 0.1)
        mu_label = MathTex(r"\mu=0.2", font_size=34, color=ACCENT_ORANGE).next_to(bars, UP, buff=0.32)
        note = Text("μ が大きいほど x=1 が出やすい", font_size=24, color=TEXT_GREY)
        note.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(label), Write(title))
        self.play(Write(original), run_time=1.1)
        self.play(TransformFromCopy(original, rewritten), run_time=1.5)
        self.play(Write(eta), run_time=1.2)
        self.play(FadeIn(bars), Write(mu_label), Write(note), run_time=1.0)
        current_bars = bars
        current_label = mu_label
        for mu in [0.5, 0.8, 0.35, 0.7]:
            next_bars = self.make_bernoulli_bars(mu).shift(RIGHT * 3.45 + UP * 0.1)
            next_label = MathTex(fr"\mu={mu:.2g}", font_size=34, color=ACCENT_ORANGE).move_to(current_label)
            self.play(Transform(current_bars, next_bars), Transform(current_label, next_label), run_time=0.8)
        self.wait(0.6)
        self.finish_narration(narration)
        self.clear_scene()

    def gaussian_example(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("PRML 2.4 / Gaussian distribution")
        title = self.scene_title("ガウス分布では x と x² が十分統計量になる", font_size=32)

        axes = Axes(
            x_range=[-4.0, 4.0, 1.0],
            y_range=[0.0, 0.5, 0.1],
            x_length=5.4,
            y_length=3.0,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 3.0 + DOWN * 0.15)
        curve = axes.plot(lambda x: normal_pdf(x, mu=-1.2, sigma=1.0), x_range=[-4, 4], color=ACCENT_GREEN)
        curve.set_stroke(width=4)
        mu_label = MathTex(r"\mu=-1.2,\ \sigma^2=1", font_size=31, color=ACCENT_GREEN)
        mu_label.next_to(axes, UP, buff=0.25)

        original = MathTex(
            r"\mathcal{N}(x|\mu,\sigma^2)",
            r"=",
            r"\frac{1}{(2\pi\sigma^2)^{1/2}}",
            r"\exp\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}",
            font_size=34,
        )
        expanded = MathTex(
            r"\exp\left\{",
            r"\frac{\mu}{\sigma^2}x",
            r"-",
            r"\frac{1}{2\sigma^2}x^2",
            r"-",
            r"\frac{\mu^2}{2\sigma^2}",
            r"\right\}",
            font_size=34,
        )
        natural = MathTex(
            r"u(x)=(x,x^2)",
            r",\quad",
            r"\eta=\left(\frac{\mu}{\sigma^2},-\frac{1}{2\sigma^2}\right)",
            font_size=33,
            color=ACCENT_YELLOW,
        )
        formulas = VGroup(original, expanded, natural).arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        formulas.scale(0.9).shift(RIGHT * 2.35 + UP * 0.15)

        self.play(FadeIn(label), Write(title), Create(axes), Create(curve), Write(mu_label))
        self.play(FadeIn(original, shift=UP * 0.08), run_time=0.9)
        self.play(FadeIn(expanded, shift=UP * 0.08), run_time=0.9)
        self.play(FadeIn(natural, shift=UP * 0.08), run_time=0.9)
        current_curve = curve
        current_label = mu_label
        for mu in [-0.2, 1.0, 0.3]:
            next_curve = axes.plot(lambda x, m=mu: normal_pdf(x, mu=m, sigma=1.0), x_range=[-4, 4], color=ACCENT_GREEN)
            next_curve.set_stroke(width=4)
            next_label = MathTex(fr"\mu={mu:.1f},\ \sigma^2=1", font_size=31, color=ACCENT_GREEN).move_to(current_label)
            self.play(Transform(current_curve, next_curve), Transform(current_label, next_label), run_time=1.0)
        self.wait(0.7)
        self.finish_narration(narration)
        self.clear_scene()

    def sufficient_statistics(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("PRML 2.4 / sufficient statistics")
        title = self.scene_title("データは十分統計量の和として効く", font_size=34)

        bern_data = [1, 0, 1, 1, 0, 1, 0, 1]
        coins = VGroup()
        for value in bern_data:
            circle = Circle(radius=0.22, color=ACCENT_ORANGE if value else ACCENT_BLUE)
            circle.set_fill(circle.get_color(), opacity=0.65)
            text = MathTex(str(value), font_size=24).move_to(circle)
            coins.add(VGroup(circle, text))
        coins.arrange(RIGHT, buff=0.12).shift(UP * 1.35)
        bern_eq = MathTex(r"\sum_n x_n=5", r"\quad \Rightarrow \quad", r"\bar{x}=5/8", font_size=38, color=ACCENT_YELLOW)
        bern_eq.next_to(coins, DOWN, buff=0.35)

        xs = [-1.4, -0.8, -0.1, 0.2, 0.7, 1.1]
        number_line = NumberLine(x_range=[-2, 2, 1], length=5.4, color=GREY_B).shift(DOWN * 1.1 + LEFT * 2.3)
        dots = VGroup(*[Dot(number_line.n2p(x), color=ACCENT_GREEN, radius=0.06) for x in xs])
        gauss_stats = MathTex(
            r"\sum_n x_n",
            r"\quad",
            r"\sum_n x_n^2",
            font_size=39,
            color=ACCENT_GREEN,
        ).shift(DOWN * 1.1 + RIGHT * 3.0)
        arrow = Arrow(number_line.get_right() + RIGHT * 0.2, gauss_stats.get_left() + LEFT * 0.2, buff=0.05, color=GREY_A)

        caption = Text("十分統計量 = 推定に必要なデータの要約", font_size=29, color=WHITE)
        caption.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(coins, lag_ratio=0.12), run_time=1.2)
        self.play(Write(bern_eq), run_time=1.1)
        self.wait(0.5)
        self.play(Create(number_line), FadeIn(dots, lag_ratio=0.1), GrowArrow(arrow), Write(gauss_stats), run_time=1.5)
        self.play(Write(caption), run_time=1.0)
        self.wait(0.7)
        self.finish_narration(narration)
        self.clear_scene()

    def maximum_likelihood(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("PRML 2.4 / maximum likelihood")
        title = self.scene_title("最尤推定は、十分統計量の平均を合わせる", font_size=32)

        condition = MathTex(
            r"-\nabla \ln g(\eta)",
            r"=",
            r"\frac{1}{N}\sum_{n=1}^{N}u(x_n)",
            font_size=44,
        ).shift(UP * 1.55)
        left_label = Text("モデル側の平均", font_size=24, color=ACCENT_GREEN).next_to(condition[0], DOWN, buff=0.22)
        right_label = Text("データ側の平均", font_size=24, color=ACCENT_ORANGE).next_to(condition[2], DOWN, buff=0.22)

        balance = VGroup()
        beam = Line(LEFT * 2.7, RIGHT * 2.7, color=GREY_A, stroke_width=5)
        pivot = Triangle(color=GREY_A).scale(0.35).rotate(PI).next_to(beam, DOWN, buff=0.0)
        left_pan = Circle(radius=0.45, color=ACCENT_GREEN).set_fill(ACCENT_GREEN, opacity=0.2).move_to(beam.get_left() + DOWN * 0.7)
        right_pan = Circle(radius=0.45, color=ACCENT_ORANGE).set_fill(ACCENT_ORANGE, opacity=0.2).move_to(beam.get_right() + DOWN * 0.7)
        balance.add(beam, pivot, left_pan, right_pan)
        balance.shift(DOWN * 0.25)

        bern = MathTex(r"\text{Bernoulli:}\quad \mu_{\mathrm{ML}}=\frac{1}{N}\sum_n x_n", font_size=38, color=ACCENT_YELLOW)
        bern.to_edge(DOWN, buff=0.65)
        data = Text("例: 8回中5回が1なら μML = 0.625", font_size=27, color=TEXT_GREY)
        data.next_to(bern, UP, buff=0.28)

        self.play(FadeIn(label), Write(title), Write(condition), run_time=1.5)
        self.play(FadeIn(left_label), FadeIn(right_label), run_time=0.8)
        self.play(FadeIn(balance), run_time=1.0)
        self.play(beam.animate.rotate(0.18), left_pan.animate.shift(UP * 0.2), right_pan.animate.shift(DOWN * 0.2), run_time=0.7)
        self.play(beam.animate.rotate(-0.18), left_pan.animate.shift(DOWN * 0.2), right_pan.animate.shift(UP * 0.2), run_time=0.7)
        self.play(Write(data), Write(bern), run_time=1.2)
        self.wait(0.8)
        self.finish_narration(narration)
        self.clear_scene()

    def conjugate_prior(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("PRML 2.4 / conjugate prior")
        title = self.scene_title("共役事前分布は、更新を足し算にする", font_size=33)

        prior = MathTex(
            r"p(\eta|\chi,\nu)",
            r"\propto",
            r"g(\eta)^\nu",
            r"\exp\{\nu\eta^{\mathrm T}\chi\}",
            font_size=41,
        ).shift(UP * 1.55)

        prior_box = self.named_box("事前", "νχ, ν", ACCENT_PURPLE, width=2.3, height=1.35)
        data_box = self.named_box("データ", "Σu(xn), N", ACCENT_ORANGE, width=2.7, height=1.35)
        post_box = self.named_box("事後", "νχ+Σu(xn), ν+N", ACCENT_GREEN, width=3.4, height=1.35)
        plus = MathTex("+", font_size=46)
        arrow = MathTex(r"\longrightarrow", font_size=46)
        flow = VGroup(prior_box, plus, data_box, arrow, post_box).arrange(RIGHT, buff=0.25).shift(DOWN * 0.35)

        update = MathTex(
            r"\chi'=\frac{\nu\chi+\sum_n u(x_n)}{\nu+N}",
            r",\quad",
            r"\nu'=\nu+N",
            font_size=36,
            color=ACCENT_YELLOW,
        ).to_edge(DOWN, buff=0.65)

        self.play(FadeIn(label), Write(title), Write(prior), run_time=1.5)
        self.play(FadeIn(prior_box), run_time=0.6)
        self.play(Write(plus), FadeIn(data_box), run_time=0.8)
        self.play(Write(arrow), FadeIn(post_box), run_time=0.9)
        self.play(Write(update), run_time=1.2)
        self.wait(0.8)
        self.finish_narration(narration)
        self.clear_scene()

    def summary(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("PRML 2.4 summary")
        title = Text("2.4 指数型分布族の要点", font_size=40, color=WHITE).to_edge(UP, buff=0.65)

        points = VGroup(
            self.named_box("枠組み", "多くの分布を同じ形で扱う", ACCENT_BLUE, width=4.2, height=1.25),
            self.named_box("三つの部品", "η, u(x), g(η)", ACCENT_YELLOW, width=4.2, height=1.25),
            self.named_box("推定", "十分統計量の平均を合わせる", ACCENT_GREEN, width=4.2, height=1.25),
            self.named_box("ベイズ更新", "共役事前分布で足し算になる", ACCENT_PURPLE, width=4.2, height=1.25),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.35, 0.35)).shift(UP * 0.1)

        formula = MathTex(
            r"p(x|\eta)=h(x)g(\eta)\exp\{\eta^{\mathrm T}u(x)\}",
            font_size=38,
            color=WHITE,
        ).to_edge(DOWN, buff=0.65)

        next_note = Text("次: 有限個のパラメータに固定しないノンパラメトリック手法へ", font_size=25, color=TEXT_GREY)
        next_note.next_to(formula, UP, buff=0.25)

        self.play(FadeIn(label), Write(title), run_time=1.0)
        self.play(FadeIn(points, lag_ratio=0.18), run_time=1.8)
        self.play(Write(next_note), Write(formula), run_time=1.4)
        self.wait(0.8)
        self.finish_narration(narration)
        self.clear_scene()
