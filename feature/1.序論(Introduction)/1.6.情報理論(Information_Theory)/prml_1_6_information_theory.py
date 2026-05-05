from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


INFO_BLUE = BLUE_C
MODEL_RED = RED_C
TRUE_GREEN = GREEN_C
KL_PURPLE = PURPLE_C
WARN_ORANGE = ORANGE
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def entropy(probabilities: np.ndarray, base: float = math.e) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)) / math.log(base))


def gaussian_pdf(x: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    z = (np.asarray(x) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


class PRML16InformationTheory(Scene):
    """PRML 1.6 information theory overview.

    Render example:
        uv run manim -pql prml_1_6_information_theory.py PRML16InformationTheory
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"

        self.information_as_surprise()
        self.entropy_as_average_code_length()
        self.entropy_as_spread()
        self.differential_entropy_gaussian()
        self.conditional_entropy()
        self.kl_divergence()
        self.kl_and_likelihood()
        self.mutual_information()

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

    def probability_bars(
        self,
        values: list[float] | np.ndarray,
        labels: list[str],
        width: float = 5.2,
        height: float = 2.4,
        color: ManimColor = INFO_BLUE,
    ) -> VGroup:
        values = np.asarray(values, dtype=float)
        bar_width = width / len(values) * 0.72
        gap = width / len(values)
        base_y = -height / 2
        group = VGroup()
        bars = VGroup()
        label_group = VGroup()
        value_group = VGroup()
        max_value = max(float(np.max(values)), 1e-6)
        for i, value in enumerate(values):
            x = -width / 2 + gap * (i + 0.5)
            bar_height = max(0.025, height * float(value) / max_value)
            bar = Rectangle(width=bar_width, height=bar_height)
            bar.set_fill(color, opacity=0.72)
            bar.set_stroke(color, width=1.0)
            bar.move_to(np.array([x, base_y + bar_height / 2, 0.0]))
            bars.add(bar)
            label_group.add(Text(labels[i], font_size=18, color=TEXT_GREY).move_to([x, base_y - 0.28, 0.0]))
            value_group.add(Text(f"{value:.2f}", font_size=16, color=WHITE).move_to([x, base_y + bar_height + 0.2, 0.0]))
        baseline = Line(LEFT * width / 2, RIGHT * width / 2, color=GREY_B).move_to([0, base_y, 0])
        group.add(baseline, bars, label_group, value_group)
        return group

    def information_as_surprise(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 1.6 / 式 (1.92)")
        title = self.scene_title("情報量は「どれくらい驚いたか」", font_size=34)

        formula = MathTex(r"h(x)=-\log_2 p(x)", font_size=48, color=WHITE)
        formula.to_edge(DOWN, buff=0.45)

        bar_values = [0.80, 0.25, 0.05]
        captions = ["よく起きる", "たまに起きる", "めったにない"]
        bars = []
        info_cards = []
        for p, caption in zip(bar_values, captions):
            bar = self.probability_bars([p], ["p(x)"], width=2.0, height=2.7, color=INFO_BLUE).shift(LEFT * 3.0)
            prob_text = Text(f"{caption}:  p(x)={p:.2f}", font_size=28, color=INFO_BLUE)
            prob_text.next_to(bar, UP, buff=0.28)
            info = -math.log2(p)
            card_box = RoundedRectangle(width=4.2, height=1.5, corner_radius=0.12, color=KL_PURPLE)
            card_text = VGroup(
                Text("受け取る情報量", font_size=24, color=TEXT_GREY),
                MathTex(f"h(x)={info:.2f}\\;\\mathrm{{bits}}", font_size=36, color=KL_PURPLE),
            ).arrange(DOWN, buff=0.18)
            card_text.move_to(card_box)
            card = VGroup(card_box, card_text).shift(RIGHT * 2.5)
            bars.append(VGroup(bar, prob_text))
            info_cards.append(card)

        axis = Axes(
            x_range=[0, 1.0, 0.25],
            y_range=[0, 5, 1],
            x_length=4.1,
            y_length=2.4,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 2.5 + DOWN * 1.1)
        curve = axis.plot(lambda u: -np.log2(max(u, 0.01)), x_range=[0.04, 1.0], color=KL_PURPLE)
        curve.set_stroke(width=4)
        axis_label = Text("p が小さいほど h は大きい", font_size=20, color=TEXT_GREY)
        axis_label.next_to(axis, UP, buff=0.08)

        self.play(FadeIn(label), Write(title), Write(formula))
        self.play(FadeIn(bars[0]), FadeIn(info_cards[0]), Create(axis), Create(curve), FadeIn(axis_label))
        for i in [1, 2]:
            self.play(Transform(bars[0], bars[i]), Transform(info_cards[0], info_cards[i]), run_time=1.1)
            self.wait(0.5)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(formula), FadeOut(bars[0]), FadeOut(info_cards[0]), FadeOut(axis), FadeOut(curve), FadeOut(axis_label))

    def entropy_as_average_code_length(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("PRML 1.6 / 式 (1.93)")
        title = self.scene_title("エントロピーは、平均して何ビット必要か", font_size=33)

        uniform = np.ones(8) / 8
        skewed = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 64, 1 / 64, 1 / 64, 1 / 64])
        labels = list("abcdefgh")
        left = self.probability_bars(uniform, labels, width=4.55, height=2.0, color=INFO_BLUE).shift(LEFT * 3.25 + DOWN * 0.15)
        right = self.probability_bars(skewed, labels, width=4.55, height=2.0, color=TRUE_GREEN).shift(RIGHT * 2.05 + DOWN * 0.15)
        left_title = Text("一様: 8通り", font_size=25, color=INFO_BLUE).next_to(left, UP, buff=0.22)
        right_title = Text("非一様: よく出る値に短い符号", font_size=23, color=TRUE_GREEN).next_to(right, UP, buff=0.22)
        left_h = MathTex(r"H=3\ \mathrm{bits}", font_size=34, color=INFO_BLUE).next_to(left, DOWN, buff=0.55)
        right_h = MathTex(r"H=2\ \mathrm{bits}", font_size=34, color=TRUE_GREEN).next_to(right, DOWN, buff=0.55)

        code_words = VGroup(
            Text("a: 0", font_size=19),
            Text("b: 10", font_size=19),
            Text("c: 110", font_size=19),
            Text("d: 1110", font_size=19),
            Text("e-h: 111100 ...", font_size=19),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        code_box = SurroundingRectangle(code_words, color=TRUE_GREEN, buff=0.18)
        code_group = VGroup(code_box, code_words).scale(0.84).next_to(right, RIGHT, buff=0.18)

        formula = MathTex(r"H[x]=-\sum_x p(x)\log_2 p(x)", font_size=38)
        formula.to_edge(DOWN, buff=0.25)

        self.play(FadeIn(label), Write(title), Write(formula))
        self.play(FadeIn(left), Write(left_title), Write(left_h), run_time=1.0)
        self.play(FadeIn(right), Write(right_title), Write(right_h), FadeIn(code_group), run_time=1.2)
        self.play(Indicate(right_h, scale_factor=1.12), Indicate(code_group, scale_factor=1.03), run_time=1.1)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(formula), FadeOut(left), FadeOut(right), FadeOut(left_title), FadeOut(right_title), FadeOut(left_h), FadeOut(right_h), FadeOut(code_group))

    def entropy_as_spread(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("PRML 1.6 / Fig. 1.30")
        title = self.scene_title("広がった分布ほど、結果は読みにくい", font_size=34)

        x = np.arange(30)
        narrow = np.exp(-0.5 * ((x - 10) / 2.2) ** 2)
        broad = np.exp(-0.5 * ((x - 14) / 7.0) ** 2)
        narrow = narrow / narrow.sum()
        broad = broad / broad.sum()

        labels = [str(i + 1) if i in [0, 9, 19, 29] else "" for i in range(30)]
        narrow_bars = self.probability_bars(narrow, labels, width=9.0, height=2.65, color=WARN_ORANGE).shift(DOWN * 0.1)
        broad_bars = self.probability_bars(broad, labels, width=9.0, height=2.65, color=INFO_BLUE).shift(DOWN * 0.1)
        h_narrow = entropy(narrow)
        h_broad = entropy(broad)
        h_text = MathTex(f"H={h_narrow:.2f}", font_size=38, color=WARN_ORANGE).to_edge(DOWN, buff=0.55)
        note = Text("一部の値に集中: 予想しやすい", font_size=27, color=WARN_ORANGE).next_to(h_text, UP, buff=0.25)

        h_text_broad = MathTex(f"H={h_broad:.2f}", font_size=38, color=INFO_BLUE).move_to(h_text)
        note_broad = Text("多くの値に広がる: 予想しにくい", font_size=27, color=INFO_BLUE).move_to(note)
        max_note = Text("同じ状態数なら、一様分布が最大エントロピー", font_size=25, color=TRUE_GREEN)
        max_note.next_to(h_text, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(narrow_bars), Write(h_text), Write(note), run_time=1.2)
        self.wait(0.8)
        self.play(Transform(narrow_bars, broad_bars), Transform(h_text, h_text_broad), Transform(note, note_broad), run_time=1.7)
        self.play(Write(max_note), run_time=0.9)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(narrow_bars), FadeOut(h_text), FadeOut(note), FadeOut(max_note))

    def differential_entropy_gaussian(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("PRML 1.6 / 式 (1.104)-(1.110)")
        title = self.scene_title("連続値では、ガウス分布が最大エントロピーになる", font_size=31)

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.85, 0.2],
            x_length=7.2,
            y_length=3.3,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 0.55 + DOWN * 0.1)

        sigmas = [0.55, 1.0, 1.65]
        curves = [
            axes.plot(lambda u, s=sigma: gaussian_pdf(u, sigma=s), x_range=[-4, 4], color=INFO_BLUE)
            for sigma in sigmas
        ]
        for curve in curves:
            curve.set_stroke(width=4)

        sigma_text = MathTex(r"\sigma=0.55", font_size=34, color=INFO_BLUE).next_to(axes, RIGHT, buff=0.35).shift(UP * 0.95)
        h_value = 0.5 * (1.0 + math.log(2.0 * math.pi * sigmas[0] ** 2))
        entropy_text = MathTex(f"H={h_value:.2f}", font_size=34, color=TRUE_GREEN).next_to(sigma_text, DOWN, buff=0.22)
        formula = MathTex(r"H[x]=-\int p(x)\ln p(x)\,dx", font_size=34).to_edge(DOWN, buff=0.55)
        gaussian_formula = MathTex(r"H_{\mathcal{N}}[x]=\frac12\{1+\ln(2\pi\sigma^2)\}", font_size=32, color=TRUE_GREEN)
        gaussian_formula.next_to(formula, UP, buff=0.2)
        constraint = Text("平均と分散だけを固定すると、最大は Gaussian", font_size=25, color=TEXT_GREY)
        constraint.next_to(gaussian_formula, UP, buff=0.2)

        self.play(FadeIn(label), Write(title), Create(axes), Write(formula))
        self.play(Create(curves[0]), Write(sigma_text), Write(entropy_text), Write(constraint), Write(gaussian_formula))
        current_curve = curves[0]
        for sigma, curve in zip(sigmas[1:], curves[1:]):
            next_sigma = MathTex(fr"\sigma={sigma:.2f}", font_size=34, color=INFO_BLUE).move_to(sigma_text)
            next_h = 0.5 * (1.0 + math.log(2.0 * math.pi * sigma**2))
            next_entropy = MathTex(f"H={next_h:.2f}", font_size=34, color=TRUE_GREEN).move_to(entropy_text)
            self.play(Transform(current_curve, curve), Transform(sigma_text, next_sigma), Transform(entropy_text, next_entropy), run_time=1.2)
            self.wait(0.4)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(current_curve), FadeOut(sigma_text), FadeOut(entropy_text), FadeOut(formula), FadeOut(gaussian_formula), FadeOut(constraint))

    def conditional_entropy(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("PRML 1.6 / 式 (1.111)-(1.112)")
        title = self.scene_title("x を知ると、y の残りの不確かさが変わる", font_size=32)

        x_box = RoundedRectangle(width=2.1, height=1.05, corner_radius=0.12, color=INFO_BLUE)
        x_text = Text("x を観測", font_size=27, color=INFO_BLUE).move_to(x_box)
        x_group = VGroup(x_box, x_text).shift(LEFT * 3.8 + UP * 0.45)

        before_points = VGroup()
        rng = np.random.default_rng(8)
        for _ in range(36):
            before_points.add(Dot([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0], radius=0.045, color=WARN_ORANGE))
        before_panel = RoundedRectangle(width=2.7, height=2.4, corner_radius=0.1, color=WARN_ORANGE)
        before_points.move_to(before_panel)
        before_title = Text("y: 候補が広い", font_size=23, color=WARN_ORANGE).next_to(before_panel, UP, buff=0.15)
        before = VGroup(before_panel, before_points, before_title).shift(LEFT * 0.75 + DOWN * 0.15)

        after_points = VGroup()
        for _ in range(16):
            after_points.add(Dot([rng.normal(0, 0.18), rng.normal(0, 0.18), 0], radius=0.052, color=TRUE_GREEN))
        after_panel = RoundedRectangle(width=2.7, height=2.4, corner_radius=0.1, color=TRUE_GREEN)
        after_points.move_to(after_panel)
        after_title = Text("x の後: 候補が絞れる", font_size=23, color=TRUE_GREEN).next_to(after_panel, UP, buff=0.15)
        after = VGroup(after_panel, after_points, after_title).shift(RIGHT * 3.0 + DOWN * 0.15)

        arrow = Arrow(before.get_right() + RIGHT * 0.15, after.get_left() + LEFT * 0.15, buff=0.05, color=INFO_BLUE)
        formula = MathTex(r"H[x,y]=H[x]+H[y|x]", font_size=42, color=WHITE).to_edge(DOWN, buff=0.55)
        sub = Text("同時に必要な情報 = まず x + x を知った後の y", font_size=25, color=TEXT_GREY).next_to(formula, UP, buff=0.2)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(before), FadeIn(x_group), run_time=1.0)
        self.play(GrowArrow(arrow), TransformFromCopy(before_points, after_points), FadeIn(after_panel), FadeIn(after_title), run_time=1.4)
        self.play(Write(sub), Write(formula), run_time=1.0)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(before), FadeOut(after), FadeOut(x_group), FadeOut(arrow), FadeOut(sub), FadeOut(formula))

    def kl_divergence(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("PRML 1.6.1 / 式 (1.113)")
        title = self.scene_title("KL は、近似分布で代用した余分な情報量", font_size=32)

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.55, 0.1],
            x_length=6.5,
            y_length=3.2,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 1.6 + DOWN * 0.05)
        p_curve = axes.plot(lambda u: gaussian_pdf(u, mu=-0.4, sigma=0.85), x_range=[-4, 4], color=TRUE_GREEN)
        q_curve = axes.plot(lambda u: gaussian_pdf(u, mu=0.75, sigma=1.25), x_range=[-4, 4], color=MODEL_RED)
        p_curve.set_stroke(width=4)
        q_curve.set_stroke(width=4)
        p_label = MathTex("p(x)", font_size=32, color=TRUE_GREEN).next_to(p_curve, UP, buff=0.1).shift(LEFT * 1.0)
        q_label = MathTex("q(x)", font_size=32, color=MODEL_RED).next_to(q_curve, UP, buff=0.1).shift(RIGHT * 0.6)

        formula = MathTex(r"\mathrm{KL}(p\Vert q)=-\int p(x)\ln {q(x)\over p(x)}\,dx", font_size=32)
        formula.to_edge(DOWN, buff=0.55)
        note = Text("q で符号化したときの追加コスト", font_size=26, color=KL_PURPLE).next_to(formula, UP, buff=0.18)

        asym_box = RoundedRectangle(width=3.5, height=2.0, corner_radius=0.12, color=KL_PURPLE)
        asym_text = VGroup(
            MathTex(r"\mathrm{KL}(p\Vert q)", font_size=30, color=TRUE_GREEN),
            MathTex(r"\ne", font_size=28, color=WHITE),
            MathTex(r"\mathrm{KL}(q\Vert p)", font_size=30, color=MODEL_RED),
            Text("普通の距離ではない", font_size=23, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.1)
        asym_text.move_to(asym_box)
        asym = VGroup(asym_box, asym_text).shift(RIGHT * 4.0 + UP * 0.35)
        zero = Text("一致したときだけ 0", font_size=26, color=TRUE_GREEN).next_to(asym, DOWN, buff=0.25)

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(Create(p_curve), Create(q_curve), FadeIn(p_label), FadeIn(q_label), run_time=1.2)
        self.play(Write(note), Write(formula), run_time=1.0)
        self.play(FadeIn(asym), Write(zero), run_time=1.0)
        self.play(Indicate(q_curve, scale_factor=1.04), Indicate(formula, scale_factor=1.03), run_time=1.1)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(p_curve), FadeOut(q_curve), FadeOut(p_label), FadeOut(q_label), FadeOut(note), FadeOut(formula), FadeOut(asym), FadeOut(zero))

    def kl_and_likelihood(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("PRML 1.6.1 / 式 (1.119)")
        title = self.scene_title("KL 最小化は、最尤推定へつながる", font_size=34)

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.55, 0.1],
            x_length=6.5,
            y_length=3.2,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 1.5 + DOWN * 0.05)
        q1 = axes.plot(lambda u: gaussian_pdf(u, mu=1.0, sigma=1.35), x_range=[-4, 4], color=MODEL_RED)
        q2 = axes.plot(lambda u: gaussian_pdf(u, mu=-0.25, sigma=0.95), x_range=[-4, 4], color=TRUE_GREEN)
        q1.set_stroke(width=4)
        q2.set_stroke(width=4)

        rng = np.random.default_rng(16)
        samples = np.array([-1.6, -1.0, -0.45, 0.15, 0.55, 1.05]) + rng.normal(0, 0.08, 6)
        dots = VGroup(*[Dot(axes.c2p(float(x), 0), radius=0.065, color=INFO_BLUE) for x in samples])
        stems = VGroup(*[Line(axes.c2p(float(x), 0), axes.c2p(float(x), gaussian_pdf(float(x), mu=1.0, sigma=1.35)), color=WARN_ORANGE, stroke_width=3) for x in samples])
        stems2 = VGroup(*[Line(axes.c2p(float(x), 0), axes.c2p(float(x), gaussian_pdf(float(x), mu=-0.25, sigma=0.95)), color=TRUE_GREEN, stroke_width=3) for x in samples])

        model_label = MathTex(r"q(x|\theta)", font_size=34, color=MODEL_RED).next_to(axes, RIGHT, buff=0.35).shift(UP * 1.0)
        cost = MathTex(r"\sum_n -\ln q(x_n|\theta)", font_size=34, color=WARN_ORANGE).next_to(model_label, DOWN, buff=0.35)
        relation = VGroup(
            MathTex(r"\min_\theta \mathrm{KL}(p\Vert q_\theta)", font_size=30, color=KL_PURPLE),
            MathTex(r"\Longleftrightarrow", font_size=28),
            MathTex(r"\max_\theta \prod_n q(x_n|\theta)", font_size=30, color=TRUE_GREEN),
        ).arrange(DOWN, buff=0.12)
        relation.next_to(cost, DOWN, buff=0.45)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(dots))
        self.play(Create(q1), Write(model_label), Create(stems), Write(cost), run_time=1.4)
        self.play(Transform(q1, q2), Transform(stems, stems2), model_label.animate.set_color(TRUE_GREEN), run_time=1.4)
        self.play(FadeIn(relation), run_time=1.0)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(axes), FadeOut(dots), FadeOut(q1), FadeOut(stems), FadeOut(model_label), FadeOut(cost), FadeOut(relation))

    def mutual_information(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("PRML 1.6.1 / 式 (1.120)-(1.121)")
        title = self.scene_title("相互情報量は、観測で減った不確かさ", font_size=34)

        rng = np.random.default_rng(21)
        independent = rng.normal(size=(55, 2))
        dependent_x = rng.normal(size=55)
        dependent = np.column_stack([dependent_x, 0.82 * dependent_x + rng.normal(scale=0.45, size=55)])

        def scatter_panel(points: np.ndarray, color: ManimColor, name: str) -> VGroup:
            axes = Axes(
                x_range=[-2.8, 2.8, 1],
                y_range=[-2.8, 2.8, 1],
                x_length=3.2,
                y_length=3.2,
                tips=False,
                axis_config={"color": GREY_B, "stroke_width": 2},
            )
            dots = VGroup(*[Dot(axes.c2p(float(x), float(y)), radius=0.035, color=color) for x, y in points])
            panel_title = Text(name, font_size=24, color=color).next_to(axes, UP, buff=0.18)
            return VGroup(axes, dots, panel_title)

        left = scatter_panel(independent, TEXT_GREY, "独立に近い").shift(LEFT * 3.2 + DOWN * 0.1)
        right = scatter_panel(dependent, INFO_BLUE, "関係がある").shift(RIGHT * 3.2 + DOWN * 0.1)
        formula1 = MathTex(r"I[x,y]=\mathrm{KL}(p(x,y)\Vert p(x)p(y))", font_size=34, color=KL_PURPLE)
        formula1.to_edge(DOWN, buff=0.55)
        formula2 = MathTex(r"I[x,y]=H[x]-H[x|y]", font_size=36, color=TRUE_GREEN)
        formula2.next_to(formula1, UP, buff=0.22)
        note = Text("y を知ったことで、x の不確かさがどれだけ減ったか", font_size=25, color=TEXT_GREY)
        note.next_to(formula2, UP, buff=0.22)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(left), run_time=1.0)
        self.play(FadeIn(right), run_time=1.0)
        self.play(Write(note), Write(formula2), Write(formula1), run_time=1.2)
        self.play(Indicate(right, scale_factor=1.03), Indicate(formula2, scale_factor=1.05), run_time=1.1)
        self.wait(1.0)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(left), FadeOut(right), FadeOut(note), FadeOut(formula2), FadeOut(formula1))
