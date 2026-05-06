from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
POSTERIOR_GREEN = GREEN_C
LIKELIHOOD_ORANGE = ORANGE
DIRICHLET_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


class PRML22MultinomialVariables(Scene):
    """PRML 2.2 Multinomial Variables overview.

    Render example:
        uv run manim -ql prml_2_2_multinomial_variables.py PRML22MultinomialVariables
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"

        self.from_binary_to_multinomial()
        self.one_of_k_representation()
        self.categorical_distribution()
        self.counts_are_sufficient()
        self.maximum_likelihood()
        self.multinomial_distribution()
        self.dirichlet_prior()
        self.posterior_update()

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
        labels: list[str],
        values: list[float],
        colors: list[ManimColor],
        width: float = 5.2,
        max_height: float = 2.75,
    ) -> tuple[VGroup, VGroup, VGroup]:
        bars = VGroup()
        captions = VGroup()
        values_text = VGroup()
        spacing = width / len(values)
        max_value = max(values)
        for index, (label, value, color) in enumerate(zip(labels, values, colors)):
            height = max_height * value / max_value
            x = (index - (len(values) - 1) / 2) * spacing
            bar = Rectangle(width=0.72, height=height, color=color, fill_color=color, fill_opacity=0.78)
            bar.move_to([x, -1.25 + height / 2, 0])
            bars.add(bar)
            captions.add(Text(label, font_size=22).next_to(bar, DOWN, buff=0.18))
            values_text.add(Text(f"{value:.2f}", font_size=20, color=color).next_to(bar, UP, buff=0.12))
        return VGroup(bars, captions, values_text), bars, values_text

    def one_hot_vector(self, active_index: int, k: int = 6) -> VGroup:
        cells = VGroup()
        for index in range(k):
            value = 1 if index == active_index else 0
            color = POSTERIOR_GREEN if value == 1 else GREY_D
            rect = Square(side_length=0.58, color=GREY_B, fill_color=color, fill_opacity=0.72)
            rect.shift(RIGHT * (index - (k - 1) / 2) * 0.72)
            digit = Text(str(value), font_size=24, color=WHITE).move_to(rect)
            cells.add(VGroup(rect, digit))
        bracket_l = Text("(", font_size=42, color=TEXT_GREY).next_to(cells, LEFT, buff=0.12)
        bracket_r = Text(")", font_size=42, color=TEXT_GREY).next_to(cells, RIGHT, buff=0.12)
        transpose = MathTex(r"^{T}", font_size=24, color=TEXT_GREY).next_to(bracket_r, RIGHT, buff=0.02)
        return VGroup(bracket_l, cells, bracket_r, transpose)

    def from_binary_to_multinomial(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 2.2 Multinomial Variables")
        title = Text("二値の外へ: K 通りのどれかを扱う", font_size=38)
        subtitle = Text("複数クラスの確率分布を作る", font_size=25, color=TEXT_GREY).next_to(title, DOWN, buff=0.25)

        binary = VGroup(
            Text("Binary", font_size=26, color=BLUE_DATA),
            MathTex(r"x\in\{0,1\}", font_size=40),
            Text("表 / 裏", font_size=22, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.28)
        multi = VGroup(
            Text("Multinomial", font_size=26, color=POSTERIOR_GREEN),
            MathTex(r"x\in\{1,\ldots,K\}", font_size=40),
            Text("数字・品詞・カテゴリ", font_size=22, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.28)
        binary.shift(LEFT * 3.0 + DOWN * 0.35)
        multi.shift(RIGHT * 3.0 + DOWN * 0.35)
        arrow = Arrow(binary.get_right() + RIGHT * 0.25, multi.get_left() + LEFT * 0.25, buff=0, color=YELLOW_C)
        examples = VGroup(
            Text("手書き数字: 10 通り", font_size=24),
            Text("文書分類: K 通り", font_size=24),
            Text("観測される状態は一つだけ", font_size=24, color=YELLOW_C),
        ).arrange(DOWN, buff=0.22).to_edge(DOWN).shift(UP * 0.3)

        self.play(FadeIn(label), Write(title), FadeIn(subtitle, shift=DOWN))
        self.play(title.animate.scale(0.74).to_edge(UP).shift(DOWN * 0.22), FadeOut(subtitle))
        self.play(FadeIn(binary, shift=LEFT * 0.25), GrowArrow(arrow), FadeIn(multi, shift=RIGHT * 0.25))
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.12) for item in examples], lag_ratio=0.18))
        self.play(Indicate(multi, color=POSTERIOR_GREEN), Indicate(examples[-1], color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, binary, multi, arrow, examples)))

    def one_of_k_representation(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("1-of-K representation")
        title = self.scene_title("一つだけ 1 が立つベクトル")
        states = VGroup()
        for index in range(6):
            card = RoundedRectangle(corner_radius=0.08, width=0.82, height=0.72, color=GREY_B, fill_color=GREY_E, fill_opacity=0.45)
            card.shift(RIGHT * (index - 2.5) * 1.02)
            text = Text(f"k={index + 1}", font_size=19).move_to(card)
            states.add(VGroup(card, text))
        highlight = SurroundingRectangle(states[2], color=POSTERIOR_GREEN, buff=0.06)
        chosen = Text("3 番目が観測された", font_size=25, color=POSTERIOR_GREEN).next_to(states, UP, buff=0.45)
        vector = self.one_hot_vector(active_index=2, k=6).shift(DOWN * 1.35)
        formula = MathTex(r"\sum_{k=1}^{K}x_k=1", font_size=42, color=YELLOW_C).to_edge(DOWN).shift(UP * 0.35)
        arrow = Arrow(states[2].get_bottom(), vector[1][2].get_top(), buff=0.15, color=POSTERIOR_GREEN)

        self.play(FadeIn(label), Write(title))
        self.play(LaggedStart(*[FadeIn(state, shift=UP * 0.12) for state in states], lag_ratio=0.08))
        self.play(Create(highlight), FadeIn(chosen), GrowArrow(arrow))
        self.play(FadeIn(vector, shift=DOWN * 0.2))
        self.play(Write(formula), Indicate(vector[1][2], color=POSTERIOR_GREEN))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, states, highlight, chosen, vector, formula, arrow)))

    def categorical_distribution(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Categorical distribution")
        title = self.scene_title("選ばれた成分の確率だけが残る")
        values = [0.12, 0.20, 0.43, 0.15, 0.10]
        labels = [r"k=1", r"k=2", r"k=3", r"k=4", r"k=5"]
        colors = [BLUE_C, TEAL_C, POSTERIOR_GREEN, ORANGE, PURPLE_C]
        chart, bars, value_text = self.probability_bars(labels, values, colors)
        chart.shift(LEFT * 2.85 + DOWN * 0.05)
        selected = SurroundingRectangle(bars[2], color=YELLOW_C, buff=0.08)
        equations = VGroup(
            MathTex(r"\mu_k\geq 0,\quad \sum_{k=1}^{K}\mu_k=1", font_size=35),
            MathTex(r"p(\mathbf{x}\mid\boldsymbol{\mu})=\prod_{k=1}^{K}\mu_k^{x_k}", font_size=38, color=YELLOW_C),
            Text("x3=1 なら  p(x|mu)=mu3", font_size=25, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.35).to_edge(RIGHT).shift(LEFT * 0.35 + DOWN * 0.05)

        self.play(FadeIn(label), Write(title))
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.12), FadeIn(chart[1]), FadeIn(value_text))
        self.play(Write(equations[0]), Write(equations[1]))
        self.play(Create(selected), FadeIn(equations[2], shift=UP * 0.1))
        self.play(Indicate(equations[1], color=YELLOW_C), Indicate(value_text[2], color=WHITE))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, chart, selected, equations)))

    def counts_are_sufficient(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Likelihood and sufficient statistics")
        title = self.scene_title("順番を消して、カウントだけを見る")
        sequence_values = [3, 2, 3, 1, 3, 4, 2, 3, 5, 1, 3, 4]
        colors = [BLUE_C, TEAL_C, POSTERIOR_GREEN, ORANGE, PURPLE_C]
        chips = VGroup()
        for index, value in enumerate(sequence_values):
            chip = Circle(radius=0.22, color=colors[value - 1], fill_color=colors[value - 1], fill_opacity=0.8)
            chip.shift(RIGHT * (index - (len(sequence_values) - 1) / 2) * 0.48 + UP * 1.1)
            chips.add(VGroup(chip, Text(str(value), font_size=16).move_to(chip)))
        counts = [sequence_values.count(k) for k in range(1, 6)]
        count_chart, bars, count_text = self.probability_bars([str(k) for k in range(1, 6)], counts, colors, width=4.8, max_height=2.2)
        count_chart.shift(DOWN * 1.1)
        count_label = Text("m = (2, 2, 5, 2, 1)", font_size=27, color=YELLOW_C).next_to(count_chart, UP, buff=0.35)
        likelihood = MathTex(r"p(D\mid\mu)=\prod_{k=1}^{K}\mu_k^{m_k}", font_size=40, color=YELLOW_C).to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(label), Write(title))
        self.play(LaggedStart(*[FadeIn(chip, shift=DOWN * 0.1) for chip in chips], lag_ratio=0.04))
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.12), FadeIn(count_chart[1]), FadeIn(count_text), FadeIn(count_label))
        self.play(Write(likelihood), Indicate(count_label, color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, chips, count_chart, count_label, likelihood)))

    def maximum_likelihood(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("Maximum likelihood")
        title = self.scene_title("最尤推定 = 観測された割合")
        counts = [2, 2, 5, 2, 1]
        n = sum(counts)
        fractions = [count / n for count in counts]
        colors = [BLUE_C, TEAL_C, POSTERIOR_GREEN, ORANGE, PURPLE_C]
        count_chart, _, _ = self.probability_bars([str(k) for k in range(1, 6)], counts, colors, width=4.5, max_height=2.25)
        count_chart.shift(LEFT * 3.0 + DOWN * 0.35)
        frac_chart, _, _ = self.probability_bars([str(k) for k in range(1, 6)], fractions, colors, width=4.5, max_height=2.25)
        frac_chart.shift(RIGHT * 3.0 + DOWN * 0.35)
        left_label = Text("count m", font_size=25, color=TEXT_GREY).next_to(count_chart, UP, buff=0.25)
        right_label = Text("probability mu", font_size=25, color=TEXT_GREY).next_to(frac_chart, UP, buff=0.25)
        arrow = Arrow(count_chart.get_right() + RIGHT * 0.35, frac_chart.get_left() + LEFT * 0.35, buff=0, color=YELLOW_C)
        equation = MathTex(r"\mu_k^{ML}=\frac{m_k}{N}", font_size=48, color=YELLOW_C).to_edge(DOWN).shift(UP * 0.28)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(count_chart), FadeIn(left_label))
        self.play(GrowArrow(arrow), Write(equation))
        self.play(FadeIn(frac_chart, shift=RIGHT * 0.2), FadeIn(right_label))
        self.play(Indicate(frac_chart, color=POSTERIOR_GREEN), Indicate(equation, color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, count_chart, frac_chart, left_label, right_label, arrow, equation)))

    def multinomial_distribution(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Multinomial distribution")
        title = self.scene_title("同じカウントを作る並び方を数える")
        chips_a = VGroup()
        chips_b = VGroup()
        patterns = [[1, 2, 3, 1, 3], [3, 1, 1, 2, 3]]
        colors = [BLUE_C, TEAL_C, POSTERIOR_GREEN]
        for row, pattern in enumerate(patterns):
            group = chips_a if row == 0 else chips_b
            for index, value in enumerate(pattern):
                chip = Circle(radius=0.24, color=colors[value - 1], fill_color=colors[value - 1], fill_opacity=0.82)
                chip.shift(RIGHT * (index - 2) * 0.62 + UP * (0.8 - row * 0.95) + LEFT * 3.2)
                group.add(VGroup(chip, Text(str(value), font_size=17).move_to(chip)))
        same_count = Text("どちらも m=(2,1,2)", font_size=25, color=YELLOW_C).next_to(VGroup(chips_a, chips_b), DOWN, buff=0.35)
        coefficient = VGroup(
            MathTex(r"\mathrm{Mult}(m_1,\ldots,m_K\mid \mu,N)", font_size=34),
            MathTex(r"=\frac{N!}{m_1!\cdots m_K!}\prod_{k=1}^{K}\mu_k^{m_k}", font_size=42, color=YELLOW_C),
        ).arrange(DOWN, buff=0.35).to_edge(RIGHT).shift(LEFT * 0.25)
        note = Text("係数 = 並び替えの数", font_size=25, color=TEXT_GREY).next_to(coefficient, DOWN, buff=0.45)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(chips_a, shift=UP * 0.1), FadeIn(chips_b, shift=UP * 0.1))
        self.play(FadeIn(same_count), Write(coefficient[0]))
        self.play(Write(coefficient[1]), FadeIn(note))
        self.play(Indicate(coefficient[1][0][6:15], color=LIKELIHOOD_ORANGE), Indicate(same_count, color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, chips_a, chips_b, same_count, coefficient, note)))

    def simplex(self) -> tuple[Polygon, VGroup, VGroup]:
        vertices = [
            np.array([-2.3, -1.4, 0.0]),
            np.array([2.3, -1.4, 0.0]),
            np.array([0.0, 1.95, 0.0]),
        ]
        triangle = Polygon(*vertices, color=GREY_B, fill_color=BLUE_E, fill_opacity=0.16)
        labels = VGroup(
            MathTex(r"\mu_1=1", font_size=24).next_to(vertices[0], DOWN, buff=0.15),
            MathTex(r"\mu_2=1", font_size=24).next_to(vertices[1], DOWN, buff=0.15),
            MathTex(r"\mu_3=1", font_size=24).next_to(vertices[2], UP, buff=0.12),
        )
        dots = VGroup()
        rng = np.random.default_rng(8)
        for _ in range(45):
            weights = rng.dirichlet([2.5, 1.5, 3.2])
            point = weights[0] * vertices[0] + weights[1] * vertices[1] + weights[2] * vertices[2]
            color = interpolate_color(BLUE_C, DIRICHLET_PURPLE, weights[2])
            dots.add(Dot(point, radius=0.035, color=color))
        return triangle, labels, dots

    def dirichlet_prior(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Dirichlet prior")
        title = self.scene_title("確率ベクトルは単体の上を動く")
        triangle, simplex_labels, dots = self.simplex()
        simplex_group = VGroup(triangle, simplex_labels, dots).shift(LEFT * 2.65 + DOWN * 0.1)
        equations = VGroup(
            MathTex(r"\mathrm{Dir}(\mu\mid\alpha)", font_size=40, color=DIRICHLET_PURPLE),
            MathTex(r"\propto \prod_{k=1}^{K}\mu_k^{\alpha_k-1}", font_size=42, color=YELLOW_C),
            Text("alpha_k は事前カウントのように効く", font_size=24, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.38).to_edge(RIGHT).shift(LEFT * 0.28)
        alpha_bars, _, _ = self.probability_bars(["a1", "a2", "a3"], [2.5, 1.5, 3.2], [BLUE_C, TEAL_C, DIRICHLET_PURPLE], width=3.2, max_height=1.6)
        alpha_bars.next_to(equations, DOWN, buff=0.45)

        self.play(FadeIn(label), Write(title))
        self.play(Create(triangle), FadeIn(simplex_labels))
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.015))
        self.play(Write(equations[0]), Write(equations[1]))
        self.play(FadeIn(equations[2]), FadeIn(alpha_bars, shift=UP * 0.1))
        self.play(Indicate(dots, color=DIRICHLET_PURPLE), Indicate(equations[2], color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, simplex_group, equations, alpha_bars)))

    def posterior_update(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("Conjugate update")
        title = self.scene_title("Dirichlet + counts = Dirichlet")
        prior = VGroup(
            Text("prior", font_size=24, color=DIRICHLET_PURPLE),
            MathTex(r"\alpha=(2,1,2)", font_size=36, color=DIRICHLET_PURPLE),
        ).arrange(DOWN, buff=0.22)
        data = VGroup(
            Text("data", font_size=24, color=LIKELIHOOD_ORANGE),
            MathTex(r"m=(2,2,5)", font_size=36, color=LIKELIHOOD_ORANGE),
        ).arrange(DOWN, buff=0.22)
        posterior = VGroup(
            Text("posterior", font_size=24, color=POSTERIOR_GREEN),
            MathTex(r"\alpha+m=(4,3,7)", font_size=36, color=POSTERIOR_GREEN),
        ).arrange(DOWN, buff=0.22)
        prior.shift(LEFT * 4.0 + UP * 0.6)
        data.shift(LEFT * 0.7 + UP * 0.6)
        posterior.shift(RIGHT * 3.0 + UP * 0.6)
        plus = MathTex("+", font_size=42).move_to((prior.get_right() + data.get_left()) / 2)
        arrow = Arrow(data.get_right() + RIGHT * 0.25, posterior.get_left() + LEFT * 0.25, buff=0, color=YELLOW_C)
        equations = VGroup(
            MathTex(r"p(\mu\mid D,\alpha)=\mathrm{Dir}(\mu\mid\alpha+m)", font_size=40, color=YELLOW_C),
            Text("ベータ・ベルヌーイの関係を K 通りへ拡張", font_size=25, color=TEXT_GREY),
            Text("次は連続値の中心: ガウス分布", font_size=27, color=WHITE),
        ).arrange(DOWN, buff=0.35).to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(prior, shift=UP * 0.1), FadeIn(data, shift=UP * 0.1), Write(plus))
        self.play(GrowArrow(arrow), FadeIn(posterior, shift=RIGHT * 0.2))
        self.play(Write(equations[0]))
        self.play(FadeIn(equations[1], shift=UP * 0.1), FadeIn(equations[2], shift=UP * 0.1))
        self.play(Indicate(posterior, color=POSTERIOR_GREEN), Indicate(equations[0], color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, prior, data, posterior, plus, arrow, equations)))
