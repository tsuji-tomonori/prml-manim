from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
POSTERIOR_GREEN = GREEN_C
LIKELIHOOD_ORANGE = ORANGE
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def gaussian(x: float | np.ndarray, mean: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    z = (np.asarray(x) - mean) / sigma
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))


class PRML12ProbabilityTheory(Scene):
    """PRML 1.2 Probability Theory overview.

    Render example:
        uv run manim -ql prml_1_2_probability_theory.py PRML12ProbabilityTheory
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"

        self.uncertainty_as_distribution()
        self.probability_weights()
        self.joint_and_marginal()
        self.conditional_and_product_rule()
        self.bayes_theorem()
        self.probability_density()
        self.expectation_and_variance()
        self.probabilistic_regression()

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

    def make_axes(self, width: float = 7.0, height: float = 3.8) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.6, 1.6, 0.5],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def curve_mean(self, x: float | np.ndarray) -> float | np.ndarray:
        return np.sin(2.0 * np.pi * np.asarray(x)) * 0.85

    def make_data_dots(self, axes: Axes, seed: int = 12, n: int = 16) -> VGroup:
        rng = np.random.default_rng(seed)
        xs = np.linspace(0.04, 0.96, n)
        ts = self.curve_mean(xs) + rng.normal(0.0, 0.23, size=n)
        return VGroup(*[Dot(axes.c2p(float(x), float(t)), radius=0.055, color=BLUE_DATA) for x, t in zip(xs, ts)])

    def bar_chart(
        self,
        labels: list[str],
        values: list[float],
        colors: list[ManimColor],
        width: float = 5.0,
        max_height: float = 2.7,
        base_y: float = -1.25,
    ) -> tuple[VGroup, VGroup, VGroup]:
        bars = VGroup()
        captions = VGroup()
        value_labels = VGroup()
        spacing = width / len(values)
        max_value = max(values)
        for index, (label, value, color) in enumerate(zip(labels, values, colors)):
            height = max_height * value / max_value
            x = (index - (len(values) - 1) / 2) * spacing
            bar = Rectangle(width=0.78, height=height, color=color, fill_color=color, fill_opacity=0.78)
            bar.move_to([x, base_y + height / 2, 0])
            bars.add(bar)
            captions.add(MathTex(label, font_size=28).next_to(bar, DOWN, buff=0.18))
            value_labels.add(Text(f"{value:.2f}", font_size=22, color=color).next_to(bar, UP, buff=0.12))
        chart = VGroup(bars, captions, value_labels)
        return chart, bars, value_labels

    def uncertainty_as_distribution(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 1.2 Probability Theory")
        title = Text("一本の答えから、不確かさを持つ答えへ", font_size=36)
        subtitle = Text("観測ノイズを確率で扱う", font_size=24, color=TEXT_GREY).next_to(title, DOWN, buff=0.25)
        axes = self.make_axes(width=8.2, height=3.7).shift(DOWN * 0.55)
        mean_curve = axes.plot(lambda u: float(self.curve_mean(u)), x_range=[0, 1], color=MODEL_RED)
        upper_curve = axes.plot(lambda u: float(self.curve_mean(u) + 0.32), x_range=[0, 1], color=BLUE_E)
        lower_curve = axes.plot(lambda u: float(self.curve_mean(u) - 0.32), x_range=[0, 1], color=BLUE_E)
        band = axes.get_area(upper_curve, x_range=[0, 1], bounded_graph=lower_curve, color=BLUE_E, opacity=0.18)
        dots = self.make_data_dots(axes)
        one_answer = Text("点予測", font_size=24, color=MODEL_RED).next_to(mean_curve, UP, buff=0.25)
        uncertainty = Text("ありそうな幅", font_size=24, color=BLUE_B).to_edge(RIGHT).shift(DOWN * 1.1)
        arrow = Arrow(uncertainty.get_left() + LEFT * 0.2, axes.c2p(0.78, self.curve_mean(0.78) + 0.3), buff=0, color=BLUE_B)

        self.play(FadeIn(label), Write(title), FadeIn(subtitle, shift=DOWN))
        self.play(title.animate.scale(0.72).to_edge(UP).shift(DOWN * 0.25), FadeOut(subtitle))
        self.play(Create(axes), Create(mean_curve), FadeIn(one_answer))
        self.play(FadeIn(band), FadeIn(dots, shift=UP * 0.1), GrowArrow(arrow), FadeIn(uncertainty))
        self.play(Indicate(band, color=BLUE_B), Indicate(dots, color=BLUE_DATA))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, mean_curve, upper_curve, lower_curve, band, dots, one_answer, uncertainty, arrow)))

    def probability_weights(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("1.2.1 確率の基本")
        title = self.scene_title("確率は、可能性に付ける重み")
        labels = [r"C_1", r"C_2", r"C_3"]
        values = [0.18, 0.57, 0.25]
        colors = [BLUE_C, POSTERIOR_GREEN, LIKELIHOOD_ORANGE]
        chart, bars, value_labels = self.bar_chart(labels, values, colors, width=5.8)
        chart.shift(DOWN * 0.2)
        rule_box = VGroup(
            MathTex(r"0 \leq p(C_k) \leq 1", font_size=36),
            MathTex(r"\sum_k p(C_k)=1", font_size=40, color=YELLOW_C),
        ).arrange(DOWN, buff=0.35).to_edge(RIGHT).shift(LEFT * 0.65)
        winner = SurroundingRectangle(bars[1], color=POSTERIOR_GREEN, buff=0.08)
        note = Text("最大のクラス以外にも重みが残る", font_size=24, color=TEXT_GREY).next_to(chart, DOWN, buff=0.6)

        self.play(FadeIn(label), Write(title))
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.16), FadeIn(chart[1]), FadeIn(value_labels))
        self.play(Write(rule_box[0]), Write(rule_box[1]))
        self.play(Create(winner), FadeIn(note))
        self.play(Indicate(rule_box[1], color=YELLOW_C), Indicate(value_labels, color=WHITE))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, chart, rule_box, winner, note)))

    def make_probability_table(self, values: np.ndarray) -> tuple[VGroup, list[list[Rectangle]], list[list[Text]]]:
        table = VGroup()
        cells: list[list[Rectangle]] = []
        texts: list[list[Text]] = []
        cell_w = 1.2
        cell_h = 0.72
        origin = np.array([-2.2, 0.95, 0.0])
        for row in range(values.shape[0]):
            cell_row = []
            text_row = []
            for col in range(values.shape[1]):
                value = float(values[row, col])
                color = interpolate_color(BLUE_E, GREEN_C, value / values.max())
                rect = Rectangle(width=cell_w, height=cell_h, color=GREY_B, fill_color=color, fill_opacity=0.55)
                rect.move_to(origin + np.array([col * cell_w, -row * cell_h, 0.0]))
                text = Text(f"{value:.2f}", font_size=20)
                text.move_to(rect)
                table.add(rect, text)
                cell_row.append(rect)
                text_row.append(text)
            cells.append(cell_row)
            texts.append(text_row)
        x_labels = VGroup(*[MathTex(fr"x_{i + 1}", font_size=24).next_to(cells[0][i], UP, buff=0.16) for i in range(3)])
        y_labels = VGroup(*[MathTex(fr"c_{i + 1}", font_size=24).next_to(cells[i][0], LEFT, buff=0.18) for i in range(3)])
        table.add(x_labels, y_labels)
        return table, cells, texts

    def joint_and_marginal(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("和の法則")
        title = self.scene_title("同時確率から、見たい変数だけを残す")
        values = np.array(
            [
                [0.04, 0.10, 0.06],
                [0.06, 0.23, 0.11],
                [0.02, 0.12, 0.26],
            ]
        )
        table, cells, _ = self.make_probability_table(values)
        table.shift(LEFT * 1.05 + DOWN * 0.05)
        row_sums = values.sum(axis=1)
        col_sums = values.sum(axis=0)
        row_labels = VGroup()
        for index, value in enumerate(row_sums):
            row_labels.add(Text(f"{value:.2f}", font_size=24, color=YELLOW_C).next_to(cells[index][-1], RIGHT, buff=0.55))
        col_labels = VGroup()
        for index, value in enumerate(col_sums):
            col_labels.add(Text(f"{value:.2f}", font_size=24, color=YELLOW_C).next_to(cells[-1][index], DOWN, buff=0.42))
        row_arrow = Arrow(cells[1][-1].get_right() + RIGHT * 0.1, row_labels[1].get_left() + LEFT * 0.05, buff=0, color=YELLOW_C)
        col_arrow = Arrow(cells[-1][1].get_bottom() + DOWN * 0.08, col_labels[1].get_top() + UP * 0.05, buff=0, color=YELLOW_C)
        equations = VGroup(
            MathTex(r"p(c)=\sum_x p(c,x)", font_size=36),
            MathTex(r"p(x)=\sum_c p(c,x)", font_size=36),
        ).arrange(DOWN, buff=0.35).to_edge(RIGHT).shift(LEFT * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(table, shift=UP * 0.2))
        self.play(Create(row_arrow), FadeIn(row_labels), Write(equations[0]))
        self.play(Create(col_arrow), FadeIn(col_labels), Write(equations[1]))
        self.play(Indicate(VGroup(row_labels, col_labels), color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, table, row_labels, col_labels, row_arrow, col_arrow, equations)))

    def conditional_and_product_rule(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("条件付き確率と積の法則")
        title = self.scene_title("条件を聞いたあとで、重みを読み直す")
        values = np.array(
            [
                [0.04, 0.10, 0.06],
                [0.06, 0.23, 0.11],
                [0.02, 0.12, 0.26],
            ]
        )
        table, cells, _ = self.make_probability_table(values)
        table.shift(LEFT * 1.25 + DOWN * 0.05)
        column_index = 1
        column_group = VGroup(*[cells[row][column_index] for row in range(3)])
        highlight = SurroundingRectangle(column_group, color=YELLOW_C, buff=0.08)
        px = values[:, column_index].sum()
        conditional = values[:, column_index] / px
        condition_title = MathTex(r"x=x_2", font_size=32, color=YELLOW_C).next_to(highlight, UP, buff=0.2)
        right_panel = VGroup(
            Text(f"入口の重み  p(x_2) = {px:.2f}", font_size=25, color=YELLOW_C),
            MathTex(r"p(c\mid x_2)", font_size=34),
            VGroup(
                *[
                    Text(f"c{i + 1}: {value:.2f}", font_size=24, color=[BLUE_C, GREEN_C, ORANGE][i])
                    for i, value in enumerate(conditional)
                ]
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15),
            MathTex(r"p(c,x)=p(c\mid x)p(x)", font_size=36, color=POSTERIOR_GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).to_edge(RIGHT).shift(LEFT * 0.15 + DOWN * 0.1)
        arrows = VGroup(
            *[
                Arrow(cells[row][column_index].get_right() + RIGHT * 0.12, right_panel[2][row].get_left() + LEFT * 0.08, buff=0, color=[BLUE_C, GREEN_C, ORANGE][row])
                for row in range(3)
            ]
        )

        self.play(FadeIn(label), Write(title), FadeIn(table, shift=UP * 0.2))
        self.play(Create(highlight), FadeIn(condition_title))
        self.play(FadeIn(right_panel[0]), FadeIn(right_panel[1]))
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.12), FadeIn(right_panel[2]))
        self.play(Write(right_panel[3]), Indicate(highlight, color=YELLOW_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, table, highlight, condition_title, right_panel, arrows)))

    def bayes_theorem(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("ベイズの定理")
        title = self.scene_title("データを見たあとで、原因の重みを更新する", font_size=32)
        classes = [r"C_1", r"C_2", r"C_3"]
        prior = np.array([0.55, 0.30, 0.15])
        likelihood = np.array([0.20, 0.75, 0.45])
        unnormalized = prior * likelihood
        posterior = unnormalized / unnormalized.sum()
        colors = [BLUE_C, POSTERIOR_GREEN, LIKELIHOOD_ORANGE]

        prior_chart, _, _ = self.bar_chart(classes, prior.tolist(), colors, width=3.0, max_height=1.75, base_y=-0.6)
        likelihood_chart, _, _ = self.bar_chart(classes, likelihood.tolist(), colors, width=3.0, max_height=1.75, base_y=-0.6)
        posterior_chart, _, _ = self.bar_chart(classes, posterior.tolist(), colors, width=3.0, max_height=1.75, base_y=-0.6)
        prior_chart.scale(0.8).to_edge(LEFT).shift(RIGHT * 0.35 + DOWN * 0.45)
        likelihood_chart.scale(0.8).move_to(ORIGIN + DOWN * 0.45)
        posterior_chart.scale(0.8).to_edge(RIGHT).shift(LEFT * 0.35 + DOWN * 0.45)
        headings = VGroup(
            Text("事前確率", font_size=24, color=BLUE_C).next_to(prior_chart, UP, buff=0.35),
            Text("データの出やすさ", font_size=24, color=LIKELIHOOD_ORANGE).next_to(likelihood_chart, UP, buff=0.35),
            Text("事後確率", font_size=24, color=POSTERIOR_GREEN).next_to(posterior_chart, UP, buff=0.35),
        )
        formula = MathTex(r"p(C\mid x)=\frac{p(x\mid C)p(C)}{p(x)}", font_size=42, color=YELLOW_C)
        formula.to_edge(DOWN).shift(UP * 0.25)
        times = MathTex(r"\times", font_size=42).move_to((prior_chart.get_right() + likelihood_chart.get_left()) / 2 + UP * 0.4)
        normalize = Arrow(likelihood_chart.get_right() + RIGHT * 0.18, posterior_chart.get_left() + LEFT * 0.18, buff=0, color=YELLOW_C)
        normalize_label = Text("足して 1 に割り直す", font_size=20, color=YELLOW_C).next_to(normalize, UP, buff=0.15)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(headings[0]), FadeIn(prior_chart))
        self.play(FadeIn(headings[1]), FadeIn(likelihood_chart), Write(times))
        self.play(GrowArrow(normalize), FadeIn(normalize_label), FadeIn(headings[2]), FadeIn(posterior_chart))
        self.play(Write(formula), Indicate(posterior_chart, color=POSTERIOR_GREEN))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, prior_chart, likelihood_chart, posterior_chart, headings, formula, times, normalize, normalize_label)))

    def probability_density(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("確率密度")
        title = self.scene_title("連続値の確率は、曲線の下の面積で読む")
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            x_length=8.2,
            y_length=3.6,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.35)
        density = axes.plot(lambda u: float(gaussian(u, 0, 1)), x_range=[-4, 4], color=POSTERIOR_GREEN)
        density.set_stroke(width=4)
        area = axes.get_area(density, x_range=[-1.0, 1.0], color=BLUE_C, opacity=0.42)
        interval = MathTex(r"a \leq x \leq b", font_size=32, color=BLUE_C).next_to(area, UP, buff=0.2)
        formula = MathTex(r"p(a\leq x\leq b)=\int_a^b p(x)\,dx", font_size=38, color=YELLOW_C)
        formula.to_edge(DOWN).shift(UP * 0.25)
        height_note = Text("高さではなく、面積", font_size=26, color=TEXT_GREY).to_edge(RIGHT).shift(UP * 0.7 + LEFT * 0.3)
        arrow = Arrow(height_note.get_left() + LEFT * 0.1, axes.c2p(1.0, gaussian(1.0)), buff=0, color=TEXT_GREY)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), Create(density))
        self.play(FadeIn(area), FadeIn(interval))
        self.play(GrowArrow(arrow), FadeIn(height_note), Write(formula))
        self.play(Indicate(area, color=BLUE_C))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, density, area, interval, formula, height_note, arrow)))

    def expectation_and_variance(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("期待値と分散")
        title = self.scene_title("分布の中心と広がりを分けて見る")
        xs = np.array([1, 2, 3, 4, 5])
        probs = np.array([0.08, 0.17, 0.42, 0.22, 0.11])
        mean = float((xs * probs).sum())
        variance = float(((xs - mean) ** 2 * probs).sum())
        labels = [str(x) for x in xs]
        colors = [interpolate_color(BLUE_C, POSTERIOR_GREEN, float(p / probs.max())) for p in probs]
        chart, bars, _ = self.bar_chart(labels, probs.tolist(), colors, width=6.6, max_height=2.5, base_y=-1.2)
        chart.shift(DOWN * 0.15)
        x_left = bars[0].get_center()[0]
        step = bars[1].get_center()[0] - bars[0].get_center()[0]
        mean_x = x_left + (mean - 1) * step
        mean_line = Line([mean_x, -1.35, 0], [mean_x, 1.85, 0], color=YELLOW_C, stroke_width=5)
        mean_label = MathTex(r"\mathbb{E}[x]", font_size=34, color=YELLOW_C).next_to(mean_line, UP, buff=0.18)
        spread_arrows = VGroup()
        for bar in [bars[0], bars[-1]]:
            spread_arrows.add(DoubleArrow(mean_line.get_bottom() + UP * 0.35, [bar.get_center()[0], -1.0, 0], buff=0.08, color=LIKELIHOOD_ORANGE))
        formulas = VGroup(
            MathTex(r"\mathbb{E}[x]=\sum_x x\,p(x)", font_size=34, color=YELLOW_C),
            MathTex(r"\mathrm{var}[x]=\mathbb{E}\{(x-\mathbb{E}[x])^2\}", font_size=32, color=LIKELIHOOD_ORANGE),
            Text(f"この例の分散: {variance:.2f}", font_size=22, color=TEXT_GREY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(RIGHT).shift(LEFT * 0.2 + DOWN * 0.15)

        self.play(FadeIn(label), Write(title))
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.12), FadeIn(chart[1]), FadeIn(chart[2]))
        self.play(Create(mean_line), FadeIn(mean_label), Write(formulas[0]))
        self.play(FadeIn(spread_arrows), Write(formulas[1]), FadeIn(formulas[2]))
        self.play(Indicate(mean_line, color=YELLOW_C), Indicate(spread_arrows, color=LIKELIHOOD_ORANGE))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, chart, mean_line, mean_label, spread_arrows, formulas)))

    def probabilistic_regression(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("確率モデルとしての回帰")
        title = self.scene_title("曲線は平均、ノイズは分布として持つ")
        axes = self.make_axes(width=8.1, height=3.7).shift(DOWN * 0.45)
        mean_curve = axes.plot(lambda u: float(self.curve_mean(u)), x_range=[0, 1], color=MODEL_RED)
        mean_curve.set_stroke(width=4)
        upper = axes.plot(lambda u: float(self.curve_mean(u) + 0.38), x_range=[0, 1], color=POSTERIOR_GREEN)
        lower = axes.plot(lambda u: float(self.curve_mean(u) - 0.38), x_range=[0, 1], color=POSTERIOR_GREEN)
        band = axes.get_area(upper, x_range=[0, 1], bounded_graph=lower, color=POSTERIOR_GREEN, opacity=0.18)
        dots = self.make_data_dots(axes, seed=27, n=20)
        formula = MathTex(r"p(t\mid x,\mathbf{w},\beta)=\mathcal{N}(t\mid y(x,\mathbf{w}),\beta^{-1})", font_size=34)
        formula.to_edge(DOWN).shift(UP * 0.18)
        mean_label = Text("平均 y(x,w)", font_size=23, color=MODEL_RED).next_to(mean_curve, UP, buff=0.26)
        noise_label = Text("ノイズの幅", font_size=23, color=POSTERIOR_GREEN).to_edge(RIGHT).shift(LEFT * 0.3)
        arrow = Arrow(noise_label.get_left() + LEFT * 0.1, axes.c2p(0.72, self.curve_mean(0.72) + 0.35), buff=0, color=POSTERIOR_GREEN)
        bridge = Text("尤度・ベイズ推論・モデル選択へ", font_size=28, color=YELLOW_C).next_to(formula, UP, buff=0.35)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(band), Create(mean_curve), FadeIn(dots))
        self.play(FadeIn(mean_label), GrowArrow(arrow), FadeIn(noise_label))
        self.play(Write(formula), FadeIn(bridge, shift=UP * 0.1))
        self.play(Indicate(formula, color=YELLOW_C), Indicate(band, color=POSTERIOR_GREEN))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, mean_curve, upper, lower, band, dots, formula, mean_label, noise_label, arrow, bridge)))
