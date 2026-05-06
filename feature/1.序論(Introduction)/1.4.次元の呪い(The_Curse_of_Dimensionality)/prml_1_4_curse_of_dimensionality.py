from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


JAPANESE_FONT = "Noto Sans CJK JP"
BG = "#101010"
INK = WHITE
MUTED = GREY_B
BLUE_DATA = BLUE_C
ACCENT = TEAL_C
WARNING = ORANGE
DANGER = RED_C
GOOD = GREEN_C
PURPLE_DIM = PURPLE_C
YELLOW_NOTE = YELLOW_C

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def fmt_big_number(value: int) -> str:
    if value < 10_000:
        return f"{value:,}"
    if value < 100_000_000:
        return f"{value / 10_000:.1f}万"
    return f"{value / 100_000_000:.1f}億"


def nearest_distance_summary(sample_count: int = 300, dimensions: tuple[int, ...] = (1, 2, 5, 10, 20)) -> list[tuple[int, float]]:
    rng = np.random.default_rng(14)
    summary: list[tuple[int, float]] = []
    for dim in dimensions:
        points = rng.random((sample_count, dim))
        query = np.full(dim, 0.5)
        raw = np.linalg.norm(points - query, axis=1).min()
        normalized = raw / math.sqrt(dim)
        summary.append((dim, normalized))
    return summary


class PRML14CurseOfDimensionality(Scene):
    """PRML 1.4 overview of the curse of dimensionality.

    Render example:
        uv run manim -ql prml_1_4_curse_of_dimensionality.py PRML14CurseOfDimensionality
    """

    def construct(self) -> None:
        self.camera.background_color = BG
        self.opening()
        self.grid_growth()
        self.local_volume()
        self.boundary_volume()
        self.nearest_neighbors()
        self.empty_cells()
        self.what_helps()
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

    def finish_narration(self, narration: tuple[float, float | None], pad: float = 0.25) -> None:
        start_time, duration = narration
        if duration is None:
            return
        elapsed = float(getattr(self, "time", 0.0)) - start_time
        remaining = duration - elapsed + pad
        if remaining > 0:
            self.wait(remaining)

    def section_label(self, text: str) -> Text:
        label = Text(text, font_size=18, color=MUTED)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 34) -> Text:
        title = Text(text, font_size=font_size, color=INK)
        title.to_edge(UP).shift(DOWN * 0.28)
        return title

    def formula(self, text: str, color: ManimColor = INK, font_size: int = 34) -> Text:
        return Text(text, font_size=font_size, color=color)

    def metric_card(self, title: str, value: str, caption: str, color: ManimColor) -> VGroup:
        box = RoundedRectangle(width=3.05, height=2.05, corner_radius=0.08)
        box.set_fill("#181818", opacity=1)
        box.set_stroke(color, width=2)
        title_mob = Text(title, font_size=24, color=color)
        value_mob = Text(value, font_size=34, color=INK)
        caption_mob = Text(caption, font_size=17, color=MUTED)
        content = VGroup(title_mob, value_mob, caption_mob).arrange(DOWN, buff=0.16)
        content.move_to(box)
        return VGroup(box, content)

    def simple_axes(self, x_label: str, y_label: str, width: float = 7.6, height: float = 4.0) -> tuple[Axes, VGroup]:
        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 1, 0.25],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )
        labels = VGroup(
            Text(x_label, font_size=20, color=MUTED).next_to(axes.x_axis, DOWN, buff=0.18),
            Text(y_label, font_size=20, color=MUTED).next_to(axes.y_axis, LEFT, buff=0.18),
        )
        return axes, labels

    def bar_chart(
        self,
        values: list[float],
        labels: list[str],
        colors: list[ManimColor],
        max_height: float = 3.2,
        width: float = 0.62,
        value_formatter=lambda x: f"{x:.2f}",
    ) -> VGroup:
        max_value = max(values) if values else 1.0
        bars = VGroup()
        for value, label, color in zip(values, labels, colors):
            height = max(0.08, max_height * value / max_value)
            rect = Rectangle(width=width, height=height)
            rect.set_fill(color, opacity=0.84)
            rect.set_stroke(color, width=1)
            number = Text(value_formatter(value), font_size=18, color=INK).next_to(rect, UP, buff=0.10)
            x_label = Text(label, font_size=18, color=MUTED).next_to(rect, DOWN, buff=0.12)
            bars.add(VGroup(rect, number, x_label))
        bars.arrange(RIGHT, buff=0.42, aligned_edge=DOWN)
        return bars

    def opening(self) -> None:
        narration = self.start_narration("scene01")
        section = self.section_label("PRML 1.4")
        title = Text("次元の呪い", font_size=52, color=INK)
        subtitle = Text("データ空間は、次元が増えると急に広くなる", font_size=28, color=MUTED)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.18).to_edge(UP).shift(DOWN * 0.72)

        line = NumberLine(x_range=[0, 1, 0.25], length=4.6, color=GREY_B, include_numbers=False)
        line_dots = VGroup(*[Dot(line.n2p(x), radius=0.055, color=BLUE_DATA) for x in np.linspace(0.08, 0.92, 8)])
        one_d = VGroup(Text("1次元", font_size=24, color=ACCENT), line, line_dots).arrange(DOWN, buff=0.20)

        square = Square(side_length=3.15).set_stroke(GREY_B, width=2)
        rng = np.random.default_rng(4)
        square_dots = VGroup(
            *[
                Dot(square.get_center() + np.array([rng.uniform(-1.42, 1.42), rng.uniform(-1.42, 1.42), 0]), radius=0.038, color=BLUE_DATA)
                for _ in range(35)
            ]
        )
        two_d = VGroup(Text("2次元", font_size=24, color=ACCENT), square, square_dots).arrange(DOWN, buff=0.20)

        vector = VGroup(
            *[
                RoundedRectangle(width=0.62, height=0.62, corner_radius=0.05)
                .set_fill(PURPLE_DIM if i < 4 else "#1d1d1d", opacity=0.9)
                .set_stroke(GREY_B, width=1)
                for i in range(10)
            ]
        ).arrange(RIGHT, buff=0.06)
        vector_label = Text("10次元以上: 見えない座標が増える", font_size=24, color=PURPLE_DIM)
        high_d = VGroup(vector_label, vector).arrange(DOWN, buff=0.28)

        panels = VGroup(one_d, two_d, high_d).arrange(RIGHT, buff=0.70, aligned_edge=DOWN)
        panels.next_to(header, DOWN, buff=0.70)

        self.play(FadeIn(section), Write(title), FadeIn(subtitle, shift=DOWN * 0.2))
        self.play(Create(line), LaggedStart(*[FadeIn(dot, scale=0.5) for dot in line_dots], lag_ratio=0.06))
        self.play(Create(square), LaggedStart(*[FadeIn(dot, scale=0.5) for dot in square_dots], lag_ratio=0.02))
        self.play(FadeIn(VGroup(one_d[0], two_d[0], high_d), shift=UP * 0.15))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(section, header, panels)))

    def grid_growth(self) -> None:
        narration = self.start_narration("scene02")
        title = self.scene_title("1軸を10分割するだけで、格子数は 10^D")
        formula = self.formula("cells = 10^D", color=YELLOW_NOTE, font_size=40).next_to(title, DOWN, buff=0.35)

        cards = VGroup(
            self.metric_card("D = 1", "10", "線上の区間", ACCENT),
            self.metric_card("D = 2", "100", "平面の小さな四角", BLUE_DATA),
            self.metric_card("D = 3", "1,000", "立方体の小部屋", WARNING),
            self.metric_card("D = 10", "100億", "同じ分割が爆発", DANGER),
        ).arrange(RIGHT, buff=0.28)
        cards.next_to(formula, DOWN, buff=0.55)

        grid = VGroup()
        for i in range(10):
            for j in range(10):
                sq = Square(side_length=0.155)
                sq.set_fill(BLUE_E if (i + j) % 2 == 0 else "#222222", opacity=0.78)
                sq.set_stroke("#303030", width=0.5)
                sq.move_to(np.array([i * 0.155, j * 0.155, 0]))
                grid.add(sq)
        grid.arrange_in_grid(rows=10, cols=10, buff=0)
        grid.scale(0.95)
        grid.to_edge(DOWN).shift(UP * 0.25 + LEFT * 3.3)

        note = Text("同じ細かさを保つには、データ数も指数的に必要", font_size=25, color=INK)
        note.to_edge(DOWN).shift(UP * 0.42 + RIGHT * 2.0)

        self.play(FadeIn(title), Write(formula))
        for card in cards:
            self.play(FadeIn(card, shift=UP * 0.18), run_time=0.55)
        self.play(LaggedStart(*[FadeIn(cell, scale=0.7) for cell in grid], lag_ratio=0.004), FadeIn(note), run_time=1.8)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, formula, cards, grid, note)))

    def local_volume(self) -> None:
        narration = self.start_narration("scene03")
        title = self.scene_title("「近い範囲」のつもりが、高次元では広くなる")
        formula = self.formula("side length = 0.1^(1/D)", color=YELLOW_NOTE, font_size=34).next_to(title, DOWN, buff=0.34)
        explanation = Text("体積の10%を含む中心領域の、1軸あたりの幅", font_size=22, color=MUTED)
        explanation.next_to(formula, DOWN, buff=0.18)

        dims = [1, 2, 5, 10, 50]
        widths = [0.1 ** (1 / d) for d in dims]
        chart = self.bar_chart(
            widths,
            [f"D={d}" for d in dims],
            [ACCENT, BLUE_DATA, WARNING, DANGER, PURPLE_DIM],
            value_formatter=lambda x: f"{x * 100:.0f}%",
        )
        chart.next_to(explanation, DOWN, buff=0.58)

        line = NumberLine(x_range=[0, 1, 0.25], length=6.5, color=GREY_B, include_numbers=False)
        intervals = VGroup()
        for width, color in [(widths[0], ACCENT), (widths[1], BLUE_DATA), (widths[3], DANGER)]:
            start = 0.5 - width / 2
            end = 0.5 + width / 2
            intervals.add(Line(line.n2p(start), line.n2p(end), color=color, stroke_width=9))
        intervals.arrange(DOWN, buff=0.22)
        line_group = VGroup(Text("同じ10%でも、軸方向には広がる", font_size=22, color=INK), line, intervals)
        line_group.arrange(DOWN, buff=0.18)
        line_group.to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(title), Write(formula), FadeIn(explanation))
        self.play(LaggedStart(*[GrowFromEdge(bar[0], DOWN) for bar in chart], lag_ratio=0.12), FadeIn(chart))
        self.play(FadeIn(line_group[0]), Create(line), LaggedStart(*[Create(seg) for seg in intervals], lag_ratio=0.25))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, formula, explanation, chart, line_group)))

    def boundary_volume(self) -> None:
        narration = self.start_narration("scene04")
        title = self.scene_title("高次元では、多くの体積が境界の近くに寄る")
        formula = self.formula("inside fraction = 0.8^D", color=YELLOW_NOTE, font_size=36).next_to(title, DOWN, buff=0.34)
        note = Text("各軸の端から10%を境界帯とみなす", font_size=22, color=MUTED).next_to(formula, DOWN, buff=0.16)

        outer = Square(side_length=3.4).set_stroke(INK, width=2)
        outer.set_fill(DANGER, opacity=0.20)
        inner = Square(side_length=2.72).set_stroke(GOOD, width=3)
        inner.set_fill(GOOD, opacity=0.35)
        square_group = VGroup(outer, inner, Text("2D: 内側 64%", font_size=22, color=GOOD).next_to(outer, DOWN, buff=0.2))
        square_group.next_to(note, DOWN, buff=0.58).shift(LEFT * 3.2)

        dims = [2, 5, 10, 20]
        inside = [0.8**d for d in dims]
        bars = self.bar_chart(
            inside,
            [f"D={d}" for d in dims],
            [GOOD, ACCENT, WARNING, DANGER],
            value_formatter=lambda x: f"{x * 100:.1f}%",
        )
        bars.next_to(note, DOWN, buff=0.78).shift(RIGHT * 2.4)
        self.play(FadeIn(title), Write(formula), FadeIn(note))
        self.play(FadeIn(outer), FadeIn(inner, scale=0.7), FadeIn(square_group[2]))
        self.play(LaggedStart(*[GrowFromEdge(bar[0], DOWN) for bar in bars], lag_ratio=0.14), FadeIn(bars))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, formula, note, square_group, bars)))

    def nearest_neighbors(self) -> None:
        narration = self.start_narration("scene05")
        title = self.scene_title("点が多く見えても、高次元では近所が遠い")
        axes, labels = self.simple_axes("x1", "x2", width=4.2, height=3.2)
        axes_group = VGroup(axes, labels).to_edge(LEFT).shift(DOWN * 0.25 + RIGHT * 0.45)
        rng = np.random.default_rng(22)
        points = VGroup(
            *[
                Dot(axes.c2p(float(x), float(y)), radius=0.025, color=BLUE_DATA)
                for x, y in rng.random((130, 2))
            ]
        )
        query = Dot(axes.c2p(0.5, 0.5), radius=0.075, color=YELLOW_NOTE)
        circle = Circle(radius=0.38, color=YELLOW_NOTE, stroke_width=3).move_to(query)
        left_caption = Text("2Dでは、中心の近くに点が見つかる", font_size=21, color=INK).next_to(axes_group, DOWN, buff=0.20)

        summary = nearest_distance_summary()
        dims = [d for d, _ in summary]
        distances = [v for _, v in summary]
        bars = self.bar_chart(
            distances,
            [f"D={d}" for d in dims],
            [ACCENT, BLUE_DATA, WARNING, DANGER, PURPLE_DIM],
            max_height=2.85,
            value_formatter=lambda x: f"{x:.2f}",
        )
        bars.to_edge(RIGHT).shift(LEFT * 0.55 + DOWN * 0.1)
        bar_title = Text("300点から一番近い点までの距離", font_size=23, color=INK).next_to(bars, UP, buff=0.34)
        bar_subtitle = Text("最大距離を1に近づけて正規化", font_size=18, color=MUTED).next_to(bar_title, DOWN, buff=0.11)

        self.play(FadeIn(title))
        self.play(Create(axes), FadeIn(labels), LaggedStart(*[FadeIn(dot, scale=0.6) for dot in points], lag_ratio=0.006))
        self.play(FadeIn(query, scale=1.5), Create(circle), FadeIn(left_caption))
        self.play(FadeIn(bar_title), FadeIn(bar_subtitle))
        self.play(LaggedStart(*[GrowFromEdge(bar[0], DOWN) for bar in bars], lag_ratio=0.13), FadeIn(bars))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, axes_group, points, query, circle, left_caption, bars, bar_title, bar_subtitle)))

    def empty_cells(self) -> None:
        narration = self.start_narration("scene06")
        title = self.scene_title("格子で空間を覚えようとすると、空の箱だらけになる")
        formula = self.formula("cells = 5^D,  samples = 1,000", color=YELLOW_NOTE, font_size=34).next_to(title, DOWN, buff=0.33)

        dims = [2, 4, 6, 10]
        cells = [5**d for d in dims]
        cards = VGroup()
        for dim, cell_count in zip(dims, cells):
            expected_empty = math.exp(-1000 / cell_count) if cell_count > 0 else 0
            caption = f"空箱 約{expected_empty * 100:.0f}%"
            if dim == 2:
                caption = "ほぼ埋まる"
            cards.add(self.metric_card(f"D = {dim}", fmt_big_number(cell_count), caption, DANGER if dim >= 6 else ACCENT))
        cards.arrange(RIGHT, buff=0.28).next_to(formula, DOWN, buff=0.55)

        grid = VGroup()
        occupied = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2), (1, 1), (3, 0)}
        for r in range(5):
            for c in range(5):
                sq = Square(side_length=0.42)
                is_occupied = (r, c) in occupied
                sq.set_fill(BLUE_DATA if is_occupied else "#202020", opacity=0.88)
                sq.set_stroke(GREY_D, width=1)
                grid.add(sq)
        grid.arrange_in_grid(rows=5, cols=5, buff=0.02)
        grid.to_edge(DOWN).shift(UP * 0.28 + LEFT * 3.1)
        grid_label = Text("2Dなら観察できる箱がまだ多い", font_size=21, color=INK).next_to(grid, UP, buff=0.18)

        warning = Text("D=10 では 5^10 = 9,765,625 個", font_size=30, color=DANGER)
        warning.to_edge(DOWN).shift(UP * 0.85 + RIGHT * 2.2)
        warning2 = Text("1,000点では、ほとんどの場所を一度も見ない", font_size=23, color=INK)
        warning2.next_to(warning, DOWN, buff=0.20)

        self.play(FadeIn(title), Write(formula))
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.16) for card in cards], lag_ratio=0.16))
        self.play(FadeIn(grid_label), LaggedStart(*[FadeIn(cell, scale=0.85) for cell in grid], lag_ratio=0.02))
        self.play(FadeIn(warning, shift=UP * 0.15), FadeIn(warning2, shift=UP * 0.15))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, formula, cards, grid, grid_label, warning, warning2)))

    def what_helps(self) -> None:
        narration = self.start_narration("scene07")
        title = self.scene_title("呪いへの対策は、空間をそのまま埋めないこと")
        vector = VGroup()
        labels = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
        for i, label in enumerate(labels):
            color = GOOD if i in (1, 6) else "#242424"
            stroke = GOOD if i in (1, 6) else GREY_D
            cell = RoundedRectangle(width=0.74, height=0.70, corner_radius=0.06)
            cell.set_fill(color, opacity=0.86)
            cell.set_stroke(stroke, width=2)
            vector.add(VGroup(cell, Text(label, font_size=19, color=INK).move_to(cell)))
        vector.arrange(RIGHT, buff=0.07).next_to(title, DOWN, buff=0.55)
        vector_caption = Text("10個の座標すべてが同じくらい重要とは限らない", font_size=24, color=INK)
        vector_caption.next_to(vector, DOWN, buff=0.24)

        ideas = VGroup(
            self.metric_card("仮定", "smooth", "近い入力は近い出力", ACCENT),
            self.metric_card("正則化", "small", "複雑すぎる形を抑える", WARNING),
            self.metric_card("特徴選択", "relevant", "効く座標に集中する", GOOD),
        ).arrange(RIGHT, buff=0.42)
        ideas.next_to(vector_caption, DOWN, buff=0.55)

        bottom = Text("データだけで全空間を埋める代わりに、構造を使う", font_size=26, color=YELLOW_NOTE)
        bottom.to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(cell, scale=0.85) for cell in vector], lag_ratio=0.05), FadeIn(vector_caption))
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.18) for card in ideas], lag_ratio=0.18))
        self.play(FadeIn(bottom, shift=UP * 0.15))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, vector, vector_caption, ideas, bottom)))

    def summary(self) -> None:
        narration = self.start_narration("scene08")
        title = Text("PRML 1.4 の要点", font_size=44, color=INK).to_edge(UP).shift(DOWN * 0.45)
        bullets = VGroup(
            Text("1. 次元が増えると、同じ細かさの格子数は指数的に増える", font_size=27, color=INK),
            Text("2. 体積は境界側に寄り、近所の点も見つけにくくなる", font_size=27, color=INK),
            Text("3. 汎化には、データ量だけでなく仮定と正則化が必要", font_size=27, color=INK),
        ).arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        bullets.next_to(title, DOWN, buff=0.75).shift(LEFT * 0.15)

        bridge = Text("次は、予測をどう判断に変えるか: Decision Theory", font_size=26, color=ACCENT)
        bridge.to_edge(DOWN).shift(UP * 0.65)

        self.play(FadeIn(title))
        for bullet in bullets:
            self.play(FadeIn(bullet, shift=RIGHT * 0.15), run_time=0.55)
        self.play(FadeIn(bridge, shift=UP * 0.16))
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(title, bullets, bridge)))
