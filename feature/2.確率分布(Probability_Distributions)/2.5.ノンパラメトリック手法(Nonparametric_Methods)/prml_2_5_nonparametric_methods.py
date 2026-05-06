from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
TRUE_GREEN = GREEN_C
MODEL_RED = RED_C
TEST_ORANGE = ORANGE
KNN_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def gaussian_pdf(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    return np.exp(-0.5 * ((np.asarray(x) - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def make_mixture_data(n: int = 50, seed: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(0.29, 0.055, n // 2)
    right = rng.normal(0.72, 0.07, n - n // 2)
    x = np.concatenate([left, right])
    return np.clip(np.sort(x), 0.03, 0.97)


def true_density(x: np.ndarray | float) -> np.ndarray | float:
    return 0.48 * gaussian_pdf(x, 0.29, 0.06) + 0.52 * gaussian_pdf(x, 0.72, 0.08)


def single_gaussian_fit(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return gaussian_pdf(grid, mu, sigma)


def kde_density(samples: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    diffs = (grid[:, None] - samples[None, :]) / h
    values = np.exp(-0.5 * diffs**2) / (h * math.sqrt(2.0 * math.pi))
    return values.mean(axis=1)


def knn_density(samples: np.ndarray, grid: np.ndarray, k: int) -> np.ndarray:
    densities = []
    n = len(samples)
    for value in grid:
        distances = np.sort(np.abs(samples - value))
        radius = max(float(distances[k - 1]), 0.012)
        densities.append(k / (n * 2.0 * radius))
    return np.minimum(np.array(densities), 7.5)


class PRML25NonparametricMethods(Scene):
    """PRML 2.5 nonparametric methods overview.

    Render example:
        uv run manim -pql prml_2_5_nonparametric_methods.py PRML25NonparametricMethods
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.samples = make_mixture_data()
        self.grid = np.linspace(0.0, 1.0, 500)

        self.opening_parametric_limit()
        self.histogram_bandwidth()
        self.local_counting_rule()
        self.kernel_density_estimator()
        self.knn_density_estimator()
        self.knn_classifier()
        self.summary_tradeoffs()

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

    def make_density_axes(self, width: float = 8.2, height: float = 4.4) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 8, 2],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_density_curve(self, axes: Axes, y_values: np.ndarray, color: ManimColor, width: float = 4.0, opacity: float = 1.0) -> VMobject:
        points = [axes.c2p(float(x), float(y)) for x, y in zip(self.grid, y_values)]
        curve = VMobject(color=color)
        curve.set_points_smoothly(points)
        curve.set_stroke(width=width, opacity=opacity)
        return curve

    def make_rug(self, axes: Axes, samples: np.ndarray | None = None, color: ManimColor = BLUE_DATA) -> VGroup:
        if samples is None:
            samples = self.samples
        return VGroup(
            *[
                Line(
                    axes.c2p(float(x), 0.0),
                    axes.c2p(float(x), 0.45),
                    color=color,
                    stroke_width=3,
                )
                for x in samples
            ]
        )

    def make_histogram(self, axes: Axes, bin_width: float) -> VGroup:
        bins = np.arange(0.0, 1.0001 + bin_width, bin_width)
        counts, edges = np.histogram(self.samples, bins=bins)
        bars = VGroup()
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            height = count / (len(self.samples) * (right - left))
            scene_width = abs(axes.c2p(float(right), 0)[0] - axes.c2p(float(left), 0)[0]) * 0.94
            scene_height = abs(axes.c2p(0, float(height))[1] - axes.c2p(0, 0)[1])
            bar = Rectangle(
                width=scene_width,
                height=max(scene_height, 0.001),
                stroke_color=BLUE_E,
                stroke_width=1.5,
                fill_color=BLUE_D,
                fill_opacity=0.55,
            )
            bar.move_to(axes.c2p(float((left + right) / 2.0), float(height / 2.0)))
            bars.add(bar)
        return bars

    def make_slider(self, label: str, values: list[str], index: int, color: ManimColor = MODEL_RED) -> tuple[VGroup, Dot]:
        line = Line(LEFT * 2.1, RIGHT * 2.1, color=GREY_B, stroke_width=4)
        ticks = VGroup()
        labels = VGroup()
        for i, value in enumerate(values):
            point = line.point_from_proportion(i / (len(values) - 1))
            ticks.add(Line(point + DOWN * 0.07, point + UP * 0.07, color=GREY_B, stroke_width=3))
            labels.add(Text(value, font_size=17, color=TEXT_GREY).next_to(point, DOWN, buff=0.12))
        knob = Dot(line.point_from_proportion(index / (len(values) - 1)), color=color, radius=0.09)
        title = Text(label, font_size=21, color=color).next_to(line, UP, buff=0.15)
        return VGroup(line, ticks, labels, title), knob

    def opening_parametric_limit(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 2.5 Nonparametric Methods")
        title = self.scene_title("形を決めすぎると、見えない山がある", font_size=34)
        axes = self.make_density_axes().shift(DOWN * 0.35)
        true_curve = self.make_density_curve(axes, true_density(self.grid), TRUE_GREEN, width=4.2)
        gaussian_curve = self.make_density_curve(axes, single_gaussian_fit(self.samples, self.grid), MODEL_RED, width=4.2)
        rug = self.make_rug(axes)
        legend = VGroup(
            Line(LEFT * 0.35, RIGHT * 0.35, color=TRUE_GREEN, stroke_width=5),
            Text("2つの山を持つデータ生成分布", font_size=20),
            Line(LEFT * 0.35, RIGHT * 0.35, color=MODEL_RED, stroke_width=5),
            Text("1つのガウスで近似", font_size=20),
        ).arrange_in_grid(rows=2, cols=2, col_alignments="lr", buff=(0.25, 0.15))
        legend.to_corner(UR).shift(DOWN * 1.05 + LEFT * 0.15)
        note = Text("parametric: 形を先に固定", font_size=26, color=MODEL_RED).next_to(axes, DOWN, buff=0.2)

        self.play(FadeIn(label), Write(title), run_time=1.6)
        self.play(Create(axes), FadeIn(rug), run_time=1.6)
        self.play(Create(true_curve), FadeIn(legend[0:2]), run_time=1.8)
        self.play(Create(gaussian_curve), FadeIn(legend[2:4]), Write(note), run_time=2.2)
        self.play(Indicate(gaussian_curve, color=YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, true_curve, gaussian_curve, rug, legend, note)), run_time=0.8)

    def histogram_bandwidth(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Histogram density estimation")
        title = self.scene_title("ビン幅 Delta が、見える形を決める", font_size=34)
        axes = self.make_density_axes().shift(DOWN * 0.25)
        true_curve = self.make_density_curve(axes, true_density(self.grid), TRUE_GREEN, width=3.5, opacity=0.75)
        rug = self.make_rug(axes)
        formula = MathTex(r"p_i=\frac{n_i}{N\Delta_i}", font_size=34, color=WHITE).to_corner(UR).shift(DOWN * 0.55)
        values = ["0.04", "0.08", "0.25"]
        slider, knob = self.make_slider("bin width Delta", values, 0, TEST_ORANGE)
        slider_group = VGroup(slider, knob).next_to(axes, DOWN, buff=0.2)

        hist_small = self.make_histogram(axes, 0.04)
        hist_mid = self.make_histogram(axes, 0.08)
        hist_large = self.make_histogram(axes, 0.25)
        captions = [
            Text("小さすぎる: 偶然のギザギザまで拾う", font_size=24, color=YELLOW),
            Text("中間: 二つの山が見えやすい", font_size=24, color=TRUE_GREEN),
            Text("大きすぎる: 山がつぶれる", font_size=24, color=YELLOW),
        ]
        for caption in captions:
            caption.next_to(slider_group, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), Create(axes), Create(true_curve), FadeIn(rug), Write(formula), run_time=2.0)
        self.play(FadeIn(hist_small), FadeIn(slider_group), Write(captions[0]), run_time=1.8)
        self.play(
            ReplacementTransform(hist_small, hist_mid),
            knob.animate.move_to(slider[0].point_from_proportion(0.5)),
            ReplacementTransform(captions[0], captions[1]),
            run_time=2.0,
        )
        self.play(
            ReplacementTransform(hist_mid, hist_large),
            knob.animate.move_to(slider[0].point_from_proportion(1.0)),
            ReplacementTransform(captions[1], captions[2]),
            run_time=2.0,
        )
        self.play(
            ReplacementTransform(hist_large, hist_mid.copy()),
            knob.animate.move_to(slider[0].point_from_proportion(0.5)),
            ReplacementTransform(captions[2], captions[1].copy()),
            run_time=1.5,
        )
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def local_counting_rule(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Local counting")
        title = self.scene_title("近くの点を数えて、体積で割る", font_size=34)
        axes = self.make_density_axes().shift(DOWN * 0.2)
        rug = self.make_rug(axes)
        x0 = 0.68
        width = 0.18
        left, right = x0 - width / 2.0, x0 + width / 2.0
        region = Rectangle(
            width=abs(axes.c2p(right, 0)[0] - axes.c2p(left, 0)[0]),
            height=abs(axes.c2p(0, 7.2)[1] - axes.c2p(0, 0)[1]),
            stroke_color=TEST_ORANGE,
            fill_color=TEST_ORANGE,
            fill_opacity=0.16,
            stroke_width=3,
        ).move_to(axes.c2p(x0, 3.6))
        target = RegularPolygon(n=4, radius=0.11, color=WHITE, fill_opacity=1).rotate(PI / 4)
        target.move_to(axes.c2p(x0, 0.15))
        inside_count = int(np.sum((self.samples >= left) & (self.samples <= right)))
        count_text = Text(f"K = {inside_count} points", font_size=25, color=TEST_ORANGE).next_to(region, UP, buff=0.2)
        volume_text = MathTex(r"V=\mathrm{width}(R)", font_size=32, color=TEXT_GREY).next_to(region, RIGHT, buff=0.35)
        formula = MathTex(r"p(x)\simeq\frac{K}{NV}", font_size=42, color=WHITE).to_corner(DR).shift(UP * 0.2 + LEFT * 0.25)
        caution = Text("小さい R: 局所的 / 大きい R: 安定", font_size=22, color=YELLOW)
        caution.to_corner(DL).shift(UP * 0.45 + RIGHT * 0.2)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(rug), run_time=1.7)
        self.play(FadeIn(target, scale=0.8), FadeIn(region), run_time=1.5)
        self.play(Write(count_text), Write(volume_text), run_time=1.5)
        self.play(Write(formula), Write(caution), run_time=1.8)
        self.play(region.animate.stretch_to_fit_width(region.width * 0.55), run_time=0.9)
        self.play(region.animate.stretch_to_fit_width(region.width * 1.9), run_time=0.9)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, rug, region, target, count_text, volume_text, formula, caution)), run_time=0.8)

    def kernel_density_estimator(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Kernel density estimator")
        title = self.scene_title("各データ点に山を置いて足し合わせる", font_size=33)
        axes = self.make_density_axes().shift(DOWN * 0.25)
        rug = self.make_rug(axes)
        formula = MathTex(r"p(x)=\frac{1}{N}\sum_{n=1}^{N}\mathcal{N}(x|x_n,h^2)", font_size=32)
        formula.to_corner(UR).shift(DOWN * 0.55)
        values = ["0.025", "0.07", "0.18"]
        slider, knob = self.make_slider("bandwidth h", values, 0, MODEL_RED)
        slider_group = VGroup(slider, knob).next_to(axes, DOWN, buff=0.2)

        h_values = [0.025, 0.07, 0.18]
        curves = [self.make_density_curve(axes, kde_density(self.samples, self.grid, h), MODEL_RED) for h in h_values]
        true_curve = self.make_density_curve(axes, true_density(self.grid), TRUE_GREEN, width=3, opacity=0.45)
        kernel_samples = self.samples[::5]
        kernels = VGroup(
            *[
                self.make_density_curve(
                    axes,
                    gaussian_pdf(self.grid, float(x), h_values[0]) / len(self.samples),
                    BLUE_E,
                    width=1.4,
                    opacity=0.35,
                )
                for x in kernel_samples
            ]
        )
        captions = [
            Text("小さい h: 点ごとの針になる", font_size=24, color=YELLOW),
            Text("中間の h: 2つの山を残して滑らか", font_size=24, color=TRUE_GREEN),
            Text("大きい h: 谷が埋まる", font_size=24, color=YELLOW),
        ]
        for caption in captions:
            caption.next_to(slider_group, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), Create(axes), Create(true_curve), FadeIn(rug), Write(formula), run_time=2.0)
        self.play(FadeIn(kernels), Create(curves[0]), FadeIn(slider_group), Write(captions[0]), run_time=2.1)
        self.play(
            ReplacementTransform(curves[0], curves[1]),
            knob.animate.move_to(slider[0].point_from_proportion(0.5)),
            ReplacementTransform(captions[0], captions[1]),
            FadeOut(kernels),
            run_time=2.0,
        )
        self.play(
            ReplacementTransform(curves[1], curves[2]),
            knob.animate.move_to(slider[0].point_from_proportion(1.0)),
            ReplacementTransform(captions[1], captions[2]),
            run_time=2.0,
        )
        self.play(
            ReplacementTransform(curves[2], curves[1].copy()),
            knob.animate.move_to(slider[0].point_from_proportion(0.5)),
            ReplacementTransform(captions[2], captions[1].copy()),
            run_time=1.4,
        )
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def knn_density_estimator(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("K-nearest-neighbour density")
        title = self.scene_title("K 個集まるまで、近傍を広げる", font_size=34)
        axes = self.make_density_axes().shift(DOWN * 0.25)
        rug = self.make_rug(axes)
        formula = MathTex(r"p(x)\simeq\frac{K}{N\,V(x)}", font_size=40).to_corner(UR).shift(DOWN * 0.55)
        curve = self.make_density_curve(axes, knn_density(self.samples, self.grid, 5), KNN_PURPLE, width=4)
        k_text = Text("K = 5", font_size=28, color=KNN_PURPLE).next_to(formula, DOWN, buff=0.2)
        positions = [0.29, 0.50, 0.72]
        intervals = []
        markers = []
        labels = []
        for x0 in positions:
            distances = np.sort(np.abs(self.samples - x0))
            radius = float(distances[4])
            interval = Rectangle(
                width=abs(axes.c2p(x0 + radius, 0)[0] - axes.c2p(x0 - radius, 0)[0]),
                height=abs(axes.c2p(0, 1.05)[1] - axes.c2p(0, 0)[1]),
                stroke_color=TEST_ORANGE,
                fill_color=TEST_ORANGE,
                fill_opacity=0.18,
                stroke_width=3,
            ).move_to(axes.c2p(x0, 0.55))
            marker = RegularPolygon(n=4, radius=0.1, color=WHITE, fill_opacity=1).rotate(PI / 4).move_to(axes.c2p(x0, 0.15))
            text = Text(f"radius {radius:.2f}", font_size=21, color=TEST_ORANGE).next_to(interval, UP, buff=0.12)
            intervals.append(interval)
            markers.append(marker)
            labels.append(text)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(rug), Write(formula), Write(k_text), run_time=2.0)
        self.play(Create(curve), run_time=1.5)
        current = VGroup(intervals[0], markers[0], labels[0])
        self.play(FadeIn(current), run_time=1.3)
        for interval, marker, text in zip(intervals[1:], markers[1:], labels[1:]):
            nxt = VGroup(interval, marker, text)
            self.play(ReplacementTransform(current, nxt), run_time=1.5)
            current = nxt
        small_k = self.make_density_curve(axes, knn_density(self.samples, self.grid, 1), YELLOW, width=3, opacity=0.85)
        large_k = self.make_density_curve(axes, knn_density(self.samples, self.grid, 24), BLUE_B, width=3, opacity=0.85)
        note = Text("K が小さいほど鋭く、大きいほど滑らか", font_size=25, color=WHITE).next_to(axes, DOWN, buff=0.25)
        self.play(Create(small_k), Create(large_k), Write(note), run_time=2.0)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, rug, formula, k_text, curve, current, small_k, large_k, note)), run_time=0.8)

    def make_classification_axes(self) -> Axes:
        return Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=7.2,
            y_length=4.8,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_class_data(self) -> tuple[np.ndarray, np.ndarray]:
        red = np.array([
            [-2.2, 0.8], [-1.9, 1.2], [-1.4, 0.4], [-0.9, 0.9],
            [-0.4, 0.25], [0.1, 0.85], [0.5, 0.4], [0.9, 1.1],
        ])
        blue = np.array([
            [-1.7, -1.0], [-1.1, -0.55], [-0.4, -1.2], [0.2, -0.45],
            [0.8, -1.0], [1.2, -0.25], [1.7, -0.75], [2.1, 0.05],
        ])
        return red, blue

    def knn_classifier(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("K-nearest-neighbour classification")
        title = self.scene_title("近くの K 個で多数決する", font_size=34)
        axes = self.make_classification_axes().shift(DOWN * 0.1)
        red, blue = self.make_class_data()
        red_dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.07, color=MODEL_RED) for x, y in red])
        blue_dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.07, color=BLUE_DATA) for x, y in blue])
        test_point = np.array([0.55, 0.12])
        test_dot = RegularPolygon(n=4, radius=0.12, color=WHITE, fill_opacity=1).rotate(PI / 4).move_to(axes.c2p(*test_point))
        all_points = np.vstack([red, blue])
        all_labels = np.array(["red"] * len(red) + ["blue"] * len(blue))
        distances = np.linalg.norm(all_points - test_point, axis=1)
        order = np.argsort(distances)
        k = 5
        radius = float(distances[order[k - 1]])
        scene_radius = np.linalg.norm(axes.c2p(test_point[0] + radius, test_point[1]) - axes.c2p(*test_point))
        circle = Circle(radius=scene_radius, color=TEST_ORANGE, stroke_width=4).move_to(axes.c2p(*test_point))
        red_count = int(np.sum(all_labels[order[:k]] == "red"))
        blue_count = k - red_count
        vote = VGroup(
            Text("K = 5", font_size=25, color=TEST_ORANGE),
            Text(f"red {red_count}", font_size=25, color=MODEL_RED),
            Text(f"blue {blue_count}", font_size=25, color=BLUE_DATA),
            Text("=> red class", font_size=28, color=MODEL_RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        vote.to_corner(UR).shift(DOWN * 0.75)
        posterior = MathTex(r"p(C_k|x)=\frac{K_k}{K}", font_size=40).next_to(vote, DOWN, buff=0.35)
        boundary_left = VGroup(
            Rectangle(width=2.6, height=1.35, fill_color=MODEL_RED, fill_opacity=0.18, stroke_color=GREY_B),
            Text("K=1", font_size=24),
            Text("細かい境界", font_size=20, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.08)
        boundary_right = VGroup(
            Rectangle(width=2.6, height=1.35, fill_color=BLUE_DATA, fill_opacity=0.16, stroke_color=GREY_B),
            Text("K=11", font_size=24),
            Text("滑らかな境界", font_size=20, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.08)
        boundary_pair = VGroup(boundary_left, boundary_right).arrange(RIGHT, buff=0.45).to_edge(DOWN).shift(UP * 0.05)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(red_dots), FadeIn(blue_dots), run_time=2.0)
        self.play(FadeIn(test_dot, scale=0.8), Create(circle), run_time=1.6)
        highlights = VGroup(
            *[
                Dot(axes.c2p(*all_points[i]), radius=0.105, color=YELLOW, fill_opacity=0.45)
                for i in order[:k]
            ]
        )
        self.play(FadeIn(highlights), Write(vote), run_time=1.8)
        self.play(Write(posterior), run_time=1.3)
        self.play(FadeIn(boundary_pair), run_time=1.6)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, red_dots, blue_dots, test_dot, circle, highlights, vote, posterior, boundary_pair)), run_time=0.8)

    def summary_tradeoffs(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Summary")
        title = self.scene_title("柔軟さは、データ保持と計算コストを連れてくる", font_size=32)
        headers = ["Histogram", "KDE", "KNN"]
        top = VGroup()
        for header, color in zip(headers, [BLUE_DATA, MODEL_RED, KNN_PURPLE]):
            box = Rectangle(width=3.1, height=1.0, stroke_color=color, fill_color=color, fill_opacity=0.18)
            text = Text(header, font_size=28, color=color)
            top.add(VGroup(box, text).move_to(box))
        top.arrange(RIGHT, buff=0.35).shift(UP * 1.1)
        points = VGroup(
            Text("形を先に強く固定しない", font_size=27, color=TRUE_GREEN),
            Text("局所近傍と平滑化パラメータが鍵", font_size=27, color=WHITE),
            Text("KDE / KNN は訓練データを評価時にも参照", font_size=27, color=YELLOW),
            Text("次: 柔軟さと複雑さ制御を両立するモデルへ", font_size=27, color=TEST_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        points.next_to(top, DOWN, buff=0.55)
        storage = VGroup(
            *[Square(side_length=0.22, fill_color=GREY_B, fill_opacity=0.7, stroke_width=0) for _ in range(40)]
        ).arrange_in_grid(rows=4, cols=10, buff=0.04)
        storage.next_to(points, DOWN, buff=0.45)
        storage_label = Text("stored data", font_size=21, color=TEXT_GREY).next_to(storage, DOWN, buff=0.12)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.2) for item in top], lag_ratio=0.15), run_time=1.5)
        self.play(LaggedStart(*[Write(item) for item in points], lag_ratio=0.22), run_time=3.0)
        self.play(FadeIn(storage), Write(storage_label), run_time=1.4)
        self.play(Indicate(storage, color=YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, top, points, storage, storage_label)), run_time=0.8)
