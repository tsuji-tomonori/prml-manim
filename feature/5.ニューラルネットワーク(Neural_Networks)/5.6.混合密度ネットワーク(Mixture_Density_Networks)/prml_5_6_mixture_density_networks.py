from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *
from PIL import Image


BLUE_DATA = BLUE_C
GREEN_CURVE = GREEN_C
RED_MODEL = RED_C
ORANGE_ALT = ORANGE
YELLOW_NOTE = YELLOW_C
PURPLE_COMPONENT = PURPLE_C
TEAL_DENSITY = TEAL_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def normal_pdf(x: np.ndarray | float, mu: np.ndarray | float = 0.0, sigma: np.ndarray | float = 1.0) -> np.ndarray | float:
    x_array = np.asarray(x)
    return np.exp(-0.5 * ((x_array - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=0, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


def branch_means(x: np.ndarray | float) -> np.ndarray:
    x_array = np.asarray(x)
    upper = 0.74 - 0.18 * np.cos(2 * np.pi * x_array)
    middle = 0.50 + 0.08 * np.sin(2 * np.pi * x_array)
    lower = 0.26 + 0.18 * np.cos(2 * np.pi * x_array)
    return np.array([upper, middle, lower])


def branch_sigmas(x: np.ndarray | float) -> np.ndarray:
    x_array = np.asarray(x)
    base = 0.035 + 0.018 * np.sin(np.pi * x_array) ** 2
    return np.array([base * 1.05, base * 0.92, base * 1.12])


def branch_logits(x: np.ndarray | float) -> np.ndarray:
    x_array = np.asarray(x)
    middle_gate = np.exp(-((x_array - 0.5) / 0.22) ** 2)
    left_gate = np.exp(-(x_array / 0.18) ** 2)
    right_gate = np.exp(-((x_array - 1.0) / 0.18) ** 2)
    return np.array([
        1.6 * left_gate + 0.3 * middle_gate + 0.25,
        1.35 * middle_gate + 0.2,
        1.6 * right_gate + 0.3 * middle_gate + 0.25,
    ])


def branch_weights(x: np.ndarray | float) -> np.ndarray:
    return softmax(branch_logits(x))


def mdn_density(t: np.ndarray | float, x: np.ndarray | float) -> np.ndarray | float:
    weights = branch_weights(x)
    means = branch_means(x)
    sigmas = branch_sigmas(x)
    return np.sum(weights * normal_pdf(t, means, sigmas), axis=0)


def mean_prediction(x: np.ndarray | float) -> np.ndarray | float:
    return np.sum(branch_weights(x) * branch_means(x), axis=0)


def mode_approximation(x: np.ndarray | float) -> np.ndarray | float:
    weights = branch_weights(x)
    means = branch_means(x)
    return means[np.argmax(weights, axis=0), np.arange(np.asarray(x).size)] if np.asarray(x).ndim else means[int(np.argmax(weights))]


class PRML56MixtureDensityNetworks(Scene):
    """PRML 5.6 overview for a high-school math audience.

    Render example:
        uv run manim -pql prml_5_6_mixture_density_networks.py PRML56MixtureDensityNetworks
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.inverse_problem()
        self.conditional_mixture()
        self.network_parameterization()
        self.likelihood_training()
        self.density_map_scene()
        self.mean_and_mode()
        self.summary_scene()

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

    def unit_axes(self, width: float = 5.1, height: float = 3.5) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 1, 0.25],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_inverse_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(56)
        x = np.linspace(0.04, 0.96, 72)
        forward_t = x + 0.25 * np.sin(2 * np.pi * x) + rng.uniform(-0.055, 0.055, size=x.size)
        forward_t = np.clip(forward_t, 0.04, 0.96)
        return x, forward_t, rng.uniform(-0.012, 0.012, size=x.size)

    def make_density_image(self, width: int = 360, height: int = 260) -> ImageMobject:
        xs = np.linspace(0, 1, width)
        ts = np.linspace(1, 0, height)
        grid_x, grid_t = np.meshgrid(xs, ts)
        density = mdn_density(grid_t, grid_x)
        density = density / density.max()
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = (20 + 35 * density).astype(np.uint8)
        rgba[..., 1] = (40 + 185 * density).astype(np.uint8)
        rgba[..., 2] = (55 + 150 * np.sqrt(density)).astype(np.uint8)
        rgba[..., 3] = (45 + 210 * density).astype(np.uint8)
        return ImageMobject(Image.fromarray(rgba, mode="RGBA"))

    def line_graph(self, axes: Axes, values: np.ndarray, color: ManimColor, width: float = 3.4) -> VMobject:
        xs = np.linspace(0, 1, len(values))
        points = [axes.c2p(float(x), float(y)) for x, y in zip(xs, values)]
        graph = VMobject(color=color)
        graph.set_points_smoothly(points)
        graph.set_stroke(width=width)
        return graph

    def inverse_problem(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 5.6 / Fig. 5.18, 5.19")
        title = self.scene_title("逆問題では、正解が一つとは限らない", font_size=33)
        left_axes = self.unit_axes(width=4.7, height=3.35).shift(LEFT * 3.05 + DOWN * 0.15)
        right_axes = self.unit_axes(width=4.7, height=3.35).shift(RIGHT * 3.05 + DOWN * 0.15)
        x, forward_t, jitter = self.make_inverse_data()
        forward_dots = VGroup(*[Dot(left_axes.c2p(float(a), float(b)), radius=0.035, color=BLUE_DATA) for a, b in zip(x, forward_t)])
        inverse_dots = VGroup(*[Dot(right_axes.c2p(float(b), float(a + j)), radius=0.035, color=BLUE_DATA) for a, b, j in zip(x, forward_t, jitter)])
        xs = np.linspace(0, 1, 240)
        forward_curve = left_axes.plot(lambda z: z + 0.25 * math.sin(2 * math.pi * z), x_range=[0, 1], color=GREEN_CURVE).set_stroke(width=4)
        single_mean = right_axes.plot(lambda z: 0.5 + 0.05 * math.sin(2 * math.pi * z), x_range=[0, 1], color=RED_MODEL).set_stroke(width=4)
        good_branches = VGroup(
            self.line_graph(right_axes, branch_means(xs)[0], GREEN_CURVE, width=3),
            self.line_graph(right_axes, branch_means(xs)[1], ORANGE_ALT, width=3),
            self.line_graph(right_axes, branch_means(xs)[2], PURPLE_COMPONENT, width=3),
        )
        left_name = Text("forward: x -> t", font_size=25, color=GREEN_CURVE).next_to(left_axes, DOWN, buff=0.25)
        right_name = Text("inverse: t -> x", font_size=25, color=YELLOW_NOTE).next_to(right_axes, DOWN, buff=0.25)
        bad_note = Text("平均だけでは谷を通る", font_size=25, color=RED_MODEL).next_to(right_axes, UP, buff=0.15)

        self.play(FadeIn(label), Write(title), Create(left_axes), Create(right_axes), Write(left_name), Write(right_name))
        self.play(FadeIn(forward_dots, lag_ratio=0.01), Create(forward_curve), run_time=1.2)
        self.play(TransformFromCopy(forward_dots, inverse_dots), run_time=1.2)
        self.play(Create(single_mean), Write(bad_note))
        self.play(FadeIn(good_branches), run_time=1.1)
        self.wait(0.7)
        self.finish_narration(narration)
        self.play(*[FadeOut(mob) for mob in [label, title, left_axes, right_axes, forward_dots, inverse_dots, forward_curve, single_mean, good_branches, left_name, right_name, bad_note]])

    def conditional_mixture(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("PRML 5.6 / Eq. (5.148)")
        title = self.scene_title("p(t|x) を、入力で変わるガウス混合として表す", font_size=31)
        formula = MathTex(
            r"p(t|x)=\sum_{k=1}^{K}\pi_k(x)\,\mathcal{N}\left(t|\mu_k(x),\sigma_k^2(x)\right)",
            font_size=38,
        ).to_edge(UP).shift(DOWN * 1.2)
        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 7, 1],
            x_length=7.0,
            y_length=3.3,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.75)
        x0 = 0.5
        ts = np.linspace(0, 1, 280)
        colors = [GREEN_CURVE, ORANGE_ALT, PURPLE_COMPONENT]
        components = VGroup()
        for k, color in enumerate(colors):
            curve = axes.plot(
                lambda z, kk=k: float(branch_weights(x0)[kk] * normal_pdf(z, branch_means(x0)[kk], branch_sigmas(x0)[kk])),
                x_range=[0.02, 0.98],
                color=color,
            ).set_stroke(width=3)
            components.add(curve)
        mixture = axes.plot(lambda z: float(mdn_density(z, x0)), x_range=[0.02, 0.98], color=YELLOW_NOTE).set_stroke(width=5)
        x_note = MathTex(r"x=0.5", font_size=34, color=YELLOW_NOTE).next_to(axes, RIGHT, buff=0.4).shift(UP * 0.75)
        pi_note = MathTex(r"\pi_k(x):\ sum\ to\ 1", font_size=28, color=GREEN_CURVE).next_to(x_note, DOWN, buff=0.25)
        mu_note = MathTex(r"\mu_k(x):\ centers", font_size=28, color=ORANGE_ALT).next_to(pi_note, DOWN, buff=0.18)
        sigma_note = MathTex(r"\sigma_k(x):\ widths", font_size=28, color=PURPLE_COMPONENT).next_to(mu_note, DOWN, buff=0.18)

        self.play(FadeIn(label), Write(title), Write(formula), Create(axes))
        self.play(FadeIn(components, lag_ratio=0.15), Write(x_note))
        self.play(Create(mixture), Write(pi_note), Write(mu_note), Write(sigma_note))
        for new_x in [0.18, 0.82, 0.5]:
            new_components = VGroup()
            for k, color in enumerate(colors):
                new_components.add(axes.plot(
                    lambda z, kk=k, xx=new_x: float(branch_weights(xx)[kk] * normal_pdf(z, branch_means(xx)[kk], branch_sigmas(xx)[kk])),
                    x_range=[0.02, 0.98],
                    color=color,
                ).set_stroke(width=3))
            new_mixture = axes.plot(lambda z, xx=new_x: float(mdn_density(z, xx)), x_range=[0.02, 0.98], color=YELLOW_NOTE).set_stroke(width=5)
            new_note = MathTex(f"x={new_x:.2f}", font_size=34, color=YELLOW_NOTE).move_to(x_note)
            self.play(Transform(components, new_components), Transform(mixture, new_mixture), Transform(x_note, new_note), run_time=1.25)
        self.wait(0.6)
        self.finish_narration(narration)
        self.play(*[FadeOut(mob) for mob in [label, title, formula, axes, components, mixture, x_note, pi_note, mu_note, sigma_note]])

    def network_parameterization(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("PRML 5.6 / Fig. 5.20 / Eq. (5.149)-(5.152)")
        title = self.scene_title("ネットワーク出力を、分布パラメータへ変換する", font_size=32)
        input_box = RoundedRectangle(width=1.25, height=0.75, corner_radius=0.08, color=BLUE_DATA).shift(LEFT * 5.0)
        input_text = MathTex(r"x", font_size=38).move_to(input_box)
        hidden = VGroup(*[Circle(radius=0.22, color=TEAL_DENSITY).shift(LEFT * 2.9 + UP * y) for y in [-1.0, -0.5, 0, 0.5, 1.0]])
        output_boxes = VGroup()
        specs = [
            (r"a^\pi_k", r"\pi_k(x)=\frac{e^{a^\pi_k}}{\sum_l e^{a^\pi_l}}", GREEN_CURVE),
            (r"a^\sigma_k", r"\sigma_k(x)=e^{a^\sigma_k}", ORANGE_ALT),
            (r"a^\mu_{kj}", r"\mu_{kj}(x)=a^\mu_{kj}", PURPLE_COMPONENT),
        ]
        for i, (raw, converted, color) in enumerate(specs):
            box = RoundedRectangle(width=2.1, height=0.65, corner_radius=0.08, color=color).shift(RIGHT * 0.3 + UP * (1.0 - i))
            raw_text = MathTex(raw, font_size=31, color=color).move_to(box)
            eq = MathTex(converted, font_size=29, color=color).next_to(box, RIGHT, buff=0.45)
            output_boxes.add(VGroup(box, raw_text, eq))
        arrows = VGroup()
        for node in hidden:
            arrows.add(Line(input_box.get_right(), node.get_left(), stroke_width=1.5, color=GREY_B))
        for node in hidden:
            for box_group in output_boxes:
                arrows.add(Line(node.get_right(), box_group[0].get_left(), stroke_width=1.0, color=GREY_B))
        count_note = MathTex(r"K\ components,\ L\ target\ dims\ \Rightarrow\ (L+2)K\ outputs", font_size=31, color=YELLOW_NOTE).to_edge(DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(input_box), Write(input_text), FadeIn(hidden), FadeIn(arrows), run_time=1.0)
        for group in output_boxes:
            self.play(FadeIn(group[0]), Write(group[1]), Write(group[2]), run_time=0.75)
        self.play(Write(count_note))
        self.wait(0.7)
        self.finish_narration(narration)
        self.play(*[FadeOut(mob) for mob in [label, title, input_box, input_text, hidden, arrows, output_boxes, count_note]])

    def likelihood_training(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("PRML 5.6 / Eq. (5.153)-(5.157)")
        title = self.scene_title("二乗誤差ではなく、混合分布の尤度を最大化する", font_size=30)
        formula = MathTex(
            r"E(w)=-\sum_{n=1}^{N}\ln\left\{\sum_{k=1}^{K}\pi_k(x_n,w)"
            r"\mathcal{N}(t_n|\mu_k(x_n,w),\sigma_k^2(x_n,w))\right\}",
            font_size=31,
        ).to_edge(UP).shift(DOWN * 1.1)
        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 8, 1],
            x_length=5.5,
            y_length=3.1,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 2.65 + DOWN * 0.8)
        x0 = 0.5
        t0 = float(branch_means(x0)[0] + 0.012)
        colors = [GREEN_CURVE, ORANGE_ALT, PURPLE_COMPONENT]
        components = VGroup(*[
            axes.plot(
                lambda z, kk=k: float(branch_weights(x0)[kk] * normal_pdf(z, branch_means(x0)[kk], branch_sigmas(x0)[kk])),
                x_range=[0.02, 0.98],
                color=color,
            ).set_stroke(width=3)
            for k, color in enumerate(colors)
        ])
        point = Dot(axes.c2p(t0, 0), radius=0.075, color=YELLOW_NOTE)
        point_line = DashedLine(axes.c2p(t0, 0), axes.c2p(t0, 7.5), color=YELLOW_NOTE, dash_length=0.08)
        likelihoods = branch_weights(x0) * normal_pdf(t0, branch_means(x0), branch_sigmas(x0))
        gamma = likelihoods / likelihoods.sum()
        bars = VGroup()
        for i, (g, color) in enumerate(zip(gamma, colors)):
            base = RIGHT * 2.05 + DOWN * (0.35 + i * 0.55)
            bar_bg = Rectangle(width=2.6, height=0.22, color=GREY_B, stroke_width=1).move_to(base)
            bar_fg = Rectangle(width=2.6 * float(g), height=0.22, color=color, stroke_width=0).set_fill(color, opacity=0.8)
            bar_fg.align_to(bar_bg, LEFT).move_to(bar_bg.get_left() + RIGHT * bar_fg.width / 2)
            label_text = MathTex(rf"\gamma_{i+1}={g:.2f}", font_size=26, color=color).next_to(bar_bg, LEFT, buff=0.18)
            bars.add(VGroup(bar_bg, bar_fg, label_text))
        gamma_formula = MathTex(r"\gamma_k(t|x)=\frac{\pi_k\mathcal{N}_k}{\sum_l\pi_l\mathcal{N}_l}", font_size=34).next_to(bars, UP, buff=0.45)
        backprop = Text("責務を誤差信号として逆伝播", font_size=25, color=YELLOW_NOTE).next_to(bars, DOWN, buff=0.35)

        self.play(FadeIn(label), Write(title), Write(formula))
        self.play(Create(axes), FadeIn(components, lag_ratio=0.1))
        self.play(FadeIn(point), Create(point_line))
        self.play(Write(gamma_formula), FadeIn(bars, lag_ratio=0.12), Write(backprop))
        self.wait(0.7)
        self.finish_narration(narration)
        self.play(*[FadeOut(mob) for mob in [label, title, formula, axes, components, point, point_line, gamma_formula, bars, backprop]])

    def density_map_scene(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("PRML 5.6 / Fig. 5.21(a)-(c)")
        title = self.scene_title("連続な出力関数から、多峰な条件付き密度が生まれる", font_size=30)
        axes = self.unit_axes(width=7.0, height=4.0).shift(LEFT * 1.65 + DOWN * 0.2)
        density_image = self.make_density_image()
        density_image.set_width(axes.x_length)
        density_image.set_height(axes.y_length)
        density_image.move_to(axes.c2p(0.5, 0.5))
        xs = np.linspace(0, 1, 240)
        mean_curves = VGroup(
            self.line_graph(axes, branch_means(xs)[0], GREEN_CURVE, width=2.4),
            self.line_graph(axes, branch_means(xs)[1], ORANGE_ALT, width=2.4),
            self.line_graph(axes, branch_means(xs)[2], PURPLE_COMPONENT, width=2.4),
        )
        slices = VGroup()
        for x0 in [0.15, 0.5, 0.85]:
            slices.add(DashedLine(axes.c2p(x0, 0), axes.c2p(x0, 1), color=YELLOW_NOTE, dash_length=0.08))
        notes = VGroup(
            Text("端: ほぼ単峰", font_size=24, color=YELLOW_NOTE).next_to(axes, RIGHT, buff=0.4).shift(UP * 0.85),
            Text("中央: 三つの候補", font_size=24, color=YELLOW_NOTE).next_to(axes, RIGHT, buff=0.4).shift(UP * 0.35),
            MathTex(r"\pi_k(x),\mu_k(x),\sigma_k(x)", font_size=30, color=TEXT_GREY).next_to(axes, RIGHT, buff=0.4).shift(DOWN * 0.3),
        )

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(FadeIn(density_image), run_time=1.0)
        self.play(FadeIn(mean_curves, lag_ratio=0.12), FadeIn(slices, lag_ratio=0.12), FadeIn(notes, lag_ratio=0.12))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(*[FadeOut(mob) for mob in [label, title, axes, density_image, mean_curves, slices, notes]])

    def mean_and_mode(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("PRML 5.6 / Eq. (5.158)-(5.160)")
        title = self.scene_title("分布を得たあと、用途に合う要約を選ぶ", font_size=32)
        axes = self.unit_axes(width=7.0, height=4.0).shift(LEFT * 1.65 + DOWN * 0.2)
        density_image = self.make_density_image()
        density_image.set_width(axes.x_length)
        density_image.set_height(axes.y_length)
        density_image.move_to(axes.c2p(0.5, 0.5))
        xs = np.linspace(0, 1, 260)
        mean_line = self.line_graph(axes, mean_prediction(xs), RED_MODEL, width=4)
        mode_line = self.line_graph(axes, mode_approximation(xs), YELLOW_NOTE, width=4)
        mean_formula = MathTex(r"\mathbb{E}[t|x]=\sum_k \pi_k(x)\mu_k(x)", font_size=31, color=RED_MODEL).next_to(axes, RIGHT, buff=0.35).shift(UP * 0.85)
        mode_note = Text("近似モード: 最大 pi の中心", font_size=24, color=YELLOW_NOTE).next_to(mean_formula, DOWN, buff=0.28)
        valley_note = Text("平均は谷を通ることがある", font_size=24, color=RED_MODEL).next_to(mode_note, DOWN, buff=0.25)
        sample_x = 0.5
        mean_dot = Dot(axes.c2p(sample_x, float(mean_prediction(sample_x))), radius=0.08, color=RED_MODEL)
        mode_dot = Dot(axes.c2p(sample_x, float(mode_approximation(sample_x))), radius=0.08, color=YELLOW_NOTE)
        sample_line = DashedLine(axes.c2p(sample_x, 0), axes.c2p(sample_x, 1), color=TEXT_GREY, dash_length=0.08)

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(density_image))
        self.play(Create(mean_line), Write(mean_formula), FadeIn(mean_dot), Create(sample_line))
        self.play(Create(mode_line), Write(mode_note), Write(valley_note), FadeIn(mode_dot))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(*[FadeOut(mob) for mob in [label, title, axes, density_image, mean_line, mode_line, mean_formula, mode_note, valley_note, mean_dot, mode_dot, sample_line]])

    def summary_scene(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("PRML 5.6")
        title = self.scene_title("混合密度ネットワークの要点", font_size=35)
        bullets = VGroup(
            Text("1. 回帰の出力を一点から p(t|x) へ拡張", font_size=29),
            Text("2. ネットワークが pi, mu, sigma を入力ごとに出す", font_size=29),
            Text("3. softmax と exp で分布の制約を満たす", font_size=29),
            Text("4. 負の対数尤度を通常の逆伝播で最小化", font_size=29),
            Text("5. 平均、分散、モードを用途に応じて選べる", font_size=29),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34).shift(LEFT * 0.2 + DOWN * 0.1)
        equation = MathTex(r"point\ prediction\quad\longrightarrow\quad conditional\ density", font_size=38, color=YELLOW_NOTE)
        equation.next_to(bullets, DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(bullets, lag_ratio=0.13), run_time=1.6)
        self.play(Write(equation))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(*[FadeOut(mob) for mob in [label, title, bullets, equation]])
