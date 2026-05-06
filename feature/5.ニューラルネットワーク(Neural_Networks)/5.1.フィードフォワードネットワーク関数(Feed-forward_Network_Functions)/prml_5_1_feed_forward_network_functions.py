from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


INPUT_BLUE = BLUE_C
HIDDEN_PURPLE = PURPLE_C
OUTPUT_GREEN = GREEN_C
MODEL_RED = RED_C
ACCENT_ORANGE = ORANGE
TARGET_YELLOW = YELLOW
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def tanh_unit(x: np.ndarray, weight: float, bias: float) -> np.ndarray:
    return np.tanh(weight * x + bias)


class PRML51FeedForwardNetworkFunctions(Scene):
    """PRML 5.1 feed-forward network functions overview.

    Render example:
        uv run manim -pql prml_5_1_feed_forward_network_functions.py PRML51FeedForwardNetworkFunctions
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.grid = np.linspace(-1.0, 1.0, 500)

        self.fixed_to_adaptive_basis()
        self.forward_propagation_function()
        self.hidden_units_as_basis()
        self.output_activation_by_task()
        self.nonlinearity_is_essential()
        self.feed_forward_topology()
        self.universal_approximation_view()
        self.weight_space_symmetry()

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

    def make_axes(self, width: float = 5.2, height: float = 3.0) -> Axes:
        return Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-1.4, 1.4, 0.7],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def curve_from_values(
        self,
        axes: Axes,
        values: np.ndarray,
        color: ManimColor,
        width: float = 3.2,
        opacity: float = 1.0,
        dashed: bool = False,
    ) -> VMobject:
        curve = VMobject(color=color)
        points = [axes.c2p(float(x), float(y)) for x, y in zip(self.grid, values)]
        curve.set_points_smoothly(points)
        curve.set_stroke(width=width, opacity=opacity)
        if dashed:
            return DashedVMobject(curve, num_dashes=46)
        return curve

    def node(self, label: str, position: np.ndarray, color: ManimColor, radius: float = 0.24) -> VGroup:
        circle = Circle(radius=radius, stroke_color=color, fill_color=color, fill_opacity=0.2, stroke_width=3)
        circle.move_to(position)
        text = Text(label, font_size=22, color=WHITE).move_to(circle)
        return VGroup(circle, text)

    def arrow_between(self, source: Mobject, target: Mobject, color: ManimColor = GREY_B, width: float = 2.0) -> Arrow:
        return Arrow(
            source.get_center(),
            target.get_center(),
            buff=0.28,
            color=color,
            stroke_width=width,
            max_tip_length_to_length_ratio=0.08,
        )

    def fixed_to_adaptive_basis(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 5.1 Feed-forward Network Functions")
        title = self.scene_title("固定基底を、学習される基底へ変える", font_size=33)

        fixed_eq = MathTex(r"y(x,\mathbf{w})=f\!\left(\sum_j w_j\phi_j(x)\right)", font_size=34)
        fixed_caption = Text("fixed basis", font_size=24, color=TEXT_GREY).next_to(fixed_eq, DOWN, buff=0.15)
        adaptive_eq = MathTex(r"y(x,\mathbf{w})=f\!\left(\sum_j w_j\phi_j(x;\theta_j)\right)", font_size=34)
        adaptive_caption = Text("adaptive hidden units", font_size=24, color=HIDDEN_PURPLE).next_to(adaptive_eq, DOWN, buff=0.15)
        left_card = VGroup(fixed_eq, fixed_caption).move_to(LEFT * 3.2 + UP * 1.0)
        right_card = VGroup(adaptive_eq, adaptive_caption).move_to(RIGHT * 3.0 + UP * 1.0)
        arrow = Arrow(left_card.get_right(), right_card.get_left(), buff=0.3, color=ACCENT_ORANGE)

        axes = self.make_axes(width=7.5, height=2.5).to_edge(DOWN).shift(UP * 0.25)
        fixed_curves = VGroup(
            self.curve_from_values(axes, np.exp(-18 * (self.grid + 0.55) ** 2), BLUE_B, width=2.2, opacity=0.55, dashed=True),
            self.curve_from_values(axes, np.exp(-18 * self.grid**2), BLUE_B, width=2.2, opacity=0.55, dashed=True),
            self.curve_from_values(axes, np.exp(-18 * (self.grid - 0.55) ** 2), BLUE_B, width=2.2, opacity=0.55, dashed=True),
        )
        adaptive_curves = VGroup(
            self.curve_from_values(axes, tanh_unit(self.grid, 4.0, 1.3) * 0.55, HIDDEN_PURPLE, width=2.5, opacity=0.7, dashed=True),
            self.curve_from_values(axes, tanh_unit(self.grid, -4.0, 0.2) * 0.45, HIDDEN_PURPLE, width=2.5, opacity=0.7, dashed=True),
            self.curve_from_values(axes, tanh_unit(self.grid, 3.0, -1.4) * 0.5, HIDDEN_PURPLE, width=2.5, opacity=0.7, dashed=True),
        )
        note = Text("特徴への変換も、重みとバイアスで動かす", font_size=25, color=ACCENT_ORANGE)
        note.next_to(axes, UP, buff=0.25)

        self.play(FadeIn(label), Write(title), run_time=1.3)
        self.play(Write(left_card), Create(axes), Create(fixed_curves), run_time=2.0)
        self.play(Create(arrow), Write(right_card), run_time=1.5)
        self.play(ReplacementTransform(fixed_curves, adaptive_curves), Write(note), run_time=2.2)
        self.play(Indicate(adaptive_caption, color=TARGET_YELLOW), run_time=1.0)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, left_card, right_card, arrow, axes, adaptive_curves, note)), run_time=0.8)

    def make_layered_network(self) -> tuple[VGroup, VGroup, VGroup, VGroup]:
        inputs = VGroup(
            self.node("x1", LEFT * 4.4 + UP * 1.15, INPUT_BLUE),
            self.node("x2", LEFT * 4.4, INPUT_BLUE),
            self.node("x0", LEFT * 4.4 + DOWN * 1.15, GREY_B, radius=0.2),
        )
        hidden = VGroup(
            self.node("z1", LEFT * 0.65 + UP * 1.35, HIDDEN_PURPLE),
            self.node("z2", LEFT * 0.65, HIDDEN_PURPLE),
            self.node("z3", LEFT * 0.65 + DOWN * 1.35, HIDDEN_PURPLE),
        )
        outputs = VGroup(
            self.node("y1", RIGHT * 3.7 + UP * 0.55, OUTPUT_GREEN),
            self.node("y2", RIGHT * 3.7 + DOWN * 0.55, OUTPUT_GREEN),
        )
        arrows = VGroup()
        for source in inputs:
            for target in hidden:
                arrows.add(self.arrow_between(source, target, GREY_C, width=1.6))
        for source in hidden:
            for target in outputs:
                arrows.add(self.arrow_between(source, target, GREY_C, width=1.8))
        return inputs, hidden, outputs, arrows

    def forward_propagation_function(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Forward propagation")
        title = self.scene_title("左から右へ、活性を順に計算する", font_size=34)
        inputs, hidden, outputs, arrows = self.make_layered_network()
        network = VGroup(arrows, inputs, hidden, outputs).shift(DOWN * 0.05)
        equations = VGroup(
            MathTex(r"a_j=\sum_i w_{ji}^{(1)}x_i+w_{j0}^{(1)}", font_size=31, color=HIDDEN_PURPLE),
            MathTex(r"z_j=h(a_j)", font_size=31, color=HIDDEN_PURPLE),
            MathTex(r"a_k=\sum_j w_{kj}^{(2)}z_j+w_{k0}^{(2)}", font_size=31, color=OUTPUT_GREEN),
            MathTex(r"y_k=\sigma(a_k)", font_size=31, color=OUTPUT_GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        equations.to_corner(DR).shift(UP * 0.25 + LEFT * 0.15)
        flow_text = Text("forward propagation", font_size=25, color=ACCENT_ORANGE).next_to(network, DOWN, buff=0.25)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(inputs), Create(arrows[: len(arrows) // 2]), run_time=1.6)
        self.play(FadeIn(hidden), Write(equations[0]), Write(equations[1]), run_time=1.8)
        self.play(Create(arrows[len(arrows) // 2 :]), FadeIn(outputs), run_time=1.5)
        self.play(Write(equations[2]), Write(equations[3]), run_time=1.7)
        self.play(
            LaggedStart(
                *[Indicate(group, color=ACCENT_ORANGE, scale_factor=1.08) for group in [inputs, hidden, outputs]],
                lag_ratio=0.35,
            ),
            Write(flow_text),
            run_time=2.0,
        )
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, network, equations, flow_text)), run_time=0.8)

    def hidden_units_as_basis(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Hidden units as adaptive basis functions")
        title = self.scene_title("単純な隠れユニットを足して、複雑な形を作る", font_size=32)
        axes = self.make_axes(width=8.2, height=4.0).shift(DOWN * 0.2)
        target = 0.65 * np.sin(math.pi * self.grid) + 0.15 * self.grid
        h1 = 0.55 * tanh_unit(self.grid, 4.2, 1.25)
        h2 = -0.95 * tanh_unit(self.grid, 4.0, 0.0)
        h3 = 0.55 * tanh_unit(self.grid, 4.2, -1.25)
        combined = 0.88 * (h1 + h2 + h3)
        initial = 0.45 * tanh_unit(self.grid, 1.4, 0.0)

        target_curve = self.curve_from_values(axes, target, TARGET_YELLOW, width=3.5, opacity=0.8)
        hidden_curves = VGroup(
            self.curve_from_values(axes, h1, BLUE_B, width=2.1, opacity=0.55, dashed=True),
            self.curve_from_values(axes, h2, PURPLE_B, width=2.1, opacity=0.55, dashed=True),
            self.curve_from_values(axes, h3, GREEN_B, width=2.1, opacity=0.55, dashed=True),
        )
        initial_curve = self.curve_from_values(axes, initial, MODEL_RED, width=3.4)
        combined_curve = self.curve_from_values(axes, combined, MODEL_RED, width=3.8)
        formula = MathTex(r"y(x)=\sum_j w_j^{(2)}\,h(w_j^{(1)}x+b_j)", font_size=30)
        formula.to_corner(DR).shift(UP * 0.45 + LEFT * 0.15)
        legend = VGroup(
            Line(LEFT * 0.3, RIGHT * 0.3, color=TARGET_YELLOW, stroke_width=4),
            Text("target", font_size=19),
            Line(LEFT * 0.3, RIGHT * 0.3, color=MODEL_RED, stroke_width=4),
            Text("network", font_size=19),
            DashedLine(LEFT * 0.3, RIGHT * 0.3, color=HIDDEN_PURPLE, stroke_width=3),
            Text("hidden units", font_size=19),
        ).arrange_in_grid(rows=3, cols=2, col_alignments="lr", buff=(0.18, 0.12))
        legend.to_corner(UL).shift(DOWN * 0.9 + RIGHT * 0.1)

        self.play(FadeIn(label), Write(title), Create(axes), Write(formula), run_time=1.8)
        self.play(Create(target_curve), FadeIn(legend[0:2]), run_time=1.3)
        self.play(Create(initial_curve), FadeIn(legend[2:4]), run_time=1.4)
        self.play(Create(hidden_curves), FadeIn(legend[4:6]), run_time=1.8)
        self.play(ReplacementTransform(initial_curve, combined_curve), run_time=2.1)
        self.play(Indicate(hidden_curves, color=TARGET_YELLOW), run_time=1.3)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, axes, formula, target_curve, hidden_curves, combined_curve, legend)), run_time=0.8)

    def output_activation_by_task(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Output activations")
        title = self.scene_title("最後の活性化は、予測したい量に合わせる", font_size=33)

        cards = VGroup()
        names = ["Regression", "Multiple binary", "Multiclass"]
        colors = [OUTPUT_GREEN, INPUT_BLUE, ACCENT_ORANGE]
        formulas = [
            MathTex(r"y_k=a_k", font_size=30, color=OUTPUT_GREEN),
            MathTex(r"y_k=\sigma(a_k)", font_size=30, color=INPUT_BLUE),
            MathTex(r"y_k=\frac{\exp(a_k)}{\sum_l\exp(a_l)}", font_size=30, color=ACCENT_ORANGE),
        ]
        subtitles = ["real value", "0 <= y <= 1", "sum y_k = 1"]
        for name, color, formula, subtitle in zip(names, colors, formulas, subtitles):
            box = RoundedRectangle(width=3.5, height=4.2, corner_radius=0.08, stroke_color=color, fill_color=color, fill_opacity=0.08)
            heading = Text(name, font_size=25, color=color).move_to(box.get_top() + DOWN * 0.45)
            formula.move_to(box.get_center() + UP * 0.75)
            sub = Text(subtitle, font_size=22, color=TEXT_GREY).next_to(formula, DOWN, buff=0.25)
            if name == "Regression":
                axis = NumberLine(x_range=[-2, 2, 1], length=2.4, color=GREY_B, include_numbers=False)
                marker = Dot(axis.n2p(1.25), color=OUTPUT_GREEN, radius=0.08)
                visual = VGroup(axis, marker).move_to(box.get_center() + DOWN * 1.15)
            elif name == "Multiple binary":
                mini_axes = Axes(x_range=[-4, 4, 2], y_range=[0, 1, 0.5], x_length=2.4, y_length=1.25, tips=False, axis_config={"color": GREY_B})
                xs = np.linspace(-4, 4, 200)
                curve = VMobject(color=INPUT_BLUE)
                curve.set_points_smoothly([mini_axes.c2p(float(x), float(sigmoid(x))) for x in xs])
                visual = VGroup(mini_axes, curve).move_to(box.get_center() + DOWN * 1.15)
            else:
                bars = VGroup()
                for value, bar_color in zip([0.18, 0.55, 0.27], [BLUE_C, ORANGE, GREEN_C]):
                    bar = Rectangle(width=0.42, height=1.7 * value, fill_color=bar_color, fill_opacity=0.75, stroke_width=0)
                    bars.add(bar)
                bars.arrange(RIGHT, buff=0.22, aligned_edge=DOWN)
                visual = bars.move_to(box.get_center() + DOWN * 1.1)
            cards.add(VGroup(box, heading, formula, sub, visual))
        cards.arrange(RIGHT, buff=0.35).shift(DOWN * 0.25)
        pointer = Text("output activation", font_size=25, color=TARGET_YELLOW).next_to(cards, DOWN, buff=0.2)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.2) for card in cards], lag_ratio=0.2), run_time=2.4)
        self.play(Write(pointer), run_time=1.0)
        self.play(LaggedStart(*[Indicate(card[2], color=TARGET_YELLOW) for card in cards], lag_ratio=0.25), run_time=2.1)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, cards, pointer)), run_time=0.8)

    def nonlinearity_is_essential(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("Why nonlinear hidden units matter")
        title = self.scene_title("線形だけなら、層を重ねても一つの線形変換", font_size=32)

        left_box = RoundedRectangle(width=5.1, height=4.4, corner_radius=0.08, stroke_color=GREY_B, fill_color=GREY_E, fill_opacity=0.08)
        left_box.shift(LEFT * 3.0 + DOWN * 0.15)
        right_box = RoundedRectangle(width=5.1, height=4.4, corner_radius=0.08, stroke_color=HIDDEN_PURPLE, fill_color=HIDDEN_PURPLE, fill_opacity=0.07)
        right_box.shift(RIGHT * 3.0 + DOWN * 0.15)

        left_title = Text("linear hidden units", font_size=24, color=TEXT_GREY).move_to(left_box.get_top() + DOWN * 0.4)
        right_title = Text("with tanh", font_size=24, color=HIDDEN_PURPLE).move_to(right_box.get_top() + DOWN * 0.4)
        chain = MathTex(r"\mathbf{x}\to W^{(1)}\mathbf{x}\to W^{(2)}W^{(1)}\mathbf{x}", font_size=30)
        collapse = MathTex(r"=\,W\mathbf{x}", font_size=38, color=TARGET_YELLOW).next_to(chain, DOWN, buff=0.35)
        chain_group = VGroup(chain, collapse).move_to(left_box.get_center() + UP * 0.3)
        left_axes = Axes(x_range=[-1, 1, 1], y_range=[-1, 1, 1], x_length=3.2, y_length=1.45, tips=False, axis_config={"color": GREY_B})
        line = self.curve_from_values(left_axes, 0.8 * self.grid, MODEL_RED, width=3.0)
        left_plot = VGroup(left_axes, line).move_to(left_box.get_center() + DOWN * 1.15)

        right_axes = Axes(x_range=[-1, 1, 1], y_range=[-1.1, 1.1, 1], x_length=3.2, y_length=1.45, tips=False, axis_config={"color": GREY_B})
        bend = self.curve_from_values(right_axes, 0.85 * np.tanh(4 * self.grid), HIDDEN_PURPLE, width=3.2)
        right_formula = MathTex(r"\mathbf{x}\to W^{(1)}\mathbf{x}\to h(\cdot)\to W^{(2)}\mathbf{z}", font_size=29)
        right_formula.move_to(right_box.get_center() + UP * 0.45)
        right_plot = VGroup(right_axes, bend).move_to(right_box.get_center() + DOWN * 1.15)
        note = Text("linear + nonlinear を交互に組む", font_size=26, color=ACCENT_ORANGE)
        note.to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title), run_time=1.1)
        self.play(FadeIn(left_box), Write(left_title), Write(chain), run_time=1.6)
        self.play(Write(collapse), Create(left_plot), run_time=1.7)
        self.play(FadeIn(right_box), Write(right_title), Write(right_formula), Create(right_plot), run_time=2.2)
        self.play(Indicate(bend, color=TARGET_YELLOW), Write(note), run_time=1.5)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, left_box, right_box, left_title, right_title, chain_group, left_plot, right_formula, right_plot, note)), run_time=0.8)

    def feed_forward_topology(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("General feed-forward topology")
        title = self.scene_title("閉路なしなら、順番に値を計算できる", font_size=34)

        positions = {
            "x1": LEFT * 4.2 + UP * 0.95,
            "x2": LEFT * 4.2 + DOWN * 0.95,
            "z1": LEFT * 1.35 + UP * 1.2,
            "z2": LEFT * 1.0 + DOWN * 0.55,
            "z3": RIGHT * 1.2 + UP * 0.25,
            "y1": RIGHT * 4.0 + UP * 0.75,
            "y2": RIGHT * 4.0 + DOWN * 0.75,
        }
        colors = {"x1": INPUT_BLUE, "x2": INPUT_BLUE, "z1": HIDDEN_PURPLE, "z2": HIDDEN_PURPLE, "z3": HIDDEN_PURPLE, "y1": OUTPUT_GREEN, "y2": OUTPUT_GREEN}
        nodes = {name: self.node(name, pos, colors[name]) for name, pos in positions.items()}
        edge_specs = [
            ("x1", "z1", GREY_C), ("x1", "z2", GREY_C), ("x2", "z2", GREY_C),
            ("z1", "z3", GREY_C), ("z2", "z3", GREY_C), ("z3", "y1", GREY_C),
            ("z2", "y2", GREY_C), ("x1", "y1", ACCENT_ORANGE),
        ]
        edges = VGroup(*[self.arrow_between(nodes[a], nodes[b], color, 2.0 if color == ACCENT_ORANGE else 1.7) for a, b, color in edge_specs])
        node_group = VGroup(*nodes.values())
        formula = MathTex(r"z_k=h\!\left(\sum_{j\to k} w_{kj}z_j\right)", font_size=35)
        formula.to_corner(DR).shift(UP * 0.2 + LEFT * 0.15)
        tags = VGroup(
            Text("skip", font_size=23, color=ACCENT_ORANGE).next_to(edges[-1], UP, buff=0.05),
            Text("sparse", font_size=23, color=TEXT_GREY).to_corner(DL).shift(UP * 0.55 + RIGHT * 0.2),
            Text("no directed cycle", font_size=25, color=TARGET_YELLOW).to_edge(DOWN).shift(UP * 0.22),
        )

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(node_group), run_time=1.2)
        self.play(Create(edges[:-1]), Write(formula), run_time=2.0)
        self.play(Create(edges[-1]), Write(tags[0]), Write(tags[1]), run_time=1.4)
        ordered = [nodes["x1"], nodes["x2"], nodes["z1"], nodes["z2"], nodes["z3"], nodes["y1"], nodes["y2"]]
        self.play(LaggedStart(*[Indicate(item, color=TARGET_YELLOW, scale_factor=1.08) for item in ordered], lag_ratio=0.16), Write(tags[2]), run_time=2.5)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, node_group, edges, formula, tags)), run_time=0.8)

    def make_function_panel(self, title: str, target_values: np.ndarray, approx_values: np.ndarray, color: ManimColor) -> VGroup:
        axes = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1.1, 1.1, 1],
            x_length=2.45,
            y_length=1.65,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 1.6},
        )
        panel_title = Text(title, font_size=21, color=color).next_to(axes, UP, buff=0.12)
        target = self.curve_from_values(axes, target_values, TARGET_YELLOW, width=2.2, opacity=0.65, dashed=True)
        approx = self.curve_from_values(axes, approx_values, MODEL_RED, width=2.8)
        sample_x = np.linspace(-0.9, 0.9, 11)
        sample_y = np.interp(sample_x, self.grid, target_values)
        dots = VGroup(*[Dot(axes.c2p(float(x), float(y)), radius=0.035, color=INPUT_BLUE) for x, y in zip(sample_x, sample_y)])
        return VGroup(axes, target, approx, dots, panel_title)

    def universal_approximation_view(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Approximation properties")
        title = self.scene_title("広い関数族を近似できる。ただし学習は別問題", font_size=32)

        x = self.grid
        panels = VGroup(
            self.make_function_panel("x^2", 1.6 * x**2 - 0.8, 1.55 * x**2 - 0.75 + 0.04 * np.sin(3 * math.pi * x), BLUE_B),
            self.make_function_panel("sin(x)", 0.8 * np.sin(math.pi * x), 0.78 * np.sin(math.pi * x) + 0.05 * np.sin(3 * math.pi * x), PURPLE_B),
            self.make_function_panel("|x|", 1.6 * np.abs(x) - 0.85, 1.55 * np.sqrt(x**2 + 0.015) - 0.82, GREEN_B),
            self.make_function_panel("step", np.where(x >= 0, 0.75, -0.75), 1.55 * sigmoid(16 * x) - 0.78, ACCENT_ORANGE),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.45, 0.55))
        panels.shift(DOWN * 0.2)
        legend = VGroup(
            Dot(radius=0.05, color=INPUT_BLUE),
            Text("data", font_size=19),
            DashedLine(LEFT * 0.25, RIGHT * 0.25, color=TARGET_YELLOW, stroke_width=3),
            Text("target", font_size=19),
            Line(LEFT * 0.25, RIGHT * 0.25, color=MODEL_RED, stroke_width=4),
            Text("network", font_size=19),
        ).arrange(RIGHT, buff=0.12)
        legend.to_corner(UR).shift(DOWN * 0.75 + LEFT * 0.1)
        note = Text("存在定理は安心材料。実際の重みを見つけるには学習が必要。", font_size=24, color=TARGET_YELLOW)
        note.to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(panel, shift=UP * 0.15) for panel in panels], lag_ratio=0.12), FadeIn(legend), run_time=3.0)
        self.play(Write(note), run_time=1.5)
        self.play(LaggedStart(*[Indicate(panel[2], color=TARGET_YELLOW) for panel in panels], lag_ratio=0.14), run_time=2.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, panels, legend, note)), run_time=0.8)

    def weight_space_symmetry(self) -> None:
        narration = self.start_narration("scene08")
        label = self.section_label("Weight-space symmetries")
        title = self.scene_title("違う重みでも、同じ関数を表せる", font_size=34)

        inputs = VGroup(self.node("x", LEFT * 4.0, INPUT_BLUE))
        hidden = VGroup(self.node("z1", LEFT * 0.8 + UP * 0.85, HIDDEN_PURPLE), self.node("z2", LEFT * 0.8 + DOWN * 0.85, HIDDEN_PURPLE))
        output = VGroup(self.node("y", RIGHT * 3.0, OUTPUT_GREEN))
        arrows = VGroup(
            self.arrow_between(inputs[0], hidden[0], GREY_C),
            self.arrow_between(inputs[0], hidden[1], GREY_C),
            self.arrow_between(hidden[0], output[0], GREY_C),
            self.arrow_between(hidden[1], output[0], GREY_C),
        )
        network = VGroup(arrows, inputs, hidden, output).shift(UP * 0.2)
        sign_box = VGroup(
            Text("sign flip", font_size=25, color=TARGET_YELLOW),
            MathTex(r"w_{\mathrm{in}}\to-w_{\mathrm{in}}", font_size=27),
            MathTex(r"w_{\mathrm{out}}\to-w_{\mathrm{out}}", font_size=27),
            Text("product contribution unchanged", font_size=20, color=TEXT_GREY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        sign_box.to_corner(DL).shift(UP * 0.35 + RIGHT * 0.15)
        perm_box = VGroup(
            Text("permutation", font_size=25, color=ACCENT_ORANGE),
            Text("z1 と z2 を入れ替える", font_size=22),
            Text("足し合わせる順番だけが変わる", font_size=20, color=TEXT_GREY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        perm_box.to_corner(DR).shift(UP * 0.45 + LEFT * 0.15)
        same_function = Text("different weights, same mapping x -> y", font_size=26, color=OUTPUT_GREEN)
        same_function.to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(inputs), FadeIn(hidden), FadeIn(output), Create(arrows), run_time=1.8)
        self.play(Write(sign_box), run_time=1.8)
        self.play(Indicate(VGroup(arrows[0], arrows[2], hidden[0]), color=TARGET_YELLOW), run_time=1.4)
        self.play(
            hidden[0].animate.move_to(hidden[1].get_center()),
            hidden[1].animate.move_to(hidden[0].get_center()),
            Write(perm_box),
            run_time=1.6,
        )
        self.play(Write(same_function), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(VGroup(label, title, network, sign_box, perm_box, same_function)), run_time=0.8)
