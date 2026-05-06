from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


FORWARD_BLUE = BLUE_C
BACKWARD_RED = RED_C
HIDDEN_GREEN = GREEN_C
ACCENT_ORANGE = ORANGE
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


class PRML53ErrorBackpropagation(Scene):
    """PRML 5.3 Error Backpropagation overview.

    Render example:
        uv run manim -pql prml_5_3_error_backpropagation.py PRML53ErrorBackpropagation
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"

        self.opening_backprop_scope()
        self.local_chain_rule()
        self.forward_values()
        self.output_delta()
        self.hidden_delta()
        self.two_layer_example()
        self.efficiency_and_checks()

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

    def scene_title(self, text: str, font_size: int = 33) -> Text:
        title = Text(text, font_size=font_size)
        title.to_edge(UP).shift(DOWN * 0.35)
        return title

    def node(self, label: str, color: ManimColor = WHITE, radius: float = 0.28) -> VGroup:
        circle = Circle(radius=radius, stroke_color=color, stroke_width=3, fill_color="#171717", fill_opacity=1)
        text = Text(label, font_size=22, color=color)
        return VGroup(circle, text)

    def make_network(
        self,
        include_bias: bool = True,
        output_count: int = 2,
        scale: float = 1.0,
    ) -> tuple[VGroup, dict[str, VGroup], VGroup, VGroup]:
        input_labels = ["x0", "x1", "x2"] if include_bias else ["x1", "x2"]
        input_y = [-1.4, 0.0, 1.4] if include_bias else [-0.75, 0.75]
        hidden_labels = ["z1", "z2"]
        output_labels = ["y1", "y2"] if output_count == 2 else ["y"]

        positions: dict[str, np.ndarray] = {}
        for label, y in zip(input_labels, input_y):
            positions[label] = np.array([-4.6, y, 0.0])
        for label, y in zip(hidden_labels, [0.85, -0.85]):
            positions[label] = np.array([-1.45, y, 0.0])
        for label, y in zip(output_labels, [0.75, -0.75] if output_count == 2 else [0.0]):
            positions[label] = np.array([1.7, y, 0.0])

        nodes: dict[str, VGroup] = {}
        for label in input_labels:
            nodes[label] = self.node(label, FORWARD_BLUE).move_to(positions[label])
        for label in hidden_labels:
            nodes[label] = self.node(label, HIDDEN_GREEN).move_to(positions[label])
        for label in output_labels:
            nodes[label] = self.node(label, BACKWARD_RED).move_to(positions[label])

        forward_edges = VGroup()
        backward_edges = VGroup()
        for i_label in input_labels:
            for h_label in hidden_labels:
                start = nodes[i_label][0].get_right()
                end = nodes[h_label][0].get_left()
                forward_edges.add(Arrow(start, end, buff=0.08, color=GREY_B, stroke_width=2.3, max_tip_length_to_length_ratio=0.08))
                backward_edges.add(Arrow(end, start, buff=0.08, color=BACKWARD_RED, stroke_width=3.2, max_tip_length_to_length_ratio=0.08))
        for h_label in hidden_labels:
            for o_label in output_labels:
                start = nodes[h_label][0].get_right()
                end = nodes[o_label][0].get_left()
                forward_edges.add(Arrow(start, end, buff=0.08, color=GREY_B, stroke_width=2.3, max_tip_length_to_length_ratio=0.08))
                backward_edges.add(Arrow(end, start, buff=0.08, color=BACKWARD_RED, stroke_width=3.2, max_tip_length_to_length_ratio=0.08))

        VGroup(forward_edges, backward_edges, *nodes.values()).scale(scale)
        graph = VGroup(forward_edges, *nodes.values())
        return graph, nodes, forward_edges, backward_edges

    def flash_edges(self, edges: VGroup, color: ManimColor, width: float = 6.0, run_time: float = 1.3) -> Animation:
        flashes = [ShowPassingFlash(edge.copy().set_color(color).set_stroke(width=width), time_width=0.45) for edge in edges]
        return AnimationGroup(*flashes, lag_ratio=0.03, run_time=run_time)

    def opening_backprop_scope(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 5.3 Error Backpropagation")
        title = self.scene_title("逆伝播は、勾配を効率よく計算する手順", font_size=32)
        graph, _, forward_edges, backward_edges = self.make_network(scale=1.0)
        graph.shift(DOWN * 0.1)
        backward_edges.shift(DOWN * 0.1)

        forward_tag = VGroup(
            Dot(color=FORWARD_BLUE, radius=0.07),
            Text("forward: values", font_size=24, color=FORWARD_BLUE),
        ).arrange(RIGHT, buff=0.12)
        backward_tag = VGroup(
            Dot(color=BACKWARD_RED, radius=0.07),
            Text("backward: deltas", font_size=24, color=BACKWARD_RED),
        ).arrange(RIGHT, buff=0.12)
        tags = VGroup(forward_tag, backward_tag).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        tags.to_corner(UR).shift(DOWN * 0.95 + LEFT * 0.15)

        stage1 = VGroup(
            Text("1. 勾配を評価", font_size=27, color=WHITE),
            MathTex(r"\nabla E(w)", font_size=36, color=ACCENT_ORANGE),
        ).arrange(DOWN, buff=0.12)
        stage2 = VGroup(
            Text("2. 最適化法が利用", font_size=27, color=TEXT_GREY),
            MathTex(r"w \leftarrow w + \Delta w", font_size=34, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.12)
        stages = VGroup(stage1, stage2).arrange(RIGHT, buff=1.0).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(Create(graph), run_time=1.8)
        self.play(self.flash_edges(forward_edges, FORWARD_BLUE), FadeIn(forward_tag), run_time=1.5)
        self.play(FadeIn(backward_edges, lag_ratio=0.08), FadeIn(backward_tag), run_time=1.7)
        self.play(FadeIn(stage1, shift=UP * 0.2), FadeIn(stage2, shift=UP * 0.2), run_time=1.4)
        self.play(Indicate(stage1, color=ACCENT_ORANGE), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def local_chain_rule(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Local chain rule")
        title = self.scene_title("1本の重みを見ると、勾配は局所情報の積", font_size=32)

        left = self.node("z_i", FORWARD_BLUE, radius=0.32).move_to(LEFT * 3.7)
        weight_box = RoundedRectangle(width=1.4, height=0.72, corner_radius=0.08, stroke_color=ACCENT_ORANGE, fill_color="#21180c", fill_opacity=1)
        weight_text = MathTex(r"w_{ji}", font_size=38, color=ACCENT_ORANGE).move_to(weight_box)
        weight = VGroup(weight_box, weight_text).move_to(LEFT * 1.25)
        right = self.node("a_j", HIDDEN_GREEN, radius=0.34).move_to(RIGHT * 1.25)
        after = self.node("E_n", BACKWARD_RED, radius=0.34).move_to(RIGHT * 3.7)
        arrows = VGroup(
            Arrow(left[0].get_right(), weight_box.get_left(), buff=0.1, color=FORWARD_BLUE, stroke_width=4),
            Arrow(weight_box.get_right(), right[0].get_left(), buff=0.1, color=FORWARD_BLUE, stroke_width=4),
            Arrow(right[0].get_right(), after[0].get_left(), buff=0.1, color=BACKWARD_RED, stroke_width=4),
        )
        chain = MathTex(
            r"\frac{\partial E_n}{\partial w_{ji}}",
            r"=",
            r"\frac{\partial E_n}{\partial a_j}",
            r"\frac{\partial a_j}{\partial w_{ji}}",
            font_size=38,
        ).to_edge(DOWN).shift(UP * 1.15)
        definitions = VGroup(
            MathTex(r"\delta_j \equiv \frac{\partial E_n}{\partial a_j}", font_size=37, color=BACKWARD_RED),
            MathTex(r"\frac{\partial a_j}{\partial w_{ji}} = z_i", font_size=37, color=FORWARD_BLUE),
            MathTex(r"\frac{\partial E_n}{\partial w_{ji}} = \delta_j z_i", font_size=43, color=ACCENT_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        definitions.to_corner(UR).shift(DOWN * 0.85 + LEFT * 0.2)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(left), FadeIn(weight), FadeIn(right), FadeIn(after), Create(arrows), run_time=1.7)
        self.play(Write(chain), run_time=1.7)
        self.play(Write(definitions[0]), Indicate(right, color=BACKWARD_RED), run_time=1.4)
        self.play(Write(definitions[1]), Indicate(left, color=FORWARD_BLUE), run_time=1.4)
        self.play(Write(definitions[2]), Circumscribe(weight, color=ACCENT_ORANGE), run_time=1.8)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def forward_values(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Forward propagation")
        title = self.scene_title("前向き計算で a と z を保存する", font_size=33)
        graph, nodes, forward_edges, _ = self.make_network(output_count=1, scale=1.0)
        graph.shift(DOWN * 0.15)

        formulas = VGroup(
            MathTex(r"a_j=\sum_i w_{ji}z_i", font_size=38, color=WHITE),
            MathTex(r"z_j=h(a_j)", font_size=38, color=HIDDEN_GREEN),
            MathTex(r"y_k=g(a_k)", font_size=38, color=BACKWARD_RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        formulas.to_corner(UR).shift(DOWN * 0.65 + LEFT * 0.2)

        saved = VGroup(
            Text("backward pass uses", font_size=22, color=TEXT_GREY),
            VGroup(
                MathTex(r"a_j", font_size=34, color=HIDDEN_GREEN),
                Text("and", font_size=20, color=TEXT_GREY),
                MathTex(r"z_i", font_size=34, color=FORWARD_BLUE),
            ).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.1)
        saved.to_edge(DOWN).shift(UP * 0.4)

        a_labels = VGroup(
            MathTex(r"a_1,z_1", font_size=27, color=HIDDEN_GREEN).next_to(nodes["z1"], UP, buff=0.16),
            MathTex(r"a_2,z_2", font_size=27, color=HIDDEN_GREEN).next_to(nodes["z2"], DOWN, buff=0.16),
            MathTex(r"a_k,y_k", font_size=27, color=BACKWARD_RED).next_to(nodes["y"], RIGHT, buff=0.2),
        ).shift(DOWN * 0.15)

        self.play(FadeIn(label), Write(title), Create(graph), run_time=1.8)
        self.play(self.flash_edges(forward_edges, FORWARD_BLUE), run_time=1.5)
        self.play(Write(formulas[0]), run_time=1.0)
        self.play(Write(formulas[1]), FadeIn(a_labels[:2]), run_time=1.2)
        self.play(Write(formulas[2]), FadeIn(a_labels[2]), run_time=1.2)
        self.play(FadeIn(saved), Indicate(a_labels, color=ACCENT_ORANGE), run_time=1.6)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def output_delta(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Output delta")
        title = self.scene_title("後ろ向き計算は出力デルタから始まる", font_size=32)

        output = self.node("y_k", BACKWARD_RED, radius=0.45).move_to(LEFT * 2.0)
        target = VGroup(
            RoundedRectangle(width=1.3, height=0.75, corner_radius=0.08, stroke_color=ACCENT_ORANGE, fill_color="#21180c", fill_opacity=1),
            MathTex(r"t_k", font_size=40, color=ACCENT_ORANGE),
        ).move_to(RIGHT * 1.15)
        minus = MathTex(r"-", font_size=52, color=WHITE).move_to((output.get_center() + target.get_center()) / 2)
        delta = VGroup(
            MathTex(r"\delta_k", font_size=48, color=BACKWARD_RED),
            MathTex(r"=y_k-t_k", font_size=48, color=BACKWARD_RED),
        ).arrange(RIGHT, buff=0.2)
        delta.to_edge(DOWN).shift(UP * 1.3)
        error_signal = Text("output-side error signal", font_size=25, color=TEXT_GREY).next_to(delta, DOWN, buff=0.18)
        network, _, _, backward_edges = self.make_network(output_count=1, scale=0.72)
        network.to_corner(UL).shift(DOWN * 0.85 + RIGHT * 0.4)
        backward_edges.to_corner(UL).shift(DOWN * 0.85 + RIGHT * 0.4)

        self.play(FadeIn(label), Write(title), run_time=1.1)
        self.play(FadeIn(output), FadeIn(target), Write(minus), run_time=1.4)
        self.play(Write(delta), FadeIn(error_signal), run_time=1.4)
        self.play(Indicate(output, color=BACKWARD_RED), Indicate(target, color=ACCENT_ORANGE), run_time=1.3)
        self.play(FadeIn(network), FadeIn(backward_edges), run_time=1.5)
        self.play(Indicate(backward_edges, color=BACKWARD_RED), run_time=1.2)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def hidden_delta(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("Hidden delta")
        title = self.scene_title("隠れユニットは、先のデルタを逆向きに集める", font_size=31)

        hidden = self.node("j", HIDDEN_GREEN, radius=0.38).move_to(LEFT * 2.9)
        outs = VGroup(
            self.node("k1", BACKWARD_RED, radius=0.32).move_to(RIGHT * 1.8 + UP * 1.25),
            self.node("k2", BACKWARD_RED, radius=0.32).move_to(RIGHT * 1.8),
            self.node("k3", BACKWARD_RED, radius=0.32).move_to(RIGHT * 1.8 + DOWN * 1.25),
        )
        forward = VGroup()
        backward = VGroup()
        weight_labels = VGroup()
        for index, out in enumerate(outs, start=1):
            forward.add(Arrow(hidden[0].get_right(), out[0].get_left(), buff=0.08, color=GREY_B, stroke_width=2.4))
            backward.add(Arrow(out[0].get_left(), hidden[0].get_right(), buff=0.08, color=BACKWARD_RED, stroke_width=4))
            weight_labels.add(MathTex(fr"w_{{k_{index}j}}", font_size=25, color=ACCENT_ORANGE).move_to((hidden.get_center() + out.get_center()) / 2 + UP * 0.12))
        delta_labels = VGroup(
            MathTex(r"\delta_{k_1}", font_size=28, color=BACKWARD_RED).next_to(outs[0], RIGHT, buff=0.14),
            MathTex(r"\delta_{k_2}", font_size=28, color=BACKWARD_RED).next_to(outs[1], RIGHT, buff=0.14),
            MathTex(r"\delta_{k_3}", font_size=28, color=BACKWARD_RED).next_to(outs[2], RIGHT, buff=0.14),
        )
        formula = MathTex(
            r"\delta_j",
            r"=",
            r"h'(a_j)",
            r"\sum_k w_{kj}\delta_k",
            font_size=42,
        ).to_edge(DOWN).shift(UP * 0.95)
        formula[0].set_color(HIDDEN_GREEN)
        formula[2].set_color(HIDDEN_GREEN)
        formula[3].set_color(BACKWARD_RED)
        note = Text("逆向きに集めて、局所微分を掛ける", font_size=24, color=TEXT_GREY).next_to(formula, DOWN, buff=0.16)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(hidden), FadeIn(outs), Create(forward), FadeIn(weight_labels), run_time=1.8)
        self.play(FadeIn(delta_labels), FadeIn(backward, lag_ratio=0.12), run_time=1.6)
        self.play(Write(formula[0:2]), run_time=0.8)
        self.play(Write(formula[3]), Indicate(backward, color=BACKWARD_RED), run_time=1.5)
        self.play(Write(formula[2]), Circumscribe(hidden, color=HIDDEN_GREEN), run_time=1.3)
        self.play(FadeIn(note), run_time=0.8)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def two_layer_example(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Two-layer example")
        title = self.scene_title("tanh 隠れ層では、勾配が外積として並ぶ", font_size=31)
        graph, _, forward_edges, backward_edges = self.make_network(output_count=2, scale=0.82)
        graph.to_edge(LEFT).shift(RIGHT * 0.35 + DOWN * 0.15)
        backward_edges.to_edge(LEFT).shift(RIGHT * 0.35 + DOWN * 0.15)

        equations = VGroup(
            MathTex(r"a_j=\sum_{i=0}^{D}w^{(1)}_{ji}x_i,\quad z_j=\tanh(a_j)", font_size=30),
            MathTex(r"y_k=\sum_{j=0}^{M}w^{(2)}_{kj}z_j", font_size=31),
            MathTex(r"\delta_k=y_k-t_k", font_size=34, color=BACKWARD_RED),
            MathTex(r"\delta_j=(1-z_j^2)\sum_k w^{(2)}_{kj}\delta_k", font_size=32, color=HIDDEN_GREEN),
            MathTex(r"\frac{\partial E_n}{\partial w^{(1)}_{ji}}=\delta_jx_i,\quad \frac{\partial E_n}{\partial w^{(2)}_{kj}}=\delta_kz_j", font_size=30, color=ACCENT_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        equations.to_corner(UR).shift(DOWN * 0.72 + LEFT * 0.1)

        outer = VGroup(
            Text("gradient\nmatrix", font_size=22, color=TEXT_GREY, line_spacing=0.8),
            Matrix(
                [[r"\delta_1 z_0", r"\delta_1 z_1", r"\delta_1 z_2"],
                 [r"\delta_2 z_0", r"\delta_2 z_1", r"\delta_2 z_2"]],
                element_alignment_corner=ORIGIN,
                h_buff=1.55,
                v_buff=0.78,
            ).scale(0.55).set_color(ACCENT_ORANGE),
        ).arrange(DOWN, buff=0.08)
        outer.to_edge(DOWN).shift(UP * 0.2 + RIGHT * 2.05)

        self.play(FadeIn(label), Write(title), Create(graph), run_time=1.8)
        self.play(self.flash_edges(forward_edges, FORWARD_BLUE), Write(equations[0]), run_time=1.5)
        self.play(Write(equations[1]), run_time=1.0)
        self.play(FadeIn(backward_edges, lag_ratio=0.08), Write(equations[2]), run_time=1.5)
        self.play(Write(equations[3]), Indicate(backward_edges, color=BACKWARD_RED), run_time=1.6)
        self.play(Write(equations[4]), FadeIn(outer), run_time=1.8)
        self.play(Indicate(outer, color=ACCENT_ORANGE), run_time=1.1)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

    def efficiency_and_checks(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Efficiency")
        title = self.scene_title("有限差分は確認用、学習中の勾配は逆伝播で求める", font_size=30)

        left = VGroup(
            Text("finite differences", font_size=30, color=TEXT_GREY),
            MathTex(r"W\ \mathrm{weights}", font_size=34, color=WHITE),
            MathTex(r"\times\ O(W)\ \mathrm{forward}", font_size=34, color=WHITE),
            MathTex(r"=O(W^2)", font_size=48, color=BACKWARD_RED),
        ).arrange(DOWN, buff=0.18)
        left_box = SurroundingRectangle(left, color=BACKWARD_RED, buff=0.28)
        left_group = VGroup(left_box, left).move_to(LEFT * 3.0 + DOWN * 0.05)

        right = VGroup(
            Text("backpropagation", font_size=30, color=FORWARD_BLUE),
            MathTex(r"1\ \mathrm{forward}", font_size=34, color=FORWARD_BLUE),
            MathTex(r"+\ 1\ \mathrm{backward}", font_size=34, color=BACKWARD_RED),
            MathTex(r"=O(W)", font_size=48, color=HIDDEN_GREEN),
        ).arrange(DOWN, buff=0.18)
        right_box = SurroundingRectangle(right, color=HIDDEN_GREEN, buff=0.28)
        right_group = VGroup(right_box, right).move_to(RIGHT * 3.05 + DOWN * 0.05)

        check = VGroup(
            Text("implementation check", font_size=24, color=TEXT_GREY),
            MathTex(
                r"\frac{E(w+\epsilon)-E(w-\epsilon)}{2\epsilon}",
                font_size=36,
                color=ACCENT_ORANGE,
            ),
        ).arrange(DOWN, buff=0.12)
        check.to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(left_group, shift=UP * 0.2), run_time=1.5)
        self.play(FadeIn(right_group, shift=UP * 0.2), run_time=1.5)
        self.play(Indicate(right[-1], color=HIDDEN_GREEN), run_time=1.2)
        self.play(FadeIn(check), run_time=1.4)
        self.play(Indicate(check[1], color=ACCENT_ORANGE), run_time=1.1)
        self.finish_narration(narration)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
