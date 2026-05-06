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


def quadratic_value(x: float, y: float, h11: float = 4.0, h22: float = 0.8, h12: float = 0.0) -> float:
    return 0.5 * (h11 * x * x + 2.0 * h12 * x * y + h22 * y * y)


class PRML54HessianMatrix(Scene):
    """PRML 5.4 explanatory video: The Hessian Matrix.

    Render example:
        uv run manim --disable_caching --flush_cache -ql prml_5_4_hessian_matrix.py PRML54HessianMatrix
    """

    def construct(self) -> None:
        self.camera.background_color = BG

        self.opening()
        self.local_quadratic_model()
        self.eigenvalue_geometry()
        self.why_hessian_matters()
        self.approximation_methods()
        self.inverse_and_finite_difference()
        self.exact_and_hessian_vector()

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

    def make_weight_axes(self, width: float = 5.8, height: float = 4.0) -> Axes:
        return Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-1.8, 1.8, 1],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def contour_ellipse(self, axes: Axes, level: float, h11: float, h22: float, angle: float, color: ManimColor) -> Ellipse:
        width = abs(axes.c2p(math.sqrt(2.0 * level / h11), 0)[0] - axes.c2p(-math.sqrt(2.0 * level / h11), 0)[0])
        height = abs(axes.c2p(0, math.sqrt(2.0 * level / h22))[1] - axes.c2p(0, -math.sqrt(2.0 * level / h22))[1])
        return Ellipse(width=width, height=height, color=color, stroke_width=2.4).rotate(angle).move_to(axes.c2p(0, 0))

    def matrix_grid(self, size: int = 5, mode: str = "full", color: ManimColor = ACCENT_BLUE) -> VGroup:
        cells = VGroup()
        for r in range(size):
            for c in range(size):
                active = mode == "full" or (mode == "diagonal" and r == c) or (mode == "outer" and (r in (1, 3) or c in (1, 3)))
                cell = Square(side_length=0.42, stroke_color=GREY_D, stroke_width=1.0)
                cell.set_fill(color if active else GREY_E, opacity=0.68 if active else 0.12)
                cells.add(cell)
        cells.arrange_in_grid(rows=size, cols=size, buff=0.035)
        return cells

    def method_card(self, title: str, subtitle: str, formula: Mobject, color: ManimColor) -> VGroup:
        box = RoundedRectangle(width=3.55, height=2.1, corner_radius=0.1, stroke_color=color, stroke_width=2.2)
        heading = Text(title, font_size=24, color=color)
        sub = Text(subtitle, font_size=18, color=TEXT_GREY)
        body = VGroup(heading, sub, formula).arrange(DOWN, buff=0.16)
        body.move_to(box)
        return VGroup(box, body)

    def opening(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 5.4 The Hessian Matrix")
        title = Text("ヘッセ行列", font_size=46, color=WHITE).to_edge(UP, buff=0.75)
        subtitle = Text("勾配の次に、誤差面の曲がり方を見る", font_size=28, color=TEXT_GREY)
        subtitle.next_to(title, DOWN, buff=0.22)

        grad = MathTex(r"\nabla E(w)", font_size=52, color=ACCENT_BLUE)
        arrow = Arrow(LEFT, RIGHT, color=TEXT_GREY, stroke_width=5)
        hessian = MathTex(r"H=\nabla\nabla E(w)", font_size=52, color=ACCENT_ORANGE)
        row = VGroup(grad, arrow, hessian).arrange(RIGHT, buff=0.42).shift(UP * 0.4)

        matrix = MathTex(
            r"H_{ij}=\frac{\partial^2 E}{\partial w_i\,\partial w_j}",
            font_size=43,
            color=WHITE,
        ).next_to(row, DOWN, buff=0.55)

        use_cases = VGroup(
            Text("最適化", font_size=25, color=ACCENT_GREEN),
            Text("刈り込み", font_size=25, color=ACCENT_RED),
            Text("ラプラス近似", font_size=25, color=ACCENT_PURPLE),
        ).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.85)

        self.play(FadeIn(label), Write(title), FadeIn(subtitle), run_time=1.2)
        self.play(Write(grad), GrowArrow(arrow), Write(hessian), run_time=1.6)
        self.play(Write(matrix), run_time=1.4)
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.15) for item in use_cases], lag_ratio=0.2), run_time=1.5)
        self.finish_narration(narration)
        self.clear_scene()

    def local_quadratic_model(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Local quadratic approximation")
        title = self.scene_title("小さな範囲では、誤差面を二次式で近似する")

        axes = self.make_weight_axes().shift(LEFT * 3.0 + DOWN * 0.25)
        contours = VGroup(
            self.contour_ellipse(axes, 0.35, 3.6, 0.7, 0.0, ACCENT_BLUE),
            self.contour_ellipse(axes, 0.75, 3.6, 0.7, 0.0, ACCENT_BLUE),
            self.contour_ellipse(axes, 1.25, 3.6, 0.7, 0.0, ACCENT_BLUE),
        )
        center = Dot(axes.c2p(0, 0), radius=0.08, color=ACCENT_GREEN)
        point = Dot(axes.c2p(1.55, 0.95), radius=0.08, color=ACCENT_ORANGE)
        delta = Arrow(axes.c2p(0, 0), axes.c2p(1.55, 0.95), color=ACCENT_ORANGE, buff=0.1)
        delta_label = MathTex(r"w-\widehat{w}", font_size=30, color=ACCENT_ORANGE).next_to(delta, UP, buff=0.08)

        formula = MathTex(
            r"E(w)\simeq E(\widehat{w})+(w-\widehat{w})^{\mathrm T}b"
            r"+\frac{1}{2}(w-\widehat{w})^{\mathrm T}H(w-\widehat{w})",
            font_size=34,
        ).shift(RIGHT * 2.15 + UP * 1.05)
        grad_formula = MathTex(r"\nabla E \simeq b+H(w-\widehat{w})", font_size=39, color=WHITE)
        min_formula = MathTex(r"\nabla E(\widehat{w})=0 \Rightarrow b=0", font_size=36, color=ACCENT_GREEN)
        simplified = MathTex(
            r"E(w)=E(\widehat{w})+\frac{1}{2}(w-\widehat{w})^{\mathrm T}H(w-\widehat{w})",
            font_size=35,
            color=ACCENT_YELLOW,
        )
        equations = VGroup(grad_formula, min_formula, simplified).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        equations.next_to(formula, DOWN, buff=0.55).shift(LEFT * 0.1)

        self.play(FadeIn(label), Write(title), Create(axes), run_time=1.5)
        self.play(Create(contours), FadeIn(center), run_time=1.5)
        self.play(FadeIn(point), GrowArrow(delta), Write(delta_label), run_time=1.3)
        self.play(Write(formula), run_time=2.1)
        self.play(LaggedStart(Write(grad_formula), Write(min_formula), Write(simplified), lag_ratio=0.35), run_time=2.6)
        self.play(Indicate(contours[0], color=ACCENT_YELLOW), run_time=1.0)
        self.finish_narration(narration)
        self.clear_scene()

    def eigenvalue_geometry(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Eigenvalues and curvature")
        title = self.scene_title("固有ベクトルは向き、固有値は曲がりの強さ")

        axes = self.make_weight_axes(width=7.0, height=4.7).shift(DOWN * 0.25)
        contours = VGroup(
            self.contour_ellipse(axes, 0.4, 4.2, 0.55, 0.45, ACCENT_BLUE),
            self.contour_ellipse(axes, 0.9, 4.2, 0.55, 0.45, ACCENT_BLUE),
            self.contour_ellipse(axes, 1.55, 4.2, 0.55, 0.45, ACCENT_BLUE),
        )
        u1 = Arrow(axes.c2p(0, 0), axes.c2p(1.3, 0.62), color=ACCENT_RED, buff=0, stroke_width=5)
        u2 = Arrow(axes.c2p(0, 0), axes.c2p(-0.38, 1.45), color=ACCENT_GREEN, buff=0, stroke_width=5)
        u1_label = MathTex(r"u_1,\ \lambda_1\ \mathrm{large}", font_size=29, color=ACCENT_RED).next_to(u1, RIGHT, buff=0.1)
        u2_label = MathTex(r"u_2,\ \lambda_2\ \mathrm{small}", font_size=29, color=ACCENT_GREEN).next_to(u2, UP, buff=0.1)
        formula = MathTex(
            r"H u_i=\lambda_i u_i,\qquad "
            r"E=E(\widehat{w})+\frac{1}{2}\sum_i \lambda_i \alpha_i^2",
            font_size=37,
        ).to_edge(DOWN, buff=0.45)
        length_note = Text("等高線の幅は 1/sqrt(lambda) に比例", font_size=24, color=TEXT_GREY)
        length_note.next_to(formula, UP, buff=0.18)

        tests = VGroup(
            Text("全て正: 局所最小", font_size=24, color=ACCENT_GREEN),
            Text("負を含む: 鞍点または最大", font_size=24, color=ACCENT_RED),
            Text("ゼロに近い: 平坦で不安定", font_size=24, color=ACCENT_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_corner(UR).shift(DOWN * 0.75 + LEFT * 0.15)

        self.play(FadeIn(label), Write(title), Create(axes), run_time=1.5)
        self.play(Create(contours), run_time=1.4)
        self.play(GrowArrow(u1), GrowArrow(u2), Write(u1_label), Write(u2_label), run_time=1.8)
        self.play(Write(formula), FadeIn(length_note), run_time=1.7)
        self.play(LaggedStart(*[FadeIn(item, shift=LEFT * 0.2) for item in tests], lag_ratio=0.2), run_time=1.8)
        self.play(Indicate(tests[0], color=ACCENT_GREEN), run_time=0.9)
        self.finish_narration(narration)
        self.clear_scene()

    def why_hessian_matters(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("Why use H")
        title = self.scene_title("ヘッセ行列は、訓練後の判断にも使われる")

        center = self.matrix_grid(size=5, mode="full", color=ACCENT_ORANGE).scale(1.05).shift(UP * 0.2)
        h_label = MathTex(r"H\in\mathbb{R}^{W\times W}", font_size=35, color=ACCENT_ORANGE).next_to(center, UP, buff=0.25)

        items = [
            ("二次の最適化", "曲率でステップを調整", ACCENT_GREEN, LEFT * 4.25 + UP * 1.2),
            ("再学習の近似", "データが少し変わった時", ACCENT_BLUE, RIGHT * 4.1 + UP * 1.2),
            ("重みの刈り込み", "影響の小さい重みを探す", ACCENT_RED, LEFT * 4.15 + DOWN * 1.55),
            ("ベイズ推論", "ラプラス近似と証拠", ACCENT_PURPLE, RIGHT * 4.0 + DOWN * 1.55),
        ]
        cards = VGroup()
        arrows = VGroup()
        for heading, desc, color, pos in items:
            box = RoundedRectangle(width=3.25, height=1.2, corner_radius=0.1, stroke_color=color, stroke_width=2.2)
            text = VGroup(Text(heading, font_size=24, color=color), Text(desc, font_size=18, color=TEXT_GREY)).arrange(DOWN, buff=0.08)
            card = VGroup(box, text).move_to(pos)
            cards.add(card)
            arrows.add(Arrow(center.get_center(), card.get_center(), color=color, buff=1.25, stroke_width=3.2))

        complexity = Text("W 個のパラメータなら H は W x W、扱いは O(W^2) 規模", font_size=25, color=ACCENT_YELLOW)
        complexity.to_edge(DOWN, buff=0.65)

        self.play(FadeIn(label), Write(title), FadeIn(center), Write(h_label), run_time=1.7)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.14), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(card, scale=0.95) for card in cards], lag_ratio=0.15), run_time=2.1)
        self.play(Write(complexity), run_time=1.3)
        self.play(Indicate(center, color=ACCENT_YELLOW), run_time=1.0)
        self.finish_narration(narration)
        self.clear_scene()

    def approximation_methods(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("Approximations")
        title = self.scene_title("近似は、何を捨てるかを意識して使う")

        diag_grid = self.matrix_grid(mode="diagonal", color=ACCENT_GREEN).scale(0.75)
        diag_formula = MathTex(r"H\simeq \mathrm{diag}(H)", font_size=30, color=WHITE)
        diag_card = self.method_card("Diagonal", "逆行列は簡単 / 非対角を捨てる", VGroup(diag_grid, diag_formula).arrange(DOWN, buff=0.18), ACCENT_GREEN)

        outer_grid = self.matrix_grid(mode="outer", color=ACCENT_ORANGE).scale(0.75)
        outer_formula = MathTex(r"H\simeq\sum_n b_n b_n^{\mathrm T}", font_size=30, color=WHITE)
        outer_card = self.method_card("Outer product", "訓練後の残差が小さい時", VGroup(outer_grid, outer_formula).arrange(DOWN, buff=0.18), ACCENT_ORANGE)

        exact_grid = self.matrix_grid(mode="full", color=ACCENT_BLUE).scale(0.75)
        exact_formula = MathTex(r"H=\nabla\nabla E", font_size=32, color=WHITE)
        exact_card = self.method_card("Exact", "二階の backprop で O(W^2)", VGroup(exact_grid, exact_formula).arrange(DOWN, buff=0.18), ACCENT_BLUE)

        cards = VGroup(diag_card, outer_card, exact_card).arrange(RIGHT, buff=0.32).shift(UP * 0.25)
        residual = MathTex(
            r"\nabla\nabla E=\sum_n\nabla y_n\nabla y_n^{\mathrm T}"
            r"+\sum_n(y_n-t_n)\nabla\nabla y_n",
            font_size=34,
        ).to_edge(DOWN, buff=0.8)
        brace = Brace(residual[0][21:], UP, color=ACCENT_RED)
        brace_text = Text("ここを小さいとみなす", font_size=21, color=ACCENT_RED).next_to(brace, UP, buff=0.08)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.25) for card in cards], lag_ratio=0.18), run_time=2.0)
        self.play(Write(residual), run_time=1.8)
        self.play(GrowFromCenter(brace), FadeIn(brace_text), run_time=1.0)
        self.play(Indicate(diag_card[0], color=ACCENT_YELLOW), Indicate(outer_card[0], color=ACCENT_YELLOW), run_time=1.1)
        self.finish_narration(narration)
        self.clear_scene()

    def inverse_and_finite_difference(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Inverse Hessian and finite differences")
        title = self.scene_title("逆ヘッセ行列は逐次更新でき、差分は検査に使う")

        left_title = Text("Outer product を 1 点ずつ足す", font_size=26, color=ACCENT_ORANGE)
        matrices = VGroup(
            MathTex(r"H_L^{-1}", font_size=39, color=ACCENT_BLUE),
            MathTex(r"\rightarrow", font_size=36, color=TEXT_GREY),
            MathTex(r"H_{L+1}^{-1}", font_size=39, color=ACCENT_GREEN),
        ).arrange(RIGHT, buff=0.25)
        update = MathTex(
            r"H_{L+1}^{-1}=H_L^{-1}-"
            r"\frac{H_L^{-1}bb^{\mathrm T}H_L^{-1}}{1+b^{\mathrm T}H_L^{-1}b}",
            font_size=32,
        )
        left = VGroup(left_title, matrices, update).arrange(DOWN, buff=0.28).move_to(LEFT * 3.05 + UP * 0.45)

        right_title = Text("有限差分は実装チェック向き", font_size=26, color=ACCENT_PURPLE)
        stencil = VGroup()
        positions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        signs = ["+", "-", "-", "+"]
        for (x, y), sign in zip(positions, signs):
            dot = Dot([x * 0.55, y * 0.38, 0], radius=0.08, color=ACCENT_YELLOW)
            sign_text = Text(sign, font_size=24, color=WHITE).next_to(dot, UP, buff=0.04)
            stencil.add(VGroup(dot, sign_text))
        square = DashedVMobject(Rectangle(width=1.1, height=0.76, color=ACCENT_PURPLE), num_dashes=16)
        stencil.add(square)
        diff = MathTex(
            r"\frac{\partial^2E}{\partial w_i\partial w_j}"
            r"\approx \frac{E_{++}-E_{+-}-E_{-+}+E_{--}}{4\epsilon^2}",
            font_size=31,
        )
        cost = Text("全要素を直接差分: O(W^3) / 勾配差分: O(W^2)", font_size=21, color=TEXT_GREY)
        right = VGroup(right_title, stencil, diff, cost).arrange(DOWN, buff=0.28).move_to(RIGHT * 3.1 + UP * 0.25)

        divider = Line(UP * 2.3, DOWN * 2.4, color=GREY_D)

        self.play(FadeIn(label), Write(title), Create(divider), run_time=1.3)
        self.play(LaggedStart(Write(left_title), FadeIn(matrices), Write(update), lag_ratio=0.25), run_time=2.2)
        self.play(LaggedStart(Write(right_title), FadeIn(stencil), Write(diff), Write(cost), lag_ratio=0.2), run_time=2.3)
        self.play(Indicate(matrices[-1], color=ACCENT_GREEN), Indicate(stencil, color=ACCENT_YELLOW), run_time=1.1)
        self.finish_narration(narration)
        self.clear_scene()

    def exact_and_hessian_vector(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Exact Hessian and Hessian-vector product")
        title = self.scene_title("実用上は、H 全体より H とベクトルの積が欲しいことも多い")

        network = VGroup()
        xs = [-3.4, -1.2, 1.2, 3.3]
        layer_sizes = [3, 4, 3, 2]
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED]
        layers: list[VGroup] = []
        for x, count, color in zip(xs, layer_sizes, colors):
            nodes = VGroup(*[Circle(radius=0.16, color=color, fill_color=color, fill_opacity=0.35).move_to([x, (i - (count - 1) / 2) * 0.58, 0]) for i in range(count)])
            layers.append(nodes)
            network.add(nodes)
        edges = VGroup()
        for left, right in zip(layers[:-1], layers[1:]):
            for a in left:
                for b in right:
                    edges.add(Line(a.get_center(), b.get_center(), color=GREY_D, stroke_width=1.0))
        network.add_to_back(edges)
        network.shift(UP * 0.9)

        forward = Arrow(LEFT * 3.7 + DOWN * 1.03, RIGHT * 3.7 + DOWN * 1.03, color=ACCENT_BLUE, stroke_width=5)
        backward = Arrow(RIGHT * 3.7 + DOWN * 1.45, LEFT * 3.7 + DOWN * 1.45, color=ACCENT_ORANGE, stroke_width=5)
        forward_label = Text("forward: activations と R{activations}", font_size=20, color=ACCENT_BLUE).next_to(forward, UP, buff=0.05)
        backward_label = Text("backward: errors と R{errors}", font_size=20, color=ACCENT_ORANGE).next_to(backward, DOWN, buff=0.05)

        equations = VGroup(
            MathTex(r"v^{\mathrm T}H=v^{\mathrm T}\nabla(\nabla E)", font_size=30, color=WHITE),
            MathTex(r"R\{w\}=v", font_size=30, color=ACCENT_YELLOW),
            MathTex(r"H\ \mathrm{full}: O(W^2),\qquad Hv: O(W)", font_size=29, color=ACCENT_GREEN),
        ).arrange(DOWN, buff=0.11).to_edge(DOWN, buff=0.22)

        summary = VGroup(
            Text("曲率を読む", font_size=24, color=ACCENT_GREEN),
            Text("近似の前提を確認する", font_size=24, color=ACCENT_YELLOW),
            Text("必要なら Hv だけ計算する", font_size=24, color=ACCENT_BLUE),
        ).arrange(RIGHT, buff=0.55).to_corner(UR).shift(DOWN * 0.75)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(FadeIn(network), run_time=1.8)
        self.play(GrowArrow(forward), FadeIn(forward_label), run_time=1.2)
        self.play(GrowArrow(backward), FadeIn(backward_label), run_time=1.2)
        self.play(LaggedStart(*[Write(eq) for eq in equations], lag_ratio=0.25), run_time=2.1)
        self.play(LaggedStart(*[FadeIn(item, shift=LEFT * 0.15) for item in summary], lag_ratio=0.18), run_time=1.3)
        self.finish_narration(narration)
        self.clear_scene()
