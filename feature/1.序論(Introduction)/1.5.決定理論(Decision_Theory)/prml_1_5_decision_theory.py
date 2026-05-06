from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_CLASS = BLUE_C
ORANGE_CLASS = ORANGE
GREEN_DECISION = GREEN_C
RED_LOSS = RED_C
YELLOW_RISK = YELLOW_C
PURPLE_REJECT = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def gaussian(x: np.ndarray | float, mean: float, sigma: float) -> np.ndarray | float:
    return np.exp(-0.5 * ((np.asarray(x) - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def class1_joint(x: np.ndarray | float) -> np.ndarray | float:
    return 0.52 * gaussian(x, 0.36, 0.15)


def class2_joint(x: np.ndarray | float) -> np.ndarray | float:
    return 0.48 * gaussian(x, 0.66, 0.14)


def posterior_c1(x: np.ndarray | float) -> np.ndarray | float:
    p1 = class1_joint(x)
    p2 = class2_joint(x)
    return p1 / (p1 + p2)


def posterior_c2(x: np.ndarray | float) -> np.ndarray | float:
    return 1 - posterior_c1(x)


class PRML15DecisionTheory(Scene):
    """PRML 1.5 decision theory overview.

    Render example:
        uv run manim -pql prml_1_5_decision_theory.py PRML15DecisionTheory
    """

    def construct(self) -> None:
        self.camera.background_color = "#101010"
        self.inference_to_decision()
        self.posterior_boundary()
        self.expected_loss()
        self.reject_option()
        self.inference_decision_paths()
        self.regression_loss()
        self.summary_bridge()

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

    def make_probability_axes(self, width: float = 8.2, height: float = 3.7) -> Axes:
        return Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 2.1, 0.5],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def make_posterior_axes(self, width: float = 8.2, height: float = 3.4) -> Axes:
        return Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1.1, 0.25],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def inference_to_decision(self) -> None:
        narration = self.start_narration("scene01")
        title = Text("PRML 1.5 Decision Theory", font_size=40, color=WHITE)
        subtitle = Text("不確かさを、具体的な行動へ変換する", font_size=29, color=TEXT_GREY)
        VGroup(title, subtitle).arrange(DOWN, buff=0.22).to_edge(UP, buff=0.9)

        input_box = self.flow_box("input x\n画像・測定値", BLUE_CLASS)
        inference_box = self.flow_box("inference\np(Ck|x)", GREEN_DECISION)
        loss_box = self.flow_box("loss Lkj\n判断の重さ", RED_LOSS)
        decision_box = self.flow_box("decision\n治療・保留・分類", ORANGE_CLASS)
        flow = VGroup(
            input_box,
            Arrow(RIGHT, RIGHT * 1.2, buff=0),
            inference_box,
            Arrow(RIGHT, RIGHT * 1.2, buff=0),
            loss_box,
            Arrow(RIGHT, RIGHT * 1.2, buff=0),
            decision_box,
        ).arrange(RIGHT, buff=0.24).scale(0.78).move_to(ORIGIN)

        equation = MathTex(
            r"\text{probability theory} + \text{loss} \Rightarrow \text{optimal action}",
            font_size=34,
        ).to_edge(DOWN, buff=0.72)

        self.play(Write(title), FadeIn(subtitle))
        self.play(FadeIn(flow[0]), GrowArrow(flow[1]), FadeIn(flow[2]))
        self.play(GrowArrow(flow[3]), FadeIn(flow[4]), GrowArrow(flow[5]), FadeIn(flow[6]))
        self.play(Write(equation), run_time=1.0)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(flow), FadeOut(equation))

    def flow_box(self, text: str, color: ManimColor) -> VGroup:
        box = RoundedRectangle(width=2.55, height=1.35, corner_radius=0.1, color=color)
        label = Text(text, font_size=23, color=WHITE, line_spacing=0.75).move_to(box)
        return VGroup(box, label)

    def posterior_boundary(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("PRML 1.5.1 / Fig. 1.24")
        title = self.scene_title("誤分類率を最小にするなら、最大の事後確率を選ぶ", font_size=31)
        axes = self.make_probability_axes().shift(DOWN * 0.12)
        x_label = MathTex("x", font_size=30).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex(r"p(x,C_k)", font_size=30).next_to(axes.y_axis.get_end(), UP)

        curve1 = axes.plot(lambda u: class1_joint(u), x_range=[0.02, 0.98], color=BLUE_CLASS)
        curve2 = axes.plot(lambda u: class2_joint(u), x_range=[0.02, 0.98], color=ORANGE_CLASS)
        curve1.set_stroke(width=4)
        curve2.set_stroke(width=4)
        c1_label = MathTex(r"p(x,C_1)", font_size=30, color=BLUE_CLASS).move_to(axes.c2p(0.25, 1.65))
        c2_label = MathTex(r"p(x,C_2)", font_size=30, color=ORANGE_CLASS).move_to(axes.c2p(0.77, 1.55))

        boundary = ValueTracker(0.36)

        def boundary_line() -> Line:
            x = boundary.get_value()
            line = DashedLine(axes.c2p(x, 0), axes.c2p(x, 1.95), dash_length=0.12, color=GREEN_DECISION)
            return line

        line = always_redraw(boundary_line)

        def boundary_text() -> VGroup:
            x = boundary.get_value()
            marker = MathTex(r"\hat{x}", font_size=29, color=GREEN_DECISION)
            marker.next_to(axes.c2p(x, 0), DOWN, buff=0.12)
            return VGroup(marker)

        marker = always_redraw(boundary_text)
        rule = MathTex(
            r"\text{choose } C_j \text{ where } p(C_j|x) \text{ is largest}",
            font_size=33,
        ).to_edge(DOWN, buff=0.55)

        comment = Text("境界を交点へ動かす", font_size=25, color=GREEN_DECISION).next_to(rule, UP, buff=0.18)
        crossing = 0.511

        self.play(FadeIn(label), Write(title), Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(Create(curve1), Create(curve2), FadeIn(c1_label), FadeIn(c2_label))
        self.play(FadeIn(line), FadeIn(marker), Write(rule))
        self.play(Write(comment), boundary.animate.set_value(crossing), run_time=2.2)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(
            FadeOut(label),
            FadeOut(title),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(curve1),
            FadeOut(curve2),
            FadeOut(c1_label),
            FadeOut(c2_label),
            FadeOut(line),
            FadeOut(marker),
            FadeOut(rule),
            FadeOut(comment),
        )

    def expected_loss(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("PRML 1.5.2 / Fig. 1.25 / 式 (1.80)-(1.81)")
        title = self.scene_title("正解率ではなく、期待損失を最小にする", font_size=33)

        matrix_title = Text("loss matrix  Lkj", font_size=26, color=TEXT_GREY)
        headers = VGroup(
            Text("判断: 治療", font_size=22, color=GREEN_DECISION),
            Text("判断: 健康", font_size=22, color=ORANGE_CLASS),
        ).arrange(RIGHT, buff=0.85)
        row_labels = VGroup(
            Text("真: 病気", font_size=22, color=BLUE_CLASS),
            Text("真: 健康", font_size=22, color=BLUE_CLASS),
        ).arrange(DOWN, buff=0.42, aligned_edge=RIGHT)
        cells = VGroup()
        values = [["0", "1000"], ["1", "0"]]
        for row in values:
            cells.add(VGroup(*[Text(value, font_size=28) for value in row]).arrange(RIGHT, buff=1.35))
        cells.arrange(DOWN, buff=0.28)
        table = VGroup(matrix_title, headers, row_labels, cells)
        headers.next_to(matrix_title, DOWN, buff=0.28)
        cells.next_to(headers, DOWN, buff=0.22)
        row_labels.next_to(cells, LEFT, buff=0.32)
        table = VGroup(matrix_title, headers, row_labels, cells).move_to(LEFT * 3.25 + DOWN * 0.05)

        p = 0.08
        posterior = VGroup(
            MathTex(r"p(C_1|x)=0.08", font_size=34, color=BLUE_CLASS),
            MathTex(r"p(C_2|x)=0.92", font_size=34, color=ORANGE_CLASS),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to(RIGHT * 3.0 + UP * 1.25)

        risk_treat = (1 - p) * 1
        risk_normal = p * 1000
        risks = VGroup(
            MathTex(r"R(a_1|x)=0\cdot0.08+1\cdot0.92=0.92", font_size=27, color=GREEN_DECISION),
            MathTex(r"R(a_2|x)=1000\cdot0.08+0\cdot0.92=80", font_size=27, color=RED_LOSS),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT).move_to(RIGHT * 2.28 + DOWN * 0.35)
        choice = Text("選ぶ: 期待損失が小さい 治療", font_size=28, color=YELLOW_RISK).to_edge(DOWN, buff=0.58)

        formula = MathTex(r"\sum_k L_{kj}p(C_k|x)", font_size=38, color=YELLOW_RISK)
        formula.next_to(risks, DOWN, buff=0.45)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(table, lag_ratio=0.18), run_time=1.2)
        self.play(FadeIn(posterior), run_time=0.9)
        self.play(Write(risks[0]), Write(risks[1]), run_time=1.5)
        self.play(Write(formula), Write(choice), run_time=1.0)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(table), FadeOut(posterior), FadeOut(risks), FadeOut(formula), FadeOut(choice))

    def reject_option(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("PRML 1.5.3 / Fig. 1.26")
        title = self.scene_title("あいまいな入力は、判断を保留できる", font_size=34)
        axes = self.make_posterior_axes().shift(DOWN * 0.1)
        c1 = axes.plot(lambda u: posterior_c1(u), x_range=[0.02, 0.98], color=BLUE_CLASS)
        c2 = axes.plot(lambda u: posterior_c2(u), x_range=[0.02, 0.98], color=ORANGE_CLASS)
        c1.set_stroke(width=4)
        c2.set_stroke(width=4)
        c1_label = MathTex(r"p(C_1|x)", font_size=28, color=BLUE_CLASS).move_to(axes.c2p(0.23, 0.88))
        c2_label = MathTex(r"p(C_2|x)", font_size=28, color=ORANGE_CLASS).move_to(axes.c2p(0.78, 0.88))
        theta = ValueTracker(0.72)

        threshold_line = always_redraw(
            lambda: DashedLine(
                axes.c2p(0.02, theta.get_value()),
                axes.c2p(0.98, theta.get_value()),
                color=PURPLE_REJECT,
                dash_length=0.12,
            )
        )
        theta_label = always_redraw(
            lambda: MathTex(r"\theta", font_size=30, color=PURPLE_REJECT).next_to(
                axes.c2p(0.02, theta.get_value()), LEFT, buff=0.12
            )
        )
        reject_band = Rectangle(width=1.45, height=3.15, color=PURPLE_REJECT, fill_opacity=0.18, stroke_opacity=0.0)
        reject_band.move_to(axes.c2p(0.511, 0.52))
        reject_text = Text("reject region\n専門家へ", font_size=25, color=PURPLE_REJECT, line_spacing=0.8)
        reject_text.move_to(reject_band.get_center() + DOWN * 0.35)
        rule = MathTex(r"\max_k p(C_k|x) \leq \theta \Rightarrow \text{reject}", font_size=34)
        rule.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title), Create(axes))
        self.play(Create(c1), Create(c2), FadeIn(c1_label), FadeIn(c2_label))
        self.play(FadeIn(threshold_line), FadeIn(theta_label), FadeIn(reject_band), Write(reject_text), Write(rule))
        self.play(theta.animate.set_value(0.62), reject_band.animate.set_width(0.9), run_time=1.3)
        self.play(theta.animate.set_value(0.82), reject_band.animate.set_width(2.0), run_time=1.3)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(
            FadeOut(label),
            FadeOut(title),
            FadeOut(axes),
            FadeOut(c1),
            FadeOut(c2),
            FadeOut(c1_label),
            FadeOut(c2_label),
            FadeOut(threshold_line),
            FadeOut(theta_label),
            FadeOut(reject_band),
            FadeOut(reject_text),
            FadeOut(rule),
        )

    def inference_decision_paths(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("PRML 1.5.4 / 式 (1.82)-(1.85)")
        title = self.scene_title("分類器の作り方: 確率をどこまで求めるか", font_size=33)

        panels = VGroup(
            self.path_panel(
                "generative",
                [r"p(x|C_k)", r"p(C_k)", r"\Downarrow", r"p(C_k|x)", r"\Downarrow", "decision"],
                BLUE_CLASS,
            ),
            self.path_panel(
                "discriminative",
                [r"p(C_k|x)", r"\Downarrow", "decision"],
                GREEN_DECISION,
            ),
            self.path_panel(
                "discriminant",
                [r"f(x)", r"\Downarrow", "class label"],
                ORANGE_CLASS,
            ),
        ).arrange(RIGHT, buff=0.32).scale(0.93).shift(DOWN * 0.15)

        note = Text("事後確率があると、損失変更・棄却・モデル結合に使いやすい", font_size=27, color=YELLOW_RISK)
        note.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(panels[0], shift=UP * 0.2), run_time=0.9)
        self.play(FadeIn(panels[1], shift=UP * 0.2), run_time=0.9)
        self.play(FadeIn(panels[2], shift=UP * 0.2), run_time=0.9)
        self.play(Write(note))
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(label), FadeOut(title), FadeOut(panels), FadeOut(note))

    def path_panel(self, heading: str, lines: list[str], color: ManimColor) -> VGroup:
        box = RoundedRectangle(width=3.7, height=4.2, corner_radius=0.1, color=color)
        head = Text(heading, font_size=27, color=color).move_to(box.get_top() + DOWN * 0.42)
        body = VGroup()
        for item in lines:
            if item in {"decision", "class label"}:
                body.add(Text(item, font_size=24, color=WHITE))
            else:
                body.add(MathTex(item, font_size=29, color=WHITE))
        body.arrange(DOWN, buff=0.13).move_to(box.get_center() + DOWN * 0.25)
        return VGroup(box, head, body)

    def regression_loss(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("PRML 1.5.5 / Fig. 1.28 / 式 (1.86)-(1.90)")
        title = self.scene_title("二乗損失では、条件付き平均が最適な予測", font_size=32)
        axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[-1.4, 1.6, 0.5],
            x_length=6.3,
            y_length=3.7,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 2.0 + DOWN * 0.15)
        mean_curve = axes.plot(lambda u: 0.95 * np.sin(2 * np.pi * u) + 0.15, x_range=[0, 1], color=GREEN_DECISION)
        mean_curve.set_stroke(width=4)
        mean_label = MathTex(r"E[t|x]", font_size=30, color=GREEN_DECISION).move_to(axes.c2p(0.83, -0.65))
        x0 = 0.32
        mean0 = 0.95 * np.sin(2 * np.pi * x0) + 0.15
        vertical = DashedLine(axes.c2p(x0, -1.25), axes.c2p(x0, 1.35), dash_length=0.1, color=TEXT_GREY)
        x0_label = MathTex(r"x_0", font_size=28).next_to(axes.c2p(x0, -1.4), DOWN, buff=0.08)

        side_axes = Axes(
            x_range=[0, 1.1, 0.5],
            y_range=[-1.4, 1.6, 0.5],
            x_length=2.15,
            y_length=3.7,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 3.25 + DOWN * 0.15)
        density = ParametricFunction(
            lambda u: side_axes.c2p(0.26 * gaussian(u, mean0, 0.34), u),
            t_range=[-1.15, 1.45],
            color=BLUE_CLASS,
        )
        mean_marker = Line(side_axes.c2p(0.0, mean0), side_axes.c2p(1.0, mean0), color=GREEN_DECISION, stroke_width=4)
        dist_label = MathTex(r"p(t|x_0)", font_size=30, color=BLUE_CLASS).next_to(side_axes, UP, buff=0.12)

        predictor = ValueTracker(mean0 - 0.75)

        def y_line() -> VGroup:
            y = predictor.get_value()
            line = Line(side_axes.c2p(0.0, y), side_axes.c2p(1.0, y), color=RED_LOSS, stroke_width=4)
            text = MathTex(r"y(x_0)", font_size=27, color=RED_LOSS).next_to(line, RIGHT, buff=0.12)
            return VGroup(line, text)

        pred_line = always_redraw(y_line)
        formula = MathTex(r"y(x)=E_t[t|x]", font_size=38, color=GREEN_DECISION)
        formula.to_edge(DOWN, buff=0.6)
        loss = MathTex(r"E[L]=\iint \{y(x)-t\}^2 p(x,t)\,dx\,dt", font_size=31)
        loss.next_to(formula, UP, buff=0.22)

        self.play(FadeIn(label), Write(title), Create(axes), Create(side_axes))
        self.play(Create(mean_curve), FadeIn(mean_label), FadeIn(vertical), FadeIn(x0_label))
        self.play(Create(density), FadeIn(dist_label), FadeIn(mean_marker), FadeIn(pred_line))
        self.play(predictor.animate.set_value(mean0 + 0.65), run_time=1.3)
        self.play(predictor.animate.set_value(mean0), run_time=1.2)
        self.play(Write(loss), Write(formula), run_time=1.1)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(
            FadeOut(label),
            FadeOut(title),
            FadeOut(axes),
            FadeOut(side_axes),
            FadeOut(mean_curve),
            FadeOut(mean_label),
            FadeOut(vertical),
            FadeOut(x0_label),
            FadeOut(density),
            FadeOut(dist_label),
            FadeOut(mean_marker),
            FadeOut(pred_line),
            FadeOut(loss),
            FadeOut(formula),
        )

    def summary_bridge(self) -> None:
        narration = self.start_narration("scene07")
        title = Text("1.5 Decision Theory の要点", font_size=38)
        title.to_edge(UP, buff=0.75)
        items = VGroup(
            self.summary_item("posterior", "p(Ck|x) が分類判断の材料になる", BLUE_CLASS),
            self.summary_item("loss", "Lkj が変わると最適な行動も変わる", RED_LOSS),
            self.summary_item("reject", "不確かな入力は保留できる", PURPLE_REJECT),
            self.summary_item("regression", "二乗損失では条件付き平均を選ぶ", GREEN_DECISION),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to(ORIGIN)
        next_section = Text("次: 1.6 Information Theory", font_size=30, color=YELLOW_RISK)
        next_section.to_edge(DOWN, buff=0.65)

        self.play(Write(title))
        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.25), run_time=0.45)
        self.play(Write(next_section), run_time=0.8)
        self.wait(0.8)
        self.finish_narration(narration)
        self.play(FadeOut(title), FadeOut(items), FadeOut(next_section))

    def summary_item(self, key: str, description: str, color: ManimColor) -> VGroup:
        tag_box = RoundedRectangle(width=2.2, height=0.58, corner_radius=0.08, color=color)
        tag = Text(key, font_size=22, color=color).move_to(tag_box)
        desc = Text(description, font_size=27, color=WHITE).next_to(tag_box, RIGHT, buff=0.28)
        return VGroup(tag_box, tag, desc)
