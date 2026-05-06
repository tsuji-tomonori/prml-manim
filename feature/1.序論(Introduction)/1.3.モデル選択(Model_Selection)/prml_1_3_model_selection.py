from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
VALIDATION_ORANGE = ORANGE
TEST_TEAL = TEAL_C
TRAIN_GREEN = GREEN_C
PENALTY_PURPLE = PURPLE_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def make_sine_data(n: int = 10, noise_std: float = 0.25, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    t = np.sin(2.0 * np.pi * x) + rng.normal(0.0, noise_std, size=n)
    return x, t


def design_matrix(x: np.ndarray | float, degree: int) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    return np.vander(x_array, N=degree + 1, increasing=True)


def fit_polynomial(x: np.ndarray, t: np.ndarray, degree: int, lam: float = 0.0) -> np.ndarray:
    phi = design_matrix(x, degree)
    if lam == 0:
        return np.linalg.lstsq(phi, t, rcond=None)[0]
    penalty = lam * np.eye(degree + 1)
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ t)


def eval_poly(w: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    return design_matrix(x, len(w) - 1) @ w


def rms_error(w: np.ndarray, x: np.ndarray, t: np.ndarray) -> float:
    residual = eval_poly(w, x) - t
    return float(np.sqrt(np.mean(residual**2)))


class PRML13ModelSelection(Scene):
    """PRML 1.3 model selection overview.

    Render example:
        uv run manim -pql prml_1_3_model_selection.py PRML13ModelSelection
    """

    def construct(self) -> None:
        self.camera.background_color = "#111111"
        self.x_train, self.t_train = make_sine_data(n=10, seed=7)
        self.x_test, self.t_test = make_sine_data(n=80, seed=17)

        self.opening_model_selection()
        self.training_error_trap()
        self.validation_and_test_split()
        self.cross_validation()
        self.leave_one_out()
        self.search_cost()
        self.information_criteria()
        self.bridge_to_dimensionality()

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
            self.wait(0.6)
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
        title.to_edge(UP).shift(DOWN * 0.28)
        return title

    def make_axes(self, width: float = 6.4, height: float = 3.8) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.6, 1.6, 0.8],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def data_dots(self, axes: Axes, color: ManimColor = BLUE_DATA, radius: float = 0.06) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(x), float(t)), color=color, radius=radius)
                for x, t in zip(self.x_train, self.t_train)
            ]
        )

    def model_curve(self, axes: Axes, degree: int, color: ManimColor = MODEL_RED, lam: float = 0.0) -> VMobject:
        w = fit_polynomial(self.x_train, self.t_train, degree, lam=lam)

        def model(u: float) -> float:
            return float(np.clip(eval_poly(w, u)[0], -1.7, 1.7))

        curve = axes.plot(model, x_range=[0, 1], color=color, use_smoothing=False)
        curve.set_stroke(width=4)
        return curve

    def error_axes(self) -> Axes:
        return Axes(
            x_range=[0, 9, 1],
            y_range=[0, 1.2, 0.3],
            x_length=7.0,
            y_length=3.8,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def error_curves(self, axes: Axes) -> tuple[VMobject, VMobject, list[float], list[float]]:
        degrees = list(range(10))
        train = []
        test = []
        for degree in degrees:
            w = fit_polynomial(self.x_train, self.t_train, degree)
            train.append(min(rms_error(w, self.x_train, self.t_train), 1.15))
            test.append(min(rms_error(w, self.x_test, self.t_test), 1.15))
        train_curve = axes.plot_line_graph(
            x_values=degrees,
            y_values=train,
            line_color=BLUE_C,
            add_vertex_dots=False,
            stroke_width=4,
        )
        test_curve = axes.plot_line_graph(
            x_values=degrees,
            y_values=test,
            line_color=VALIDATION_ORANGE,
            add_vertex_dots=False,
            stroke_width=4,
        )
        return train_curve, test_curve, train, test

    def block_row(
        self,
        count: int,
        width: float,
        height: float,
        colors: list[ManimColor],
        labels: list[str] | None = None,
    ) -> VGroup:
        blocks = VGroup()
        for i in range(count):
            rect = Rectangle(width=width, height=height, stroke_width=2, stroke_color=GREY_D, fill_opacity=0.85)
            rect.set_fill(colors[i])
            if i > 0:
                rect.next_to(blocks[-1], RIGHT, buff=0.03)
            label = Text(labels[i], font_size=18, color=WHITE) if labels else VMobject()
            if labels:
                label.move_to(rect)
                blocks.add(VGroup(rect, label))
            else:
                blocks.add(rect)
        blocks.move_to(ORIGIN)
        return blocks

    def opening_model_selection(self) -> None:
        self.clear()
        narration = self.start_narration("scene01")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("モデルの複雑さを選ぶ")

        axes = self.make_axes(width=5.6, height=3.5).shift(LEFT * 2.4 + DOWN * 0.15)
        dots = self.data_dots(axes)
        curve_simple = self.model_curve(axes, 1)
        curve_good = self.model_curve(axes, 3, color=TRAIN_GREEN)
        curve_complex = self.model_curve(axes, 9)
        axis_labels = VGroup(
            MathTex("x", font_size=28).next_to(axes.x_axis, RIGHT, buff=0.12),
            MathTex("t", font_size=28).next_to(axes.y_axis, UP, buff=0.12),
        )

        dial_center = RIGHT * 3.6 + DOWN * 0.15
        dial = Circle(radius=1.05, color=GREY_B, stroke_width=4).move_to(dial_center)
        pointer = Line(dial_center, dial_center + UP * 0.75, color=MODEL_RED, stroke_width=7)
        dial_title = Text("複雑さ M", font_size=28, color=MODEL_RED).next_to(dial, UP, buff=0.35)
        low = Text("硬い", font_size=22, color=TEXT_GREY).next_to(dial, LEFT, buff=0.35)
        high = Text("自由", font_size=22, color=TEXT_GREY).next_to(dial, RIGHT, buff=0.35)
        message = Text("選びたいのは初見データに強いモデル", font_size=27, color=WHITE)
        message.to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(axis_labels), FadeIn(dots), run_time=1.0)
        self.play(Create(curve_simple), FadeIn(dial), Write(dial_title), FadeIn(low), FadeIn(high), Create(pointer))
        for curve, angle in [(curve_good, -0.75), (curve_complex, -1.3), (curve_good.copy(), 0.75)]:
            new_pointer = pointer.copy().rotate(angle, about_point=dial_center)
            self.play(Transform(pointer, new_pointer), Transform(curve_simple, curve), run_time=1.0)
            self.wait(0.15)
        self.play(Write(message))
        self.finish_narration(narration)

    def training_error_trap(self) -> None:
        self.clear()
        narration = self.start_narration("scene02")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("訓練誤差だけでは選べない")

        axes = self.error_axes().shift(DOWN * 0.2)
        train_curve, test_curve, train, test = self.error_curves(axes)
        x_label = Text("次数 M", font_size=22, color=TEXT_GREY).next_to(axes.x_axis, DOWN, buff=0.35)
        y_label = Text("誤差", font_size=22, color=TEXT_GREY).next_to(axes.y_axis, LEFT, buff=0.25)
        y_label.rotate(PI / 2)
        legend = VGroup(
            Line(ORIGIN, RIGHT * 0.45, color=BLUE_C, stroke_width=5),
            Text("訓練", font_size=22),
            Line(ORIGIN, RIGHT * 0.45, color=VALIDATION_ORANGE, stroke_width=5),
            Text("初見", font_size=22),
        ).arrange(RIGHT, buff=0.18).to_corner(UR).shift(DOWN * 0.45)

        train_best = Dot(axes.c2p(9, train[9]), color=BLUE_C, radius=0.08)
        test_best_degree = int(np.argmin(test))
        test_best = Dot(axes.c2p(test_best_degree, test[test_best_degree]), color=VALIDATION_ORANGE, radius=0.08)
        train_callout = Text("訓練だけなら M=9", font_size=22, color=BLUE_C).next_to(train_best, DOWN, buff=0.25)
        test_callout = Text("初見では中くらい", font_size=22, color=VALIDATION_ORANGE).next_to(test_best, UP, buff=0.2)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), Write(x_label), Write(y_label), FadeIn(legend))
        self.play(Create(train_curve), run_time=1.4)
        self.play(FadeIn(train_best), Write(train_callout))
        self.play(Create(test_curve), run_time=1.4)
        self.play(FadeIn(test_best), Write(test_callout))
        warning = Text("training error は predictive performance の代理にならない", font_size=25, color=WHITE)
        warning.to_edge(DOWN).shift(UP * 0.25)
        self.play(Write(warning))
        self.finish_narration(narration)

    def validation_and_test_split(self) -> None:
        self.clear()
        narration = self.start_narration("scene03")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("データを役割で分ける")

        colors = [TRAIN_GREEN] * 12 + [VALIDATION_ORANGE] * 4 + [TEST_TEAL] * 4
        row = self.block_row(20, width=0.48, height=0.55, colors=colors).shift(UP * 1.75)
        captions = VGroup(
            Text("train", font_size=24, color=TRAIN_GREEN).next_to(row[5], DOWN, buff=0.28),
            Text("validation", font_size=24, color=VALIDATION_ORANGE).next_to(row[13], DOWN, buff=0.28),
            Text("test", font_size=24, color=TEST_TEAL).next_to(row[17], DOWN, buff=0.28),
        )
        candidates = VGroup(
            self.model_card("M=1", "underfit", BLUE_D),
            self.model_card("M=3", "selected", TRAIN_GREEN),
            self.model_card("M=9", "overfit", MODEL_RED),
        ).arrange(RIGHT, buff=0.55).shift(DOWN * 0.3)
        arrows = VGroup(
            Arrow(captions[0].get_bottom(), candidates[0].get_top(), buff=0.25, color=TRAIN_GREEN),
            Arrow(captions[1].get_bottom(), candidates[1].get_top(), buff=0.25, color=VALIDATION_ORANGE),
        )
        lock = VGroup(
            RoundedRectangle(width=1.25, height=0.62, corner_radius=0.08, color=TEST_TEAL, stroke_width=4),
            Text("final check", font_size=17, color=TEST_TEAL),
        )
        lock[1].move_to(lock[0])
        lock.next_to(row[17], UP, buff=0.22)
        note = Text("test は最後に一度だけ", font_size=26, color=WHITE).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(row, lag_ratio=0.04), FadeIn(captions))
        self.play(FadeIn(candidates, lag_ratio=0.2), GrowArrow(arrows[0]))
        self.play(GrowArrow(arrows[1]), candidates[1][0].animate.set_stroke(WHITE, width=5))
        self.play(FadeIn(lock), Write(note))
        self.finish_narration(narration)

    def model_card(self, main: str, sub: str, color: ManimColor) -> VGroup:
        rect = RoundedRectangle(width=2.2, height=1.35, corner_radius=0.08, stroke_color=color, stroke_width=3)
        rect.set_fill("#1B1B1B", opacity=0.85)
        main_text = MathTex(main, font_size=34, color=color)
        sub_text = Text(sub, font_size=20, color=TEXT_GREY).next_to(main_text, DOWN, buff=0.16)
        content = VGroup(main_text, sub_text).move_to(rect)
        return VGroup(rect, content)

    def cross_validation(self) -> None:
        self.clear()
        narration = self.start_narration("scene04")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("S-fold cross-validation")

        rows = VGroup()
        score_labels = VGroup()
        for run in range(4):
            colors = [TRAIN_GREEN] * 4
            colors[run] = MODEL_RED
            labels = ["eval" if i == run else "train" for i in range(4)]
            row = self.block_row(4, width=1.55, height=0.5, colors=colors, labels=labels)
            row.shift(UP * (1.25 - run * 0.62))
            run_label = Text(f"run {run + 1}", font_size=20, color=TEXT_GREY).next_to(row, LEFT, buff=0.35)
            score = Text(f"score {run + 1}", font_size=20, color=VALIDATION_ORANGE).next_to(row, RIGHT, buff=0.35)
            rows.add(VGroup(run_label, row))
            score_labels.add(score)

        formula = VGroup(
            Text("平均スコア", font_size=25, color=WHITE),
            MathTex(r"= {s_1+s_2+s_3+s_4 \over 4}", font_size=36, color=VALIDATION_ORANGE),
        ).arrange(RIGHT, buff=0.25).to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title))
        for i, row in enumerate(rows):
            self.play(FadeIn(row), FadeIn(score_labels[i]), run_time=0.55)
        brace = Brace(rows, LEFT, color=GREY_B)
        ratio = MathTex(r"{S-1 \over S}", r"\ \mathrm{for\ training}", font_size=32)
        ratio.next_to(brace, LEFT, buff=0.2)
        self.play(FadeIn(brace), Write(ratio))
        self.play(Write(formula))
        self.finish_narration(narration)

    def leave_one_out(self) -> None:
        self.clear()
        narration = self.start_narration("scene05")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("データが少ないとき")

        count = 18
        colors = [TRAIN_GREEN] * count
        colors[0] = MODEL_RED
        row = self.block_row(count, width=0.42, height=0.55, colors=colors).shift(UP * 0.8)
        marker = SurroundingRectangle(row[0], color=MODEL_RED, buff=0.05, stroke_width=4)
        equations = VGroup(
            MathTex("S=N", font_size=46, color=WHITE),
            Text("leave-one-out", font_size=32, color=MODEL_RED),
            Text("学習回数も N 回", font_size=28, color=TEXT_GREY),
        ).arrange(DOWN, buff=0.25).shift(DOWN * 1.15)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(row, lag_ratio=0.03), Create(marker))
        for index in [4, 9, 13, 17]:
            self.play(marker.animate.move_to(row[index]), run_time=0.35)
        self.play(Write(equations[0]), Write(equations[1]))
        self.play(Write(equations[2]))
        self.finish_narration(narration)

    def search_cost(self) -> None:
        self.clear()
        narration = self.start_narration("scene06")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("複雑さのつまみが増える")

        grid = VGroup()
        rows, cols = 4, 5
        for r in range(rows):
            for c in range(cols):
                cell = Square(side_length=0.55, color=GREY_B, stroke_width=2)
                cell.set_fill("#202020", opacity=0.8)
                cell.move_to(np.array([(c - 2) * 0.62, (1.5 - r) * 0.62, 0.0]))
                grid.add(cell)
        grid.shift(LEFT * 2.2 + DOWN * 0.1)
        x_label = Text("M", font_size=28, color=MODEL_RED).next_to(grid, DOWN, buff=0.25)
        y_label = MathTex(r"\lambda", font_size=36, color=PENALTY_PURPLE).next_to(grid, LEFT, buff=0.25)
        layers = VGroup(
            grid.copy().set_opacity(0.35).shift(RIGHT * 0.25 + UP * 0.2),
            grid.copy().set_opacity(0.2).shift(RIGHT * 0.5 + UP * 0.4),
        )
        count = VGroup(
            Text("候補", font_size=26, color=TEXT_GREY),
            MathTex("5\\times4=20", font_size=42, color=WHITE),
            Text("4-fold なら 80 回学習", font_size=25, color=VALIDATION_ORANGE),
        ).arrange(DOWN, buff=0.25).shift(RIGHT * 3.0)
        explosion = Text("つまみが増えると組み合わせが増える", font_size=25, color=WHITE)
        explosion.to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(grid), Write(x_label), Write(y_label))
        self.play(LaggedStart(*[cell.animate.set_fill(VALIDATION_ORANGE, opacity=0.6) for cell in grid], lag_ratio=0.025), run_time=1.1)
        self.play(FadeIn(layers), Write(count))
        self.play(Write(explosion))
        self.finish_narration(narration)

    def information_criteria(self) -> None:
        self.clear()
        narration = self.start_narration("scene07")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("訓練データだけで補正する考え方")

        formula = MathTex(r"\mathrm{AIC\ score}", "=", r"\ln p(D|w_{\mathrm{ML}})", "-", "M", font_size=42)
        formula.shift(UP * 1.55)
        formula[2].set_color(TRAIN_GREEN)
        formula[4].set_color(PENALTY_PURPLE)

        fit_box = VGroup(
            RoundedRectangle(width=3.0, height=1.35, corner_radius=0.08, color=TRAIN_GREEN, stroke_width=3),
            Text("当てはまり", font_size=25, color=TRAIN_GREEN),
            MathTex(r"\ln p(D|w_{\mathrm{ML}})", font_size=31, color=TRAIN_GREEN),
        )
        fit_box[1].move_to(fit_box[0].get_center() + UP * 0.25)
        fit_box[2].move_to(fit_box[0].get_center() + DOWN * 0.25)
        penalty_box = VGroup(
            RoundedRectangle(width=3.0, height=1.35, corner_radius=0.08, color=PENALTY_PURPLE, stroke_width=3),
            Text("複雑さ", font_size=25, color=PENALTY_PURPLE),
            MathTex("M", font_size=35, color=PENALTY_PURPLE),
        )
        penalty_box[1].move_to(penalty_box[0].get_center() + UP * 0.25)
        penalty_box[2].move_to(penalty_box[0].get_center() + DOWN * 0.25)
        boxes = VGroup(fit_box, penalty_box).arrange(RIGHT, buff=0.8).shift(DOWN * 0.2)

        bars = VGroup()
        labels = VGroup()
        values = [1.1, 1.55, 1.35]
        names = ["M=1", "M=3", "M=9"]
        colors = [BLUE_D, TRAIN_GREEN, MODEL_RED]
        for i, (value, name, color) in enumerate(zip(values, names, colors)):
            bar = Rectangle(width=0.55, height=value, color=color, fill_opacity=0.85, stroke_width=0)
            bar.align_to(ORIGIN, DOWN).shift(RIGHT * (i - 1) * 0.8)
            bars.add(bar)
            labels.add(MathTex(name, font_size=24, color=color).next_to(bar, DOWN, buff=0.18))
        chart = VGroup(bars, labels).shift(DOWN * 2.05 + RIGHT * 3.7)
        chart_title = Text("補正後スコア", font_size=22, color=TEXT_GREY).next_to(chart, UP, buff=0.2)

        self.play(FadeIn(label), Write(title))
        self.play(Write(formula))
        self.play(FadeIn(boxes, lag_ratio=0.2))
        self.play(FadeIn(chart_title), GrowFromEdge(bars, DOWN), FadeIn(labels))
        self.finish_narration(narration)

    def bridge_to_dimensionality(self) -> None:
        self.clear()
        narration = self.start_narration("scene08")
        label = self.section_label("1.3 Model Selection")
        title = self.scene_title("見えている点と初見への強さを分ける")

        items = VGroup(
            self.summary_item("train / validation / test", TRAIN_GREEN),
            self.summary_item("cross-validation", VALIDATION_ORANGE),
            self.summary_item("information criteria", PENALTY_PURPLE),
            self.summary_item("Bayesian approach", TEST_TEAL),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).shift(LEFT * 2.45 + DOWN * 0.05)

        grid_1d = self.dimension_grid(1).shift(RIGHT * 2.55 + UP * 0.8)
        grid_2d = self.dimension_grid(2).shift(RIGHT * 2.55 + UP * 0.8)
        grid_3d = self.dimension_grid(3).shift(RIGHT * 2.55 + UP * 0.8)
        next_label = Text("次: 1.4 次元の呪い", font_size=30, color=WHITE).to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(items, lag_ratio=0.15))
        self.play(FadeIn(grid_1d))
        self.play(Transform(grid_1d, grid_2d), run_time=0.8)
        self.play(Transform(grid_1d, grid_3d), run_time=0.8)
        self.play(Write(next_label))
        self.finish_narration(narration)

    def summary_item(self, text: str, color: ManimColor) -> VGroup:
        bullet = Dot(color=color, radius=0.08)
        label = Text(text, font_size=26, color=WHITE)
        return VGroup(bullet, label).arrange(RIGHT, buff=0.2)

    def dimension_grid(self, dimension: int) -> VGroup:
        if dimension == 1:
            cells = VGroup(*[Square(side_length=0.38, color=GREY_B, stroke_width=2) for _ in range(5)]).arrange(RIGHT, buff=0.03)
            label = MathTex("D=1", font_size=30, color=TEXT_GREY).next_to(cells, DOWN, buff=0.25)
            return VGroup(cells, label)
        if dimension == 2:
            cells = VGroup()
            for r in range(5):
                for c in range(5):
                    cell = Square(side_length=0.28, color=GREY_B, stroke_width=1.4)
                    cell.move_to(np.array([(c - 2) * 0.31, (2 - r) * 0.31, 0.0]))
                    cells.add(cell)
            label = MathTex("D=2", font_size=30, color=TEXT_GREY).next_to(cells, DOWN, buff=0.25)
            return VGroup(cells, label)
        cubes = VGroup()
        for layer in range(3):
            for r in range(4):
                for c in range(4):
                    cell = Square(side_length=0.24, color=GREY_B, stroke_width=1.2)
                    cell.set_fill("#1F1F1F", opacity=0.35)
                    cell.move_to(np.array([(c - 1.5) * 0.27 + layer * 0.18, (1.5 - r) * 0.27 + layer * 0.14, 0.0]))
                    cubes.add(cell)
        label = MathTex("D=3", font_size=30, color=TEXT_GREY).next_to(cubes, DOWN, buff=0.25)
        return VGroup(cubes, label)
