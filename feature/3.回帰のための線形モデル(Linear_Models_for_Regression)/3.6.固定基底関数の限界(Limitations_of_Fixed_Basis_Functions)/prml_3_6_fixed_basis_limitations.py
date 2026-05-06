from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
ADAPTIVE_GREEN = GREEN_C
GRID_ORANGE = ORANGE
KERNEL_TEAL = TEAL_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def target_curve(x: np.ndarray | float) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    return 0.78 * np.sin(2 * np.pi * x_array) + 0.24 * np.sin(6 * np.pi * x_array)


def rbf_features(x: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x[:, None] - centers[None, :]) / width) ** 2)


def fit_rbf(x: np.ndarray, t: np.ndarray, centers: np.ndarray, width: float, lam: float = 1e-4) -> np.ndarray:
    phi = np.column_stack([np.ones_like(x), rbf_features(x, centers, width)])
    penalty = lam * np.eye(phi.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ t)


def eval_rbf(x: np.ndarray, centers: np.ndarray, width: float, weights: np.ndarray) -> np.ndarray:
    phi = np.column_stack([np.ones_like(x), rbf_features(x, centers, width)])
    return phi @ weights


class PRML36FixedBasisLimitations(Scene):
    """PRML 3.6 limitations of fixed basis functions.

    Render example:
        uv run manim -pql prml_3_6_fixed_basis_limitations.py PRML36FixedBasisLimitations
    """

    def construct(self) -> None:
        self.camera.background_color = "#111111"
        self.x_train, self.t_train = self.make_training_data()

        self.opening_fixed_basis()
        self.local_basis_width_tradeoff()
        self.high_dimensional_growth()
        self.fixed_grid_waste()
        self.boundary_misalignment()
        self.data_adaptive_basis()
        self.bridge_to_next_methods()

    def make_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(36)
        x = np.array([0.03, 0.10, 0.18, 0.28, 0.38, 0.50, 0.62, 0.74, 0.86, 0.96])
        t = target_curve(x) + rng.normal(0.0, 0.11, size=x.shape)
        return x, t

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
            self.wait(0.7)
            return
        elapsed = float(getattr(self, "time", 0.0)) - start_time
        remaining = duration - elapsed + pad
        if remaining > 0:
            self.wait(remaining)

    def section_label(self) -> Text:
        label = Text("3.6 Limitations of Fixed Basis Functions", font_size=18, color=TEXT_GREY)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 32) -> Text:
        title = Text(text, font_size=font_size)
        title.to_edge(UP).shift(DOWN * 0.28)
        return title

    def make_axes(self, width: float = 6.4, height: float = 3.7) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.4, 1.4, 0.7],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def data_dots(self, axes: Axes, radius: float = 0.06) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(x), float(t)), color=BLUE_DATA, radius=radius)
                for x, t in zip(self.x_train, self.t_train)
            ]
        )

    def rbf_curve(self, axes: Axes, centers: np.ndarray, width: float, color: ManimColor = MODEL_RED) -> VMobject:
        weights = fit_rbf(self.x_train, self.t_train, centers, width)

        def model(u: float) -> float:
            value = eval_rbf(np.array([u]), centers, width, weights)[0]
            return float(np.clip(value, -1.5, 1.5))

        curve = axes.plot(model, x_range=[0, 1], color=color, use_smoothing=False)
        curve.set_stroke(width=4)
        return curve

    def basis_bumps(self, axes: Axes, centers: np.ndarray, width: float, scale: float = 0.75) -> VGroup:
        bumps = VGroup()
        for center in centers:
            bump = axes.plot(
                lambda u, c=float(center): -1.2 + scale * np.exp(-0.5 * ((u - c) / width) ** 2),
                x_range=[0, 1],
                color=GRID_ORANGE,
                use_smoothing=False,
            )
            bump.set_stroke(width=2, opacity=0.75)
            bumps.add(bump)
        return bumps

    def opening_fixed_basis(self) -> None:
        self.clear()
        narration = self.start_narration("scene01")
        label = self.section_label()
        title = self.scene_title("先に決めた特徴量で学習する")

        axes = self.make_axes(width=6.1, height=3.45).shift(LEFT * 2.15 + DOWN * 0.15)
        dots = self.data_dots(axes)
        centers = np.linspace(0.08, 0.92, 7)
        bumps = self.basis_bumps(axes, centers, 0.10, scale=0.55)
        curve = self.rbf_curve(axes, centers, 0.10)
        formula = MathTex(r"y(x,\mathbf{w})=\sum_j w_j\phi_j(x)", font_size=40)
        formula.shift(RIGHT * 3.1 + UP * 1.15)
        cards = VGroup(
            self.small_card("fixed", "基底の位置と幅", GRID_ORANGE),
            self.small_card("learned", "重み w", ADAPTIVE_GREEN),
        ).arrange(DOWN, buff=0.25).shift(RIGHT * 3.15 + DOWN * 0.65)
        note = Text("便利さの代わりに、表現力は基底の選び方へ依存する", font_size=25)
        note.to_edge(DOWN).shift(UP * 0.24)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(dots), run_time=0.9)
        self.play(LaggedStart(*[Create(bump) for bump in bumps], lag_ratio=0.08), run_time=1.2)
        self.play(Create(curve), Write(formula))
        self.play(FadeIn(cards, lag_ratio=0.18), Write(note))
        self.finish_narration(narration)

    def local_basis_width_tradeoff(self) -> None:
        self.clear()
        narration = self.start_narration("scene02")
        label = self.section_label()
        title = self.scene_title("幅の選び方だけでもトレードオフ")

        axes = self.make_axes(width=6.4, height=3.55).shift(DOWN * 0.25)
        dots = self.data_dots(axes)
        centers = np.linspace(0.05, 0.95, 11)
        narrow = self.rbf_curve(axes, centers, 0.035, color=MODEL_RED)
        good = self.rbf_curve(axes, centers, 0.105, color=ADAPTIVE_GREEN)
        wide = self.rbf_curve(axes, centers, 0.28, color=BLUE_D)
        captions = VGroup(
            Text("狭すぎる: 点ごとに反応", font_size=23, color=MODEL_RED),
            Text("ほどよい: 近くを共有", font_size=23, color=ADAPTIVE_GREEN),
            Text("広すぎる: 細部をならす", font_size=23, color=BLUE_D),
        ).arrange(RIGHT, buff=0.55).to_edge(DOWN).shift(UP * 0.24)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(dots))
        self.play(Create(narrow), Write(captions[0]), run_time=1.0)
        self.play(Transform(narrow, good), Transform(captions[0], captions[1]), run_time=1.0)
        self.play(Transform(narrow, wide), Transform(captions[0], captions[2]), run_time=1.0)
        self.play(Transform(narrow, good), Transform(captions[0], captions[1]), run_time=0.8)
        self.finish_narration(narration)

    def high_dimensional_growth(self) -> None:
        self.clear()
        narration = self.start_narration("scene03")
        label = self.section_label()
        title = self.scene_title("次元が増えると基底数が増える")

        grids = VGroup(
            self.grid_1d(7, "D=1", "7"),
            self.grid_2d(5, "D=2", "25"),
            self.grid_3d(4, "D=3", "64"),
        ).arrange(RIGHT, buff=0.95).shift(UP * 0.35)
        formula = MathTex(r"K^D", font_size=54, color=GRID_ORANGE).shift(DOWN * 1.75)
        caption = Text("各次元に K 個ずつ置くと、総数は指数的に増える", font_size=25)
        caption.next_to(formula, DOWN, buff=0.25)

        self.play(FadeIn(label), Write(title))
        for grid in grids:
            self.play(FadeIn(grid, lag_ratio=0.03), run_time=0.7)
        self.play(Write(formula), Write(caption))
        self.finish_narration(narration)

    def fixed_grid_waste(self) -> None:
        self.clear()
        narration = self.start_narration("scene04")
        label = self.section_label()
        title = self.scene_title("固定グリッドはデータのない場所にも置かれる")

        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=7.0,
            y_length=4.6,
            background_line_style={"stroke_color": GREY_D, "stroke_width": 1, "stroke_opacity": 0.5},
            axis_config={"stroke_color": GREY_B, "stroke_width": 2},
        ).shift(LEFT * 1.0 + DOWN * 0.05)
        centers = VGroup()
        for x in np.linspace(-2.5, 2.5, 6):
            for y in np.linspace(-1.5, 1.5, 4):
                centers.add(Dot(plane.c2p(float(x), float(y)), color=GRID_ORANGE, radius=0.045))
        data = VGroup()
        for value in np.linspace(-2.5, 2.5, 18):
            y = 0.55 * np.sin(1.25 * value)
            data.add(Dot(plane.c2p(float(value), float(y)), color=BLUE_DATA, radius=0.07))
        tube = VMobject(color=BLUE_DATA, stroke_width=18, stroke_opacity=0.16)
        points = [plane.c2p(float(x), float(0.55 * np.sin(1.25 * x))) for x in np.linspace(-2.6, 2.6, 60)]
        tube.set_points_smoothly(points)
        explanation = VGroup(
            Text("使われる場所", font_size=25, color=BLUE_DATA),
            Text("ほとんど空の場所", font_size=25, color=GRID_ORANGE),
        ).arrange(DOWN, buff=0.25).shift(RIGHT * 4.1 + UP * 0.45)
        waste = Text("高次元では、この空白がさらに大きくなる", font_size=24, color=WHITE)
        waste.to_edge(DOWN).shift(UP * 0.28)

        self.play(FadeIn(label), Write(title))
        self.play(Create(plane), FadeIn(centers, lag_ratio=0.02), run_time=1.1)
        self.play(Create(tube), FadeIn(data, lag_ratio=0.03), run_time=1.1)
        self.play(Write(explanation), Write(waste))
        self.finish_narration(narration)

    def boundary_misalignment(self) -> None:
        self.clear()
        narration = self.start_narration("scene05")
        label = self.section_label()
        title = self.scene_title("形が合わないと、重みだけでは苦しい")

        axes = self.make_axes(width=6.1, height=3.55).shift(LEFT * 2.1 + DOWN * 0.15)
        dots = self.data_dots(axes)
        centers_bad = np.linspace(0.0, 1.0, 4)
        centers_good = np.array([0.05, 0.17, 0.31, 0.50, 0.69, 0.83, 0.95])
        bad_bumps = self.basis_bumps(axes, centers_bad, 0.18, scale=0.48)
        bad_curve = self.rbf_curve(axes, centers_bad, 0.18, color=MODEL_RED)
        good_curve = self.rbf_curve(axes, centers_good, 0.10, color=ADAPTIVE_GREEN)
        right_text = VGroup(
            Text("基底は固定", font_size=30, color=GRID_ORANGE),
            Text("調整できるのは重みだけ", font_size=24, color=TEXT_GREY),
            Text("必要な場所に部品がないと遠回り", font_size=24, color=WHITE),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT).shift(RIGHT * 3.15)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), FadeIn(dots), FadeIn(bad_bumps, lag_ratio=0.08))
        self.play(Create(bad_curve), Write(right_text[0]), Write(right_text[1]))
        self.play(Transform(bad_curve, good_curve), Write(right_text[2]), run_time=1.2)
        self.finish_narration(narration)

    def data_adaptive_basis(self) -> None:
        self.clear()
        narration = self.start_narration("scene06")
        label = self.section_label()
        title = self.scene_title("基底をデータに合わせて動かす発想")

        fixed = self.center_panel("fixed basis", GRID_ORANGE).shift(LEFT * 3.0)
        adaptive = self.center_panel("adaptive basis", ADAPTIVE_GREEN).shift(RIGHT * 3.0)
        arrow = Arrow(fixed.get_right(), adaptive.get_left(), color=WHITE, buff=0.35)
        points = self.cluster_points().shift(RIGHT * 3.0 + DOWN * 0.2)
        moving_centers = VGroup(
            Dot(RIGHT * 3.0 + LEFT * 1.0 + UP * 0.6, color=GRID_ORANGE, radius=0.08),
            Dot(RIGHT * 3.0 + RIGHT * 0.2 + DOWN * 0.5, color=GRID_ORANGE, radius=0.08),
            Dot(RIGHT * 3.0 + RIGHT * 1.0 + UP * 0.35, color=GRID_ORANGE, radius=0.08),
        )
        target_centers = VGroup(
            Dot(points[1].get_center(), color=ADAPTIVE_GREEN, radius=0.09),
            Dot(points[7].get_center(), color=ADAPTIVE_GREEN, radius=0.09),
            Dot(points[13].get_center(), color=ADAPTIVE_GREEN, radius=0.09),
        )
        labels = VGroup(
            Text("線形モデルの良さは残す", font_size=25, color=WHITE),
            Text("特徴量の作り方を学習する方向へ", font_size=25, color=ADAPTIVE_GREEN),
        ).arrange(DOWN, buff=0.25).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(fixed), FadeIn(adaptive), GrowArrow(arrow))
        self.play(FadeIn(points, lag_ratio=0.03), FadeIn(moving_centers))
        self.play(Transform(moving_centers, target_centers), run_time=1.2)
        self.play(Write(labels))
        self.finish_narration(narration)

    def bridge_to_next_methods(self) -> None:
        self.clear()
        narration = self.start_narration("scene07")
        label = self.section_label()
        title = self.scene_title("限界が、次のモデルへの入口になる")

        items = VGroup(
            self.summary_item("固定基底: 解析しやすい", ADAPTIVE_GREEN),
            self.summary_item("高次元: 基底数が増えすぎる", GRID_ORANGE),
            self.summary_item("データ形状: 空白に部品を置きがち", MODEL_RED),
            self.summary_item("次: カーネル法やニューラルネット", KERNEL_TEAL),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT).shift(LEFT * 2.55 + DOWN * 0.05)
        flow = VGroup(
            self.method_card("kernel", "内積で特徴を扱う", KERNEL_TEAL),
            self.method_card("sparse", "必要な基底だけ使う", GRID_ORANGE),
            self.method_card("neural net", "基底を学習する", ADAPTIVE_GREEN),
        ).arrange(DOWN, buff=0.2).shift(RIGHT * 3.0)
        next_label = Text("固定した特徴量から、学習する特徴量へ", font_size=28, color=WHITE)
        next_label.to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(items, lag_ratio=0.14))
        self.play(FadeIn(flow, lag_ratio=0.18))
        self.play(Write(next_label))
        self.finish_narration(narration)

    def small_card(self, title: str, body: str, color: ManimColor) -> VGroup:
        rect = RoundedRectangle(width=2.35, height=1.0, corner_radius=0.08, stroke_color=color, stroke_width=3)
        rect.set_fill("#1B1B1B", opacity=0.86)
        title_text = Text(title, font_size=25, color=color)
        body_text = Text(body, font_size=19, color=WHITE).next_to(title_text, DOWN, buff=0.12)
        content = VGroup(title_text, body_text).move_to(rect)
        return VGroup(rect, content)

    def grid_1d(self, count: int, dim: str, total: str) -> VGroup:
        cells = VGroup(*[Square(side_length=0.34, color=GRID_ORANGE, stroke_width=2) for _ in range(count)]).arrange(RIGHT, buff=0.04)
        label = VGroup(Text(dim, font_size=26), MathTex(total, font_size=34, color=GRID_ORANGE)).arrange(DOWN, buff=0.12)
        label.next_to(cells, DOWN, buff=0.28)
        return VGroup(cells, label)

    def grid_2d(self, size: int, dim: str, total: str) -> VGroup:
        cells = VGroup()
        for r in range(size):
            for c in range(size):
                cell = Square(side_length=0.22, color=GRID_ORANGE, stroke_width=1.4)
                cell.move_to(np.array([(c - (size - 1) / 2) * 0.25, ((size - 1) / 2 - r) * 0.25, 0.0]))
                cells.add(cell)
        label = VGroup(Text(dim, font_size=26), MathTex(total, font_size=34, color=GRID_ORANGE)).arrange(DOWN, buff=0.12)
        label.next_to(cells, DOWN, buff=0.28)
        return VGroup(cells, label)

    def grid_3d(self, size: int, dim: str, total: str) -> VGroup:
        cells = VGroup()
        for layer in range(size):
            for r in range(size):
                for c in range(size):
                    cell = Square(side_length=0.14, color=GRID_ORANGE, stroke_width=0.9)
                    cell.set_fill("#242424", opacity=0.35)
                    cell.move_to(
                        np.array(
                            [
                                (c - (size - 1) / 2) * 0.17 + layer * 0.12,
                                ((size - 1) / 2 - r) * 0.17 + layer * 0.09,
                                0.0,
                            ]
                        )
                    )
                    cells.add(cell)
        label = VGroup(Text(dim, font_size=26), MathTex(total, font_size=34, color=GRID_ORANGE)).arrange(DOWN, buff=0.12)
        label.next_to(cells, DOWN, buff=0.28)
        return VGroup(cells, label)

    def center_panel(self, title: str, color: ManimColor) -> VGroup:
        rect = RoundedRectangle(width=3.0, height=3.0, corner_radius=0.08, stroke_color=color, stroke_width=3)
        rect.set_fill("#1A1A1A", opacity=0.75)
        label = Text(title, font_size=27, color=color).next_to(rect, UP, buff=0.18)
        centers = VGroup(
            Dot(rect.get_center() + LEFT * 0.8 + UP * 0.6, color=color, radius=0.08),
            Dot(rect.get_center() + RIGHT * 0.55 + UP * 0.35, color=color, radius=0.08),
            Dot(rect.get_center() + LEFT * 0.25 + DOWN * 0.65, color=color, radius=0.08),
            Dot(rect.get_center() + RIGHT * 0.95 + DOWN * 0.55, color=color, radius=0.08),
        )
        return VGroup(rect, label, centers)

    def cluster_points(self) -> VGroup:
        points = VGroup()
        coordinates = [
            (-0.9, 0.55),
            (-0.72, 0.42),
            (-0.58, 0.65),
            (-0.38, 0.48),
            (-0.18, 0.38),
            (0.02, 0.16),
            (0.22, -0.02),
            (0.36, -0.28),
            (0.52, -0.46),
            (0.72, -0.36),
            (0.84, -0.12),
            (0.98, 0.2),
            (1.12, 0.45),
            (1.25, 0.62),
        ]
        for x, y in coordinates:
            points.add(Dot(np.array([x, y, 0.0]), color=BLUE_DATA, radius=0.055))
        return points

    def summary_item(self, text: str, color: ManimColor) -> VGroup:
        bullet = Dot(color=color, radius=0.08)
        label = Text(text, font_size=25, color=WHITE)
        return VGroup(bullet, label).arrange(RIGHT, buff=0.2)

    def method_card(self, title: str, body: str, color: ManimColor) -> VGroup:
        rect = RoundedRectangle(width=3.1, height=0.95, corner_radius=0.08, stroke_color=color, stroke_width=3)
        rect.set_fill("#1B1B1B", opacity=0.86)
        title_text = Text(title, font_size=24, color=color)
        body_text = Text(body, font_size=18, color=WHITE)
        content = VGroup(title_text, body_text).arrange(DOWN, buff=0.08).move_to(rect)
        return VGroup(rect, content)
