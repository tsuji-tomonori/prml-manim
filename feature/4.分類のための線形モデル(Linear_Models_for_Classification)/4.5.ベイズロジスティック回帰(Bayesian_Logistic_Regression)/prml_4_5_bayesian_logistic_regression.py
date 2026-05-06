from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from manim import *


JAPANESE_FONT = "Noto Sans CJK JP"
SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

BACKGROUND = "#101010"
TEXT_GREY = GREY_B
CLASS0_BLUE = BLUE_C
CLASS1_RED = RED_C
POSTERIOR_YELLOW = YELLOW_C
MAP_GREEN = GREEN_C
UNCERTAINTY_ORANGE = ORANGE
PURPLE_MID = PURPLE_C

X_MIN, X_MAX = -2.4, 2.4
Y_MIN, Y_MAX = -1.75, 1.85

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def make_demo_data() -> tuple[np.ndarray, np.ndarray]:
    class0 = np.array(
        [
            [-1.85, -0.70],
            [-1.55, 0.15],
            [-1.25, 0.95],
            [-1.05, -0.20],
            [-0.72, 0.72],
            [-0.48, -1.02],
            [-0.22, 0.28],
        ]
    )
    class1 = np.array(
        [
            [0.42, -0.42],
            [0.72, 0.62],
            [1.05, -0.92],
            [1.28, 0.12],
            [1.55, 0.92],
            [1.78, -0.35],
            [2.02, 0.45],
        ]
    )
    x = np.vstack([class0, class1])
    t = np.concatenate([np.zeros(len(class0)), np.ones(len(class1))])
    return x, t


def feature_matrix(points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points)
    return np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])


def fit_bayesian_logistic_regression(x: np.ndarray, t: np.ndarray, alpha: float = 0.65) -> tuple[np.ndarray, np.ndarray]:
    phi = feature_matrix(x)
    w = np.zeros(phi.shape[1])
    precision = alpha * np.eye(phi.shape[1])
    for _ in range(40):
        y = sigmoid(phi @ w)
        r = np.maximum(y * (1.0 - y), 1.0e-6)
        gradient = phi.T @ (t - y) - precision @ w
        hessian_precision = precision + (phi.T * r) @ phi
        step = np.linalg.solve(hessian_precision, gradient)
        w = w + step
        if np.linalg.norm(step) < 1.0e-9:
            break
    y = sigmoid(phi @ w)
    r = np.maximum(y * (1.0 - y), 1.0e-6)
    posterior_precision = precision + (phi.T * r) @ phi
    covariance = np.linalg.inv(posterior_precision)
    return w, covariance


class PRML45BayesianLogisticRegression(Scene):
    """PRML 4.5 Bayesian logistic regression explanation video.

    Render example:
        uv run manim --disable_caching --flush_cache -ql prml_4_5_bayesian_logistic_regression.py PRML45BayesianLogisticRegression
    """

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        self.x_data, self.t_data = make_demo_data()
        self.w_map, self.s_n = fit_bayesian_logistic_regression(self.x_data, self.t_data)

        self.opening_weight_distribution()
        self.posterior_from_prior_and_likelihood()
        self.laplace_approximation_scene()
        self.map_logistic_classifier()
        self.logit_uncertainty_scene()
        self.predictive_distribution_scene()
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

    def finish_narration(self, narration: tuple[float, float | None], pad: float = 0.25) -> None:
        start_time, duration = narration
        if duration is None:
            return
        elapsed = float(getattr(self, "time", 0.0)) - start_time
        remaining = duration - elapsed + pad
        if remaining > 0:
            self.wait(remaining)

    def clear_scene(self) -> None:
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.7)

    def section_label(self, text: str) -> Text:
        label = Text(text, font_size=18, color=TEXT_GREY)
        label.to_corner(UL)
        return label

    def scene_title(self, text: str, font_size: int = 34) -> Text:
        title = Text(text, font_size=font_size)
        title.to_edge(UP).shift(DOWN * 0.35)
        return title

    def map_probability(self, points: np.ndarray) -> np.ndarray:
        return sigmoid(feature_matrix(points) @ self.w_map)

    def bayesian_probability(self, points: np.ndarray) -> np.ndarray:
        phi = feature_matrix(points)
        mu = phi @ self.w_map
        var = np.einsum("ij,jk,ik->i", phi, self.s_n, phi)
        kappa = 1.0 / np.sqrt(1.0 + math.pi * var / 8.0)
        return sigmoid(kappa * mu)

    def logit_stats(self, point: np.ndarray) -> tuple[float, float]:
        phi = feature_matrix(np.asarray(point))[0]
        mu = float(phi @ self.w_map)
        var = float(phi @ self.s_n @ phi)
        return mu, math.sqrt(max(var, 1.0e-8))

    def make_curve(self, axes: Axes, x_values: np.ndarray, y_values: np.ndarray, color: ManimColor, stroke_width: float = 4.0) -> VMobject:
        points = [axes.c2p(float(x), float(y)) for x, y in zip(x_values, y_values)]
        curve = VMobject(color=color)
        curve.set_points_smoothly(points)
        curve.set_stroke(width=stroke_width)
        return curve

    def make_classification_axes(self, width: float, height: float, shift: np.ndarray) -> Axes:
        axes = Axes(
            x_range=[X_MIN, X_MAX, 1.2],
            y_range=[Y_MIN, Y_MAX, 1.2],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )
        axes.shift(shift)
        return axes

    def make_probability_heatmap(
        self,
        axes: Axes,
        probability_fn,
        nx: int = 28,
        ny: int = 22,
        opacity: float = 0.73,
    ) -> VGroup:
        x_edges = np.linspace(X_MIN, X_MAX, nx + 1)
        y_edges = np.linspace(Y_MIN, Y_MAX, ny + 1)
        cells = VGroup()
        for left, right in zip(x_edges[:-1], x_edges[1:]):
            for bottom, top in zip(y_edges[:-1], y_edges[1:]):
                center = np.array([(left + right) / 2.0, (bottom + top) / 2.0])
                probability = float(np.clip(probability_fn(center[None, :])[0], 0.0, 1.0))
                color = interpolate_color(CLASS0_BLUE, CLASS1_RED, probability)
                scene_width = abs(axes.c2p(float(right), 0.0)[0] - axes.c2p(float(left), 0.0)[0]) * 1.02
                scene_height = abs(axes.c2p(0.0, float(top))[1] - axes.c2p(0.0, float(bottom))[1]) * 1.02
                rect = Rectangle(
                    width=scene_width,
                    height=scene_height,
                    stroke_width=0,
                    fill_color=color,
                    fill_opacity=opacity,
                )
                rect.move_to(axes.c2p(float(center[0]), float(center[1])))
                cells.add(rect)
        cells.set_z_index(-3)
        return cells

    def make_data_points(self, axes: Axes, radius: float = 0.065) -> VGroup:
        dots = VGroup()
        for point, target in zip(self.x_data, self.t_data):
            color = CLASS1_RED if target > 0.5 else CLASS0_BLUE
            dot = Dot(axes.c2p(float(point[0]), float(point[1])), radius=radius, color=color)
            dot.set_stroke(WHITE, width=1.0, opacity=0.85)
            dot.set_z_index(3)
            dots.add(dot)
        return dots

    def make_boundary_line(self, axes: Axes, w: np.ndarray, color: ManimColor = WHITE) -> Mobject:
        candidates: list[tuple[float, float]] = []
        if abs(w[2]) > 1.0e-8:
            for x in (X_MIN, X_MAX):
                y = -(w[0] + w[1] * x) / w[2]
                if Y_MIN <= y <= Y_MAX:
                    candidates.append((float(x), float(y)))
        if abs(w[1]) > 1.0e-8:
            for y in (Y_MIN, Y_MAX):
                x = -(w[0] + w[2] * y) / w[1]
                if X_MIN <= x <= X_MAX:
                    candidates.append((float(x), float(y)))
        unique: list[tuple[float, float]] = []
        for candidate in candidates:
            if not any(np.linalg.norm(np.array(candidate) - np.array(existing)) < 1.0e-5 for existing in unique):
                unique.append(candidate)
        if len(unique) < 2:
            return VGroup()
        unique.sort(key=lambda item: (item[0], item[1]))
        line = Line(
            axes.c2p(unique[0][0], unique[0][1]),
            axes.c2p(unique[-1][0], unique[-1][1]),
            color=color,
            stroke_width=4.5,
        )
        line.set_z_index(2)
        return line

    def make_classifier_panel(
        self,
        title: str,
        probability_kind: str,
        shift: np.ndarray,
        width: float = 5.2,
        height: float = 3.55,
        resolution: tuple[int, int] = (28, 22),
    ) -> VGroup:
        axes = self.make_classification_axes(width, height, shift)
        probability_fn = self.map_probability if probability_kind == "map" else self.bayesian_probability
        heatmap = self.make_probability_heatmap(axes, probability_fn, nx=resolution[0], ny=resolution[1])
        points = self.make_data_points(axes)
        boundary = self.make_boundary_line(axes, self.w_map)
        title_mob = Text(title, font_size=22).next_to(axes, UP, buff=0.15)
        x_label = Text("x1", font_size=16, color=TEXT_GREY).next_to(axes.x_axis.get_end(), DOWN, buff=0.08)
        y_label = Text("x2", font_size=16, color=TEXT_GREY).next_to(axes.y_axis.get_end(), LEFT, buff=0.08)
        return VGroup(heatmap, axes, boundary, points, x_label, y_label, title_mob)

    def make_formula_box(self, label: str, formula: str, color: ManimColor, width: float = 4.8, height: float = 1.35) -> VGroup:
        rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.08,
            stroke_color=color,
            stroke_width=2.0,
            fill_color="#181818",
            fill_opacity=0.92,
        )
        title = Text(label, font_size=21, color=color).move_to(rect.get_top() + DOWN * 0.28)
        math = MathTex(formula, font_size=26, color=WHITE)
        math.next_to(title, DOWN, buff=0.18)
        if math.width > width - 0.35:
            math.scale((width - 0.35) / math.width)
        return VGroup(rect, title, math)

    def make_legend(self) -> VGroup:
        legend = VGroup(
            Dot(color=CLASS0_BLUE, radius=0.07),
            Text("C2", font_size=18, color=CLASS0_BLUE),
            Dot(color=CLASS1_RED, radius=0.07),
            Text("C1", font_size=18, color=CLASS1_RED),
            Line(LEFT * 0.22, RIGHT * 0.22, color=WHITE, stroke_width=4),
            Text("p=0.5", font_size=18, color=WHITE),
        ).arrange(RIGHT, buff=0.13)
        return legend

    def one_dimensional_curves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w_values = np.linspace(-4.0, 4.0, 500)
        prior = np.exp(-0.5 * (w_values / 1.25) ** 2)
        feature_positions = np.array([-1.8, -1.1, -0.35, 0.45, 1.05, 1.7])
        targets = np.array([0, 0, 0, 1, 1, 1])
        y = sigmoid(np.outer(w_values, feature_positions))
        likelihood = np.prod((y**targets) * ((1.0 - y) ** (1.0 - targets)), axis=1)
        posterior = prior * likelihood
        prior = prior / np.max(prior)
        likelihood = likelihood / np.max(likelihood)
        posterior = posterior / np.max(posterior)
        return w_values, prior, likelihood, posterior

    def opening_weight_distribution(self) -> None:
        narration = self.start_narration("scene01")
        label = self.section_label("PRML 4.5 Bayesian Logistic Regression")
        title = self.scene_title("重み w を、一点ではなく分布として扱う", font_size=33)
        map_panel = self.make_classifier_panel("一点推定: w_MAP", "map", LEFT * 3.15 + DOWN * 0.25, width=4.4, height=3.05, resolution=(22, 16))

        weight_axes = Axes(
            x_range=[-0.6, 2.4, 1.0],
            y_range=[-1.6, 1.6, 1.0],
            x_length=4.35,
            y_length=3.05,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 3.05 + DOWN * 0.25)
        center = weight_axes.c2p(float(self.w_map[1]), float(self.w_map[2]))
        ellipse_wide = Ellipse(width=1.85, height=0.82, color=POSTERIOR_YELLOW, stroke_width=4)
        ellipse_wide.rotate(0.22).move_to(center)
        ellipse_narrow = Ellipse(width=0.92, height=0.41, color=POSTERIOR_YELLOW, stroke_width=3)
        ellipse_narrow.rotate(0.22).move_to(center)
        mean_dot = Dot(center, color=WHITE, radius=0.08)
        weight_title = Text("ベイズ: p(w|t)", font_size=22).next_to(weight_axes, UP, buff=0.15)
        w1_label = Text("w1", font_size=16, color=TEXT_GREY).next_to(weight_axes.x_axis.get_end(), DOWN, buff=0.08)
        w2_label = Text("w2", font_size=16, color=TEXT_GREY).next_to(weight_axes.y_axis.get_end(), LEFT, buff=0.08)
        weight_panel = VGroup(weight_axes, ellipse_wide, ellipse_narrow, mean_dot, weight_title, w1_label, w2_label)

        formula = MathTex(
            r"p(\mathbf{w}\mid\mathbf{t}) \propto p(\mathbf{w})\prod_n p(t_n\mid\mathbf{w})",
            font_size=31,
            color=WHITE,
        )
        formula.to_edge(DOWN).shift(UP * 0.2)
        note = Text("正確な正規化と予測積分が難しい", font_size=24, color=UNCERTAINTY_ORANGE).next_to(formula, UP, buff=0.18)
        arrow = Arrow(map_panel.get_right() + RIGHT * 0.1, weight_panel.get_left() + LEFT * 0.1, buff=0.15, color=TEXT_GREY)

        self.play(FadeIn(label), Write(title), run_time=1.4)
        self.play(FadeIn(map_panel), run_time=1.7)
        self.play(GrowArrow(arrow), FadeIn(weight_panel), run_time=1.8)
        self.play(Write(formula), FadeIn(note), run_time=1.5)
        self.play(Indicate(ellipse_wide, color=YELLOW), Indicate(note, color=YELLOW), run_time=1.4)
        self.finish_narration(narration)
        self.clear_scene()

    def posterior_from_prior_and_likelihood(self) -> None:
        narration = self.start_narration("scene02")
        label = self.section_label("Posterior from prior and likelihood")
        title = self.scene_title("事前分布とシグモイド尤度を掛け合わせる", font_size=33)
        w_values, prior, likelihood, posterior = self.one_dimensional_curves()
        axes = Axes(
            x_range=[-4, 4, 2],
            y_range=[0, 1.1, 0.5],
            x_length=8.6,
            y_length=3.35,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.35)
        prior_curve = self.make_curve(axes, w_values, prior, CLASS0_BLUE, 4)
        like_curve = self.make_curve(axes, w_values, likelihood, CLASS1_RED, 4)
        post_curve = self.make_curve(axes, w_values, posterior, POSTERIOR_YELLOW, 5)
        legend = VGroup(
            Line(LEFT * 0.28, RIGHT * 0.28, color=CLASS0_BLUE, stroke_width=5),
            Text("prior", font_size=18),
            Line(LEFT * 0.28, RIGHT * 0.28, color=CLASS1_RED, stroke_width=5),
            Text("likelihood", font_size=18),
            Line(LEFT * 0.28, RIGHT * 0.28, color=POSTERIOR_YELLOW, stroke_width=5),
            Text("posterior", font_size=18),
        ).arrange(RIGHT, buff=0.13)
        legend.next_to(axes, UP, buff=0.15)
        formula_1 = MathTex(
            r"\ln p(\mathbf{w}\mid\mathbf{t}) = -\frac12(\mathbf{w}-\mathbf{m}_0)^T\mathbf{S}_0^{-1}(\mathbf{w}-\mathbf{m}_0)",
            font_size=25,
        )
        formula_2 = MathTex(
            r"+\sum_n\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\}+\mathrm{const}",
            font_size=25,
        )
        formulas = VGroup(formula_1, formula_2).arrange(DOWN, buff=0.12)
        formulas.to_edge(DOWN).shift(UP * 0.12)

        self.play(FadeIn(label), Write(title), Create(axes), run_time=1.4)
        self.play(Create(prior_curve), FadeIn(legend[0:2]), run_time=1.3)
        self.play(Create(like_curve), FadeIn(legend[2:4]), run_time=1.3)
        self.play(Create(post_curve), FadeIn(legend[4:6]), run_time=1.5)
        self.play(Write(formulas), run_time=1.7)
        self.play(Indicate(post_curve, color=YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.clear_scene()

    def laplace_approximation_scene(self) -> None:
        narration = self.start_narration("scene03")
        label = self.section_label("Laplace approximation")
        title = self.scene_title("MAP の近くで、事後分布をガウスに置き換える", font_size=32)
        w_values, _, _, posterior = self.one_dimensional_curves()
        axes = Axes(
            x_range=[-4, 4, 2],
            y_range=[0, 1.15, 0.5],
            x_length=8.4,
            y_length=3.15,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.25)
        posterior_curve = self.make_curve(axes, w_values, posterior, POSTERIOR_YELLOW, 5)
        mode = float(w_values[int(np.argmax(posterior))])
        gaussian = np.exp(-0.5 * ((w_values - mode) / 0.72) ** 2)
        gaussian = gaussian / np.max(gaussian) * np.max(posterior)
        gaussian_curve = self.make_curve(axes, w_values, gaussian, MAP_GREEN, 4)
        mode_line = DashedLine(axes.c2p(mode, 0), axes.c2p(mode, 1.08), color=WHITE, stroke_width=3)
        mode_label = Text("w_MAP", font_size=22, color=WHITE).next_to(mode_line, UP, buff=0.05)
        formula_a = MathTex(
            r"q(\mathbf{w})=\mathcal{N}(\mathbf{w}\mid\mathbf{w}_{MAP},\mathbf{S}_N)",
            font_size=29,
            color=WHITE,
        )
        formula_b = MathTex(
            r"\mathbf{S}_N^{-1}=\mathbf{S}_0^{-1}+\sum_n y_n(1-y_n)\phi_n\phi_n^T",
            font_size=27,
            color=WHITE,
        )
        formulas = VGroup(formula_a, formula_b).arrange(DOWN, buff=0.14)
        formulas.to_edge(DOWN).shift(UP * 0.1)
        note = Text("曲率が大きい方向ほど、分散は小さい", font_size=24, color=UNCERTAINTY_ORANGE)
        note.next_to(axes, UP, buff=0.15)

        self.play(FadeIn(label), Write(title), Create(axes), run_time=1.3)
        self.play(Create(posterior_curve), run_time=1.4)
        self.play(Create(mode_line), FadeIn(mode_label), run_time=1.1)
        self.play(Create(gaussian_curve), FadeIn(note), run_time=1.6)
        self.play(Write(formulas), run_time=1.8)
        self.play(Indicate(gaussian_curve, color=YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.clear_scene()

    def map_logistic_classifier(self) -> None:
        narration = self.start_narration("scene04")
        label = self.section_label("MAP logistic classifier")
        title = self.scene_title("MAP 解で作るロジスティック分類器", font_size=34)
        panel = self.make_classifier_panel("p(C1 | phi, w_MAP)", "map", DOWN * 0.2, width=8.2, height=4.25, resolution=(34, 24))
        legend = self.make_legend().next_to(panel[1], DOWN, buff=0.22)
        formula = MathTex(r"p(C_1\mid\phi,\mathbf{w}_{MAP})=\sigma(\mathbf{w}_{MAP}^T\phi)", font_size=31)
        formula.next_to(title, DOWN, buff=0.12)

        self.play(FadeIn(label), Write(title), Write(formula), run_time=1.5)
        self.play(FadeIn(panel[0]), Create(panel[1]), FadeIn(panel[4:]), run_time=1.6)
        self.play(FadeIn(panel[3]), run_time=1.2)
        self.play(Create(panel[2]), FadeIn(legend), run_time=1.4)
        self.play(Indicate(panel[2], color=YELLOW), run_time=1.2)
        self.finish_narration(narration)
        self.clear_scene()

    def logit_uncertainty_scene(self) -> None:
        narration = self.start_narration("scene05")
        label = self.section_label("Uncertainty in the logit")
        title = self.scene_title("重みの分布は、ロジット a の分布になる", font_size=33)

        axes_left = self.make_classification_axes(4.8, 3.45, LEFT * 3.05 + DOWN * 0.2)
        points = self.make_data_points(axes_left)
        boundary = self.make_boundary_line(axes_left, self.w_map)
        near_point = np.array([1.12, 0.38])
        far_point = np.array([2.15, 1.55])
        near_dot = Star(n=5, outer_radius=0.14, inner_radius=0.06, color=WHITE, fill_opacity=1.0)
        near_dot.move_to(axes_left.c2p(float(near_point[0]), float(near_point[1])))
        far_dot = Star(n=5, outer_radius=0.14, inner_radius=0.06, color=UNCERTAINTY_ORANGE, fill_opacity=1.0)
        far_dot.move_to(axes_left.c2p(float(far_point[0]), float(far_point[1])))
        near_label = Text("データの近く", font_size=18, color=WHITE).next_to(near_dot, DOWN, buff=0.12)
        far_label = Text("離れた場所", font_size=18, color=UNCERTAINTY_ORANGE).next_to(far_dot, UP, buff=0.12)
        left_title = Text("入力空間", font_size=22).next_to(axes_left, UP, buff=0.13)

        mu_near, sigma_near = self.logit_stats(near_point)
        mu_far, sigma_far = self.logit_stats(far_point)
        a_values = np.linspace(-6.0, 6.0, 500)
        near_pdf = normal_pdf(a_values, mu_near, sigma_near)
        far_pdf = normal_pdf(a_values, mu_far, sigma_far)
        near_pdf = near_pdf / np.max(near_pdf)
        far_pdf = far_pdf / np.max(far_pdf)
        axes_right = Axes(
            x_range=[-6, 6, 3],
            y_range=[0, 1.1, 0.5],
            x_length=4.9,
            y_length=3.1,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 3.0 + DOWN * 0.25)
        near_curve = self.make_curve(axes_right, a_values, near_pdf, WHITE, 4)
        far_curve = self.make_curve(axes_right, a_values, far_pdf, UNCERTAINTY_ORANGE, 4)
        right_title = Text("a = w^T phi の分布", font_size=22).next_to(axes_right, UP, buff=0.13)
        a_label = Text("logit a", font_size=17, color=TEXT_GREY).next_to(axes_right.x_axis.get_end(), DOWN, buff=0.08)
        curve_labels = VGroup(
            Line(LEFT * 0.25, RIGHT * 0.25, color=WHITE, stroke_width=5),
            Text("狭い", font_size=18),
            Line(LEFT * 0.25, RIGHT * 0.25, color=UNCERTAINTY_ORANGE, stroke_width=5),
            Text("広い", font_size=18),
        ).arrange(RIGHT, buff=0.14)
        curve_labels.next_to(axes_right, DOWN, buff=0.15)
        formula = MathTex(
            r"\mu_a=\mathbf{w}_{MAP}^T\phi",
            r"\qquad",
            r"\sigma_a^2=\phi^T\mathbf{S}_N\phi",
            font_size=31,
        )
        formula.to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(label), Write(title), run_time=1.3)
        self.play(Create(axes_left), FadeIn(points), Create(boundary), FadeIn(left_title), run_time=1.5)
        self.play(FadeIn(near_dot), FadeIn(far_dot), FadeIn(near_label), FadeIn(far_label), run_time=1.4)
        self.play(Create(axes_right), FadeIn(right_title), FadeIn(a_label), run_time=1.1)
        self.play(Create(near_curve), Create(far_curve), FadeIn(curve_labels), run_time=1.7)
        self.play(Write(formula), run_time=1.4)
        self.play(Indicate(far_curve, color=YELLOW), run_time=1.1)
        self.finish_narration(narration)
        self.clear_scene()

    def predictive_distribution_scene(self) -> None:
        narration = self.start_narration("scene06")
        label = self.section_label("Predictive distribution")
        title = self.scene_title("積分すると、不確かな場所の確率は 0.5 に近づく", font_size=32)
        formula_1 = MathTex(
            r"p(C_1\mid\phi,\mathbf{t})\simeq\int\sigma(a)\mathcal{N}(a\mid\mu_a,\sigma_a^2)\,da",
            font_size=23,
        )
        formula_2 = MathTex(
            r"\simeq \sigma(\kappa(\sigma_a^2)\mu_a),\quad \kappa(\sigma_a^2)=(1+\pi\sigma_a^2/8)^{-1/2}",
            font_size=23,
        )
        formulas = VGroup(formula_1, formula_2).arrange(DOWN, buff=0.1)
        formulas.next_to(title, DOWN, buff=0.1)
        map_panel = self.make_classifier_panel("MAP 予測", "map", LEFT * 2.95 + DOWN * 0.92, width=5.15, height=3.1, resolution=(26, 20))
        bayes_panel = self.make_classifier_panel("ベイズ予測", "bayes", RIGHT * 2.95 + DOWN * 0.92, width=5.15, height=3.1, resolution=(26, 20))
        legend = self.make_legend().to_edge(DOWN).shift(UP * 0.08)
        note = Text("境界は同じ。遠い領域の色だけが弱くなる。", font_size=24, color=UNCERTAINTY_ORANGE)
        note.next_to(legend, UP, buff=0.12)

        self.play(FadeIn(label), Write(title), run_time=1.2)
        self.play(Write(formulas), run_time=1.8)
        self.play(FadeIn(map_panel), run_time=1.8)
        self.play(FadeIn(bayes_panel), FadeIn(legend), run_time=1.8)
        self.play(FadeIn(note), Indicate(bayes_panel[0], color=YELLOW), run_time=1.5)
        self.finish_narration(narration)
        self.clear_scene()

    def make_step_box(self, title: str, formula: str, color: ManimColor) -> VGroup:
        rect = RoundedRectangle(
            width=2.85,
            height=1.35,
            corner_radius=0.08,
            stroke_color=color,
            fill_color="#181818",
            fill_opacity=0.95,
            stroke_width=2.0,
        )
        title_mob = Text(title, font_size=19, color=color).move_to(rect.get_top() + DOWN * 0.28)
        formula_mob = MathTex(formula, font_size=25, color=WHITE).next_to(title_mob, DOWN, buff=0.18)
        if formula_mob.width > 2.55:
            formula_mob.scale(2.55 / formula_mob.width)
        return VGroup(rect, title_mob, formula_mob)

    def summary_scene(self) -> None:
        narration = self.start_narration("scene07")
        label = self.section_label("Summary")
        title = self.scene_title("4.5 ベイズロジスティック回帰の流れ", font_size=34)
        steps = VGroup(
            self.make_step_box("1. 事前分布", r"p(\mathbf{w})", CLASS0_BLUE),
            self.make_step_box("2. MAP", r"\mathbf{w}_{MAP}", MAP_GREEN),
            self.make_step_box("3. ラプラス近似", r"q(\mathbf{w})", POSTERIOR_YELLOW),
            self.make_step_box("4. 予測積分", r"\int\sigma(\mathbf{w}^T\phi)q(\mathbf{w})d\mathbf{w}", UNCERTAINTY_ORANGE),
        ).arrange(RIGHT, buff=0.35)
        steps.shift(UP * 0.55)
        arrows = VGroup()
        for left_box, right_box in zip(steps[:-1], steps[1:]):
            arrows.add(Arrow(left_box.get_right(), right_box.get_left(), buff=0.08, color=TEXT_GREY, max_tip_length_to_length_ratio=0.18))

        takeaways = VGroup(
            Text("正確なベイズ推論は intractable", font_size=24, color=TEXT_GREY),
            Text("MAP とヘッセ行列からガウス近似を作る", font_size=24, color=WHITE),
            Text("予測確率は、不確実な場所ほど 0.5 側へ戻る", font_size=24, color=UNCERTAINTY_ORANGE),
            Text("決定境界 p=0.5 は MAP と同じ", font_size=24, color=MAP_GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        takeaways.next_to(steps, DOWN, buff=0.65)
        bridge = Text("後のガウス過程分類・変分ロジスティック回帰への入口", font_size=25, color=POSTERIOR_YELLOW)
        bridge.to_edge(DOWN).shift(UP * 0.35)

        self.play(FadeIn(label), Write(title), run_time=1.3)
        self.play(FadeIn(steps[0]), run_time=0.8)
        for arrow, step in zip(arrows, steps[1:]):
            self.play(GrowArrow(arrow), FadeIn(step), run_time=0.8)
        self.play(FadeIn(takeaways, shift=UP * 0.15), run_time=1.7)
        self.play(FadeIn(bridge), run_time=1.1)
        self.finish_narration(narration)
        self.clear_scene()
