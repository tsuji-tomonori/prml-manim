from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from manim import *


BLUE_DATA = BLUE_C
MODEL_RED = RED_C
EVIDENCE_ORANGE = ORANGE
PRIOR_PURPLE = PURPLE_C
POSTERIOR_GREEN = GREEN_C
NOISE_TEAL = TEAL_C
TEXT_GREY = GREY_B
JAPANESE_FONT = "Noto Sans CJK JP"

SCENE_DIR = Path(__file__).resolve().parent
VOICEOVER_DIR = SCENE_DIR / "assets" / "voicevox"

ManimText = Text


def Text(*args, **kwargs) -> ManimText:
    kwargs.setdefault("font", JAPANESE_FONT)
    return ManimText(*args, **kwargs)


def make_sine_data(n: int = 25, noise_std: float = 0.28, seed: int = 35) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    t = np.sin(2.0 * np.pi * x) + rng.normal(0.0, noise_std, size=n)
    return x, t


def gaussian_design_matrix(x: np.ndarray | float, basis_count: int = 9, scale: float = 0.16) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    centers = np.linspace(0.0, 1.0, basis_count)
    features = [np.ones_like(x_array)]
    for center in centers:
        features.append(np.exp(-0.5 * ((x_array - center) / scale) ** 2))
    return np.column_stack(features)


def posterior_stats(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float,
    beta: float,
    basis_count: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    phi = gaussian_design_matrix(x, basis_count=basis_count)
    a_matrix = alpha * np.eye(phi.shape[1]) + beta * phi.T @ phi
    m_n = beta * np.linalg.solve(a_matrix, phi.T @ t)
    lambdas = beta * np.linalg.eigvalsh(phi.T @ phi)
    gamma = float(np.sum(lambdas / (alpha + lambdas)))
    return phi, a_matrix, m_n, gamma


def predictive_mean(w: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    return gaussian_design_matrix(x, basis_count=len(w) - 1) @ w


def log_evidence(x: np.ndarray, t: np.ndarray, alpha: float, beta: float, basis_count: int = 9) -> float:
    phi, a_matrix, m_n, _ = posterior_stats(x, t, alpha, beta, basis_count=basis_count)
    n_points, m_dim = phi.shape
    residual = t - phi @ m_n
    energy = 0.5 * beta * float(residual @ residual) + 0.5 * alpha * float(m_n @ m_n)
    sign, log_det = np.linalg.slogdet(a_matrix)
    if sign <= 0:
        return float("-inf")
    return 0.5 * m_dim * np.log(alpha) + 0.5 * n_points * np.log(beta) - energy - 0.5 * log_det - 0.5 * n_points * np.log(2 * np.pi)


def evidence_iteration(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float = 1.0,
    beta: float = 8.0,
    steps: int = 6,
) -> list[dict[str, float]]:
    phi = gaussian_design_matrix(x)
    raw_eigs = np.linalg.eigvalsh(phi.T @ phi)
    states: list[dict[str, float]] = []
    for _ in range(steps):
        _, _, m_n, gamma = posterior_stats(x, t, alpha, beta)
        residual = t - phi @ m_n
        rss = max(float(residual @ residual), 1e-8)
        states.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "logev": log_evidence(x, t, alpha, beta),
            }
        )
        alpha = max(gamma / max(float(m_n @ m_n), 1e-8), 1e-6)
        lambdas = beta * raw_eigs
        gamma = float(np.sum(lambdas / (alpha + lambdas)))
        beta = max((len(t) - gamma) / rss, 1e-6)
    return states


class PRML35EvidenceApproximation(Scene):
    """PRML 3.5 evidence approximation overview.

    Render example:
        uv run manim -pql prml_3_5_evidence_approximation.py PRML35EvidenceApproximation
    """

    def construct(self) -> None:
        self.camera.background_color = "#111111"
        self.x_train, self.t_train = make_sine_data()
        self.beta_fixed = 11.1
        self.alpha_grid = np.exp(np.linspace(-5.0, 5.0, 80))
        self.log_alpha_grid = np.log(self.alpha_grid)
        self.log_evidence_values = np.array([log_evidence(self.x_train, self.t_train, a, self.beta_fixed) for a in self.alpha_grid])
        self.alpha_star = float(self.alpha_grid[int(np.argmax(self.log_evidence_values))])

        self.opening_evidence_approximation()
        self.evidence_integral()
        self.log_evidence_breakdown()
        self.alpha_evidence_slider()
        self.effective_parameter_count()
        self.beta_update()
        self.reestimation_loop()
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

    def make_data_axes(self, width: float = 5.7, height: float = 3.4) -> Axes:
        return Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.7, 1.7, 0.8],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def data_dots(self, axes: Axes) -> VGroup:
        return VGroup(
            *[
                Dot(axes.c2p(float(x), float(t)), color=BLUE_DATA, radius=0.045)
                for x, t in zip(self.x_train, self.t_train)
            ]
        )

    def regression_curve(self, axes: Axes, alpha: float, color: ManimColor = MODEL_RED) -> VMobject:
        _, _, m_n, _ = posterior_stats(self.x_train, self.t_train, alpha, self.beta_fixed)

        def model(u: float) -> float:
            return float(np.clip(predictive_mean(m_n, u)[0], -1.65, 1.65))

        curve = axes.plot(model, x_range=[0, 1], color=color, use_smoothing=False)
        curve.set_stroke(width=4)
        return curve

    def evidence_axes(self, width: float = 4.8, height: float = 3.2) -> Axes:
        low = float(np.min(self.log_evidence_values)) - 0.5
        high = float(np.max(self.log_evidence_values)) + 0.5
        return Axes(
            x_range=[-5, 5, 2.5],
            y_range=[low, high, 2.0],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )

    def evidence_curve(self, axes: Axes) -> VMobject:
        def curve_func(z: float) -> float:
            return float(np.interp(z, self.log_alpha_grid, self.log_evidence_values))

        curve = axes.plot(curve_func, x_range=[-5, 5], color=EVIDENCE_ORANGE, use_smoothing=False)
        curve.set_stroke(width=4)
        return curve

    def opening_evidence_approximation(self) -> None:
        self.clear()
        narration = self.start_narration("scene01")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("重みは積分し、ハイパーパラメータは選ぶ", font_size=32)

        full_box = self.flow_box("Fully Bayesian", [r"w", r"\alpha", r"\beta"], "integrate").shift(LEFT * 3.05 + DOWN * 0.15)
        approx_box = self.flow_box("Evidence approximation", [r"w"], "integrate").shift(RIGHT * 3.05 + DOWN * 0.15)
        maximize = VGroup(
            Text("maximize", font_size=24, color=EVIDENCE_ORANGE),
            MathTex(r"p(t|\alpha,\beta)", font_size=34, color=EVIDENCE_ORANGE),
            MathTex(r"\alpha,\beta", font_size=34, color=WHITE),
        ).arrange(DOWN, buff=0.16).next_to(approx_box, DOWN, buff=0.35)
        arrow = Arrow(full_box.get_right(), approx_box.get_left(), color=TEXT_GREY, buff=0.35)
        ratio = VGroup(
            MathTex(r"{\alpha \over \beta}", font_size=44, color=PRIOR_PURPLE),
            Text("正則化の強さ", font_size=24, color=TEXT_GREY),
        ).arrange(RIGHT, buff=0.25).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(full_box), GrowArrow(arrow), FadeIn(approx_box), run_time=1.2)
        self.play(Write(maximize), Write(ratio))
        self.finish_narration(narration)

    def flow_box(self, heading: str, variables: list[str], action: str) -> VGroup:
        rect = RoundedRectangle(width=4.15, height=2.4, corner_radius=0.08, stroke_color=GREY_B, stroke_width=3)
        rect.set_fill("#1B1B1B", opacity=0.9)
        h = Text(heading, font_size=24, color=WHITE)
        var_group = VGroup(*[MathTex(v, font_size=38, color=PRIOR_PURPLE if v != "w" else POSTERIOR_GREEN) for v in variables]).arrange(RIGHT, buff=0.35)
        action_text = Text(action, font_size=23, color=TEXT_GREY)
        content = VGroup(h, var_group, action_text).arrange(DOWN, buff=0.22).move_to(rect)
        return VGroup(rect, content)

    def evidence_integral(self) -> None:
        self.clear()
        narration = self.start_narration("scene02")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("evidence は重みで積分した説明力")

        axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[0, 1.05, 0.25],
            x_length=7.4,
            y_length=3.5,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        ).shift(DOWN * 0.35)
        likelihood = axes.plot(lambda w: np.exp(-0.5 * ((w - 0.85) / 0.62) ** 2), x_range=[-3.5, 3.5], color=MODEL_RED)
        prior = axes.plot(lambda w: 0.92 * np.exp(-0.5 * (w / 1.25) ** 2), x_range=[-3.5, 3.5], color=PRIOR_PURPLE)
        product = axes.plot(lambda w: 0.72 * np.exp(-0.5 * ((w - 0.55) / 0.48) ** 2), x_range=[-3.5, 3.5], color=EVIDENCE_ORANGE)
        area = axes.get_area(product, x_range=[-0.7, 1.8], color=EVIDENCE_ORANGE, opacity=0.45)
        formula = MathTex(r"p(t|\alpha,\beta)=\int p(t|w,\beta)p(w|\alpha)\,dw", font_size=40)
        formula.to_edge(UP).shift(DOWN * 1.0)
        legend = VGroup(
            self.legend_item("likelihood", MODEL_RED),
            self.legend_item("prior", PRIOR_PURPLE),
            self.legend_item("area = evidence", EVIDENCE_ORANGE),
        ).arrange(RIGHT, buff=0.45).to_edge(DOWN).shift(UP * 0.2)

        self.play(FadeIn(label), Write(title))
        self.play(Create(axes), Write(formula))
        self.play(Create(likelihood), Create(prior), FadeIn(legend[0]), FadeIn(legend[1]), run_time=1.2)
        self.play(Create(product), FadeIn(area), FadeIn(legend[2]), run_time=1.0)
        note = Text("高さだけでなく、幅も効く", font_size=28, color=WHITE).next_to(formula, DOWN, buff=0.35)
        self.play(Write(note))
        self.finish_narration(narration)

    def legend_item(self, text: str, color: ManimColor) -> VGroup:
        return VGroup(Line(ORIGIN, RIGHT * 0.45, color=color, stroke_width=5), Text(text, font_size=22, color=WHITE)).arrange(RIGHT, buff=0.15)

    def log_evidence_breakdown(self) -> None:
        self.clear()
        narration = self.start_narration("scene03")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("log evidence の中身")

        formula = MathTex(
            r"\ln p(t|\alpha,\beta)",
            "=",
            r"{M\over2}\ln\alpha",
            "+",
            r"{N\over2}\ln\beta",
            "-",
            r"E(m_N)",
            "-",
            r"{1\over2}\ln|A|",
            "-",
            r"{N\over2}\ln(2\pi)",
            font_size=35,
        ).shift(UP * 1.85)
        formula[2].set_color(PRIOR_PURPLE)
        formula[4].set_color(NOISE_TEAL)
        formula[6].set_color(MODEL_RED)
        formula[8].set_color(EVIDENCE_ORANGE)
        energy = MathTex(
            r"E(w)={\beta\over2}\|t-\Phi w\|^2+{\alpha\over2}w^Tw",
            font_size=38,
            color=WHITE,
        ).shift(UP * 0.55)
        cards = VGroup(
            self.term_card("data fit", r"\|t-\Phi m_N\|^2", MODEL_RED),
            self.term_card("weight size", r"m_N^Tm_N", PRIOR_PURPLE),
            self.term_card("volume penalty", r"\ln|A|", EVIDENCE_ORANGE),
        ).arrange(RIGHT, buff=0.35).shift(DOWN * 0.95)
        balance = Text("当てはまりと複雑さのつり合い", font_size=28, color=WHITE).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Write(formula), run_time=1.4)
        self.play(Write(energy))
        self.play(FadeIn(cards, lag_ratio=0.15))
        self.play(Write(balance))
        self.finish_narration(narration)

    def term_card(self, title: str, formula: str, color: ManimColor) -> VGroup:
        rect = RoundedRectangle(width=3.25, height=1.35, corner_radius=0.08, stroke_color=color, stroke_width=3)
        rect.set_fill("#1B1B1B", opacity=0.9)
        t = Text(title, font_size=22, color=color)
        f = MathTex(formula, font_size=29, color=WHITE)
        content = VGroup(t, f).arrange(DOWN, buff=0.16).move_to(rect)
        return VGroup(rect, content)

    def alpha_evidence_slider(self) -> None:
        self.clear()
        narration = self.start_narration("scene04")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("alpha は evidence の山で選ぶ")

        data_axes = self.make_data_axes(width=5.0, height=3.05).shift(LEFT * 3.05 + DOWN * 0.15)
        dots = self.data_dots(data_axes)
        evidence_axes = self.evidence_axes(width=4.7, height=3.05).shift(RIGHT * 3.0 + DOWN * 0.15)
        evidence_curve = self.evidence_curve(evidence_axes)
        x_label = MathTex(r"\ln\alpha", font_size=25, color=TEXT_GREY).next_to(evidence_axes.x_axis, DOWN, buff=0.28)
        y_label = Text("log evidence", font_size=20, color=TEXT_GREY).next_to(evidence_axes.y_axis, LEFT, buff=0.22)
        y_label.rotate(PI / 2)

        alpha_values = [np.exp(4.0), np.exp(-4.0), self.alpha_star]
        captions = ["硬すぎる", "自由すぎる", "evidence 最大"]
        colors = [PRIOR_PURPLE, MODEL_RED, POSTERIOR_GREEN]
        current_curve = self.regression_curve(data_axes, alpha_values[0], color=colors[0])
        log_alpha = float(np.log(alpha_values[0]))
        point = Dot(evidence_axes.c2p(log_alpha, log_evidence(self.x_train, self.t_train, alpha_values[0], self.beta_fixed)), color=colors[0], radius=0.075)
        caption = Text(captions[0], font_size=26, color=colors[0]).to_edge(DOWN).shift(UP * 0.25)
        slider = self.alpha_slider(log_alpha, colors[0]).next_to(caption, UP, buff=0.25)

        self.play(FadeIn(label), Write(title))
        self.play(Create(data_axes), FadeIn(dots), Create(evidence_axes), Write(x_label), Write(y_label), Create(evidence_curve))
        self.play(Create(current_curve), FadeIn(point), FadeIn(slider), Write(caption))
        for alpha, text, color in zip(alpha_values[1:], captions[1:], colors[1:]):
            log_a = float(np.log(alpha))
            new_curve = self.regression_curve(data_axes, alpha, color=color)
            new_point = Dot(evidence_axes.c2p(log_a, log_evidence(self.x_train, self.t_train, alpha, self.beta_fixed)), color=color, radius=0.075)
            new_caption = Text(text, font_size=26, color=color).to_edge(DOWN).shift(UP * 0.25)
            new_slider = self.alpha_slider(log_a, color).next_to(new_caption, UP, buff=0.25)
            self.play(
                Transform(current_curve, new_curve),
                Transform(point, new_point),
                Transform(slider, new_slider),
                Transform(caption, new_caption),
                run_time=1.2,
            )
        self.finish_narration(narration)

    def alpha_slider(self, log_alpha: float, color: ManimColor) -> VGroup:
        line = Line(LEFT * 2.1, RIGHT * 2.1, color=GREY_B, stroke_width=4)
        knob_x = np.interp(log_alpha, [-5, 5], [-2.1, 2.1])
        knob = Dot(line.get_center() + RIGHT * knob_x, color=color, radius=0.1)
        label = MathTex(r"\ln\alpha", font_size=25, color=TEXT_GREY).next_to(line, LEFT, buff=0.25)
        value = MathTex(f"{log_alpha:.1f}", font_size=25, color=color).next_to(line, RIGHT, buff=0.25)
        return VGroup(label, line, knob, value)

    def effective_parameter_count(self) -> None:
        self.clear()
        narration = self.start_narration("scene05")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("gamma は有効なパラメータ数")

        alpha = self.alpha_star
        phi = gaussian_design_matrix(self.x_train)
        lambdas = self.beta_fixed * np.linalg.eigvalsh(phi.T @ phi)
        ratios = np.sort(lambdas / (alpha + lambdas))[::-1]
        meters = VGroup()
        for i, ratio in enumerate(ratios):
            meter = self.gamma_meter(i, float(ratio))
            meters.add(meter)
        meters.arrange(RIGHT, buff=0.14).shift(UP * 0.35)
        gamma = float(np.sum(ratios))
        formula = MathTex(r"\gamma=\sum_i {\lambda_i\over \alpha+\lambda_i}", font_size=44, color=WHITE).shift(UP * 2.0)
        total = VGroup(
            Text("合計", font_size=26, color=TEXT_GREY),
            MathTex(rf"\gamma \approx {gamma:.1f}", font_size=48, color=POSTERIOR_GREEN),
            Text("data が決めた方向だけを数える", font_size=25, color=WHITE),
        ).arrange(DOWN, buff=0.18).shift(DOWN * 1.75)

        self.play(FadeIn(label), Write(title))
        self.play(Write(formula))
        self.play(FadeIn(meters, lag_ratio=0.08), run_time=1.5)
        self.play(Write(total))
        self.finish_narration(narration)

    def gamma_meter(self, index: int, ratio: float) -> VGroup:
        height = 2.0
        base = Rectangle(width=0.42, height=height, stroke_color=GREY_B, stroke_width=2)
        fill = Rectangle(width=0.42, height=max(height * ratio, 0.02), stroke_width=0, fill_color=POSTERIOR_GREEN, fill_opacity=0.85)
        fill.align_to(base, DOWN)
        label = MathTex(str(index + 1), font_size=18, color=TEXT_GREY).next_to(base, DOWN, buff=0.12)
        value = Text(f"{ratio:.2f}", font_size=16, color=WHITE).next_to(base, UP, buff=0.1)
        return VGroup(base, fill, label, value)

    def beta_update(self) -> None:
        self.clear()
        narration = self.start_narration("scene06")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("beta は N ではなく N - gamma で補正する", font_size=32)

        phi, _, m_n, gamma = posterior_stats(self.x_train, self.t_train, self.alpha_star, self.beta_fixed)
        residuals = self.t_train - phi @ m_n
        rss = float(residuals @ residuals)
        n_points = len(self.t_train)
        ml_var = rss / n_points
        bayes_var = rss / (n_points - gamma)

        bars = VGroup()
        labels = VGroup()
        for i, (name, value, color) in enumerate([("ML", ml_var, MODEL_RED), ("Bayes", bayes_var, NOISE_TEAL)]):
            bar = Rectangle(width=0.8, height=value * 6.0, stroke_width=0, fill_color=color, fill_opacity=0.85)
            bar.align_to(ORIGIN, DOWN).shift(RIGHT * (i - 0.5) * 1.25)
            bars.add(bar)
            labels.add(Text(name, font_size=24, color=color).next_to(bar, DOWN, buff=0.2))
        chart = VGroup(bars, labels).shift(LEFT * 3.1 + DOWN * 0.6)
        chart_title = Text("推定ノイズ分散", font_size=25, color=WHITE).next_to(chart, UP, buff=0.45)

        formula_ml = MathTex(r"\sigma^2_{\mathrm{ML}}={RSS\over N}", font_size=39, color=MODEL_RED)
        formula_bayes = MathTex(r"{1\over\beta}={RSS\over N-\gamma}", font_size=43, color=NOISE_TEAL)
        numbers = VGroup(
            MathTex(rf"N={n_points}", font_size=32, color=WHITE),
            MathTex(rf"\gamma\approx {gamma:.1f}", font_size=32, color=POSTERIOR_GREEN),
            MathTex(rf"N-\gamma\approx {n_points - gamma:.1f}", font_size=32, color=NOISE_TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        right = VGroup(formula_ml, formula_bayes, numbers).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(RIGHT * 2.45 + DOWN * 0.1)
        note = Text("使った自由度を差し引く", font_size=28, color=WHITE).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(chart_title), GrowFromEdge(bars, DOWN), FadeIn(labels))
        self.play(Write(formula_ml))
        self.play(Write(formula_bayes), Write(numbers))
        self.play(Write(note))
        self.finish_narration(narration)

    def reestimation_loop(self) -> None:
        self.clear()
        narration = self.start_narration("scene07")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("alpha, beta, gamma を反復して合わせる")

        cycle = self.update_cycle().shift(LEFT * 3.0 + DOWN * 0.15)
        states = evidence_iteration(self.x_train, self.t_train, alpha=0.25, beta=5.0, steps=6)
        points = self.iteration_points(states).shift(RIGHT * 3.05 + DOWN * 0.15)
        table = VGroup()
        for i, state in enumerate(states[:4]):
            row = VGroup(
                Text(f"{i + 1}", font_size=20, color=TEXT_GREY),
                MathTex(rf"\alpha={state['alpha']:.2f}", font_size=22, color=PRIOR_PURPLE),
                MathTex(rf"\beta={state['beta']:.1f}", font_size=22, color=NOISE_TEAL),
                MathTex(rf"\gamma={state['gamma']:.1f}", font_size=22, color=POSTERIOR_GREEN),
            ).arrange(RIGHT, buff=0.18)
            table.add(row)
        table.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(DOWN).shift(UP * 0.18)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(cycle, lag_ratio=0.12))
        self.play(FadeIn(points[0]), Write(points[1]))
        for dot in points[2]:
            self.play(FadeIn(dot), run_time=0.25)
        self.play(Write(table))
        self.finish_narration(narration)

    def update_cycle(self) -> VGroup:
        labels = [
            (r"\alpha,\beta", PRIOR_PURPLE, UP * 1.25),
            (r"m_N", MODEL_RED, RIGHT * 1.55),
            (r"\gamma", POSTERIOR_GREEN, DOWN * 1.25),
            (r"\alpha_{\mathrm{new}},\beta_{\mathrm{new}}", NOISE_TEAL, LEFT * 1.55),
        ]
        nodes = VGroup()
        for formula, color, pos in labels:
            circle = Circle(radius=0.55, color=color, stroke_width=3).set_fill("#1B1B1B", opacity=0.9).move_to(pos)
            text = MathTex(formula, font_size=28, color=color).move_to(circle)
            nodes.add(VGroup(circle, text))
        arrows = VGroup(
            CurvedArrow(nodes[0].get_right(), nodes[1].get_top(), angle=-TAU / 8, color=TEXT_GREY),
            CurvedArrow(nodes[1].get_bottom(), nodes[2].get_right(), angle=-TAU / 8, color=TEXT_GREY),
            CurvedArrow(nodes[2].get_left(), nodes[3].get_bottom(), angle=-TAU / 8, color=TEXT_GREY),
            CurvedArrow(nodes[3].get_top(), nodes[0].get_left(), angle=-TAU / 8, color=TEXT_GREY),
        )
        caption = Text("再推定", font_size=26, color=WHITE).move_to(ORIGIN)
        return VGroup(arrows, nodes, caption)

    def iteration_points(self, states: list[dict[str, float]]) -> VGroup:
        log_alphas = [np.log(s["alpha"]) for s in states]
        logev = [s["logev"] for s in states]
        axes = Axes(
            x_range=[min(log_alphas) - 0.4, max(log_alphas) + 0.4, 1],
            y_range=[min(logev) - 0.5, max(logev) + 0.5, 1],
            x_length=4.7,
            y_length=3.0,
            tips=False,
            axis_config={"color": GREY_B, "stroke_width": 2},
        )
        title = Text("反復ごとの evidence", font_size=23, color=WHITE).next_to(axes, UP, buff=0.25)
        dots = VGroup()
        for i, (x_val, y_val) in enumerate(zip(log_alphas, logev)):
            dot = Dot(axes.c2p(float(x_val), float(y_val)), color=EVIDENCE_ORANGE, radius=0.065)
            number = Text(str(i + 1), font_size=15, color=WHITE).move_to(dot)
            dots.add(VGroup(dot, number))
        path = VMobject(color=EVIDENCE_ORANGE, stroke_width=3)
        path.set_points_as_corners([dot[0].get_center() for dot in dots])
        return VGroup(VGroup(axes, path), title, dots)

    def summary(self) -> None:
        self.clear()
        narration = self.start_narration("scene08")
        label = self.section_label("3.5 The Evidence Approximation")
        title = self.scene_title("エビデンス近似の要点")

        items = VGroup(
            self.summary_item(r"\int p(t|w,\beta)p(w|\alpha)\,dw", "重み w は積分する", POSTERIOR_GREEN),
            self.summary_item(r"\max_{\alpha,\beta}\ p(t|\alpha,\beta)", "ハイパーパラメータは evidence 最大化", EVIDENCE_ORANGE),
            self.summary_item(r"\gamma=\sum_i{\lambda_i\over\alpha+\lambda_i}", "有効パラメータ数を数える", PRIOR_PURPLE),
            self.summary_item(r"{1\over\beta}={RSS\over N-\gamma}", "ノイズも自由度補正つきで推定", NOISE_TEAL),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).shift(LEFT * 0.35 + DOWN * 0.05)
        next_label = Text("次: 3.6 固定基底関数の限界", font_size=30, color=WHITE).to_edge(DOWN).shift(UP * 0.28)

        self.play(FadeIn(label), Write(title))
        self.play(FadeIn(items, lag_ratio=0.14), run_time=1.5)
        self.play(Write(next_label))
        self.finish_narration(narration)

    def summary_item(self, formula: str, text: str, color: ManimColor) -> VGroup:
        dot = Dot(color=color, radius=0.08)
        math = MathTex(formula, font_size=30, color=color)
        label = Text(text, font_size=24, color=WHITE)
        return VGroup(dot, math, label).arrange(RIGHT, buff=0.25)
