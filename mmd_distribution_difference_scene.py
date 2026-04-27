from manimlib import *
import numpy as np
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from layout_verifier import LayoutVerifier
from distribution_metrics_shared import (
    BG_COLOR,
    DAVID_COLOR,
    THOMAS_COLOR,
    LABEL_COLOR,
    FRAME_COLOR,
    OverlapScatterPlot,
)

ACCENT_COLOR = YELLOW_C
ACROSS_COLOR = "#78d6a3"
NUMBER_LINE_COLOR = "#c7ccdc"
KERNEL_SIGMA = 0.62
BASE_DOT_OPACITY = 0.18
DIM_DOT_OPACITY = 0.08
FULL_DOT_OPACITY = 0.96
TEXT_MARGIN = 0.1
BAR_WIDTH = 3.05
BAR_HEIGHT = 0.32
ORIGINAL_BAR_VALUES = (0.76, 0.72, 0.58)

# Number-line positions (0–1) used after the expression lands on the MMD line
MMD_MARKER_BASELINE = 0.12
MMD_MARKER_HIGH = 0.86

# Clip to match OverlapScatterPlot defaults
_MMD_XY_LO = np.array([-2.35, -1.55], dtype=float)
_MMD_XY_HI = np.array([2.35, 1.55], dtype=float)


def title_text(text, scale=0.68, color=WHITE):
    mob = TexText(text, color=color)
    mob.scale(scale)
    return mob


def caption_text(text, scale=0.42, color="#ccccdd"):
    mob = TexText(str(text), color=color)
    mob.scale(scale)
    return mob


def kernel_weight(anchor_point, point, sigma=KERNEL_SIGMA):
    offset = point - anchor_point
    return float(np.exp(-np.dot(offset, offset) / (2 * sigma ** 2)))


def kernel_weight_array(anchor_point, points, sigma=KERNEL_SIGMA):
    return np.array([kernel_weight(anchor_point, point, sigma=sigma) for point in points])


def weight_to_opacity(weight, low=BASE_DOT_OPACITY, high=FULL_DOT_OPACITY, power=0.72):
    return low + (high - low) * (weight ** power)


def opacity_anims(dots, opacities):
    return [dot.animate.set_opacity(float(opacity)) for dot, opacity in zip(dots, opacities)]


def uniform_opacity_anims(dots, opacity):
    return [dot.animate.set_opacity(float(opacity)) for dot in dots]


def clamp_to_frame(mob, margin=TEXT_MARGIN):
    if mob.get_left()[0] < -FRAME_X_RADIUS + margin:
        mob.shift(RIGHT * (-FRAME_X_RADIUS + margin - mob.get_left()[0]))
    if mob.get_right()[0] > FRAME_X_RADIUS - margin:
        mob.shift(LEFT * (mob.get_right()[0] - (FRAME_X_RADIUS - margin)))
    if mob.get_bottom()[1] < -FRAME_Y_RADIUS + margin:
        mob.shift(UP * (-FRAME_Y_RADIUS + margin - mob.get_bottom()[1]))
    if mob.get_top()[1] > FRAME_Y_RADIUS - margin:
        mob.shift(DOWN * (mob.get_top()[1] - (FRAME_Y_RADIUS - margin)))
    return mob


def place_lower_block(lower_block, scatter, buff=0.16):
    available_top = scatter.frame_box.get_bottom()[1] - buff
    available_bottom = -FRAME_Y_RADIUS + TEXT_MARGIN
    available_height = available_top - available_bottom
    available_width = 2 * FRAME_X_RADIUS - 2 * TEXT_MARGIN

    if lower_block.get_height() > available_height:
        lower_block.scale(0.96 * available_height / lower_block.get_height())
    if lower_block.get_width() > available_width:
        lower_block.scale(0.98 * available_width / lower_block.get_width())

    lower_block.next_to(scatter.frame_box, DOWN, buff=buff)
    lower_block.align_to(scatter.frame_box, LEFT)
    lower_block.shift(RIGHT * 0.2)
    clamp_to_frame(lower_block)

    if lower_block.get_bottom()[1] < available_bottom:
        lower_block.shift(UP * (available_bottom - lower_block.get_bottom()[1]))
    return clamp_to_frame(lower_block)


def place_metric_bars(bars, scatter):
    place_lower_block(bars, scatter)
    bars.shift(RIGHT * (0 - bars.get_center()[0]))
    return clamp_to_frame(bars)


def style_scatter_points(scatter):
    scatter.david_dots.set_fill(DAVID_COLOR, opacity=FULL_DOT_OPACITY)
    scatter.david_dots.set_stroke(WHITE, width=0.7, opacity=0.34)
    scatter.thomas_dots.set_fill(THOMAS_COLOR, opacity=FULL_DOT_OPACITY)
    scatter.thomas_dots.set_stroke(WHITE, width=0.7, opacity=0.34)


def place_cloud_labels(scatter):
    scatter.david_label.move_to(scatter.frame_box.get_corner(UL) + RIGHT * 0.58 + DOWN * 0.25)
    scatter.thomas_label.move_to(scatter.frame_box.get_corner(UR) + LEFT * 0.68 + DOWN * 0.25)


def use_latex_cloud_labels(scatter):
    scatter.david_label = caption_text("David", scale=0.44, color=DAVID_COLOR)
    scatter.thomas_label = caption_text("Thomas", scale=0.44, color=THOMAS_COLOR)


def soften_thomas_right_edge(scatter, max_x=1.45):
    right_edge_ids = np.where(scatter.thomas_data[:, 0] > max_x)[0]
    for idx in right_edge_ids:
        scatter.thomas_data[idx, 0] = max_x
        px, py = scatter.thomas_data[idx]
        scatter.thomas_dots[idx].move_to(scatter.axes.c2p(float(px), float(py)))


def make_influence_field(dot, color, max_radius=1.05):
    field = VGroup()
    for radius, opacity in [(1.00, 0.035), (0.76, 0.055), (0.52, 0.085), (0.28, 0.14)]:
        circle = Circle(radius=max_radius * radius)
        circle.set_fill(color, opacity=opacity)
        circle.set_stroke(color, width=1.0, opacity=opacity * 1.55)
        circle.move_to(dot.get_center())
        field.add(circle)
    return field


def make_metric_bar(label_text, fill_ratio, fill_color, max_width=BAR_WIDTH):
    label = caption_text(label_text, scale=0.42, color=LABEL_COLOR)

    track = RoundedRectangle(
        width=max_width,
        height=BAR_HEIGHT,
        corner_radius=0.08,
        stroke_color=FRAME_COLOR,
        stroke_width=1.0,
        fill_color=GREY_E,
        fill_opacity=0.12,
    )

    fill = RoundedRectangle(
        width=max(0.02, max_width * fill_ratio),
        height=BAR_HEIGHT,
        corner_radius=0.08,
        stroke_width=0,
        fill_color=fill_color,
        fill_opacity=0.9,
    )
    fill.move_to(track.get_left() + RIGHT * fill.get_width() / 2)

    meter = VGroup(track, fill)
    row = VGroup(label, meter)
    row.arrange(RIGHT, buff=0.34, aligned_edge=DOWN)
    row.label = label
    row.track = track
    row.fill = fill
    row.fill_color = fill_color
    return row


def make_metric_bars(values):
    labels = ("David", "Thomas", "across")
    colors = (DAVID_COLOR, THOMAS_COLOR, ACROSS_COLOR)
    bars = VGroup(*[make_metric_bar(label, value, color) for label, value, color in zip(labels, values, colors)])
    bars.arrange(DOWN, buff=0.24, aligned_edge=LEFT)
    return bars


def fill_target_for(bar, fill_ratio, fill_color=None):
    color = fill_color or bar.fill_color
    target = RoundedRectangle(
        width=max(0.02, BAR_WIDTH * fill_ratio),
        height=BAR_HEIGHT,
        corner_radius=0.08,
        stroke_width=0,
        fill_color=color,
        fill_opacity=0.9,
    )
    target.move_to(bar.track.get_left() + RIGHT * target.get_width() / 2)
    return target


def bar_fill_anims(bars, values):
    return [Transform(bar.fill, fill_target_for(bar, value)) for bar, value in zip(bars, values)]


def make_connection_lines(dots_a, data_a, dots_b, data_b, color, count=None, threshold=None, cross=False, opacity=0.54, width=None):
    candidates = []
    for i, point_a in enumerate(data_a):
        start_j = 0 if cross else i + 1
        for j in range(start_j, len(data_b)):
            dist = float(np.linalg.norm(point_a - data_b[j]))
            if threshold is None or dist <= threshold:
                candidates.append((dist, i, j))
    candidates.sort(key=lambda item: item[0])

    lines = VGroup()
    selected = candidates if count is None else candidates[:count]
    line_width = width if width is not None else (1.35 if not cross else 1.1)
    for _, i, j in selected:
        line = Line(dots_a[i].get_center(), dots_b[j].get_center())
        line.set_stroke(color, width=line_width, opacity=opacity)
        lines.add(line)
    return lines


def thomas_xy_approach_david(david_xy, thomas_xy, weight=0.9):
    """Interpolate each Thomas point toward the nearest David point (strong overlap)."""
    david_xy = np.asarray(david_xy, dtype=float)
    thomas_xy = np.asarray(thomas_xy, dtype=float)
    out = np.empty_like(thomas_xy)
    for i, p in enumerate(thomas_xy):
        d = np.linalg.norm(david_xy - p, axis=1)
        j = int(np.argmin(d))
        out[i] = (1.0 - weight) * p + weight * david_xy[j]
    return np.clip(out, _MMD_XY_LO, _MMD_XY_HI)


def thomas_xy_spread_from_david(david_xy, thomas_xy, spread=1.5, mean_shift=0.44):
    """Widen the Thomas cloud and nudge it away from David (visually 'more MMD')."""
    david_xy = np.asarray(david_xy, dtype=float)
    thomas_xy = np.asarray(thomas_xy, dtype=float)
    t_mean = thomas_xy.mean(axis=0)
    d_mean = david_xy.mean(axis=0)
    away = t_mean - d_mean
    n = float(np.linalg.norm(away))
    u = (away / n) if n > 1e-5 else np.array([0.55, 0.25], dtype=float)
    out = np.empty_like(thomas_xy)
    for i, p in enumerate(thomas_xy):
        v = p - t_mean
        out[i] = t_mean + v * spread + u * mean_shift
    return np.clip(out, _MMD_XY_LO, _MMD_XY_HI)


def thomas_dot_animate_moves(scatter, thomas_xy):
    thomas_xy = np.asarray(thomas_xy, dtype=float)
    thomas_xy = np.clip(thomas_xy, _MMD_XY_LO, _MMD_XY_HI)
    return [
        scatter.thomas_dots[i].animate.move_to(
            scatter.axes.c2p(float(thomas_xy[i, 0]), float(thomas_xy[i, 1]))
        )
        for i in range(len(scatter.thomas_dots))
    ]


def write_thomas_data_to_scatter(scatter, thomas_xy):
    thomas_xy = np.clip(np.asarray(thomas_xy, dtype=float), _MMD_XY_LO, _MMD_XY_HI)
    scatter.thomas_data[:, :] = thomas_xy


def make_expression_text(text, color=WHITE, font_size=42):
    return TexText(text, font_size=font_size, color=color)


def make_mmd_number_line():
    number_line = NumberLine(
        x_range=[0, 1, 0.25],
        width=5.35,
        color=NUMBER_LINE_COLOR,
        stroke_width=1.7,
        include_ticks=True,
        tick_size=0.045,
        longer_tick_multiple=2.0,
        big_tick_numbers=[0, 1],
        include_numbers=False,
    )
    number_line.set_stroke(NUMBER_LINE_COLOR, width=1.7, opacity=0.68)

    zero_label = caption_text("0", scale=0.42, color=NUMBER_LINE_COLOR)
    one_label = caption_text("1", scale=0.42, color=NUMBER_LINE_COLOR)
    zero_label.next_to(number_line.n2p(0), DOWN, buff=0.13)
    one_label.next_to(number_line.n2p(1), DOWN, buff=0.13)

    number_line.add(zero_label, one_label)
    return number_line


def mmd_marker_position(number_line, value):
    return number_line.n2p(value) + UP * 0.24


def make_mmd_marker(number_line, value):
    halo = Circle(radius=0.18, stroke_color=ACCENT_COLOR, stroke_width=1.2, stroke_opacity=0.34)
    halo.set_fill(ACCENT_COLOR, opacity=0.12)

    dot = Dot(radius=0.055, color=ACCENT_COLOR)
    dot.set_stroke(WHITE, width=0.8, opacity=0.72)

    stem = Line(ORIGIN, DOWN * 0.18)
    stem.set_stroke(ACCENT_COLOR, width=2.0, opacity=0.86)
    stem.next_to(dot, DOWN, buff=0.02)

    pointer = Triangle(fill_color=ACCENT_COLOR, fill_opacity=1.0, stroke_width=0)
    pointer.scale(0.075)
    pointer.rotate(PI)
    pointer.next_to(stem, DOWN, buff=-0.015)

    marker = VGroup(halo, dot, stem, pointer)
    marker.move_to(mmd_marker_position(number_line, value))
    return marker


class MMDDistributionDifferenceScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.camera.frame.set_width(14)

        self.stage1_introduce_distributions()
        self.stage2_local_similarity()
        self.stage3_within_group_similarity()
        self.stage4_across_group_similarity()
        self.stage5_aggregate_comparison()
        self.stage6_form_mmd_expression()
        self.stage7_counterfactual_identical()
        self.stage8_return_to_original_state()
        self.stage9_outro()

    def stage1_introduce_distributions(self):
        verifier = LayoutVerifier(scene_name="MMDDistributionDifferenceScene.stage1")

        question = caption_text("Are these two distributions measurably different?", scale=0.64, color=WHITE)
        question.to_edge(UP, buff=0.45)

        scatter = OverlapScatterPlot()
        scatter.move_to(UP * 0.42)
        style_scatter_points(scatter)
        soften_thomas_right_edge(scatter)
        use_latex_cloud_labels(scatter)
        place_cloud_labels(scatter)

        verifier.check_inside_frame("question", question, margin=TEXT_MARGIN)
        verifier.check_inside_frame("scatter_frame", scatter.frame_box, margin=TEXT_MARGIN)
        verifier.check_inside_frame("david_label", scatter.david_label, margin=TEXT_MARGIN)
        verifier.check_inside_frame("thomas_label", scatter.thomas_label, margin=TEXT_MARGIN)
        verifier.check_vertical_order("question", question, "scatter_frame", scatter.frame_box, min_gap=0.16)
        verifier.assert_ok()

        self.play(FadeIn(question, shift=UP * 0.08), run_time=0.75)
        self.play(FadeIn(scatter.axes), FadeIn(scatter.frame_box), run_time=0.55)
        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.55) for dot in scatter.david_dots], lag_ratio=0.035),
            FadeIn(scatter.david_label, shift=UP * 0.06),
            run_time=1.05,
        )
        self.play(scatter.david_dots.animate.scale(1.035).set_opacity(0.82), run_time=0.45)
        self.play(scatter.david_dots.animate.scale(1 / 1.035).set_opacity(FULL_DOT_OPACITY), run_time=0.45)
        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.55) for dot in scatter.thomas_dots], lag_ratio=0.025),
            FadeIn(scatter.thomas_label, shift=UP * 0.06),
            run_time=0.85,
        )
        self.wait(0.25)

        # Baseline Thomas for later MMD number-line sweeps (after soften / right-edge)
        self._thomas_mmd_baseline = scatter.thomas_data.copy()
        self._thomas_mmd_close = thomas_xy_approach_david(
            scatter.david_data, self._thomas_mmd_baseline
        )
        self._thomas_mmd_spread = thomas_xy_spread_from_david(
            scatter.david_data, self._thomas_mmd_baseline
        )

        self._question = question
        self._scatter = scatter

    def stage2_local_similarity(self):
        scatter = self._scatter
        samples = [
            ("david", 1, DAVID_COLOR),
            ("thomas", 8, THOMAS_COLOR),
            ("david", 10, ACROSS_COLOR),
        ]

        all_dots = VGroup(scatter.david_dots, scatter.thomas_dots)
        self.play(
            *uniform_opacity_anims(scatter.david_dots, 0.42),
            *uniform_opacity_anims(scatter.thomas_dots, 0.42),
            run_time=0.45,
        )

        for group, idx, color in samples:
            if group == "david":
                anchor_dot = scatter.david_dots[idx]
                anchor_xy = scatter.david_data[idx]
            else:
                anchor_dot = scatter.thomas_dots[idx]
                anchor_xy = scatter.thomas_data[idx]

            david_opacities = [weight_to_opacity(weight) for weight in kernel_weight_array(anchor_xy, scatter.david_data)]
            thomas_opacities = [weight_to_opacity(weight) for weight in kernel_weight_array(anchor_xy, scatter.thomas_data)]
            field = make_influence_field(anchor_dot, color)

            self.play(
                FadeIn(field, scale=0.9),
                anchor_dot.animate.scale(1.55),
                *opacity_anims(scatter.david_dots, david_opacities),
                *opacity_anims(scatter.thomas_dots, thomas_opacities),
                run_time=0.75,
            )
            self.play(
                FadeOut(field, scale=1.05),
                anchor_dot.animate.scale(1 / 1.55),
                all_dots.animate.set_opacity(0.46),
                run_time=0.55,
            )

        self.wait(0.2)

    def stage3_within_group_similarity(self):
        verifier = LayoutVerifier(scene_name="MMDDistributionDifferenceScene.stage3")
        scatter = self._scatter

        bar_group = make_metric_bars((0.05, 0.05, 0.05))
        place_metric_bars(bar_group, scatter)

        verifier.check_inside_frame("bar_group", bar_group, margin=TEXT_MARGIN)
        verifier.check_vertical_order("scatter_frame", scatter.frame_box, "bar_group", bar_group, min_gap=0.10)
        verifier.assert_ok()

        david_lines = make_connection_lines(
            scatter.david_dots,
            scatter.david_data,
            scatter.david_dots,
            scatter.david_data,
            DAVID_COLOR,
            opacity=0.24,
            width=0.78,
        )
        thomas_lines = make_connection_lines(
            scatter.thomas_dots,
            scatter.thomas_data,
            scatter.thomas_dots,
            scatter.thomas_data,
            THOMAS_COLOR,
            opacity=0.17,
            width=0.62,
        )

        self.play(FadeIn(bar_group, shift=UP * 0.08), run_time=0.45)
        self.play(
            *uniform_opacity_anims(scatter.david_dots, 0.88),
            *uniform_opacity_anims(scatter.thomas_dots, DIM_DOT_OPACITY),
            ShowCreation(david_lines),
            Transform(bar_group[0].fill, fill_target_for(bar_group[0], ORIGINAL_BAR_VALUES[0])),
            run_time=1.15,
        )
        self.play(FadeOut(david_lines), run_time=0.45)
        self.play(
            *uniform_opacity_anims(scatter.david_dots, DIM_DOT_OPACITY),
            *uniform_opacity_anims(scatter.thomas_dots, 0.88),
            ShowCreation(thomas_lines),
            Transform(bar_group[1].fill, fill_target_for(bar_group[1], ORIGINAL_BAR_VALUES[1])),
            run_time=1.15,
        )
        self.play(FadeOut(thomas_lines), run_time=0.45)
        self.wait(0.2)

        self._bar_group = bar_group
        self._bars = bar_group

    def stage4_across_group_similarity(self):
        scatter = self._scatter
        across_lines = make_connection_lines(
            scatter.david_dots,
            scatter.david_data,
            scatter.thomas_dots,
            scatter.thomas_data,
            ACROSS_COLOR,
            cross=True,
            opacity=0.13,
            width=0.52,
        )

        self.play(
            *uniform_opacity_anims(scatter.david_dots, 0.56),
            *uniform_opacity_anims(scatter.thomas_dots, 0.56),
            ShowCreation(across_lines),
            Transform(self._bars[2].fill, fill_target_for(self._bars[2], ORIGINAL_BAR_VALUES[2])),
            run_time=1.2,
        )
        self.play(FadeOut(across_lines), run_time=0.45)
        self.wait(0.25)

    def stage5_aggregate_comparison(self):
        scatter = self._scatter
        self.play(
            *uniform_opacity_anims(scatter.david_dots, 0.42),
            *uniform_opacity_anims(scatter.thomas_dots, 0.42),
            self._bar_group.animate.set_opacity(1.0),
            run_time=0.75,
        )
        self.wait(0.35)

    def stage6_form_mmd_expression(self):
        scalar = make_expression_text("MMD", color=WHITE, font_size=42)
        scalar.move_to(DOWN * 2)

        number_line = make_mmd_number_line()
        number_line.next_to(scalar, DOWN, buff=0.58)

        marker = make_mmd_marker(number_line, MMD_MARKER_BASELINE)

        landing_group = VGroup(scalar, number_line, marker)
        clamp_to_frame(landing_group)

        within_david = make_expression_text("Within", color=DAVID_COLOR, font_size=30)
        plus = make_expression_text("+", color=WHITE, font_size=32)
        within_thomas = make_expression_text("Within", color=THOMAS_COLOR, font_size=30)
        minus = make_expression_text("-", color=WHITE, font_size=32)
        two_x = make_expression_text("2 x", color=WHITE, font_size=30)
        across = make_expression_text("Across", color=ACROSS_COLOR, font_size=30)

        expression = VGroup(
            within_david,
            plus,
            within_thomas,
            minus,
            two_x,
            across,
        )
        expression.arrange(RIGHT, buff=0.1)
        expression.move_to(scalar.get_center())

        bar_meters = VGroup(*[bar[1] for bar in self._bars])
        self.play(
            FadeOut(bar_meters, shift=DOWN * 0.04),
            FadeTransform(self._bars[0].label, within_david),
            FadeIn(plus, shift=LEFT * 0.05),
            FadeTransform(self._bars[1].label, within_thomas),
            FadeIn(minus, shift=LEFT * 0.05),
            FadeIn(two_x, shift=LEFT * 0.05),
            FadeTransform(self._bars[2].label, across),
            run_time=0.95,
        )
        self.play(FadeTransform(expression, scalar), run_time=0.75)
        self.play(ShowCreation(number_line), GrowFromCenter(marker), run_time=0.75)
        self.wait(0.45)

        self._scalar = scalar
        self._number_line = number_line
        self._marker = marker

    def stage7_counterfactual_identical(self):
        scatter = self._scatter
        close_xy = self._thomas_mmd_close

        self.play(
            *uniform_opacity_anims(scatter.david_dots, 0.52),
            *uniform_opacity_anims(scatter.thomas_dots, 0.52),
            run_time=0.65,
        )
        self.play(
            *thomas_dot_animate_moves(scatter, close_xy),
            self._marker.animate.move_to(mmd_marker_position(self._number_line, 0.0)),
            run_time=1.05,
        )
        write_thomas_data_to_scatter(scatter, close_xy)
        self.wait(0.45)

    def stage8_return_to_original_state(self):
        scatter = self._scatter
        base_xy = self._thomas_mmd_baseline
        spread_xy = self._thomas_mmd_spread

        self.play(
            *uniform_opacity_anims(scatter.david_dots, 0.74),
            *uniform_opacity_anims(scatter.thomas_dots, 0.74),
            *thomas_dot_animate_moves(scatter, spread_xy),
            self._marker.animate.move_to(mmd_marker_position(self._number_line, MMD_MARKER_HIGH)),
            run_time=1.1,
        )
        write_thomas_data_to_scatter(scatter, spread_xy)
        self.play(
            *thomas_dot_animate_moves(scatter, base_xy),
            self._marker.animate.move_to(mmd_marker_position(self._number_line, MMD_MARKER_BASELINE)),
            run_time=1.05,
        )
        write_thomas_data_to_scatter(scatter, base_xy)
        self.wait(0.8)

    def stage9_outro(self):
        scatter = self._scatter
        visible_scene = VGroup(
            self._question,
            scatter.axes,
            scatter.frame_box,
            scatter.david_dots,
            scatter.thomas_dots,
            scatter.david_label,
            scatter.thomas_label,
            self._scalar,
            self._number_line,
            self._marker,
        )

        self.play(
            *uniform_opacity_anims(scatter.david_dots, 0.22),
            *uniform_opacity_anims(scatter.thomas_dots, 0.22),
            run_time=0.55,
        )
        self.wait(0.35)
        self.play(FadeOut(visible_scene), run_time=0.95)
