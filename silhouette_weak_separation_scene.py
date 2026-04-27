from manimlib import *
import sys
from pathlib import Path
from statistics import mean
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import (
    BG_COLOR,
    DAVID_COLOR,
    THOMAS_COLOR,
    OverlapScatterPlot,
    get_distance,
)
from layout_verifier import LayoutVerifier


class SilhouetteWeakSeparationScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="SilhouetteWeakSeparationScene")

        def latex_text(text, scale=0.42, color="#ccccdd"):
            mob = TexText(str(text), color=color)
            mob.scale(scale)
            return mob

        def latex_math(tex, scale=0.42, color="#ccccdd"):
            mob = Tex(tex, color=color)
            mob.scale(scale)
            return mob

        def make_measurement_bar(distance, color, max_distance):
            width = interpolate(0.10, 0.36, distance / max_distance)
            bar = Line(ORIGIN, RIGHT * width)
            bar.set_stroke(color, width=3.2, opacity=0.86)
            return bar

        def make_result_group(symbol, value, color):
            mean_label = latex_math(R"\text{mean} =", scale=0.34, color=WHITE)
            value_label = latex_math(Rf"{symbol} \approx {value:.2f}", scale=0.38, color=color)
            result = VGroup(mean_label, value_label)
            result.arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
            return result

        def make_distance_row(title_text_str, distances, color, symbol, value, max_distance):
            title = latex_text(title_text_str, scale=0.36, color=color)
            bars = VGroup(*[
                make_measurement_bar(distance, color, max_distance)
                for distance in distances
            ])
            bars.arrange(RIGHT, buff=0.04, aligned_edge=DOWN)
            result = make_result_group(symbol, value, color)
            row = VGroup(title, bars, result)
            row.arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
            return row, title, bars, result

        def make_anchor_lines(dot_getter, ids, color, opacity):
            lines = VGroup(*[
                Line(anchor.get_center(), dot_getter(i).get_center())
                for i in ids
            ])
            lines.set_stroke(color, width=1.18, opacity=opacity)
            return lines

        def make_score_number_line():
            number_line = NumberLine(
                x_range=[-1, 1, 1],
                width=5.6,
                color="#8b93a8",
                stroke_width=1.6,
                include_ticks=True,
                tick_size=0.05,
                longer_tick_multiple=1.8,
                big_tick_numbers=[-1, 0, 1],
                include_numbers=False,
            )
            number_line.set_stroke("#8b93a8", width=1.6, opacity=0.68)

            minus_label = latex_math("-1", scale=0.38, color="#8b93a8")
            zero_label = latex_math("0", scale=0.38, color="#8b93a8")
            plus_label = latex_math("+1", scale=0.38, color="#8b93a8")
            minus_label.next_to(number_line.n2p(-1), DOWN, buff=0.13)
            zero_label.next_to(number_line.n2p(0), DOWN, buff=0.13)
            plus_label.next_to(number_line.n2p(1), DOWN, buff=0.13)
            number_line.add(minus_label, zero_label, plus_label)
            return number_line

        def score_marker_position(number_line, value):
            return number_line.n2p(value) + UP * 0.13

        def make_score_marker(number_line, value):
            marker = Triangle(fill_color=YELLOW, fill_opacity=1.0, stroke_width=0)
            marker.scale(0.085)
            marker.rotate(PI)
            marker.move_to(score_marker_position(number_line, value))
            return marker

        def concentrate_clouds(scatter, david_scale=0.91, thomas_scale=0.73, thomas_shift=0.03):
            david_mean = scatter.david_data.mean(axis=0)
            thomas_mean = scatter.thomas_data.mean(axis=0)
            away = thomas_mean - david_mean
            norm = np.linalg.norm(away)
            if norm > 1e-5:
                away = away / norm

            scatter.david_data[:, :] = david_mean + (scatter.david_data - david_mean) * david_scale
            scatter.thomas_data[:, :] = (
                thomas_mean
                + (scatter.thomas_data - thomas_mean) * thomas_scale
                + away * thomas_shift
            )

            for i, (px, py) in enumerate(scatter.david_data):
                scatter.david_dots[i].move_to(scatter.axes.c2p(float(px), float(py)))
            for i, (px, py) in enumerate(scatter.thomas_data):
                scatter.thomas_dots[i].move_to(scatter.axes.c2p(float(px), float(py)))

        scatter = OverlapScatterPlot(n_david=10, n_thomas=10)
        concentrate_clouds(scatter)
        scatter.move_to(UP * 1.25)
        scatter.david_label = latex_text("David", scale=0.44, color=DAVID_COLOR)
        scatter.thomas_label = latex_text("Thomas", scale=0.44, color=THOMAS_COLOR)
        scatter.david_dots.set_fill(DAVID_COLOR, opacity=0.96)
        scatter.david_dots.set_stroke(WHITE, width=0.7, opacity=0.35)
        scatter.thomas_dots.set_fill(THOMAS_COLOR, opacity=0.96)
        scatter.thomas_dots.set_stroke(WHITE, width=0.7, opacity=0.35)
        scatter.david_label.move_to(
            scatter.frame_box.get_corner(UL) + RIGHT * 0.82 + DOWN * 0.28
        )
        scatter.thomas_label.move_to(
            scatter.frame_box.get_corner(UR) + LEFT * 1.0 + DOWN * 0.28
        )
        self.add(scatter, scatter.david_label, scatter.thomas_label)

        anchor_idx = 9
        anchor = scatter.get_david_dot(anchor_idx)

        self.play(anchor.animate.scale(1.9), run_time=0.6)

        david_positions = scatter.david_data
        thomas_positions = scatter.thomas_data
        anchor_xy = david_positions[anchor_idx]

        blue_dists = [
            (i, get_distance(anchor_xy, p))
            for i, p in enumerate(david_positions)
            if i != anchor_idx
        ]
        blue_dists.sort(key=lambda item: item[1])
        same_class = blue_dists
        same_class_ids = [i for i, _ in same_class]

        orange_dists = [(i, get_distance(anchor_xy, p)) for i, p in enumerate(thomas_positions)]
        orange_dists.sort(key=lambda item: item[1])
        other_class = orange_dists
        other_class_ids = [i for i, _ in other_class]
        a_value = mean(distance for _, distance in same_class)
        b_value = mean(distance for _, distance in other_class)
        score = (b_value - a_value) / max(a_value, b_value)
        displayed_distances = [distance for _, distance in same_class + other_class]
        max_display_distance = max(displayed_distances)

        blue_lines = make_anchor_lines(
            scatter.get_david_dot,
            same_class_ids,
            DAVID_COLOR,
            opacity=0.36,
        )
        orange_lines = make_anchor_lines(
            scatter.get_thomas_dot,
            other_class_ids,
            THOMAS_COLOR,
            opacity=0.30,
        )

        a_row, a_title, blue_bar_targets, a_result = make_distance_row(
            "all within-class distances",
            [distance for _, distance in same_class],
            DAVID_COLOR,
            "a",
            a_value,
            max_display_distance,
        )
        b_row, b_title, orange_bar_targets, b_result = make_distance_row(
            "all other-class distances",
            [distance for _, distance in other_class],
            THOMAS_COLOR,
            "b",
            b_value,
            max_display_distance,
        )
        summary_rows = VGroup(a_row, b_row)
        summary_rows.arrange(DOWN, buff=0.22)

        formula = latex_math(R"Silhouette Score = \frac{b - a}{\max(a, b)}", scale=0.48, color=WHITE)
        formula_row = VGroup(formula)
        formula_row.arrange(RIGHT, buff=0.22, aligned_edge=DOWN)

        number_line = make_score_number_line()
        marker = make_score_marker(number_line, score)

        score_tag = latex_math(Rf"s \approx {score:.2f}", scale=0.38, color=YELLOW)

        bottom_caption = latex_text(
            "Weak separation, not clean clustering",
            scale=0.40,
            color=WHITE,
        )

        formula_row.next_to(summary_rows, DOWN, buff=0.18)
        number_line.next_to(formula_row, DOWN, buff=0.50)
        bottom_caption.next_to(number_line, DOWN, buff=0.24)

        lower_block = VGroup(
            summary_rows,
            formula_row,
            number_line,
            marker,
            score_tag,
            bottom_caption,
        )
        available_top = scatter.frame_box.get_bottom()[1] - 0.14
        available_bottom = -FRAME_Y_RADIUS + 0.12
        available_height = available_top - available_bottom
        available_width = 2 * FRAME_X_RADIUS - 0.3

        if lower_block.get_height() > available_height:
            lower_block.scale(0.96 * available_height / lower_block.get_height())
        if lower_block.get_width() > available_width:
            lower_block.scale(0.98 * available_width / lower_block.get_width())

        lower_block.next_to(scatter.frame_box, DOWN, buff=0.14)
        center_x = scatter.frame_box.get_center()[0]
        for mob in (summary_rows, formula_row, number_line, bottom_caption):
            mob.set_x(center_x)
        marker.move_to(score_marker_position(number_line, score))
        score_tag.next_to(marker, UP, buff=0.1)

        if lower_block.get_bottom()[1] < available_bottom:
            lower_block.shift(UP * (available_bottom - lower_block.get_bottom()[1]))
        if lower_block.get_left()[0] < -FRAME_X_RADIUS + 0.1:
            lower_block.shift(RIGHT * (-FRAME_X_RADIUS + 0.1 - lower_block.get_left()[0]))
        if lower_block.get_right()[0] > FRAME_X_RADIUS - 0.1:
            lower_block.shift(LEFT * (lower_block.get_right()[0] - (FRAME_X_RADIUS - 0.1)))

        self.play(
            scatter.david_dots.animate.set_opacity(0.92),
            scatter.thomas_dots.animate.set_opacity(0.36),
            run_time=0.45,
        )
        self.play(LaggedStart(*[ShowCreation(line) for line in blue_lines], lag_ratio=0.035))
        self.play(FadeIn(a_title, shift=RIGHT * 0.12))
        self.play(
            LaggedStart(
                *[
                    TransformFromCopy(line, bar)
                    for line, bar in zip(blue_lines, blue_bar_targets)
                ],
                lag_ratio=0.08,
            ),
            FadeIn(a_result, shift=RIGHT * 0.08),
            run_time=0.85,
        )

        self.play(
            scatter.david_dots.animate.set_opacity(0.40),
            scatter.thomas_dots.animate.set_opacity(0.86),
            run_time=0.45,
        )
        self.play(LaggedStart(*[ShowCreation(line) for line in orange_lines], lag_ratio=0.025))
        self.play(FadeIn(b_title, shift=RIGHT * 0.12))
        self.play(
            LaggedStart(
                *[
                    TransformFromCopy(line, bar)
                    for line, bar in zip(orange_lines, orange_bar_targets)
                ],
                lag_ratio=0.08,
            ),
            FadeIn(b_result, shift=RIGHT * 0.08),
            run_time=0.85,
        )
        self.play(
            scatter.david_dots.animate.set_opacity(0.72),
            scatter.thomas_dots.animate.set_opacity(0.72),
            run_time=0.35,
        )
        self.play(Write(formula))

        verifier.check_vertical_order("scatter_frame", scatter.frame_box, "summary_rows", summary_rows, min_gap=0.12)
        verifier.check_vertical_order("scatter_frame", scatter.frame_box, "score_tag", score_tag, min_gap=0.08)
        verifier.check_vertical_order("summary_rows", summary_rows, "formula_row", formula_row, min_gap=0.12)
        verifier.check_vertical_order("formula_row", formula_row, "number_line", number_line, min_gap=0.24)
        verifier.check_vertical_order("score_tag", score_tag, "number_line", number_line, min_gap=0.08)
        verifier.check_no_overlap("formula_row", formula_row, "number_line", number_line, min_gap=0.24)
        verifier.check_no_overlap("summary_rows", summary_rows, "score_tag", score_tag, min_gap=0.08)
        verifier.check_no_overlap("formula_row", formula_row, "bottom_caption", bottom_caption, min_gap=0.16)
        verifier.check_no_overlap("number_line", number_line, "bottom_caption", bottom_caption, min_gap=0.10)
        verifier.check_inside_frame("summary_rows", summary_rows, margin=0.1)
        verifier.check_inside_frame("formula_row", formula_row, margin=0.1)
        verifier.check_inside_frame("number_line", number_line, margin=0.1)
        verifier.check_inside_frame("score_tag", score_tag, margin=0.1)
        verifier.check_inside_frame("bottom_caption", bottom_caption, margin=0.1)
        verifier.check_inside_frame("david_label", scatter.david_label, margin=0.1)
        verifier.check_inside_frame("thomas_label", scatter.thomas_label, margin=0.1)
        verifier.assert_ok()

        self.play(ShowCreation(number_line))
        self.play(GrowFromCenter(marker), FadeIn(score_tag, shift=UP * 0.06))
        self.play(Write(bottom_caption))
        self.wait(1.5)
