from manimlib import *
import sys
from pathlib import Path
from statistics import mean

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import (
    BG_COLOR,
    DAVID_COLOR,
    THOMAS_COLOR,
    caption_text,
    OverlapScatterPlot,
    get_distance,
)
from layout_verifier import LayoutVerifier


class SilhouetteWeakSeparationScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="SilhouetteWeakSeparationScene")

        def make_measurement_bar(distance, color, max_distance):
            width = interpolate(0.34, 0.88, distance / max_distance)
            bar = Line(ORIGIN, RIGHT * width)
            bar.set_stroke(color, width=5, opacity=0.92)
            return bar

        def make_result_group(symbol, value, color):
            mean_label = caption_text("mean =", scale=0.34, color=WHITE)
            value_label = caption_text(f"{symbol} ~ {value:.2f}", scale=0.38, color=color)
            result = VGroup(mean_label, value_label)
            result.arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
            return result

        def make_distance_row(title_text_str, distances, color, symbol, value, max_distance):
            title = caption_text(title_text_str, scale=0.36, color=color)
            bars = VGroup(*[
                make_measurement_bar(distance, color, max_distance)
                for distance in distances
            ])
            bars.arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
            result = make_result_group(symbol, value, color)
            row = VGroup(title, bars, result)
            row.arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
            return row, title, bars, result

        scatter = OverlapScatterPlot()
        scatter.move_to(UP * 1.25)
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

        anchor_idx = 5
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
        near_blue = blue_dists[:4]
        near_blue_ids = [i for i, _ in near_blue]

        orange_dists = [(i, get_distance(anchor_xy, p)) for i, p in enumerate(thomas_positions)]
        orange_dists.sort(key=lambda item: item[1])
        near_orange = orange_dists[:4]
        near_orange_ids = [i for i, _ in near_orange]
        a_value = mean(distance for _, distance in near_blue)
        b_value = mean(distance for _, distance in near_orange)
        score = (b_value - a_value) / max(a_value, b_value)
        displayed_distances = [distance for _, distance in near_blue + near_orange]
        max_display_distance = max(displayed_distances)

        blue_lines = VGroup(*[
            Line(
                anchor.get_center(),
                scatter.get_david_dot(i).get_center(),
                stroke_color=DAVID_COLOR,
                stroke_width=2.5,
                stroke_opacity=0.6,
            )
            for i in near_blue_ids
        ])

        orange_lines = VGroup(*[
            Line(
                anchor.get_center(),
                scatter.get_thomas_dot(i).get_center(),
                stroke_color=THOMAS_COLOR,
                stroke_width=2.5,
                stroke_opacity=0.5,
            )
            for i in near_orange_ids
        ])

        a_row, a_title, blue_bar_targets, a_result = make_distance_row(
            "4 same-name distances",
            [distance for _, distance in near_blue],
            DAVID_COLOR,
            "a",
            a_value,
            max_display_distance,
        )
        b_row, b_title, orange_bar_targets, b_result = make_distance_row(
            "4 other-name distances",
            [distance for _, distance in near_orange],
            THOMAS_COLOR,
            "b",
            b_value,
            max_display_distance,
        )
        summary_rows = VGroup(a_row, b_row)
        summary_rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)

        b_symbol = caption_text("b", scale=0.40, color=THOMAS_COLOR)
        comparison_text = caption_text("only slightly larger than", scale=0.34, color=WHITE)
        a_symbol = caption_text("a", scale=0.40, color=DAVID_COLOR)
        comparison = VGroup(b_symbol, comparison_text, a_symbol)
        comparison.arrange(RIGHT, buff=0.08, aligned_edge=DOWN)

        formula = caption_text("s = (b - a) / max(a, b)", scale=0.40, color=WHITE)

        number_line = NumberLine(
            x_range=[-1, 1, 1],
            width=5.8,
            include_ticks=True,
            include_numbers=True,
        )
        marker = Triangle(fill_color=YELLOW, fill_opacity=1, stroke_width=0)
        marker.scale(0.12)
        marker.rotate(PI)
        marker.next_to(number_line.n2p(score), UP, buff=0.06)

        score_tag = caption_text(f"s ~ {score:.2f}", scale=0.38, color=YELLOW)
        score_tag.next_to(marker, UP, buff=0.08)

        bottom_caption = caption_text(
            "Weak separation, not clean clustering",
            scale=0.40,
            color=WHITE,
        )

        comparison.next_to(summary_rows, DOWN, buff=0.16)
        comparison.align_to(summary_rows, LEFT)
        formula.next_to(comparison, DOWN, buff=0.18)
        formula.align_to(summary_rows, LEFT)
        number_line.next_to(formula, DOWN, buff=0.32)
        number_line.align_to(summary_rows, LEFT)
        marker.next_to(number_line.n2p(score), UP, buff=0.06)
        score_tag.next_to(marker, UP, buff=0.08)
        bottom_caption.next_to(number_line, DOWN, buff=0.24)
        bottom_caption.align_to(summary_rows, LEFT)

        lower_block = VGroup(
            summary_rows,
            comparison,
            formula,
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
        lower_block.align_to(scatter.frame_box, LEFT)
        lower_block.shift(RIGHT * 0.24)

        if lower_block.get_bottom()[1] < available_bottom:
            lower_block.shift(UP * (available_bottom - lower_block.get_bottom()[1]))
        if lower_block.get_left()[0] < -FRAME_X_RADIUS + 0.1:
            lower_block.shift(RIGHT * (-FRAME_X_RADIUS + 0.1 - lower_block.get_left()[0]))
        if lower_block.get_right()[0] > FRAME_X_RADIUS - 0.1:
            lower_block.shift(LEFT * (lower_block.get_right()[0] - (FRAME_X_RADIUS - 0.1)))

        self.play(LaggedStart(*[ShowCreation(line) for line in blue_lines], lag_ratio=0.12))
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

        self.play(LaggedStart(*[ShowCreation(line) for line in orange_lines], lag_ratio=0.12))
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
        self.play(FadeIn(comparison, shift=RIGHT * 0.08))
        self.play(Write(formula))

        verifier.check_vertical_order("scatter_frame", scatter.frame_box, "summary_rows", summary_rows, min_gap=0.12)
        verifier.check_vertical_order("scatter_frame", scatter.frame_box, "score_tag", score_tag, min_gap=0.08)
        verifier.check_vertical_order("summary_rows", summary_rows, "comparison", comparison, min_gap=0.10)
        verifier.check_vertical_order("comparison", comparison, "formula", formula, min_gap=0.12)
        verifier.check_vertical_order("formula", formula, "number_line", number_line, min_gap=0.16)
        verifier.check_vertical_order("score_tag", score_tag, "number_line", number_line, min_gap=0.08)
        verifier.check_no_overlap("formula", formula, "number_line", number_line, min_gap=0.16)
        verifier.check_no_overlap("summary_rows", summary_rows, "score_tag", score_tag, min_gap=0.08)
        verifier.check_no_overlap("formula", formula, "bottom_caption", bottom_caption, min_gap=0.16)
        verifier.check_no_overlap("number_line", number_line, "bottom_caption", bottom_caption, min_gap=0.10)
        verifier.check_inside_frame("summary_rows", summary_rows, margin=0.1)
        verifier.check_inside_frame("comparison", comparison, margin=0.1)
        verifier.check_inside_frame("formula", formula, margin=0.1)
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
