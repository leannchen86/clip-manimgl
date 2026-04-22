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
    caption_text,
    OverlapScatterPlot,
)
ACCENT_COLOR = YELLOW_C
SUBDUED_LABEL_COLOR = "#a4abc5"
NEUTRAL_LINE_COLOR = "#aeb4c7"
KERNEL_SIGMA = 0.6
BASE_DOT_OPACITY = 0.2
DIM_DOT_OPACITY = 0.08
FULL_DOT_OPACITY = 0.96
TEXT_MARGIN = 0.1


def title_text(text, scale=0.68, color=WHITE):
    mob = Text(text, font="Helvetica Neue", color=color)
    mob.scale(scale)
    return mob


def kernel_circle_around(dot, color, radius=0.38):
    ring = Circle(
        radius=radius,
        stroke_color=color,
        stroke_width=1.2,
        stroke_opacity=0.22,
    )
    ring.move_to(dot.get_center())
    return ring


def kernel_weight(anchor_point, point, sigma=KERNEL_SIGMA):
    offset = point - anchor_point
    return float(np.exp(-np.dot(offset, offset) / (2 * sigma ** 2)))


def kernel_weight_array(anchor_point, points, sigma=KERNEL_SIGMA):
    return np.array([kernel_weight(anchor_point, point, sigma=sigma) for point in points])


def weight_to_opacity(weight, low=BASE_DOT_OPACITY, high=FULL_DOT_OPACITY, power=0.7):
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


def place_explainer(caption, scatter, buff=0.18):
    max_width = scatter.frame_box.get_width() - 0.45
    if caption.get_width() > max_width:
        caption.scale(0.98 * max_width / caption.get_width())
    caption.next_to(scatter.frame_box, DOWN, buff=buff)
    caption.align_to(scatter.frame_box, LEFT)
    caption.shift(RIGHT * 0.24)
    return clamp_to_frame(caption)


def place_term_labels(term_labels, scatter, buff=0.34):
    term_labels.next_to(scatter.frame_box, RIGHT, buff=buff)
    term_labels.shift(DOWN * 0.18)
    return clamp_to_frame(term_labels)


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
    lower_block.shift(RIGHT * 0.24)
    clamp_to_frame(lower_block)

    if lower_block.get_bottom()[1] < available_bottom:
        lower_block.shift(UP * (available_bottom - lower_block.get_bottom()[1]))
    return clamp_to_frame(lower_block)


def style_scatter_points(scatter):
    scatter.david_dots.set_fill(DAVID_COLOR, opacity=0.96)
    scatter.david_dots.set_stroke(WHITE, width=0.7, opacity=0.35)
    scatter.thomas_dots.set_fill(THOMAS_COLOR, opacity=0.96)
    scatter.thomas_dots.set_stroke(WHITE, width=0.7, opacity=0.35)


def make_summary_bar(label_text, fill_ratio, fill_color, max_width=2.9):
    label = caption_text(label_text, scale=0.46, color=LABEL_COLOR)

    track = RoundedRectangle(
        width=max_width,
        height=0.34,
        corner_radius=0.08,
        stroke_color=FRAME_COLOR,
        stroke_width=1.0,
        fill_color=GREY_E,
        fill_opacity=0.12,
    )

    fill = RoundedRectangle(
        width=max_width * fill_ratio,
        height=0.34,
        corner_radius=0.08,
        stroke_width=0,
        fill_color=fill_color,
        fill_opacity=0.9,
    )
    fill.align_to(track, LEFT)
    fill.move_to(track.get_left() + RIGHT * fill.get_width() / 2)

    row = VGroup(label, VGroup(track, fill))
    row.arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
    return row


class MMDDistributionDifferenceScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.camera.frame.set_width(14)

        self.stage1_intro_scatter()
        self.stage2_kernel_neighborhoods()
        self.stage3_kernel_terms()
        self.stage4_summary_bars()
        self.stage5_scalar_landing()

    def stage1_intro_scatter(self):
        verifier = LayoutVerifier(scene_name="MMDDistributionDifferenceScene.stage1")

        title = title_text("MMD: overlapping distributions", scale=0.62)
        title.to_edge(UP, buff=0.2)

        question = caption_text(
            "Are these two distributions measurably different?",
            scale=0.5,
            color=WHITE,
        )
        question.next_to(title, DOWN, buff=0.12)

        scatter = OverlapScatterPlot()
        scatter.move_to(UP * 0.35)
        style_scatter_points(scatter)

        verifier.check_inside_frame("title", title, margin=TEXT_MARGIN)
        verifier.check_inside_frame("question", question, margin=TEXT_MARGIN)
        verifier.check_inside_frame("scatter_frame", scatter.frame_box, margin=TEXT_MARGIN)
        verifier.check_inside_frame("david_label", scatter.david_label, margin=TEXT_MARGIN)
        verifier.check_inside_frame("thomas_label", scatter.thomas_label, margin=TEXT_MARGIN)
        verifier.check_no_overlap("title", title, "question", question, min_gap=0.08)
        verifier.check_vertical_order("question", question, "scatter_frame", scatter.frame_box, min_gap=0.12)
        verifier.check_no_overlap("question", question, "david_label", scatter.david_label, min_gap=0.05)
        verifier.check_no_overlap("question", question, "thomas_label", scatter.thomas_label, min_gap=0.05)
        verifier.assert_ok()

        self.play(FadeIn(title, shift=UP * 0.12), FadeIn(question, shift=UP * 0.08), run_time=0.8)
        self.play(FadeIn(scatter.axes), FadeIn(scatter.frame_box), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.55) for d in scatter.david_dots], lag_ratio=0.035),
            FadeIn(scatter.david_label, shift=UP * 0.08),
            run_time=1.1,
        )
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.55) for d in scatter.thomas_dots], lag_ratio=0.035),
            FadeIn(scatter.thomas_label, shift=UP * 0.08),
            run_time=1.1,
        )
        self.wait(0.3)

        self._scatter = scatter

    def stage2_kernel_neighborhoods(self):
        verifier = LayoutVerifier(scene_name="MMDDistributionDifferenceScene.stage2")
        scatter = self._scatter
        david_anchor_id = 0
        thomas_anchor_id = 21
        david_anchor = scatter.get_david_dot(david_anchor_id)
        thomas_anchor = scatter.get_thomas_dot(thomas_anchor_id)
        david_probe = kernel_circle_around(david_anchor, DAVID_COLOR, radius=0.56)
        thomas_probe = kernel_circle_around(thomas_anchor, THOMAS_COLOR, radius=0.56)

        self.play(
            david_anchor.animate.scale(1.7),
            thomas_anchor.animate.scale(1.7),
            run_time=0.75,
        )

        kernel_caption = caption_text(
            "Kernel similarity comes from\nlocal neighborhoods",
            scale=0.42,
            color=SUBDUED_LABEL_COLOR,
        )
        place_explainer(kernel_caption, scatter)

        verifier.check_inside_frame("kernel_caption", kernel_caption, margin=TEXT_MARGIN)
        verifier.check_vertical_order("scatter_frame", scatter.frame_box, "kernel_caption", kernel_caption, min_gap=0.12)
        verifier.check_no_overlap("scatter_frame", scatter.frame_box, "kernel_caption", kernel_caption, min_gap=0.10)
        verifier.assert_ok()

        self.play(
            ShowCreation(david_probe),
            ShowCreation(thomas_probe),
            FadeIn(kernel_caption, shift=UP * 0.08),
            run_time=0.95,
        )
        self.wait(0.25)

        self._david_anchor_id = david_anchor_id
        self._thomas_anchor_id = thomas_anchor_id
        self._david_probe = david_probe
        self._thomas_probe = thomas_probe
        self._kernel_caption = kernel_caption

    def stage3_kernel_terms(self):
        verifier = LayoutVerifier(scene_name="MMDDistributionDifferenceScene.stage3")
        scatter = self._scatter
        david_anchor_idx = self._david_anchor_id
        thomas_anchor_idx = self._thomas_anchor_id
        david_anchor_xy = scatter.david_data[david_anchor_idx]
        thomas_anchor_xy = scatter.thomas_data[thomas_anchor_idx]

        david_weights = kernel_weight_array(david_anchor_xy, scatter.david_data)
        thomas_weights = kernel_weight_array(thomas_anchor_xy, scatter.thomas_data)
        across_weights = kernel_weight_array(david_anchor_xy, scatter.thomas_data)

        david_opacities = [weight_to_opacity(weight) for weight in david_weights]
        thomas_opacities = [weight_to_opacity(weight) for weight in thomas_weights]
        across_target_opacities = [weight_to_opacity(weight, low=0.18, high=FULL_DOT_OPACITY) for weight in across_weights]
        across_source_opacities = np.full(len(scatter.david_dots), DIM_DOT_OPACITY)
        across_source_opacities[david_anchor_idx] = FULL_DOT_OPACITY

        within_david_label = caption_text("within David", scale=0.42, color=DAVID_COLOR)
        within_thomas_label = caption_text("within Thomas", scale=0.42, color=THOMAS_COLOR)
        across_label = caption_text("across groups", scale=0.42, color=NEUTRAL_LINE_COLOR)
        term_labels = VGroup(within_david_label, within_thomas_label, across_label)
        term_labels.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        place_term_labels(term_labels, scatter)
        term_labels.set_opacity(0.28)

        within_david_caption = caption_text(
            "Within David:\nnearby blue points get larger kernel weight",
            scale=0.39,
            color=SUBDUED_LABEL_COLOR,
        )
        place_explainer(within_david_caption, scatter)

        within_thomas_caption = caption_text(
            "Within Thomas:\ndo the same for orange neighborhoods",
            scale=0.39,
            color=SUBDUED_LABEL_COLOR,
        )
        place_explainer(within_thomas_caption, scatter)

        across_caption = caption_text(
            "Across groups:\ncompare orange points to a blue anchor",
            scale=0.39,
            color=SUBDUED_LABEL_COLOR,
        )
        place_explainer(across_caption, scatter)

        averaging_caption = caption_text(
            "Repeat over many anchors,\nthen average the three kernel terms",
            scale=0.39,
            color=SUBDUED_LABEL_COLOR,
        )
        place_explainer(averaging_caption, scatter)

        verifier.check_inside_frame("term_labels", term_labels, margin=TEXT_MARGIN)
        verifier.check_min_horizontal_gap("scatter_frame", scatter.frame_box, "term_labels", term_labels, min_gap=0.10)
        verifier.check_no_overlap("scatter_frame", scatter.frame_box, "term_labels", term_labels, min_gap=0.08)
        for name, caption in [
            ("within_david_caption", within_david_caption),
            ("within_thomas_caption", within_thomas_caption),
            ("across_caption", across_caption),
            ("averaging_caption", averaging_caption),
        ]:
            verifier.check_inside_frame(name, caption, margin=TEXT_MARGIN)
            verifier.check_vertical_order("scatter_frame", scatter.frame_box, name, caption, min_gap=0.10)
            verifier.check_no_overlap("term_labels", term_labels, name, caption, min_gap=0.08)
        verifier.assert_ok()

        self.play(FadeIn(term_labels, shift=LEFT * 0.08), run_time=0.45)
        self.play(
            within_david_label.animate.set_opacity(1.0),
            *opacity_anims(scatter.david_dots, david_opacities),
            *uniform_opacity_anims(scatter.thomas_dots, DIM_DOT_OPACITY),
            self._david_probe.animate.set_stroke(DAVID_COLOR, width=1.35, opacity=0.28),
            self._thomas_probe.animate.set_stroke(THOMAS_COLOR, width=1.0, opacity=0.08),
            FadeOut(self._kernel_caption, shift=UP * 0.04),
            FadeIn(within_david_caption, shift=UP * 0.04),
            run_time=1.0,
        )
        self._kernel_caption = within_david_caption

        self.play(
            within_david_label.animate.set_opacity(0.28),
            within_thomas_label.animate.set_opacity(1.0),
            *uniform_opacity_anims(scatter.david_dots, DIM_DOT_OPACITY),
            *opacity_anims(scatter.thomas_dots, thomas_opacities),
            self._david_probe.animate.set_stroke(DAVID_COLOR, width=1.0, opacity=0.08),
            self._thomas_probe.animate.set_stroke(THOMAS_COLOR, width=1.35, opacity=0.28),
            FadeOut(self._kernel_caption, shift=UP * 0.04),
            FadeIn(within_thomas_caption, shift=UP * 0.04),
            run_time=1.0,
        )
        self._kernel_caption = within_thomas_caption

        self.play(
            within_thomas_label.animate.set_opacity(0.28),
            across_label.animate.set_opacity(1.0),
            *opacity_anims(scatter.david_dots, across_source_opacities),
            *opacity_anims(scatter.thomas_dots, across_target_opacities),
            self._david_probe.animate.set_stroke(NEUTRAL_LINE_COLOR, width=1.35, opacity=0.3),
            self._thomas_probe.animate.set_stroke(THOMAS_COLOR, width=1.0, opacity=0.06),
            FadeOut(self._kernel_caption, shift=UP * 0.04),
            FadeIn(across_caption, shift=UP * 0.04),
            run_time=1.0,
        )
        self._kernel_caption = across_caption

        self.play(
            term_labels.animate.set_opacity(0.72),
            *uniform_opacity_anims(scatter.david_dots, 0.58),
            *uniform_opacity_anims(scatter.thomas_dots, 0.58),
            self._david_probe.animate.set_stroke(DAVID_COLOR, width=1.2, opacity=0.14),
            self._thomas_probe.animate.set_stroke(THOMAS_COLOR, width=1.2, opacity=0.14),
            FadeOut(self._kernel_caption, shift=UP * 0.04),
            FadeIn(averaging_caption, shift=UP * 0.04),
            run_time=0.9,
        )
        self._kernel_caption = averaging_caption
        self.wait(0.35)

        self._metric_label = term_labels

    def stage4_summary_bars(self):
        verifier = LayoutVerifier(scene_name="MMDDistributionDifferenceScene.stage4")
        summary_title = caption_text(
            "Average those three kernel terms",
            scale=0.4,
            color=SUBDUED_LABEL_COLOR,
        )

        david_bar = make_summary_bar("within David", 0.74, DAVID_COLOR)
        thomas_bar = make_summary_bar("within Thomas", 0.69, THOMAS_COLOR)
        across_bar = make_summary_bar("across names", 0.56, GREY_B)

        bar_group = VGroup(david_bar, thomas_bar, across_bar)
        bar_group.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        summary_title.next_to(bar_group, UP, buff=0.18)
        summary_title.align_to(bar_group, LEFT)

        lower_block = VGroup(summary_title, bar_group)
        place_lower_block(lower_block, self._scatter)

        verifier.check_vertical_order("scatter_frame", self._scatter.frame_box, "summary_title", summary_title, min_gap=0.10)
        verifier.check_vertical_order("summary_title", summary_title, "bar_group", bar_group, min_gap=0.12)
        verifier.check_inside_frame("summary_title", summary_title, margin=TEXT_MARGIN)
        verifier.check_inside_frame("bar_group", bar_group, margin=TEXT_MARGIN)
        verifier.assert_ok()

        self.play(
            FadeIn(bar_group, shift=UP * 0.12),
            FadeOut(self._metric_label, shift=RIGHT * 0.08),
            FadeOut(self._kernel_caption, shift=DOWN * 0.08),
            FadeOut(self._david_probe, scale=0.96),
            FadeOut(self._thomas_probe, scale=0.96),
            *uniform_opacity_anims(self._scatter.david_dots, 0.34),
            *uniform_opacity_anims(self._scatter.thomas_dots, 0.34),
            run_time=0.95,
        )
        self.play(FadeIn(summary_title, shift=UP * 0.08), run_time=0.45)
        self.wait(0.25)

        self._bar_group = bar_group
        self._summary_title = summary_title

    def stage5_scalar_landing(self):
        verifier = LayoutVerifier(scene_name="MMDDistributionDifferenceScene.stage5")
        scalar = Text("MMD > 0", font="Helvetica Neue", font_size=42, color=WHITE)
        scalar.move_to(self._bar_group.get_center()).shift(RIGHT * 0.45)

        number_line = NumberLine(
            x_range=[0, 1, 0.25],
            width=5.5,
            include_ticks=True,
            include_numbers=True,
            big_tick_numbers=[0, 1],
            decimal_number_config={"num_decimal_places": 2},
        )
        number_line.next_to(scalar, DOWN, buff=0.7)

        marker_value = 0.12
        marker = Triangle(fill_color=ACCENT_COLOR, fill_opacity=1, stroke_width=0)
        marker.scale(0.12)
        marker.rotate(PI)
        marker.next_to(number_line.n2p(marker_value), UP, buff=0.06)

        p_value = caption_text("p < 0.05", scale=0.4, color=ACCENT_COLOR)
        p_value.next_to(marker, UP, buff=0.15)

        bottom_caption = caption_text(
            "Small but measurable distribution difference.",
            scale=0.44,
            color=WHITE,
        )
        bottom_caption.next_to(number_line, DOWN, buff=0.26)

        landing_group = VGroup(scalar, number_line, marker, p_value, bottom_caption)
        clamp_to_frame(landing_group)

        verifier.check_inside_frame("scalar", scalar, margin=TEXT_MARGIN)
        verifier.check_inside_frame("number_line", number_line, margin=TEXT_MARGIN)
        verifier.check_inside_frame("p_value", p_value, margin=TEXT_MARGIN)
        verifier.check_inside_frame("bottom_caption", bottom_caption, margin=TEXT_MARGIN)
        verifier.check_vertical_order("scalar", scalar, "number_line", number_line, min_gap=0.14)
        verifier.check_vertical_order("number_line", number_line, "bottom_caption", bottom_caption, min_gap=0.10)
        verifier.check_no_overlap("scalar", scalar, "p_value", p_value, min_gap=0.08)
        verifier.check_no_overlap("number_line", number_line, "bottom_caption", bottom_caption, min_gap=0.10)
        verifier.assert_ok()

        self.play(
            FadeOut(self._summary_title, shift=DOWN * 0.08),
            FadeTransform(self._bar_group, scalar),
            run_time=0.95,
        )
        self.play(ShowCreation(number_line), run_time=0.6)
        self.play(GrowFromCenter(marker), Flash(marker, color=ACCENT_COLOR, flash_radius=0.35), run_time=0.7)
        self.play(FadeIn(p_value, shift=UP * 0.08), FadeIn(bottom_caption, shift=UP * 0.08), run_time=0.6)
        self.wait(1.6)
