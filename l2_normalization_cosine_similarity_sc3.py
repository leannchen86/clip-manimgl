from manimlib import *
import numpy as np
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from layout_verifier import LayoutVerifier


BG_COLOR = "#0f1117"
GRID_COLOR = "#2d425c"
RAW_COLOR = BLUE_D
NORM_COLOR = GREEN_C
ANCHOR_COLOR = GOLD_A
MOVING_COLOR = TEAL_C
OTHER_POINT_COLOR = BLUE_B
CAPTION_COLOR = "#d7d9e5"


def title_text(text, scale=0.7, color=WHITE):
    t = Text(text, color=color)
    t.scale(scale)
    return t


def caption_text(text, scale=0.42, color=CAPTION_COLOR):
    t = Text(text, color=color)
    t.scale(scale)
    return t


def spherical_to_cartesian(theta, phi):
    return np.array([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ])


class L2NormalizationCosineSimilarity(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="L2NormalizationCosineSimilarity")

        frame = self.camera.frame
        frame.reorient(0, 0)

        # ------------------------------------------------------------------
        # Stage 1 — 2D normalization on the unit circle
        # ------------------------------------------------------------------
        title = title_text("L2 Normalization and Cosine Similarity", scale=0.68)
        title.to_edge(UP, buff=0.24)
        title.fix_in_frame()

        explainer = caption_text(
            "After normalization, direction matters more than raw magnitude.",
            scale=0.42,
        )
        explainer.to_edge(DOWN, buff=0.22)
        explainer.fix_in_frame()

        norm_formula = Text("normalized vector = v / ||v||", font_size=27, color=WHITE)
        norm_formula.next_to(title, DOWN, buff=0.18)
        norm_formula.fix_in_frame()

        plane = NumberPlane(
            x_range=(-3, 3, 1),
            y_range=(-3, 3, 1),
            height=6.0,
            width=6.0,
            axis_config={
                "stroke_color": GREY_B,
                "stroke_width": 2,
            },
            background_line_style={
                "stroke_color": GRID_COLOR,
                "stroke_width": 1,
                "stroke_opacity": 0.28,
            },
        )
        unit_circle = Circle(radius=1.0, color=TEAL_A, stroke_width=3)
        unit_circle.set_fill(opacity=0)
        circle_label = caption_text(
            "unit circle: every point here has length 1",
            scale=0.38,
            color=TEAL_A,
        )
        circle_label.next_to(unit_circle, DOWN, buff=0.25)
        circle_label.shift(LEFT * 0.1)

        raw_vec = np.array([2.75, 1.45, 0.0])
        norm_vec = raw_vec / np.linalg.norm(raw_vec)

        raw_arrow = Arrow(ORIGIN, raw_vec, buff=0.0, stroke_width=5, color=RAW_COLOR)
        raw_dot = GlowDot(raw_vec, color=RAW_COLOR, radius=0.12)
        raw_label = caption_text(
            "before normalization:\nmagnitude can exceed 1",
            scale=0.34,
            color=RAW_COLOR,
        )
        raw_label.next_to(raw_arrow.get_end(), UR, buff=0.16)
        raw_label.shift(RIGHT * 0.05)

        norm_arrow_target = Arrow(ORIGIN, norm_vec, buff=0.0, stroke_width=5, color=NORM_COLOR)
        norm_dot_target = GlowDot(norm_vec, color=NORM_COLOR, radius=0.12)
        norm_label_target = caption_text(
            "after normalization:\nlength = 1, same direction",
            scale=0.34,
            color=NORM_COLOR,
        )
        norm_label_target.next_to(norm_arrow_target.get_end(), UR, buff=0.16)
        norm_label_target.shift(RIGHT * 0.1)

        stage_label = caption_text(
            "normalization moves the tip onto the unit circle while preserving direction",
            scale=0.38,
        )
        stage_label.next_to(explainer, UP, buff=0.18)
        stage_label.fix_in_frame()

        verifier.check_inside_frame("title", title, margin=0.08)
        verifier.check_inside_frame("norm_formula", norm_formula, margin=0.08)
        verifier.check_inside_frame("explainer", explainer, margin=0.08)
        verifier.check_inside_frame("stage_label", stage_label, margin=0.08)
        verifier.check_inside_frame("circle_label", circle_label, margin=0.08)
        verifier.check_no_overlap("title", title, "norm_formula", norm_formula, min_gap=0.08)
        verifier.check_no_overlap("explainer", explainer, "stage_label", stage_label, min_gap=0.08)
        verifier.assert_ok()

        self.play(ShowCreation(plane), ShowCreation(unit_circle), FadeIn(circle_label, shift=UP * 0.08), run_time=1.1)
        self.play(FadeIn(title, shift=DOWN * 0.1), FadeIn(norm_formula, shift=DOWN * 0.1), run_time=0.8)
        self.play(
            GrowArrow(raw_arrow),
            FadeIn(raw_dot, scale=0.8),
            FadeIn(raw_label, shift=UP * 0.08),
            run_time=0.9,
        )
        self.play(FadeIn(stage_label, shift=UP * 0.08), run_time=0.5)
        self.wait(0.3)
        self.play(
            Transform(raw_arrow, norm_arrow_target),
            Transform(raw_dot, norm_dot_target),
            Transform(raw_label, norm_label_target),
            run_time=1.6,
        )
        self.play(FadeIn(explainer, shift=UP * 0.08), run_time=0.6)
        self.wait(0.5)

        # ------------------------------------------------------------------
        # Stage 2 — 3D sphere, points on the surface, angle-based similarity
        # ------------------------------------------------------------------
        group_2d = Group(plane, unit_circle, raw_arrow, raw_dot, raw_label, circle_label)
        self.play(FadeOut(group_2d), FadeOut(stage_label), run_time=0.8)
        self.play(
            ApplyMethod(frame.reorient, 85, -10),
            ApplyMethod(frame.set_height, 6.4),
            run_time=1.8,
        )

        axes = ThreeDAxes(
            x_range=(-1.3, 1.3, 1),
            y_range=(-1.3, 1.3, 1),
            z_range=(-1.3, 1.3, 1),
            width=5.2,
            height=5.2,
            depth=5.2,
        )
        axes.set_stroke(GREY_B, width=1.4, opacity=0.42)

        sphere = Sphere(radius=1.0)
        sphere.set_color(BLUE_E)
        sphere.set_opacity(0.14)

        cosine_formula = Text("cosine similarity = cos(theta)", font_size=30, color=WHITE)
        cosine_formula.to_edge(UP, buff=0.7)
        cosine_formula.fix_in_frame()

        sphere_note = caption_text("sphere = all unit-length directions", scale=0.38, color=TEAL_A)
        sphere_note.next_to(cosine_formula, DOWN, buff=0.14)
        sphere_note.fix_in_frame()

        background_note = caption_text(
            "faint dots = other normalized embeddings on the same sphere",
            scale=0.33,
            color=GREY_B,
        )
        background_note.next_to(sphere_note, DOWN, buff=0.12)
        background_note.fix_in_frame()

        sphere_caption = caption_text("On the hypersphere, only direction changes similarity.", scale=0.4)
        sphere_caption.next_to(explainer, UP, buff=0.18)
        sphere_caption.fix_in_frame()

        anchor_vec = spherical_to_cartesian(0.35, 1.05)
        nearby_start = spherical_to_cartesian(1.15, 1.0)
        nearby_end = spherical_to_cartesian(0.55, 1.03)
        far_end = spherical_to_cartesian(2.15, 1.08)
        anchor_surface = 1.03 * anchor_vec
        anchor_dot = Sphere(radius=0.04).set_color(ANCHOR_COLOR).move_to(anchor_surface)
        background_dirs = [
            spherical_to_cartesian(-2.0, 1.25),
            spherical_to_cartesian(2.55, 1.3),
            spherical_to_cartesian(-0.55, 1.9),
        ]
        background_points = Group(
            *[
                Sphere(radius=0.03)
                .set_color(OTHER_POINT_COLOR)
                .set_opacity(0.6)
                .move_to(1.02 * vec)
                for vec in background_dirs
            ]
        )

        t_tracker = ValueTracker(0.0)

        def tracked_direction():
            if t_tracker.get_value() <= 1.0:
                alpha = t_tracker.get_value()
                start = nearby_start
                end = nearby_end
            else:
                alpha = t_tracker.get_value() - 1.0
                start = nearby_end
                end = far_end
            blended = interpolate(start, end, alpha)
            return blended / np.linalg.norm(blended)

        moving_dot = always_redraw(
            lambda: Sphere(radius=0.04)
            .set_color(MOVING_COLOR)
            .move_to(1.03 * tracked_direction())
        )

        anchor_arrow = always_redraw(
            lambda: Arrow(
                ORIGIN,
                1.03 * anchor_vec,
                buff=0.0,
                stroke_width=5,
                color=ANCHOR_COLOR,
            )
        )
        moving_arrow = always_redraw(
            lambda: Arrow(
                ORIGIN,
                1.03 * tracked_direction(),
                buff=0.0,
                stroke_width=5,
                color=MOVING_COLOR,
            )
        )

        def great_circle_arc_points(a, b, n=36):
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            dot_ab = np.clip(np.dot(a, b), -1, 1)
            omega = np.arccos(dot_ab)
            if omega < 1e-5:
                return [a.copy() for _ in range(n)]
            pts = []
            for s in np.linspace(0, 1, n):
                p = (
                    np.sin((1 - s) * omega) / np.sin(omega) * a
                    + np.sin(s * omega) / np.sin(omega) * b
                )
                pts.append(p / np.linalg.norm(p))
            return pts

        arc = always_redraw(
            lambda: VMobject(stroke_color=WHITE, stroke_width=4).set_points_smoothly(
                [1.01 * p for p in great_circle_arc_points(anchor_vec, tracked_direction(), n=40)]
            )
        )

        theta_label = Text("theta", font_size=20, color=WHITE)

        def update_theta_label(mob):
            mob.move_to(
                1.2 * great_circle_arc_points(anchor_vec, tracked_direction(), n=25)[12]
                + RIGHT * 0.08
            )
            return mob

        theta_label.add_updater(update_theta_label)

        role_text = caption_text(
            "gold = anchor direction   teal = comparison direction",
            scale=0.35,
        )
        role_text.next_to(background_note, DOWN, buff=0.12)
        role_text.fix_in_frame()

        cosine_text = Text("cosine =", font_size=24, color=WHITE)
        cosine_value = DecimalNumber(np.dot(anchor_vec, nearby_start), num_decimal_places=2)
        cosine_value.set_color(WHITE)

        def update_cosine(mob):
            mob.set_value(np.clip(np.dot(anchor_vec, tracked_direction()), -1, 1))
            return mob

        cosine_value.add_updater(update_cosine)
        cosine_group = VGroup(cosine_text, cosine_value).arrange(RIGHT, buff=0.14)
        cosine_group.next_to(explainer, UP, buff=0.38)
        cosine_group.fix_in_frame()

        self.play(
            FadeIn(axes),
            FadeIn(sphere),
            FadeIn(cosine_formula, shift=DOWN * 0.08),
            FadeIn(sphere_note, shift=DOWN * 0.08),
            FadeIn(background_note, shift=DOWN * 0.08),
            Transform(norm_formula, cosine_formula),
            run_time=1.2,
        )
        self.remove(norm_formula)
        self.add(cosine_formula)
        self.play(
            FadeIn(background_points),
            FadeIn(anchor_dot),
            FadeIn(moving_dot),
            run_time=0.8,
        )
        self.play(
            ShowCreation(anchor_arrow),
            ShowCreation(moving_arrow),
            FadeIn(role_text, shift=RIGHT * 0.1),
            run_time=0.9,
        )
        self.play(
            ShowCreation(arc),
            FadeIn(theta_label),
            FadeIn(sphere_caption, shift=UP * 0.08),
            FadeIn(cosine_group, shift=UP * 0.08),
            run_time=0.8,
        )
        self.wait(0.4)

        # Move closer in direction, then farther apart.
        self.play(t_tracker.animate.set_value(1.0), run_time=1.8)
        self.wait(0.3)
        self.play(t_tracker.animate.set_value(2.0), run_time=2.0)
        self.wait(1.2)
