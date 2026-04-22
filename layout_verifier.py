from dataclasses import dataclass
from itertools import combinations

from manimlib import FRAME_X_RADIUS, FRAME_Y_RADIUS


@dataclass
class ChartAnnotation:
    name: str
    label: object
    arrow: object | None = None


@dataclass
class ChartVerificationSpec:
    scene_name: str
    x_axis: object
    y_axis: object
    plot_graph: object
    x_tick_labels: object
    y_tick_labels: object
    annotations: list[ChartAnnotation]
    reference_labels: list[tuple[str, object]]
    axis_labels: list[tuple[str, object]] | None = None
    nonlinear_x: bool = False
    scale_note: object | None = None
    x_tick_values: list[float] | None = None


class LayoutVerifier:
    """
    Lightweight geometry-based verifier for ManimGL layouts.

    This is intended to catch the most common LLM layout failures:
    - overlapping text / formulas / charts
    - not enough gap in vertical stacks
    - content drifting outside the camera frame
    """

    def __init__(self, scene_name="Scene"):
        self.scene_name = scene_name
        self.issues = []
        self.warnings = []

    def _bounds(self, mob, pad=0.0):
        return {
            "left": mob.get_left()[0] - pad,
            "right": mob.get_right()[0] + pad,
            "bottom": mob.get_bottom()[1] - pad,
            "top": mob.get_top()[1] + pad,
        }

    def _record(self, message):
        self.issues.append(message)

    def _warn(self, message):
        self.warnings.append(message)

    def check_no_overlap(self, name_a, mob_a, name_b, mob_b, min_gap=0.0):
        a = self._bounds(mob_a, pad=min_gap / 2)
        b = self._bounds(mob_b, pad=min_gap / 2)

        overlaps_x = a["left"] < b["right"] and a["right"] > b["left"]
        overlaps_y = a["bottom"] < b["top"] and a["top"] > b["bottom"]

        if overlaps_x and overlaps_y:
            self._record(
                f"{name_a} overlaps {name_b} "
                f"(required minimum gap {min_gap:.2f})."
            )

    def check_curve_clearance(self, label_name, label_mob, curve_name, curve_mob, min_gap=0.0):
        points = curve_mob.get_points()
        if len(points) == 0:
            self.check_no_overlap(label_name, label_mob, curve_name, curve_mob, min_gap=min_gap)
            return

        bounds = self._bounds(label_mob, pad=min_gap)
        for point in points:
            x, y = point[:2]
            if bounds["left"] <= x <= bounds["right"] and bounds["bottom"] <= y <= bounds["top"]:
                self._record(
                    f"{label_name} overlaps {curve_name} "
                    f"(required minimum gap {min_gap:.2f})."
                )
                return

    def check_arrow_label_clearance(self, label_name, label_mob, arrow_name, arrow_mob):
        points = arrow_mob.get_points()
        if len(points) == 0:
            return

        bounds = self._bounds(label_mob, pad=-0.01)
        interior_hits = 0
        for point in points[3:]:
            x, y = point[:2]
            if bounds["left"] <= x <= bounds["right"] and bounds["bottom"] <= y <= bounds["top"]:
                interior_hits += 1

        if interior_hits > 3:
            self._record(f"{arrow_name} passes through {label_name}.")

    def check_vertical_order(self, upper_name, upper_mob, lower_name, lower_mob, min_gap=0.0):
        gap = upper_mob.get_bottom()[1] - lower_mob.get_top()[1]
        if gap < min_gap:
            self._record(
                f"{upper_name} is too close to {lower_name} "
                f"(gap {gap:.2f}, expected at least {min_gap:.2f})."
            )

    def check_inside_frame(self, name, mob, margin=0.0):
        b = self._bounds(mob)
        if b["left"] < -FRAME_X_RADIUS + margin:
            self._record(f"{name} extends past the left frame margin.")
        if b["right"] > FRAME_X_RADIUS - margin:
            self._record(f"{name} extends past the right frame margin.")
        if b["bottom"] < -FRAME_Y_RADIUS + margin:
            self._record(f"{name} extends past the bottom frame margin.")
        if b["top"] > FRAME_Y_RADIUS - margin:
            self._record(f"{name} extends past the top frame margin.")

    def check_min_horizontal_gap(self, left_name, left_mob, right_name, right_mob, min_gap=0.0):
        gap = right_mob.get_left()[0] - left_mob.get_right()[0]
        if gap < min_gap:
            self._record(
                f"{left_name} is too close to {right_name} "
                f"(horizontal gap {gap:.2f}, expected at least {min_gap:.2f})."
            )

    def check_group_pairwise_no_overlap(self, prefix, mobs, min_gap=0.0):
        items = list(mobs)
        for (index_a, mob_a), (index_b, mob_b) in combinations(enumerate(items), 2):
            self.check_no_overlap(
                f"{prefix}[{index_a}]",
                mob_a,
                f"{prefix}[{index_b}]",
                mob_b,
                min_gap=min_gap,
            )

    def check_chart_spec(self, spec: ChartVerificationSpec):
        self.check_inside_frame("x_axis", spec.x_axis, margin=0.1)
        self.check_inside_frame("y_axis", spec.y_axis, margin=0.1)
        self.check_inside_frame("plot_graph", spec.plot_graph, margin=0.1)

        if spec.axis_labels:
            for label_name, label_mob in spec.axis_labels:
                self.check_inside_frame(label_name, label_mob, margin=0.1)

        self.check_group_pairwise_no_overlap("x_tick_labels", spec.x_tick_labels, min_gap=0.04)
        self.check_group_pairwise_no_overlap("y_tick_labels", spec.y_tick_labels, min_gap=0.02)

        for annotation in spec.annotations:
            self.check_inside_frame(annotation.name, annotation.label, margin=0.1)
            self.check_curve_clearance(annotation.name, annotation.label, "plot_graph", spec.plot_graph, min_gap=0.08)
            if annotation.arrow is not None:
                self.check_arrow_label_clearance(annotation.name, annotation.label, f"{annotation.name}_arrow", annotation.arrow)
            for ref_name, ref_mob in spec.reference_labels:
                self.check_no_overlap(annotation.name, annotation.label, ref_name, ref_mob, min_gap=0.05)
            if annotation.arrow is not None and annotation.arrow.get_length() < 0.35:
                self._warn(f"{annotation.name} arrow is very short and may read ambiguously.")

        if spec.axis_labels:
            for label_name, label_mob in spec.axis_labels:
                self.check_no_overlap(label_name, label_mob, "x_axis", spec.x_axis, min_gap=0.05)
                self.check_no_overlap(label_name, label_mob, "y_axis", spec.y_axis, min_gap=0.05)

        if spec.nonlinear_x and spec.scale_note is None:
            self._warn("chart uses nonlinear x spacing without a scale disclosure note.")

        if spec.nonlinear_x and spec.x_tick_values and len(spec.x_tick_values) > 5:
            self._warn("dense numeric ticks on a nonlinear x-axis may read confusingly.")

        y_axis_gap = spec.plot_graph.get_left()[0] - spec.y_axis.get_right()[0]
        if y_axis_gap > 1.6:
            self._warn("plot data appears visually detached from the y-axis.")

        if spec.scale_note is not None:
            self.check_inside_frame("scale_note", spec.scale_note, margin=0.1)

    def get_report(self):
        lines = []
        if self.issues:
            lines.extend(f"- {issue}" for issue in self.issues)
        if self.warnings:
            lines.extend(f"- warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def assert_ok(self):
        if not self.issues:
            return

        issue_text = "\n".join(f"- {issue}" for issue in self.issues)
        raise ValueError(
            f"Layout verification failed in {self.scene_name}:\n{issue_text}"
        )
