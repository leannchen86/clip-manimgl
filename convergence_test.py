from manimlib import *
import numpy as np
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from layout_verifier import ChartAnnotation, ChartVerificationSpec, LayoutVerifier


class ConvergenceTest(Scene):
    def construct(self):
        verifier = LayoutVerifier(scene_name="ConvergenceTest")

        # Toggle this depending on your experiment
        two_way = True

        if two_way:
            baseline_y = 50
            baseline_label_text = "50% random baseline"
            y_max = 100
            y_step = 10
            curve_points = [
                (5, 54),
                (10, 59),
                (50, 69),
                (100, 74),
                (500, 81),
                (1000, 83),
                (5000, 84.5),
                (7500, 84.8),
            ]
            plateau_y = 85
        else:
            baseline_y = 2
            baseline_label_text = "~2% random baseline"
            y_max = 20
            y_step = 2
            curve_points = [
                (5, 3.2),
                (10, 4.5),
                (50, 8.8),
                (100, 10.7),
                (500, 13.4),
                (1000, 14.2),
                (5000, 14.8),
                (7500, 14.9),
            ]
            plateau_y = 15

        x_min = curve_points[0][0]
        x_max = curve_points[-1][0]
        log_x_max = np.log10(x_max / x_min)

        axes = Axes(
            x_range=[0, log_x_max, 0.5],
            y_range=[0, y_max, y_step],
            width=10.5,
            height=5.8,
            axis_config={
                "include_tip": True,
                "stroke_width": 2,
            },
            x_axis_config={
                "include_numbers": False,
                "include_ticks": False,
            },
            y_axis_config={
                "include_numbers": False,
            },
        )
        axes.center().shift(DOWN * 0.2)

        x_label = Text("training images per name", font_size=32)
        x_label.next_to(axes.x_axis, DOWN, buff=0.32)

        scale_note = Text("x-axis spacing is logarithmic", font_size=20)
        scale_note.set_color(GREY_B)
        scale_note.next_to(axes.x_axis, UP, buff=0.16)
        scale_note.align_to(axes.x_axis, RIGHT)

        y_label = Text("accuracy", font_size=30)
        y_label.rotate(PI / 2)
        y_label.next_to(axes.y_axis, LEFT, buff=0.72)
        y_label.shift(DOWN * 0.12)

        x_tick_values = [5, 50, 500, 5000]
        x_ticks = VGroup()
        x_tick_labels = VGroup()
        for x in x_tick_values:
            point = axes.c2p(self.x_to_axis(x, x_min), 0)
            tick = Line(UP, DOWN)
            tick.set_height(0.16)
            tick.set_stroke(GREY_B, width=2)
            tick.move_to(point)

            label = Text(f"{x:,}", font_size=20)
            label.next_to(point, DOWN, buff=0.14)

            x_ticks.add(tick)
            x_tick_labels.add(label)

        endpoint_label = Text("7,500", font_size=18)
        endpoint_label.next_to(
            axes.c2p(self.x_to_axis(7500, x_min), 0),
            DOWN,
            buff=0.14,
        )
        endpoint_label.shift(RIGHT * 0.18)

        y_tick_labels = VGroup()
        y_label_values = [baseline_y]
        if two_way:
            y_label_values += [70, 80, 90]
        else:
            y_label_values += [6, 10, 14, 18]

        for y in y_label_values:
            label = Text(f"{y:g}%", font_size=22)
            label.next_to(axes.c2p(0, y), LEFT, buff=0.14)
            y_tick_labels.add(label)

        baseline = DashedLine(
            axes.c2p(0, baseline_y),
            axes.c2p(log_x_max, baseline_y),
            dash_length=0.08,
            positive_space_ratio=0.55,
            stroke_width=2,
        )
        baseline.set_stroke(GREY_B, opacity=0.7)

        baseline_label = Text(baseline_label_text, font_size=26)
        baseline_label.set_color(GREY_B)
        baseline_label.next_to(
            axes.c2p(self.x_to_axis(2500, x_min), baseline_y),
            UP,
            buff=0.16,
        )

        graph = axes.get_graph(
            lambda axis_x: self.interp_curve(self.axis_to_x(axis_x, x_min), curve_points),
            x_range=[0, log_x_max],
            color=BLUE,
        )
        graph.set_stroke(width=4)

        markers = VGroup()
        for x, y in curve_points:
            dot = Dot(
                axes.c2p(self.x_to_axis(x, x_min), y),
                radius=0.055,
                color=BLUE,
            )
            markers.add(dot)

        asymptote = DashedLine(
            axes.c2p(self.x_to_axis(3200, x_min), plateau_y),
            axes.c2p(log_x_max, plateau_y),
            dash_length=0.09,
            positive_space_ratio=0.45,
            stroke_width=2,
        )
        asymptote.set_stroke(YELLOW, opacity=0.35)

        signal_point = axes.c2p(
            self.x_to_axis(10, x_min),
            self.interp_curve(10, curve_points),
        )
        slow_point = axes.c2p(
            self.x_to_axis(500, x_min),
            self.interp_curve(500, curve_points),
        )
        plateau_point = axes.c2p(
            self.x_to_axis(7500, x_min),
            self.interp_curve(7500, curve_points),
        )

        signal_label = Text("signal appears", font_size=28)
        signal_label.next_to(signal_point, DOWN + RIGHT, buff=0.42)
        signal_label.shift(RIGHT * 0.38 + DOWN * 0.22)
        signal_arrow = Arrow(
            signal_label.get_top() + LEFT * 0.34,
            signal_point + DOWN * 0.03,
            buff=0.08,
            stroke_width=3,
        )

        slow_label = Text("gains slow", font_size=28)
        slow_label.next_to(slow_point, UP + RIGHT, buff=0.3)
        slow_label.shift(RIGHT * 0.42 + UP * 0.1)
        slow_arrow = Arrow(
            slow_label.get_left() + DOWN * 0.1,
            slow_point + UP * 0.03,
            buff=0.08,
            stroke_width=3,
        )

        plateau_label = Text("plateau", font_size=30)
        plateau_label.next_to(plateau_point, UP + LEFT, buff=0.2)
        plateau_label.shift(LEFT * 0.34 + UP * 0.04)
        plateau_arrow = Arrow(
            plateau_label.get_right() + DOWN * 0.02,
            plateau_point + LEFT * 0.03,
            buff=0.08,
            stroke_width=3,
        )

        spec = ChartVerificationSpec(
            scene_name="ConvergenceTest",
            x_axis=axes.x_axis,
            y_axis=axes.y_axis,
            axis_labels=[("x_label", x_label), ("y_label", y_label)],
            plot_graph=graph,
            x_tick_labels=VGroup(*x_tick_labels, endpoint_label),
            y_tick_labels=y_tick_labels,
            annotations=[
                ChartAnnotation("signal_label", signal_label, signal_arrow),
                ChartAnnotation("slow_label", slow_label, slow_arrow),
                ChartAnnotation("plateau_label", plateau_label, plateau_arrow),
            ],
            reference_labels=[("baseline_label", baseline_label), ("scale_note", scale_note)],
            nonlinear_x=True,
            scale_note=scale_note,
            x_tick_values=[5, 50, 500, 5000, 7500],
        )
        verifier.check_chart_spec(spec)
        if verifier.warnings:
            print(
                "[layout warnings] ConvergenceTest\n"
                + "\n".join(f"- {warning}" for warning in verifier.warnings)
            )
        verifier.assert_ok()

        self.play(
            ShowCreation(axes),
            FadeIn(x_label, shift=UP * 0.15),
            FadeIn(scale_note, shift=UP * 0.1),
            FadeIn(y_label, shift=RIGHT * 0.15),
            run_time=1.6,
        )
        self.play(
            LaggedStart(*[ShowCreation(tick) for tick in x_ticks], lag_ratio=0.08),
            LaggedStart(*[FadeIn(m, scale=0.9) for m in x_tick_labels], lag_ratio=0.08),
            FadeIn(endpoint_label, scale=0.9),
            LaggedStart(*[FadeIn(m, scale=0.9) for m in y_tick_labels], lag_ratio=0.08),
            run_time=1.2,
        )

        self.play(
            ShowCreation(baseline),
            FadeIn(baseline_label, shift=UP * 0.15),
            run_time=1.0,
        )

        self.play(
            ShowCreation(graph),
            run_time=2.6,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(dot) for dot in markers], lag_ratio=0.08),
            run_time=1.0,
        )

        self.play(
            ShowCreation(signal_arrow),
            FadeIn(signal_label, shift=UP * 0.15),
            run_time=0.8,
        )
        self.play(
            ShowCreation(slow_arrow),
            FadeIn(slow_label, shift=UP * 0.15),
            run_time=0.8,
        )
        self.play(
            ShowCreation(plateau_arrow),
            FadeIn(plateau_label, shift=UP * 0.15),
            ShowCreation(asymptote),
            run_time=1.0,
        )

        self.wait(2)

    def interp_curve(self, x, points):
        """
        Smooth interpolation in log-x space so the early rise feels steep
        and later growth feels flattened, matching the scene description.
        """
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)

        lx = np.log10(xs)
        xq = np.log10(max(x, xs[0]))

        if xq <= lx[0]:
            return ys[0]
        if xq >= lx[-1]:
            return ys[-1]

        idx = np.searchsorted(lx, xq) - 1
        idx = np.clip(idx, 0, len(lx) - 2)

        t = (xq - lx[idx]) / (lx[idx + 1] - lx[idx])
        return interpolate(ys[idx], ys[idx + 1], t)

    def x_to_axis(self, x, x_min):
        return np.log10(max(x, x_min) / x_min)

    def axis_to_x(self, axis_x, x_min):
        return x_min * (10 ** axis_x)
