from manimlib import *
import numpy as np


BG_COLOR = "#0f1117"
LABEL_COLOR = "#d5d8e8"


def caption_text(text, scale=0.42, color=LABEL_COLOR):
    mob = Text(text, font="Helvetica Neue", color=color)
    mob.scale(scale)
    return mob


class SoftmaxProbabilityBudgetScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Probabilities must sum to 1", font="Helvetica Neue", font_size=34)
        title.to_edge(UP, buff=0.45)

        names = ["William", "Jessica", "Sarah", "Thomas", "Laura", "Robert"]
        colors = [BLUE_B, PURPLE_B, PINK, ORANGE, TEAL, GREY_B]

        x_axis = Line(LEFT * 4.4, RIGHT * 4.4, stroke_width=2, stroke_color=GREY_B)
        x_axis.shift(DOWN * 1.8)

        chart_labels = VGroup()
        bars = VGroup()
        x_positions = np.linspace(-3.6, 3.6, len(names))
        initial = np.array([0.18, 0.16, 0.17, 0.17, 0.16, 0.16])

        for x, name, color, value in zip(x_positions, names, colors, initial):
            rect = Rectangle(
                width=0.78,
                height=3.6 * value,
                fill_color=color,
                fill_opacity=0.9,
                stroke_width=0,
            )
            rect.move_to(np.array([x, x_axis.get_y() + (3.6 * value) / 2, 0]))

            label = Text(name, font="Helvetica Neue", font_size=24)
            label.rotate(PI / 8)
            label.next_to(x_axis, DOWN, buff=0.25)
            label.set_x(x)

            bars.add(rect)
            chart_labels.add(label)

        bottom_caption = caption_text(
            "As the model gets more confident about some names, harder names can get crowded out",
            scale=0.44,
            color=WHITE,
        )
        bottom_caption.to_edge(DOWN, buff=0.35)

        robert_note = caption_text("Robert gets crowded out", scale=0.48, color=YELLOW)
        robert_note.next_to(bars[-1], RIGHT, buff=0.6).shift(UP * 0.7)

        self.play(Write(title))
        self.play(ShowCreation(x_axis))
        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.08),
            FadeIn(chart_labels, shift=UP * 0.15),
            run_time=1.4,
        )

        steps = [
            np.array([0.20, 0.18, 0.19, 0.16, 0.15, 0.12]),
            np.array([0.24, 0.21, 0.20, 0.15, 0.13, 0.07]),
            np.array([0.28, 0.23, 0.22, 0.13, 0.10, 0.04]),
            np.array([0.30, 0.25, 0.23, 0.11, 0.09, 0.02]),
        ]

        def animate_to_distribution(values):
            anims = []
            for rect, x, value in zip(bars, x_positions, values):
                target = Rectangle(
                    width=0.78,
                    height=3.6 * value,
                    fill_color=rect.get_fill_color(),
                    fill_opacity=0.9,
                    stroke_width=0,
                )
                target.move_to(np.array([x, x_axis.get_y() + (3.6 * value) / 2, 0]))
                anims.append(Transform(rect, target))
            return anims

        for values in steps:
            self.play(*animate_to_distribution(values), run_time=0.9)

        self.play(FadeIn(bottom_caption, shift=UP * 0.2))
        self.play(FadeIn(robert_note, shift=LEFT * 0.2))
        self.wait(1.5)
