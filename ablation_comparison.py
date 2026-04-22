from manimlib import *
import numpy as np


class Scene10AblationComparison(Scene):
    """
    Scene 10 — Ablation: Frozen CLIP + MLP vs Fine-Tuned CLIP

    Notes:
    - Replace LEFT_FINAL / RIGHT_FINAL and series values with your actual results.
    - The scene only highlights the winner based on the values you set here.
    """

    LEFT_FINAL = 0.62
    RIGHT_FINAL = 0.57

    LEFT_SERIES = [0.50, 0.54, 0.57, 0.59, 0.60, 0.61, 0.62]
    RIGHT_SERIES = [0.50, 0.53, 0.55, 0.56, 0.57, 0.57, 0.57]

    def construct(self):
        title = Text(
            "Ablation: Frozen CLIP + MLP vs Fine-tuned CLIP",
            font_size=40
        )
        title.to_edge(UP, buff=0.3)

        left_panel = self.build_pipeline_panel(
            title_text="Frozen CLIP + MLP",
            subtitle_text="only head updates",
            frozen_backbone=True,
            chart_color=BLUE,
        )
        right_panel = self.build_pipeline_panel(
            title_text="Fine-tuned CLIP",
            subtitle_text="many layers update",
            frozen_backbone=False,
            chart_color=YELLOW,
        )

        left_panel.move_to(LEFT * 3.75 + UP * 0.15)
        right_panel.move_to(RIGHT * 3.75 + UP * 0.15)

        self.play(FadeIn(title, shift=0.2 * DOWN))
        self.play(
            FadeIn(left_panel, shift=0.3 * RIGHT),
            FadeIn(right_panel, shift=0.3 * LEFT),
            run_time=1.1,
        )
        self.wait(0.3)

        # Left: explicitly frozen backbone
        self.play(
            ShowCreation(left_panel.backbone.freeze_glow),
            FadeIn(left_panel.lock_icon, scale=0.7),
            run_time=0.8,
        )
        self.play(
            left_panel.backbone.layers.animate.set_opacity(0.6),
            run_time=0.4,
        )

        # Left: trainable head
        self.play(
            ShowCreation(left_panel.head.train_glow),
            FadeIn(left_panel.head.train_badge, shift=0.1 * UP),
            left_panel.head.body.animate.scale(1.05),
            run_time=0.8,
        )
        self.play(
            left_panel.head.body.animate.scale(1 / 1.05),
            run_time=0.3,
        )

        # Right: many trainable layers
        self.play(
            LaggedStart(
                *[ShowCreation(glow) for glow in right_panel.backbone.train_glows],
                lag_ratio=0.12
            ),
            FadeIn(right_panel.backbone.badges, lag_ratio=0.08),
            ShowCreation(right_panel.head.train_glow),
            FadeIn(right_panel.head.train_badge, shift=0.1 * UP),
            run_time=1.0,
        )

        # Right: pulse trainable modules so it feels active
        for _ in range(2):
            self.play(
                right_panel.backbone.layer1.animate.scale(1.06),
                right_panel.backbone.layer2.animate.scale(1.06),
                right_panel.backbone.layer3.animate.scale(1.06),
                right_panel.head.body.animate.scale(1.04),
                run_time=0.35,
            )
            self.play(
                right_panel.backbone.layer1.animate.scale(1 / 1.06),
                right_panel.backbone.layer2.animate.scale(1 / 1.06),
                right_panel.backbone.layer3.animate.scale(1 / 1.06),
                right_panel.head.body.animate.scale(1 / 1.04),
                run_time=0.35,
            )

        self.wait(0.2)

        # Metric labels
        left_metric_label = Text("validation accuracy", font_size=24)
        right_metric_label = Text("validation accuracy", font_size=24)
        left_metric_label.next_to(left_panel.chart_group, UP, buff=0.12)
        right_metric_label.next_to(right_panel.chart_group, UP, buff=0.12)

        self.play(
            FadeIn(left_metric_label, shift=0.1 * DOWN),
            FadeIn(right_metric_label, shift=0.1 * DOWN),
            run_time=0.5,
        )

        # Animate counters + charts
        self.play(
            UpdateFromAlphaFunc(
                left_panel.counter,
                self.make_counter_updater(self.LEFT_SERIES[0], self.LEFT_FINAL),
            ),
            UpdateFromAlphaFunc(
                right_panel.counter,
                self.make_counter_updater(self.RIGHT_SERIES[0], self.RIGHT_FINAL),
            ),
            UpdateFromAlphaFunc(
                left_panel.chart_group,
                self.make_chart_updater(
                    left_panel.chart_group,
                    self.LEFT_SERIES,
                    BLUE,
                ),
            ),
            UpdateFromAlphaFunc(
                right_panel.chart_group,
                self.make_chart_updater(
                    right_panel.chart_group,
                    self.RIGHT_SERIES,
                    YELLOW,
                ),
            ),
            run_time=2.8,
            rate_func=linear,
        )
        self.wait(0.4)

        comparison_card = self.build_comparison_card(
            left_name="Frozen CLIP + MLP",
            right_name="Fine-tuned CLIP",
            left_acc=self.LEFT_FINAL,
            right_acc=self.RIGHT_FINAL,
        )
        comparison_card.to_edge(DOWN, buff=0.28)

        self.play(FadeIn(comparison_card, shift=0.2 * UP), run_time=0.8)
        self.wait(0.3)

        # Highlight winner using actual numbers
        if self.LEFT_FINAL > self.RIGHT_FINAL:
            winner_box = SurroundingRectangle(
                left_panel,
                buff=0.16,
                color=GREEN,
                stroke_width=5
            )
            winner_label = Text("better here", font_size=28, color=GREEN)
            winner_label.next_to(left_panel, UP, buff=0.1)
            acc_word = comparison_card.accuracy_left
            stable_word = comparison_card.stability_left
            gen_word = comparison_card.generalization_left
        elif self.RIGHT_FINAL > self.LEFT_FINAL:
            winner_box = SurroundingRectangle(
                right_panel,
                buff=0.16,
                color=GREEN,
                stroke_width=5
            )
            winner_label = Text("better here", font_size=28, color=GREEN)
            winner_label.next_to(right_panel, UP, buff=0.1)
            acc_word = comparison_card.accuracy_right
            stable_word = comparison_card.stability_right
            gen_word = comparison_card.generalization_right
        else:
            winner_box = SurroundingRectangle(
                VGroup(left_panel, right_panel),
                buff=0.18,
                color=GREEN,
                stroke_width=5
            )
            winner_label = Text("tie on this metric", font_size=28, color=GREEN)
            winner_label.next_to(VGroup(left_panel, right_panel), UP, buff=0.1)
            acc_word = VGroup(comparison_card.accuracy_left, comparison_card.accuracy_right)
            stable_word = VGroup(comparison_card.stability_left, comparison_card.stability_right)
            gen_word = VGroup(
                comparison_card.generalization_left,
                comparison_card.generalization_right,
            )

        self.play(
            ShowCreation(winner_box),
            FadeIn(winner_label, shift=0.1 * UP),
            Flash(acc_word, flash_radius=0.45),
            run_time=1.0,
        )
        self.play(
            Flash(stable_word, flash_radius=0.45),
            Flash(gen_word, flash_radius=0.45),
            run_time=0.9,
        )

        self.wait(1.5)

    # ---------------------------------------------------------
    # Panel builders
    # ---------------------------------------------------------

    def build_pipeline_panel(
        self,
        title_text,
        subtitle_text,
        frozen_backbone=True,
        chart_color=BLUE,
    ):
        panel_box = RoundedRectangle(
            width=5.9,
            height=5.95,
            corner_radius=0.2,
        )
        panel_box.set_stroke(GREY_B, 2)
        panel_box.set_fill(GREY_E, 0.08)

        title = Text(title_text, font_size=31)
        subtitle = Text(subtitle_text, font_size=22)
        subtitle.set_color(GREY_B)

        image_box = self.build_image_box()
        backbone = self.build_clip_backbone(frozen=frozen_backbone)
        head = self.build_mlp_head()

        arrow1 = Arrow(
            image_box.get_right(),
            backbone.get_left(),
            buff=0.12,
            stroke_width=4,
        )
        arrow2 = Arrow(
            backbone.get_right(),
            head.get_left(),
            buff=0.12,
            stroke_width=4,
        )

        pipeline_row = VGroup(image_box, backbone, head)
        pipeline_row.arrange(RIGHT, buff=0.33, aligned_edge=DOWN)

        arrow1.put_start_and_end_on(
            image_box.get_right() + RIGHT * 0.03,
            backbone.get_left() + LEFT * 0.03,
        )
        arrow2.put_start_and_end_on(
            backbone.get_right() + RIGHT * 0.03,
            head.get_left() + LEFT * 0.03,
        )

        chart_group, counter = self.build_chart_and_counter(chart_color)

        keyword_row = VGroup(
            Text("stability", font_size=21),
            Text("accuracy", font_size=21),
            Text("generalization", font_size=21),
        )
        keyword_row.arrange(RIGHT, buff=0.18)

        content = VGroup(
            title,
            subtitle,
            VGroup(pipeline_row, arrow1, arrow2),
            chart_group,
            keyword_row,
        )
        content.arrange(DOWN, buff=0.23)
        content.move_to(panel_box.get_center() + DOWN * 0.02)

        lock_icon = self.build_lock_icon()
        lock_icon.scale(0.55)
        lock_icon.next_to(backbone, UP, buff=0.1)
        lock_icon.set_opacity(0)

        group = VGroup(panel_box, content, lock_icon)
        group.backbone = backbone
        group.head = head
        group.chart_group = chart_group
        group.counter = counter
        group.lock_icon = lock_icon
        return group

    def build_image_box(self):
        image_box = RoundedRectangle(
            width=1.18,
            height=1.5,
            corner_radius=0.1,
        )
        image_box.set_stroke(GREY_B, 2)
        image_box.set_fill(GREY_D, 0.35)

        # simple portrait placeholder
        head = Circle(radius=0.16)
        shoulders = ArcBetweenPoints(
            LEFT * 0.23 + DOWN * 0.05,
            RIGHT * 0.23 + DOWN * 0.05,
            angle=-PI / 2,
        )
        shoulders.set_stroke(WHITE, 2)

        head.move_to(image_box.get_center() + UP * 0.23)
        shoulders.move_to(image_box.get_center() + DOWN * 0.18)

        label = Text("image", font_size=20)
        label.next_to(image_box, DOWN, buff=0.08)

        return VGroup(image_box, head, shoulders, label)

    def build_clip_backbone(self, frozen=True):
        label = Text("CLIP backbone", font_size=22)

        layer1 = RoundedRectangle(width=0.38, height=1.02, corner_radius=0.08)
        layer2 = RoundedRectangle(width=0.38, height=1.18, corner_radius=0.08)
        layer3 = RoundedRectangle(width=0.38, height=0.94, corner_radius=0.08)

        layers = VGroup(layer1, layer2, layer3)
        layers.arrange(RIGHT, buff=0.11)

        if frozen:
            fills = [GREY_D, GREY_C, GREY_D]
            opacities = [0.35, 0.42, 0.35]
        else:
            fills = [TEAL_E, BLUE_E, YELLOW_E]
            opacities = [0.72, 0.78, 0.84]

        badges = VGroup()
        train_glows = VGroup()

        for layer, fill, opacity in zip(layers, fills, opacities):
            layer.set_fill(fill, opacity=opacity)
            layer.set_stroke(WHITE, 1.5)

            glow = SurroundingRectangle(
                layer,
                buff=0.03,
                color=YELLOW,
                stroke_width=4,
            )
            glow.set_opacity(0)
            layer.train_glow = glow
            train_glows.add(glow)

            if frozen:
                badge = Text("frozen", font_size=13, color=GREY_B)
            else:
                badge = Text("train", font_size=13, color=YELLOW)

            badge.next_to(layer, DOWN, buff=0.05)
            badges.add(badge)

        if frozen:
            badges.set_opacity(0.9)
        else:
            badges.set_opacity(0)

        freeze_glow = SurroundingRectangle(
            layers,
            buff=0.08,
            color=BLUE,
            stroke_width=5,
        )
        freeze_glow.set_opacity(0)

        body = VGroup(layers, badges)
        body.arrange(DOWN, buff=0.12)

        full = VGroup(label, body, train_glows, freeze_glow)
        full.arrange(DOWN, buff=0.14)

        full.layers = layers
        full.layer1 = layer1
        full.layer2 = layer2
        full.layer3 = layer3
        full.badges = badges
        full.train_glows = train_glows
        full.freeze_glow = freeze_glow
        return full

    def build_mlp_head(self):
        label = Text("MLP head", font_size=22)

        node_cols = VGroup()
        for n_nodes in [2, 4, 2]:
            col = VGroup(*[Dot(radius=0.05) for _ in range(n_nodes)])
            col.arrange(DOWN, buff=0.15)
            node_cols.add(col)
        node_cols.arrange(RIGHT, buff=0.24)

        connections = VGroup()
        for i in range(len(node_cols) - 1):
            for a in node_cols[i]:
                for b in node_cols[i + 1]:
                    line = Line(
                        a.get_center(),
                        b.get_center(),
                        stroke_width=1.2,
                        color=GREY_B,
                    )
                    connections.add(line)

        body = VGroup(connections, node_cols)

        body_box = RoundedRectangle(
            width=1.75,
            height=1.45,
            corner_radius=0.08,
        )
        body_box.set_stroke(GREY_B, 1.5)
        body_box.set_fill(BLACK, 0)

        head_group = VGroup(body_box, body)
        body.move_to(body_box.get_center())

        train_badge = Text("trainable", font_size=15, color=YELLOW)
        train_badge.set_opacity(0)

        train_glow = SurroundingRectangle(
            head_group,
            buff=0.05,
            color=YELLOW,
            stroke_width=5,
        )
        train_glow.set_opacity(0)

        full = VGroup(label, head_group, train_badge, train_glow)
        full.arrange(DOWN, buff=0.1)

        full.body = head_group
        full.train_badge = train_badge
        full.train_glow = train_glow
        return full

    def build_lock_icon(self):
        body = RoundedRectangle(
            width=0.34,
            height=0.24,
            corner_radius=0.05,
        )
        body.set_fill(GREY_C, 1)
        body.set_stroke(WHITE, 1.5)

        shackle = Arc(
            start_angle=PI,
            angle=PI,
            radius=0.12,
        )
        shackle.rotate(-PI / 2)
        shackle.set_stroke(GREY_C, 6)
        shackle.next_to(body, UP, buff=-0.01)

        return VGroup(shackle, body)

    def build_chart_and_counter(self, color):
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0.45, 0.70, 0.05],
            width=2.55,
            height=1.35,
            axis_config={
                "include_tip": False,
                "stroke_width": 2,
            },
        )
        axes.set_stroke(GREY_B, 2)

        line = VMobject()
        line.set_points_as_corners([
            axes.c2p(0, 0.50),
            axes.c2p(1, 0.50),
        ])
        line.set_stroke(color, 4)

        counter = DecimalNumber(
            50.0,
            num_decimal_places=1,
            font_size=32,
        )
        percent = Text("%", font_size=26)

        counter_group = VGroup(counter, percent)
        counter_group.arrange(RIGHT, buff=0.03)
        counter_group.next_to(axes, RIGHT, buff=0.22)

        chart_group = VGroup(axes, line, counter_group)
        chart_group.axes = axes
        chart_group.line = line
        chart_group.counter = counter
        return chart_group, counter

    def build_comparison_card(self, left_name, right_name, left_acc, right_acc):
        card = RoundedRectangle(
            width=10.1,
            height=1.7,
            corner_radius=0.18,
        )
        card.set_stroke(GREY_B, 2)
        card.set_fill(GREY_E, 0.2)

        divider = Line(UP * 0.58, DOWN * 0.58)
        divider.set_stroke(GREY_B, 2)

        left_title = Text(left_name, font_size=26)
        right_title = Text(right_name, font_size=26)

        left_acc_text = Text(f"accuracy: {left_acc * 100:.1f}%", font_size=23)
        right_acc_text = Text(f"accuracy: {right_acc * 100:.1f}%", font_size=23)

        stability_left = Text("stability", font_size=21)
        accuracy_left = Text("accuracy", font_size=21)
        generalization_left = Text("generalization", font_size=21)

        stability_right = Text("stability", font_size=21)
        accuracy_right = Text("accuracy", font_size=21)
        generalization_right = Text("generalization", font_size=21)

        left_keywords = VGroup(
            stability_left,
            accuracy_left,
            generalization_left,
        ).arrange(RIGHT, buff=0.12)

        right_keywords = VGroup(
            stability_right,
            accuracy_right,
            generalization_right,
        ).arrange(RIGHT, buff=0.12)

        left_col = VGroup(left_title, left_acc_text, left_keywords)
        left_col.arrange(DOWN, buff=0.07)

        right_col = VGroup(right_title, right_acc_text, right_keywords)
        right_col.arrange(DOWN, buff=0.07)

        content = VGroup(left_col, divider, right_col)
        content.arrange(RIGHT, buff=0.35)
        content.move_to(card.get_center())

        group = VGroup(card, content)
        group.stability_left = stability_left
        group.accuracy_left = accuracy_left
        group.generalization_left = generalization_left
        group.stability_right = stability_right
        group.accuracy_right = accuracy_right
        group.generalization_right = generalization_right
        return group

    # ---------------------------------------------------------
    # Animation helpers
    # ---------------------------------------------------------

    def make_counter_updater(self, start, end):
        def updater(mob, alpha):
            value = interpolate(start, end, alpha)
            mob.set_value(value * 100)
        return updater

    def make_chart_updater(self, chart_group, series, chart_color):
        axes = chart_group.axes
        line = chart_group.line
        xs = np.linspace(0, 6, len(series))

        def updater(mob, alpha):
            n = max(2, int(np.ceil(interpolate(2, len(series), alpha))))
            visible_xs = xs[:n]
            visible_ys = series[:n]
            points = [
                axes.c2p(x, y)
                for x, y in zip(visible_xs, visible_ys)
            ]

            new_line = VMobject()
            new_line.set_points_as_corners(points)
            new_line.set_stroke(chart_color, 4)
            line.become(new_line)

        return updater
