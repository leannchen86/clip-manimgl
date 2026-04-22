from manimlib import *
import numpy as np


class VLMArchitectureComparison(Scene):
    def construct(self):
        title = Text("VLM Architecture Comparison", font_size=42)
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN))
        self.wait(0.2)

        divider = Line(UP * 3.3, DOWN * 3.3, stroke_opacity=0.25)
        self.play(ShowCreation(divider), run_time=0.8)

        # ----------------------------
        # Left side: CLIP pipeline
        # ----------------------------
        left_header = Text("CLIP", font_size=34, color=BLUE_B)
        left_header.move_to(LEFT * 3.4 + UP * 2.75)

        portrait_left = self.make_portrait()
        portrait_left.move_to(LEFT * 5.3 + UP * 1.2)

        image_encoder = self.make_box("Image Encoder", width=2.2, height=0.8, color=BLUE_D)
        image_encoder.move_to(LEFT * 3.4 + UP * 1.2)

        image_embedding = self.make_embedding_column(label="Image\nEmbedding", color=BLUE_C)
        image_embedding.move_to(LEFT * 1.6 + UP * 1.2)

        candidate_names = self.make_candidate_names()
        candidate_names.move_to(LEFT * 5.0 + DOWN * 1.35)

        text_encoder = self.make_box("Text Encoder", width=2.2, height=0.8, color=TEAL_D)
        text_encoder.move_to(LEFT * 3.4 + DOWN * 1.35)

        text_embedding = self.make_embedding_column(label="Text\nEmbeddings", color=TEAL_C)
        text_embedding.move_to(LEFT * 1.6 + DOWN * 1.35)

        similarity = self.make_box("Similarity", width=1.9, height=0.85, color=YELLOW_D)
        similarity.move_to(LEFT * 0.15)

        ranked_output = self.make_ranked_output()
        ranked_output.move_to(LEFT * 3.2 + DOWN * 2.65)

        clip_label = Text("CLIP compares embeddings", font_size=28, color=BLUE_B)
        clip_label.move_to(LEFT * 3.4 + DOWN * 3.2)

        # left arrows
        l_arrow_1 = Arrow(
            portrait_left.get_right(),
            image_encoder.get_left(),
            buff=0.12,
            stroke_width=5,
        )
        l_arrow_2 = Arrow(
            image_encoder.get_right(),
            image_embedding.get_left(),
            buff=0.12,
            stroke_width=5,
        )
        l_arrow_3 = Arrow(
            candidate_names.get_right(),
            text_encoder.get_left(),
            buff=0.12,
            stroke_width=5,
        )
        l_arrow_4 = Arrow(
            text_encoder.get_right(),
            text_embedding.get_left(),
            buff=0.12,
            stroke_width=5,
        )
        l_arrow_5 = Arrow(
            image_embedding.get_right(),
            similarity.get_left() + UP * 0.2,
            buff=0.1,
            stroke_width=5,
        )
        l_arrow_6 = Arrow(
            text_embedding.get_right(),
            similarity.get_left() + DOWN * 0.2,
            buff=0.1,
            stroke_width=5,
        )
        l_arrow_7 = Arrow(
            similarity.get_right(),
            ranked_output.get_left(),
            buff=0.15,
            stroke_width=5,
        )

        left_group = VGroup(
            left_header,
            portrait_left,
            image_encoder,
            image_embedding,
            candidate_names,
            text_encoder,
            text_embedding,
            similarity,
            ranked_output,
        )

        # ----------------------------
        # Right side: generative VLM
        # ----------------------------
        right_header = Text("Generative VLM", font_size=34, color=MAROON_B)
        right_header.move_to(RIGHT * 3.4 + UP * 2.75)

        portrait_right = self.make_portrait()
        portrait_right.move_to(RIGHT * 1.7 + UP * 1.2)

        vision_encoder = self.make_box("Vision Encoder", width=2.35, height=0.8, color=PURPLE_D)
        vision_encoder.move_to(RIGHT * 3.55 + UP * 1.2)

        llm_box = self.make_box("LLM", width=1.5, height=1.0, color=MAROON_D)
        llm_box.move_to(RIGHT * 5.35 + UP * 1.2)

        generated_panel = self.make_generated_output_box()
        generated_panel.move_to(RIGHT * 3.7 + DOWN * 1.2)

        vlm_label = Text("VLM generates text", font_size=28, color=MAROON_B)
        vlm_label.move_to(RIGHT * 3.45 + DOWN * 3.2)

        r_arrow_1 = Arrow(
            portrait_right.get_right(),
            vision_encoder.get_left(),
            buff=0.12,
            stroke_width=5,
        )
        r_arrow_2 = Arrow(
            vision_encoder.get_right(),
            llm_box.get_left(),
            buff=0.12,
            stroke_width=5,
        )
        r_arrow_3 = Arrow(
            llm_box.get_bottom(),
            generated_panel.get_top(),
            buff=0.12,
            stroke_width=5,
        )

        right_group = VGroup(
            right_header,
            portrait_right,
            vision_encoder,
            llm_box,
            generated_panel,
        )

        # top labels
        top_left_label = Text("ranking", font_size=24, color=BLUE_B)
        top_left_label.next_to(left_header, DOWN, buff=0.12)

        top_right_label = Text("free-form generation", font_size=24, color=MAROON_B)
        top_right_label.next_to(right_header, DOWN, buff=0.12)

        # bottom conclusion
        conclusion = Text(
            "Embedding comparison is easier to test consistently",
            font_size=30,
            color=YELLOW_B,
        )
        conclusion.to_edge(DOWN, buff=0.22)

        self.play(
            FadeIn(left_group, lag_ratio=0.05),
            FadeIn(right_group, lag_ratio=0.05),
            FadeIn(top_left_label, shift=DOWN * 0.2),
            FadeIn(top_right_label, shift=DOWN * 0.2),
            run_time=1.5,
        )
        self.wait(0.4)

        # Feed portrait into both sides
        self.play(
            Indicate(portrait_left, scale_factor=1.03),
            Indicate(portrait_right, scale_factor=1.03),
            run_time=0.8,
        )
        self.play(
            ShowCreation(l_arrow_1),
            ShowCreation(r_arrow_1),
            run_time=0.8,
        )
        self.play(
            image_encoder[0].animate.set_stroke(width=5).set_color(BLUE_B),
            vision_encoder[0].animate.set_stroke(width=5).set_color(PURPLE_B),
            run_time=0.8,
        )
        self.wait(0.2)

        # Left side text names into encoder
        self.play(FadeIn(candidate_names, shift=RIGHT * 0.2), run_time=0.6)
        self.play(ShowCreation(l_arrow_3), run_time=0.6)
        self.play(text_encoder[0].animate.set_stroke(width=5).set_color(TEAL_B), run_time=0.6)

        # Embeddings
        self.play(
            ShowCreation(l_arrow_2),
            ShowCreation(l_arrow_4),
            ShowCreation(r_arrow_2),
            run_time=0.8,
        )
        self.play(
            FadeIn(image_embedding, shift=RIGHT * 0.2),
            FadeIn(text_embedding, shift=RIGHT * 0.2),
            llm_box[0].animate.set_stroke(width=5).set_color(MAROON_B),
            run_time=0.9,
        )
        self.wait(0.2)

        # Similarity vs generation
        self.play(
            ShowCreation(l_arrow_5),
            ShowCreation(l_arrow_6),
            FadeIn(similarity, scale=0.9),
            run_time=0.8,
        )
        self.play(
            similarity[0].animate.set_stroke(width=5).set_color(YELLOW_B),
            FadeIn(clip_label, shift=UP * 0.2),
            FadeIn(vlm_label, shift=UP * 0.2),
            run_time=0.8,
        )

        # Ranked output list
        self.play(ShowCreation(l_arrow_7), run_time=0.6)
        self.play(FadeIn(ranked_output, shift=RIGHT * 0.25), run_time=0.8)

        scores = VGroup(
            Text("0.91", font_size=24, color=GREY_B),
            Text("0.84", font_size=24, color=GREY_B),
            Text("0.79", font_size=24, color=GREY_B),
        )
        scores.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        scores.next_to(ranked_output, RIGHT, buff=0.18).shift(UP * 0.04)
        self.play(
            LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in scores], lag_ratio=0.15),
            run_time=0.7,
        )
        self.wait(0.2)

        # Generated output sentence and instability
        gen_text_1 = Text("This person looks like a David.", font_size=28)
        gen_text_2 = Text("This might be Michael... or Daniel.", font_size=26)
        gen_text_3 = Text("Hard to say, maybe David.", font_size=28)

        for t in [gen_text_1, gen_text_2, gen_text_3]:
            t.move_to(generated_panel[1].get_center())

        self.play(ShowCreation(r_arrow_3), run_time=0.6)
        self.play(FadeIn(generated_panel, scale=0.95), FadeIn(gen_text_1), run_time=0.9)
        self.wait(0.6)

        unstable_box = SurroundingRectangle(gen_text_1, buff=0.1, color=RED, stroke_width=3)
        unstable_label = Text("changes", font_size=20, color=RED)
        unstable_label.next_to(unstable_box, UP, buff=0.08)

        self.play(
            ShowCreation(unstable_box),
            FadeIn(unstable_label, shift=UP * 0.1),
            run_time=0.5,
        )
        self.wait(0.25)

        self.play(
            TransformMatchingShapes(gen_text_1, gen_text_2),
            run_time=0.8,
        )
        unstable_box_2 = SurroundingRectangle(gen_text_2, buff=0.1, color=RED, stroke_width=3)
        self.play(Transform(unstable_box, unstable_box_2), run_time=0.4)
        self.wait(0.35)

        self.play(
            TransformMatchingShapes(gen_text_2, gen_text_3),
            run_time=0.8,
        )
        unstable_box_3 = SurroundingRectangle(gen_text_3, buff=0.1, color=RED, stroke_width=3)
        self.play(Transform(unstable_box, unstable_box_3), run_time=0.4)
        self.wait(0.4)

        # Emphasize concrete contrast
        left_emphasis = SurroundingRectangle(ranked_output, buff=0.16, color=GREEN, stroke_width=4)
        right_emphasis = SurroundingRectangle(gen_text_3, buff=0.16, color=RED, stroke_width=4)

        self.play(
            ShowCreation(left_emphasis),
            ShowCreation(right_emphasis),
            run_time=0.7,
        )
        self.wait(0.3)

        self.play(FadeIn(conclusion, shift=UP * 0.2), run_time=0.8)
        self.wait(1.2)

    # ---------------------------------------------------
    # Helper constructors
    # ---------------------------------------------------
    def make_box(self, label, width=2.2, height=0.8, color=BLUE):
        rect = RoundedRectangle(
            corner_radius=0.16,
            width=width,
            height=height,
            stroke_color=color,
            stroke_width=3,
            fill_color=color,
            fill_opacity=0.08,
        )
        text = Text(label, font_size=26)
        text.move_to(rect.get_center())
        return VGroup(rect, text)

    def make_portrait(self):
        frame = RoundedRectangle(
            corner_radius=0.12,
            width=1.5,
            height=1.9,
            stroke_color=GREY_B,
            stroke_width=2.5,
            fill_color=GREY_E,
            fill_opacity=0.35,
        )

        head = Circle(radius=0.26, stroke_width=0, fill_color=GREY_C, fill_opacity=1)
        head.move_to(frame.get_center() + UP * 0.33)

        shoulders = ArcBetweenPoints(
            frame.get_center() + LEFT * 0.42 + DOWN * 0.2,
            frame.get_center() + RIGHT * 0.42 + DOWN * 0.2,
            angle=-PI / 2,
            stroke_color=GREY_C,
            stroke_width=8,
        )

        face_lines = VGroup(
            Line(head.get_center() + LEFT * 0.1 + UP * 0.03, head.get_center() + LEFT * 0.03 + UP * 0.03),
            Line(head.get_center() + RIGHT * 0.03 + UP * 0.03, head.get_center() + RIGHT * 0.1 + UP * 0.03),
            Arc(radius=0.07, angle=-PI, start_angle=PI, stroke_width=2).move_to(head.get_center() + DOWN * 0.06),
        )
        face_lines.set_stroke(WHITE, 2)

        return VGroup(frame, head, shoulders, face_lines)

    def make_embedding_column(self, label="Embedding", color=BLUE):
        squares = VGroup()
        for _ in range(8):
            sq = Square(side_length=0.22)
            sq.set_stroke(color, 2)
            sq.set_fill(color, 0.2)
            squares.add(sq)
        squares.arrange(DOWN, buff=0.04)

        lbl = Text(label, font_size=20)
        lbl.next_to(squares, DOWN, buff=0.12)
        return VGroup(squares, lbl)

    def make_candidate_names(self):
        names = ["Jessica", "Jennifer", "Sarah", "Emily"]
        labels = VGroup()
        for name in names:
            t = Text(name, font_size=24)
            chip = RoundedRectangle(
                corner_radius=0.12,
                width=t.get_width() + 0.42,
                height=0.42,
                stroke_color=TEAL_B,
                stroke_width=2,
                fill_color=TEAL_D,
                fill_opacity=0.08,
            )
            t.move_to(chip.get_center())
            labels.add(VGroup(chip, t))
        labels.arrange(DOWN, buff=0.12, aligned_edge=LEFT)

        title = Text("candidate names", font_size=23, color=GREY_B)
        title.next_to(labels, UP, buff=0.14)
        return VGroup(title, labels)

    def make_ranked_output(self):
        lines = VGroup(
            Text("1. Jessica", font_size=28, color=GREEN_B),
            Text("2. Jennifer", font_size=28),
            Text("3. Sarah", font_size=28),
        )
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        panel = RoundedRectangle(
            corner_radius=0.14,
            width=2.55,
            height=1.65,
            stroke_color=GREEN_B,
            stroke_width=3,
            fill_color=GREEN_D,
            fill_opacity=0.08,
        )
        lines.move_to(panel.get_center())

        return VGroup(panel, lines)

    def make_generated_output_box(self):
        panel = RoundedRectangle(
            corner_radius=0.14,
            width=4.0,
            height=1.45,
            stroke_color=MAROON_B,
            stroke_width=3,
            fill_color=MAROON_D,
            fill_opacity=0.08,
        )
        return VGroup(panel, VectorizedPoint(panel.get_center()))
