from manimlib import *
import numpy as np


class CLIPExplainer(Scene):
    def construct(self):
        self.set_background_color(BLACK)

        # ── Colors ──
        IMG_COLOR = BLUE
        TXT_COLOR = GREEN
        EMBED_COLOR = YELLOW
        PROJ_COLOR = PURPLE

        # ================================================================
        # 1. Title
        # ================================================================
        title = Tex(R"\text{CLIP: Contrastive Language--Image Pre-training}", font_size=40)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)

        # ================================================================
        # 2. Show image placeholder (left) and text (right)
        # ================================================================
        # --- Image: a 4x4 grid of colored squares ---
        patch_colors = [
            BLUE_E, BLUE_D, BLUE_C, TEAL_E,
            BLUE_D, TEAL_D, TEAL_C, GREEN_E,
            TEAL_E, TEAL_D, GREEN_D, GREEN_C,
            GREEN_E, GREEN_D, GREEN_C, YELLOW_E,
        ]
        patch_size = 0.45
        image_grid = VGroup()
        for i in range(4):
            for j in range(4):
                sq = Square(side_length=patch_size)
                sq.set_fill(patch_colors[i * 4 + j], opacity=1)
                sq.set_stroke(WHITE, 0.5)
                sq.move_to(np.array([j * patch_size, -i * patch_size, 0]))
                image_grid.add(sq)
        image_grid.center()
        image_grid.move_to(LEFT * 4.5 + DOWN * 0.3)

        img_label = Tex(R"\text{Image}", font_size=28, color=IMG_COLOR)
        img_label.next_to(image_grid, UP, buff=0.25)

        # --- Text sentence ---
        sentence = Tex(R"\text{A photo of a cat}", font_size=30, color=TXT_COLOR)
        sentence.move_to(RIGHT * 4.5 + DOWN * 0.3)

        txt_label = Tex(R"\text{Text}", font_size=28, color=TXT_COLOR)
        txt_label.next_to(sentence, UP, buff=0.25)

        self.play(
            FadeIn(image_grid, shift=UP * 0.3),
            FadeIn(img_label),
            FadeIn(sentence, shift=UP * 0.3),
            FadeIn(txt_label),
            run_time=1.2,
        )
        self.wait(0.5)

        # ================================================================
        # 3. Image → Patches  (highlight grid lines)
        # ================================================================
        patch_border = SurroundingRectangle(image_grid, buff=0.02, color=WHITE)
        patch_border.set_stroke(width=2)

        # Draw lines to emphasize patches
        grid_lines = VGroup()
        left_x = image_grid[0].get_left()[0]
        right_x = image_grid[3].get_right()[0]
        top_y = image_grid[0].get_top()[1]
        bot_y = image_grid[12].get_bottom()[1]
        for k in range(1, 4):
            y = top_y - k * patch_size
            grid_lines.add(Line(
                np.array([left_x, y, 0]),
                np.array([right_x, y, 0]),
                stroke_color=WHITE, stroke_width=2,
            ))
            x = left_x + k * patch_size
            grid_lines.add(Line(
                np.array([x, top_y, 0]),
                np.array([x, bot_y, 0]),
                stroke_color=WHITE, stroke_width=2,
            ))

        patches_label = Tex(R"\text{Split into patches}", font_size=22, color=GREY_A)
        patches_label.next_to(image_grid, DOWN, buff=0.3)

        self.play(
            ShowCreation(patch_border),
            ShowCreation(grid_lines, lag_ratio=0.1),
            FadeIn(patches_label, shift=DOWN * 0.2),
            run_time=1.2,
        )
        self.wait(0.5)

        # ================================================================
        # 4. Text → Tokens  (highlight individual words)
        # ================================================================
        tokens_text = ["A", "photo", "of", "a", "cat"]
        token_boxes = VGroup()
        for i, word in enumerate(tokens_text):
            t = Tex(R"\text{" + word + "}", font_size=26, color=TXT_COLOR)
            box = SurroundingRectangle(t, buff=0.08, color=TXT_COLOR)
            box.set_stroke(width=1.5)
            token_boxes.add(VGroup(box, t))
        token_boxes.arrange(RIGHT, buff=0.15)
        token_boxes.move_to(sentence.get_center())

        tokens_label = Tex(R"\text{Tokenize}", font_size=22, color=GREY_A)
        tokens_label.next_to(token_boxes, DOWN, buff=0.3)

        self.play(
            FadeOut(sentence),
            LaggedStartMap(FadeIn, token_boxes, lag_ratio=0.15),
            FadeIn(tokens_label, shift=DOWN * 0.2),
            run_time=1.2,
        )
        self.wait(0.5)

        # ================================================================
        # 5. Shrink inputs and move up to make room for encoders
        # ================================================================
        self.play(
            FadeOut(title),
            FadeOut(patches_label),
            FadeOut(tokens_label),
            run_time=0.6,
        )

        # Group image side and text side
        img_group = VGroup(image_grid, patch_border, grid_lines, img_label)
        txt_group = VGroup(token_boxes, txt_label)

        self.play(
            img_group.animate.scale(0.7).move_to(LEFT * 5 + UP * 2.8),
            txt_group.animate.scale(0.7).move_to(RIGHT * 5 + UP * 2.8),
            run_time=1,
        )
        self.wait(0.3)

        # ================================================================
        # 6. Patch embeddings (image side)
        # ================================================================
        num_patches = 16
        patch_emb_rects = VGroup()
        for i in range(num_patches):
            r = Rectangle(width=0.6, height=0.15)
            r.set_fill(patch_colors[i], opacity=0.8)
            r.set_stroke(WHITE, 0.5)
            patch_emb_rects.add(r)
        patch_emb_rects.arrange(DOWN, buff=0.02)
        patch_emb_rects.set_height(2.2)
        patch_emb_rects.move_to(LEFT * 5 + DOWN * 0.2)

        emb_label_img = VGroup(
            Tex(R"\text{Patch}", font_size=18, color=GREY_A),
            Tex(R"\text{Embeddings}", font_size=18, color=GREY_A),
        )
        emb_label_img.arrange(DOWN, buff=0.05)
        emb_label_img.next_to(patch_emb_rects, LEFT, buff=0.2)

        arrow_img_to_emb = Arrow(
            img_group.get_bottom() + DOWN * 0.1,
            patch_emb_rects.get_top() + UP * 0.1,
            buff=0.1, stroke_color=IMG_COLOR, stroke_width=3,
        )

        self.play(
            ShowCreation(arrow_img_to_emb),
            run_time=0.6,
        )
        self.play(
            LaggedStartMap(FadeIn, patch_emb_rects, shift=DOWN * 0.1, lag_ratio=0.05),
            FadeIn(emb_label_img),
            run_time=1,
        )
        self.wait(0.3)

        # ================================================================
        # 7. Token embeddings (text side)
        # ================================================================
        tok_emb_rects = VGroup()
        tok_colors = [GREEN_E, GREEN_D, GREEN_C, GREEN_B, GREEN_A]
        for i in range(5):
            r = Rectangle(width=0.6, height=0.3)
            r.set_fill(tok_colors[i], opacity=0.8)
            r.set_stroke(WHITE, 0.5)
            tok_emb_rects.add(r)
        tok_emb_rects.arrange(DOWN, buff=0.04)
        tok_emb_rects.set_height(2.2)
        tok_emb_rects.move_to(RIGHT * 5 + DOWN * 0.2)

        emb_label_txt = VGroup(
            Tex(R"\text{Token}", font_size=18, color=GREY_A),
            Tex(R"\text{Embeddings}", font_size=18, color=GREY_A),
        )
        emb_label_txt.arrange(DOWN, buff=0.05)
        emb_label_txt.next_to(tok_emb_rects, RIGHT, buff=0.2)

        arrow_txt_to_emb = Arrow(
            txt_group.get_bottom() + DOWN * 0.1,
            tok_emb_rects.get_top() + UP * 0.1,
            buff=0.1, stroke_color=TXT_COLOR, stroke_width=3,
        )

        self.play(
            ShowCreation(arrow_txt_to_emb),
            run_time=0.6,
        )
        self.play(
            LaggedStartMap(FadeIn, tok_emb_rects, shift=DOWN * 0.1, lag_ratio=0.1),
            FadeIn(emb_label_txt),
            run_time=1,
        )
        self.wait(0.3)

        # ================================================================
        # 8. Encoder blocks
        # ================================================================
        img_encoder = RoundedRectangle(
            width=2.2, height=0.8, corner_radius=0.15,
        )
        img_encoder.set_fill(BLUE_E, opacity=0.6)
        img_encoder.set_stroke(IMG_COLOR, 2)
        img_encoder.next_to(patch_emb_rects, DOWN, buff=0.6)
        img_enc_text = Tex(R"\text{Image Encoder (ViT)}", font_size=16, color=WHITE)
        img_enc_text.move_to(img_encoder)

        txt_encoder = RoundedRectangle(
            width=2.2, height=0.8, corner_radius=0.15,
        )
        txt_encoder.set_fill(GREEN_E, opacity=0.6)
        txt_encoder.set_stroke(TXT_COLOR, 2)
        txt_encoder.next_to(tok_emb_rects, DOWN, buff=0.6)
        txt_enc_text = Tex(R"\text{Text Encoder}", font_size=16, color=WHITE)
        txt_enc_text.move_to(txt_encoder)

        arrow_emb_to_enc_img = Arrow(
            patch_emb_rects.get_bottom(),
            img_encoder.get_top(),
            buff=0.1, stroke_color=IMG_COLOR, stroke_width=3,
        )
        arrow_emb_to_enc_txt = Arrow(
            tok_emb_rects.get_bottom(),
            txt_encoder.get_top(),
            buff=0.1, stroke_color=TXT_COLOR, stroke_width=3,
        )

        self.play(
            ShowCreation(arrow_emb_to_enc_img),
            ShowCreation(arrow_emb_to_enc_txt),
            run_time=0.6,
        )
        self.play(
            FadeIn(img_encoder),
            FadeIn(img_enc_text),
            FadeIn(txt_encoder),
            FadeIn(txt_enc_text),
            run_time=0.8,
        )
        self.wait(0.5)

        # ================================================================
        # 9. Projection into shared embedding space
        # ================================================================
        shared_label = Tex(
            R"\text{Joint Embedding Space}", font_size=28, color=EMBED_COLOR,
        )
        shared_label.move_to(DOWN * 3.3)

        # Image embedding vector (projected) — use Tex for math notation
        img_emb_dot = Dot(color=IMG_COLOR, radius=0.12)
        img_emb_dot.move_to(LEFT * 0.8 + DOWN * 2.5)
        img_emb_label = Tex(R"\vec{e}_I", font_size=28, color=IMG_COLOR)
        img_emb_label.next_to(img_emb_dot, UP, buff=0.15)

        # Text embedding vector (projected)
        txt_emb_dot = Dot(color=TXT_COLOR, radius=0.12)
        txt_emb_dot.move_to(RIGHT * 0.8 + DOWN * 2.5)
        txt_emb_label = Tex(R"\vec{e}_T", font_size=28, color=TXT_COLOR)
        txt_emb_label.next_to(txt_emb_dot, UP, buff=0.15)

        # Surrounding ellipse for the space
        space_ellipse = Ellipse(width=5, height=1.8)
        space_ellipse.set_stroke(EMBED_COLOR, 1.5, opacity=0.5)
        space_ellipse.move_to(DOWN * 2.5)

        # Arrows from encoders to embedding space
        arrow_enc_img_to_space = Arrow(
            img_encoder.get_bottom(),
            img_emb_dot.get_center() + UP * 0.2 + LEFT * 0.5,
            buff=0.15, stroke_color=PROJ_COLOR, stroke_width=3,
        )
        arrow_enc_txt_to_space = Arrow(
            txt_encoder.get_bottom(),
            txt_emb_dot.get_center() + UP * 0.2 + RIGHT * 0.5,
            buff=0.15, stroke_color=PROJ_COLOR, stroke_width=3,
        )

        proj_label_img = Tex(R"\text{project}", font_size=16, color=PROJ_COLOR)
        proj_label_img.set_backstroke(width=3)
        proj_label_img.next_to(arrow_enc_img_to_space, LEFT, buff=0.1)
        proj_label_txt = Tex(R"\text{project}", font_size=16, color=PROJ_COLOR)
        proj_label_txt.set_backstroke(width=3)
        proj_label_txt.next_to(arrow_enc_txt_to_space, RIGHT, buff=0.1)

        self.play(
            ShowCreation(arrow_enc_img_to_space),
            ShowCreation(arrow_enc_txt_to_space),
            FadeIn(proj_label_img),
            FadeIn(proj_label_txt),
            run_time=1,
        )
        self.play(
            FadeIn(space_ellipse),
            FadeIn(shared_label, shift=UP * 0.2),
            run_time=0.8,
        )
        self.play(
            FadeIn(img_emb_dot, scale=0.3),
            FadeIn(img_emb_label),
            FadeIn(txt_emb_dot, scale=0.3),
            FadeIn(txt_emb_label),
            run_time=0.8,
        )
        self.wait(0.5)

        # ================================================================
        # 10. Show cosine similarity — dots move close together
        # ================================================================
        sim_line = DashedLine(
            img_emb_dot.get_center(),
            txt_emb_dot.get_center(),
            stroke_color=YELLOW, stroke_width=2,
        )
        sim_label = Tex(R"\text{cosine similarity}", font_size=22, color=YELLOW)
        sim_label.set_backstroke(width=3)
        sim_label.next_to(sim_line, DOWN, buff=0.15)

        self.play(
            ShowCreation(sim_line),
            FadeIn(sim_label),
            run_time=0.8,
        )
        self.wait(0.3)

        # Animate dots moving closer (matching pair → high similarity)
        target_center = (img_emb_dot.get_center() + txt_emb_dot.get_center()) / 2
        offset = np.array([0.15, 0, 0])

        new_sim_line = DashedLine(
            target_center - offset,
            target_center + offset,
            stroke_color=YELLOW, stroke_width=2,
        )

        self.play(
            img_emb_dot.animate.move_to(target_center - offset),
            img_emb_label.animate.move_to(target_center - offset + UP * 0.3),
            txt_emb_dot.animate.move_to(target_center + offset),
            txt_emb_label.animate.move_to(target_center + offset + UP * 0.3),
            Transform(sim_line, new_sim_line),
            run_time=1.5,
        )

        match_text = Tex(
            R"\text{Matching pair} \rightarrow \text{High similarity}",
            font_size=28, color=YELLOW,
        )
        match_text.set_backstroke(width=4)
        match_text.move_to(DOWN * 3.3)

        self.play(
            FadeOut(shared_label),
            FadeIn(match_text, shift=UP * 0.2),
            run_time=0.8,
        )
        self.wait(2)
        self.embed()
