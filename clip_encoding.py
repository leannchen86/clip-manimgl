from manimlib import *
import numpy as np


class CLIPEncoding(Scene):
    def construct(self):
        self.set_background_color(BLACK)

        patch_colors = [
            BLUE_E, BLUE_D, BLUE_C, TEAL_E,
            BLUE_D, TEAL_D, TEAL_C, GREEN_E,
            TEAL_E, TEAL_D, GREEN_D, GREEN_C,
            GREEN_E, GREEN_D, GREEN_C, YELLOW_E,
        ]
        tok_colors = [GREEN_E, GREEN_D, GREEN_C, GREEN_B, GREEN_A]

        # ================================================================
        # STAGE 1 — Show text–image pair
        # ================================================================
        patch_size = 0.5
        image_grid = VGroup()
        for i in range(4):
            for j in range(4):
                sq = Square(side_length=patch_size)
                sq.set_fill(patch_colors[i * 4 + j], opacity=1)
                sq.set_stroke(WHITE, 0.8)
                sq.move_to(np.array([j * patch_size, -i * patch_size, 0]))
                image_grid.add(sq)
        image_grid.center()
        image_grid.move_to(LEFT * 5)

        sentence = Tex(R"\text{a photo of a cat}", font_size=44, color=GREEN_C)
        sentence.move_to(RIGHT * 4)

        img_lbl = Tex(R"\text{Image}", font_size=26, color=GREY_B)
        img_lbl.next_to(image_grid, UP, buff=0.3)
        txt_lbl = Tex(R"\text{Text}", font_size=26, color=GREY_B)
        txt_lbl.next_to(sentence, UP, buff=0.3)

        self.play(
            FadeIn(image_grid, shift=UP * 0.3),
            FadeIn(img_lbl),
            Write(sentence),
            FadeIn(txt_lbl),
            run_time=1.5,
        )
        self.wait(1)

        # ================================================================
        # STAGE 2 — Text → token boxes → text embeddings
        # ================================================================
        self.play(FadeOut(img_lbl), FadeOut(txt_lbl), run_time=0.4)

        # Build token boxes vertically at the sentence's position
        words = ["a", "photo", "of", "a", "cat"]
        token_mobs = VGroup()
        for word, col in zip(words, tok_colors):
            t = Tex(R"\text{" + word + "}", font_size=34, color=col)
            box = SurroundingRectangle(t, buff=0.1, color=col)
            box.set_stroke(width=1.5)
            token_mobs.add(VGroup(box, t))
        token_mobs.arrange(DOWN, buff=0.12)
        token_mobs.move_to(sentence.get_center())

        # Sentence morphs into a column of tokens
        self.play(ReplacementTransform(sentence, token_mobs), run_time=1.2)
        self.wait(0.4)

        # Build text embedding targets to the left of the tokens
        txt_emb_rects = VGroup()
        for col in tok_colors:
            r = Rectangle(width=1.5, height=0.22)
            r.set_fill(col, opacity=0.9)
            r.set_stroke(WHITE, 0.5)
            txt_emb_rects.add(r)
        txt_emb_rects.arrange(DOWN, buff=0.07)
        txt_emb_rects.move_to(RIGHT * 1.8)

        # Each token morphs into its embedding bar
        self.play(
            LaggedStart(
                *[ReplacementTransform(tok.copy(), emb)
                  for tok, emb in zip(token_mobs, txt_emb_rects)],
                lag_ratio=0.25,
            ),
            run_time=1.8,
        )
        self.play(FadeOut(token_mobs), run_time=0.5)

        txt_emb_lbl = Tex(R"\text{Text Embeddings}", font_size=22, color=GREY_A)
        txt_emb_lbl.next_to(txt_emb_rects, UP, buff=0.25)
        self.play(FadeIn(txt_emb_lbl), run_time=0.5)
        self.wait(0.8)

        # ================================================================
        # STAGE 3 — Image → patches → patch embeddings
        # ================================================================
        # Reveal patch grid lines over the image
        grid_lines = VGroup()
        left_x  = image_grid[0].get_left()[0]
        right_x = image_grid[3].get_right()[0]
        top_y   = image_grid[0].get_top()[1]
        bot_y   = image_grid[12].get_bottom()[1]
        for k in range(1, 4):
            y = top_y - k * patch_size
            grid_lines.add(Line(
                [left_x, y, 0], [right_x, y, 0],
                stroke_color=WHITE, stroke_width=2.5,
            ))
            x = left_x + k * patch_size
            grid_lines.add(Line(
                [x, top_y, 0], [x, bot_y, 0],
                stroke_color=WHITE, stroke_width=2.5,
            ))

        patch_lbl = Tex(R"\text{Split into patches}", font_size=22, color=GREY_A)
        patch_lbl.next_to(image_grid, DOWN, buff=0.3)

        self.play(
            ShowCreation(grid_lines, lag_ratio=0.05),
            FadeIn(patch_lbl, shift=DOWN * 0.15),
            run_time=0.9,
        )
        self.wait(0.5)
        self.play(FadeOut(patch_lbl), run_time=0.4)

        # Build patch embedding targets
        patch_emb_rects = VGroup()
        for col in patch_colors:
            r = Rectangle(width=1.5, height=0.135)
            r.set_fill(col, opacity=0.9)
            r.set_stroke(WHITE, 0.4)
            patch_emb_rects.add(r)
        patch_emb_rects.arrange(DOWN, buff=0.03)
        patch_emb_rects.move_to(LEFT * 1.8)

        # Each patch morphs into its embedding bar
        self.play(
            LaggedStart(
                *[ReplacementTransform(image_grid[i].copy(), patch_emb_rects[i])
                  for i in range(16)],
                lag_ratio=0.06,
            ),
            run_time=2.5,
        )

        patch_emb_lbl = Tex(R"\text{Patch Embeddings}", font_size=22, color=GREY_A)
        patch_emb_lbl.next_to(patch_emb_rects, UP, buff=0.25)
        self.play(FadeIn(patch_emb_lbl), run_time=0.5)
        self.wait(2)
        self.embed()
