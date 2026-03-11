from manimlib import *
import numpy as np


class CLIPEncoding(Scene):
    def construct(self):
        tok_colors = [BLUE_C, GREEN_C, TEAL_C, BLUE_B, YELLOW_C]
        patch_colors = [
            BLUE_E, BLUE_D, BLUE_C, TEAL_E,
            BLUE_D, TEAL_D, TEAL_C, GREEN_E,
            TEAL_E, TEAL_D, GREEN_D, GREEN_C,
            GREEN_E, GREEN_D, GREEN_C, YELLOW_E,
        ]
        LX, RX = -4.0, 4.0

        def make_encoder(label, color, pos):
            layers = VGroup()
            for i in range(6):
                r = Rectangle(width=2.2, height=0.22)
                r.set_fill(color, opacity=0.25 + 0.1 * i)
                r.set_stroke(interpolate_color(color, WHITE, 0.3), 0.8)
                layers.add(r)
            layers.arrange(DOWN, buff=0.05)
            box = SurroundingRectangle(layers, buff=0.2, color=color)
            box.set_stroke(width=2.5)
            box.set_fill(BLACK, opacity=0.6)
            lbl = Tex(R"\text{" + label + "}", font_size=22, color=WHITE)
            lbl.next_to(box, UP, buff=0.15)
            grp = VGroup(box, layers, lbl)
            grp.move_to(pos)
            return grp, box

        def make_embedding(colors, pos, h=0.2, w=1.2):
            bars = VGroup()
            for c in colors:
                r = Rectangle(width=w, height=h)
                r.set_fill(c, opacity=0.85)
                r.set_stroke(WHITE, 0.5)
                bars.add(r)
            bars.arrange(DOWN, buff=0.04)
            bars.move_to(pos)
            return bars

        # ================================================================
        # STAGE 1 — Text input on the left, image on the right
        # ================================================================
        sentence = Tex(
            R"\text{a photo of a dog}",
            font_size=40, color=GREEN_C,
        )
        sentence.move_to([LX, 3.0, 0])

        ps = 0.42
        image_grid = VGroup()
        for i in range(4):
            for j in range(4):
                sq = Square(side_length=ps)
                sq.set_fill(patch_colors[i * 4 + j], opacity=1)
                sq.set_stroke(WHITE, 0.5)
                sq.move_to(np.array([j * ps, -i * ps, 0]))
                image_grid.add(sq)
        image_grid.center()
        image_grid.move_to([RX, 3.0, 0])

        img_lbl = Tex(R"\text{Image}", font_size=22, color=GREY_B)
        img_lbl.next_to(image_grid, UP, buff=0.25)
        txt_lbl = Tex(R"\text{Text}", font_size=22, color=GREY_B)
        txt_lbl.next_to(sentence, UP, buff=0.25)

        self.play(
            Write(sentence), FadeIn(txt_lbl),
            FadeIn(image_grid, shift=UP * 0.2), FadeIn(img_lbl),
            run_time=1.5,
        )
        self.wait(0.8)

        # ================================================================
        # STAGE 2 — Sentence splits into coloured word tokens
        # ================================================================
        self.play(FadeOut(txt_lbl), FadeOut(img_lbl), run_time=0.3)

        words = ["a", "photo", "of", "a", "dog"]
        toks = VGroup()
        for w, c in zip(words, tok_colors):
            t = Tex(R"\text{" + w + "}", font_size=30, color=WHITE)
            bx = SurroundingRectangle(t, buff=0.1, color=c)
            bx.set_fill(c, opacity=0.25)
            bx.set_stroke(width=2)
            toks.add(VGroup(bx, t))
        toks.arrange(RIGHT, buff=0.12)
        toks.move_to(sentence.get_center())

        self.play(
            FadeOut(sentence),
            LaggedStart(
                *[FadeIn(tk, shift=DOWN * 0.1) for tk in toks],
                lag_ratio=0.12,
            ),
            run_time=1.0,
        )
        self.wait(0.4)

        # ================================================================
        # STAGE 3 — Tokens pass through Text Encoder → text embedding
        # ================================================================
        te_grp, te_box = make_encoder("Text Encoder", BLUE_D, [LX, 0.7, 0])
        self.play(FadeIn(te_grp, shift=UP * 0.15), run_time=0.7)
        self.wait(0.2)

        entry_t = te_box.get_top() + DOWN * 0.15
        self.play(
            LaggedStart(
                *[tk.animate.move_to(entry_t).scale(0.2).set_opacity(0)
                  for tk in toks],
                lag_ratio=0.12,
            ),
            run_time=1.4,
        )
        self.remove(toks)

        te_colors = [BLUE_D, BLUE_C, TEAL_D, TEAL_C, GREEN_D]
        txt_emb = make_embedding(te_colors, [LX, -1.4, 0])

        arr_t = Arrow(
            te_box.get_bottom(), txt_emb.get_top(),
            buff=0.1, stroke_color=BLUE_C, thickness=2,
        )
        te_lbl = Tex(R"\text{Text Embedding}", font_size=18, color=GREY_A)
        te_lbl.next_to(txt_emb, DOWN, buff=0.2)

        self.play(
            GrowArrow(arr_t),
            LaggedStart(
                *[FadeIn(r, shift=DOWN * 0.15) for r in txt_emb],
                lag_ratio=0.08,
            ),
            run_time=1.0,
        )
        self.play(FadeIn(te_lbl), run_time=0.3)
        self.wait(0.5)

        # ================================================================
        # STAGE 4 — Grid overlay on image, patches into Image Encoder
        # ================================================================
        grid_lines = VGroup()
        le = image_grid[0].get_left()[0]
        re_ = image_grid[3].get_right()[0]
        tp = image_grid[0].get_top()[1]
        bt = image_grid[12].get_bottom()[1]
        for k in range(1, 4):
            y = tp - k * ps
            grid_lines.add(Line(
                [le, y, 0], [re_, y, 0],
                stroke_color=WHITE, stroke_width=2,
            ))
            x = le + k * ps
            grid_lines.add(Line(
                [x, tp, 0], [x, bt, 0],
                stroke_color=WHITE, stroke_width=2,
            ))

        spl = Tex(R"\text{Split into patches}", font_size=18, color=GREY_A)
        spl.next_to(image_grid, DOWN, buff=0.25)

        self.play(
            ShowCreation(grid_lines, lag_ratio=0.06),
            FadeIn(spl, shift=DOWN * 0.1),
            run_time=0.8,
        )
        self.wait(0.3)
        self.play(FadeOut(spl), run_time=0.3)

        ie_grp, ie_box = make_encoder("Image Encoder", TEAL_D, [RX, 0.7, 0])
        self.play(FadeIn(ie_grp, shift=UP * 0.15), run_time=0.7)
        self.wait(0.2)

        entry_i = ie_box.get_top() + DOWN * 0.15
        self.play(
            LaggedStart(
                *[sq.animate.move_to(entry_i).scale(0.2).set_opacity(0)
                  for sq in image_grid],
                lag_ratio=0.04,
            ),
            FadeOut(grid_lines),
            run_time=2.0,
        )
        self.remove(image_grid, grid_lines)

        ie_colors = [TEAL_E, TEAL_D, TEAL_C, GREEN_D, GREEN_C]
        img_emb = make_embedding(ie_colors, [RX, -1.4, 0])

        arr_i = Arrow(
            ie_box.get_bottom(), img_emb.get_top(),
            buff=0.1, stroke_color=TEAL_C, thickness=2,
        )
        ie_lbl = Tex(R"\text{Image Embedding}", font_size=18, color=GREY_A)
        ie_lbl.next_to(img_emb, DOWN, buff=0.2)

        self.play(
            GrowArrow(arr_i),
            LaggedStart(
                *[FadeIn(r, shift=DOWN * 0.15) for r in img_emb],
                lag_ratio=0.08,
            ),
            run_time=1.0,
        )
        self.play(FadeIn(ie_lbl), run_time=0.3)
        self.wait(0.8)

        # ================================================================
        # STAGE 5 — Both embeddings travel to the shared embedding space
        # ================================================================
        self.play(
            FadeOut(te_grp), FadeOut(ie_grp),
            FadeOut(arr_t), FadeOut(arr_i),
            FadeOut(te_lbl), FadeOut(ie_lbl),
            run_time=0.6,
        )

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            width=6, height=5,
            axis_config=dict(stroke_color=GREY_C, stroke_width=1.2),
        )
        axes.move_to(DOWN * 0.3)

        bg = Rectangle(width=6.3, height=5.3)
        bg.set_fill("#0d1a2a", opacity=0.5)
        bg.set_stroke(BLUE_E, width=1.5)
        bg.move_to(axes.get_center())

        sp_lbl = Tex(
            R"\text{Shared Embedding Space}",
            font_size=26, color=BLUE_B,
        )
        sp_lbl.next_to(bg, UP, buff=0.25)

        self.play(
            FadeIn(bg), ShowCreation(axes), FadeIn(sp_lbl),
            run_time=1.2,
        )

        center_left = axes.c2p(-0.5, 0)
        center_right = axes.c2p(0.5, 0)
        self.play(
            txt_emb.animate.move_to(center_left),
            img_emb.animate.move_to(center_right),
            run_time=1.0,
        )

        txt_pt = axes.c2p(1.0, 1.4)
        img_pt = axes.c2p(1.4, 1.0)

        txt_dot = Dot(txt_pt, color=BLUE_C, radius=0.12)
        img_dot = Dot(img_pt, color=TEAL_C, radius=0.12)

        txt_glow = Circle(radius=0.28, color=BLUE_C)
        txt_glow.set_stroke(width=3, opacity=0.4)
        txt_glow.set_fill(BLUE_C, opacity=0.12)
        txt_glow.move_to(txt_pt)

        img_glow = Circle(radius=0.28, color=TEAL_C)
        img_glow.set_stroke(width=3, opacity=0.4)
        img_glow.set_fill(TEAL_C, opacity=0.12)
        img_glow.move_to(img_pt)

        self.play(
            ReplacementTransform(txt_emb, txt_dot),
            ReplacementTransform(img_emb, img_dot),
            run_time=1.5,
        )
        self.play(FadeIn(txt_glow), FadeIn(img_glow), run_time=0.5)

        td_lbl = Tex(
            R'\text{"a photo of a dog"}',
            font_size=16, color=BLUE_B,
        )
        td_lbl.next_to(txt_dot, UP + LEFT, buff=0.15)

        id_lbl = Tex(
            R"\text{dog image}",
            font_size=16, color=TEAL_B,
        )
        id_lbl.next_to(img_dot, DOWN + RIGHT, buff=0.15)

        self.play(FadeIn(td_lbl), FadeIn(id_lbl), run_time=0.5)

        # ================================================================
        # Cosine similarity line
        # ================================================================
        sim_line = DashedLine(
            txt_pt, img_pt,
            dash_length=0.08, stroke_color=YELLOW_C,
        )
        mid = (np.array(txt_pt) + np.array(img_pt)) / 2
        sim_lbl = Tex(
            R"\text{cosine similarity}",
            font_size=16, color=YELLOW_C,
        )
        sim_lbl.move_to(mid + np.array([1.2, 0.3, 0]))

        self.play(ShowCreation(sim_line), FadeIn(sim_lbl), run_time=0.8)
        self.wait(1.0)

        # ================================================================
        # STAGE 6 — Multiple text-image pairs populate the space
        # ================================================================
        pair_data = [
            (R'\text{"a red car"}',  RED_C,    -1.8, -1.5,
             R"\text{car image}",    RED_D,    -1.5, -1.8),
            (R'\text{"sunset"}',     ORANGE,   -2.0,  1.2,
             R"\text{sunset image}", GOLD_D,   -1.7,  0.8),
            (R'\text{"a cute cat"}', PURPLE_C,  0.8, -1.8,
             R"\text{cat image}",    PURPLE_D,  1.1, -2.1),
            (R'\text{"blue ocean"}', BLUE_E,    2.2,  0.3,
             R"\text{ocean image}",  BLUE_D,    2.5,  0.0),
        ]

        new_mobs = VGroup()
        for ts, tc, tx, ty, is_, ic, ix, iy in pair_data:
            t_p = axes.c2p(tx, ty)
            i_p = axes.c2p(ix, iy)

            td = Dot(t_p, color=tc, radius=0.08)
            id_ = Dot(i_p, color=ic, radius=0.08)
            dl = DashedLine(
                t_p, i_p,
                dash_length=0.05, stroke_color=GREY_B, stroke_width=1,
            )
            tl = Tex(ts, font_size=13, color=tc)
            tl.next_to(td, UP, buff=0.08)
            il = Tex(is_, font_size=13, color=ic)
            il.next_to(id_, DOWN, buff=0.08)

            new_mobs.add(td, id_, dl, tl, il)

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in new_mobs],
                lag_ratio=0.06,
            ),
            run_time=3.0,
        )
        self.wait(3)
