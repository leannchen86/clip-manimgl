from manimlib import *
import numpy as np
import random

class CLIPSharedEmbeddingSpace(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        # Use a TeX-like font via Pango (no LaTeX install needed)
        manim_config.text.font = "CMU Serif"

        # -----------------------------
        # Style helpers
        # -----------------------------
        def make_encoder(title, w=2.2, h=1.6, layers=6):
            # stacked layers (Transformer-ish)
            stack = VGroup()
            for i in range(layers):
                r = RoundedRectangle(width=w, height=h, corner_radius=0.12)
                r.set_fill(GREY_E, opacity=0.35)
                r.set_stroke(GREY_B, width=1.3)
                r.shift(UP * (0.03 * i) + RIGHT * (0.03 * i))
                stack.add(r)
            label = Text(title, font_size=24, color=GREY_A)
            label.move_to(stack.get_center())
            box = VGroup(stack, label)
            return box

        def make_token_block(word, color, font_size=30):
            t = Text(word, font_size=font_size, color=color)
            box = SurroundingRectangle(t, buff=0.12, color=color)
            box.set_stroke(width=1.6)
            box.set_fill(color, opacity=0.15)
            return VGroup(box, t)

        def make_number_vector(n=6, buff=0.12, offset=0, pad=0.18):
            # Vertical vector: numbers in a column with bracket-like top/bottom (single math entity)
            # offset 用來讓兩邊向量數字不同（e.g. text 用 0，image 用非 0）
            nums = VGroup()
            for i in range(n):
                val = (i * 17 + offset * 13) % 100 / 100.0
                if i % 2 == 1:
                    val = -val
                num = Text(f"{val:.2f}", font_size=22, color=WHITE)
                nums.add(num)
            nums.arrange(DOWN, buff=buff)
            nums.move_to(ORIGIN)
            w = nums.get_width() + pad
            tick = 0.08
            top_y = nums.get_top()[1] + 0.08
            bot_y = nums.get_bottom()[1] - 0.08
            left_x, right_x = -w / 2, w / 2
            # 上下括號對調：上方用下括號樣式（豎線向下），下方用上括號樣式（豎線向上）
            top_line = Line([left_x, top_y, 0], [right_x, top_y, 0], stroke_width=1.8, color=WHITE)
            top_left_tick = Line([left_x, top_y, 0], [left_x, top_y - tick, 0], stroke_width=1.8, color=WHITE)
            top_right_tick = Line([right_x, top_y, 0], [right_x, top_y - tick, 0], stroke_width=1.8, color=WHITE)
            bot_line = Line([left_x, bot_y, 0], [right_x, bot_y, 0], stroke_width=1.8, color=WHITE)
            bot_left_tick = Line([left_x, bot_y, 0], [left_x, bot_y + tick, 0], stroke_width=1.8, color=WHITE)
            bot_right_tick = Line([right_x, bot_y, 0], [right_x, bot_y + tick, 0], stroke_width=1.8, color=WHITE)
            brackets = VGroup(top_line, top_left_tick, top_right_tick, bot_line, bot_left_tick, bot_right_tick)
            vec = VGroup(brackets, nums)
            return vec

        def make_image_grid(n=8, patch_size=0.28):
            # lightweight "image" placeholder: colored grid
            # (14x14 is heavy; this keeps it smooth while still conveying the idea)
            palette = [BLUE_E, BLUE_D, BLUE_C, TEAL_E, TEAL_D, TEAL_C, GREEN_E, GREEN_D, GREEN_C, YELLOW_E]
            grid = VGroup()
            for i in range(n):
                for j in range(n):
                    sq = Square(side_length=patch_size)
                    sq.set_fill(random.choice(palette), opacity=1)
                    sq.set_stroke(BLACK, 0.0)
                    sq.move_to(np.array([j * patch_size, -i * patch_size, 0]))
                    grid.add(sq)
            grid.center()
            frame = SurroundingRectangle(grid, buff=0.05, color=GREY_B)
            frame.set_stroke(width=1.2)
            return VGroup(grid, frame)

        def make_grid_overlay_for(grid_group, n=8, patch_size=0.28):
            # overlay lines on top of the image grid
            grid = grid_group[0]
            left_x = grid[0].get_left()[0]
            right_x = grid[n-1].get_right()[0]
            top_y = grid[0].get_top()[1]
            bot_y = grid[n*(n-1)].get_bottom()[1]

            lines = VGroup()
            for k in range(1, n):
                y = top_y - k * patch_size
                lines.add(Line([left_x, y, 0], [right_x, y, 0], stroke_color=WHITE, stroke_width=1.6))
                x = left_x + k * patch_size
                lines.add(Line([x, top_y, 0], [x, bot_y, 0], stroke_color=WHITE, stroke_width=1.6))
            return lines

        def make_glow_dot(color=BLUE_C, r=0.08):
            core = Dot(radius=r)
            core.set_fill(color, opacity=1)
            halo = Dot(radius=r*2.4)
            halo.set_fill(color, opacity=0.18)
            halo.set_stroke(width=0)
            return VGroup(halo, core)

        # -----------------------------
        # Layout: left and right paths (no track lines)
        # -----------------------------
        left_track_x  = -6.0
        right_track_x =  6.0

        # Titles for text / image paths
        left_title = Text("Text path", font_size=40, color=GREY_B)
        right_title = Text("Image path", font_size=40, color=GREY_B)
        left_title.move_to(np.array([left_track_x + 1.8, 3.6, 0]))
        right_title.move_to(np.array([right_track_x - 1.8, 3.6, 0]))

        # -----------------------------
        # Left: sentence -> tokens -> Text Encoder -> text embedding
        # -----------------------------
        sentence = Text("a photo of a dog", font_size=42, color=GREEN_C)
        sentence.move_to(np.array([left_track_x + 1.8, 2.7, 0]))
        self.play(FadeIn(left_title), run_time=1)
        self.play(FadeOut(left_title), run_time=0.5)
        self.play(Write(sentence), run_time=0.9)

        words = ["a", "photo", "of", "a", "dog"]
        tok_colors = [GREEN_E, GREEN_D, GREEN_C, GREEN_B, GREEN_A]
        tokens = VGroup(*[make_token_block(w, c) for w, c in zip(words, tok_colors)])
        tokens.arrange(RIGHT, buff=0.18)
        tokens.move_to(sentence.get_center())

        # Split into tokens
        self.play(ReplacementTransform(sentence, tokens), run_time=1.0)
        self.wait(0.2)

        # Move tokens downward along the left track
        tokens_target = tokens.copy().move_to(np.array([left_track_x + 1.8, 1.2, 0]))
        self.play(Transform(tokens, tokens_target), run_time=0.9)

        # Text encoder box
        text_encoder = make_encoder("Text Encoder", w=2.4, h=1.6, layers=7)
        text_encoder.move_to(np.array([left_track_x + 1.8, -0.3, 0]))
        self.play(FadeIn(text_encoder, shift=UP*0.2), run_time=0.6)

        # Tokens pass through encoder (shrink into box center)
        into_box = tokens.copy().scale(0.55).move_to(text_encoder.get_center())
        self.play(Transform(tokens, into_box), run_time=0.8)
        self.play(FadeOut(tokens), run_time=0.25)

        # Emerge as text embedding vector（[可調] n=維度列數，這裡用 6 列）
        text_vec = make_number_vector(n=6, buff=0.12, offset=0)
        text_vec.move_to(np.array([left_track_x + 1.8, -2.3, 0]))
        text_vec_lbl = Text("Text embedding", font_size=22, color=GREY_A)
        text_vec_lbl.next_to(text_vec, DOWN, buff=0.2)

        self.play(FadeIn(text_vec, shift=DOWN*0.2), run_time=0.8)
        self.play(FadeIn(text_vec_lbl), run_time=0.4)

        # -----------------------------
        # Right: real dog image -> patch grid -> Image Encoder -> image embedding
        # -----------------------------
        # 使用實際狗狗圖片檔
        self.play(FadeIn(right_title), run_time=1)
        self.play(FadeOut(right_title), run_time=0.5)
        dog_img = ImageMobject("example_photos/dog_photo.jpg")
        dog_img.set_height(2.0)  # [可調] 圖片高度
        dog_img.move_to(np.array([right_track_x - 1.8, 1.8, 0]))  # [可調] 位置

        # 圖片上方的說明文字
        img_cap = Text("(image)", font_size=22, color=GREY_A)
        img_cap.next_to(dog_img, UP, buff=0.1)

        # 動畫：圖片與說明淡入
        self.play(FadeIn(dog_img, shift=UP*0.2), FadeIn(img_cap), run_time=0.9)

        # 在狗狗圖片上畫出 6x6 的網格線
        n_grid = 6
        grid_lines = VGroup()
        left_x  = dog_img.get_left()[0]
        right_x = dog_img.get_right()[0]
        top_y   = dog_img.get_top()[1]
        bot_y   = dog_img.get_bottom()[1]

        dx = (right_x - left_x) / n_grid
        dy = (top_y   - bot_y) / n_grid

        for k in range(1, n_grid):
            # 水平線
            y = top_y - k * dy
            grid_lines.add(Line([left_x, y, 0], [right_x, y, 0], stroke_color=WHITE, stroke_width=1.6))
            # 垂直線
            x = left_x + k * dx
            grid_lines.add(Line([x, top_y, 0], [x, bot_y, 0], stroke_color=WHITE, stroke_width=1.6))

        # 「Split into patches」標籤
        split_lbl = Text("Split into patches", font_size=22, color=GREY_A)
        split_lbl.next_to(dog_img, DOWN, buff=0.25)

        # 動畫：網格線畫出 + 標籤出現；之後標籤淡出
        self.play(ShowCreation(grid_lines, lag_ratio=0.02), FadeIn(split_lbl, shift=DOWN*0.12), run_time=0.9)
        self.play(FadeOut(split_lbl), run_time=0.3)

        # Image encoder 方塊（外觀在 make_encoder 裡可調）
        # [可調] 編碼器位置：y=0.0 可改上下
        img_encoder = make_encoder("Image Encoder", w=2.4, h=1.6, layers=7)
        img_encoder.move_to(np.array([right_track_x - 1.8, -0.3, 0]))
        self.play(FadeIn(img_encoder, shift=UP*0.2), run_time=0.6)

        # Patches lift off one-by-one into the encoder
        # 依照剛才畫的 6x6 grid，在圖片範圍上建立「虛擬 patch 方塊」（實際畫的是 encoder 前要飛的方塊）
        patches = VGroup()
        for iy in range(n_grid):
            for ix in range(n_grid):
                cx = left_x + (ix + 0.5) * dx
                cy = top_y  - (iy + 0.5) * dy
                rect = Rectangle(
                    width=dx * 0.9,
                    height=dy * 0.9,
                    stroke_color=YELLOW_E,
                    stroke_width=1.2,
                    fill_color=YELLOW_E,
                    fill_opacity=0.35,
                ).move_to(np.array([cx, cy, 0]))
                patches.add(rect)

        # pick a path (simple scan) and animate a subset for speed while still conveying \"one by one\"
        idxs = list(range(len(patches)))
        step = max(1, len(idxs)//20)  # animate ~20 patches
        idxs = idxs[::step]

        fly_anims = []
        flying_patches = VGroup()
        for k, i in enumerate(idxs):
            p = patches[i].copy()
            p.set_stroke(WHITE, 0.8)
            p.set_fill(patches[i].get_fill_color(), opacity=1)
            # start at patch, then fly into encoder center
            flying_patches.add(p)
            fly_anims.append(Transform(p, p.copy().scale(0.6).move_to(img_encoder.get_center())))
        self.play(LaggedStart(*fly_anims, lag_ratio=0.07), run_time=2.0)
        # patches that flew into the encoder fade out after arriving
        self.play(FadeOut(flying_patches), run_time=0.6)

        # Image embedding vector（[可調] 同樣改成 6 列）
        img_vec = make_number_vector(n=6, buff=0.12, offset=31)
        img_vec.move_to(np.array([right_track_x - 1.8, -2.3, 0]))
        img_vec_lbl = Text("Image embedding", font_size=22, color=GREY_A)
        img_vec_lbl.next_to(img_vec, DOWN, buff=0.2)

        self.play(FadeIn(img_vec, shift=DOWN*0.2), run_time=0.8)
        self.play(FadeIn(img_vec_lbl), run_time=0.4)
        # 狗狗圖片被 map 成向量後淡出（含網格線與說明）
        self.play(
            FadeOut(dog_img),
            FadeOut(img_cap),
            FadeOut(grid_lines),
            run_time=0.6
        )
        self.wait(0.4)

        # -----------------------------
        # Center: shared 2D embedding space + projection + cosine similarity
        # -----------------------------
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_width": 1.4, "stroke_color": BLUE_E, "stroke_opacity": 0.35},
            axis_config={"stroke_color": BLUE_C, "stroke_width": 2},
        )
        plane.scale(0.8)
        plane.move_to(ORIGIN)

        space_lbl = Text("Shared 2D embedding space", font_size=26, color=BLUE_B)
        space_lbl.next_to(plane, UP, buff=0.25)

        # Bring in the plane
        self.play(FadeIn(plane, lag_ratio=0.0), FadeIn(space_lbl), run_time=0.9)

        # Move both vectors toward center, then "project" into points
        text_vec_to_center = text_vec.copy().scale(0.65).move_to(np.array([-1.7, -0.4, 0]))
        img_vec_to_center  = img_vec.copy().scale(0.65).move_to(np.array([ 1.7, -0.4, 0]))
        self.play(
            Transform(text_vec, text_vec_to_center),
            Transform(img_vec,  img_vec_to_center),
            FadeOut(text_vec_lbl),
            FadeOut(img_vec_lbl),
            run_time=1.0
        )

        # Project to points: vector transforms into a dot in the embedding space
        text_point = make_glow_dot(color=GREEN_C, r=0.08).move_to(plane.c2p(-0.7, 0.5))
        img_point  = make_glow_dot(color=YELLOW_C, r=0.08).move_to(plane.c2p(-0.35, 0.25))

        proj_lbl = Text("project", font_size=20, color=GREY_B)
        proj_lbl.move_to(np.array([0, -0.9, 0]))

        self.play(
            ReplacementTransform(text_vec, text_point),
            ReplacementTransform(img_vec, img_point),
            run_time=0.9
        )

        # Cosine similarity dotted line
        sim_line = DashedLine(text_point.get_center(), img_point.get_center(), dash_length=0.12)
        sim_line.set_stroke(WHITE, 2.2, opacity=0.9)
        sim_lbl = Text("cosine similarity", font_size=22, color=GREY_A)
        sim_lbl.next_to(sim_line.get_center(), UP, buff=0.15)

        self.play(ShowCreation(sim_line), FadeIn(sim_lbl), run_time=0.7)
        self.play(FadeOut(sim_lbl), run_time=0.3)
        self.wait(0.6)

        # Fade out surrounding (titles + encoders), zoom in on shared embedding space
        surrounding = VGroup(left_title, right_title, text_encoder, img_encoder)
        center_group = VGroup(plane, space_lbl, text_point, img_point, sim_line)
        self.play(
            FadeOut(surrounding),
            center_group.animate.scale(1.55).move_to(ORIGIN),
            run_time=0.9
        )
        self.wait(0.2)

        # -----------------------------
        # Ending: shared clusters in embedding space — 狗 / 貓 / 車子
        # -----------------------------
        clusters = [
            {"center": (-1.5,  1.0), "n": 4, "c1": GREEN_C,  "c2": YELLOW_C, "label": "dog"},
            {"center": ( 1.4,  0.9), "n": 4, "c1": TEAL_C,   "c2": BLUE_C,   "label": "cat"},
            {"center": ( 0.0, -1.1), "n": 4, "c1": PURPLE_C, "c2": RED_C,   "label": "car"},
        ]

        new_pairs = VGroup()
        new_lines = VGroup()
        cluster_labels = VGroup()

        for cl in clusters:
            cx, cy = cl["center"]
            # 文字與圖片點對（同一語意聚在一起）
            for _ in range(cl["n"]):
                dx = random.uniform(-0.32, 0.32)
                dy = random.uniform(-0.25, 0.25)
                p_text = make_glow_dot(color=cl["c1"], r=0.06).move_to(plane.c2p(cx + dx, cy + dy))
                p_img  = make_glow_dot(color=cl["c2"], r=0.06).move_to(plane.c2p(
                    cx + dx + random.uniform(-0.15, 0.15),
                    cy + dy + random.uniform(-0.15, 0.15)
                ))
                line = DashedLine(p_text.get_center(), p_img.get_center(), dash_length=0.10)
                line.set_stroke(GREY_A, 1.5, opacity=0.55)
                new_pairs.add(p_text, p_img)
                new_lines.add(line)
            # 每個 cluster 的標籤（狗 / 貓 / 車子）
            lbl = Text(cl["label"], font_size=24, color=cl["c1"])
            lbl.next_to(plane.c2p(cx, cy), UP, buff=0.35)
            cluster_labels.add(lbl)

        end_lbl = Text("Similar text-image pairs cluster by meaning", font_size=26, color=GREY_A)
        end_lbl.next_to(plane, DOWN, buff=0.25)

        self.play(
            LaggedStart(*[FadeIn(m, shift=UP*0.05) for m in new_pairs], lag_ratio=0.03),
            LaggedStart(*[ShowCreation(l) for l in new_lines], lag_ratio=0.03),
            run_time=1.6
        )
        self.play(FadeIn(cluster_labels), run_time=0.6)
        self.play(FadeIn(end_lbl), run_time=0.5)

        self.wait(2)
