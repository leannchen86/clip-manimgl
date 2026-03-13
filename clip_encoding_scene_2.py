from manimlib import *
import numpy as np


def softmax_row(x):
    """Softmax over a 1D array (row)."""
    e = np.exp(np.array(x, dtype=float) - np.max(x))
    return e / e.sum()


def create_matrix_cells(N, grid_anchor, step, cell_size, scale_factor):
    """
    Create NxN grid of cells. Each cell is VGroup(box, num) where num is a Tex mobject.
    Returns (cells, row_centers).
    """
    cells = []
    for i in range(N):
        for j in range(N):
            box = RoundedRectangle(
                width=cell_size,
                height=cell_size,
                corner_radius=0.06 * scale_factor,
                stroke_color=GREY_B,
                stroke_width=1.2 * scale_factor,
                fill_color=GREY_E,
                fill_opacity=0,
            )
            box.move_to(grid_anchor + np.array([j * step, -i * step, 0]))
            num = Tex("0", font_size=int(22 * scale_factor), color=WHITE)
            num.move_to(box.get_center())
            cells.append(VGroup(box, num))
    row_centers = [
        grid_anchor + np.array([(N - 1) * step / 2, -i * step, 0])
        for i in range(N)
    ]
    return cells, row_centers


def logit_color(is_diag, value):
    """Warm for high logits (matched), cool/dark blue for low (mismatched)."""
    if is_diag:
        return YELLOW_C
    return BLUE_E


def populate_raw_logits(cells, logits_matrix, N, scale_factor):
    """
    Set each cell's number to the raw logit string and color the cell.
    logits_matrix[i][j] = logit for row i, col j.
    """
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            val = logits_matrix[i][j]
            is_diag = i == j
            c = logit_color(is_diag, val)
            cells[idx][0].set_fill(c, opacity=0.5 if is_diag else 0.3)
            cells[idx][0].set_stroke(c, width=1.2 * scale_factor)
            s = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
            cells[idx][1].become(Tex(s, font_size=int(22 * scale_factor), color=WHITE))
            cells[idx][1].move_to(cells[idx][0].get_center())
    return cells


class CLIPSimilarityMatrix(Scene):
    """
    Stage 1: Raw similarity logits (image vs text); diagonal = matched, off-diagonal = mismatched.
    Stage 2: Row-wise softmax → probabilities; then L_image, L_text, L_CLIP.
    """
    def construct(self):
        self.camera.background_color = BLACK

        N = 3
        scale_factor = 1.35
        cell_size = 0.65 * scale_factor
        step = cell_size + 0.08 * scale_factor
        thumb_w, thumb_h = 0.5 * scale_factor, 0.4 * scale_factor
        left_col_w = 1.8 * scale_factor
        layout_shift_right = 2.5
        layout_shift_up = 1

        # [可調] 向量 [n,m,k] 的位置與大小
        vec_font_size_final = 10 * scale_factor      # 最終停在圖片上方/文字左方時的字體大小
        vec_font_size_center = 18 * scale_factor     # 一開始在畫面中央出現時的較大字體
        vec_buff_above_thumb = 0.3 * scale_factor
        vec_buff_left_prompt = 0.3 * scale_factor
        vec_spread_at_bottom = 0.35 * scale_factor
        vec_below_cell_offset = 0.15 * scale_factor

        # 中間出場時的排版
        img_center_x = 0
        img_center_y = 1.2 * scale_factor
        img_center_gap = 0.75 * scale_factor

        # text_vec 先在畫面中間偏右，之後只做「水平位移」到左邊 prompt
        txt_start_x = 0 * scale_factor

        # 中間大字縮到最終小字的比例
        img_scale_ratio = vec_font_size_final / vec_font_size_center
        txt_scale_ratio = vec_font_size_final / vec_font_size_center

        # 表格下方一排的 y
        bottom_row_y = -N * step - 0 * scale_factor

        grid_anchor = np.array([
            -FRAME_X_RADIUS + left_col_w + 0.4 * scale_factor + step / 2 + layout_shift_right,
            FRAME_Y_RADIUS - 1.8 * scale_factor - (N - 1) * step / 2 + layout_shift_up,
            0
        ])

        # ---------- Axes: thumbnails (row), text prompts (column) ----------
        thumb_files = ["dog_w.jpg", "cat_w.jpg", "car_w.jpg"]
        text_prompts = [r"\text{dog}", r"\text{cat}", r"\text{car}"]

        thumb_images = []
        for i in range(N):
            img = ImageMobject(thumb_files[i])
            img.set_height(thumb_h)
            img.move_to(grid_anchor + np.array([i * step, step * 0.8, 0]))
            thumb_images.append(img)

        prompts = VGroup()
        for i, t in enumerate(text_prompts):
            p = Tex(t, font_size=int(28 * scale_factor), color=GREY_A)
            p.move_to(grid_anchor + np.array([-step * 0.9, -i * step, 0]))
            prompts.add(p)

        self.play(
            LaggedStart(*[FadeIn(thumb_images[i], shift=DOWN * 0.1) for i in range(N)], lag_ratio=0.1),
            LaggedStart(*[FadeIn(p, shift=RIGHT * 0.1) for p in prompts], lag_ratio=0.1),
            run_time=1.2
        )
        self.wait(0.3)

        # ---------- 固定向量：每張圖片一個、每段文字一個 ----------
        def vec_str(a, b, c):
            return r"[" + f"{a:.1f}, {b:.1f}, {c:.1f}" + r"]"

        img_vecs = [
            (0.9, 0.4, -0.2),
            (0.3, 0.85, 0.15),
            (-0.4, 0.2, 0.9),
        ]

        txt_vecs = [
            (0.95, 0.2, 0.0),
            (0.35, 0.9, 0.25),
            (-0.35, 0.1, 0.85),
        ]

        def dot(u, v):
            return sum(a * b for a, b in zip(u, v))

        logits_matrix = [[dot(img_vecs[i], txt_vecs[j]) * 4.5 for j in range(N)] for i in range(N)]

        cells, row_centers = create_matrix_cells(N, grid_anchor, step, cell_size, scale_factor)
        populate_raw_logits(cells, logits_matrix, N, scale_factor)

        # ---------- img_vec / txt_vec 先出現在中間，再移到原本位置 ----------
        img_vec_mobs = []
        txt_vec_mobs = []
        fs_final = int(vec_font_size_final)
        fs_center = int(vec_font_size_center)

        # 最終目標位置
        img_targets = []
        txt_targets = []

        for i in range(N):
            thumb_center = grid_anchor + np.array([i * step, step * 0.8, 0])
            dummy = Tex(vec_str(*img_vecs[i]), font_size=fs_final, color=YELLOW_C)
            dummy.next_to(thumb_center, UP, buff=vec_buff_above_thumb)
            img_targets.append(dummy.get_center())

        for j in range(N):
            prompt_center = grid_anchor + np.array([-step * 0.9, -j * step, 0])
            dummy = Tex(vec_str(*txt_vecs[j]), font_size=fs_final, color=TEAL_C)
            dummy.next_to(prompt_center, LEFT, buff=vec_buff_left_prompt)
            txt_targets.append(dummy.get_center())

        # 1) img_vec 先在畫面中間垂直排列，以較大字體出現
        for i in range(N):
            v = Tex(vec_str(*img_vecs[i]), font_size=fs_center, color=YELLOW_C)
            v.move_to(np.array([
                img_center_x,
                img_center_y - i * img_center_gap,
                0
            ]))
            img_vec_mobs.append(v)

        self.play(
            LaggedStart(*[FadeIn(v, scale=1.15) for v in img_vec_mobs], lag_ratio=0.12),
            run_time=0.7
        )
        self.wait(0.2)

        # 2) img_vec 從中間垂直排列移到上方水平排列，同時變小
        self.play(
            LaggedStart(
                *[
                    img_vec_mobs[i].animate.move_to(img_targets[i]).scale(img_scale_ratio)
                    for i in range(N)
                ],
                lag_ratio=0.12
            ),
            run_time=1.0
        )
        self.wait(0.2)

        # 3) text_vec 再出現在畫面中間，以較大字體出現；y 直接對齊最終位置
        for j in range(N):
            v = Tex(vec_str(*txt_vecs[j]), font_size=fs_center, color=TEAL_C)
            v.move_to(np.array([
                txt_start_x,
                txt_targets[j][1],
                0
            ]))
            txt_vec_mobs.append(v)

        self.play(
            LaggedStart(*[FadeIn(v, scale=1.15) for v in txt_vec_mobs], lag_ratio=0.12),
            run_time=0.7
        )
        self.wait(0.2)

        # 4) text_vec 水平位移到 prompt 左邊，同時變小
        self.play(
            LaggedStart(
                *[
                    txt_vec_mobs[j].animate.move_to(txt_targets[j]).scale(txt_scale_ratio)
                    for j in range(N)
                ],
                lag_ratio=0.12
            ),
            run_time=1.0
        )
        self.wait(0.2)

        # 每個格子：從「圖片上方／文字左方」把兩向量移到下方 → 內積 → 數字進格子
        below_offset = cell_size * 0.5 + vec_below_cell_offset
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                cell = cells[idx]
                self.add(cell[0])
                center = cell[0].get_center()
                below_center = center + DOWN * below_offset

                copy_img = Tex(vec_str(*img_vecs[i]), font_size=fs_final, color=YELLOW_C)
                copy_txt = Tex(vec_str(*txt_vecs[j]), font_size=fs_final, color=TEAL_C)
                copy_img.move_to(img_vec_mobs[i].get_center())
                copy_txt.move_to(txt_vec_mobs[j].get_center())
                self.add(copy_img, copy_txt)

                dot_sym = Tex(r"\cdot", font_size=int(22 * scale_factor), color=GREY_A)
                target_left = below_center + LEFT * vec_spread_at_bottom
                target_right = below_center + RIGHT * vec_spread_at_bottom
                self.play(
                    copy_img.animate.move_to(target_left),
                    copy_txt.animate.move_to(target_right),
                    run_time=0.4
                )
                dot_sym.move_to(below_center)
                self.play(FadeIn(dot_sym), run_time=0.12)

                val = logits_matrix[i][j]
                s = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
                result_num = Tex(s, font_size=int(22 * scale_factor), color=WHITE)
                result_num.move_to(below_center)
                self.play(
                    FadeOut(copy_img), FadeOut(copy_txt), FadeOut(dot_sym),
                    FadeIn(result_num),
                    run_time=0.2
                )
                self.play(result_num.animate.move_to(center), run_time=0.35)
                self.remove(result_num)
                self.add(cell[1])

        self.play(
            FadeOut(img_vec_mobs[0]), FadeOut(img_vec_mobs[1]), FadeOut(img_vec_mobs[2]),
            FadeOut(txt_vec_mobs[0]), FadeOut(txt_vec_mobs[1]), FadeOut(txt_vec_mobs[2]),
            run_time=0.4
        )
        self.wait(0.2)

        line1 = Tex(r"\text{Diagonal = matched pairs} \rightarrow \text{ push similarity HIGH}", font_size=22, color=GREY_A)
        line2 = Tex(r"\text{Off-diagonal = mismatched} \rightarrow \text{ push similarity LOW}", font_size=22, color=GREY_A)
        subtitle = VGroup(line1, line2).arrange(DOWN, buff=0.12)
        subtitle.move_to(grid_anchor + np.array([(N - 1) * step / 2, bottom_row_y, 0]))
        self.play(FadeIn(subtitle, shift=UP * 0.1), run_time=0.6)
        self.wait(0.4)

        self.play(FadeOut(subtitle), run_time=0.5)
        self.wait(1)
        norm_lbl = Tex(r"\text{Normalization}", font_size=28, color=YELLOW_C)
        norm_lbl.move_to(grid_anchor + np.array([(N - 1) * step / 2, bottom_row_y, 0]))
        self.play(FadeIn(norm_lbl), run_time=0.5)
        self.wait(0.2)

        bar_chart_offset_right = 0.9 * scale_factor

        for row_idx in range(N):
            logits_row = logits_matrix[row_idx]
            probs_row = softmax_row(logits_row)

            bar_center = grid_anchor + np.array([
                (N - 1) * step + bar_chart_offset_right,
                -row_idx * step,
                0
            ])

            bars = VGroup()
            max_l = max(logits_row)
            min_l = min(logits_row)
            span = max_l - min_l if max_l != min_l else 1
            for k in range(N):
                h_raw = 0.15 + 0.4 * (logits_row[k] - min_l) / span
                bar = Rectangle(
                    width=0.18 * scale_factor,
                    height=h_raw * scale_factor,
                    fill_color=YELLOW_C,
                    fill_opacity=0.8,
                    stroke_width=0.5 * scale_factor,
                )
                bars.add(bar)

            bars.arrange(RIGHT, buff=0.05 * scale_factor)
            bars.move_to(bar_center)

            self.play(FadeIn(bars), run_time=0.25)
            self.wait(0.1)

            for k, bar in enumerate(bars):
                new_h = (0.2 + 0.5 * probs_row[k]) * scale_factor
                bar.stretch(new_h / bar.get_height(), 1)
            bars.arrange(RIGHT, buff=0.05 * scale_factor)
            bars.move_to(bar_center)
            self.play(bars.animate.stretch(0.9, 1), run_time=0.4)
            self.play(FadeOut(bars), run_time=0.15)

            for j in range(N):
                idx = row_idx * N + j
                p_val = probs_row[j]
                s = f"{p_val:.2f}" if p_val >= 0.01 else "<0.01"
                new_num = Tex(s, font_size=int(20 * scale_factor), color=WHITE)
                new_num.move_to(cells[idx][0].get_center())
                self.play(FadeOut(cells[idx][1]), FadeIn(new_num), run_time=0.2)
                self.remove(cells[idx][1])
                self.add(new_num)
            self.wait(0.1)

        self.wait(0.2)

        self.play(FadeOut(norm_lbl), run_time=0.4)
        self.wait(0.15)

        row_arrow_below = -0.1 * scale_factor
        row_arrow_left = 0.3 * scale_factor
        row_arrow_right = 0.3 * scale_factor
        col_arrow_right = 0.5 * scale_factor
        col_arrow_above = 0.35 * scale_factor
        col_arrow_below = -0.5 * scale_factor

        below_y = bottom_row_y
        below_left_x = (N - 1) * step / 2 - 1.1 * scale_factor
        below_right_x = (N - 1) * step / 2 + 1.1 * scale_factor

        row_arrow_y = -N * step - row_arrow_below
        arrow_row = Arrow(
            grid_anchor + np.array([-row_arrow_left, row_arrow_y, 0]),
            grid_anchor + np.array([(N - 1) * step + row_arrow_right, row_arrow_y, 0]),
            buff=0.08 * scale_factor,
            stroke_width=4 * scale_factor,
            stroke_color=MAROON_B,
        )
        L_img_label = Tex(r"\mathcal{L}_{\text{image}}", font_size=int(28 * scale_factor), color=MAROON_B)
        L_img_label.next_to(arrow_row, DOWN, buff=0.12 * scale_factor)
        self.play(ShowCreation(arrow_row), FadeIn(L_img_label), run_time=0.7)
        self.wait(0.2)
        self.play(FadeOut(arrow_row), run_time=0.25)
        target_L_img = grid_anchor + np.array([below_left_x, below_y, 0])
        self.play(L_img_label.animate.move_to(target_L_img), run_time=0.5)
        self.wait(0.15)

        col_arrow_x = (N - 1) * step + col_arrow_right
        arrow_col = Arrow(
            grid_anchor + np.array([col_arrow_x, col_arrow_above, 0]),
            grid_anchor + np.array([col_arrow_x, -N * step - col_arrow_below, 0]),
            buff=0.08 * scale_factor,
            stroke_width=4 * scale_factor,
            stroke_color=TEAL_B,
        )
        L_txt_label = Tex(r"\mathcal{L}_{\text{text}}", font_size=int(28 * scale_factor), color=TEAL_B)
        L_txt_label.next_to(arrow_col, RIGHT, buff=0.12 * scale_factor)
        self.play(ShowCreation(arrow_col), FadeIn(L_txt_label), run_time=0.7)
        self.wait(0.2)
        self.play(FadeOut(arrow_col), run_time=0.25)
        target_L_txt = grid_anchor + np.array([below_right_x, below_y, 0])
        self.play(L_txt_label.animate.move_to(target_L_txt), run_time=0.5)
        self.wait(0.2)

        merge_center = grid_anchor + np.array([(N - 1) * step / 2, below_y, 0])
        self.play(
            L_img_label.animate.move_to(merge_center),
            L_txt_label.animate.move_to(merge_center),
            run_time=0.6
        )
        self.wait(0.2)

        L_clip_label = Tex(r"\mathcal{L}_{\text{CLIP}}", font_size=int(32 * scale_factor), color=YELLOW_C)
        L_clip_label.move_to(merge_center)
        self.play(
            FadeOut(L_img_label),
            FadeOut(L_txt_label),
            FadeIn(L_clip_label, scale=1.2),
            run_time=0.7
        )
        self.wait(1.2)

        self.embed()