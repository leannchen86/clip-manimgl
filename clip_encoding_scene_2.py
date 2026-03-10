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
        return YELLOW_C  # matched pair → push high
    return BLUE_E  # mismatched → push low


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
        vec_font_size = 10 * scale_factor      # 向量字體大小（數字越大向量越大）
        vec_buff_above_thumb = 0.3 * scale_factor   # 圖片向量在圖片上方的間距
        vec_buff_left_prompt = 0.3 * scale_factor  # 文字向量在文字左方的間距
        vec_spread_at_bottom = 0.35 * scale_factor  # 內積時兩向量在格子下方的左右間距
        vec_below_cell_offset = 0.15 * scale_factor  # 下方相乘處與格子的垂直距離（越大離格子越遠）
        # 表格下方一排的 y（與 L_CLIP、subtitle、Normalization 同高）
        bottom_row_y = -N * step - 0 * scale_factor

        grid_anchor = np.array([
            -FRAME_X_RADIUS + left_col_w + 0.4 * scale_factor + step / 2 + layout_shift_right,
            FRAME_Y_RADIUS - 1.8 * scale_factor - (N - 1) * step / 2 + layout_shift_up,
            0
        ])

        # ----------  Axes: thumbnails (row), text prompts (column)  ----------
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

        # ----------  固定向量：每張圖片一個、每段文字一個  ----------
        def vec_str(a, b, c):
            return r"[" + f"{a:.1f}, {b:.1f}, {c:.1f}" + r"]"

        # 圖片 embedding（狗、貓、車）固定數值
        img_vecs = [
            (0.9, 0.4, -0.2),   # image 0 dog
            (0.3, 0.85, 0.15),  # image 1 cat
            (-0.4, 0.2, 0.9),   # image 2 car
        ]
        # 文字 embedding（狗、貓、車）固定數值
        txt_vecs = [
            (0.95, 0.2, 0.0),   # text 0 dog
            (0.35, 0.9, 0.25),  # text 1 cat
            (-0.35, 0.1, 0.85), # text 2 car
        ]
        # 內積得到 logits（縮放後對角線約 4.x、非對角線較小）
        def dot(u, v):
            return sum(a * b for a, b in zip(u, v))
        logits_matrix = [[dot(img_vecs[i], txt_vecs[j]) * 4.5 for j in range(N)] for i in range(N)]

        cells, row_centers = create_matrix_cells(N, grid_anchor, step, cell_size, scale_factor)
        populate_raw_logits(cells, logits_matrix, N, scale_factor)

        # 在對應圖片上方、文字左方顯示固定向量
        img_vec_mobs = []
        txt_vec_mobs = []
        fs = int(vec_font_size)
        for i in range(N):
            v = Tex(vec_str(*img_vecs[i]), font_size=fs, color=YELLOW_C)
            thumb_center = grid_anchor + np.array([i * step, step * 0.8, 0])
            v.next_to(thumb_center, UP, buff=vec_buff_above_thumb)
            img_vec_mobs.append(v)
        for j in range(N):
            v = Tex(vec_str(*txt_vecs[j]), font_size=fs, color=TEAL_C)
            prompt_center = grid_anchor + np.array([-step * 0.9, -j * step, 0])
            v.next_to(prompt_center, LEFT, buff=vec_buff_left_prompt)
            txt_vec_mobs.append(v)
        self.play(
            LaggedStart(*[FadeIn(v) for v in img_vec_mobs], lag_ratio=0.08),
            LaggedStart(*[FadeIn(v) for v in txt_vec_mobs], lag_ratio=0.08),
            run_time=0.6
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

                # 複製：圖片向量 i、文字向量 j，起點在對應圖上／文左
                copy_img = Tex(vec_str(*img_vecs[i]), font_size=fs, color=YELLOW_C)
                copy_txt = Tex(vec_str(*txt_vecs[j]), font_size=fs, color=TEAL_C)
                copy_img.move_to(img_vec_mobs[i].get_center())
                copy_txt.move_to(txt_vec_mobs[j].get_center())
                self.add(copy_img, copy_txt)

                # 兩向量位移到該格下方（左右並排）
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
                # 內積結果出現，再移進格子
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
        # 相乘完成後，上方／左方的固定向量淡出
        self.play(
            FadeOut(img_vec_mobs[0]), FadeOut(img_vec_mobs[1]), FadeOut(img_vec_mobs[2]),
            FadeOut(txt_vec_mobs[0]), FadeOut(txt_vec_mobs[1]), FadeOut(txt_vec_mobs[2]),
            run_time=0.4
        )
        self.wait(0.2)

        # Subtitle: Diagonal = matched → HIGH; Off-diagonal = mismatched → LOW（與 L_CLIP 同高）
        line1 = Tex(r"\text{Diagonal = matched pairs} \rightarrow \text{ push similarity HIGH}", font_size=22, color=GREY_A)
        line2 = Tex(r"\text{Off-diagonal = mismatched} \rightarrow \text{ push similarity LOW}", font_size=22, color=GREY_A)
        subtitle = VGroup(line1, line2).arrange(DOWN, buff=0.12)
        subtitle.move_to(grid_anchor + np.array([(N - 1) * step / 2, bottom_row_y, 0]))
        self.play(FadeIn(subtitle, shift=UP * 0.1), run_time=0.6)
        self.wait(0.4)

        # ----------  Stage 2: Row-wise softmax  ----------
        # 兩行字幕 fadeout 後等約 1 秒，再 fadein Normalization
        self.play(FadeOut(subtitle), run_time=0.5)
        self.wait(1)
        norm_lbl = Tex(r"\text{Normalization}", font_size=28, color=YELLOW_C)
        norm_lbl.move_to(grid_anchor + np.array([(N - 1) * step / 2, bottom_row_y, 0]))
        self.play(FadeIn(norm_lbl), run_time=0.5)
        self.wait(0.2)

        # Bar chart 顯示在圖表右方（與該列對齊）
        bar_chart_offset_right = 0.9 * scale_factor  # 矩陣右緣到 bar chart 的距離

        for row_idx in range(N):
            logits_row = logits_matrix[row_idx]
            probs_row = softmax_row(logits_row)
            # 位置：矩陣右側，與當前列同高
            bar_center = grid_anchor + np.array([
                (N - 1) * step + bar_chart_offset_right,
                -row_idx * step,
                0
            ])

            # Bars: first show raw logits (shifted for visibility), then normalize to probs
            bars = VGroup()
            max_l = max(logits_row)
            min_l = min(logits_row)
            span = max_l - min_l if max_l != min_l else 1
            for k in range(N):
                # Height proportional to logit (shifted to be positive for display)
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
            # Normalize: bars become probabilities (heights sum to 1 visually)
            for k, bar in enumerate(bars):
                new_h = (0.2 + 0.5 * probs_row[k]) * scale_factor
                bar.stretch(new_h / bar.get_height(), 1)
            bars.arrange(RIGHT, buff=0.05 * scale_factor)
            bars.move_to(bar_center)
            self.play(bars.animate.stretch(0.9, 1), run_time=0.4)
            self.play(FadeOut(bars), run_time=0.15)

            # Snap back: update cell numbers to probabilities
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
        # Normalization 文字在 L_image 出現前淡出
        self.play(FadeOut(norm_lbl), run_time=0.4)
        self.wait(0.15)

        # ----------  Loss: L_image → 留下移表下偏左；L_text → 留下移表下偏右；合併後 FadeOut 兩者、FadeIn L_CLIP  ----------
        # [可調] 箭頭與標籤位置
        row_arrow_below = -0.1 * scale_factor
        row_arrow_left = 0.3 * scale_factor
        row_arrow_right = 0.3 * scale_factor
        col_arrow_right = 0.5 * scale_factor
        col_arrow_above = 0.35 * scale_factor
        col_arrow_below = -0.5 * scale_factor
        # [可調] 移下來後的三個文字位置（與 subtitle / Normalization 同高 = bottom_row_y）
        below_y = bottom_row_y
        below_left_x = (N - 1) * step / 2 - 1.1 * scale_factor   # L_image 移下來後的 x（越小越左）
        below_right_x = (N - 1) * step / 2 + 1.1 * scale_factor   # L_text 移下來後的 x（越大越右）
        # merge_center = 表格水平中線 + below_y，L_CLIP 也出現在此

        grid_center = grid_anchor + np.array([(N - 1) * step / 2, -(N - 1) * step / 2, 0])

        # L_image：橫箭頭掃過 → 箭頭淡出，標籤留下並移到表下方偏左
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

        # L_text：直箭頭掃過 → 箭頭淡出，標籤留下並移到表下方偏右
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

        # 兩標籤合併到中央 → FadeOut 兩者、FadeIn L_CLIP
        merge_center = grid_anchor + np.array([(N - 1) * step / 2, below_y, 0])
        self.play(
            L_img_label.animate.move_to(merge_center),
            L_txt_label.animate.move_to(merge_center),
            run_time=0.6
        )
        self.wait(0.2)
        L_clip_label = Tex(r"\mathcal{L}_{\text{CLIP}}", font_size=int(32 * scale_factor), color=YELLOW_C)
        L_clip_label.move_to(merge_center)  # 在 L_image 與 L_text 合體處出現
        self.play(
            FadeOut(L_img_label),
            FadeOut(L_txt_label),
            FadeIn(L_clip_label, scale=1.2),
            run_time=0.7
        )
        self.wait(1.2)

        self.embed()
