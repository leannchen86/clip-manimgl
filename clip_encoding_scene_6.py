from manimlib import *
import numpy as np


def cluster_gaussian(n, center, spread, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=spread, size=(n, 2)) + np.array(center, dtype=float)
    return pts


def make_confusion_cells(
    matrix,
    anchor,
    cell,
    gap,
    font_scale=1.0,
):
    """
    matrix[i, j] = P(pred=j | true=i), row i true class, col j predicted.
    Returns (cell_mobs VGroup, col_groups[j] = column j for highlighting).
    """
    n = int(matrix.shape[0])
    mobs = VGroup()
    col_groups = [VGroup() for _ in range(n)]

    for i in range(n):
        for j in range(n):
            v = float(matrix[i, j])
            is_diag = i == j
            if is_diag:
                col = GREEN_C
                op = 0.22 + 0.62 * min(1.0, v * 1.15)
            else:
                col = RED_C if v > 0.12 else ORANGE
                op = 0.12 + 0.78 * min(1.0, v * 1.2)

            box = RoundedRectangle(
                width=cell,
                height=cell,
                corner_radius=0.05,
                stroke_color=GREY_B,
                stroke_width=1.2,
                fill_color=col,
                fill_opacity=op,
            )
            cx = anchor[0] + j * (cell + gap)
            cy = anchor[1] - i * (cell + gap)
            box.move_to(np.array([cx, cy, 0]))

            pct = int(round(v * 100))
            num = Tex(rf"{pct}\%", font_size=int(20 * font_scale), color=WHITE)
            num.move_to(box.get_center())
            g = VGroup(box, num)
            mobs.add(g)
            col_groups[j].add(g)

    return mobs, col_groups


CLASS_COLORS = [TEAL_C, BLUE_C, PURPLE_B, MAROON_B, GOLD_E]


class BadSinksClustersScene(Scene):
    """
    Scene 6 — bad sinks / clusters: tight vs crowded regions, sorted metrics as a
    descending curve, more classes → more infighting, confusion matrix + attractor column.
    """

    def construct(self):
        self.camera.background_color = BLACK

        AX_COLOR = GREY_B
        LABEL_COLOR = GREY_A

        np.random.seed(11)

        names = ["David", "Daniel", "Mike", "Michael", "Matt"]
        N_CLS = len(names)

        # ------------------------------------------------------------------
        # Stage 0 — title
        # ------------------------------------------------------------------
        title = Tex(
            R"\text{Bad sinks: name clusters \& confusion}",
            font_size=36,
            color=WHITE,
        )
        title.to_edge(UP, buff=0.32)
        self.play(FadeIn(title, shift=DOWN * 0.12), run_time=0.7)
        self.wait(0.35)

        # ------------------------------------------------------------------
        # Stage 1 — embedding: tight cluster vs crowded overlap
        # ------------------------------------------------------------------
        emb_axes = Axes(
            x_range=[-3.4, 3.4, 1],
            y_range=[-2.5, 2.5, 1],
            width=6.2,
            height=4.2,
            axis_config=dict(stroke_color=AX_COLOR, stroke_width=2),
        )
        emb_axes.to_edge(LEFT, buff=0.55).shift(DOWN * 0.15)

        cap_tight = Tex(R"\text{tight, clean cluster}", font_size=22, color=TEAL_C)
        cap_crowd = Tex(R"\text{crowded region (overlap)}", font_size=22, color=ORANGE)
        cap_tight.next_to(emb_axes.c2p(-1.35, 1.15), UP, buff=0.08)
        cap_crowd.next_to(emb_axes.c2p(1.4, 1.15), UP, buff=0.08)

        cluster_specs = [
            (0, (-1.35, 0.75), 0.22, 55, 1),   # David — tight
            (1, (0.95, 0.15), 0.38, 48, 2),     # Daniel — crowded
            (2, (1.45, -0.55), 0.36, 52, 3),    # Mike — crowded, overlaps Daniel
        ]

        all_dot_groups = {}
        for idx, ctr, sprd, n, sd in cluster_specs:
            P = cluster_gaussian(n, ctr, sprd, seed=sd)
            grp = VGroup()
            for p in P:
                d = Dot(emb_axes.c2p(p[0], p[1]), radius=0.042)
                d.set_fill(CLASS_COLORS[idx], 0.92).set_stroke(BLACK, 0)
                d.set_z_index(-1)
                grp.add(d)
            all_dot_groups[idx] = grp

        lbl_david = Tex(
            R"\text{David}", font_size=24, color=TEAL_C
        ).next_to(cap_tight, UP, buff=0.06)
        lbl_crowd = Tex(
            R"\text{Daniel, Mike}", font_size=22, color=GREY_A
        ).next_to(cap_crowd, UP, buff=0.06)

        self.play(ShowCreation(emb_axes), run_time=0.65)
        self.play(
            LaggedStart(FadeIn(all_dot_groups[0], scale=0.3), lag_ratio=0.02),
            run_time=1.0,
        )
        self.play(FadeIn(cap_tight), FadeIn(lbl_david), run_time=0.45)
        self.play(
            LaggedStart(FadeIn(all_dot_groups[1], scale=0.3), lag_ratio=0.02),
            LaggedStart(FadeIn(all_dot_groups[2], scale=0.3), lag_ratio=0.02),
            run_time=1.1,
        )
        self.play(FadeIn(cap_crowd), FadeIn(lbl_crowd), run_time=0.45)
        self.wait(0.55)

        note_embed = Tex(
            R"\text{Some names sit in tight basins; others share a crowded sink.}",
            font_size=24,
            color=LABEL_COLOR,
        )
        note_embed.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(note_embed, shift=UP * 0.1), run_time=0.55)
        self.wait(0.7)
        self.play(FadeOut(note_embed), run_time=0.4)

        # ------------------------------------------------------------------
        # Stage 2 — bar chart: unsorted → sorted (descending "curve")
        # ------------------------------------------------------------------
        recall = np.array([0.84, 0.46, 0.61, 0.79, 0.34], dtype=float)
        perm_display = np.array([2, 4, 0, 1, 3], dtype=int)

        bar_axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 1.05, 0.25],
            width=5.4,
            height=3.05,
            axis_config=dict(stroke_color=AX_COLOR, stroke_width=1.8),
        )
        bar_axes.next_to(emb_axes, RIGHT, buff=0.55).align_to(emb_axes, UP)

        bar_w = 0.42
        gap_x = 1.0

        def bar_group_at_slots(slots, heights, labels, color_indices=None):
            grp = VGroup()
            for k, (sx, h, nm) in enumerate(zip(slots, heights, labels)):
                bl = bar_axes.c2p(sx - bar_w / 2, 0)
                tr = bar_axes.c2p(sx + bar_w / 2, h)
                w = abs(tr[0] - bl[0])
                hgt = abs(tr[1] - bl[1])
                fill_c = CLASS_COLORS[color_indices[k]] if color_indices else BLUE_D
                r = Rectangle(width=w, height=hgt, stroke_width=1.2)
                r.set_stroke(GREY_B, 1.2)
                r.set_fill(fill_c, 0.75)
                r.move_to((bl + tr) / 2)
                lab = Tex(rf"\text{{{nm}}}", font_size=18, color=GREY_A)
                lab.next_to(r, DOWN, buff=0.12)
                grp.add(VGroup(r, lab))
            return grp

        x_slots = 1.0 + gap_x * np.arange(N_CLS)
        bars_unsorted = bar_group_at_slots(
            x_slots,
            recall[perm_display],
            [names[i] for i in perm_display],
            color_indices=list(perm_display),
        )

        chart_title = Tex(
            R"\text{Per-name recall}",
            font_size=26,
            color=WHITE,
        )
        chart_title.next_to(bar_axes, UP, buff=0.35)

        sort_lbl = Tex(
            R"\text{sort} \Rightarrow \text{descending curve}",
            font_size=24,
            color=YELLOW_C,
        )
        sort_lbl.next_to(bar_axes, DOWN, buff=0.5)

        self.play(
            ShowCreation(bar_axes),
            FadeIn(chart_title, shift=DOWN * 0.08),
            run_time=0.7,
        )
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.2) for b in bars_unsorted], lag_ratio=0.12),
            run_time=1.1,
        )
        self.wait(0.35)

        sort_idx = np.argsort(-recall)
        sorted_names = [names[i] for i in sort_idx]
        sorted_heights = recall[sort_idx]
        bars_sorted = bar_group_at_slots(
            x_slots, sorted_heights, sorted_names,
            color_indices=list(sort_idx),
        )

        curve_pts = [bar_axes.c2p(x_slots[i], sorted_heights[i]) for i in range(N_CLS)]
        curve = VMobject()
        curve.set_points_smoothly(curve_pts)
        curve.set_stroke(YELLOW_A, width=3, opacity=0.85)

        self.play(
            Transform(bars_unsorted, bars_sorted),
            FadeIn(sort_lbl, shift=UP * 0.08),
            run_time=1.35,
            rate_func=smooth,
        )
        self.play(ShowCreation(curve), run_time=0.75)
        self.wait(0.65)

        # ------------------------------------------------------------------
        # Stage 3 — more classes → more infighting (extra dots + recall drop)
        # ------------------------------------------------------------------
        crowd_note = Tex(
            R"\text{More names} \Rightarrow \text{more infighting in the same sinks}",
            font_size=24,
            color=LABEL_COLOR,
        )
        crowd_note.to_edge(DOWN, buff=0.26)

        extra_specs = [
            (3, (2.05, 0.55), 0.25, 22, 10),   # Michael — right of Daniel
            (4, (0.35, -1.15), 0.25, 22, 12),   # Matt — below-left of Mike
        ]
        extra_groups = VGroup()
        for idx, ctr, sprd, n, sd in extra_specs:
            P = cluster_gaussian(n, ctr, sprd, seed=sd)
            for p in P:
                d = Dot(emb_axes.c2p(p[0], p[1]), radius=0.038)
                d.set_fill(CLASS_COLORS[idx], 0.88).set_stroke(BLACK, 0)
                d.set_z_index(-1)
                extra_groups.add(d)

        extra_lbl = Tex(
            R"\text{+ Michael, Matt}",
            font_size=22, color=GREY_A,
        ).next_to(lbl_crowd, RIGHT, buff=0.05)

        recall_after = np.array([0.82, 0.31, 0.42, 0.55, 0.22], dtype=float)
        sort_idx2 = np.argsort(-recall_after)
        sorted_h2 = recall_after[sort_idx2]
        sorted_n2 = [names[i] for i in sort_idx2]
        bars_after = bar_group_at_slots(
            x_slots, sorted_h2, sorted_n2,
            color_indices=list(sort_idx2),
        )
        curve_pts2 = [bar_axes.c2p(x_slots[i], sorted_h2[i]) for i in range(N_CLS)]
        curve2 = VMobject()
        curve2.set_points_smoothly(curve_pts2)
        curve2.set_stroke(RED_C, width=3, opacity=0.85)

        self.play(FadeIn(crowd_note, shift=UP * 0.1), run_time=0.45)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.15) for d in extra_groups], lag_ratio=0.03),
            FadeIn(extra_lbl, shift=UP * 0.06),
            run_time=1.4,
        )
        self.wait(0.3)

        self.play(
            Transform(bars_unsorted, bars_after),
            Transform(curve, curve2),
            run_time=1.2,
            rate_func=smooth,
        )
        self.wait(0.75)
        self.play(FadeOut(crowd_note), run_time=0.35)

        # ------------------------------------------------------------------
        # Stage 4 — confusion matrix + strong attractor column
        # ------------------------------------------------------------------
        self.play(
            FadeOut(chart_title),
            FadeOut(sort_lbl),
            FadeOut(curve),
            FadeOut(bars_unsorted),
            FadeOut(bar_axes),
            run_time=0.65,
        )

        # Michael column steals from Daniel / Mike / Matt
        C = np.array(
            [
                [0.78, 0.14, 0.03, 0.03, 0.02],  # David
                [0.06, 0.46, 0.04, 0.38, 0.06],   # Daniel → Michael
                [0.02, 0.05, 0.42, 0.36, 0.15],   # Mike → Michael, Matt
                [0.03, 0.08, 0.06, 0.79, 0.04],   # Michael (strong self)
                [0.02, 0.04, 0.18, 0.42, 0.34],   # Matt → Michael
            ],
            dtype=float,
        )

        cell_s = 0.58
        gap_m = 0.07
        grid_w = N_CLS * cell_s + (N_CLS - 1) * gap_m
        cm_anchor = np.array([2.05, 0.55, 0])

        pred_hdr = Tex(R"\text{predicted}", font_size=22, color=LABEL_COLOR)
        pred_hdr.next_to(
            np.array([cm_anchor[0] + grid_w / 2 - cell_s / 2,
                       cm_anchor[1] + cell_s + 0.55, 0]),
            UP, buff=0.08,
        )

        true_hdr = Tex(R"\text{true}", font_size=22, color=LABEL_COLOR)
        true_hdr.next_to(
            np.array([cm_anchor[0] - 1.15,
                       cm_anchor[1] - grid_w / 2 + cell_s / 2, 0]),
            LEFT, buff=0.12,
        )

        cells_mob, col_groups = make_confusion_cells(C, cm_anchor, cell_s, gap_m)

        row_labels = VGroup()
        col_labels = VGroup()
        for i, nm in enumerate(names):
            t = Tex(rf"\text{{{nm}}}", font_size=18, color=GREY_A)
            t.next_to(
                np.array([cm_anchor[0] - 0.55,
                           cm_anchor[1] - i * (cell_s + gap_m), 0]),
                LEFT, buff=0.12,
            )
            row_labels.add(t)
        for j, nm in enumerate(names):
            t = Tex(rf"\text{{{nm}}}", font_size=18, color=GREY_A)
            t.next_to(
                np.array([cm_anchor[0] + j * (cell_s + gap_m),
                           cm_anchor[1] + 0.52, 0]),
                UP, buff=0.1,
            )
            col_labels.add(t)

        cm_title = Tex(
            R"\text{Confusion matrix: who steals whom?}",
            font_size=28, color=WHITE,
        )
        cm_title.next_to(pred_hdr, UP, buff=0.35)

        attractor_col_idx = 3  # Michael
        attractor_brace = Brace(col_groups[attractor_col_idx], DOWN, buff=0.06)
        attractor_txt = Tex(
            R"\text{strong attractor}", font_size=22, color=YELLOW_C,
        )
        attractor_txt.next_to(attractor_brace, DOWN, buff=0.12)

        self.play(FadeIn(cm_title, shift=DOWN * 0.1), run_time=0.5)
        self.play(FadeIn(pred_hdr), FadeIn(true_hdr), run_time=0.45)
        self.play(FadeIn(row_labels), FadeIn(col_labels), run_time=0.55)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.5) for c in cells_mob], lag_ratio=0.04),
            run_time=1.5,
        )
        self.wait(0.4)

        col_highlight = SurroundingRectangle(
            col_groups[attractor_col_idx],
            color=YELLOW_C,
            stroke_width=2.5,
            buff=0.04,
        )
        self.play(
            ShowCreation(col_highlight),
            GrowFromCenter(attractor_brace),
            FadeIn(attractor_txt, shift=UP * 0.08),
            run_time=0.85,
        )

        steal_note = Tex(
            R"\text{Daniel, Mike, Matt} \to \text{Michael}",
            font_size=22,
            color=GREY_A,
        )
        steal_note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(steal_note, shift=UP * 0.08), run_time=0.55)
        self.wait(1.0)

        for _ in range(2):
            self.play(
                col_highlight.animate.set_stroke(width=5, opacity=1.0),
                run_time=0.3,
            )
            self.play(
                col_highlight.animate.set_stroke(width=2.5, opacity=0.6),
                run_time=0.3,
            )
        self.wait(0.8)

        # self.embed()
