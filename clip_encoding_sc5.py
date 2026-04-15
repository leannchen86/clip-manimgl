from manimlib import *
import numpy as np


# ============================================================
# Palette
# ============================================================
BG_COLOR = "#0f1117"
CLASS_A_COLOR = "#4fc3f7"
CLASS_B_COLOR = "#ef5350"
NETWORK_COLOR = "#666688"


# ============================================================
# Helpers
# ============================================================
def title_text(s, scale=0.72, color=WHITE):
    t = Tex(r"\text{" + s + "}", color=color)
    t.scale(scale)
    return t


def caption_text(s, scale=0.42, color="#ccccdd"):
    t = Tex(r"\text{" + s + "}", color=color)
    t.scale(scale)
    return t


def make_moons(n=40, noise=0.08, seed=42):
    """Two interleaved half-moon clusters (similar to sklearn.make_moons)."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, np.pi, n)
    xa = np.cos(theta) + rng.normal(0, noise, n)
    ya = np.sin(theta) + rng.normal(0, noise, n)
    xb = 1 - np.cos(theta) + rng.normal(0, noise, n)
    yb = -np.sin(theta) + 0.5 + rng.normal(0, noise, n)
    return np.column_stack([xa, ya]), np.column_stack([xb, yb])


def boundary_curve(x):
    """Sinusoidal decision boundary that separates the two moons."""
    return 0.4 * np.sin(np.pi * x) + 0.25


def make_shaded_region(plane, t_vals, bfn, side,
                       x_lo, x_hi, y_lo, y_hi, color, opacity):
    """Build a filled VMobject covering one side of a boundary curve."""
    region = VMobject()
    if side == "upper":
        verts = [plane.c2p(x_lo, y_hi)]
        verts.append(plane.c2p(x_lo, bfn(t_vals[0])))
        verts += [plane.c2p(t, bfn(t)) for t in t_vals]
        verts.append(plane.c2p(x_hi, bfn(t_vals[-1])))
        verts.append(plane.c2p(x_hi, y_hi))
        verts.append(plane.c2p(x_lo, y_hi))
    else:
        verts = [plane.c2p(x_hi, y_lo)]
        verts.append(plane.c2p(x_hi, bfn(t_vals[-1])))
        verts += [plane.c2p(t, bfn(t)) for t in t_vals[::-1]]
        verts.append(plane.c2p(x_lo, bfn(t_vals[0])))
        verts.append(plane.c2p(x_lo, y_lo))
        verts.append(plane.c2p(x_hi, y_lo))
    region.set_points_as_corners(verts)
    region.set_fill(color, opacity)
    region.set_stroke(width=0)
    return region


# ============================================================
# Main Scene
# ============================================================
class MLPDecisionBoundaryScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.camera.frame.set_width(14)

        self.stage1_scatter_with_linear()
        self.stage2_mlp_diagram()
        self.stage3_feature_warping()
        self.stage4_nonlinear_boundary()
        self.stage5_epoch_refinement()

    # --------------------------------------------------------
    # Stage 1 — 2D scatter plot + failed linear classifier
    # --------------------------------------------------------
    def stage1_scatter_with_linear(self):
        plane = NumberPlane(
            x_range=[-1.5, 2.5, 1],
            y_range=[-1.5, 2.0, 1],
            width=8,
            height=5.5,
            axis_config=dict(stroke_color=GREY_B, stroke_width=2),
            background_line_style=dict(
                stroke_color="#1a3a4a",
                stroke_width=1,
                stroke_opacity=0.4,
            ),
        )
        plane.shift(DOWN * 0.15)

        pts_a, pts_b = make_moons(40, 0.08, 42)

        dots_a = VGroup(*[
            Dot(plane.c2p(p[0], p[1]), radius=0.06)
            .set_fill(CLASS_A_COLOR, 0.92)
            for p in pts_a
        ])
        dots_b = VGroup(*[
            Dot(plane.c2p(p[0], p[1]), radius=0.06)
            .set_fill(CLASS_B_COLOR, 0.92)
            for p in pts_b
        ])

        title = title_text("MLP Decision Boundary Transformation", scale=0.85)
        title.to_edge(UP, buff=0.35)

        self.play(FadeIn(title, shift=DOWN * 0.12), run_time=0.7)
        self.play(ShowCreation(plane), run_time=0.8)
        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.4) for d in [*dots_a, *dots_b]],
                lag_ratio=0.012,
            ),
            run_time=2.0,
        )

        linear_line = Line(
            plane.c2p(-1.0, -0.8), plane.c2p(2.0, 1.8),
            stroke_color=YELLOW, stroke_width=2.5,
        )
        fail_label = caption_text(
            "Linear classifier: insufficient", scale=0.4, color=YELLOW,
        )
        fail_label.next_to(linear_line.get_end(), UR, buff=0.15)
        x_mark = Tex(r"\times", color=RED, font_size=36)
        x_mark.next_to(fail_label, LEFT, buff=0.1)

        self.play(ShowCreation(linear_line), run_time=1.0)
        self.play(FadeIn(fail_label), FadeIn(x_mark), run_time=0.7)
        self.wait(1.0)

        self._plane = plane
        self._dots_a, self._dots_b = dots_a, dots_b
        self._pts_a, self._pts_b = pts_a, pts_b
        self._title = title
        self._linear_stuff = VGroup(linear_line, fail_label, x_mark)

    # --------------------------------------------------------
    # Stage 2 — MLP diagram + ReLU on the side
    # --------------------------------------------------------
    def stage2_mlp_diagram(self):
        layer_sizes = [2, 5, 5, 2]
        layer_xs = [3.4, 4.2, 5.0, 5.8]
        gap_y = 0.36

        neurons = []
        all_neurons = VGroup()
        for ns, xp in zip(layer_sizes, layer_xs):
            layer = []
            y0 = -(ns - 1) * gap_y / 2
            for i in range(ns):
                c = Circle(
                    radius=0.12,
                    stroke_color=NETWORK_COLOR,
                    stroke_width=1.5,
                )
                c.set_fill(BG_COLOR, 0.8)
                c.move_to(np.array([xp, y0 + i * gap_y, 0]))
                layer.append(c)
                all_neurons.add(c)
            neurons.append(layer)

        edge_groups = []
        edges = VGroup()
        for l_idx in range(len(layer_sizes) - 1):
            layer_edges = VGroup()
            for n1 in neurons[l_idx]:
                for n2 in neurons[l_idx + 1]:
                    layer_edges.add(Line(
                        n1.get_center(), n2.get_center(),
                        stroke_color=NETWORK_COLOR,
                        stroke_width=0.8,
                        stroke_opacity=0.4,
                    ))
            edge_groups.append(layer_edges)
            edges.add(*layer_edges)

        mlp_lbl = caption_text("MLP", scale=0.5, color=WHITE)
        mlp_lbl.move_to(np.array([4.6, 1.4, 0]))

        relu_org = np.array([4.6, -1.6, 0])
        relu_neg = Line(
            relu_org + LEFT * 0.5, relu_org,
            stroke_color=YELLOW_C, stroke_width=2,
        )
        relu_pos = Line(
            relu_org, relu_org + np.array([0.5, 0.4, 0]),
            stroke_color=YELLOW_C, stroke_width=2,
        )
        relu_lbl = caption_text("ReLU", scale=0.35, color=YELLOW_C)
        relu_lbl.next_to(VGroup(relu_neg, relu_pos), DOWN, buff=0.12)

        self.play(FadeOut(self._linear_stuff), run_time=0.6)

        plot_group = VGroup(self._plane, self._dots_a, self._dots_b)
        self.play(plot_group.animate.shift(LEFT * 1.8), run_time=0.8)

        self.play(
            FadeIn(edges),
            LaggedStart(
                *[FadeIn(n, scale=0.5) for n in all_neurons],
                lag_ratio=0.04,
            ),
            FadeIn(mlp_lbl),
            run_time=1.2,
        )
        self.play(
            ShowCreation(relu_neg),
            ShowCreation(relu_pos),
            FadeIn(relu_lbl),
            run_time=0.8,
        )
        self.wait(0.8)

        self._edge_groups = edge_groups
        self._mlp_all = VGroup(
            all_neurons, edges, mlp_lbl,
            relu_neg, relu_pos, relu_lbl,
        )

    # --------------------------------------------------------
    # Stage 3 — Layer-by-layer feature space warping
    # --------------------------------------------------------
    def stage3_feature_warping(self):
        plane = self._plane
        pts_a, pts_b = self._pts_a.copy(), self._pts_b.copy()
        dots_a, dots_b = self._dots_a, self._dots_b

        all_pts = np.vstack([pts_a, pts_b])
        ctr = all_pts.mean(axis=0)

        def layer1(pts):
            c = pts - ctr
            th = 0.4
            R = np.array([
                [np.cos(th), -np.sin(th)],
                [np.sin(th),  np.cos(th)],
            ])
            S = np.diag([1.3, 0.8])
            return (S @ R @ c.T).T + ctr

        def layer2(pts):
            out = pts.copy()
            mask = out[:, 1] < ctr[1]
            out[mask, 1] = ctr[1] + (out[mask, 1] - ctr[1]) * 0.08
            out[:, 0] *= 1.1
            return out

        def layer3_split(pa, pb):
            a, b = pa.copy(), pb.copy()
            a[:, 1] += 0.55
            b[:, 1] -= 0.55
            return a, b

        steps = [
            ("Layer 1: rotate + stretch", YELLOW_C),
            ("Layer 2: ReLU non-linearity", YELLOW_C),
            ("Layer 3: linearly separable", GREEN_C),
        ]

        cur_a, cur_b = pts_a, pts_b

        for idx, (label, col) in enumerate(steps):
            if idx == 0:
                new_a, new_b = layer1(cur_a), layer1(cur_b)
            elif idx == 1:
                new_a, new_b = layer2(cur_a), layer2(cur_b)
            else:
                new_a, new_b = layer3_split(cur_a, cur_b)

            eg = self._edge_groups[idx]
            self.play(
                eg.animate.set_stroke(WHITE, width=1.5, opacity=0.9),
                run_time=0.5,
            )

            move_anims = [
                d.animate.move_to(plane.c2p(p[0], p[1]))
                for d, p in zip(dots_a, new_a)
            ]
            move_anims += [
                d.animate.move_to(plane.c2p(p[0], p[1]))
                for d, p in zip(dots_b, new_b)
            ]
            self.play(*move_anims, run_time=2.0, rate_func=smooth)

            self.play(
                eg.animate.set_stroke(NETWORK_COLOR, width=0.8, opacity=0.4),
                run_time=0.4,
            )
            cur_a, cur_b = new_a, new_b

        sep_y = (cur_a[:, 1].min() + cur_b[:, 1].max()) / 2
        sep_line = Line(
            plane.c2p(-1.5, sep_y), plane.c2p(2.5, sep_y),
            stroke_color=GREEN_C, stroke_width=2.5,
        )
        sep_lbl = caption_text("Linearly separable!", scale=0.42, color=GREEN_C)
        sep_lbl.next_to(sep_line, UP, buff=0.12).shift(RIGHT * 2)

        self.play(ShowCreation(sep_line), FadeIn(sep_lbl), run_time=1.0)
        self.wait(0.6)

        depth_lbl = caption_text(
            "Depth = Expressive Power", scale=0.52, color=YELLOW_C,
        )
        depth_lbl.next_to(self._title, DOWN, buff=0.15)
        self.play(FadeIn(depth_lbl, scale=1.1), run_time=0.8)
        self.wait(1.0)

        self._stage3_extras = VGroup(sep_line, sep_lbl, depth_lbl)

    # --------------------------------------------------------
    # Stage 4 — Map back to original space, non-linear boundary
    # --------------------------------------------------------
    def stage4_nonlinear_boundary(self):
        plane = self._plane
        pts_a, pts_b = self._pts_a, self._pts_b
        dots_a, dots_b = self._dots_a, self._dots_b

        self.play(FadeOut(self._stage3_extras), run_time=0.6)

        anims = [
            d.animate.move_to(plane.c2p(p[0], p[1]))
            for d, p in zip(dots_a, pts_a)
        ]
        anims += [
            d.animate.move_to(plane.c2p(p[0], p[1]))
            for d, p in zip(dots_b, pts_b)
        ]
        self.play(*anims, run_time=2.0, rate_func=smooth)

        t_vals = np.linspace(-1.0, 2.0, 100)
        bnd_pts = [plane.c2p(t, boundary_curve(t)) for t in t_vals]
        boundary = VMobject().set_points_smoothly(bnd_pts)
        boundary.set_stroke(WHITE, width=2.5)

        self.play(ShowCreation(boundary), run_time=1.5)

        upper_r = make_shaded_region(
            plane, t_vals, boundary_curve, "upper",
            -1.5, 2.5, -1.5, 2.0, CLASS_A_COLOR, 0.12,
        )
        lower_r = make_shaded_region(
            plane, t_vals, boundary_curve, "lower",
            -1.5, 2.5, -1.5, 2.0, CLASS_B_COLOR, 0.12,
        )

        lbl_a = caption_text("Class A", scale=0.45, color=CLASS_A_COLOR)
        lbl_a.move_to(plane.c2p(0.5, 1.4))
        lbl_b = caption_text("Class B", scale=0.45, color=CLASS_B_COLOR)
        lbl_b.move_to(plane.c2p(0.5, -0.8))

        self.play(
            FadeIn(upper_r), FadeIn(lower_r),
            FadeIn(lbl_a), FadeIn(lbl_b),
            run_time=1.0,
        )
        self.wait(1.0)

        self._stage4_extras = VGroup(
            boundary, upper_r, lower_r, lbl_a, lbl_b,
        )

    # --------------------------------------------------------
    # Stage 5 — Epoch counter with boundary sharpening
    # --------------------------------------------------------
    def stage5_epoch_refinement(self):
        plane = self._plane

        self.play(
            FadeOut(self._stage4_extras),
            FadeOut(self._mlp_all),
            run_time=0.6,
        )

        plot_group = VGroup(self._plane, self._dots_a, self._dots_b)
        self.play(plot_group.animate.shift(RIGHT * 1.8), run_time=0.8)

        epoch_tracker = ValueTracker(0)
        epoch_label = Tex(
            r"\text{Epoch:}", font_size=36, color=WHITE,
        )
        epoch_label.to_corner(UL, buff=0.6)

        epoch_num = always_redraw(lambda: Tex(
            r"\text{" + str(int(epoch_tracker.get_value())) + "}",
            font_size=36, color=WHITE,
        ).next_to(epoch_label, RIGHT, buff=0.2))

        self.play(FadeIn(epoch_label), FadeIn(epoch_num), run_time=0.5)

        rng = np.random.default_rng(42)
        fixed_noise = rng.normal(0, 1, 80)
        t_vals = np.linspace(-1.0, 2.0, 80)

        def make_bnd(frac):
            amp = 0.05 + 0.35 * frac
            ns = 0.15 * (1.0 - frac)
            pts = [
                plane.c2p(
                    t,
                    amp * np.sin(np.pi * t) + 0.25 + ns * fixed_noise[k],
                )
                for k, t in enumerate(t_vals)
            ]
            m = VMobject().set_points_smoothly(pts)
            m.set_stroke(WHITE, width=2.5)
            return m

        bnd = make_bnd(0)
        self.play(ShowCreation(bnd), run_time=0.8)

        for step in range(1, 11):
            frac = step / 10
            new_bnd = make_bnd(frac)
            self.play(
                Transform(bnd, new_bnd),
                epoch_tracker.animate.set_value(int(100 * frac)),
                run_time=0.5,
            )

        t_fine = np.linspace(-1.0, 2.0, 100)
        upper_r = make_shaded_region(
            plane, t_fine, boundary_curve, "upper",
            -1.5, 2.5, -1.5, 2.0, CLASS_A_COLOR, 0.12,
        )
        lower_r = make_shaded_region(
            plane, t_fine, boundary_curve, "lower",
            -1.5, 2.5, -1.5, 2.0, CLASS_B_COLOR, 0.12,
        )

        lbl_a = caption_text("Class A", scale=0.45, color=CLASS_A_COLOR)
        lbl_a.move_to(plane.c2p(0.5, 1.4))
        lbl_b = caption_text("Class B", scale=0.45, color=CLASS_B_COLOR)
        lbl_b.move_to(plane.c2p(0.5, -0.8))

        self.play(
            FadeIn(upper_r), FadeIn(lower_r),
            FadeIn(lbl_a), FadeIn(lbl_b),
            run_time=1.0,
        )
        self.wait(2.5)
