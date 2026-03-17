from manimlib import *
import numpy as np


# ============================================================
# Palette
# ============================================================
BG_COLOR = "#0f1117"
DOG_COLOR = BLUE
CAT_COLOR = PURPLE
CAR_COLOR = ORANGE
EDGE_COLOR = "#aaaaff"
LABEL_COLOR = WHITE
DIM_COLOR = "#555577"


# ============================================================
# Helpers
# ============================================================
def random_sphere_points(n, radius=2.8, seed=42):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    r = radius * rng.random(n) ** (1 / 3)
    pts = pts * r[:, None]
    return pts


def cluster_2d_points(n, center, spread, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=spread, size=(n, 2))
    pts += np.array(center)
    return pts


def make_dot(pos, color, radius=0.055, opacity=0.92):
    d = Dot(point=np.array(pos), radius=radius, color=color)
    d.set_opacity(opacity)
    return d


def glowing_dot(pos, color, radius=0.065):
    core = Dot(point=np.array(pos), radius=radius, color=color)
    core.set_color(color).set_opacity(0.95)

    halo = Dot(point=np.array(pos), radius=radius * 2.8, color=color)
    halo.set_color(color).set_opacity(0.18)

    return VGroup(halo, core)


def title_text(s, scale=0.72, color=WHITE):
    t = Text(s, font="Helvetica Neue", color=color)
    t.scale(scale)
    return t


def caption_text(s, scale=0.42, color="#ccccdd"):
    t = Text(s, font="Helvetica Neue", color=color)
    t.scale(scale)
    return t


# ============================================================
# Main Scene
# ============================================================
class UMAPVisualizationScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        frame = self.camera.frame
        frame.reorient(0, 0)
        frame.set_width(14)

        self.scene1_highdim_cloud()
        self.scene2_compression_funnel()
        self.scene3_local_neighborhood()
        self.scene4_global_clusters()
        self.scene5_tsne_comparison()
        self.scene_final_landing()

    # --------------------------------------------------------
    # Scene 1 — High-Dim Cloud
    # --------------------------------------------------------
    def scene1_highdim_cloud(self):
        frame = self.camera.frame

        title = title_text("768-Dimensional CLIP Embedding Space", scale=0.62)
        title.to_edge(UP, buff=0.35)

        pts = random_sphere_points(250, radius=2.8, seed=42)
        rng = np.random.default_rng(7)
        colors = rng.choice([DOG_COLOR, CAT_COLOR, CAR_COLOR], size=250)

        cloud = VGroup()
        for p, c in zip(pts, colors):
            d = Dot(
                point=np.array([p[0], p[1], p[2]]),
                radius=0.055,
                color=c
            )
            d.set_opacity(0.75)
            cloud.add(d)

        self.play(
            FadeIn(title, shift=UP * 0.2),
            run_time=1.2,
        )

        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.4) for d in cloud],
                lag_ratio=0.008
            ),
            run_time=2.8,
        )

        def orbit(mob, alpha):
            # 從初始 (0, 0) 平滑旋轉到斜側視角
            mob.reorient(
                8 * alpha,    # theta: 0 → 8
                56 * alpha    # phi: 0 → 56
            )

        self.play(
            UpdateFromAlphaFunc(frame, orbit),
            run_time=4.5,
            rate_func=smooth,
        )
        self.wait(0.5)

        self._cloud_3d = cloud
        self._title1 = VGroup(title)

    # --------------------------------------------------------
    # Scene 2 — Compression / Projection
    # --------------------------------------------------------
    def scene2_compression_funnel(self):
        frame = self.camera.frame
        cloud = self._cloud_3d

        funnel_title = title_text("UMAP Projection", scale=0.62)
        funnel_title.to_edge(UP, buff=0.35)

        arrow_label = caption_text("High-Dimensional  →  2D", scale=0.42)
        arrow_label.next_to(funnel_title, DOWN, buff=0.14)

        self.play(
            FadeOut(self._title1),
            FadeIn(funnel_title, shift=UP * 0.15),
            run_time=0.7,
        )
        self.play(FadeIn(arrow_label), run_time=0.5)

        rng = np.random.default_rng(3)
        targets_2d = rng.uniform(-2.5, 2.5, (len(cloud), 2))

        start_theta = -20 + 28
        start_phi = 68 - 12

        def flatten_camera(mob, alpha):
            a = smooth(alpha)
            mob.reorient(
                interpolate(start_theta, 0, a),
                interpolate(start_phi, 0, a),
            )

        anims = [UpdateFromAlphaFunc(frame, flatten_camera)]
        for i, dot in enumerate(cloud):
            tx, ty = targets_2d[i]
            anims.append(dot.animate.move_to(np.array([tx, ty, 0])))

        self.play(*anims, run_time=3.0, rate_func=smooth)
        self.wait(0.6)

        self._cloud_2d = cloud
        self._targets_2d = targets_2d
        self._funnel_labels = VGroup(funnel_title, arrow_label)

    # --------------------------------------------------------
    # Scene 3 — Local Neighborhood
    # --------------------------------------------------------
    def scene3_local_neighborhood(self):
        cloud = self._cloud_2d
        targets_2d = self._targets_2d

        dists_to_center = np.linalg.norm(targets_2d, axis=1)
        focal_idx = int(np.argmin(dists_to_center))

        focal_pos = np.array([
            targets_2d[focal_idx, 0],
            targets_2d[focal_idx, 1],
            0
        ])

        dists = np.linalg.norm(targets_2d - targets_2d[focal_idx], axis=1)
        neighbor_indices = np.argsort(dists)[1:7]

        focal_dot = glowing_dot(focal_pos, YELLOW, radius=0.1)

        label_struct = title_text("Preserve Local Structure", scale=0.58)
        label_struct.to_edge(UP, buff=0.35)

        label_knn = caption_text(
            "k-Nearest Neighbor Graph (k = 6)",
            scale=0.40
        )
        label_knn.next_to(label_struct, DOWN, buff=0.14)

        self.play(
            FadeOut(self._funnel_labels),
            FadeIn(label_struct),
            FadeIn(label_knn),
            FadeIn(focal_dot),
            run_time=0.8,
        )

        edges = VGroup()
        for ni in neighbor_indices:
            np2 = np.array([targets_2d[ni, 0], targets_2d[ni, 1], 0])
            line = Line(
                focal_pos,
                np2,
                stroke_width=1.8,
                stroke_color=EDGE_COLOR,
                stroke_opacity=0.65
            )
            edges.add(line)

        self.play(
            LaggedStart(*[ShowCreation(e) for e in edges], lag_ratio=0.18),
            run_time=2.0,
        )

        neighbor_halos = VGroup()
        for ni in neighbor_indices:
            np2 = np.array([targets_2d[ni, 0], targets_2d[ni, 1], 0])
            h = Dot(point=np2, radius=0.13, color=YELLOW)
            h.set_opacity(0.22)
            neighbor_halos.add(h)

        self.play(FadeIn(neighbor_halos, scale=1.4), run_time=0.7)
        self.wait(1.2)

        self.play(
            FadeOut(focal_dot),
            FadeOut(neighbor_halos),
            FadeOut(edges),
            FadeOut(label_struct),
            FadeOut(label_knn),
            run_time=0.8,
        )

        self._cloud_2d = cloud
        self._targets_2d = targets_2d

    # --------------------------------------------------------
    # Scene 4 — UMAP Cluster Arrangement
    # --------------------------------------------------------
    def scene4_global_clusters(self):
        cloud = self._cloud_2d

        dog_center = np.array([-1.2, 0.6])
        cat_center = np.array([0.0, 0.6])
        car_center = np.array([2.8, -1.0])

        n_per = len(cloud) // 3

        dog_pts = cluster_2d_points(n_per, dog_center, 0.55, seed=10)
        cat_pts = cluster_2d_points(n_per, cat_center, 0.55, seed=20)
        car_pts = cluster_2d_points(
            len(cloud) - 2 * n_per,
            car_center,
            0.6,
            seed=30
        )

        all_pts = np.vstack([dog_pts, cat_pts, car_pts])
        all_colors = (
            [DOG_COLOR] * n_per +
            [CAT_COLOR] * n_per +
            [CAR_COLOR] * (len(cloud) - 2 * n_per)
        )

        anims = []
        for dot, (px, py), col in zip(cloud, all_pts, all_colors):
            anims.append(
                dot.animate
                .move_to(np.array([px, py, 0]))
                .set_color(col)
                .set_opacity(0.85)
            )

        cluster_title = title_text("UMAP: Semantic Clusters", scale=0.62)
        cluster_title.to_edge(UP, buff=0.35)

        self.play(FadeIn(cluster_title), run_time=0.5)
        self.play(
            LaggedStart(*anims, lag_ratio=0.004),
            run_time=3.2,
            rate_func=smooth,
        )

        lbl_dogs = caption_text("Dogs", scale=0.5, color=DOG_COLOR)
        lbl_cats = caption_text("Cats", scale=0.5, color=CAT_COLOR)
        lbl_cars = caption_text("Cars", scale=0.5, color=CAR_COLOR)

        lbl_dogs.move_to(np.array([dog_center[0], dog_center[1] + 0.85, 0]))
        lbl_cats.move_to(np.array([cat_center[0], cat_center[1] + 0.85, 0]))
        lbl_cars.move_to(np.array([car_center[0], car_center[1] + 0.9, 0]))

        caption = caption_text(
            "Semantically related clusters remain close",
            scale=0.40
        )
        caption.to_edge(DOWN, buff=0.38)

        ref_line = Line(
            np.array([dog_center[0], dog_center[1] + 0.55, 0]),
            np.array([cat_center[0], cat_center[1] + 0.55, 0]),
        )
        brace = Brace(ref_line, direction=UP, color="#aaaaee", buff = 0.5)
        brace.scale(0.8)

        brace_txt = caption_text("similar", scale=0.35, color="#aaaaee")
        brace_txt.next_to(brace, UP, buff=0.1)

        self.play(
            FadeIn(lbl_dogs),
            FadeIn(lbl_cats),
            FadeIn(lbl_cars),
            run_time=0.7,
        )
        self.play(
            FadeIn(brace),
            FadeIn(brace_txt),
            FadeIn(caption, shift=UP * 0.1),
            run_time=0.8,
        )
        self.wait(1.8)

        self._cluster_cloud = cloud
        self._cluster_pts = all_pts
        self._cluster_colors = all_colors
        self._dog_center = dog_center
        self._cat_center = cat_center
        self._car_center = car_center
        self._scene4_labels = VGroup(
            cluster_title, lbl_dogs, lbl_cats, lbl_cars,
            brace, brace_txt, caption
        )

    # --------------------------------------------------------
    # Scene 5 — t-SNE vs UMAP
    # --------------------------------------------------------
    def scene5_tsne_comparison(self):
        cloud = self._cluster_cloud
        all_colors = self._cluster_colors

        self.play(FadeOut(self._scene4_labels), run_time=0.6)

        offset = 2.6

        tsne_dog_center = np.array([-offset - 1.2, 0.5])
        tsne_cat_center = np.array([-offset - 1.2, -1.5])
        tsne_car_center = np.array([-offset + 0.2, 0.5])

        n_per = len(cloud) // 3

        tsne_dog_pts = cluster_2d_points(n_per, tsne_dog_center, 0.48, seed=10)
        tsne_cat_pts = cluster_2d_points(n_per, tsne_cat_center, 0.48, seed=20)
        tsne_car_pts = cluster_2d_points(
            len(cloud) - 2 * n_per,
            tsne_car_center,
            0.5,
            seed=30
        )
        tsne_pts = np.vstack([tsne_dog_pts, tsne_cat_pts, tsne_car_pts])

        umap_dog_center = self._dog_center + np.array([offset, 0])
        umap_cat_center = self._cat_center + np.array([offset, 0])
        umap_car_center = self._car_center + np.array([offset, 0])

        umap_dog_pts = cluster_2d_points(n_per, umap_dog_center, 0.5, seed=10)
        umap_cat_pts = cluster_2d_points(n_per, umap_cat_center, 0.5, seed=20)
        umap_car_pts = cluster_2d_points(
            len(cloud) - 2 * n_per,
            umap_car_center,
            0.55,
            seed=30
        )
        umap_pts = np.vstack([umap_dog_pts, umap_cat_pts, umap_car_pts])

        tsne_header = title_text("t-SNE Projection", scale=0.55, color="#ddddee")
        umap_header = title_text("UMAP Projection", scale=0.55, color="#ddddee")
        tsne_header.move_to(np.array([-offset, 2.7, 0]))
        umap_header.move_to(np.array([offset, 2.7, 0]))

        divider = Line(
            UP * 3.0,
            DOWN * 3.0,
            stroke_color="#444466",
            stroke_width=1.5
        )

        self.play(
            FadeIn(tsne_header),
            FadeIn(umap_header),
            FadeIn(divider),
            run_time=0.7,
        )

        anims = []
        for i, dot in enumerate(cloud):
            tx, ty = tsne_pts[i]
            anims.append(dot.animate.move_to(np.array([tx, ty, 0])))

        self.play(
            LaggedStart(*anims, lag_ratio=0.004),
            run_time=2.5,
            rate_func=smooth,
        )

        umap_dots = VGroup()
        for i, col in enumerate(all_colors):
            ux, uy = umap_pts[i]
            d = Dot(point=np.array([ux, uy, 0]), radius=0.055, color=col)
            d.set_color(col).set_opacity(0.85)
            umap_dots.add(d)

        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.3) for d in umap_dots],
                lag_ratio=0.006
            ),
            run_time=2.2,
        )

        tl_dog = caption_text("Dogs", scale=0.42, color=DOG_COLOR)
        tl_cat = caption_text("Cats", scale=0.42, color=CAT_COLOR)
        tl_car = caption_text("Cars", scale=0.42, color=CAR_COLOR)
        tl_dog.move_to(np.array([tsne_dog_center[0], tsne_dog_center[1] + 0.72, 0]))
        tl_cat.move_to(np.array([tsne_cat_center[0], tsne_cat_center[1] + 0.72, 0]))
        tl_car.move_to(np.array([tsne_car_center[0], tsne_car_center[1] + 0.72, 0]))

        ul_dog = caption_text("Dogs", scale=0.42, color=DOG_COLOR)
        ul_cat = caption_text("Cats", scale=0.42, color=CAT_COLOR)
        ul_car = caption_text("Cars", scale=0.42, color=CAR_COLOR)
        ul_dog.move_to(np.array([umap_dog_center[0], umap_dog_center[1] + 0.72, 0]))
        ul_cat.move_to(np.array([umap_cat_center[0], umap_cat_center[1] + 0.72, 0]))
        ul_car.move_to(np.array([umap_car_center[0], umap_car_center[1] + 0.72, 0]))

        self.play(
            FadeIn(VGroup(tl_dog, tl_cat, tl_car)),
            FadeIn(VGroup(ul_dog, ul_cat, ul_car)),
            run_time=0.8,
        )
        self.wait(2.5)

        self.play(
            FadeOut(VGroup(
                cloud, umap_dots,
                tsne_header, umap_header, divider,
                tl_dog, tl_cat, tl_car,
                ul_dog, ul_cat, ul_car,
            )),
            run_time=0.9,
        )

    # --------------------------------------------------------
    # Final Scene
    # --------------------------------------------------------
    def scene_final_landing(self):
        dog_center = self._dog_center
        cat_center = self._cat_center
        car_center = self._car_center

        n_flying = 36
        rng = np.random.default_rng(55)
        categories = rng.choice(["dog", "cat", "car"], size=n_flying)

        color_map = {"dog": DOG_COLOR, "cat": CAT_COLOR, "car": CAR_COLOR}
        center_map = {"dog": dog_center, "cat": cat_center, "car": car_center}

        final_title = title_text("UMAP Projection of CLIP Embeddings", scale=0.65)
        final_title.to_edge(UP, buff=0.35)

        self.play(FadeIn(final_title, shift=UP * 0.15), run_time=0.7)

        n_bg = 220
        n_per = n_bg // 3

        bg_dog = cluster_2d_points(n_per, dog_center, 0.5, seed=10)
        bg_cat = cluster_2d_points(n_per, cat_center, 0.5, seed=20)
        bg_car = cluster_2d_points(n_bg - 2 * n_per, car_center, 0.55, seed=30)

        bg_pts = np.vstack([bg_dog, bg_cat, bg_car])
        bg_colors = (
            [DOG_COLOR] * n_per +
            [CAT_COLOR] * n_per +
            [CAR_COLOR] * (n_bg - 2 * n_per)
        )

        bg_cloud = VGroup()
        for (px, py), col in zip(bg_pts, bg_colors):
            d = Dot(point=np.array([px, py, 0]), radius=0.055, color=col)
            d.set_color(col).set_opacity(0.8)
            bg_cloud.add(d)

        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.3) for d in bg_cloud],
                lag_ratio=0.006
            ),
            run_time=1.8,
        )

        screen_w, screen_h = 8.5, 5.0
        flying_anims = []
        flying_dots = []

        rng2 = np.random.default_rng(1234)

        for cat in categories:
            col = color_map[cat]
            ctr = center_map[cat]

            lx = ctr[0] + rng2.normal(scale=0.45)
            ly = ctr[1] + rng2.normal(scale=0.45)

            edge = rng2.choice(["L", "R", "T", "B"])
            if edge == "L":
                sx, sy = -screen_w, rng2.uniform(-screen_h, screen_h)
            elif edge == "R":
                sx, sy = screen_w, rng2.uniform(-screen_h, screen_h)
            elif edge == "T":
                sx, sy = rng2.uniform(-screen_w, screen_w), screen_h
            else:
                sx, sy = rng2.uniform(-screen_w, screen_w), -screen_h

            gd = glowing_dot(np.array([sx, sy, 0]), col, radius=0.08)
            flying_dots.append(gd)
            flying_anims.append(gd.animate.move_to(np.array([lx, ly, 0])))

        for gd in flying_dots:
            self.add(gd)

        self.play(
            LaggedStart(*flying_anims, lag_ratio=0.07),
            run_time=3.5,
            rate_func=smooth,
        )

        lbl_d = caption_text("Dogs", scale=0.5, color=DOG_COLOR)
        lbl_c = caption_text("Cats", scale=0.5, color=CAT_COLOR)
        lbl_r = caption_text("Cars", scale=0.5, color=CAR_COLOR)

        lbl_d.move_to(np.array([dog_center[0], dog_center[1] + 0.85, 0]))
        lbl_c.move_to(np.array([cat_center[0], cat_center[1] + 0.85, 0]))
        lbl_r.move_to(np.array([car_center[0], car_center[1] + 0.9, 0]))

        self.play(
            FadeIn(lbl_d),
            FadeIn(lbl_c),
            FadeIn(lbl_r),
            run_time=0.8,
        )
        self.wait(2.5)