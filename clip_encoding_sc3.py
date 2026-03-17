from manimlib import *
import numpy as np


class L2NormalizationCosineSimilarity(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#0f1117"

        # Camera frame control for ManimGL
        frame = self.camera.frame

        # Start by facing the XY plane head-on
        frame.reorient(0, 0)

        # ============================================================
        # PART 1 — 2D: L2 Normalization on the Unit Circle
        # ============================================================
        plane = NumberPlane(
            x_range=(-3, 3, 1),
            y_range=(-3, 3, 1),
            height=6,
            width=6,
            axis_config={
                "stroke_color": GREY_B,
                "stroke_width": 2,
            },
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.25,
            },
        )
        # 先加入網格，讓它在最底層，不會擋住箭頭與標籤
        self.add(plane)

        unit_circle = Circle(radius=1, color=TEAL_A, stroke_width=3)
        unit_circle.set_fill(opacity=0)

        title_2d = Text(
            "L2 Normalization in Embedding Space",
            font_size=32,
            color=WHITE
        )
        title_2d.to_edge(DOWN)


        raw_vec = np.array([2.4, 1.4, 0.0])
        norm_vec = raw_vec / np.linalg.norm(raw_vec)

        raw_arrow = Arrow(
            ORIGIN, raw_vec,
            buff=0,
            stroke_width=2,
            color=BLUE
        )

        raw_label = Tex(r"\mathbf{v}", color=YELLOW)
        raw_label.scale(0.9)
        raw_label.next_to(raw_arrow.get_end(), UR, buff=0.12)

        norm_arrow_target = Arrow(
            ORIGIN, norm_vec,
            buff=0,
            stroke_width=2,
            color=GREEN_B
        )

        norm_label_target = Tex(
            r"\hat{\mathbf{v}} = \frac{\mathbf{v}}{\left\lVert \mathbf{v} \right\rVert}",
            color=GREEN_B
        )
        norm_label_target.scale(0.85)
        norm_label_target.next_to(norm_arrow_target.get_end(), UR, buff=0.12)

        self.play(ShowCreation(plane), run_time=1.2)
        self.play(ShowCreation(unit_circle), run_time=0.8)
        self.play(GrowArrow(raw_arrow), FadeIn(raw_label), run_time=1.0)
        self.wait(0.5)

        self.play(
            Transform(raw_arrow, norm_arrow_target),
            Transform(raw_label, norm_label_target),
            run_time=1.8
        )
        self.wait(1.0)
        self.play(FadeIn(title_2d, shift=UP * 0.2))
        self.wait(1.0)

        group_2d = VGroup(
            plane, unit_circle, raw_arrow, raw_label, title_2d
        )

        # Fade out 2D elements
        self.play(FadeOut(group_2d), run_time=1.0)

        # Camera transition: from 2D front view to 3D perspective
        self.play(
            ApplyMethod(frame.reorient, 135, -315),
            run_time=2.0
        )

        # ============================================================
        # PART 2 — 3D Sphere Setup
        # ============================================================
        axes = ThreeDAxes(
            x_range=(-1.4, 1.4, 1),
            y_range=(-1.4, 1.4, 1),
            z_range=(-1.4, 1.4, 1),
            width=6,
            height=6,
            depth=6,
        ) 

        sphere = Sphere(radius=1)
        sphere.set_color(BLUE_D)
        sphere.set_opacity(0.1)

        formula = Tex(
            r"\text{cosine similarity}=\cos(\theta)=\hat{\mathbf{u}}\cdot\hat{\mathbf{v}}",
            color=WHITE
        )
        formula.scale(0.85)
        formula.to_edge(UP)
        formula.fix_in_frame()

        self.play(ShowCreation(axes), run_time=1.0)
        self.play(FadeIn(sphere), run_time=1.2)

        # ============================================================
        # 左側：三張圖片 + 對應文字 (car, cat, dog) → embedding
        # ============================================================
        thumb_files = ["car_w.jpg", "cat_w.jpg", "dog_w.jpg"]
        text_labels = ["car", "cat", "dog"]
        thumb_h = 0.45
        rows = Group()
        for fname, label in zip(thumb_files, text_labels):
            img = ImageMobject(fname)
            img.set_height(thumb_h)
            txt = Text(label, font_size=20, color=WHITE)
            txt.next_to(img, DOWN, buff=0.08)
            rows.add(Group(img, txt))
        rows.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        left_panel = Group(rows)
        left_panel.to_corner(LEFT, buff=0.5).shift(DOWN * 0.4)
        left_panel.fix_in_frame()

        self.play(FadeIn(left_panel), run_time=0.8)
        self.wait(0.3)

        # ============================================================
        # PART 3 — Embedding Points on Sphere (car, cat, dog)
        # 圖片／文字分別變化成球面上的點
        # ============================================================
        def spherical_to_cartesian(theta, phi):
            return np.array([
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi)
            ])

        # 球面上位置 (theta, phi) 弧度：car/cat/dog 的 image 與 text 各一顆點
        # phi≈1 在赤道附近較好見；car_text 靠近 car_image 才會有高 cosine
        image_params = [
            (0.60, 1.05),   # car（示範用 u，箭頭先指 car_text 再移向 cat_text）
            (2.00, 1.10),   # cat
            (3.40, 1.00),   # dog
        ]
        text_params = [
            (0.95, 1.02),   # car（靠近 car image → 高 cosine）
            (2.25, 1.05),   # cat（箭頭移過來 → 低 cosine）
            (2.85, 1.12),   # dog
        ]

        dot_radius = 0.032
        # 小點放在單位球「外表面」上：半徑 1 + dot_radius，確保在 3D 球面上
        surface_radius = 1.0 + dot_radius
        image_colors = [BLUE_A, TEAL_B, BLUE_C]
        text_colors = [GREEN_A, GREEN_C, GREEN_E]

        image_dots = Group()
        text_dots = Group()

        for i, (th, ph) in enumerate(image_params):
            p = spherical_to_cartesian(th, ph)
            p_surface = surface_radius * p
            sphere = Sphere(radius=dot_radius)
            sphere.set_color(image_colors[i])
            sphere.move_to(p_surface)
            image_dots.add(sphere)

        for i, (th, ph) in enumerate(text_params):
            p = spherical_to_cartesian(th, ph)
            p_surface = surface_radius * p
            sphere = Sphere(radius=dot_radius)
            sphere.set_color(text_colors[i])
            sphere.move_to(p_surface)
            text_dots.add(sphere)

        # 每張圖片縮小淡出，同時對應的藍點在球面上淡入（圖片 → 球面點）
        for i in range(3):
            self.play(
                rows[i][0].animate.scale(0.15),
                FadeOut(rows[i][0]),
                FadeIn(image_dots[i], scale=0.5),
                run_time=0.75
            )
        self.wait(0.2)

        # 每個文字縮小淡出，同時對應的綠點在球面上淡入（文字 → 球面點）
        for i in range(3):
            self.play(
                rows[i][1].animate.scale(0.15),
                FadeOut(rows[i][1]),
                FadeIn(text_dots[i], scale=0.5),
                run_time=0.75
            )
        self.wait(0.2)

        # ============================================================
        # PART 4 — 球心一支箭頭：先指 car_text（與 car_image 高 cosine），再移到 cat_text（低 cosine）
        # ============================================================
        car_image_vec = spherical_to_cartesian(*image_params[0])
        car_text_vec = spherical_to_cartesian(*text_params[0])
        cat_text_vec = spherical_to_cartesian(*text_params[1])

        t_tracker = ValueTracker(0.0)

        def get_tip():
            t = t_tracker.get_value()
            a = car_text_vec / np.linalg.norm(car_text_vec)
            b = cat_text_vec / np.linalg.norm(cat_text_vec)
            dot_ab = np.clip(np.dot(a, b), -1, 1)
            omega = np.arccos(dot_ab)
            if omega < 1e-5:
                return a.copy()
            p = (
                np.sin((1 - t) * omega) / np.sin(omega) * a
                + np.sin(t * omega) / np.sin(omega) * b
            )
            return p / np.linalg.norm(p)

        highlight_radius = 0.038
        highlight_surface = 1.0 + highlight_radius
        car_image_surface = highlight_surface * (car_image_vec / np.linalg.norm(car_image_vec))

        def get_tip_surface():
            return highlight_surface * get_tip()

        car_image_dot = Sphere(radius=highlight_radius).set_color(GOLD_A).move_to(car_image_surface)

        # ========== 箭頭部件可調參數（約 259～287 行）==========
        def make_solid_arrow(start, end, color=GOLD_A, line_width=1, tip_size=0.01):
            direction = end - start
            length = np.linalg.norm(direction)
            if length < 1e-6:
                direction = np.array([1, 0, 0])
                length = 1.0
            unit = direction / length
            line_end = end - unit * tip_size * 0.1  # 0.9 = 箭桿與箭頭銜接比例，可調

            line = Line(start, line_end)
            line.set_stroke(color=color, width=line_width)  # line_width = 箭桿粗細

            tip = Triangle()
            tip.set_fill(color, opacity=1)
            tip.set_stroke(color, width=0)  # width=0 無邊線；改 >0 可加邊框
            tip.scale(tip_size)  # tip_size = 三角形箭頭大小
            tip.rotate(-PI / 2)
            tip.move_to(end)
            angle = angle_of_vector(unit[:2])
            tip.rotate(angle - PI / 2)

            return VGroup(line, tip)

        arrow_from_center = always_redraw(lambda: make_solid_arrow(
            ORIGIN,
            get_tip_surface(),
            color=GOLD_A,       # 箭頭顏色
            line_width=6,       # 箭桿粗細
            tip_size=0.16       # 箭頭三角形大小
        ))

        def great_circle_arc_points(a, b, n=40):
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            dot_ab = np.clip(np.dot(a, b), -1, 1)
            omega = np.arccos(dot_ab)
            if omega < 1e-5:
                return [a.copy() for _ in range(n)]
            pts = []
            for s in np.linspace(0, 1, n):
                p = (
                    np.sin((1 - s) * omega) / np.sin(omega) * a
                    + np.sin(s * omega) / np.sin(omega) * b
                )
                pts.append(p / np.linalg.norm(p))
            return pts

        arc = always_redraw(lambda: VMobject(
            stroke_color=WHITE,
            stroke_width=4
        ).set_points_smoothly(
            great_circle_arc_points(car_image_vec, get_tip(), n=60)
        ))

        theta_label = Tex(r"\theta", color=WHITE)
        theta_label.scale(0.9)
        theta_label.set_fill(WHITE, opacity=1)
        theta_label.set_stroke(WHITE, width=2)

        def update_theta_label(m):
            pts = great_circle_arc_points(car_image_vec, get_tip(), n=60)
            mid = pts[len(pts) // 2]
            m.move_to(mid * 1.12)
            return m

        theta_label.add_updater(update_theta_label)

        self.play(
            FadeIn(car_image_dot),
            FadeIn(arrow_from_center),
            run_time=1.0
        )
        self.play(ShowCreation(arc), FadeIn(theta_label), run_time=1.0)
        self.play(FadeIn(formula), run_time=0.8)

        # ============================================================
        # PART 5 — cosine = car_image · tip，箭指 car_text 時高、指 cat_text 時低
        # ============================================================
        cosine_text = Text("cosine =", font_size=24, color=WHITE)
        u_n = car_image_vec / np.linalg.norm(car_image_vec)
        cosine_value = DecimalNumber(
            np.dot(u_n, car_text_vec / np.linalg.norm(car_text_vec)),
            num_decimal_places=2
        )
        cosine_value.set_color(WHITE)

        def update_cosine(m):
            tip_n = get_tip()
            val = np.clip(np.dot(u_n, tip_n), -1, 1)
            m.set_value(val)
            return m

        cosine_value.add_updater(update_cosine)

        cosine_group = VGroup(cosine_text, cosine_value).arrange(RIGHT, buff=0.12)
        cosine_group.to_edge(DOWN)
        cosine_group.fix_in_frame()

        self.play(FadeIn(cosine_group), run_time=0.6)

        # 箭頭從指向 car_text 移到指向 cat_text，cosine 從高變低
        self.play(
            t_tracker.animate.set_value(1.0),
            run_time=3.0
        )
        self.wait(2.0)