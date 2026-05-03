from manimlib import *
from pathlib import Path
from PIL import Image


class RetinaFacePreprocessScene(Scene):
    def tex_escape(self, text):
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in str(text))

    def make_rgb_channel_files(self, source="croped.png", size=336):
        output_dir = Path(__file__).with_name(".generated") / "rgb_channels"
        output_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(__file__).with_name(source)
        image = Image.open(source_path).convert("RGB")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((size, size), resample)
        r, g, b = image.split()

        channel_specs = [
            ("R", Image.merge("RGB", (r, Image.new("L", image.size), Image.new("L", image.size)))),
            ("G", Image.merge("RGB", (Image.new("L", image.size), g, Image.new("L", image.size)))),
            ("B", Image.merge("RGB", (Image.new("L", image.size), Image.new("L", image.size), b))),
        ]

        paths = {}
        for name, channel_image in channel_specs:
            path = output_dir / f"croped_{name}.png"
            channel_image.save(path)
            paths[name] = str(path)
        return paths

    def make_patch_channel_files(self, source="mid.png", size=336, patch_size=14, row=12, col=12):
        output_dir = Path(__file__).with_name(".generated") / "vit_patch"
        output_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(__file__).with_name(source)
        image = Image.open(source_path).convert("RGB")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        if source_path.name == "mid.png":
            side = min(image.size)
            left = (image.width - side) // 2
            upper = (image.height - side) // 2
            patch = image.crop((left, upper, left + side, upper + side))
        else:
            image = image.resize((size, size), resample)
            left = col * patch_size
            upper = row * patch_size
            patch = image.crop((left, upper, left + patch_size, upper + patch_size))
        r, g, b = patch.split()

        paths = {}
        patch_path = output_dir / "patch_rgb.png"
        patch.save(patch_path)
        paths["RGB"] = str(patch_path)

        channel_specs = [
            ("R", Image.merge("RGB", (r, Image.new("L", patch.size), Image.new("L", patch.size)))),
            ("G", Image.merge("RGB", (Image.new("L", patch.size), g, Image.new("L", patch.size)))),
            ("B", Image.merge("RGB", (Image.new("L", patch.size), Image.new("L", patch.size), b))),
        ]
        for name, channel_image in channel_specs:
            path = output_dir / f"patch_{name}.png"
            channel_image.save(path)
            paths[name] = str(path)
        return paths

    def mono(self, text, font_size=22, color=WHITE, **kwargs):
        return TexText(
            self.tex_escape(text),
            font_size=font_size,
            color=color,
            **kwargs,
        )

    def tech_mono(self, text, font_size=22, color=WHITE, **kwargs):
        return TexText(
            r"\texttt{" + self.tex_escape(text) + "}",
            font_size=font_size,
            color=color,
            **kwargs,
        )

    def latex_text(self, text, font_size=22, color=WHITE, **kwargs):
        return TexText(
            text,
            font_size=font_size,
            color=color,
            **kwargs,
        )

    def label_stack(self, lines, font_size=20, color=WHITE, buff=0.08):
        labels = VGroup(*[
            self.mono(line, font_size=font_size, color=color)
            for line in lines
        ])
        labels.arrange(DOWN, buff=buff)
        return labels

    def technical_tag(self, lines, font_size=18, width=1.85):
        text = VGroup(*[
            self.tech_mono(line, font_size=font_size, color=WHITE)
            for line in lines
        ])
        text.arrange(DOWN, buff=0.05)
        bg = RoundedRectangle(
            width=width,
            height=text.get_height() + 0.28,
            corner_radius=0.08,
            stroke_color=WHITE,
            stroke_width=1.2,
            fill_color=BLACK,
            fill_opacity=0.72,
        )
        text.move_to(bg)
        return VGroup(bg, text)

    def block(
        self,
        title,
        lines,
        width=2.2,
        height=1.15,
        fill_color=GREY_E,
        fill_opacity=0.85,
        stroke_color=GREY_B,
        text_color=WHITE,
        detail_color=GREY_A,
    ):
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.12,
            stroke_color=stroke_color,
            stroke_width=1.7,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
        )
        title_mob = self.mono(title, font_size=23, color=text_color)
        detail = self.label_stack(lines, font_size=15, color=detail_color, buff=0.05)
        content = VGroup(title_mob, detail)
        content.arrange(DOWN, buff=0.13)
        content.move_to(box)
        return VGroup(box, content)

    def deterministic_block(self, title, lines, width=2.45, height=1.2):
        return self.block(
            title,
            lines,
            width=width,
            height=height,
            fill_color=WHITE,
            fill_opacity=0.94,
            stroke_color=WHITE,
            text_color=BLACK,
            detail_color=GREY_E,
        )

    def arrow_between(self, left, right, color=GREY_A, buff=0.12):
        return Arrow(
            left.get_right() + RIGHT * buff,
            right.get_left() - RIGHT * buff,
            buff=0,
            fill_color=color,
            stroke_width=2.0,
            thickness=0.025,
            max_tip_length_to_length_ratio=0.12,
        )

    def make_face_crop(self, side=1.95, show_text=True):
        crop_image = ImageMobject("croped.png")
        crop_image.set_height(side)
        if crop_image.get_width() > side:
            crop_image.set_width(side)

        frame = Square(side_length=side + 0.05)
        frame.set_fill("#111318", opacity=0.24)
        frame.set_stroke(WHITE, width=1.7, opacity=0.88)
        frame.move_to(crop_image)

        if not show_text:
            return Group(crop_image, frame)

        label_bg = RoundedRectangle(
            width=side * 0.74,
            height=0.28,
            corner_radius=0.05,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=0.68,
        )
        label_bg.move_to(frame.get_bottom() + UP * 0.2)
        label = self.mono("Face Crop", font_size=15, color=WHITE)
        label.move_to(label_bg)

        return Group(crop_image, frame, label_bg, label)

    def make_tensor_stack(self, side=1.65, channel_files=None):
        if channel_files is None:
            channel_files = self.make_rgb_channel_files()

        stack = Group()
        specs = [
            ("B", BLUE_B, RIGHT * 0.24 + UP * 0.18),
            ("G", GREEN_B, RIGHT * 0.12 + UP * 0.09),
            ("R", RED_B, ORIGIN),
        ]
        for channel, color, offset in specs:
            channel_image = ImageMobject(channel_files[channel])
            channel_image.set_height(side)
            sq = Square(side_length=side)
            sq.set_fill(BLACK, opacity=0)
            sq.set_stroke(color, width=1.6, opacity=0.86)
            sq.shift(offset)
            channel_image.move_to(sq)
            label = self.mono(channel, font_size=18, color=color)
            label.move_to(sq.get_corner(UL) + RIGHT * 0.17 + DOWN * 0.17)
            stack.add(Group(channel_image, sq, label))
        shape = self.label_stack(
            ["CLIP-READY TENSOR", "(3, 336, 336)", "normalized RGB"],
            font_size=16,
            color=WHITE,
            buff=0.06,
        )
        shape.next_to(stack, DOWN, buff=0.22)
        chw = self.mono("C x H x W = 3 x 336 x 336", font_size=15, color=GREY_A)
        chw.next_to(shape, DOWN, buff=0.12)
        return Group(stack, shape, chw)

    def make_resize_grid(self, square_mob, rows=6, cols=6, color=WHITE):
        lines = VGroup()
        left = square_mob.get_left()[0]
        right = square_mob.get_right()[0]
        bottom = square_mob.get_bottom()[1]
        top = square_mob.get_top()[1]

        for i in range(1, cols):
            x = left + (right - left) * i / cols
            line = Line([x, bottom, 0], [x, top, 0])
            line.set_stroke(color, width=0.8, opacity=0.45)
            lines.add(line)

        for i in range(1, rows):
            y = bottom + (top - bottom) * i / rows
            line = Line([left, y, 0], [right, y, 0])
            line.set_stroke(color, width=0.8, opacity=0.45)
            lines.add(line)

        return lines

    def make_rgb_channels_row(self, channel_files, side=0.92):
        channels = Group()
        specs = [
            ("R", RED_B, channel_files["R"]),
            ("G", GREEN_B, channel_files["G"]),
            ("B", BLUE_B, channel_files["B"]),
        ]
        for name, color, path in specs:
            image = ImageMobject(path)
            image.set_height(side)
            frame = Square(side_length=side + 0.04)
            frame.set_stroke(color, width=1.4, opacity=0.9)
            frame.move_to(image)
            grid = self.make_resize_grid(frame, rows=7, cols=7, color=WHITE)
            grid.set_stroke(opacity=0.32)
            label = Tex(rf"\text{{{name}}}", font_size=18, color=color)
            label.next_to(frame, DOWN, buff=0.07)
            channels.add(Group(image, frame, grid, label))
        channels.arrange(RIGHT, buff=0.32)
        return channels

    def make_patch_stack(self, patch_files, side=1.22):
        stack = Group()
        channel_specs = [
            ("B", BLUE_B, RIGHT * 0.42 + UP * 0.32),
            ("G", GREEN_B, RIGHT * 0.28 + UP * 0.21),
            ("R", RED_B, RIGHT * 0.14 + UP * 0.10),
        ]
        for channel, color, offset in channel_specs:
            channel_image = ImageMobject(patch_files[channel])
            channel_image.set_height(side)
            frame = Square(side_length=side)
            frame.set_fill(BLACK, opacity=0.12)
            frame.set_stroke(color, width=1.4, opacity=0.9)
            frame.shift(offset)
            channel_image.move_to(frame)
            label = self.mono(channel, font_size=15, color=color)
            label.move_to(frame.get_corner(UR) + LEFT * 0.15 + DOWN * 0.15)
            stack.add(Group(channel_image, frame, label))

        patch_image = ImageMobject(patch_files["RGB"])
        patch_image.set_height(side)
        patch_frame = Square(side_length=side)
        patch_frame.set_fill(BLACK, opacity=0)
        patch_frame.set_stroke(WHITE, width=1.9, opacity=0.95)
        patch_image.move_to(patch_frame)

        return Group(stack, Group(patch_image, patch_frame))

    def make_patch_dimension_marks(self, patch_front, color=YELLOW_B):
        frame = patch_front[1]
        width_label = Tex(r"14", font_size=28, color=color)
        width_label.next_to(frame, DOWN, buff=0.1)
        height_label = Tex(r"14", font_size=28, color=color)
        height_label.next_to(frame, LEFT, buff=0.12)
        return VGroup(width_label, height_label)

    def make_channel_depth_mark(self, channel_stack, color=WHITE):
        corners = [
            channel_stack[2][1].get_corner(UR),
            channel_stack[1][1].get_corner(UR),
            channel_stack[0][1].get_corner(UR),
        ]
        line = VMobject()
        line.set_points_as_corners(corners)
        line.set_stroke(color, width=1.6, opacity=0.95)

        label = Tex(r"3", font_size=28, color=color)
        label.next_to(line, RIGHT, buff=0.12)
        label.shift(UP * 0.02)
        return VGroup(line, label)

    def make_patch_sweep_grid(self, square_mob, rows=24, cols=24, color=YELLOW_B):
        cells = VGroup()
        left = square_mob.get_left()[0]
        right = square_mob.get_right()[0]
        bottom = square_mob.get_bottom()[1]
        top = square_mob.get_top()[1]
        cell_w = (right - left) / cols
        cell_h = (top - bottom) / rows

        for i in range(rows):
            for j in range(cols):
                cell = Rectangle(width=cell_w, height=cell_h)
                cell.move_to([
                    left + (j + 0.5) * cell_w,
                    top - (i + 0.5) * cell_h,
                    0,
                ])
                cell.set_fill(color, opacity=0)
                cell.set_stroke(color, width=0.45, opacity=0)
                cells.add(cell)
        return cells

    def make_patch_pixel_grid(self, patch_front, rows=14, cols=14, color=YELLOW_B):
        frame = patch_front[1]
        cells = VGroup()
        left = frame.get_left()[0]
        right = frame.get_right()[0]
        bottom = frame.get_bottom()[1]
        top = frame.get_top()[1]
        cell_w = (right - left) / cols
        cell_h = (top - bottom) / rows

        for i in range(rows):
            for j in range(cols):
                cell = Rectangle(width=cell_w, height=cell_h)
                cell.move_to([
                    left + (j + 0.5) * cell_w,
                    top - (i + 0.5) * cell_h,
                    0,
                ])
                cell.set_fill(YELLOW_B, opacity=0.05)
                cell.set_stroke(color, width=0.55, opacity=0.7)
                cells.add(cell)
        return cells

    def make_patch_value_vector(self):
        visible_values = [
            "0.12", "-0.48", "0.31", r"\ldots", "0.07",
        ]
        entries = Tex(
            r"\left[" + r",\ ".join(visible_values) + r"\right]",
            font_size=27,
            color=WHITE,
        )
        label = Tex(r"\text{linear projection}\ \rightarrow\ 1024\ \text{dims}", font_size=20, color=GREY_A)
        label.next_to(entries, DOWN, buff=0.18)
        return VGroup(entries, label)

    def make_patch_vector(self):
        card = RoundedRectangle(
            width=4.25,
            height=1.35,
            corner_radius=0.1,
            stroke_color=WHITE,
            stroke_width=1.4,
            fill_color="#101217",
            fill_opacity=0.94,
        )
        vector = Tex(r"[0.12,\ -0.48,\ \ldots,\ 0.07]", font_size=34, color=WHITE)
        vector.move_to(card.get_center() + UP * 0.16)
        label = Tex(r"1024\ \text{dims}", font_size=26, color=GREY_A)
        label.move_to(card.get_center() + DOWN * 0.42)
        return VGroup(card, vector, label)

    def make_photo_thumb(self, height=0.95):
        photo = ImageMobject("rf_example.png")
        photo.set_height(height)
        frame = SurroundingRectangle(photo, buff=0.02)
        frame.set_stroke(WHITE, width=1.0, opacity=0.75)
        label = self.mono("input image", font_size=13, color=GREY_A)
        label.next_to(frame, DOWN, buff=0.08)
        return Group(photo, frame, label)

    def construct(self):
        self.camera.background_color = BLACK

        # PART 1 -- RetinaFace: detect & crop
        photo = ImageMobject("rf_example.png")
        photo.set_height(3.9)
        photo.move_to(UP * 0.2)
        photo_frame = SurroundingRectangle(photo, buff=0.04)
        photo_frame.set_stroke(WHITE, width=1.15, opacity=0.72)

        input_label = Tex(r"\text{Input Image}", font_size=22, color=WHITE)
        input_label.next_to(photo_frame, UP, buff=0.22)

        retinaface_frame = SurroundingRectangle(photo, buff=0.16)
        retinaface_frame.set_fill(GREY_E, opacity=0.045)
        retinaface_frame.set_stroke(GREY_B, width=0.9, opacity=0.9)
        retinaface_tag = Tex(r"\text{RetinaFace}", font_size=24, color=GREY_A)
        retinaface_tag.next_to(retinaface_frame, DOWN, buff=0.12)

        detect_color = YELLOW_B
        expand_color = BLUE_B
        crop_color = GREEN_B

        bbox = Rectangle(
            width=photo.get_width() * 0.18,
            height=photo.get_height() * 0.30,
            stroke_color=detect_color,
            stroke_width=3.0,
        )
        bbox.move_to(
            photo.get_center()
            + RIGHT * photo.get_width() * 0.045
            + UP * photo.get_height() * 0.115
        )
        bbox_tag = self.technical_tag(["face", "score: 0.97"], font_size=13, width=1.18)
        bbox_tag.next_to(bbox, UP, buff=0.06)

        expanded_bbox = Rectangle(
            width=bbox.get_width() * 1.5,
            height=bbox.get_height() * 1.5,
            stroke_color=expand_color,
            stroke_width=3.0,
        )
        expanded_bbox.move_to(bbox)

        square_bbox = Square(side_length=expanded_bbox.get_height())
        square_bbox.set_stroke(crop_color, width=3.2)
        square_bbox.move_to(expanded_bbox)

        left_bbox = Rectangle(
            width=photo.get_width() * 0.15,
            height=photo.get_height() * 0.24,
            stroke_color=GREY_B,
            stroke_width=2.0,
        )
        left_bbox.set_stroke(opacity=0.82)
        left_bbox.move_to(
            photo.get_center()
            + LEFT * photo.get_width() * 0.405
            + UP * photo.get_height() * 0.075
        )
        left_tag = self.technical_tag(["face", "score: 0.61"], font_size=12, width=1.18)
        left_tag.next_to(left_bbox, UP, buff=0.06)

        right_bbox = Rectangle(
            width=photo.get_width() * 0.15,
            height=photo.get_height() * 0.24,
            stroke_color=GREY_B,
            stroke_width=2.0,
        )
        right_bbox.set_stroke(opacity=0.82)
        right_bbox.move_to(
            photo.get_center()
            + RIGHT * photo.get_width() * 0.295
            + UP * photo.get_height() * 0.085
        )
        right_tag = self.technical_tag(["face", "score: 0.74"], font_size=12, width=1.05)
        right_tag.next_to(right_bbox, UP, buff=0.06)

        low_score_detections = VGroup(left_bbox, left_tag, right_bbox, right_tag)

        step_1 = TexText("detect highest-confidence face", font_size=23, color=detect_color)
        step_2 = TexText(r"expand bbox by \(+50\%\) margin", font_size=23, color=expand_color)
        step_3 = TexText("square-crop", font_size=23, color=crop_color)
        for step in (step_1, step_2, step_3):
            step.next_to(retinaface_tag, DOWN, buff=0.24)

        crop_final_center = RIGHT * 4.75 + DOWN * 0.42
        crop_final_side = 1.58
        crop = self.make_face_crop(side=square_bbox.get_height(), show_text=False)
        crop.move_to(square_bbox)
        crop_label = self.mono("Face Crop", font_size=18, color=WHITE)
        crop_label.move_to(crop_final_center + UP * (crop_final_side / 2 + 0.34))
        crop_detail = self.mono("(square, arbitrary pixel)", font_size=15, color=GREY_B)
        crop_detail.move_to(crop_final_center + DOWN * (crop_final_side / 2 + 0.28))

        failure = self.mono(
            r"Failure path: no face found -> API returns 400 -> pipeline halts",
            font_size=17,
            color=GREY_B,
        )
        failure.to_edge(DOWN, buff=0.34)

        part_1 = Group(
            photo,
            photo_frame,
            input_label,
            retinaface_frame,
            retinaface_tag,
            bbox,
            bbox_tag,
            low_score_detections,
            step_1,
            step_2,
            step_3,
            crop,
            crop_label,
            crop_detail,
            failure,
        )

        self.play(
            FadeIn(photo, shift=UP * 0.12),
            FadeIn(retinaface_frame),
            FadeIn(retinaface_tag, shift=DOWN * 0.05),
            ShowCreation(photo_frame),
            FadeIn(input_label, shift=UP * 0.08),
            run_time=1.15,
        )
        self.play(
            retinaface_frame.animate.set_stroke(detect_color, width=2.8, opacity=1.0),
            FadeIn(step_1, shift=UP * 0.06),
            run_time=0.45,
        )
        self.play(
            ShowCreation(low_score_detections[0]),
            FadeIn(low_score_detections[1], shift=UP * 0.05),
            ShowCreation(low_score_detections[2]),
            FadeIn(low_score_detections[3], shift=UP * 0.05),
            ShowCreation(bbox),
            FadeIn(bbox_tag, shift=RIGHT * 0.08),
            run_time=0.95,
        )
        self.wait(0.25)
        self.play(
            retinaface_frame.animate.set_stroke(expand_color, width=2.8, opacity=1.0),
            FadeOut(step_1, shift=UP * 0.04),
            FadeIn(step_2, shift=UP * 0.06),
            Transform(bbox, expanded_bbox),
            bbox_tag.animate.next_to(expanded_bbox, UP, buff=0.06),
            run_time=0.85,
        )
        self.wait(0.2)
        self.play(
            retinaface_frame.animate.set_stroke(crop_color, width=2.8, opacity=1.0),
            FadeOut(step_2, shift=UP * 0.04),
            FadeIn(step_3, shift=UP * 0.06),
            Transform(bbox, square_bbox),
            bbox_tag.animate.next_to(square_bbox, UP, buff=0.06),
            run_time=0.85,
        )
        self.wait(0.2)
        self.play(
            FadeIn(crop[0]),
            ReplacementTransform(bbox.copy(), crop[1]),
            FadeOut(bbox_tag),
            run_time=0.75,
        )
        self.play(
            crop.animate.scale(crop_final_side / square_bbox.get_height()).move_to(crop_final_center),
            FadeOut(step_3, shift=UP * 0.04),
            FadeIn(crop_label, shift=UP * 0.08),
            FadeIn(crop_detail, shift=UP * 0.08),
            run_time=1.05,
        )
        self.play(FadeIn(failure, shift=UP * 0.08), run_time=0.55)
        self.wait(1.2)

        # PART 2 -- Preprocessing before the CLIP encoder
        title_2 = TexText(
            "Preprocessing before CLIP encoder",
            font_size=34,
            color=WHITE,
        )
        title_2.to_edge(UP, buff=0.42)

        channel_files = self.make_rgb_channel_files()
        resize_color = BLUE_B
        normalize_color = GREEN_B
        preprocess_center = ORIGIN + UP * 0.34
        preprocess_side = 2.16

        self.play(
            FadeOut(Group(photo, photo_frame, input_label)),
            FadeOut(VGroup(retinaface_frame, retinaface_tag, bbox, low_score_detections, crop_label, crop_detail, failure)),
            crop.animate.scale(preprocess_side / crop_final_side).move_to(preprocess_center),
            FadeIn(title_2, shift=UP * 0.08),
            run_time=1.15,
        )

        preprocess_frame = SurroundingRectangle(crop, buff=0.18)
        preprocess_frame.set_fill(GREY_E, opacity=0.045)
        preprocess_frame.set_stroke(resize_color, width=1.2, opacity=0.95)
        phase_label = TexText("resize", font_size=25, color=resize_color)
        phase_label.next_to(preprocess_frame, DOWN, buff=0.12)
        resize_caption = TexText("bilinear resize to 336x336", font_size=20, color=GREY_A)
        resize_caption.next_to(phase_label, DOWN, buff=0.12)

        self.play(
            FadeIn(preprocess_frame),
            FadeIn(phase_label, shift=UP * 0.06),
            FadeIn(resize_caption, shift=UP * 0.06),
            run_time=0.65,
        )

        resize_grid = self.make_resize_grid(crop[1], rows=7, cols=7, color=WHITE)
        self.play(ShowCreation(resize_grid, lag_ratio=0.08), run_time=0.95)
        self.wait(0.2)

        crop_with_grid = Group(crop, resize_grid)
        target_crop_center = UP * 1.22
        crop_with_grid_target = crop_with_grid.copy()
        crop_with_grid_target.move_to(target_crop_center)
        preprocess_frame_target = SurroundingRectangle(crop_with_grid_target, buff=0.18)
        preprocess_frame_target.set_fill(GREY_E, opacity=0.045)
        preprocess_frame_target.set_stroke(normalize_color, width=1.2, opacity=0.95)
        normalize_label = TexText("normalize", font_size=25, color=normalize_color)
        normalize_label.next_to(preprocess_frame_target, DOWN, buff=0.12)
        normalize_caption = TexText("RGB mean/std normalization", font_size=20, color=GREY_A)
        normalize_caption.next_to(normalize_label, DOWN, buff=0.12)

        self.play(
            crop_with_grid.animate.move_to(target_crop_center),
            Transform(preprocess_frame, preprocess_frame_target),
            Transform(phase_label, normalize_label),
            Transform(resize_caption, normalize_caption),
            run_time=0.9,
        )

        rgb_row = self.make_rgb_channels_row(channel_files, side=0.88)
        rgb_row.move_to(DOWN * 1.45)
        tensor_label = self.label_stack(
            ["CLIP-ready tensor", "(3, 336, 336)", "normalized RGB"],
            font_size=17,
            color=WHITE,
            buff=0.07,
        )
        tensor_label.next_to(rgb_row, DOWN, buff=0.18)

        self.play(
            FadeIn(rgb_row, shift=DOWN * 0.16),
            FadeIn(tensor_label, shift=UP * 0.08),
            run_time=0.95,
        )

        self.wait(1.0)

        # PART 3 -- CLIP ViT patchify
        patch_files = self.make_patch_channel_files()
        clip_color = BLUE_B

        clip_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=clip_color,
            stroke_width=1.2,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        clip_frame.move_to(DOWN * 0.04)
        clip_title = Tex(r"\text{CLIP ViT-L/14 @ 336px}", font_size=36, color=WHITE)
        clip_title.move_to(clip_frame.get_top() + DOWN * 0.27)
        clip_subtitle = Tex(r"\text{(frozen, }\sim\text{ 300M params)}", font_size=22, color=GREY_A)
        clip_subtitle.next_to(clip_frame, DOWN, buff=0.12)

        cell_side = crop[1].get_width() / 7
        selected_patch = Square(side_length=cell_side)
        selected_patch.set_fill(YELLOW_B, opacity=0.16)
        selected_patch.set_stroke(YELLOW_B, width=2.1, opacity=1.0)
        selected_patch.move_to(crop[1].get_center())

        patch_stack = self.make_patch_stack(patch_files, side=1.22)
        patch_stack.move_to(LEFT * 2.45 + DOWN * 0.06)
        patch_dims = self.make_patch_dimension_marks(patch_stack[1], color=YELLOW_B)
        channel_depth_mark = self.make_channel_depth_mark(patch_stack[0], color=YELLOW_B)

        patch_counter_label = Tex(r"\text{patch number:}", font_size=24, color=WHITE)
        counter_digit_templates = {
            char: Tex(char, font_size=34, color=YELLOW_B)
            for char in "0123456789"
        }
        patch_counter_number = VGroup()

        def set_patch_counter_number(value):
            chars = str(int(value))
            patch_counter_number.set_submobjects([
                counter_digit_templates[char].copy()
                for char in chars
            ])
            patch_counter_number.arrange(RIGHT, buff=0.015)

        set_patch_counter_number(0)
        patch_counter = VGroup(patch_counter_label, patch_counter_number)
        patch_counter.arrange(RIGHT, buff=0.16)
        patch_counter.next_to(patch_stack, DOWN, buff=0.46)

        cls_token = Tex(r"+\ \texttt{[CLS] token}", font_size=27, color=YELLOW_B)
        cls_token.next_to(patch_counter, DOWN, buff=0.18)

        patch_pixel_grid = self.make_patch_pixel_grid(patch_stack[1], rows=14, cols=14, color=YELLOW_B)
        patch_value_vector = self.make_patch_value_vector()
        patch_value_vector.move_to(LEFT * 2.45 + DOWN * 0.02)

        self.play(
            FadeIn(clip_frame),
            FadeIn(clip_title, shift=UP * 0.06),
            FadeIn(clip_subtitle, shift=UP * 0.04),
            ShowCreation(selected_patch),
            run_time=0.9,
        )
        self.play(
            FadeOut(Group(title_2, preprocess_frame, phase_label, resize_caption, rgb_row, tensor_label)),
            crop_with_grid.animate.move_to(RIGHT * 2.45 + DOWN * 0.06),
            ReplacementTransform(selected_patch, patch_stack[1][1]),
            FadeIn(patch_stack[1][0]),
            FadeIn(patch_stack[0], shift=RIGHT * 0.18 + UP * 0.12),
            run_time=1.05,
        )
        patch_sweep_grid = self.make_patch_sweep_grid(crop[1], rows=24, cols=24, color=YELLOW_B)

        def light_patch_cells(cells, alpha):
            total = len(cells)
            lit_count = min(total, int(np.floor(alpha * total)))
            for index, cell in enumerate(cells):
                if index < lit_count:
                    opacity = 0.48 if index == lit_count - 1 else 0.16
                    stroke_opacity = 0.75 if index == lit_count - 1 else 0.18
                else:
                    opacity = 0
                    stroke_opacity = 0
                cell.set_fill(YELLOW_B, opacity=opacity)
                cell.set_stroke(YELLOW_B, width=0.45, opacity=stroke_opacity)

        def update_patch_counter(counter, alpha):
            set_patch_counter_number(round(alpha * 576))
            counter.arrange(RIGHT, buff=0.16)
            counter.next_to(patch_stack, DOWN, buff=0.46)

        def add_cls_counter(counter, alpha):
            set_patch_counter_number(576 + int(round(alpha)))
            counter.arrange(RIGHT, buff=0.16)
            counter.next_to(patch_stack, DOWN, buff=0.46)

        self.play(
            FadeIn(patch_dims),
            ShowCreation(channel_depth_mark[0]),
            FadeIn(channel_depth_mark[1], shift=RIGHT * 0.06),
            run_time=0.75,
        )
        self.wait(0.45)
        self.play(
            FadeOut(VGroup(patch_dims, channel_depth_mark)),
            FadeOut(resize_grid),
            run_time=0.45,
        )
        self.play(
            FadeIn(patch_counter, shift=UP * 0.06),
            run_time=0.35,
        )
        self.play(
            UpdateFromAlphaFunc(patch_sweep_grid, light_patch_cells),
            UpdateFromAlphaFunc(patch_counter, update_patch_counter),
            run_time=2.25,
        )
        cls_token.next_to(patch_counter, DOWN, buff=0.18)
        self.play(
            FadeIn(cls_token, shift=UP * 0.05),
            UpdateFromAlphaFunc(patch_counter, add_cls_counter),
            run_time=0.7,
        )
        self.wait(0.55)
        self.play(
            FadeOut(VGroup(patch_counter, cls_token, patch_sweep_grid)),
            ShowCreation(patch_pixel_grid, lag_ratio=0.01),
            FadeOut(patch_stack[0], shift=LEFT * 0.05),
            FadeOut(patch_stack[1][0]),
            run_time=0.9,
        )
        self.play(
            FadeOut(patch_stack[1][1], shift=LEFT * 0.12),
            ReplacementTransform(patch_pixel_grid, patch_value_vector[0]),
            FadeIn(patch_value_vector[1], shift=UP * 0.05),
            run_time=1.45,
        )
        self.wait(1.4)
