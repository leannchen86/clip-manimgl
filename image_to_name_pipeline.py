from manimlib import *
import numpy as np
from pathlib import Path
from PIL import Image


class PipelineAnimation(Scene):
    def create_pipeline_block(
        self,
        title,
        lines,
        width=1.82,
        height=1.35,
        highlight=False,
        color=GREY_B,
    ):
        box = RoundedRectangle(width=width, height=height, corner_radius=0.08)
        box.set_fill("#101218", opacity=0.92)
        box.set_stroke(BLUE_B if highlight else color, width=2.2 if highlight else 1.2)

        title_mob = Text(title, font_size=22, color=WHITE if highlight else GREY_A)
        detail = VGroup()
        for line in lines:
            detail.add(Text(line, font_size=13, color=GREY_C))
        detail.arrange(DOWN, buff=0.055)

        content = VGroup(title_mob, detail)
        content.arrange(DOWN, buff=0.16)
        content.move_to(box.get_center())

        glow = RoundedRectangle(width=width + 0.12, height=height + 0.12, corner_radius=0.1)
        glow.set_fill(BLUE_C, opacity=0.05 if highlight else 0.0)
        glow.set_stroke(BLUE_C, width=5, opacity=0.22 if highlight else 0.0)
        glow.move_to(box)

        return VGroup(glow, box, content)

    def create_arrow(self, left_mob, right_mob, color=GREY_B, stroke_width=2.0):
        start = left_mob.get_right() + RIGHT * 0.08
        end = right_mob.get_left() - RIGHT * 0.08
        direction = end - start
        length = np.linalg.norm(direction)
        unit = direction / length if length > 0 else RIGHT
        tip_size = 0.13

        line = Line(start, end - unit * tip_size * 0.85)
        line.set_stroke(color, width=stroke_width, opacity=0.92)

        tip = Triangle()
        tip.set_fill(color, opacity=0.92)
        tip.set_stroke(width=0)
        tip.scale(tip_size)
        angle = np.arctan2(unit[1], unit[0])
        tip.rotate(angle - PI / 2)
        tip.move_to(end - unit * tip_size * 0.32)
        return VGroup(line, tip)

    def create_vector(self, n_cells=34, cell_width=0.12, cell_height=0.36, colors=None):
        if colors is None:
            colors = [BLUE_E, BLUE_D, TEAL_D, GREEN_D, GREY_B]

        cells = VGroup()
        for i in range(n_cells):
            rect = Rectangle(width=cell_width, height=cell_height)
            rect.set_fill(colors[i % len(colors)], opacity=0.78)
            rect.set_stroke(WHITE, width=0.35, opacity=0.45)
            cells.add(rect)
        cells.arrange(RIGHT, buff=0.025)
        bracket = SurroundingRectangle(cells, buff=0.06)
        bracket.set_stroke(GREY_B, width=1.2)
        return VGroup(bracket, cells)

    def create_probability_bars(self, values, labels=None, width=5.2, height=2.4, color=TEAL_C):
        max_value = max(values) if values else 1.0
        total = sum(values) if values else 1.0
        bar_width = width / len(values) * 0.58
        spacing = width / len(values)

        bars = VGroup()
        for i, value in enumerate(values):
            h = height * value / max_value
            bar = Rectangle(width=bar_width, height=h)
            bar.set_fill(color, opacity=0.86)
            bar.set_stroke(WHITE, width=0.6, opacity=0.45)
            x = (i - (len(values) - 1) / 2) * spacing
            bar.move_to(np.array([x, h / 2, 0.0]))

            if labels:
                lab = Text(labels[i], font_size=16, color=GREY_C)
                lab.next_to(bar, DOWN, buff=0.12)
                bars.add(VGroup(bar, lab))
            else:
                bars.add(bar)

        baseline = Line(
            np.array([-width / 2 - 0.12, 0, 0]),
            np.array([width / 2 + 0.12, 0, 0]),
            stroke_color=GREY_C,
            stroke_width=1.1,
        )

        mass_strip = VGroup()
        cursor = -width / 2
        strip_y = -0.48
        strip_colors = [TEAL_C, BLUE_C, BLUE_D, GREY_B]
        for i, value in enumerate(values):
            segment_width = width * value / total
            segment = Rectangle(width=segment_width, height=0.13)
            segment.set_fill(strip_colors[i % len(strip_colors)], opacity=0.86)
            segment.set_stroke(BLACK, width=0.35, opacity=0.75)
            segment.move_to(np.array([cursor + segment_width / 2, strip_y, 0.0]))
            mass_strip.add(segment)
            cursor += segment_width
        strip_frame = SurroundingRectangle(mass_strip, buff=0.02)
        strip_frame.set_stroke(WHITE, width=0.75, opacity=0.5)
        strip_label = Text("probability mass", font_size=14, color=GREY_B)
        strip_label.next_to(strip_frame, DOWN, buff=0.08)

        return VGroup(baseline, bars, VGroup(mass_strip, strip_frame, strip_label))

    def create_top10_list(self):
        names = [
            ("David", 0.31),
            ("Daniel", 0.19),
            ("Michael", 0.12),
            ("Kevin", 0.09),
            ("Jason", 0.07),
            ("Alex", 0.05),
            ("Chris", 0.04),
            ("Ryan", 0.03),
            ("Brian", 0.02),
            ("Eric", 0.02),
        ]
        rows = VGroup()
        max_bar_width = 4.35
        for rank, (name, prob) in enumerate(names, start=1):
            color = BLUE_B if rank == 1 else GREY_B
            rank_text = Text(f"{rank}.", font_size=22, color=color)
            name_text = Text(name, font_size=22, color=WHITE if rank == 1 else GREY_A)
            value_text = Text(f"{prob:.2f}", font_size=20, color=WHITE if rank == 1 else GREY_B)

            bar = Rectangle(width=max_bar_width * prob / names[0][1], height=0.2)
            bar.set_fill(color, opacity=0.92 if rank == 1 else 0.62)
            bar.set_stroke(WHITE, width=0.4, opacity=0.35)

            rank_text.move_to(LEFT * 3.05)
            name_text.move_to(LEFT * 2.38)
            bar.move_to(LEFT * 0.15 + RIGHT * bar.get_width() / 2)
            value_text.move_to(RIGHT * 2.95)
            row = VGroup(rank_text, name_text, bar, value_text)
            rows.add(row)

        rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        highlight = SurroundingRectangle(rows[0], buff=0.09)
        highlight.set_fill(BLUE_C, opacity=0.06)
        highlight.set_stroke(BLUE_B, width=1.5, opacity=0.85)
        return VGroup(highlight, rows)

    def create_logit_bars(self, logits, width=5.7, height=2.8):
        max_abs = max(abs(v) for v in logits)
        bar_width = width / len(logits) * 0.56
        spacing = width / len(logits)
        baseline = Line(
            np.array([-width / 2 - 0.2, 0, 0]),
            np.array([width / 2 + 0.2, 0, 0]),
            stroke_color=GREY_C,
            stroke_width=1.2,
        )
        bars = VGroup()
        for i, value in enumerate(logits):
            h = abs(value) / max_abs * height / 2
            bar = Rectangle(width=bar_width, height=h)
            color = BLUE_C if value >= 0 else GREY_D
            bar.set_fill(color, opacity=0.82 if value >= 0 else 0.62)
            bar.set_stroke(WHITE, width=0.55, opacity=0.42)
            x = (i - (len(logits) - 1) / 2) * spacing
            y = h / 2 if value >= 0 else -h / 2
            bar.move_to(np.array([x, y, 0.0]))
            bars.add(bar)
        return VGroup(baseline, bars)

    def create_mlp_block(self):
        layers = VGroup()
        widths = [0.56, 0.86, 0.86, 0.56]
        heights = [2.1, 2.55, 2.25, 1.72]
        labels = ["Input\n768", "Hidden", "Hidden", "Output\n138"]
        for w, h, label in zip(widths, heights, labels):
            rect = RoundedRectangle(width=w, height=h, corner_radius=0.05)
            rect.set_fill("#12141b", opacity=0.95)
            rect.set_stroke(BLUE_D, width=1.2, opacity=0.75)
            text = Text(label, font_size=15, color=GREY_A)
            text.move_to(rect)
            layers.add(VGroup(rect, text))
        layers.arrange(RIGHT, buff=0.28)
        title = Text("MLP classification head", font_size=25, color=WHITE)
        subtitle = Text("trained on our name labels", font_size=18, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.1)
        label_group = VGroup(title, subtitle)
        label_group.next_to(layers, UP, buff=0.32)
        return VGroup(layers, label_group)

    def save_crop_from_box(self, filename, face_box, output_path, output_size=336):
        image = Image.open(filename).convert("RGBA")
        img_w, img_h = image.size
        cx = face_box["x"] + face_box["width"] / 2
        cy = face_box["y"] + face_box["height"] / 2
        side = max(face_box["width"], face_box["height"]) * 1.05
        left = max(0, int(cx - side / 2))
        top = max(0, int(cy - side / 2))
        right = min(img_w, int(cx + side / 2))
        bottom = min(img_h, int(cy + side / 2))
        crop = image.crop((left, top, right, bottom)).resize((output_size, output_size))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)
        return str(output_path), (img_w, img_h)

    def box_from_pixels(self, image_mob, face_box, image_size):
        img_w, img_h = image_size
        mob_w = image_mob.get_width()
        mob_h = image_mob.get_height()
        left = image_mob.get_left()[0]
        top = image_mob.get_top()[1]

        cx = left + (face_box["x"] + face_box["width"] / 2) / img_w * mob_w
        cy = top - (face_box["y"] + face_box["height"] / 2) / img_h * mob_h
        w = face_box["width"] / img_w * mob_w
        h = face_box["height"] / img_h * mob_h

        rect = Rectangle(width=w, height=h)
        rect.set_stroke(BLUE_B, width=3.0)
        rect.set_fill(BLUE_B, opacity=0.045)
        rect.move_to(np.array([cx, cy, 0.0]))
        return rect

    def create_dim_masks(self, image_mob, focus_rect):
        left = image_mob.get_left()[0]
        right = image_mob.get_right()[0]
        top = image_mob.get_top()[1]
        bottom = image_mob.get_bottom()[1]
        fl = focus_rect.get_left()[0]
        fr = focus_rect.get_right()[0]
        ft = focus_rect.get_top()[1]
        fb = focus_rect.get_bottom()[1]

        def mask(width, height, center):
            rect = Rectangle(width=max(width, 0.001), height=max(height, 0.001))
            rect.set_fill(BLACK, opacity=0.62)
            rect.set_stroke(width=0)
            rect.move_to(center)
            return rect

        return VGroup(
            mask(fr - fl, top - ft, np.array([(fl + fr) / 2, (top + ft) / 2, 0])),
            mask(fr - fl, fb - bottom, np.array([(fl + fr) / 2, (fb + bottom) / 2, 0])),
            mask(fl - left, top - bottom, np.array([(left + fl) / 2, (top + bottom) / 2, 0])),
            mask(right - fr, top - bottom, np.array([(fr + right) / 2, (top + bottom) / 2, 0])),
        )

    def create_patch_grid(self, image_mob, rows=6, cols=6, color=WHITE):
        lines = VGroup()
        left = image_mob.get_left()[0]
        right = image_mob.get_right()[0]
        top = image_mob.get_top()[1]
        bottom = image_mob.get_bottom()[1]
        for k in range(1, cols):
            x = left + (right - left) * k / cols
            lines.add(Line([x, top, 0], [x, bottom, 0], stroke_color=color, stroke_width=1.2))
        for k in range(1, rows):
            y = bottom + (top - bottom) * k / rows
            lines.add(Line([left, y, 0], [right, y, 0], stroke_color=color, stroke_width=1.2))
        return lines

    def overview_group(self, clip_highlight=True):
        block_specs = [
            ("PHOTO", ["any resolution", "JPEG / PNG"]),
            ("RetinaFace", ["detect + crop", "square crop", "face region"]),
            ("CLIP encoder", ["ViT-L/14", "@336px", "frozen", "336x336 RGB"]),
            ("MLP", ["trained head", "768-dim vector"]),
            ("Softmax", ["138 logits"]),
            ("Top-10 names", ["138 probabilities", "sum = 1.0"]),
        ]
        blocks = VGroup()
        for title, lines in block_specs:
            blocks.add(self.create_pipeline_block(title, lines, highlight=clip_highlight and title == "CLIP encoder"))
        blocks.arrange(RIGHT, buff=0.5)
        blocks.move_to(DOWN * 0.15)

        arrows = VGroup()
        for left, right in zip(blocks[:-1], blocks[1:]):
            arrows.add(self.create_arrow(left, right))

        title = Text("Image-to-name prediction pipeline", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.5)
        return VGroup(title, blocks, arrows)

    def clear_scene(self, run_time=0.55):
        if self.mobjects:
            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=run_time)

    def construct(self):
        self.camera.background_color = BLACK

        base_dir = Path(__file__).resolve().parent
        image_file = str(base_dir / "croped1.png")

        # Edit this manually to tune the simulated RetinaFace detection.
        # Coordinates are pixels in croped1.png, measured from top-left.
        face_box = {
            "x": 215,
            "y": 95,
            "width": 650,
            "height": 855,
        }

        crop_file, image_size = self.save_crop_from_box(
            image_file,
            face_box,
            base_dir / ".generated" / "retinaface_face_crop_336.png",
        )
        logits = [-1.2, 0.4, 2.2, -0.7, 1.5, 0.1, -1.6, 0.9, 3.1, -0.3]
        probabilities = [0.26, 0.18, 0.13, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

        # FRAME 0 - Establishing overview shot
        overview = self.overview_group(clip_highlight=True)
        self.play(FadeIn(overview[0], shift=UP * 0.15), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(block, shift=UP * 0.16) for block in overview[1]], lag_ratio=0.08),
            run_time=1.15,
        )
        self.play(ShowCreation(overview[2], lag_ratio=0.12), run_time=0.9)
        clip_pulse = SurroundingRectangle(overview[1][2], buff=0.08)
        clip_pulse.set_stroke(BLUE_C, width=3.0, opacity=0.85)
        self.play(ShowCreation(clip_pulse), run_time=0.45)
        self.play(FadeOut(clip_pulse), run_time=0.35)
        self.wait(0.5)

        # FRAME 1 - RetinaFace crop
        self.clear_scene()
        photo = ImageMobject(image_file)
        photo.set_height(4.85)
        photo.move_to(LEFT * 1.95 + UP * 0.15)
        box = self.box_from_pixels(photo, face_box, image_size)
        detection_label = Text("RetinaFace detection", font_size=25, color=BLUE_B)
        detection_label.next_to(box, UP, buff=0.15)
        any_res = Text("The original photo can have any resolution.", font_size=22, color=GREY_B)
        any_res.to_edge(DOWN, buff=0.45)
        crop_img = ImageMobject(crop_file)
        crop_img.set_height(3.25)
        crop_img.move_to(RIGHT * 3.0 + UP * 0.15)
        crop_frame = Square(side_length=crop_img.get_height() + 0.06)
        crop_frame.set_stroke(WHITE, width=1.5, opacity=0.75)
        crop_frame.move_to(crop_img)
        crop_label = Text("square face crop", font_size=25, color=WHITE)
        crop_label.next_to(crop_frame, DOWN, buff=0.2)

        self.play(FadeIn(photo, shift=UP * 0.12), FadeIn(any_res), run_time=0.8)
        self.play(ShowCreation(box), FadeIn(detection_label, shift=UP * 0.08), run_time=0.75)
        masks = self.create_dim_masks(photo, box)
        self.play(FadeIn(masks), run_time=0.55)
        self.play(
            TransformFromCopy(box, crop_frame),
            FadeIn(crop_img, shift=RIGHT * 0.25),
            FadeIn(crop_label, shift=UP * 0.1),
            run_time=1.05,
        )
        self.wait(0.55)

        # FRAME 2 - CLIP encoder input
        self.clear_scene()
        face_crop = ImageMobject(crop_file)
        face_crop.set_height(3.0)
        face_crop.move_to(LEFT * 3.45 + UP * 0.15)
        face_frame = Square(side_length=face_crop.get_height() + 0.06)
        face_frame.set_stroke(WHITE, width=1.3, opacity=0.7)
        face_frame.move_to(face_crop)
        input_label = Text("Input: 336 x 336 RGB face crop", font_size=28, color=WHITE)
        input_label.to_edge(UP, buff=0.55)
        encoder = self.create_pipeline_block(
            "CLIP encoder",
            ["ViT-L/14", "@336px", "frozen"],
            width=2.35,
            height=1.75,
            highlight=True,
        )
        encoder.move_to(RIGHT * 2.85 + UP * 0.15)
        arrow_to_encoder = self.create_arrow(face_frame, encoder, color=BLUE_B, stroke_width=2.5)
        patch_grid = self.create_patch_grid(face_crop, rows=6, cols=6, color=WHITE)
        patch_label = Text("Image patches -> visual tokens", font_size=24, color=GREY_A)
        patch_label.next_to(face_frame, DOWN, buff=0.3)
        frozen_note = Text("The encoder is frozen during training and inference.", font_size=21, color=GREY_B)
        frozen_note.to_edge(DOWN, buff=0.48)

        self.play(FadeIn(input_label), FadeIn(face_crop), ShowCreation(face_frame), run_time=0.75)
        self.play(ShowCreation(patch_grid, lag_ratio=0.03), FadeIn(patch_label), run_time=0.9)
        self.play(FadeIn(encoder, shift=LEFT * 0.2), ShowCreation(arrow_to_encoder), FadeIn(frozen_note), run_time=0.8)
        self.wait(0.45)

        # FRAME 3 - CLIP encoder output embedding
        tokens = VGroup()
        for i in range(18):
            dot = Square(side_length=0.12)
            dot.set_fill(BLUE_C if i % 3 else TEAL_C, opacity=0.85)
            dot.set_stroke(width=0)
            dot.move_to(face_crop.get_center() + np.array([
                ((i % 6) - 2.5) * 0.38,
                (1.0 - (i // 6)) * 0.38,
                0.0,
            ]))
            tokens.add(dot)
        token_target = tokens.copy()
        for i, dot in enumerate(token_target):
            dot.move_to(encoder.get_center() + np.array([
                ((i % 6) - 2.5) * 0.13,
                (1.0 - (i // 6)) * 0.13,
                0.0,
            ]))
            dot.scale(0.6)

        vector = self.create_vector(n_cells=38)
        vector.move_to(RIGHT * 3.45 + DOWN * 1.75)
        vector_label = VGroup(
            Text("Face embedding", font_size=25, color=WHITE),
            Text("768-dimensional vector", font_size=21, color=GREY_B),
        )
        vector_label.arrange(DOWN, buff=0.1)
        vector_label.next_to(vector, DOWN, buff=0.2)
        caption = Text("CLIP turns the face into a reusable visual embedding.", font_size=25, color=GREY_A)
        caption.to_edge(DOWN, buff=0.43)
        idea = Text("The encoder does not output a name directly.", font_size=23, color=GREY_B)
        idea.next_to(input_label, DOWN, buff=0.2)

        self.play(FadeIn(tokens), run_time=0.25)
        self.play(Transform(tokens, token_target), run_time=0.95)
        self.play(FadeOut(tokens), run_time=0.25)
        self.play(FadeOut(patch_label), FadeOut(frozen_note), FadeIn(idea), run_time=0.35)
        self.play(FadeIn(vector, shift=RIGHT * 0.25), FadeIn(vector_label), FadeIn(caption), run_time=0.85)
        self.wait(0.7)

        # FRAME 4 - MLP trained head
        self.clear_scene()
        vector2 = self.create_vector(n_cells=38)
        vector2.scale(0.92)
        vector2.move_to(LEFT * 4.25 + UP * 0.15)
        emb_label = Text("768-dim CLIP embedding", font_size=22, color=GREY_B)
        emb_label.next_to(vector2, DOWN, buff=0.22)
        mlp = self.create_mlp_block()
        mlp.move_to(ORIGIN + UP * 0.05)
        mlp_arrow = self.create_arrow(vector2, mlp[0], color=BLUE_B, stroke_width=2.4)
        small_logits = self.create_logit_bars(logits, width=2.65, height=1.55)
        small_logits.move_to(RIGHT * 4.25 + DOWN * 0.1)
        logits_label = Text("logits", font_size=24, color=WHITE)
        logits_label.next_to(small_logits, DOWN, buff=0.22)
        caption = Text("The MLP maps the CLIP embedding to name scores.", font_size=25, color=GREY_A)
        caption.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(vector2), FadeIn(emb_label), run_time=0.55)
        self.play(FadeIn(mlp, shift=UP * 0.14), ShowCreation(mlp_arrow), run_time=0.75)
        self.play(FadeIn(small_logits, shift=RIGHT * 0.18), FadeIn(logits_label), FadeIn(caption), run_time=0.9)
        self.wait(0.55)

        # FRAME 5 - Logits
        self.clear_scene()
        logit_chart = self.create_logit_bars(logits, width=6.6, height=3.35)
        logit_chart.move_to(UP * 0.05)
        logit_title = Text("138 logits", font_size=34, color=WHITE)
        logit_title.to_edge(UP, buff=0.55)
        raw_note = Text("raw, unnormalized scores", font_size=22, color=GREY_B)
        raw_note.next_to(logit_title, DOWN, buff=0.18)
        raw_caption = Text("Logits are raw scores, not probabilities yet.", font_size=25, color=GREY_A)
        raw_caption.to_edge(DOWN, buff=0.45)
        pos_label = Text("+", font_size=24, color=BLUE_B)
        pos_label.next_to(logit_chart, LEFT, buff=0.25).shift(UP * 0.95)
        neg_label = Text("-", font_size=24, color=GREY_C)
        neg_label.next_to(logit_chart, LEFT, buff=0.25).shift(DOWN * 0.95)
        self.play(FadeIn(logit_title), FadeIn(raw_note), run_time=0.45)
        self.play(ShowCreation(logit_chart[0]), LaggedStart(*[FadeIn(b, shift=UP * 0.05) for b in logit_chart[1]], lag_ratio=0.04), run_time=1.05)
        self.play(FadeIn(pos_label), FadeIn(neg_label), FadeIn(raw_caption), run_time=0.45)
        self.wait(0.55)

        # FRAME 6 - Softmax
        self.clear_scene()
        logits_left = self.create_logit_bars(logits, width=3.1, height=2.3)
        logits_left.move_to(LEFT * 4.15 + UP * 0.1)
        softmax = self.create_pipeline_block(
            "Softmax",
            ["converts scores", "into probabilities"],
            width=2.35,
            height=1.5,
            highlight=True,
        )
        softmax.move_to(ORIGIN + UP * 0.1)
        prob_bars = self.create_probability_bars(probabilities, width=3.6, height=2.35, color=TEAL_C)
        prob_bars.move_to(RIGHT * 4.1 + DOWN * 0.95)
        sum_label = Text("sum = 1.0", font_size=25, color=WHITE)
        sum_label.next_to(prob_bars, UP, buff=0.24)
        soft_caption = Text("Softmax turns raw scores into a probability distribution.", font_size=25, color=GREY_A)
        soft_caption.to_edge(DOWN, buff=0.45)
        arrow1 = self.create_arrow(logits_left, softmax, color=BLUE_B, stroke_width=2.3)
        arrow2 = self.create_arrow(softmax, prob_bars, color=TEAL_C, stroke_width=2.3)

        self.play(FadeIn(logits_left), run_time=0.45)
        self.play(FadeIn(softmax, shift=UP * 0.12), ShowCreation(arrow1), run_time=0.65)
        self.play(ShowCreation(arrow2), FadeIn(prob_bars, shift=RIGHT * 0.18), FadeIn(sum_label), FadeIn(soft_caption), run_time=1.0)
        self.wait(0.6)

        # FRAME 7 - Top-10 names
        self.clear_scene()
        top_title = Text("Top-10 predicted names", font_size=34, color=WHITE)
        top_title.to_edge(UP, buff=0.5)
        top_list = self.create_top10_list()
        top_list.move_to(UP * 0.05)
        final_note = Text("The final output is a ranked probability list, not a generated sentence.", font_size=23, color=GREY_A)
        final_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(top_title), run_time=0.35)
        self.play(FadeIn(top_list[0]), LaggedStart(*[FadeIn(row, shift=RIGHT * 0.12) for row in top_list[1]], lag_ratio=0.045), run_time=1.2)
        self.play(FadeIn(final_note), run_time=0.4)
        self.wait(0.7)

        # FRAME 8 - Return to overview
        self.clear_scene()
        overview2 = self.overview_group(clip_highlight=False)
        final_caption = Text(
            "Frozen CLIP encoder + trained MLP head = consistent name ranking.",
            font_size=27,
            color=WHITE,
        )
        final_caption.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(overview2[0], shift=UP * 0.12), FadeIn(overview2[1]), FadeIn(overview2[2]), run_time=0.85)

        glow_arrows = VGroup()
        for left, right in zip(overview2[1][:-1], overview2[1][1:]):
            glow_arrows.add(self.create_arrow(left, right, color=BLUE_C, stroke_width=4.0))
        self.play(ShowCreation(glow_arrows, lag_ratio=0.16), run_time=1.45)

        highlight_blocks = VGroup()
        for i in [2, 3, 4, 5]:
            h = SurroundingRectangle(overview2[1][i], buff=0.07)
            h.set_stroke(BLUE_C if i == 2 else TEAL_C, width=2.5, opacity=0.9)
            h.set_fill(BLUE_C, opacity=0.035)
            highlight_blocks.add(h)
        self.play(LaggedStart(*[ShowCreation(h) for h in highlight_blocks], lag_ratio=0.12), FadeIn(final_caption), run_time=1.0)
        self.wait(1.0)
