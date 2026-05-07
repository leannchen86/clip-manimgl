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

    def make_rgb_channel_files(self, source="croped1.png", size=336):
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

    def make_patch_channel_files(self, source="mid1.png", size=336, patch_size=14, row=12, col=12):
        output_dir = Path(__file__).with_name(".generated") / "vit_patch"
        output_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(__file__).with_name(source)
        image = Image.open(source_path).convert("RGB")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        if source_path.name == "mid1.png":
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
        crop_image = ImageMobject("croped1.png")
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

    def token_color(self, index):
        palette = [
            "#2F80ED",
            "#27AE60",
            "#F2C94C",
            "#EB5757",
            "#56CCF2",
            "#BB6BD9",
            "#F2994A",
            "#6FCF97",
        ]
        return palette[index % len(palette)]

    def make_token_cell(self, index=0, side=0.24, fill_opacity=1.0):
        cell = Square(side_length=side)
        cell.set_fill(self.token_color(index), opacity=fill_opacity)
        cell.set_stroke(WHITE, width=0.75, opacity=0.82)
        return cell

    def make_transformer_token_row(self, left_count=18, right_count=10, side=0.24):
        row = VGroup()
        cells = VGroup()
        ellipsis = Tex(r"\cdots", font_size=28, color=GREY_A)

        token_index = 0
        for _ in range(left_count):
            cell = self.make_token_cell(token_index, side=side)
            row.add(cell)
            cells.add(cell)
            token_index += 1

        row.add(ellipsis)

        for _ in range(right_count):
            cell = self.make_token_cell(token_index, side=side)
            row.add(cell)
            cells.add(cell)
            token_index += 1

        row.arrange(RIGHT, buff=0.055)
        return row, cells, ellipsis

    def make_token_row_stack(self, token_row, layers=3):
        stack = VGroup()
        stack_palette = [
            "#00D1FF",
            "#FF6B35",
            "#9B5DE5",
            "#00F5A0",
            "#FF477E",
            "#FEE440",
            "#5A86FF",
            "#B8F35A",
        ]
        for depth in range(1, layers + 1):
            row_copy = token_row.copy()
            row_copy.shift(UP * 0.34 * depth)
            row_copy.set_opacity(0.48 - 0.11 * depth)
            row_copy.set_z_index(1)
            visible_index = 0
            for mob in row_copy:
                if isinstance(mob, Square):
                    color_index = (visible_index * 5 + depth * 3) % len(stack_palette)
                    mob.set_fill(stack_palette[color_index], opacity=0.48 - 0.11 * depth)
                    mob.set_stroke(WHITE, width=0.7, opacity=0.48 - 0.11 * depth)
                    visible_index += 1
            stack.add(row_copy)
        return stack

    def make_attention_connections(self, token_cells, count=38, seed=7):
        rng = np.random.default_rng(seed)
        connections = VGroup()
        colors = [BLUE_B, TEAL_B, YELLOW_B, GREEN_B]
        total = len(token_cells)
        pairs = set()

        while len(pairs) < count:
            left = int(rng.integers(0, total - 3))
            max_span = min(15, total - left - 1)
            span = int(rng.integers(3, max_span + 1))
            right = left + span
            pairs.add((left, right))

        for index, (left, right) in enumerate(sorted(pairs)):
            start = token_cells[left].get_top() + UP * 0.025
            end = token_cells[right].get_top() + UP * 0.025
            span = abs(right - left)
            arc_height = 0.22 + 0.055 * span + 0.09 * (index % 4)
            jitter = RIGHT * float(rng.uniform(-0.08, 0.08))
            mid = (start + end) / 2 + UP * arc_height + jitter

            curve = VMobject()
            curve.set_points_smoothly([start, mid, end])
            curve.set_stroke(colors[index % len(colors)], width=1.2, opacity=0.42)
            connections.add(curve)

        return connections

    def make_layernorm_stats(self, token_cells):
        mean_values = [
            "-0.18", "0.07", "-0.04", "0.13", "-0.09", "0.02", "0.16",
            "-0.11", "0.05", "-0.14", "0.09", "-0.02", "0.12", "-0.06",
            "0.01", "-0.10", "0.15", "-0.03", "0.08", "-0.12", "0.04",
            "0.11", "-0.07", "0.03", "-0.15", "0.06", "-0.01", "0.10",
        ]
        std_values = [
            "1.18", "0.91", "1.04", "0.87", "1.12", "0.96", "1.23",
            "0.89", "1.07", "0.94", "1.15", "0.98", "1.20", "0.92",
            "1.01", "0.88", "1.17", "0.95", "1.09", "0.90", "1.13",
            "0.97", "1.06", "0.93", "1.21", "0.99", "1.10", "0.86",
        ]

        mean_numbers = VGroup()
        std_numbers = VGroup()
        for index, cell in enumerate(token_cells):
            mean = Tex(mean_values[index], font_size=9, color=GREY_A)
            mean.next_to(cell, UP, buff=0.075)
            mean_numbers.add(mean)

            std = Tex(std_values[index], font_size=9, color=GREY_A)
            std.next_to(cell, DOWN, buff=0.075)
            std_numbers.add(std)

        mean_label = TexText("mean", font_size=14, color=GREY_B)
        mean_label.next_to(mean_numbers[0], LEFT, buff=0.16)
        std_label = TexText("std", font_size=14, color=GREY_B)
        std_label.next_to(std_numbers[0], LEFT, buff=0.16)

        return VGroup(mean_label, std_label, mean_numbers, std_numbers)

    def make_mlp_vector(self, output=False):
        prefix = r"\tilde{n}" if output else "n"
        vector = Tex(
            r"\left[\begin{array}{c}"
            + rf"{prefix}_1\\ {prefix}_2\\ \vdots\\ {prefix}_{{1024}}"
            + r"\end{array}\right]",
            font_size=28,
            color=WHITE if not output else YELLOW_B,
        )
        return vector

    def make_feed_forward_mlp(self):
        layer_sizes = [4, 6, 4]
        layer_colors = [GREEN_B, BLUE_B, YELLOW_B]

        node_layers = VGroup()
        for layer_index, size in enumerate(layer_sizes):
            nodes = VGroup()
            for _ in range(size):
                node = Circle(radius=0.07)
                node.set_fill(layer_colors[layer_index], opacity=0.12)
                node.set_stroke(layer_colors[layer_index], width=1.25, opacity=0.88)
                nodes.add(node)
            nodes.arrange(DOWN, buff=0.12)
            node_layers.add(nodes)
        node_layers.arrange(RIGHT, buff=0.46)

        connections = VGroup()
        for left_layer, right_layer in zip(node_layers[:-1], node_layers[1:]):
            for left_node in left_layer:
                for right_node in right_layer:
                    line = Line(left_node.get_center(), right_node.get_center())
                    line.set_stroke(GREY_B, width=0.55, opacity=0.28)
                    connections.add(line)

        label = TexText("MLP", font_size=18, color=WHITE)
        label.next_to(node_layers, UP, buff=0.13)

        return VGroup(connections, node_layers, label)

    def make_transformer_block(self, width=3.35, height=0.36, color=BLUE_B, opacity=0.92):
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.06,
            stroke_color=color,
            stroke_width=1.1,
            fill_color="#111820",
            fill_opacity=opacity,
        )
        label = TexText("Transformer", font_size=16, color=WHITE)
        label.move_to(box)
        return VGroup(box, label)

    def make_transformer_depth_stack(self):
        blocks = VGroup()
        colors = [BLUE_B, TEAL_B, GREEN_B, YELLOW_B]
        for index in range(4):
            block = self.make_transformer_block(
                color=colors[index % len(colors)],
                opacity=0.88,
            )
            blocks.add(block)
        lower_blocks = blocks.copy()

        upper_blocks = VGroup()
        for index in range(4):
            block = self.make_transformer_block(
                color=colors[(index + 1) % len(colors)],
                opacity=0.88,
            )
            upper_blocks.add(block)

        lower_blocks.arrange(UP, buff=0)
        upper_blocks.arrange(UP, buff=0)
        dots = Tex(r"\vdots", font_size=34, color=GREY_A)

        stack = VGroup(lower_blocks, dots, upper_blocks)
        stack.arrange(UP, buff=0.11)

        multiplier = Tex(r"\times 24", font_size=38, color=YELLOW_B)
        multiplier.next_to(stack, RIGHT, buff=0.5)

        return VGroup(stack, multiplier)

    def make_position_token_labels(self, token_cells):
        position_labels = VGroup()
        token_type_labels = VGroup()
        total_visible = len(token_cells)
        for index, cell in enumerate(token_cells):
            if index < total_visible - 4:
                position = index
            else:
                position = 576 - (total_visible - 1 - index)

            top_label = Tex(str(position), font_size=13, color=GREY_A)
            top_label.next_to(cell, UP, buff=0.09)
            position_labels.add(top_label)

            if position == 0:
                bottom_label = Tex(r"\mathrm{[CLS]}", font_size=16, color=YELLOW_B)
            else:
                bottom_label = Tex(str(position), font_size=13, color=GREY_A)
            bottom_label.next_to(cell, DOWN, buff=0.09)
            token_type_labels.add(bottom_label)

        pos_label = TexText("pos", font_size=17, color=GREY_B)
        pos_label.next_to(position_labels[0], LEFT, buff=0.22)
        patch_label = TexText("patch", font_size=17, color=GREY_B)
        patch_label.next_to(token_type_labels[0], LEFT, buff=0.22)

        return VGroup(pos_label, patch_label, position_labels, token_type_labels)

    def make_cls_embedding_vector(self):
        return Tex(
            r"\left[n_1,\ n_2,\ \ldots,\ n_{768}\right]",
            font_size=42,
            color=YELLOW_B,
        )

    def make_embedding_space(self):
        axes = ThreeDAxes(
            x_range=(-1.2, 1.2, 10),
            y_range=(-1.2, 1.2, 10),
            z_range=(-1.2, 1.2, 10),
            width=1.9,
            height=1.9,
            depth=1.9,
        )
        axes.x_axis.set_stroke(BLUE_B, width=1.2, opacity=0.72)
        axes.y_axis.set_stroke(TEAL_B, width=1.2, opacity=0.72)
        axes.z_axis.set_stroke(YELLOW_B, width=1.2, opacity=0.72)

        sphere = Sphere(radius=1)
        sphere.set_color(BLUE_D)
        sphere.set_opacity(0.1)

        x_label = Tex("x", font_size=18, color=BLUE_B)
        x_label.move_to(RIGHT * 1.15 + DOWN * 0.06)
        y_label = Tex("y", font_size=18, color=TEAL_B)
        y_label.move_to(UP * 1.15 + RIGHT * 0.06)
        z_label = Tex("z", font_size=18, color=YELLOW_B)
        z_label.move_to(LEFT * 0.20 + UP * 0.20)

        space = Group(axes, sphere, x_label, y_label, z_label)
        space.rotate(45 * DEGREES, axis=OUT, about_point=ORIGIN)
        space.rotate(-45 * DEGREES, axis=RIGHT, about_point=ORIGIN)

        return space

    def make_embedding_vector(self, end, color=YELLOW_B):
        vector = Line(ORIGIN, end, color=color, stroke_width=3.0)
        vector.set_opacity(1.0)
        endpoint = Sphere(radius=0.055)
        endpoint.set_color(color)
        endpoint.move_to(end)
        embedding_vector = Group(vector, endpoint)
        embedding_vector.rotate(45 * DEGREES, axis=OUT, about_point=ORIGIN)
        embedding_vector.rotate(-45 * DEGREES, axis=RIGHT, about_point=ORIGIN)
        return embedding_vector

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
        photo = ImageMobject("rf_example1.png")
        photo.set_height(height)
        frame = SurroundingRectangle(photo, buff=0.02)
        frame.set_stroke(WHITE, width=1.0, opacity=0.75)
        label = self.mono("input image", font_size=13, color=GREY_A)
        label.next_to(frame, DOWN, buff=0.08)
        return Group(photo, frame, label)

    def construct(self):
        self.camera.background_color = BLACK

        # PART 1 -- RetinaFace: detect & crop
        photo = ImageMobject("rf_example1.png")
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
            FadeOut(VGroup(retinaface_frame, retinaface_tag, bbox, crop_label, crop_detail, failure)),
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

        # PART 4 -- Transformer input tokens and self-attention
        token_seed = self.make_token_cell(index=3, side=0.36, fill_opacity=0.95)
        token_seed.move_to(patch_value_vector[0].get_center())

        token_row, token_cells, token_ellipsis = self.make_transformer_token_row()
        token_row.move_to(UP * 0.18)
        token_row_center = token_row.get_center()
        token_row.set_z_index(2)
        token_count = Tex(r"577\ \text{tokens}", font_size=30, color=YELLOW_B)
        token_count.next_to(token_row, DOWN, buff=0.26)

        self.play(
            FadeOut(crop_with_grid, shift=RIGHT * 0.18),
            FadeOut(patch_value_vector[1], shift=DOWN * 0.05),
            patch_value_vector[0].animate.scale(0.06).move_to(token_seed).set_opacity(0),
            FadeIn(token_seed, scale=0.82),
            run_time=0.85,
        )
        self.remove(patch_value_vector[0])
        self.wait(0.2)

        seed_landing_index = 3
        self.play(
            ReplacementTransform(token_seed, token_cells[seed_landing_index]),
            LaggedStart(*[
                TransformFromCopy(token_seed.copy(), cell)
                for index, cell in enumerate(token_cells)
                if index != seed_landing_index
            ], lag_ratio=0.025),
            FadeIn(token_ellipsis, shift=UP * 0.04),
            run_time=1.25,
        )
        self.play(
            FadeIn(token_count, shift=UP * 0.06),
            run_time=0.45,
        )
        self.wait(0.55)

        attention_caption = TexText(
            "multi-head self-attention",
            font_size=25,
            color=TEAL_B,
        )
        attention_caption.move_to(clip_frame.get_bottom() + UP * 0.34)

        back_token_rows = self.make_token_row_stack(token_row, layers=3)
        attention_connections = self.make_attention_connections(token_cells)
        attended_palette = [
            "#FF4D6D",
            "#FFD166",
            "#06D6A0",
            "#4CC9F0",
            "#B517FF",
            "#FF9F1C",
            "#7AE582",
            "#F15BB5",
        ]
        attended_colors = [
            attended_palette[index % len(attended_palette)]
            for index in range(len(token_cells))
        ]

        self.play(
            clip_frame.animate.set_stroke(TEAL_B, width=1.35, opacity=1.0),
            FadeIn(attention_caption, shift=UP * 0.05),
            run_time=0.55,
        )
        self.play(
            LaggedStart(*[
                FadeIn(row, shift=UP * 0.08)
                for row in back_token_rows
            ], lag_ratio=0.18),
            run_time=0.85,
        )
        self.play(
            LaggedStart(*[
                ShowCreation(curve)
                for curve in attention_connections
            ], lag_ratio=0.018),
            run_time=1.25,
        )
        token_color_anims = [
                cell.animate.set_fill(attended_colors[index], opacity=0.96).set_stroke(
                    WHITE,
                    width=1.15,
                    opacity=1.0,
                )
                for index, cell in enumerate(token_cells)
        ]
        stack_color_anims = []
        stack_attention_palette = [
            "#2EC4B6",
            "#FF3366",
            "#8338EC",
            "#FBFF12",
            "#3A86FF",
            "#FF9F1C",
            "#06FFA5",
            "#F72585",
        ]
        for row_index, row in enumerate(back_token_rows):
            row_opacity = 0.37 - 0.11 * row_index
            visible_index = 0
            for mob in row:
                if isinstance(mob, Square):
                    color_index = (visible_index * 7 + row_index * 4 + 2) % len(stack_attention_palette)
                    stack_color_anims.append(
                        mob.animate.set_fill(
                            stack_attention_palette[color_index],
                            opacity=row_opacity,
                        ).set_stroke(
                            WHITE,
                            width=0.8,
                            opacity=row_opacity,
                        )
                    )
                    visible_index += 1

        self.play(
            *token_color_anims,
            *stack_color_anims,
            run_time=0.72,
        )
        self.wait(1.0)

        # PART 5 -- LayerNorm stabilizes each token vector
        layernorm_caption = TexText(
            "LayerNorm",
            font_size=25,
            color=GREEN_B,
        )
        layernorm_caption.move_to(attention_caption)
        layernorm_stats = self.make_layernorm_stats(token_cells)
        layernorm_stats.set_z_index(4)
        token_count_layernorm = token_count.copy()
        token_count_layernorm.next_to(layernorm_stats, DOWN, buff=0.2)

        layernorm_colors = [
            interpolate_color(
                attended_colors[index],
                WHITE if index % 2 == 0 else BLACK,
                0.34 if index % 2 == 0 else 0.28,
            )
            for index in range(len(token_cells))
        ]

        self.play(
            Transform(attention_caption, layernorm_caption),
            clip_frame.animate.set_stroke(GREEN_B, width=1.35, opacity=1.0),
            FadeOut(back_token_rows, shift=UP * 0.08),
            FadeOut(attention_connections),
            Transform(token_count, token_count_layernorm),
            run_time=0.85,
        )
        self.play(
            FadeIn(layernorm_stats[0], shift=RIGHT * 0.05),
            FadeIn(layernorm_stats[1], shift=RIGHT * 0.05),
            LaggedStart(*[
                FadeIn(number, shift=DOWN * 0.025)
                for number in layernorm_stats[2]
            ], lag_ratio=0.012),
            LaggedStart(*[
                FadeIn(number, shift=UP * 0.025)
                for number in layernorm_stats[3]
            ], lag_ratio=0.012),
            run_time=1.0,
        )
        self.play(
            *[
                cell.animate.set_fill(layernorm_colors[index], opacity=1.0).set_stroke(
                    WHITE,
                    width=1.0,
                    opacity=0.96,
                )
                for index, cell in enumerate(token_cells)
            ],
            run_time=0.85,
        )
        self.wait(1.0)

        # PART 6 -- Feed-forward MLP transforms each token independently
        mlp_caption = TexText(
            "Feed-forward MLP",
            font_size=25,
            color=BLUE_B,
        )
        mlp_caption.move_to(attention_caption)

        selected_index = len(token_cells) // 2
        selected_token = token_cells[selected_index]
        token_home = selected_token.get_center()
        input_vector = self.make_mlp_vector(output=False)
        input_vector.move_to(token_home + LEFT * 2.25 + DOWN * 1.35)
        mlp = self.make_feed_forward_mlp()
        mlp.next_to(input_vector, RIGHT, buff=0.78)
        output_vector = self.make_mlp_vector(output=True)
        output_vector.next_to(mlp, RIGHT, buff=0.76)
        vector_to_mlp_arrow = self.arrow_between(input_vector, mlp, color=BLUE_B, buff=0.08)
        mlp_to_vector_arrow = self.arrow_between(mlp, output_vector, color=YELLOW_B, buff=0.08)
        output_token = self.make_token_cell(index=23, side=selected_token.get_width())
        output_token.set_fill("#2DE2E6", opacity=1.0)
        output_token.set_stroke(WHITE, width=1.15, opacity=1.0)
        output_token.move_to(token_home)

        final_palette = [
            "#2DE2E6",
            "#FF6B6B",
            "#4D96FF",
            "#FFD93D",
            "#9B5DE5",
            "#00C2A8",
            "#FF8FAB",
            "#80ED99",
        ]
        final_token_colors = [
            final_palette[(index * 3 + 1) % len(final_palette)]
            for index in range(len(token_cells))
        ]
        final_token_colors[selected_index] = "#2DE2E6"

        self.play(
            Transform(attention_caption, mlp_caption),
            clip_frame.animate.set_stroke(BLUE_B, width=1.35, opacity=1.0),
            FadeOut(layernorm_stats),
            FadeOut(token_count),
            selected_token.animate.move_to(input_vector.get_center()).scale(1.18),
            run_time=0.85,
        )
        self.play(
            ReplacementTransform(selected_token, input_vector),
            run_time=0.55,
        )
        self.play(
            FadeIn(mlp, shift=RIGHT * 0.12),
            ShowCreation(vector_to_mlp_arrow),
            run_time=0.75,
        )
        self.play(
            mlp[1][1].animate.set_fill(BLUE_B, opacity=0.58).set_stroke(WHITE, width=1.45, opacity=0.96),
            mlp[0].animate.set_stroke(BLUE_B, width=0.8, opacity=0.5),
            run_time=0.5,
        )
        self.play(
            ShowCreation(mlp_to_vector_arrow),
            FadeIn(output_vector, shift=RIGHT * 0.12),
            mlp[1][2].animate.set_fill(YELLOW_B, opacity=0.55).set_stroke(WHITE, width=1.45, opacity=0.96),
            run_time=0.75,
        )
        self.play(
            ReplacementTransform(output_vector, output_token),
            FadeOut(input_vector, shift=LEFT * 0.08),
            FadeOut(mlp),
            FadeOut(vector_to_mlp_arrow),
            FadeOut(mlp_to_vector_arrow),
            run_time=0.95,
        )
        self.play(
            *[
                cell.animate.set_fill(final_token_colors[index], opacity=1.0).set_stroke(
                    WHITE,
                    width=1.05,
                    opacity=0.98,
                )
                for index, cell in enumerate(token_cells)
                if index != selected_index
            ],
            output_token.animate.set_fill(final_token_colors[selected_index], opacity=1.0).set_stroke(
                WHITE,
                width=1.05,
                opacity=0.98,
            ),
            run_time=0.9,
        )
        self.wait(1.0)

        # PART 7 -- Residual connection adds the original tokens back
        residual_caption = TexText(
            "Residual connection",
            font_size=25,
            color=YELLOW_B,
        )
        residual_caption.move_to(attention_caption)

        residual_row, residual_cells, residual_ellipsis = self.make_transformer_token_row()
        residual_row.move_to(token_row_center + DOWN * 0.72)
        residual_row.set_opacity(0.78)

        current_token_mobs = [
            output_token if index == selected_index else token_cells[index]
            for index in range(len(token_cells))
        ]
        residual_plus_marks = VGroup()
        plus_indices = [2, 7, 12, 17, 22, 26]
        for index in plus_indices:
            plus = Tex(r"+", font_size=23, color=YELLOW_B)
            plus.move_to((residual_cells[index].get_center() + current_token_mobs[index].get_center()) / 2)
            residual_plus_marks.add(plus)

        residual_result_palette = [
            "#7EE8FA",
            "#FFB86B",
            "#8AFF80",
            "#FF79C6",
            "#A78BFA",
            "#FDE047",
            "#5EEAD4",
            "#F87171",
        ]
        residual_result_colors = [
            residual_result_palette[(index * 5 + 2) % len(residual_result_palette)]
            for index in range(len(token_cells))
        ]

        self.play(
            Transform(attention_caption, residual_caption),
            clip_frame.animate.set_stroke(YELLOW_B, width=1.35, opacity=1.0),
            FadeIn(residual_row, shift=UP * 0.08),
            LaggedStart(*[
                FadeIn(plus, scale=0.7)
                for plus in residual_plus_marks
            ], lag_ratio=0.08),
            run_time=0.85,
        )
        self.play(
            residual_row.animate.move_to(token_row_center).set_opacity(0.0),
            FadeOut(residual_plus_marks, shift=UP * 0.08),
            *[
                mob.animate.set_fill(residual_result_colors[index], opacity=1.0).set_stroke(
                    WHITE,
                    width=1.08,
                    opacity=0.98,
                )
                for index, mob in enumerate(current_token_mobs)
            ],
            run_time=1.1,
        )
        self.wait(0.85)

        # PART 8 -- Collapse one ViT block into a repeated Transformer stack
        vit_frame_copy = clip_frame.copy()
        vit_frame_copy.set_fill(BLACK, opacity=0)
        vit_frame_copy.set_stroke(BLUE_B, width=1.25, opacity=0.65)

        transformer_stack = self.make_transformer_depth_stack()
        transformer_stack.move_to(ORIGIN + DOWN * 0.08)
        stack_source_layer = transformer_stack[0][0][0]
        transformer_block = self.make_transformer_block(width=3.45, height=0.38, color=BLUE_B)
        transformer_block.move_to(stack_source_layer)
        stack_copy_layers = list(transformer_stack[0][0][1:]) + list(transformer_stack[0][2])
        residual_token_state = Group(token_ellipsis, *current_token_mobs)
        stack_caption = TexText(
            "All 577 tokens flow through, attending to each other",
            font_size=23,
            color=YELLOW_B,
        )
        stack_caption.move_to(attention_caption)

        self.play(
            FadeIn(vit_frame_copy),
            FadeOut(residual_token_state, shift=DOWN * 0.05),
            Transform(attention_caption, stack_caption),
            run_time=0.35,
        )
        self.play(
            Transform(vit_frame_copy, transformer_block),
            run_time=0.9,
        )
        stack_copy_source = vit_frame_copy.copy()
        self.play(
            ReplacementTransform(vit_frame_copy, stack_source_layer),
            LaggedStart(*[
                TransformFromCopy(stack_copy_source, layer)
                for layer in stack_copy_layers
            ], lag_ratio=0.055),
            FadeIn(transformer_stack[0][1], shift=UP * 0.04),
            run_time=1.35,
        )
        self.play(
            FadeIn(transformer_stack[1], shift=LEFT * 0.08),
            run_time=0.45,
        )
        self.wait(0.8)

        token_frame = clip_frame.copy()
        token_frame.set_fill(GREY_E, opacity=0.035)
        token_frame.set_stroke(BLUE_B, width=1.15, opacity=0.9)

        position_token_row, position_token_cells, position_ellipsis = self.make_transformer_token_row(
            left_count=10,
            right_count=4,
            side=0.34,
        )
        position_token_row.move_to(token_row_center + DOWN * 0.02)
        position_palette = [
            "#F94144",
            "#F3722C",
            "#F9C74F",
            "#90BE6D",
            "#43AA8B",
            "#577590",
            "#9B5DE5",
            "#00BBF9",
        ]
        for index, cell in enumerate(position_token_cells):
            cell.set_fill(position_palette[index % len(position_palette)], opacity=1.0)
            cell.set_stroke(WHITE, width=0.95, opacity=0.96)
        position_token_cells[0].set_fill(YELLOW_B, opacity=1.0)
        position_token_cells[0].set_stroke(WHITE, width=1.25, opacity=1.0)

        position_labels = self.make_position_token_labels(position_token_cells)
        cls_pos_label = position_labels[2][0]
        cls_patch_label = position_labels[3][0]
        non_cls_position_labels = VGroup(
            position_labels[0],
            position_labels[1],
            cls_pos_label,
            *position_labels[2][1:],
            *position_labels[3][1:],
        )
        non_cls_tokens = VGroup(
            position_ellipsis,
            *position_token_cells[1:],
        )
        cls_token = position_token_cells[0]
        cls_focus_group = VGroup(cls_token, cls_patch_label)
        cls_vector = self.make_cls_embedding_vector()
        cls_vector.move_to(ORIGIN + DOWN * 0.05)
        projection_caption = Tex(
            r"\text{linear projection }1024\text{-dim}\rightarrow768\text{-dim}",
            font_size=25,
            color=GREEN_B,
        )
        projection_caption.move_to(attention_caption)

        tower_without_bottom = VGroup(
            *transformer_stack[0][0][1:],
            transformer_stack[0][1],
            *transformer_stack[0][2],
            transformer_stack[1],
        )

        self.play(
            FadeOut(tower_without_bottom, shift=UP * 0.05),
            Transform(stack_source_layer[0], token_frame),
            FadeOut(stack_source_layer[1]),
            FadeOut(attention_caption, shift=DOWN * 0.05),
            run_time=0.9,
        )
        self.play(
            FadeIn(position_token_row, shift=UP * 0.08),
            FadeIn(position_labels[0], shift=RIGHT * 0.04),
            FadeIn(position_labels[1], shift=RIGHT * 0.04),
            LaggedStart(*[
                FadeIn(label, shift=UP * 0.02)
                for label in position_labels[2]
            ], lag_ratio=0.01),
            LaggedStart(*[
                FadeIn(label, shift=DOWN * 0.02)
                for label in position_labels[3]
            ], lag_ratio=0.01),
            run_time=1.1,
        )
        self.wait(0.65)
        self.play(
            FadeOut(non_cls_tokens, shift=DOWN * 0.06),
            FadeOut(non_cls_position_labels),
            run_time=0.8,
        )
        self.play(
            cls_token.animate.scale(2.2).move_to(ORIGIN + DOWN * 0.05),
            cls_patch_label.animate.next_to(ORIGIN + DOWN * 0.05, DOWN, buff=0.44),
            FadeIn(projection_caption, shift=UP * 0.05),
            run_time=0.8,
        )
        self.play(
            ReplacementTransform(cls_focus_group, cls_vector),
            run_time=0.9,
        )
        self.wait(0.75)

        # PART 9 -- L2 normalize the final embedding onto the unit sphere
        embedding_space = self.make_embedding_space()
        embedding_shift = UP * 0.24
        embedding_space.shift(embedding_shift)

        raw_endpoint = np.array([1.58, 0.82, 0.55])
        normalized_endpoint = raw_endpoint / np.linalg.norm(raw_endpoint)
        raw_embedding_vector = self.make_embedding_vector(raw_endpoint, color=YELLOW_B)
        raw_embedding_vector.shift(embedding_shift)
        raw_vector_label = Tex(r"\mathbf{v}", font_size=27, color=YELLOW_B)
        raw_vector_label.next_to(raw_embedding_vector[1], UR, buff=0.12)

        normalized_embedding_vector = self.make_embedding_vector(normalized_endpoint, color=GREEN_B)
        normalized_embedding_vector.shift(embedding_shift)
        normalized_vector_label = Tex(r"\hat{\mathbf{v}}", font_size=27, color=GREEN_B)
        normalized_vector_label.next_to(normalized_embedding_vector[1], UR, buff=0.12)

        l2_caption = Tex(
            r"\text{L2-normalize (rescale to unit length)}",
            font_size=25,
            color=GREEN_B,
        )
        l2_caption.move_to(projection_caption)

        self.play(
            FadeOut(stack_source_layer[0]),
            FadeIn(embedding_space[0]),
            FadeIn(embedding_space[2]),
            FadeIn(embedding_space[3]),
            FadeIn(embedding_space[4]),
            Transform(projection_caption, l2_caption),
            FadeOut(cls_vector, shift=DOWN * 0.05),
            FadeIn(raw_embedding_vector, shift=UP * 0.05),
            FadeIn(raw_vector_label, shift=UP * 0.04),
            run_time=1.05,
        )
        self.play(
            FadeIn(embedding_space[1]),
            run_time=0.75,
        )
        self.wait(0.4)
        self.play(
            Transform(raw_embedding_vector, normalized_embedding_vector),
            Transform(raw_vector_label, normalized_vector_label),
            run_time=1.2,
        )
        self.wait(0.9)

        # PART 10 -- Hand the normalized embedding to the trained MLP head
        mlp_head_frame = RoundedRectangle(
            width=clip_frame.get_width(),
            height=clip_frame.get_height(),
            corner_radius=0.14,
            stroke_color=GREEN_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        mlp_head_frame.move_to(clip_frame)
        mlp_head_title = Tex(r"\text{Trained MLP head}", font_size=36, color=WHITE)
        mlp_head_title.move_to(mlp_head_frame.get_top() + DOWN * 0.27)
        mlp_head_subtitle = Tex(
            r"\text{248K params, the only thing that learned face}\rightarrow\text{name}",
            font_size=22,
            color=GREY_A,
        )
        mlp_head_subtitle.next_to(mlp_head_frame, DOWN, buff=0.12)

        final_embedding_vector = self.make_cls_embedding_vector()
        final_embedding_vector.move_to(ORIGIN + DOWN * 0.02)
        exiting_vit = VGroup(clip_frame, clip_title, clip_subtitle)
        embedding_world = Group(
            embedding_space,
            raw_embedding_vector,
            raw_vector_label,
        )

        self.play(
            exiting_vit.animate.scale(2.7, about_point=clip_frame.get_center()).set_opacity(0),
            FadeOut(embedding_world, shift=DOWN * 0.08),
            FadeOut(projection_caption, shift=DOWN * 0.05),
            FadeIn(final_embedding_vector, shift=UP * 0.08),
            FadeIn(mlp_head_frame),
            FadeIn(mlp_head_title, shift=UP * 0.06),
            FadeIn(mlp_head_subtitle, shift=UP * 0.04),
            run_time=1.2,
        )
        self.wait(1.25)
