from manimlib import *
import numpy as np


class RetinaFaceMLPHeadScenePart2(Scene):
    def make_weight_matrix(self):
        return Tex(
            r"\mathbf{W}="
            r"\left[\begin{array}{cccccc}"
            r"0.12&-0.08&0.05&\cdots&-0.02&0.03\\"
            r"-0.04&0.19&-0.07&\cdots&0.06&-0.11\\"
            r"0.09&0.01&0.14&\cdots&-0.05&0.02\\"
            r"-0.13&0.04&0.08&\cdots&0.10&-0.06\\"
            r"\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\"
            r"0.07&0.02&-0.10&\cdots&0.04&0.16"
            r"\end{array}\right]",
            font_size=27,
            color=WHITE,
        )

    def make_vertical_vector(self, values, color=YELLOW_B):
        return Tex(
            r"\left[\begin{array}{c}"
            + r"\\".join(values)
            + r"\end{array}\right]",
            font_size=30,
            color=color,
        )

    def make_fixed_width_vertical_vector(self, values, color=YELLOW_B, cell_width="0.90cm"):
        boxed_values = [
            rf"\makebox[{cell_width}][c]{{${value}$}}"
            for value in values
        ]
        return Tex(
            r"\left[\begin{array}{c}"
            + r"\\".join(boxed_values)
            + r"\end{array}\right]",
            font_size=30,
            color=color,
        )

    def make_vector_annotations(self, vector, name=None, color=GREY_A, side=LEFT):
        height_brace = Brace(vector, side, buff=0.05)
        height_brace.set_stroke(color, width=1.0, opacity=0.82)
        height_label = Tex("256", font_size=28, color=color)
        height_label.next_to(height_brace, side, buff=0.08)
        labels = VGroup(height_brace, height_label)
        if name:
            name_label = Tex(rf"\text{{{name}}}", font_size=24, color=color)
            name_label.next_to(vector, DOWN, buff=0.16)
            labels.add(name_label)
        return labels

    def make_linear_block(self, label_text="Linear (Weight)", width=1.82, font_size=14):
        box = RoundedRectangle(
            width=width,
            height=0.48,
            corner_radius=0.09,
            stroke_color=GREEN_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        label = TexText(label_text, font_size=font_size, color=WHITE)
        label.move_to(box)
        return VGroup(box, label)

    def make_dimension_linear_block(self, in_dim, out_dim):
        box = RoundedRectangle(
            width=1.82,
            height=0.78,
            corner_radius=0.09,
            stroke_color=GREEN_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        label = Tex(
            rf"\begin{{array}}{{c}}\text{{Linear}}\\{in_dim}\rightarrow{out_dim}\end{{array}}",
            font_size=17,
            color=WHITE,
        )
        label.move_to(box)
        return VGroup(box, label)

    def make_logits_block(self):
        box = RoundedRectangle(
            width=1.52,
            height=0.78,
            corner_radius=0.09,
            stroke_color=GREEN_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        label = Tex(
            r"\begin{array}{c}138\ \text{logits}\\\text{(real-valued)}\end{array}",
            font_size=16,
            color=WHITE,
        )
        label.move_to(box)
        return VGroup(box, label)

    def make_weight_matrix_138_128(self):
        matrix = Tex(
            r"\mathbf{W}="
            r"\left[\begin{array}{cccccc}"
            r"0.18&-0.04&0.07&\cdots&0.11&-0.02\\"
            r"-0.09&0.13&-0.05&\cdots&0.04&0.08\\"
            r"0.06&0.02&0.16&\cdots&-0.03&0.10\\"
            r"\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\"
            r"0.12&-0.07&0.03&\cdots&0.15&-0.06"
            r"\end{array}\right]",
            font_size=21,
            color=WHITE,
        )

        rows_brace = Brace(matrix, RIGHT, buff=0.05)
        rows_brace.set_stroke(GREEN_B, width=0.9, opacity=0.86)
        rows_label = Tex("138", font_size=21, color=GREEN_B)
        rows_label.next_to(rows_brace, RIGHT, buff=0.06)

        cols_brace = Brace(matrix, DOWN, buff=0.05)
        cols_brace.set_stroke(GREEN_B, width=0.9, opacity=0.86)
        cols_label = Tex("128", font_size=21, color=GREEN_B)
        cols_label.next_to(cols_brace, DOWN, buff=0.06)
        return VGroup(matrix, rows_brace, rows_label, cols_brace, cols_label)

    def make_horizontal_vector(self, values, color=GREEN_B):
        return Tex(
            r"\left["
            + r",\ ".join(values)
            + r"\right]",
            font_size=34,
            color=color,
        )

    def make_relu_function(self):
        box = RoundedRectangle(
            width=4.35,
            height=0.86,
            corner_radius=0.08,
            stroke_color=YELLOW_B,
            stroke_width=1.25,
            fill_color="#111820",
            fill_opacity=0.94,
        )
        formula = Tex(r"\mathrm{ReLU}(x)=\max(0,x)", font_size=25, color=WHITE)
        formula.move_to(box.get_center() + LEFT * 0.55)

        graph_origin = box.get_center() + RIGHT * 1.45 + DOWN * 0.18
        x_axis = Line(graph_origin + LEFT * 0.42, graph_origin + RIGHT * 0.55)
        y_axis = Line(graph_origin + DOWN * 0.18, graph_origin + UP * 0.48)
        axes = VGroup(x_axis, y_axis)
        axes.set_stroke(GREY_A, width=1.0, opacity=0.65)

        flat = Line(graph_origin + LEFT * 0.38, graph_origin)
        rising = Line(graph_origin, graph_origin + RIGHT * 0.48 + UP * 0.38)
        relu_curve = VGroup(flat, rising)
        relu_curve.set_stroke(YELLOW_B, width=2.0, opacity=0.95)

        return VGroup(box, formula, axes, relu_curve)

    def make_relu_block(self):
        box = RoundedRectangle(
            width=0.78,
            height=0.78,
            corner_radius=0.08,
            stroke_color=YELLOW_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.94,
        )

        graph_origin = box.get_center() + UP * 0.10
        x_axis = Line(graph_origin + LEFT * 0.22, graph_origin + RIGHT * 0.24)
        y_axis = Line(graph_origin + DOWN * 0.12, graph_origin + UP * 0.22)
        axes = VGroup(x_axis, y_axis)
        axes.set_stroke(GREY_A, width=0.75, opacity=0.6)

        flat = Line(graph_origin + LEFT * 0.19, graph_origin)
        rising = Line(graph_origin, graph_origin + RIGHT * 0.22 + UP * 0.17)
        relu_curve = VGroup(flat, rising)
        relu_curve.set_stroke(YELLOW_B, width=1.6, opacity=0.96)

        label = TexText("ReLU", font_size=12, color=WHITE)
        label.move_to(box.get_center() + DOWN * 0.22)
        return VGroup(box, axes, relu_curve, label)

    def make_two_layer_mlp(self, layer_sizes=(5, 4)):
        node_layers = VGroup()
        for size in layer_sizes:
            nodes = VGroup()
            for _ in range(size):
                node = Circle(radius=0.095)
                node.set_fill(WHITE, opacity=0.0)
                node.set_stroke(WHITE, width=1.45, opacity=0.85)
                nodes.add(node)
            nodes.arrange(DOWN, buff=0.18)
            node_layers.add(nodes)
        node_layers.arrange(RIGHT, buff=1.0)

        connections = VGroup()
        for left_node in node_layers[0]:
            for right_node in node_layers[1]:
                line = Line(left_node.get_center(), right_node.get_center())
                line.set_stroke(GREY_B, width=0.85, opacity=0.34)
                connections.add(line)

        return VGroup(connections, node_layers)

    def construct(self):
        self.camera.background_color = BLACK

        mlp_head_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=GREEN_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        mlp_head_frame.move_to(DOWN * 0.04)

        mlp_head_title = Tex(r"\text{Trained MLP head}", font_size=36, color=WHITE)
        mlp_head_title.move_to(mlp_head_frame.get_top() + DOWN * 0.27)
        mlp_head_subtitle = Tex(
            r"\text{248K params, the only thing that learned face}\rightarrow\text{name}",
            font_size=22,
            color=GREY_A,
        )
        mlp_head_subtitle.next_to(mlp_head_frame, DOWN, buff=0.12)

        weight_matrix = self.make_weight_matrix()
        weight_matrix.move_to(RIGHT * 0.92 + UP * 0.08)
        compressed_reference = self.make_vertical_vector(
            ["0.21", "-0.09", r"\vdots", "0.27"],
            color=GREEN_B,
        )
        compressed_reference.next_to(weight_matrix, LEFT, buff=0.86)
        compressed_reference.shift(UP * 0.08)
        compressed_annotations = self.make_vector_annotations(
            compressed_reference,
            color=GREEN_B,
            side=LEFT,
        )
        activation_anchor = compressed_reference.get_center()
        activation_left_x = compressed_reference.get_left()[0]

        def place_activation_vector(vector):
            vector.move_to(activation_anchor)
            vector.shift(RIGHT * (activation_left_x - vector.get_left()[0]))
            return vector

        affine_vector = self.make_fixed_width_vertical_vector(
            ["0.12", "0.04", r"\vdots", "0.31"],
            color=GREEN_B,
        )
        place_activation_vector(affine_vector)
        affine_vector.shift(DOWN * 0.01)

        linear_block = self.make_linear_block("Linear (Weight)")
        linear_block.move_to(LEFT * 3.75 + DOWN * 1.48)
        bias_block = self.make_linear_block("Linear (Bias)")
        bias_block.next_to(linear_block, DOWN, buff=0.07)
        batchnorm_block = self.make_linear_block("BatchNorm-1D (256)", width=2.25)
        batchnorm_block.next_to(VGroup(linear_block, bias_block), RIGHT, buff=0.18)

        batchnorm_caption = Tex(
            r"\text{BatchNorm-1D }(256)",
            font_size=25,
            color=BLUE_B,
        )
        batchnorm_caption.move_to(mlp_head_frame.get_bottom() + UP * 0.34)

        self.add(
            mlp_head_frame,
            mlp_head_title,
            mlp_head_subtitle,
            affine_vector,
            compressed_annotations,
            linear_block,
            bias_block,
            batchnorm_block,
            batchnorm_caption,
        )
        self.wait(1.0)

        pre_relu_vector = self.make_horizontal_vector(
            ["0.12", "0.04", "-0.15", "0.07", "0.23", r"\cdots", "0.14", "0.31", "-0.22", "0.31"],
            color=GREEN_B,
        )
        pre_relu_vector.move_to(ORIGIN + UP * 1.55)

        relu_function = self.make_relu_function()
        relu_function.move_to(ORIGIN + UP * 0.34)

        post_relu_vector = self.make_horizontal_vector(
            ["0.12", "0.04", "0", "0.07", "0.23", r"\cdots", "0.14", "0.31", "0", "0.31"],
            color=YELLOW_B,
        )
        post_relu_vector.move_to(ORIGIN + DOWN * 0.72)

        relu_caption = Tex(r"\text{ReLU activation}", font_size=25, color=YELLOW_B)
        relu_caption.move_to(batchnorm_caption)

        self.play(
            FadeOut(compressed_annotations, shift=LEFT * 0.08),
            Transform(affine_vector, pre_relu_vector),
            Transform(batchnorm_caption, relu_caption),
            run_time=0.85,
        )
        self.play(
            FadeIn(relu_function, shift=UP * 0.05),
            run_time=0.45,
        )
        self.play(
            Transform(affine_vector, post_relu_vector),
            relu_function[0].animate.set_stroke(YELLOW_B, width=1.8, opacity=1.0),
            run_time=0.95,
        )
        self.play(
            relu_function[0].animate.set_stroke(YELLOW_B, width=1.25, opacity=0.9),
            run_time=0.25,
        )
        compact_batchnorm_block = self.make_linear_block(
            "BatchNorm-1D (256)",
            width=1.82,
        )
        compact_batchnorm_block.next_to(VGroup(linear_block, bias_block), RIGHT, buff=0.18)
        compact_batchnorm_block.shift(LEFT * 0.12)

        relu_block = self.make_relu_block()
        relu_block.next_to(compact_batchnorm_block, RIGHT, buff=0.07)
        self.play(
            Transform(batchnorm_block, compact_batchnorm_block),
            ReplacementTransform(relu_function, relu_block),
            run_time=0.75,
        )
        self.wait(0.3)

        training_vector = affine_vector.copy()
        training_vector.generate_target()
        training_vector.target.move_to(ORIGIN + UP * 1.34)

        dropout_vector = self.make_horizontal_vector(
            ["0.12", "0", "0", "0.07", "0", r"\cdots", "0.14", "0.31", "0", "0.31"],
            color=TEAL_B,
        )
        dropout_vector.move_to(training_vector.target)

        training_box = SurroundingRectangle(dropout_vector, buff=0.18)
        training_box.stretch(1.12, dim=0)
        training_box.set_fill(TEAL_B, opacity=0.035)
        training_box.set_stroke(TEAL_B, width=1.45, opacity=0.9)
        training_label = TexText("Dropout (During Training)", font_size=22, color=TEAL_B)
        training_label.next_to(training_box, UP, buff=0.14)

        dropout_caption = Tex(r"\text{Dropout }(p=0.3)", font_size=25, color=TEAL_B)
        dropout_caption.move_to(batchnorm_caption)

        self.add(training_vector)
        self.play(
            MoveToTarget(training_vector),
            run_time=0.65,
        )
        self.play(
            FadeIn(training_box, scale=0.98),
            FadeIn(training_label, shift=UP * 0.04),
            Transform(batchnorm_caption, dropout_caption),
            run_time=0.55,
        )
        self.play(
            Transform(training_vector, dropout_vector),
            run_time=0.75,
        )
        self.wait(0.35)
        self.play(
            FadeOut(batchnorm_caption),
            run_time=0.3,
        )

        centered_relu_vector = self.make_horizontal_vector(
            ["0.12", "0.04", "0", "0.07", "0.23", r"\cdots", "0.14", "0.31", "0", "0.31"],
            color=YELLOW_B,
        )
        centered_relu_vector.move_to(affine_vector.get_center() + UP * 0.35)
        length_brace = Brace(centered_relu_vector, UP, buff=0.08)
        length_brace.set_stroke(YELLOW_B, width=1.1, opacity=0.88)
        length_label = Tex("256", font_size=28, color=YELLOW_B)
        length_label.next_to(length_brace, UP, buff=0.08)
        length_group = VGroup(length_brace, length_label)

        merged_linear_block = self.make_dimension_linear_block(768, 256)
        merged_linear_block.move_to(VGroup(linear_block, bias_block).get_center())

        second_linear_block = self.make_dimension_linear_block(256, 128)
        second_linear_block.next_to(relu_block, RIGHT, buff=0.07)

        self.play(
            FadeOut(training_box),
            FadeOut(training_label, shift=UP * 0.04),
            FadeOut(training_vector, shift=UP * 0.08),
            Transform(affine_vector, centered_relu_vector),
            FadeIn(length_group, shift=UP * 0.05),
            ReplacementTransform(VGroup(linear_block, bias_block), merged_linear_block),
            FadeIn(second_linear_block, shift=RIGHT * 0.08),
            run_time=0.95,
        )

        self.wait(0.35)

        mlp_network = self.make_two_layer_mlp()
        mlp_network.scale(0.94)
        mlp_network.next_to(length_group, UP, buff=0.22)
        shortened_vector = self.make_horizontal_vector(
            ["0.41", "-0.08", "0.16", r"\cdots", "0.27", "-0.11"],
            color=GREEN_B,
        )
        shortened_vector.move_to(centered_relu_vector)
        shortened_brace = Brace(shortened_vector, UP, buff=0.08)
        shortened_brace.set_stroke(GREEN_B, width=1.1, opacity=0.88)
        shortened_label = Tex("128", font_size=28, color=GREEN_B)
        shortened_label.next_to(shortened_brace, UP, buff=0.08)
        shortened_length_group = VGroup(shortened_brace, shortened_label)

        left_layer = mlp_network[1][0]
        right_layer = mlp_network[1][1]
        mlp_connections = mlp_network[0]
        self.play(
            FadeIn(mlp_network, shift=DOWN * 0.05),
            run_time=0.55,
        )
        self.play(
            left_layer.animate.set_fill(WHITE, opacity=0.82).set_stroke(WHITE, width=2.2, opacity=1.0),
            run_time=0.55,
        )
        self.play(
            mlp_connections.animate.set_stroke(WHITE, width=1.65, opacity=0.72),
            run_time=0.55,
        )
        self.play(
            mlp_connections.animate.set_stroke(GREY_B, width=0.85, opacity=0.34),
            left_layer.animate.set_fill(WHITE, opacity=0.0).set_stroke(WHITE, width=1.45, opacity=0.85),
            right_layer.animate.set_fill(WHITE, opacity=0.82).set_stroke(WHITE, width=2.2, opacity=1.0),
            Transform(affine_vector, shortened_vector),
            Transform(length_group, shortened_length_group),
            run_time=0.85,
        )
        self.wait(0.55)

        bn_input_vector = self.make_horizontal_vector(
            ["0.41", "-0.08", "0.16", r"\cdots", "0.27", "-0.11"],
            color=GREEN_B,
        )
        bn_input_vector.move_to(shortened_vector.get_center() + UP * 0.55)

        batchnorm128_block = self.make_linear_block("BatchNorm-1D (128)", width=1.82, font_size=12)
        batchnorm128_block.next_to(second_linear_block, RIGHT, buff=0.07)

        self.play(
            FadeOut(mlp_network, shift=UP * 0.06),
            FadeOut(length_group, shift=UP * 0.04),
            Transform(affine_vector, bn_input_vector),
            FadeIn(batchnorm128_block, shift=RIGHT * 0.08),
            run_time=0.85,
        )
        self.wait(0.25)

        normalized_bn_vector = self.make_horizontal_vector(
            ["1.18", "-0.52", "0.06", r"\cdots", "0.74", "-0.31"],
            color=BLUE_B,
        )
        normalized_bn_vector.move_to(bn_input_vector)
        normalization_box = SurroundingRectangle(normalized_bn_vector, buff=0.18)
        normalization_box.stretch(1.07, dim=0)
        normalization_box.set_fill(BLUE_B, opacity=0.035)
        normalization_box.set_stroke(BLUE_B, width=1.35, opacity=0.92)
        normalization_label = TexText("Normalization", font_size=22, color=BLUE_B)
        normalization_label.next_to(normalization_box, UP, buff=0.14)

        affine_bn_vector = self.make_horizontal_vector(
            ["0.95", "-0.16", "0.22", r"\cdots", "0.63", "0.04"],
            color=GREEN_B,
        )
        affine_bn_vector.move_to(bn_input_vector)
        affine_box = SurroundingRectangle(affine_bn_vector, buff=0.18)
        affine_box.stretch(1.07, dim=0)
        affine_box.set_fill(GREEN_B, opacity=0.035)
        affine_box.set_stroke(GREEN_B, width=1.35, opacity=0.92)
        affine_label = TexText("Affine Transformation", font_size=22, color=GREEN_B)
        affine_label.next_to(affine_box, UP, buff=0.14)

        self.play(
            FadeIn(normalization_box, scale=0.98),
            FadeIn(normalization_label, shift=UP * 0.04),
            Transform(affine_vector, normalized_bn_vector),
            run_time=0.85,
        )
        self.wait(0.25)
        self.play(
            FadeOut(normalization_box),
            FadeOut(normalization_label, shift=UP * 0.03),
            FadeIn(affine_box, scale=0.98),
            FadeIn(affine_label, shift=UP * 0.04),
            Transform(affine_vector, affine_bn_vector),
            run_time=0.85,
        )
        self.wait(0.25)

        relu128_vector = self.make_horizontal_vector(
            ["0.95", "0", "0.22", r"\cdots", "0.63", "0.04"],
            color=YELLOW_B,
        )
        relu128_vector.move_to(affine_bn_vector)
        relu128_box = SurroundingRectangle(relu128_vector, buff=0.18)
        relu128_box.stretch(1.07, dim=0)
        relu128_box.set_fill(YELLOW_B, opacity=0.035)
        relu128_box.set_stroke(YELLOW_B, width=1.35, opacity=0.92)
        relu128_label = TexText("ReLU", font_size=22, color=YELLOW_B)
        relu128_label.next_to(relu128_box, UP, buff=0.14)

        relu128_block = self.make_relu_block()
        relu128_block.next_to(batchnorm128_block, RIGHT, buff=0.07)

        self.play(
            FadeOut(affine_box),
            FadeOut(affine_label, shift=UP * 0.03),
            FadeIn(relu128_box, scale=0.98),
            FadeIn(relu128_label, shift=UP * 0.04),
            FadeIn(relu128_block, shift=RIGHT * 0.08),
            run_time=0.55,
        )
        self.play(
            Transform(affine_vector, relu128_vector),
            run_time=0.75,
        )
        self.wait(0.25)

        training128_vector = affine_vector.copy()
        training128_vector.set_opacity(0.46)
        training128_vector.generate_target()
        training128_vector.target.move_to(relu128_vector.get_center() + UP * 0.82)
        training128_vector.target.set_opacity(0.46)

        dropout128_vector = self.make_horizontal_vector(
            ["0", "0", "0.22", r"\cdots", "0", "0.04"],
            color=TEAL_B,
        )
        dropout128_vector.move_to(training128_vector.target)
        dropout128_vector.set_opacity(0.58)

        dropout128_label = TexText("Dropout (During Training)", font_size=22, color=TEAL_B)
        dropout128_label.next_to(dropout128_vector, UP, buff=0.18)

        self.play(
            FadeOut(relu128_box),
            FadeOut(relu128_label, shift=UP * 0.03),
            run_time=0.35,
        )
        self.add(training128_vector)
        self.play(
            MoveToTarget(training128_vector),
            FadeIn(dropout128_label, shift=UP * 0.04),
            run_time=0.7,
        )
        self.play(
            Transform(training128_vector, dropout128_vector),
            run_time=0.75,
        )
        self.wait(0.35)

        final_input_vector = self.make_horizontal_vector(
            ["0.95", "0", "0.22", r"\cdots", "0.63", "0.04"],
            color=YELLOW_B,
        )
        final_input_vector.move_to(relu128_vector.get_center() + DOWN * 0.55)

        final_linear_block = self.make_dimension_linear_block(128, 138)
        final_linear_block.next_to(relu_block, DOWN, buff=0.12)
        final_linear_block.align_to(relu_block, RIGHT)
        final_linear_block.shift(UP * 0.08)

        self.play(
            FadeOut(training128_vector, shift=UP * 0.06),
            FadeOut(dropout128_label, shift=UP * 0.04),
            Transform(affine_vector, final_input_vector),
            run_time=0.85,
        )

        self.wait(0.3)

        final_mlp_network = self.make_two_layer_mlp(layer_sizes=(4, 6))
        final_mlp_network.scale(0.94)
        final_mlp_network.next_to(affine_vector, UP, buff=0.42)
        final_mlp_center = final_mlp_network.get_center()

        output138_vector = self.make_horizontal_vector(
            ["0.18", "-0.04", "0.33", "0.09", r"\cdots", "-0.12", "0.27", "0.05"],
            color=GREEN_B,
        )
        output138_vector.move_to(final_input_vector)
        output138_vector.shift(DOWN * 0.18)
        output138_brace = Brace(output138_vector, UP, buff=0.06)
        output138_brace.set_stroke(GREEN_B, width=0.9, opacity=0.88)
        output138_label = Tex("138", font_size=23, color=GREEN_B)
        output138_label.next_to(output138_brace, UP, buff=0.06)
        output138_length_group = VGroup(output138_brace, output138_label)

        logits_block = self.make_logits_block()
        logits_block.next_to(final_linear_block, RIGHT, buff=0.07)

        final_left_layer = final_mlp_network[1][0]
        final_right_layer = final_mlp_network[1][1]
        final_connections = final_mlp_network[0]
        self.play(
            FadeIn(final_mlp_network, shift=DOWN * 0.05),
            FadeIn(final_linear_block, shift=RIGHT * 0.08),
            run_time=0.65,
        )
        self.play(
            final_left_layer.animate.set_fill(WHITE, opacity=0.82).set_stroke(WHITE, width=2.2, opacity=1.0),
            run_time=0.55,
        )
        self.play(
            final_connections.animate.set_stroke(WHITE, width=1.65, opacity=0.72),
            run_time=0.55,
        )
        self.play(
            final_connections.animate.set_stroke(GREY_B, width=0.85, opacity=0.34),
            final_left_layer.animate.set_fill(WHITE, opacity=0.0).set_stroke(WHITE, width=1.45, opacity=0.85),
            final_right_layer.animate.set_fill(WHITE, opacity=0.82).set_stroke(WHITE, width=2.2, opacity=1.0),
            Transform(affine_vector, output138_vector),
            FadeIn(output138_length_group, shift=UP * 0.05),
            FadeIn(logits_block, shift=RIGHT * 0.08),
            run_time=0.85,
        )
        self.play(
            final_right_layer.animate.set_fill(WHITE, opacity=0.0).set_stroke(WHITE, width=1.45, opacity=0.85),
            run_time=0.25,
        )
        row_caption = TexText("one matrix row per name", font_size=22, color=GREY_A)
        row_caption.next_to(final_mlp_network, RIGHT, buff=0.35)
        right_nodes = list(final_right_layer)
        self.play(
            FadeIn(row_caption, shift=RIGHT * 0.05),
            right_nodes[0].animate.set_fill(WHITE, opacity=1.0).set_stroke(WHITE, width=3.0, opacity=1.0),
            run_time=0.16,
        )
        self.play(
            right_nodes[1].animate.set_fill(WHITE, opacity=1.0).set_stroke(WHITE, width=3.0, opacity=1.0),
            run_time=0.16,
        )
        for index in range(2, len(right_nodes)):
            self.play(
                right_nodes[index - 2].animate.set_fill(WHITE, opacity=0.0).set_stroke(WHITE, width=1.45, opacity=0.85),
                right_nodes[index].animate.set_fill(WHITE, opacity=1.0).set_stroke(WHITE, width=3.0, opacity=1.0),
                run_time=0.16,
            )
        self.play(
            right_nodes[-2].animate.set_fill(WHITE, opacity=0.0).set_stroke(WHITE, width=1.45, opacity=0.85),
            run_time=0.12,
        )
        self.play(
            right_nodes[-1].animate.set_fill(WHITE, opacity=0.0).set_stroke(WHITE, width=1.45, opacity=0.85),
            run_time=0.12,
        )
        self.wait(0.5)
        self.play(
            FadeOut(output138_length_group, shift=UP * 0.04),
            run_time=0.35,
        )
        final_vector_centered = output138_vector.copy()
        final_vector_centered.move_to(np.array([
            output138_vector.get_center()[0],
            final_mlp_center[1],
            0.0,
        ]))
        self.play(
            FadeOut(final_mlp_network, shift=UP * 0.05),
            FadeOut(row_caption, shift=RIGHT * 0.05),
            Transform(affine_vector, final_vector_centered),
            run_time=0.85,
        )

        def make_param_counter(value):
            latex_value = f"{value:,}".replace(",", "{,}")
            return Tex(
                rf"\text{{Total Trainable Params: }}{latex_value}",
                font_size=25,
                color=WHITE,
            )

        def get_param_jump_values(start_value, end_value, n_steps=5):
            values = [
                int(round(start_value + (end_value - start_value) * alpha))
                for alpha in np.linspace(1 / n_steps, 1, n_steps)
            ]
            values[-1] = end_value
            return values

        pipeline_group = VGroup(
            merged_linear_block,
            batchnorm_block,
            relu_block,
            second_linear_block,
            batchnorm128_block,
            relu128_block,
            final_linear_block,
            logits_block,
        )
        param_counter = make_param_counter(0)
        param_counter.move_to(np.array([
            pipeline_group.get_center()[0],
            pipeline_group.get_top()[1] + 0.82,
            0.0,
        ]))
        self.play(
            FadeIn(param_counter, shift=UP * 0.04),
            run_time=0.45,
        )

        param_steps = [
            (merged_linear_block, 196864),
            (batchnorm_block, 197376),
            (second_linear_block, 230272),
            (batchnorm128_block, 230528),
            (final_linear_block, 248330),
        ]
        previous_block = None
        current_param_value = 0
        for block, target_value in param_steps:
            animations = [
                block[0].animate.set_stroke(YELLOW_B, width=2.35, opacity=1.0),
                block[0].animate.set_fill("#25323d", opacity=0.98),
            ]
            if previous_block is not None:
                animations.extend([
                    previous_block[0].animate.set_stroke(GREEN_B, width=1.15, opacity=0.9),
                    previous_block[0].animate.set_fill("#111820", opacity=0.92),
                ])
            self.play(
                *animations,
                run_time=0.18,
            )
            for jump_value in get_param_jump_values(current_param_value, target_value):
                next_counter = make_param_counter(jump_value)
                next_counter.move_to(param_counter)
                self.play(
                    Transform(param_counter, next_counter),
                    run_time=0.055,
                )
            self.wait(0.08)
            previous_block = block
            current_param_value = target_value
        if previous_block is not None:
            self.play(
                previous_block[0].animate.set_stroke(GREEN_B, width=1.15, opacity=0.9),
                previous_block[0].animate.set_fill("#111820", opacity=0.92),
                run_time=0.25,
            )
        self.wait(0.8)
