from manimlib import *
import numpy as np


BG_COLOR = "#0f1117"
DAVID_COLOR = BLUE_C
THOMAS_COLOR = ORANGE
FRAME_COLOR = "#5c647d"
LABEL_COLOR = "#d5d8e8"


def caption_text(text, scale=0.42, color=LABEL_COLOR):
    mob = Text(text, font="Helvetica Neue", color=color)
    mob.scale(scale)
    return mob


def make_overlap_points(seed=7):
    """
    Create two overlapping clouds with only a slight center shift.
    The point is weak separation, not clean clustering.
    """
    rng = np.random.default_rng(seed)

    n = 24
    cov = np.array([[1.0, 0.25], [0.25, 0.75]])
    david_raw = rng.multivariate_normal(mean=[-0.18, 0.08], cov=cov, size=n)
    thomas_raw = rng.multivariate_normal(mean=[0.18, -0.08], cov=cov, size=n)

    all_pts = np.vstack([david_raw, thomas_raw])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)

    def normalize(arr):
        out = np.zeros_like(arr)
        out[:, 0] = np.interp(arr[:, 0], [mins[0], maxs[0]], [-2.85, 2.85])
        out[:, 1] = np.interp(arr[:, 1], [mins[1], maxs[1]], [-1.8, 1.8])
        return out

    return normalize(david_raw), normalize(thomas_raw)


class OverlapScatterPlot(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-2.3, 2.3, 1],
            width=8.2,
            height=5.2,
            axis_config={
                "include_ticks": False,
                "include_tip": False,
                "stroke_color": FRAME_COLOR,
                "stroke_opacity": 0.45,
                "stroke_width": 1.2,
            },
        )

        self.frame_box = SurroundingRectangle(
            self.axes,
            buff=0.0,
            stroke_color=FRAME_COLOR,
            stroke_width=1.5,
        )

        self.david_data, self.thomas_data = make_overlap_points(seed=7)
        # Backward-compatible aliases for older scene code that still uses blue/orange naming.
        self.blue_data = self.david_data
        self.orange_data = self.thomas_data

        self.david_dots = VGroup(*[
            Dot(self.axes.c2p(x, y), radius=0.055, color=DAVID_COLOR).set_opacity(0.92)
            for x, y in self.david_data
        ])
        self.thomas_dots = VGroup(*[
            Dot(self.axes.c2p(x, y), radius=0.055, color=THOMAS_COLOR).set_opacity(0.92)
            for x, y in self.thomas_data
        ])

        self.david_label = caption_text("David", scale=0.5, color=DAVID_COLOR)
        self.thomas_label = caption_text("Thomas", scale=0.5, color=THOMAS_COLOR)
        self.david_label.next_to(self.axes, UL, buff=0.14).shift(RIGHT * 0.28 + DOWN * 0.08)
        self.thomas_label.next_to(self.axes, UR, buff=0.14).shift(LEFT * 0.28 + DOWN * 0.08)

        self.add(self.axes, self.frame_box, self.david_dots, self.thomas_dots)

    def get_david_dot(self, idx):
        return self.david_dots[idx]

    def get_thomas_dot(self, idx):
        return self.thomas_dots[idx]

    def get_blue_dot(self, idx):
        return self.get_david_dot(idx)

    def get_orange_dot(self, idx):
        return self.get_thomas_dot(idx)


def get_distance(p, q):
    return np.linalg.norm(p - q)
