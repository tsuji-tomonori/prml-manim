"""Shaded objects and a projected 3D loss surface for Manim CE's Cairo renderer.

Only the coefficient inset has a moving camera. Japanese text, subtitles and
the data plot stay in screen space. The surface vertices are actual (w0, w1, E)
coordinates; perspective, lighting and depth sorting are applied together.
"""
from __future__ import annotations

import numpy as np
from manim import (
    BLACK, WHITE, DOWN, RIGHT, UP, Circle, Line, ManimColor, Polygon,
    RoundedRectangle, ValueTracker, VGroup, VMobject, interpolate_color,
)

from polynomial_model import WEIGHTS, linear_fit_error, linear_loss_ring

PANEL = ManimColor("#182231")
EDGE = ManimColor("#3C5069")
PURPLE = ManimColor("#C29AFF")
GREEN = ManimColor("#77D49A")
YELLOW = ManimColor("#FFE079")


class DepthDot(VGroup):
    """A shaded bead whose geometric center remains the exact plotted value."""
    def __init__(self, point=(0, 0, 0), radius=.06, color=WHITE):
        super().__init__()
        base = ManimColor(color)
        for radius_scale, offset, tone in [
            (1, 0, interpolate_color(base, BLACK, .47)),
            (.91, .03, base),
            (.70, .16, interpolate_color(base, WHITE, .19)),
            (.39, .29, interpolate_color(base, WHITE, .53)),
            (.15, .34, interpolate_color(base, WHITE, .85)),
        ]:
            self.add(Circle(radius=radius * radius_scale, stroke_width=0,
                            fill_color=tone, fill_opacity=1)
                     .shift(radius * offset * (UP + .65 * -RIGHT)))
        self.move_to(point)


def raised_panel(center, width, height):
    """A shallow board with a cast shadow; its face is behind chart geometry."""
    shadow = RoundedRectangle(width=width + .10, height=height + .10,
                              corner_radius=.13, stroke_width=0,
                              fill_color=BLACK, fill_opacity=.35)
    shadow.move_to(center).shift(.10 * DOWN + .08 * RIGHT).set_z_index(-7)
    edge = RoundedRectangle(width=width, height=height, corner_radius=.11,
                            stroke_width=0, fill_color=EDGE, fill_opacity=1)
    edge.move_to(center).shift(.055 * DOWN).set_z_index(-6)
    face = RoundedRectangle(width=width, height=height, corner_radius=.11,
                            stroke_color=EDGE, stroke_width=.65,
                            fill_color=PANEL, fill_opacity=1)
    face.move_to(center).set_z_index(-5)
    return VGroup(shadow, edge, face)


def relief_faces(front, color, thickness=.045):
    """Constant decorative thickness; the front retains its data-encoded area."""
    offset = thickness * (RIGHT + DOWN)
    corners = [front.get_corner(v) for v in [UP + RIGHT, DOWN + RIGHT, DOWN - RIGHT]]
    sides = VGroup(*[
        Polygon(a, b, b + offset, a + offset, stroke_width=0,
                fill_color=interpolate_color(color, BLACK, shade), fill_opacity=1)
        for a, b, shade in [(corners[0], corners[1], .35),
                            (corners[1], corners[2], .55)]])
    return VGroup(sides, front)


class LossLandscape(VGroup):
    """Lift a contour map into the exact linear-regression error bowl.

    The rim is E_min + 4, so no huge off-screen corner peaks are clipped.
    At lift=0 the view is a coefficient map. At lift=1, the SAME coefficient
    positions have height E, including the positive error at the optimum.
    """
    RIM_EXCESS = 4.
    HEIGHT_SCALE = .66
    FINAL_ELEVATION = 36.
    FINAL_AZIMUTH = -26.

    def __init__(self, center=(3.3, .18, 0), width=5.7, height=3.9):
        super().__init__()
        self.screen_center = np.array(center, dtype=float)
        self.best = WEIGHTS[1][:2]
        self.minimum = float(linear_fit_error(*self.best))
        self.lift = ValueTracker(0)
        self.elevation = ValueTracker(90)
        self.azimuth = ValueTracker(0)
        self.surface_opacity = ValueTracker(0)
        self._last_state = None

        # Radial mesh cuts the quadratic surface at a constant, meaningful E.
        ring = linear_loss_ring(self.RIM_EXCESS, 49)[:-1]
        radii = np.linspace(0, 1, 13)
        mesh = self.best + radii[:, None, None] * (ring - self.best)
        self.vertices = np.dstack((mesh, linear_fit_error(mesh[..., 0], mesh[..., 1])))
        self.quads = []
        self.surface = VGroup()
        for i in range(12):
            for j in range(48):
                k = (j + 1) % 48
                vertices = self.vertices[[i, i + 1, i + 1, i], [j, j, k, k]]
                face = Polygon(*np.zeros((4, 3)), stroke_width=.35)
                self.quads.append((face, vertices))
                self.surface.add(face)

        rim = np.c_[ring, np.zeros(len(ring))]
        self.floor_vertices = np.vstack((rim, rim[0]))
        self.floor = Polygon(*np.zeros((len(rim), 3)), fill_color=PANEL,
                             fill_opacity=1, stroke_color=EDGE, stroke_width=1)
        self.shadow = self.floor.copy().set_fill(BLACK, .4).set_stroke(width=0)
        self.contours = VGroup()
        self.contour_vertices = []
        for excess in [.10, .40, 1., 2., 4.]:
            xy = linear_loss_ring(excess)
            self.contour_vertices.append(np.c_[xy, np.full(len(xy), self.minimum + excess)])
            self.contours.add(VMobject().set_stroke(PURPLE, 1.2, .6))

        # Coordinate axes through (0,0), rather than through the optimum.
        low, high = ring.min(axis=0), ring.max(axis=0)
        self.axis_vertices = [
            np.array([[low[0], 0, 0], [high[0], 0, 0]]),
            np.array([[0, low[1], 0], [0, high[1], 0]]),
        ]
        self.floor_axes = VGroup(*[Line(UP, DOWN, color=EDGE, stroke_width=1.6) for _ in range(2)])
        self.axis_ends = [high[0], high[1]]

        # Fix one scale for the entire reveal; the height axis is never rescaled.
        bounds = []
        for elevation, azimuth, lift in [(90, 0, 0), (60, -13, .5),
                                          (self.FINAL_ELEVATION, self.FINAL_AZIMUTH, 1)]:
            points = np.vstack((self.vertices.reshape(-1, 3), self.floor_vertices))
            bounds.append(self._camera(points, elevation, azimuth, lift)[0])
        bounds = np.vstack(bounds)
        extent = np.ptp(bounds[:, :2], axis=0)
        self.scale_factor = min(width / extent[0], height / extent[1])
        self.screen_offset = .5 * (bounds.min(axis=0) + bounds.max(axis=0))
        self.add(self.shadow, self.floor, self.floor_axes, self.surface, self.contours)
        self.refresh()
        self.add_updater(lambda m: m.refresh())

    def _camera(self, points, elevation, azimuth, lift):
        points = np.asarray(points, dtype=float)
        local = points - np.r_[self.best, 0]
        local = local * np.array([.80, .65, self.HEIGHT_SCALE * lift])
        a, e = np.deg2rad([azimuth, elevation])
        right = np.array([np.cos(a), np.sin(a), 0])
        depth = np.array([-np.sin(a), np.cos(a), 0])
        x = local @ right
        y = (local @ depth) * np.sin(e) + local[:, 2] * np.cos(e)
        toward_camera = -(local @ depth) * np.cos(e) + local[:, 2] * np.sin(e)
        perspective = 18 / (18 - toward_camera)
        return np.c_[x * perspective, y * perspective, np.zeros(len(x))], toward_camera

    def project(self, points):
        points = np.atleast_2d(points)
        projected, depth = self._camera(points, self.elevation.get_value(),
                                        self.azimuth.get_value(), self.lift.get_value())
        return self.screen_center + self.scale_factor * (projected - self.screen_offset), depth

    def point(self, w0, w1, energy=None):
        energy = float(linear_fit_error(w0, w1)) if energy is None else energy
        return self.project([[w0, w1, energy]])[0][0]

    def axis_end(self, index):
        return self.point(self.axis_ends[0], 0, 0) if index == 0 else self.point(0, self.axis_ends[1], 0)

    def refresh(self):
        state = tuple(t.get_value() for t in [self.lift, self.elevation, self.azimuth,
                                             self.surface_opacity])
        if state == self._last_state:
            return
        self._last_state = state
        lift, _, _, opacity = state
        floor, _ = self.project(self.floor_vertices)
        self.floor.set_points_as_corners(floor)
        self.shadow.set_points_as_corners(floor + .09 * DOWN + .06 * RIGHT)
        for axis, vertices in zip(self.floor_axes, self.axis_vertices):
            axis.put_start_and_end_on(*self.project(vertices)[0])
        for contour, vertices in zip(self.contours, self.contour_vertices):
            contour.set_points_as_corners(self.project(vertices)[0])
            contour.set_stroke(opacity=.65 - .22 * lift)

        # Opaque, lit faces are drawn far-to-near, using the same camera.
        centers = np.array([vertices.mean(axis=0) for _, vertices in self.quads])
        order = np.argsort(self.project(centers)[1])
        for face, vertices in self.quads:
            projected, _ = self.project(np.vstack((vertices, vertices[0])))
            face.set_points_as_corners(projected)
            local = vertices * np.array([.80, .65, self.HEIGHT_SCALE * lift])
            normal = np.cross(local[1] - local[0], local[2] - local[0])
            if normal[2] < 0:
                normal *= -1
            normal /= max(np.linalg.norm(normal), 1e-9)
            light = np.array([-.4, -.6, 1.])
            light /= np.linalg.norm(light)
            shade = .42 + .50 * max(0, float(normal @ light))
            height = (vertices[:, 2].mean() - self.minimum) / self.RIM_EXCESS
            pigment = interpolate_color(GREEN, PURPLE, height * .85)
            face.set_fill(interpolate_color(BLACK, pigment, shade), opacity)
            face.set_stroke(interpolate_color(pigment, WHITE, .12), opacity=.12 * opacity)
        self.surface.submobjects = [self.quads[i][0] for i in order]
