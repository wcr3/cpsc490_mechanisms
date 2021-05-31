from abc import ABC, abstractmethod
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from ..generics import Vec3, Vec2
from ..shape import Shape
from ..curve import Curve

class Animator(ABC):
    @abstractmethod
    def write_frame(self, shapes: Iterable[Shape], curves: Iterable[Curve]) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

class Animator2D(Animator):
    def __init__(self, normal: Vec3) -> None:
        normal = normal.normalized()
        self._x_axis = Vec3(0,1,0).cross(normal)
        self._y_axis = normal.cross(self._x_axis)

    def _project_to_plane(self, point: Vec3) -> Vec2:
        return Vec2(point.dot(self._x_axis), point.dot(self._y_axis))

    def _axes_on_plane(self) -> tuple[Vec2, Vec2, Vec2]:
        return (self._project_to_plane(Vec3(1,0,0)), self._project_to_plane(Vec3(0,1,0)), self._project_to_plane(Vec3(0,0,1)))

class MPEGAnimator(Animator2D):
    def __init__(self, normal: Vec3, outfile: str, fps: int, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        self._xlim = xlim
        self._ylim = ylim

        self._fig, self._axs = plt.subplots(1)

        self._writer = anim.FFMpegWriter(fps=fps)
        self._writer.setup(self._fig, outfile, 100)
        super().__init__(normal)

    def write_frame(self, shapes: Iterable[Shape], curves: Iterable[Curve]) -> None:
        for shape in shapes:
            x = []
            y = []
            for point in shape.points():
                pt = self._project_to_plane(point)
                x.append(pt.x)
                y.append(pt.y)
            
            self._axs.plot(x, y)

        for curve in curves:
            x = []
            y = []
            for point in curve.points():
                pt = self._project_to_plane(point.location)
                x.append(pt.x)
                y.append(pt.y)

            self._axs.plot(x, y)

            x = []
            y = []
            u = []
            v = []
            for point in curve.points():
                if point.velocity.x != 0 or point.velocity.y != 0:
                    pt = self._project_to_plane(point.location)
                    vel = self._project_to_plane(point.velocity)
                    x.append(pt.x)
                    y.append(pt.y)
                    u.append(vel.x)
                    v.append(vel.y)

            if len(x) > 0:
                self._axs.quiver(x, y, u, v, width=self._axs.viewLim.width * 0.0005)

        x = []
        y = []
        u = []
        v = []
        for ax in self._axes_on_plane():
            x.append(0)
            y.append(0)
            u.append(ax.x)
            v.append(ax.y)

        self._axs.quiver(x, y, u, v, width=self._axs.viewLim.width * 0.0005)

        self._axs.set_xlim(self._xlim)
        self._axs.set_ylim(self._ylim)
        self._fig.canvas.draw()
        self._writer.grab_frame()
        self._axs.clear()

    def finish(self) -> None:
        self._writer.finish()
