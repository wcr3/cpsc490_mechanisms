from __future__ import annotations

from typing import Generator, Callable
from .generics import Vec3

class Shape:
    def __init__(self, points: list[Vec3]) -> None:
        self._points = points

    def add_point(self, point: Vec3) -> None:
        self._points.append(point)

    def points(self) -> Generator[Vec3, None, None]:
        return (point for point in self._points)

    def transform(self, callback: Callable[[Vec3], Vec3]) -> Shape:
        return Shape([callback(point) for point in self._points])

class Line(Shape):
    def __init__(self, len: float) -> None:
        super().__init__([Vec3(0,0,0), Vec3(len,0,0)])