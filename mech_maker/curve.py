from __future__ import annotations

from typing import Generator, Callable

from .generics import Vec3

class CurvePoint:
    def __init__(self, location: Vec3, velocity: Vec3) -> None:
        self.location = location
        self.velocity = velocity

class Curve:
    def __init__(self, points: list[CurvePoint]) -> None:
        self._points = points
        self._total_length = -1.0

    def points(self) -> Generator[CurvePoint, None, None]:
        return (point for point in self._points)

    def total_length(self) -> float:
        if self._total_length > 0:
            return self._total_length

        self._total_length = sum([(v2.location - v1.location).magnitude() for v1, v2 in zip(self._points, self._points[1:])])
        return self._total_length

    def transform(self, callback: Callable[[CurvePoint], CurvePoint]) -> Curve:
        return Curve([callback(point) for point in self._points])