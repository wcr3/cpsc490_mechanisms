from ..generics import Vec3
from ..curve import CurvePoint, Curve

from .member import Member

class TrackPoint():
    def __init__(self, member: Member, location: Vec3) -> None:
        self._member = member
        self._location = location

    def position(self) -> Vec3:
        return self._member.relative_location(self._location)

class MechanismOutput:
    def __init__(self, track_point: TrackPoint) -> None:
        self._track_point = track_point
        self._points: list[tuple[float, Vec3]] = []

    def _update_neightbor_velocities(index: int) -> None:
        pass

    def _velocity_at(self, index: int, first_last_diff: float) -> Vec3:
        if index == 0:
            if len(self._points) != 1: # forward difference
                if first_last_diff > 0.01:
                    return (self._points[0][1] - self._points[1][1]) / (self._points[0][0] - self._points[1][0])
                else: # cyclic
                    t0 = self._points[1][0]
                    t1 = self._points[0][0]
                    t2 = self._points[0][0] + (self._points[-2][0] - self._points[-1][0])
                    p0 = self._points[1][1]
                    p1 = self._points[0][1]
                    p2 = self._points[-2][1]
            else:
                return Vec3(0,0,0)
        elif index == len(self._points) - 1:
            if first_last_diff > 0.01: # backward difference
                return (self._points[-2][1] - self._points[-1][1]) / (self._points[-2][0] - self._points[-1][0])
            else: # cylic
                t0 = self._points[-1][0] + (self._points[1][0] - self._points[0][0])
                t1 = self._points[-1][0]
                t2 = self._points[-2][0]
                p0 = self._points[1][1]
                p1 = self._points[-1][1]
                p2 = self._points[-2][1]
        else:
            t0 = self._points[index + 1][0]
            t1 = self._points[index][0]
            t2 = self._points[index - 1][0]
            p0 = self._points[index + 1][1]
            p1 = self._points[index][1]
            p2 = self._points[index - 1][1]
                        
        # parabolic interpolation
        p0 = ((t1 - t2) / ((t0 - t1) * (t0 - t2))) * p0
        p1 = (((t1 - t2) + (t1 - t0)) / ((t1 - t2) * (t1 - t0))) * p1
        p2 = ((t1 - t0) / ((t2 - t0) * (t2 - t1))) * p2
        return p0 + p1 + p2

    def apply_time(self, time: float) -> None:
        index = 0
        for point in self._points:
            if point[0] < time:
                break

            index = index + 1

        self._points.insert(index, (time, self._track_point.position()))

    def reset(self) -> None:
        self._points = []

    def curve(self) -> Curve:
        first_last_diff = (self._points[0][1] - self._points[-1][1]).magnitude()
        return Curve([CurvePoint(point[1], self._velocity_at(index, first_last_diff)) for index, point in reversed(list(enumerate(self._points)))])
