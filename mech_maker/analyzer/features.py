from __future__ import annotations

from typing import Generator
import numpy as np
from scipy.spatial.transform import Rotation as rot
from sklearn.decomposition import PCA

from ..generics import Quaternion, Vec2, Vec3
from ..curve import CurvePoint, Curve

def _find_next_point(current_point: CurvePoint, start_index: int, points: list[CurvePoint], length: float) -> tuple[CurvePoint, int]:
    index = start_index

    current_dist = 0.0
    prev_point = current_point
    check_dist = (points[index].location - prev_point.location).magnitude()
    while current_dist + check_dist < length - 1e-8:
        current_dist = current_dist + check_dist
        index = index + 1
        prev_point = points[index - 1]
        check_dist = (points[index].location - prev_point.location).magnitude()

    ratio = (length - 1e-8 - current_dist) / check_dist
    next_pt = CurvePoint(prev_point.location.interp(points[index].location, ratio), prev_point.velocity.interp(points[index].velocity, ratio))
    return next_pt, index - 1

def _resample_curve(curve: Curve, num_samples: int) -> tuple[Curve, Vec3]:
    total_length = curve.total_length()
    segment_length = total_length / (num_samples - 1)

    points = list(curve.points())

    index = 0
    current_point = points[index]
    result = [current_point]
    avg = current_point.location
    for _ in range(num_samples - 1):
        next_point, index = _find_next_point(current_point, index + 1, points, segment_length)
        avg = avg + next_point.location
        result.append(next_point)
        current_point = next_point

    avg = avg / num_samples 

    return Curve(result), avg

class CurveFeature:
    def __init__(self, curve: Curve, num_samples: int) -> None:
        self._orig_curve = curve
        self._sampled_curve, self._avg_pos = _resample_curve(curve, num_samples)
        self._curve_translated = self._sampled_curve.transform(lambda pt: CurvePoint(pt.location - self._avg_pos, pt.velocity))
        pca = PCA(3)
        self._pca = pca.fit(np.array([v.location.np_array() for v in self._curve_translated.points()]))
        self._axes = pca.components_
        r = rot.from_matrix(self._axes.transpose())
        x, y, z, w = r.as_quat()
        self._orientation = Quaternion.build(w, x, y, z, True)
        self._curve_rotated = self._curve_translated.transform(lambda pt: CurvePoint(pt.location.rotate(self._orientation), pt.velocity.rotate(self._orientation)))
        self._l_x = max([pt.location.x for pt in self._curve_rotated.points()]) - min([pt.location.x for pt in self._curve_rotated.points()])
        self._l_y = max([pt.location.y for pt in self._curve_rotated.points()]) - min([pt.location.y for pt in self._curve_rotated.points()])
        self._l_z = max([pt.location.z for pt in self._curve_rotated.points()]) - min([pt.location.z for pt in self._curve_rotated.points()])
        self._curve_scaled = self._curve_rotated.transform(lambda pt: CurvePoint(pt.location * (1/self._l_x), pt.velocity * (1/self._l_x)))
        self.features = []
        self._calc_features()

    def _calc_plane_ellipticity(self, vec1: Vec3, vec2: Vec3) -> float:
        locs = [Vec2(pt.location.dot(vec1), pt.location.dot(vec2)) for pt in self._curve_scaled.points()]
        l_max = max([loc.x for loc in locs]) - min([loc.x for loc in locs])
        l_min = max([loc.y for loc in locs]) - min([loc.y for loc in locs])
        return l_min / l_max

    def _calc_features(self) -> None:
        f = []
        f.append(self._curve_scaled.total_length())
        f.extend([self._calc_plane_ellipticity(v1, v2) for v1, v2 in [(Vec3(1,0,0), Vec3(0,1,0)), (Vec3(0,1,0), Vec3(0,0,1)), (Vec3(1,0,0), Vec3(0,0,1))]])
        f.append(self._avg_pos.magnitude())
        f.append(self._orientation.angle())
        self.features = f

    def axes(self) -> tuple[Vec3, Vec3, Vec3]:
        return (Vec3(*self._axes[0]), Vec3(*self._axes[1]), Vec3(*self._axes[2]))

    def curves(self) -> Generator[Curve, None, None]:
        yield self._curve_translated
        yield Curve([CurvePoint(Vec3(0,0,0), axis) for axis in self.axes()])

    def compare(self, other: CurveFeature) -> float:
        return sum((f1 - f2) * (f1 - f2) for f1, f2 in zip(self.features, other.features))