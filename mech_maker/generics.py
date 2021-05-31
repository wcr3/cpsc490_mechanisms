"""A module of basic geometric building blocks"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np

class MultiD(ABC):
    """Abstract Base for Multidimensional Objects (Vectors and Quaternions)"""

    def __init__(self, *coords: float) -> None:
        self._coords = list(coords)

    def np_array(self) -> np.ndarray:
        return np.array(self._coords)

    def normalized(self):
        return self/abs(self)

    def sq_magnitude(self) -> float:
        return sum([val * val for val in self])

    def magnitude(self) -> float:
        return np.sqrt(self.sq_magnitude())

    @abstractmethod
    def interp(self, other: MultiD, ratio: float) -> MultiD:
        pass

    def __eq__(self, other) -> bool:
        return all([val1 == val2 for (val1, val2) in zip(self, other)])

    def __add__(self, other):
        return type(self)(*[val1 + val2 for (val1, val2) in zip(self, other)])

    def __sub__(self, other):
        return type(self)(*[val1 - val2 for (val1, val2) in zip(self, other)])

    def __neg__(self):
        return type(self)(*[-val for val in self])

    def __mul__(self, other: float):
        return type(self)(*[val * other for val in self])

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def __truediv__(self, other: float):
        return type(self)(*[val / other for val in self])

    def __floordiv__(self, other: int):
        return type(self)(*[val // other for val in self])

    def __abs__(self) -> float:
        return self.magnitude()

    def __getitem__(self, key: int):
        return self._coords[key]

    def __setitem__(self, key: int, value: float):
        self._coords[key] = value

    def __iter__(self) -> Iterator[float]:
        return self._coords.__iter__()

    def __len__(self) -> int:
        return len(self._coords)

    @property
    @abstractmethod
    def _str_prefix(self) -> str:
        pass

    def __str__(self) -> str:
        ret = self._str_prefix + '<'
        for coord in self._coords:
            ret+= (str(coord) + ',')

        return ret[0:-1] + '>'

class Vec(MultiD):
    """Parent Class for Vectors"""

    def __init__(self, *coords: float) -> None:
        super().__init__(*coords)
    
    def dot(self, other) -> float:
        return sum([val1 * val2 for (val1, val2) in zip(self, other)])

    def angle_to(self, other) -> float:
        val = self.dot(other) / (abs(self) * abs(other))
        val = val if val <= 1 else 1
        return np.arccos( val )

    def distance_to(self, other: Vec3) -> float:
        return (other - self).magnitude()

    def interp(self, other: Vec, ratio: float):
        return (self * (1 - ratio)) + (other * ratio)

    @property
    def _str_prefix(self) -> str:
        return 'Vec' + str(len(self._coords))

class Vec2(Vec):
    """2D Vector"""

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)

    def cross(self, other: Vec2) -> float:
        return (self.x * other.y) - (self.y * other.x)

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, val: float) -> None:
        self[0] = val

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, val: float) -> None:
        self[1] = val

class Vec3(Vec):
    """3D Vector"""

    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x, y, z)

    def cross(self, other: Vec3) -> Vec3:
        x = (self.y * other.z) - (self.z * other.y)
        y = (self.z * other.x) - (self.x * other.z)
        z = (self.x * other.y) - (self.y * other.x)
        return Vec3(x, y, z)

    def rotate(self, q: Quaternion) -> Vec3:
        p = Quaternion.build(0, *self, False)
        p = q.quat_mult(p, False).quat_div(q, False)
        return Vec3(p.x, p.y, p.z)

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, val: float) -> None:
        self[0] = val

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, val: float) -> None:
        self[1] = val

    @property
    def z(self) -> float:
        return self[2]

    @z.setter
    def z(self, val: float) -> None:
        self[2] = val
    
class Quaternion(MultiD):
    """Quaternion"""

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        super().__init__(w, x, y, z)

    @staticmethod
    def build(w: float, x: float, y: float, z: float, is_rotation: bool) -> Quaternion:
        if is_rotation:
            return Quaternion(w,x,y,z).normalized()
        else:
            return Quaternion(w,x,y,z)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> Quaternion:
        if angle == 0:
            return Quaternion(1,0,0,0)

        w = np.cos(angle / 2)
        axis = axis.normalized()
        x = axis.x * np.sin(angle / 2)
        y = axis.y * np.sin(angle / 2)
        z = axis.z * np.sin(angle / 2)
        return Quaternion.build(w,x,y,z,True)

    def to_axis_angle(self) -> tuple[Vec3, float]:
        angle = self.angle()
        return self._axis_from_angle(angle), angle

    def angle(self) -> float:
        ang = 2 * np.arccos(self.w)
        return ang #if ang <= np.pi else ang - (np.pi * 2)

    def axis(self) -> Vec3:
        return self._axis_from_angle(self.angle())

    def _axis_from_angle(self, angle: float) -> Vec3:
        if angle == 0:
            return Vec3(0,0,0)
        
        return Vec3(self.x, self.y, self.z) / np.sin(angle / 2)

    def inverse(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def quat_mult(self, other: Quaternion, is_rotation: bool) -> Quaternion:
        w = (self.w * other.w) - (self.x * other.x) - (self.y * other.y) - (self.z * other.z)
        x = (self.w * other.x) + (self.x * other.w) + (self.y * other.z) - (self.z * other.y)
        y = (self.w * other.y) - (self.x * other.z) + (self.y * other.w) + (self.z * other.x)
        z = (self.w * other.z) + (self.x * other.y) - (self.y * other.x) + (self.z * other.w)
        return Quaternion.build(w, x, y, z, is_rotation)

    def quat_div(self, other: Quaternion, is_rotation: bool) -> Quaternion:
        return self.quat_mult(other.inverse(), is_rotation)

    def interp(self, other: Quaternion, ratio: float) -> Quaternion:
        diff = other.quat_div(self, True)
        axis, angle = diff.to_axis_angle()
        return Quaternion.from_axis_angle(axis, angle * ratio).quat_mult(self, True)

    @staticmethod
    def identity() -> Quaternion:
        return Quaternion(1,0,0,0)

    @property
    def w(self) -> float:
        return self[0]

    @w.setter
    def w(self, val: float) -> None:
        self[0] = val

    @property
    def x(self) -> float:
        return self[1]

    @x.setter
    def x(self, val: float) -> None:
        self[1] = val

    @property
    def y(self) -> float:
        return self[2]

    @y.setter
    def y(self, val: float) -> None:
        self[2] = val

    @property
    def z(self) -> float:
        return self[3]

    @z.setter
    def z(self, val: float) -> None:
        self[3] = val

    @property
    def _str_prefix(self) -> str:
        return 'Quat'
