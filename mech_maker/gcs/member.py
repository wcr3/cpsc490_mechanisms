from ..generics import Quaternion, Vec3
from ..shape import Shape

class Member:
    def __init__(self, location: Vec3, orientation: Quaternion, shape: Shape) -> None:
        self.location = location
        self.orientation = orientation
        self._shape = shape

    def relative_location(self, location: Vec3) -> Vec3:
        p = location.rotate(self.orientation)
        return p + self.location

    def relative_axis(self, axis: Vec3) -> Vec3:
        return axis.rotate(self.orientation)

    def relative_orientation(self, orientation: Quaternion) -> Quaternion:
        return self.orientation.quat_mult(orientation, True)

    def shape(self) -> Shape:
        return self._shape.transform(self.relative_location)