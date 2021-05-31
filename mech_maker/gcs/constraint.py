from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..generics import MultiD, Quaternion, Vec3
from .member import Member

def _location_eval(location1: Vec3, location2: Vec3) -> float:
    v_dif = location1 - location2
    return v_dif.sq_magnitude()

def _orientation_eval(orientation1: Quaternion, orientation2: Quaternion) -> float:
    ang_dif = orientation2.quat_div(orientation1, True).angle()
    return ang_dif * ang_dif

def _axis_eval(axis1: Vec3, axis2: Vec3) -> float:
    ang_dif = axis1.angle_to(axis2)
    return ang_dif * ang_dif

class Constraint(ABC):
    @abstractmethod
    def eval(self) -> float:
        pass

class StandardConstraint(Constraint):
    def __init__(self, members: tuple[Member] | tuple[Member, Member], *params: tuple[MultiD, MultiD]) -> None:
        self._members = members
        self._params = params

class GroupConstraint(Constraint):
    @abstractmethod
    def __init__(self, *sub_constraints: Constraint) -> None:
        self._sub_constraints = sub_constraints

    def eval(self) -> float:
        return sum([constraint.eval() for constraint in self._sub_constraints])

class RelativeConstraint(StandardConstraint):
    def __init__(self, member1: Member, member2: Member, *params: tuple[MultiD, MultiD]) -> None:
        super().__init__((member1, member2), *params)

class RelativeLocationConstraint(RelativeConstraint):
    def __init__(self, member1: Member, member2: Member, locations: tuple[Vec3, Vec3]) -> None:
        super().__init__(member1, member2, locations)

    @property
    def _locations(self) -> tuple[Vec3, Vec3]:
        return self._params[0]

    def eval(self) -> float:
        return _location_eval(self._members[0].relative_location(self._locations[0]), self._members[1].relative_location(self._locations[1]))

class RelativeOrientationConstraint(RelativeConstraint):
    def __init__(self, member1: Member, member2: Member, orientations: tuple[Quaternion, Quaternion]) -> None:
        super().__init__(member1, member2, orientations)

    @property
    def _orientations(self) -> tuple[Quaternion, Quaternion]:
        return self._params[0]

    def eval(self) -> float:
        return _orientation_eval(self._members[0].relative_orientation(self._orientations[0]), self._members[1].relative_orientation(self._orientations[1]))

class RelativeAxisAlignedConstraint(RelativeConstraint):
    def __init__(self, member1: Member, member2: Member, axes: tuple[Vec3, Vec3]) -> None:
        super().__init__(member1, member2, tuple(axis.normalized() for axis in axes))

    @property
    def _axes(self) -> tuple[Vec3, Vec3]:
        return self._params[0]

    def eval(self) -> float:
        return _axis_eval(self._members[0].relative_axis(self._axes[0]), self._members[1].relative_axis(self._axes[1]))

class RelativePinConstraint(GroupConstraint, RelativeConstraint):
    def __init__(self, member1: Member, member2: Member, locations: tuple[Vec3, Vec3], axes: tuple[Vec3, Vec3]) -> None:
        location_constraint = RelativeLocationConstraint(member1, member2, locations)
        orientation_constraint = RelativeAxisAlignedConstraint(member1, member2, axes)
        GroupConstraint.__init__(self, location_constraint, orientation_constraint)

class FixedConstraint(StandardConstraint):
    def __init__(self, member: Member, *params: tuple[MultiD, MultiD]) -> None:
        super().__init__((member,), *params)

    @property
    def _member(self) -> Member:
        return self._members[0]

    @property
    def _local(self) -> MultiD:
        return self._params[0][0]

    @property
    def _global(self) -> MultiD:
        return self._params[0][1]

class FixedLocationConstraint(FixedConstraint):
    def __init__(self, member: Member, locations: tuple[Vec3, Vec3]) -> None:
        super().__init__(member, locations)

    def eval(self) -> float:
        return _location_eval(self._member.relative_location(self._local), self._global)

class FixedOrientationConstraint(FixedConstraint):
    def __init__(self, member: Member, orientations: tuple[Quaternion, Quaternion]) -> None:
        super().__init__(member, orientations)

    def eval(self) -> float:
        return _orientation_eval(self._member.relative_orientation(self._local), self._global)

class FixedAxisAlignedConstraint(FixedConstraint):
    def __init__(self, member: Member, axes: tuple[Vec3, Vec3]) -> None:
        super().__init__(member, axes)

    def eval(self) -> float:
        return _axis_eval(self._member.relative_axis(self._local), self._global)

class FixedPinConstraint(GroupConstraint, FixedConstraint):
    def __init__(self, member: Member, locations: tuple[Vec3, Vec3], axes: tuple[Vec3, Vec3]) -> None:
        location_constraint = FixedLocationConstraint(member, locations)
        orientation_constraint = FixedAxisAlignedConstraint(member, axes)
        GroupConstraint.__init__(self, location_constraint, orientation_constraint)

class FixedAllConstraint(GroupConstraint, FixedConstraint):
    def __init__(self, member: Member, locations: tuple[Vec3, Vec3], orientations: tuple[Quaternion, Quaternion]) -> None:
        location_constraint = FixedLocationConstraint(member, locations)
        orientation_constraint = FixedOrientationConstraint(member, orientations)
        GroupConstraint.__init__(self, location_constraint, orientation_constraint)

class OnPlaneConstraint(Constraint):
    def __init__(self, member: Member, local_location: Vec3, plane_eq: Callable[[Vec3], float]) -> None:
        self._member = member
        self._local_location = local_location
        self._plane_eq = plane_eq

    def eval(self) -> float:
        plane_dif = self._plane_eq(self._member.relative_location(self._local_location))
        return plane_dif * plane_dif