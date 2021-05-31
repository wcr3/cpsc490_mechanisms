from typing import Generator

from ..generics import Vec3, Quaternion
from ..shape import Shape
from .member import Member


class MechanismState:
    def __init__(self) -> None:
        self._members: list[Member] = []
        self._member_map: dict[int, list[int]] = {}

    def add_member(self, member: Member) -> None:
        self._members.append(member)

    def to_raw_values(self) -> list[float]:
        state = []
        for member in self._members:
            self._member_map[id(member)] = [len(state)]
            state.extend(member.location)
            self._member_map[id(member)].append(len(state))
            state.extend(member.orientation) # test
            self._member_map[id(member)].append(len(state)) # test
            # axis, angle = member.orientation.to_axis_angle()
            # state.extend(axis)
            # self._member_map[id(member)].append(len(state))
            # state.append(angle)
            # self._member_map[id(member)].append(len(state))

        return state

    def update_from_raw_values(self, vals: list[float]) -> None:
        for member in self._members:
            regions = self._member_map[id(member)]
            member.location = Vec3(*vals[regions[0]:regions[1]])
            member.orientation = Quaternion.build(*vals[regions[1]:regions[2]], True)
            # axis = Vec3(*vals[regions[1]:regions[2]])
            # angle = float(*vals[regions[2]:regions[3]])
            # member.orientation = Quaternion.from_axis_angle(axis, angle)

    def shapes(self) -> Generator[Shape, None, None]:
        return (member.shape() for member in self._members)

