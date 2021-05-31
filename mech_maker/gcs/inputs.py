from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from ..generics import MultiD
from .member import Member
from .constraint import StandardConstraint, FixedConstraint, RelativeConstraint

class MechanismInputParams:
    def __init__(self, start_params: tuple[MultiD, MultiD], end_params: tuple[MultiD, MultiD]) -> None:
        self._start_params = start_params
        self._end_params = end_params
        
    def params(self, ratio: float) -> tuple[MultiD, MultiD]:
        return (self._start_params[0].interp(self._end_params[0], ratio), self._start_params[1].interp(self._end_params[1], ratio))

class MechanismInput(ABC):
    @abstractmethod
    def __init__(self, constraint_type: Type[StandardConstraint], members: tuple[Member] | tuple[Member, Member], time_region: tuple[float, float], *params: MechanismInputParams) -> None:
        self._constraint_type = constraint_type
        self._members = members
        self._time_region = time_region
        self._params = params

    def constraint(self, time: float) -> StandardConstraint | None:
        if time < self._time_region[0] or time > self._time_region[1]:
            return None

        return self._constraint_type(*self._members, *[param.params((time - self._time_region[0]) / (self._time_region[1] - self._time_region[0])) for param in self._params])

class FixedMechanismInput(MechanismInput):
    def __init__(self, constraint_type: Type[FixedConstraint], member: Member, region: tuple[float, float], *params: tuple[tuple[MultiD, MultiD], tuple[MultiD, MultiD]]) -> None:
        super().__init__(constraint_type, (member,), region, *params)

class RelativeMechanismInput(MechanismInput):
    def __init__(self, constraint_type: Type[RelativeConstraint], member1: Member, member2: Member, region: tuple[float, float], *params: tuple[tuple[MultiD, MultiD], tuple[MultiD, MultiD]]) -> None:
        super().__init__(constraint_type, (member1, member2), region, *params)