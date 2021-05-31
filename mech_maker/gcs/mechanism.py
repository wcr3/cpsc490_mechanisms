from __future__ import annotations

from collections import defaultdict
from typing import Callable, Generator

from ..generics import Vec3
from ..shape import Shape
from ..curve import Curve

from .state import MechanismState
from .member import Member
from .constraint import Constraint, OnPlaneConstraint, FixedAxisAlignedConstraint
from .inputs import MechanismInput
from .outputs import MechanismOutput, TrackPoint
from .solver import Solver

class Mechanism:
    def __init__(self, solver: Solver) -> None:
        self._solver = solver
        self._state = MechanismState()
        self._time = 0.0
        self._constraints: list[Constraint] = []
        self._solved_states: defaultdict[float, list[float] | None] = defaultdict(lambda: None)
        self._inputs: list[MechanismInput] = []
        self._outputs: list[MechanismOutput] = []

    def _reset_solutions(self) -> None:
        self._solved_states = defaultdict(lambda: None)
        for output in self._outputs:
            output.reset()

    def add_member(self, member: Member) -> None:
        self._state.add_member(member)
        self._reset_solutions()

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)
        self._reset_solutions()

    def add_track_point(self, point: TrackPoint) -> None:
        self._outputs.append(MechanismOutput(point))
        for time, state in self._solved_states.items():
            self._state.update_from_raw_values(state)
            self._outputs[-1].apply_time(time)

        if len(self._solved_states):
            self._state.update_from_raw_values(self._solved_states[self._time])

    def add_input(self, input: MechanismInput) -> None:
        self._inputs.append(input)
        self._reset_solutions()

    def set_solver(self, solver: Solver) -> None:
        self._solver = solver

    def shapes(self) -> Generator[Shape, None, None]:
        return self._state.shapes()

    def curves(self) -> Generator[Curve, None, None]:
        return (output.curve() for output in self._outputs)

    def set_time(self, time: float) -> bool:
        if (self._solved_states[time] is None):
            return self._solve_time(time)
        else:
            self._time = time
            self._state.update_from_raw_values(self._solved_states[time])
            return True

    def _solve_time(self, time: float) -> bool:
        ret = False
        if (self._solved_states[time] is not None):
            ret = True
        else:
            cons = []
            cons.extend(self._constraints)
            for input in self._inputs:
                con = input.constraint(time)
                if con is not None:
                    cons.append(con)

            if (self._solver.solve(self._state, cons)):
                ret = True
                self._time = time
                self._solved_states[time] = self._state.to_raw_values()
                for output in self._outputs:
                    output.apply_time(time)

        return ret
        
    def solve_times(self, times: list[float], callback: Callable[[bool, Generator[Shape, None, None], Generator[Curve, None, None]], None] | None) -> list[bool]:
        res = []
        for time in times:
            res.append(self._solve_time(time))
            if callback is not None:
                callback(res[-1], self.shapes(), self.curves())

        return res

class Mechanism2D(Mechanism):
    def __init__(self, solver: Solver, z: float =0) -> None:
        self._z = z
        super().__init__(solver)

    def add_member(self, member: Member):
        self.add_constraint(OnPlaneConstraint(member, Vec3(0,0,0), lambda vec : vec.z - self._z))
        self.add_constraint(FixedAxisAlignedConstraint(member, (Vec3(0,0,1), Vec3(0,0,1))))
        super().add_member(member)
