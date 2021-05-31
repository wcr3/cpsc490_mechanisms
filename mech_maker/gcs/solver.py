from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable
from scipy import optimize as opt

from ..shape import Shape

from .state import MechanismState
from .constraint import Constraint

class Solver(ABC):
    """Abstract Base Class for Various Constraint Solving Methods"""
    @abstractmethod
    def solve(self, state: MechanismState, constraints: list[Constraint]) -> bool:
        pass

class ScipySLSQPSolver(Solver):
    """Constraint Solver using scipy's SLSQP implementation"""
    def __init__(self, iter_callback: Callable[[Iterable[Shape]], None] | None =None) -> None:
        self._iter_callback: Callable[[list[float]], None] | None = None if iter_callback is None else lambda x: iter_callback(self._state.shapes())
        self._state: MechanismState | None = None
        self._constraints: list[Constraint] | None = None

    def _op_func(self, inp: list[float]) -> float:
        self._state.update_from_raw_values(inp)
        return sum([constraint.eval() for constraint in self._constraints])

    def solve(self, state: MechanismState, constraints: list[Constraint]) -> bool:
        self._state = state
        self._constraints = constraints
        res = opt.minimize(self._op_func, state.to_raw_values(), method='SLSQP', jac='2-point', callback=self._iter_callback)
        return res.success
