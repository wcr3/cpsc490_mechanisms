import numpy as np

class PinConstraint:
    def __init__(self, link1, link2, point1, point2):
        self.link1 = link1
        self.link2 = link2
        self.point1 = point1
        self.point2 = point2

    def func(self, state):
        x1 = state[self.link1 * 4] + (self.point1 * np.cos(state[self.link1 * 4 + 2]) * state[self.link1 * 4 + 3])
        y1 = state[self.link1 * 4 + 1] + (self.point1 * np.sin(state[self.link1 * 4 + 2]) * state[self.link1 * 4 + 3])

        x2 = state[self.link2 * 4] + (self.point2 * np.cos(state[self.link2 * 4 + 2]) * state[self.link2 * 4 + 3])
        y2 = state[self.link2 * 4 + 1] + (self.point2 * np.sin(state[self.link2 * 4 + 2]) * state[self.link2 * 4 + 3])

        return np.power(x1 - x2, 2) + np.power(y1 - y2, 2)

class FixedConstraint:
    def __init__(self, link, rot, loc):
        self.link = link
        self.rot = rot
        self.loc = loc

    def func(self, state):
        return np.power(state[self.link * 4] - self.loc[0], 2) + np.power(state[self.link * 4 + 1] - self.loc[1], 2) + np.power(state[self.link * 4 + 2] - self.rot, 2)

class LengthConstraint:
    def __init__(self, link, length):
        self.link = link
        self.length = length

    def func(self, state):
        return np.power(state[self.link * 4 + 3] - self.length, 2)