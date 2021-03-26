import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from link import Link2D
from constraint import PinConstraint, FixedConstraint, LengthConstraint

def state_flatten(links):
    ret = []
    for link in links:
        ret.append(link.loc[0])
        ret.append(link.loc[1])
        ret.append(link.rot)
        ret.append(link.length)

    return ret

def state_rebuild(state):
    i = 0
    links = []
    for i in range(len(state) // 4):
        links.append(Link2D(state[i*4 + 3], state[i*4 + 2], [state[i*4], state[i*4 + 1]]))

    return links

def opt_func(state, constraints):
    ret = 0
    for constraint in constraints:
        ret = ret + constraint.func(state)

    return ret

if __name__ == "__main__":
    links = [Link2D(1, 0, [0,0]), Link2D(1, np.pi/2, [0,0]), Link2D(1, 0, [0,1.1]), Link2D(1.5, np.pi/2, [1.1,0])]
    constraints = [PinConstraint(0, 1, 0, 0), PinConstraint(1, 2, 1, 0), PinConstraint(2, 3, 0.5, 1), PinConstraint(0, 3, 1, 0), FixedConstraint(0, 0, [0,0]), FixedConstraint(1, np.pi/2, [0,0]), LengthConstraint(0, 1), LengthConstraint(1, 1), LengthConstraint(2, 1), LengthConstraint(3, 1.5)]

    track_x = []
    track_y = []

    fig, axs = plt.subplots(2)
    writer = anim.FFMpegWriter()
    fig.show()

    with writer.saving(fig, '..\\Images\\track.mp4', 100):
        for i in range(0, 100):
            links[1].rot = (2 * np.pi) * (i / 100) + (np.pi/2)
            constraints[5].rot = (2 * np.pi) * (i / 100) + (np.pi/2)

            for link in links:
                axs[0].plot([link.loc[0], link.loc[0] + (link.length * np.cos(link.rot))], [link.loc[1], link.loc[1] + (link.length * np.sin(link.rot))])

            state = state_flatten(links)
            tmp = op.fmin(opt_func, state, args=(constraints,), maxfun=10000)
            links = state_rebuild(tmp)

            track_x.append(links[2].loc[0] + (np.cos(links[2].rot) * links[2].length))
            track_y.append(links[2].loc[1] + (np.sin(links[2].rot) * links[2].length))

            for link in links:
                axs[1].plot([link.loc[0], link.loc[0] + (link.length * np.cos(link.rot))], [link.loc[1], link.loc[1] + (link.length * np.sin(link.rot))])

            axs[1].plot(track_x, track_y)

            axs[0].set_xlim(left=-1.25, right=2.25)
            axs[0].set_ylim(bottom=-1.75, top=1.75)
            axs[1].set_xlim(left=-1.25, right=2.25)
            axs[1].set_ylim(bottom=-1.75, top=1.75)
            fig.canvas.draw()
            writer.grab_frame()
            axs[0].clear()
            axs[1].clear()

