from fresnel import Planner
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from itertools import count

NSAMPLE = 2**10
_xx, _yy = np.meshgrid(
    np.arange(NSAMPLE) * 1.0,
    np.arange(NSAMPLE) * 1.0,
    indexing="xy",
)
xx = _xx.flatten()
yy = _yy.flatten()
xx /= NSAMPLE
yy /= NSAMPLE
xx -= xx.mean()
yy -= yy.mean()
rr = (xx**2 + yy**2) ** 0.5
arr_np = (rr < 0.02) * 1.0 + 0.0j
arr = arr_np.reshape([NSAMPLE, NSAMPLE])
planner = Planner(arr.shape)
fig, ax = plt.subplots(1, 1)
ax.imshow(np.abs(arr))
z = 0.0

phase = (np.exp(-1j * 1000*(-xx + rr**2))).reshape([NSAMPLE, NSAMPLE])


class State:
    z = 0.0

    def __init__(self, arr) -> None:
        self.arr = arr


state = State(arr)


def update(i, state):
    # units just need to be consistent, here I use microns:
    dz = 1e3
    state.z += dz
    if i == 1:
        state.arr *= phase
    planner.propagate_inplace(state.arr, dz, 0.5, (25e3, 25e3))
    ax.images[0].set_data(np.abs(state.arr))
    ax.images[0].autoscale()
    ax.set_title(f"i = {i}, z = {state.z/1e3:0.2f} mm")
    return [ax]


anim = FuncAnimation(fig, lambda i: update(i, state), count(), interval=20)
plt.show()
