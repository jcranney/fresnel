from fresnel import Planner
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation
from itertools import count


NSAMPLE = 2**11

# lets do everything in mm to get a reasonable numerical range
SUPPORT_WIDTH: float = 10000  # 200 mm
BEAM_DIAM: float = 100.0  # 20 mm
FOCAL_LEN: float = 200000.0  # f = 100 mm
WAVELENGTH: float = 0.5e-3  # 500 nm
DZ: float = 100000  # 10 mm per step

_x = np.arange(NSAMPLE) * 1.0
_x -= _x.mean()
_x *= SUPPORT_WIDTH / NSAMPLE
_y = _x
_xx, _yy = np.meshgrid(
    _x,
    _y,
    indexing="xy",
)
xx = _xx.flatten()
yy = _yy.flatten()

rr = (xx**2 + yy**2) ** 0.5
# plt.matshow(rr.reshape((NSAMPLE,NSAMPLE)))
# plt.colorbar()
# plt.show()
# exit()

arr_np = ((rr < BEAM_DIAM/2) * 1.0) + 0j # .astype(np.complex128)
arr = arr_np.reshape([NSAMPLE, NSAMPLE])

# phase = np.exp(1j * 1 * rr**2).reshape(
#     [NSAMPLE, NSAMPLE]
# )
phase = (np.exp(-1j * 2 * np.pi / WAVELENGTH * rr**2 / (2 * FOCAL_LEN))).reshape(
    [NSAMPLE, NSAMPLE]
)
arr *= phase

planner = Planner(arr.shape)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(
    np.abs(arr),
    extent=(
        -SUPPORT_WIDTH / 2,
        SUPPORT_WIDTH / 2,
        -SUPPORT_WIDTH / 2,
        SUPPORT_WIDTH / 2,
    ),
)
ax.set_xlim((-BEAM_DIAM, BEAM_DIAM))
ax.set_ylim((-BEAM_DIAM, BEAM_DIAM))


class State:
    def __init__(self, arr: NDArray[np.complexfloating]) -> None:
        self.arr = arr
        self.z = 0.0


state = State(arr)


def update(i, state: State):
    state.z += DZ
    # if i == 0:
    #     state.arr *= phase
    planner.propagate_inplace(
        state.arr,
        DZ,
        WAVELENGTH,
        (SUPPORT_WIDTH, SUPPORT_WIDTH),
    )
    ax.images[0].set_data(np.abs(state.arr))
    ax.images[0].autoscale()
    ax.set_title(f"i = {i}, z = {state.z:0.2f} mm")
    return [ax]


anim = FuncAnimation(fig, lambda i: update(i, state), count(), interval=20, cache_frame_data=False)
plt.show()
