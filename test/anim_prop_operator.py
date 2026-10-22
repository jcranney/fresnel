import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation
from itertools import count
from fresnel import InplacePropagator

NSAMPLE = 2**9

# lets do everything in mm to get a reasonable numerical range
SUPPORT_WIDTH: float = 20.0e3  # 20 m
BEAM_DIAM: float = 8.0e3  # 8 m
WAVELENGTH: float = 0.5e-3  # 500 nm
DZ: float = 1000e3  # 1km per step

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

_arr = ((rr < BEAM_DIAM / 2) * 1.0).astype(np.complex128)
arr = _arr.reshape([NSAMPLE, NSAMPLE])

propagator = InplacePropagator((NSAMPLE, NSAMPLE))

phi = 100 * np.exp(-((xx - 2e3) ** 2 + (yy - 3e3) ** 2) / 2e3**2)
modifier = np.exp(-1j * phi).reshape([NSAMPLE, NSAMPLE])
arr *= modifier


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
        self.propagator = propagator
        self.arr = arr
        self.z = 0.0


state = State(arr)


def update(i, state: State):
    state.z += DZ
    state.propagator.ang_spec(
        state.arr,
        DZ,
        WAVELENGTH,
        (SUPPORT_WIDTH, SUPPORT_WIDTH)
    )
    ax.images[0].set_data(np.abs(state.arr))
    ax.images[0].autoscale()
    ax.set_title(f"i = {i}, z = {state.z/1e6:0.2f} km")
    return [ax]


anim = FuncAnimation(
    fig, lambda i: update(i, state), count(), interval=20, cache_frame_data=False
)
plt.show()
