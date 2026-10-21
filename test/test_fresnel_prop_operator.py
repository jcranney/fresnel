import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation
from itertools import count
from fresnel import InplacePropagation

for NSAMPLE in [2**a for a in range(4, 12)]:
    # lets do everything in mm to get a reasonable numerical range
    SUPPORT_WIDTH: float = 20.0e3  # 200 mm
    BEAM_DIAM: float = 8.0e3  # 20 mm
    # FOCAL_LEN: float = 100.0  # f = 100 mm
    WAVELENGTH: float = 0.5e-3  # 500 nm
    DZ: float = 100e3  # 10 mm per step

    x = np.arange(NSAMPLE) * 1.0
    x -= x.mean()
    x *= SUPPORT_WIDTH / NSAMPLE
    y = x
    _xx, _yy = np.meshgrid(
        x,
        y,
        indexing="xy",
    )
    xx = _xx.flatten()
    yy = _yy.flatten()

    rr = (xx**2 + yy**2) ** 0.5

    _arr = ((rr < BEAM_DIAM / 2) * 1.0).astype(np.complex128)
    arr = _arr.reshape([NSAMPLE, NSAMPLE])

    propagator = InplacePropagation(
        WAVELENGTH, (SUPPORT_WIDTH, SUPPORT_WIDTH), (NSAMPLE, NSAMPLE)
    )

    phi = 100 * np.exp(-((xx - 2e3) ** 2 + (yy - 3e3) ** 2) / 1e3**2)
    modifier = np.exp(-1j * phi).reshape([NSAMPLE, NSAMPLE])
    arr *= modifier

    t1 = time.time()
    for _ in range(100):
        propagator.ang_spec_prop(arr, 1e6)
    t2 = time.time()

    print(f"NSAMPLE: {NSAMPLE}, {(t2-t1)/100:0.3e} sec per iteration")
        