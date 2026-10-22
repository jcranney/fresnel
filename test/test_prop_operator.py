import numpy as np
import time
from fresnel import InplacePropagator

print(
    "| Version | `ang_spec` 2048 | `fft2` 2048 | `np.fft2` 2048 | `ang_spec` 256 | `fft2` 256 | `np.fft2` 256 |"
)
print("|  vx.x.x | ", end="")

for NSAMPLE in [2048, 256]:
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
    arr_orig = _arr.reshape([NSAMPLE, NSAMPLE])

    propagator = InplacePropagator((NSAMPLE, NSAMPLE))

    phi = 100 * np.exp(-((xx - 2e3) ** 2 + (yy - 3e3) ** 2) / 1e3**2)
    modifier = np.exp(-1j * phi).reshape([NSAMPLE, NSAMPLE])
    arr_orig *= modifier

    arr = arr_orig.copy()
    t1 = time.time()
    for _ in range(100):
        propagator.ang_spec(arr, DZ, WAVELENGTH, (SUPPORT_WIDTH, SUPPORT_WIDTH))
    t2 = time.time()

    print(f" {(t2-t1)/100:10.3e} |", end="")

    arr = arr_orig.copy()
    t1 = time.time()
    for _ in range(100):
        propagator.fft2(arr)
    t2 = time.time()

    print(f" {(t2-t1)/100:10.3e} |", end="")

    arr = arr_orig.copy()
    t1 = time.time()
    for _ in range(100):
        arr = np.fft.fft2(arr)
    t2 = time.time()

    print(f" {(t2-t1)/100:10.3e} |", end="")
print("")
