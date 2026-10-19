from fresnel import Planner
import numpy as np
import matplotlib.pyplot as plt
import time

print(f"{'NSAMPLE':10} | {'t (rust)':10} | {'t (numpy)':10}")
print("-" * 40)
for NSAMPLE in [2**n for n in range(1, 12)]:
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
    arr_np = (rr < 0.1) * 1.0 + 0.0j
    arr_sq = arr_np.reshape([NSAMPLE, NSAMPLE])
    planner = Planner(arr_sq.shape)

    t1 = time.time()
    output = arr_sq.copy()
    # units just need to be consistent, here i use microns:
    planner.propagate_inplace(output, 1e0, 0.5, (25e3, 25e3))
    t2 = time.time()
    _ = np.fft.fft2(arr_sq)
    t3 = time.time()
    print(f"{NSAMPLE:10} | {t2 - t1:10.3e} | {t3 - t2:10.3e}")
    if NSAMPLE == 1024:
        plt.matshow(np.abs(output))
        plt.show()
