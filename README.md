# Fresnel propagation

This is a narrowly scoped python package for efficient Fresnel Propagation, with
a rust backend.

Install with:

```bash
pip install fresnel-prop
```

Use with:

```python
from fresnel import InplacePropagator

# prepare the inplace propagator based on array dimensions
prop = InplacePropagator(arr.shape)

# propagate the beam with wavelength=`wavelength`,
# and physical size `(beam_width_m, beam_height_m)`
# modelled by the complex array `arr`
# by `dz` [metres] along the z axis.
prop.ang_spec(arr, dz, wavelength, (beam_width_m, beam_height_m))
```

## Profiling on my Dell Laptop

- `fft2` is the fft2 method provided by the `InplacePropagator` object (this package),
- `np.fft2` is the defacto python `fft2` method,
- `ang_spec` is the angular spectrum propagation method, which involves 2 `fft2`s
  and a number of element-wise operations.
- `fft2` 2048 (for example) is the time-per-iteration (in seconds) of an `fft2` operation performed on a $2048\times2048$ `np.complex128` array.

| Version | `ang_spec` 2048 | `fft2` 2048 | `np.fft2` 2048 | `ang_spec` 256 | `fft2` 256 | `np.fft2` 256 |
| ------- | --------------- | ----------- | -------------- | -------------- | ---------- | ------------- |
| v0.1.2  | 1.408e-01       | 3.972e-02   | 9.831e-02      | 1.825e-03      | 3.574e-04  | 3.922e-04     |
| v0.1.3  | 1.302e-01       | 3.549e-02   | 9.887e-02      | 1.867e-03      | 2.870e-04  | 5.342e-04     |
