"""
A Python module implemented in Rust.
"""

from _typeshed import Incomplete
from typing import final

@final
class InplacePropagator:
    """
    The `InplacePropagator` object is the main python interface, exposing an
    efficient `fft2`, `ifft2`, and `ang_spec_prop` method acting inplace on
    numpy complex128 arrays.
    """
    def __new__(cls, /, shape: tuple[int, int]) -> InplacePropagator: ...
    def ang_spec_prop(self, /, array: Incomplete, dz: float, wavelength: float, wavesize_xy: tuple[float, float]) -> None: ...
    def fft2(self, /, array: Incomplete) -> None: ...
    def ifft2(self, /, array: Incomplete) -> None: ...
