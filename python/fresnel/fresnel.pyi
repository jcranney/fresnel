"""
A Python module implemented in Rust.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import final

@final
class Fft2Planner:
    @staticmethod
    def forward_planner(nrows: int, ncols: int) -> Fft2Planner: ...
    @staticmethod
    def inverse_planner(nrows: int, ncols: int) -> Fft2Planner: ...
    def ncols(self, /) -> int: ...
    def nrows(self, /) -> int: ...
    def process_numpy_inplace(self, /, array: Incomplete) -> None: ...

@final
class InplacePropagation:
    """
    The `InplacePropagation` object aims to implenent the ang_spec_prop
    algorithm of Schmidt's (2010) book, Listing 6.5, in an efficient manner.
    """
    def __new__(cls, /, wavelength: float, wavesize_xy: tuple[float, float], shape: tuple[int, int]) -> InplacePropagation: ...
    def ang_spec_prop(self, /, array: Incomplete, d: float) -> None: ...

def fresnel_transfer_inplace(array: Incomplete, dz: float, wavelength: float, wavesize: Sequence[float]) -> None: ...
