from typing import Tuple

from .fresnel import *
from numpy.typing import NDArray
import numpy as np


class Planner:
    def __init__(self, shape: Tuple[int, int]) -> None:
        self.fft2_planner = Fft2Planner(shape[0], shape[1])

    def fft2(self, array: NDArray[np.complex128]) -> NDArray[np.complex128]:
        tmp = array.copy()
        self.fft2_planner.fft2_numpy_inplace(tmp)
        return tmp