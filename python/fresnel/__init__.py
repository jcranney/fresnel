from typing import Tuple

from .fresnel import *
from numpy.typing import NDArray
import numpy as np


class Planner:
    def __init__(self, shape: Tuple[int, int]) -> None:
        self.fft2_planner = Fft2Planner.forward_planner(shape[0], shape[1])
        self.ifft2_planner = Fft2Planner.inverse_planner(shape[0], shape[1])

    def fft2(self, array: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        tmp = array.copy()
        self.fft2_planner.process_numpy_inplace(tmp)
        return tmp

    def ifft2(self, array: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        tmp = array.copy()
        self.ifft2_planner.process_numpy_inplace(tmp)
        return tmp

    def fft2_inplace(self, array: NDArray[np.complexfloating]):
        self.fft2_planner.process_numpy_inplace(array)

    def ifft2_inplace(self, array: NDArray[np.complexfloating]):
        self.ifft2_planner.process_numpy_inplace(array)

    def propagate_inplace(
        self,
        wave: NDArray[np.complexfloating],
        dz: float,
        wavelength: float,
        wavesize: Tuple[float, float],
    ):
        # forward fourier transform once:
        self.fft2_inplace(wave)

        # multiply with propagation operator
        fresnel_transfer_inplace(wave, dz, wavelength, wavesize)

        # inverse fourier transform
        self.ifft2_inplace(wave)
