use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
pub mod fresnel {
    use numpy::PyReadwriteArray2;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::*;
    use std::{f64::consts::PI, sync::Arc};

    use pyo3::prelude::*;
    use rustfft::{Fft, FftPlanner, num_complex::Complex64};

    pub struct Fft2Planner {
        ncols: usize,
        nrows: usize,
        fft_rows: Arc<dyn Fft<f64>>,
        fft_cols: Arc<dyn Fft<f64>>,
        buffer: Vec<Complex64>,
        scale: Option<f64>,
    }

    impl Fft2Planner {
        pub fn forward_planner(nrows: usize, ncols: usize) -> Self {
            let mut planner = FftPlanner::<f64>::new();
            let fft_rows = planner.plan_fft_forward(ncols);
            let fft_cols = planner.plan_fft_forward(nrows);
            let buffer = vec![Complex64::ZERO; ncols * nrows];
            Self {
                fft_rows,
                fft_cols,
                ncols,
                nrows,
                buffer,
                scale: None,
            }
        }
        pub fn inverse_planner(nrows: usize, ncols: usize) -> Self {
            let mut planner = FftPlanner::<f64>::new();
            let fft_rows = planner.plan_fft_inverse(ncols);
            let fft_cols = planner.plan_fft_inverse(nrows);
            let buffer = vec![Complex64::ZERO; ncols * nrows];
            Self {
                fft_rows,
                fft_cols,
                ncols,
                nrows,
                buffer,
                scale: Some(1.0 / (nrows * ncols) as f64),
            }
        }
        pub fn ncols(&self) -> usize {
            self.ncols
        }
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        pub fn process_inplace(&mut self, data: &mut [Complex64]) {
            // do fft of each row
            data.par_chunks_mut(self.ncols).for_each(|row| {
                self.fft_rows.process(row);
            });

            // transpose to allow efficient fourier transforming
            transpose::transpose(data, &mut self.buffer, self.ncols, self.nrows);

            // do fft of each column of intermediate buffer
            self.buffer.par_chunks_mut(self.nrows).for_each(|col| {
                self.fft_cols.process(col);
                if let Some(scale) = self.scale {
                    col.iter_mut().for_each(|x| *x *= scale);
                }
            });

            // transpose to get back to original layout
            transpose::transpose(&self.buffer, data, self.nrows, self.ncols);
        }
    }

    /// The `InplacePropagator` object is the main python interface, exposing an
    /// efficient `fft2`, `ifft2`, and `ang_spec` method acting inplace on
    /// numpy complex128 arrays.
    #[pyclass]
    pub struct InplacePropagator {
        pub forward_planner: Fft2Planner,
        pub inverse_planner: Fft2Planner,
        pub shape: (usize, usize),
    }

    #[pymethods]
    impl InplacePropagator {
        #[new]
        pub fn new(shape: (usize, usize)) -> Self {
            Self {
                forward_planner: Fft2Planner::forward_planner(shape.0, shape.1),
                inverse_planner: Fft2Planner::inverse_planner(shape.0, shape.1),
                shape,
            }
        }

        pub fn fft2(&mut self, mut array: PyReadwriteArray2<Complex64>) -> PyResult<()> {
            // perform fft2
            self.forward_planner.process_inplace(array.as_slice_mut()?);
            Ok(())
        }

        pub fn ifft2(&mut self, mut array: PyReadwriteArray2<Complex64>) -> PyResult<()> {
            // perform ifft2
            self.inverse_planner.process_inplace(array.as_slice_mut()?);
            Ok(())
        }

        pub fn ang_spec(
            &mut self,
            mut array: PyReadwriteArray2<Complex64>,
            dz: f64,
            wavelength: f64,
            wavesize_xy: (f64, f64),
        ) -> PyResult<()> {
            self.forward_planner.process_inplace(array.as_slice_mut()?);

            ang_spec_transfer_function_multiply_inplace(
                self.shape,
                array.as_slice_mut()?,
                dz,
                wavelength,
                wavesize_xy,
            );
            self.inverse_planner.process_inplace(array.as_slice_mut()?);
            Ok(())
        }
    }

    pub fn ang_spec_transfer_function_multiply_inplace(
        shape: (usize, usize),
        array_slice: &mut [Complex64],
        dz: f64,
        wavelength: f64,
        wavesize_xy: (f64, f64),
    ) {
        let nrows = shape.0;
        let ncols = shape.1;
        let k = 2.0 * PI / wavelength;
        let alpha_s = 1.0 / wavesize_xy.0;
        let beta_s = 1.0 / wavesize_xy.1;
        let prefactor = Complex64::i() * -PI.powf(2.0) * 2.0 / k * dz;
        array_slice
            .par_iter_mut()
            .enumerate()
            .map(|(i, z)| {
                (
                    (
                        ((((i % ncols) + (ncols / 2)) % ncols) as isize - (ncols / 2) as isize)
                            as f64,
                        ((((i / ncols) + (nrows / 2)) % nrows) as isize - (nrows / 2) as isize)
                            as f64,
                    ),
                    z,
                )
            })
            .for_each(|((x, y), z)| {
                let p = (x * alpha_s)*(x * alpha_s);
                let q = (y * beta_s)*(y * beta_s);
                let s = p + q;
                let f = (prefactor * s).exp();
                *z *= f;
            });
    }
}
