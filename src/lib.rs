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
        buffer_a: Vec<Complex64>,
        buffer_b: Vec<Complex64>,
        scale: Option<f64>,
    }

    impl Fft2Planner {
        pub fn forward_planner(nrows: usize, ncols: usize) -> Self {
            let mut planner = FftPlanner::<f64>::new();
            let fft_rows = planner.plan_fft_forward(ncols);
            let fft_cols = planner.plan_fft_forward(nrows);
            let buffer_a = vec![Complex64::ZERO; nrows * ncols];
            let buffer_b = vec![Complex64::ZERO; ncols * nrows];
            Self {
                fft_rows,
                fft_cols,
                ncols,
                nrows,
                buffer_a,
                buffer_b,
                scale: None,
            }
        }
        pub fn inverse_planner(nrows: usize, ncols: usize) -> Self {
            let mut planner = FftPlanner::<f64>::new();
            let fft_rows = planner.plan_fft_inverse(ncols);
            let fft_cols = planner.plan_fft_inverse(nrows);
            let buffer_a = vec![Complex64::ZERO; nrows * ncols];
            let buffer_b = vec![Complex64::ZERO; ncols * nrows];
            Self {
                fft_rows,
                fft_cols,
                ncols,
                nrows,
                buffer_a,
                buffer_b,
                scale: Some(1.0 / (nrows * ncols) as f64),
            }
        }
        pub fn load_array(&mut self, array: &[Complex64]) {
            self.buffer_a.copy_from_slice(array);
        }
        pub fn read_array(&self) -> &[Complex64] {
            &self.buffer_a
        }
        pub fn ncols(&self) -> usize {
            self.ncols
        }
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        pub fn process_inplace(&mut self) {
            // do fft of each row
            self.buffer_a.par_chunks_mut(self.ncols).for_each(|row| {
                self.fft_rows.process(row); //, &mut scratch);
            });

            // transpose to allow efficient fourier transforming
            transpose::transpose(&self.buffer_a, &mut self.buffer_b, self.ncols, self.nrows);

            // do fft of each column of intermediate buffer
            self.buffer_b.par_chunks_mut(self.nrows).for_each(|col| {
                self.fft_cols.process(col);
                if let Some(scale) = self.scale {
                    col.iter_mut().for_each(|x| *x *= scale);
                }
            });

            // transpose to get back to original layout
            transpose::transpose(&self.buffer_b, &mut self.buffer_a, self.nrows, self.ncols);
        }
    }

    /// The `InplacePropagator` object is the main python interface, exposing an
    /// efficient `fft2`, `ifft2`, and `ang_spec` method acting inplace on
    /// numpy complex128 arrays.
    #[pyclass]
    struct InplacePropagator {
        forward_planner: Fft2Planner,
        inverse_planner: Fft2Planner,
        shape: (usize, usize),
    }

    #[pymethods]
    impl InplacePropagator {
        #[new]
        fn new(shape: (usize, usize)) -> Self {
            Self {
                forward_planner: Fft2Planner::forward_planner(shape.0, shape.1),
                inverse_planner: Fft2Planner::inverse_planner(shape.0, shape.1),
                shape,
            }
        }

        fn fft2(&mut self, mut array: PyReadwriteArray2<Complex64>) -> PyResult<()> {
            // load up data into planner buffer
            self.forward_planner.load_array(array.as_slice()?);
            // perform fft2
            self.forward_planner.process_inplace();
            // read back data from buffer
            array
                .as_slice_mut()?
                .copy_from_slice(self.forward_planner.read_array());
            Ok(())
        }

        fn ifft2(&mut self, mut array: PyReadwriteArray2<Complex64>) -> PyResult<()> {
            // load up data into planner buffer
            self.inverse_planner.load_array(array.as_slice()?);
            // perform ifft2

            self.inverse_planner.process_inplace();
            // read back data from buffer
            array
                .as_slice_mut()?
                .copy_from_slice(self.inverse_planner.read_array());
            Ok(())
        }

        fn ang_spec(
            &mut self,
            mut array: PyReadwriteArray2<Complex64>,
            dz: f64,
            wavelength: f64,
            wavesize_xy: (f64, f64),
        ) -> PyResult<()> {
            self.forward_planner.load_array(array.as_slice()?);
            self.forward_planner.process_inplace();
            array
                .as_slice_mut()?
                .copy_from_slice(self.forward_planner.read_array());
            self.ang_spec_transfer_function_multiply_inplace(
                array.as_slice_mut()?,
                dz,
                wavelength,
                wavesize_xy,
            );
            self.inverse_planner.load_array(array.as_slice()?);
            self.inverse_planner.process_inplace();
            array
                .as_slice_mut()?
                .copy_from_slice(self.inverse_planner.read_array());
            Ok(())
        }
    }

    impl InplacePropagator {
        fn ang_spec_transfer_function_multiply_inplace(
            &self,
            array_slice: &mut [Complex64],
            dz: f64,
            wavelength: f64,
            wavesize_xy: (f64, f64),
        ) {
            let nrows = self.shape.0;
            let ncols = self.shape.1;
            let k = 2.0 * PI / wavelength;
            let alpha_s = 1.0 / wavesize_xy.0;
            let beta_s = 1.0 / wavesize_xy.1;
            array_slice
                .iter_mut()
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
                    let p = (x * alpha_s).powf(2.0);
                    let q = (y * beta_s).powf(2.0);
                    let s = p + q;
                    let phi = -PI.powf(2.0) * 2.0 / k * dz * s;
                    let f = (Complex64::i() * phi).exp();
                    *z *= f;
                });
        }
    }
}
