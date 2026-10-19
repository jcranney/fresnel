use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
pub mod fresnel {
    use numpy::{PyReadwriteArray2, PyUntypedArrayMethods};
    use rayon::iter::ParallelIterator;
    use rayon::prelude::*;
    use std::{f64::consts::PI, sync::Arc};

    use pyo3::prelude::*;
    use rustfft::{Fft, FftPlanner, num_complex::Complex64};

    #[pyclass]
    pub struct Fft2Planner {
        ncols: usize,
        nrows: usize,
        fft_rows: Arc<dyn Fft<f64>>,
        fft_cols: Arc<dyn Fft<f64>>,
        buffer_a: Vec<Complex64>,
        buffer_b: Vec<Complex64>,
        scale: Option<f64>,
    }

    #[pymethods]
    impl Fft2Planner {
        #[staticmethod]
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
        #[staticmethod]
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

        pub fn nrows(&self) -> usize {
            self.nrows
        }
        pub fn ncols(&self) -> usize {
            self.ncols
        }

        pub fn process_numpy_inplace(&mut self, mut array: PyReadwriteArray2<Complex64>) {
            // load up data into planner buffer
            self.load_array(array.as_slice().unwrap());

            // perform fft2
            self.process_inplace();

            // read back data from buffer
            array
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(self.read_array());
        }
    }

    impl Fft2Planner {
        pub fn load_array(&mut self, array: &[Complex64]) {
            self.buffer_a.copy_from_slice(array);
        }

        pub fn read_array(&self) -> &[Complex64] {
            &self.buffer_a
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

    #[pyfunction]
    fn fresnel_transfer_inplace(
        mut array: PyReadwriteArray2<Complex64>,
        dz: f64,
        wavelength: f64,
        wavesize: [f64; 2],
    ) {
        let [nrows, ncols] = array.shape()[..2].as_array().unwrap().clone();
        // Time to find that dastardly conversion factor from pixels to spatial
        // frequency units.
        // Just going to assume that it's the same process as a temporal
        // fourier transform...
        let alpha_s = ncols as f64 / wavesize[0];
        let beta_s = nrows as f64 / wavesize[1];

        array
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
            .map(|(i, z)| {
                (
                    (
                        ((((i % ncols) + (ncols / 2)) % ncols) as isize - (ncols / 2) as isize)
                            as f64
                            / ncols as f64,
                        ((((i / ncols) + (nrows / 2)) % nrows) as isize - (nrows / 2) as isize)
                            as f64
                            / nrows as f64,
                    ),
                    z,
                )
            })
            .for_each(|((x, y), z)| {
                let rho_sqr =
                    ((x * x) * alpha_s / wavelength + (y * y) * beta_s / wavelength).powf(2.0);
                let f = (-Complex64::i() * PI * wavelength * dz * rho_sqr).exp();
                *z *= f;
            });
    }
}
