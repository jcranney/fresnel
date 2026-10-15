use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
pub mod fresnel {
    use numpy::PyReadwriteArray2;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::*;
    use std::sync::Arc;

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
    }

    #[pymethods]
    impl Fft2Planner {
        #[new]
        pub fn new(nrows: usize, ncols: usize) -> Self {
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
            }
        }

        pub fn nrows(&self) -> usize {
            self.nrows
        }
        pub fn ncols(&self) -> usize {
            self.ncols
        }

        pub fn fft2_numpy_inplace(&mut self, mut array: PyReadwriteArray2<Complex64>) {
            // load up data into planner buffer
            self.load_array(array.as_slice().unwrap());
    
            // perform fft2
            self.fft2_inplace();

            // read back data from buffer
            array.as_slice_mut().unwrap().copy_from_slice(self.read_array());
        }
    }

    impl Fft2Planner {
        pub fn load_array(&mut self, array: &[Complex64]) {
            self.buffer_a.copy_from_slice(array);
        }

        pub fn read_array(&self) -> &[Complex64] {
            &self.buffer_a
        }

        pub fn fft2_inplace(&mut self) {
            // do fft of each row
            self.buffer_a.par_chunks_mut(self.ncols).for_each(|row| {
                self.fft_rows.process(row); //, &mut scratch);
            });

            // transpose to allow efficient fourier transforming
            transpose::transpose(&self.buffer_a, &mut self.buffer_b, self.ncols, self.nrows);

            // do fft of each column of intermediate buffer
            self.buffer_b.par_chunks_mut(self.nrows).for_each(|col| {
                self.fft_cols.process(col);
            });

            // transpose to get back to original layout
            transpose::transpose(&self.buffer_b, &mut self.buffer_a, self.nrows, self.ncols);
        }
    }
}

// struct Field<T> {
//     data: Vec<Complex<T>>,
//     nrows: usize,
//     ncols: usize,
// }

// impl Field<f64> {
//     fn zeros(nrows: usize, ncols: usize) -> Self {
//         Self {
//             data: vec![Complex { re: 0.0, im: 0.0 }; nrows * ncols],
//             nrows,
//             ncols,
//         }
//     }

//     fn fft(&self) -> Self {
//         let result = fft2(&self.data, self.nrows, self.ncols);
//         Self {
//             data: result,
//             nrows: self.nrows,
//             ncols: self.ncols,
//         }
//     }

//     fn ifft(&self) -> Self {
//         let result = ifft2(&self.data, self.nrows, self.ncols);
//         Self {
//             data: result,
//             nrows: self.nrows,
//             ncols: self.ncols,
//         }
//     }

//     fn element_wise_prod(&self, other: &Self) -> Self {
//         todo!()
//     }

//     fn fresnel_transfer_function(dz: f64) -> Self {
//         todo!()
//     }
// }

// pub fn angular_spectrum_propagate(f_in: &Field<f64>, dz: f64) -> Field<f64> {
//     // tranfer function only depends on propagation distance
//     let tf = Field::fresnel_transfer_function(dz);

//     // output field is inverse FFT of Transfer Function times FFT of input field
//     let f_out = (tf.element_wise_prod(&f_in.fft())).ifft();
//     f_out
// }
