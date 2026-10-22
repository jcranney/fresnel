use fresnel::fresnel::{ang_spec_transfer_function_multiply_inplace};
use rustfft::num_complex::Complex;
use std::time::Instant;

fn main() {
    let mut input = vec![];
    for x in -128..128 {
        for y in -128..128 {
            let r = ((x as f64).powf(2.0) + (y as f64).powf(2.0)).powf(0.5);
            let v = match r < 16.0 {
                true => Complex::new(1.0, 0.0),
                false => Complex::new(0.0, 0.0),
            };
            input.push(v);
        }
    }
    let mut prop = fresnel::fresnel::InplacePropagator::new((256, 256));

    let t1 = Instant::now();
    for _ in 0..600 {
        prop.forward_planner.process_inplace(&mut input);
        ang_spec_transfer_function_multiply_inplace((256, 256), &mut input, 0.1, 1.6e-6, (7e-3, 7e-3));
        prop.inverse_planner.process_inplace(&mut input);
    }
    let elapsed = t1.elapsed();
    
    println!(
        "{:10}: time elapsed: {:0.3e} sec",
        prop.shape.0,
        elapsed.as_secs_f64()
    );
}
