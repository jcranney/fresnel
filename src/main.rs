use fresnel::fresnel::Fft2Planner;
use rustfft::num_complex::Complex;
use std::time::Instant;

fn main() {
    let mut input = vec![];
    for x in -512..512 {
        for y in -512..512 {
            let r = ((x as f64).powf(2.0) + (y as f64).powf(2.0)).powf(0.5);
            let v = match r < 16.0 {
                true => Complex::new(1.0, 0.0),
                false => Complex::new(0.0, 0.0),
            };
            input.push(v);
        }
    }
    let mut planner = Fft2Planner::new(1024, 1024);

    let t1 = Instant::now();
    planner.load_array(&input);
    planner.fft2_inplace();
    let elapsed = t1.elapsed();
    println!("{}", planner.read_array()[0]);
    
    println!(
        "{:10}: time elapsed: {:0.3e} sec",
        planner.nrows(),
        elapsed.as_secs_f64()
    );
}
