use crate::{ising::run_ising, quantum_harmonic_oscillator::run_qho};

mod ising;
mod quantum_harmonic_oscillator;

fn main() {
    run_ising();
}

fn main2() {
    let mut cum_errs = Vec::new();
    let mut n = 0;
    for i in 0..100 {
        println!("Running {i} ising sim");
        let (cum_err, n_temps) = run_ising();
        cum_errs.push(cum_err);
        n = n_temps;
    }
    let avg_errs = cum_errs
        .iter()
        .map(|cum_err| cum_err / n as f64)
        .collect::<Vec<_>>();
    let avg_err = avg_errs.iter().sum::<f64>() / avg_errs.len() as f64;
    let err_var: f64 = avg_errs.iter().map(|err| (err - avg_err).powi(2)).sum();
    println!("cum_errs: {:.0?}", cum_errs);
    println!("<errs>: {:.0?}", avg_errs);
    println!("<err>: {:.4}", avg_err);
    println!("err_var: {:.4}", err_var);
    println!("err_sd: {:.4}", err_var.sqrt());
    //run_qho();
}
