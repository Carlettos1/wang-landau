use crate::{ising::run_ising, quantum_harmonic_oscillator::run_qho};

mod ising;
mod quantum_harmonic_oscillator;

fn main() {
    run_ising();
    //run_qho();
}
