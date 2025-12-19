use csta::{Randomizable, State};
use cstawl::wang_landau;

const N: usize = 64;

#[derive(Debug)]
struct QHO {
    spins: Vec<isize>,
}

impl Randomizable for QHO {
    fn sample<R: rand::Rng + ?Sized>(_rng: &mut R) -> Self {
        Self { spins: vec![0; N] }
    }
}

impl State for QHO {
    /// position, prev_value, next_value
    type Change = (usize, isize, isize);
    type Params = ();

    fn energy(&self, _: &mut Self::Params) -> f64 {
        self.spins
            .iter()
            .fold(0.0, |acc, spin| acc + 0.5 * (*spin as f64 + 0.5))
    }

    fn propose_change(&self, rng: &mut impl rand::Rng) -> Self::Change {
        let rand_index = rng.random_range(0..self.spins.len());
        let new_spin = if self.spins[rand_index] == 0 || rng.random_bool(0.5) {
            1
        } else {
            -1
        };
        (rand_index, self.spins[rand_index], new_spin)
    }

    fn revert_change(&mut self, change: Self::Change) {
        self.spins[change.0] = change.1;
    }

    fn apply_change(&mut self, change: Self::Change) {
        self.spins[change.0] = change.2;
    }
}

pub fn run_qho() {
    println!("Starting qho");
    let (ln_g, hist) = wang_landau::<QHO>(1e-12, 10_000, (), -0.25 * N as f64, 0.75 * N as f64, N);
    println!("Finished qho");
    println!("ln_g: {ln_g:?}");
    println!("hist: {hist:#?}");
}
