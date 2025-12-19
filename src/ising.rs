use core::f64;

use csta::{Randomizable, State, csta_derive::Randomizable};
use cstawl::{Histogram, wang_landau};
use rand::Rng;

const N: usize = 128;

#[derive(Debug)]
struct Ising1D {
    spins: Vec<i8>,
}

impl State for Ising1D {
    type Params = f64; // coupling J
    type Change = usize; // position to flip

    fn energy(&self, J: &mut f64) -> f64 {
        let mut e = 0.0;
        for i in 0..self.spins.len() {
            let j = (i + 1) % self.spins.len();
            e -= *J * (self.spins[i] as f64) * (self.spins[j] as f64);
        }
        e
    }

    fn propose_change(&self, rng: &mut impl Rng) -> usize {
        rng.random_range(0..self.spins.len())
    }

    fn apply_change(&mut self, i: usize) {
        self.spins[i] *= -1;
    }

    fn revert_change(&mut self, i: usize) {
        self.spins[i] *= -1;
    }
}

impl Randomizable for Ising1D {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let n = N;
        let spins = (0..n)
            .map(|_| if rng.random_bool(0.5) { 1 } else { -1 })
            .collect();
        Self { spins }
    }
}

pub fn run_ising() {
    println!("Starting Ising 1D");
    let J = 1.0;
    let (ln_g, hist) =
        wang_landau::<Ising1D>(1e-12, 10_000, J, -J * N as f64, J * N as f64, N / 2 + 1);
    println!("Finished Ising 1D");
    println!("ln_g: {ln_g:#?}");

    let max = ln_g.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let ln_z0: f64 = max + ln_g.iter().fold(0.0, |acc, li| acc + (li - max).exp()).ln();
    println!("{ln_z0}");
    let ln_g_normalized: Vec<f64> = ln_g
        .iter()
        .map(|li| li + N as f64 * f64::consts::LN_2 - ln_z0)
        .collect();
    println!("{ln_g_normalized:?}");
    println!(
        "Total Number of states: {}",
        ln_g_normalized
            .iter()
            .fold(0.0, |acc, li_norm| acc + li_norm.exp())
    );
    println!("(Analitical Number): \t{}", 2.0_f64.powi(N as i32));

    let ln_ge0 = *ln_g.first().unwrap();
    let anchor = f64::consts::LN_2 - ln_ge0;

    let mut error_normalized_state = 0.0;
    let mut error_anchored_state = 0.0;
    println!("Exact DoS comparison; g(E_k)=2(N k), N: {N}");
    println!(" N | k  | E_k  |ln(g(E_k))| anchored |analytical| energy");
    for (i, (normal_ln_gei, ln_gei)) in ln_g_normalized.iter().zip(ln_g.iter()).enumerate() {
        let k = 2 * i;
        let exact_ln = f64::consts::LN_2
            + ((N - k)..N)
                .filter(|x| x != &0)
                .fold(0.0, |acc, x| acc + (x as f64).ln())
            - (1..k).fold(0.0, |acc, x| acc + (x as f64).ln());
        let anchored = ln_gei + anchor;
        println!(
            "{} | {: >2} | E_{: <2} | {: >8.3} | {: >8.3} | {: >8.3} | {: >3}",
            N,
            k,
            k,
            normal_ln_gei,
            anchored,
            exact_ln,
            -(N as f64) + 2.0 * k as f64
        );
        error_anchored_state += (anchored - exact_ln).powi(2);
        error_normalized_state += (normal_ln_gei - exact_ln).powi(2);
    }
    println!("Errores:");
    println!("normalized: {error_anchored_state: >10.4}");
    println!("anchored:   {error_normalized_state: >10.4}");
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

fn free_energy(ln_g: &[f64], hist: &Histogram, beta: f64) -> f64 {
    let terms: Vec<f64> = ln_g
        .iter()
        .enumerate()
        .map(|(i, &lg)| {
            let E = hist.min + (i as f64 + 0.5) * hist.bin_width;
            lg - beta * E
        })
        .collect();

    -log_sum_exp(&terms) / beta
}
