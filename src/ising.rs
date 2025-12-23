use core::f64;

use csta::{Randomizable, State};
use cstawl::{WLData, log_sum_exp, wang_landau, wang_landau2};
use itertools::izip;
use rand::Rng;

const N: usize = 32;

#[derive(Debug)]
struct Ising1D {
    spins: Vec<i8>,
}

impl State for Ising1D {
    type Params = f64; // coupling J
    type Change = usize; // position to flip

    // fix: 31% del tiempo de ejecución:
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

pub fn run_ising() -> (f64, usize) {
    println!("Starting Ising 1D");
    let J = 1.0;
    let wl_raw_data = wang_landau2::<Ising1D>(
        5_000_000.0,
        1_000,
        5,
        500,
        J,
        -J * N as f64,
        J * N as f64,
        N / 2 + 1,
    );
    // let wl_raw_data =
    //     wang_landau::<Ising1D>(500_000.0, 10_000, J, -J * N as f64, J * N as f64, N / 2 + 1);
    let ln_g = wl_raw_data.dos.to_vec();
    println!("Finished Ising 1D");
    //println!("ln_g: {ln_g:#?}");

    let max_ln_g = ln_g.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let ln_z0: f64 = max_ln_g
        + ln_g
            .iter()
            .fold(0.0, |acc, li| acc + (li - max_ln_g).exp())
            .ln();
    //println!("{ln_z0}");
    let ln_g_normalized: Vec<f64> = ln_g
        .iter()
        .map(|li| li + (N as f64) * f64::consts::LN_2 - ln_z0)
        .collect();
    // let ln_sum = log_sum_exp(&ln_g);
    // let ln_de = wl_raw_data.bin_width.ln();
    // let ln_g_normalized: Vec<f64> = ln_g.iter().map(|li| li - ln_de - ln_sum).collect();
    //println!("{ln_g_normalized:?}");
    println!(
        "Total Number of states: {}",
        ln_g_normalized
            .iter()
            .fold(0.0, |acc, li_norm| acc + li_norm.exp())
    );
    println!("(Analitical Number): \t{}", 2.0_f64.powi(N as i32));

    let ln_ge0 = *ln_g.first().unwrap();
    let anchor = f64::consts::LN_2 - ln_ge0;
    let mut analitical = Vec::new();

    let mut error_normalized_state = 0.0;
    let mut error_anchored_state = 0.0;
    println!("Exact DoS comparison; g(E_k)=2(N k), N: {N}");
    println!(" N | k  | E_k  |ln(g(E_k))|normalized| anchored |analytical| energy");
    for (i, (normal_ln_gei, ln_gei)) in ln_g_normalized.iter().zip(ln_g.iter()).enumerate() {
        let k = 2 * i;
        let exact_ln = f64::consts::LN_2
            + ((N - k)..N)
                .filter(|x| x != &0)
                .fold(0.0, |acc, x| acc + (x as f64).ln())
            - (1..k).fold(0.0, |acc, x| acc + (x as f64).ln());
        let anchored = ln_gei + anchor;
        println!(
            "{} | {: >2} | E_{: <2} | {: >9.3}| {: >8.3} | {: >8.3} | {: >8.3} | {: >3}",
            N,
            k,
            k,
            ln_gei,
            normal_ln_gei,
            anchored,
            exact_ln,
            -(N as f64) + 2.0 * k as f64
        );
        error_anchored_state += (anchored - exact_ln).powi(2);
        error_normalized_state += (normal_ln_gei - exact_ln).powi(2);
        analitical.push(exact_ln);
    }
    println!("Errores:");
    println!("normalized: {error_anchored_state: >10.4}");
    println!("anchored:   {error_normalized_state: >10.4}");

    let data = WLData::from(wl_raw_data);
    let data2 = WLData {
        dos: analitical,
        min: -J * N as f64,
        max: J * N as f64,
        bins: vec![0; N / 2 - 1],
        bin_width: (J * N as f64 + J * N as f64) / (N / 2 - 1) as f64,
    };

    println!("Using Z(beta)");
    let mut cum_err = 0.0;
    let temps = (1..1001).map(|i| i as f64 / 100.0).collect::<Vec<_>>();
    let n_temps = temps.len();
    for temp in temps {
        let beta = 1.0 / temp;
        let log_z = data.log_partition_function(beta);

        let (e, e2) = data.energy_moments(beta);
        let c = data.specific_heat(beta, 1.0);
        let f = data.free_energy(log_z, beta, 1.0);
        let s = data.entropy(e, f, temp);

        let p_wl = data.energy_distribution(beta);
        let p_an = data2.energy_distribution(beta);

        // println!(
        //     "T={temp:.1}, Beta={beta:.3}, log_z={log_z:.4}, <E>={e:.4}, <E²>={e2:.4}, C(T)={c:.4}, F(T)={f:.4}, S(T)={s:.4}"
        // );
        if 0.4999 < temp && temp < 0.5001 {
            println!("P_wl(E|T)={p_wl:.4?}");
            println!("P_an(E|T)={p_an:.4?}");
        }
        let p_e_error: f64 = izip!(p_wl, p_an)
            .map(|(x, y)| ((x - y) * 100_f64).powi(2))
            .sum();

        if 0.4999 < temp && temp < 0.5001 {
            println!("Error of P(E|T) = {:.4}", p_e_error);
        }
        cum_err += p_e_error;
    }
    println!(
        "<err>: {:.2}, with {} temps. (cum_err: {:.2})",
        cum_err / n_temps as f64,
        n_temps,
        cum_err
    );
    // println!("Microcanonical data: ");
    let s_e = data.microcanonical_entropy(1.0);
    let t_micro = data.microcanonical_temperature(1.0);
    // println!("S(E)={s_e:.4?}");
    // println!("T(E)={t_micro:.4?}");
    (cum_err, n_temps)
}
