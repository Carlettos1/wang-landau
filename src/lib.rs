use csta::{Randomizable, State};
use itertools::*;
use rand::Rng;

/// when it switches from ln_f to 1/t
const switch_ln_f: f64 = 1e-6; // strict = 1e-9
/// how many visits (k*bins) should have before doing ln_f *= 0.5
const k: usize = 500; // strict = 1000
/// % difference with mean on for each bin
const p: f64 = 0.8; // strict = 0.9

/// Returns not normalized DoS and Histogram
pub fn wang_landau<S>(
    target_time: f64,
    sweeps: usize,
    mut params: S::Params,
    min_energy: f64,
    max_energy: f64,
    n_bins: usize,
) -> (DoS, Histogram)
where
    S: State + Randomizable,
{
    let mut ln_g = vec![0.0_f64; n_bins];
    let mut hist = Histogram::new(n_bins, min_energy, max_energy);
    let mut ln_f = 1.0;
    let mut rng = rand::rng();
    let mut state = S::sample(&mut rng);
    let mut prev_energy = state.energy(&mut params);
    let mut prev_energy_bin = hist.energy_to_bin(prev_energy);
    let mut switch: bool = false;

    'w: loop {
        for _ in 0..sweeps {
            let change = state.propose_change(&mut rng);
            state.apply_change(change.clone());
            let new_energy = state.energy(&mut params);
            let new_energy_bin = hist.energy_to_bin(new_energy);

            if rng.random::<f64>()
                < f64::min(1.0, (ln_g[prev_energy_bin] - ln_g[new_energy_bin]).exp())
            {
                prev_energy = new_energy;
                prev_energy_bin = new_energy_bin;
            } else {
                state.revert_change(change);
            }
            if switch {
                let t = hist.total_visits as f64 / hist.bins.len() as f64;
                if t >= target_time {
                    break 'w;
                }
                ln_g[prev_energy_bin] += 1.0 / t;
            } else {
                ln_g[prev_energy_bin] += ln_f;
            }
            hist.bins[prev_energy_bin] += 1;
            hist.visit();

            if prev_energy < hist.min || prev_energy > hist.max {
                println!("{prev_energy} outside bounds ({}, {})", hist.min, hist.max);
            }
        }

        if !switch && hist.is_flat() && hist.total_visits >= k * hist.bins.len() {
            ln_f *= 0.5;
            if ln_f < switch_ln_f {
                switch = true;
            } else {
                hist.clear(); // makes all bins 0
            }
        }
    }

    (ln_g, hist)
}

pub type DoS = Vec<f64>;

#[derive(Debug)]
pub struct Histogram {
    pub bins: Vec<usize>,
    pub min: f64,
    pub max: f64,
    pub bin_width: f64,
    pub total_visits: usize,
}

impl Histogram {
    pub fn new(bins: usize, min: f64, max: f64) -> Self {
        Self {
            bins: vec![0; bins],
            min,
            max,
            bin_width: (max - min) / bins as f64,
            total_visits: 0,
        }
    }

    pub fn energy_to_bin(&self, value: f64) -> usize {
        let value = value.clamp(self.min, self.max);
        let i = ((value - self.min) / self.bin_width) as usize;
        i.min(self.bins.len() - 1)
    }

    pub fn get(&self, value: f64) -> usize {
        let bin = self.energy_to_bin(value);
        self.bins[bin]
    }

    pub fn mean(&self) -> f64 {
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }

    pub fn is_flat(&self) -> bool {
        let mean = self.mean();
        for value in self.bins.iter() {
            if (*value as f64) < p * mean {
                return false;
            }
        }
        true
    }

    pub fn clear(&mut self) {
        self.bins.iter_mut().for_each(|v| *v = 0);
        self.total_visits = 0;
    }

    pub fn visit(&mut self) {
        self.total_visits += 1;
    }
}
