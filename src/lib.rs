use csta::{Randomizable, State};
use itertools::*;
use rand::Rng;

mod wl_data;
pub use wl_data::{WLData, log_sum_exp};

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
) -> RawWangLandauData
where
    S: State + Randomizable,
{
    let mut data: RawWangLandauData = RawWangLandauData::new(n_bins, min_energy, max_energy);
    let mut ln_f = 1.0;
    let mut rng = rand::rng();
    let mut state = S::sample(&mut rng);
    let mut prev_energy = state.energy(&mut params);
    let mut prev_energy_bin = data.energy_to_bin(prev_energy);
    let mut switch: bool = false;

    'w: loop {
        for _ in 0..sweeps {
            let change = state.propose_change(&mut rng);
            state.apply_change(change.clone());
            let new_energy = state.energy(&mut params);
            let new_energy_bin = data.energy_to_bin(new_energy);

            if rng.random::<f64>()
                < f64::min(
                    1.0,
                    (data.dos[prev_energy_bin] - data.dos[new_energy_bin]).exp(),
                )
            {
                prev_energy = new_energy;
                prev_energy_bin = new_energy_bin;
            } else {
                state.revert_change(change);
            }
            if switch {
                let t = (data.total_visits + 1) as f64 / data.bins.len() as f64;
                if t >= target_time {
                    break 'w;
                }
                data.dos[prev_energy_bin] += 1.0 / t;
            } else {
                data.dos[prev_energy_bin] += ln_f;
            }
            data.bins[prev_energy_bin] += 1;
            data.visit();

            if prev_energy < data.min || prev_energy > data.max {
                println!("{prev_energy} outside bounds ({}, {})", data.min, data.max);
            }
        }

        if !switch && data.is_flat() && data.visits >= k * data.bins.len() {
            ln_f *= 0.5;
            if ln_f < switch_ln_f {
                switch = true;
            } else {
            }
            data.clear_hist(); // makes all bins 0
        }
    }

    data
}

#[derive(Debug)]
pub struct RawWangLandauData {
    /// Density of States
    pub dos: Vec<f64>,
    // Histogram data
    /// Histogram bins
    pub bins: Vec<usize>,
    /// minimum energy
    pub min: f64,
    /// maximum energy
    pub max: f64,
    /// width of energy of each bin
    #[doc(alias = "de")]
    pub bin_width: f64,
    /// visits between each ln_f
    visits: usize,
    total_visits: usize,
}

impl RawWangLandauData {
    pub fn new(n_bins: usize, min: f64, max: f64) -> Self {
        Self {
            dos: vec![0.0_f64; n_bins],
            bins: vec![0; n_bins],
            min,
            max,
            bin_width: (max - min) / n_bins as f64,
            visits: 0,
            total_visits: 0,
        }
    }

    pub fn energy_to_bin(&self, energy_value: f64) -> usize {
        let energy_value = energy_value.clamp(self.min, self.max);
        let i = ((energy_value - self.min) / self.bin_width) as usize;
        i.min(self.bins.len() - 1)
    }

    pub fn get(&self, energy_value: f64) -> usize {
        let bin = self.energy_to_bin(energy_value);
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

    pub fn clear_hist(&mut self) {
        self.bins.iter_mut().for_each(|v| *v = 0);
        self.visits = 0;
    }

    pub fn visit(&mut self) {
        self.visits += 1;
        self.total_visits += 1;
    }

    pub fn process_data(self) -> WLData {
        self.into()
    }
}
