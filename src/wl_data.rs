//! Manipulation and other things done from WangLandauData

use crate::RawWangLandauData;

pub struct WLData {
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
    pub bin_width: f64,
}

impl From<RawWangLandauData> for WLData {
    fn from(value: RawWangLandauData) -> Self {
        // Density of States, ln(g(E_i)) = dos[i]
        let dos: Vec<f64> = value.dos;
        // Histogram bins, use energy_to_bin to move from continuos energy -> discrete values
        let bins: Vec<usize> = value.bins;
        let n_bins: usize = bins.len();
        let min: f64 = value.min;
        let max: f64 = value.max;
        let bin_width: f64 = value.bin_width;

        WLData {
            dos,
            bins,
            min,
            max,
            bin_width,
        }
    }
}

impl WLData {
    pub fn log_partition_function(&self, beta: f64) -> f64 {
        let log_weights: Vec<f64> = self
            .dos
            .iter()
            .enumerate()
            .map(|(i, &ln_g)| {
                let e = bin_to_energy(self.min, self.bin_width, i);
                ln_g - beta * e
            })
            .collect();

        log_sum_exp(&log_weights)
    }

    pub fn energy_moments(&self, beta: f64) -> (f64, f64) {
        let log_z = self.log_partition_function(beta);

        let mut e_avg = 0.0;
        let mut e2_avg = 0.0;

        for (i, &ln_g) in self.dos.iter().enumerate() {
            let e = bin_to_energy(self.min, self.bin_width, i);
            let log_p = ln_g - beta * e - log_z;
            let p = log_p.exp();

            e_avg += p * e;
            e2_avg += p * e * e;
        }

        (e_avg, e2_avg)
    }

    pub fn specific_heat(&self, beta: f64, k_b: f64) -> f64 {
        let (e, e2) = self.energy_moments(beta);
        let var_e = e2 - e * e;

        beta * beta * var_e / k_b
    }

    /// you pass log_z
    pub fn free_energy(&self, log_z: f64, beta: f64, k_b: f64) -> f64 {
        -k_b * (1.0 / beta) * log_z
    }

    /// log_z is calculated on function
    pub fn free_energy2(&self, beta: f64, k_b: f64) -> f64 {
        -k_b * (1.0 / beta) * self.log_partition_function(beta)
    }

    pub fn entropy(&self, e_avg: f64, free_energy: f64, temperature: f64) -> f64 {
        (e_avg - free_energy) / temperature
    }

    pub fn energy_distribution(&self, beta: f64) -> Vec<f64> {
        let log_z = self.log_partition_function(beta);

        self.dos
            .iter()
            .enumerate()
            .map(|(i, &ln_g)| {
                let e = bin_to_energy(self.min, self.bin_width, i);
                (ln_g - beta * e - log_z).exp()
            })
            .collect()
    }

    pub fn microcanonical_entropy(&self, k_b: f64) -> Vec<f64> {
        self.dos.iter().map(|&ln_g| k_b * ln_g).collect()
    }

    pub fn microcanonical_temperature(&self, k_b: f64) -> Vec<f64> {
        let s: Vec<f64> = self.microcanonical_entropy(k_b);

        let mut temp = vec![0.0; self.dos.len()];

        for i in 1..self.dos.len() - 1 {
            let dsdE = (s[i + 1] - s[i - 1]) / (2.0 * self.bin_width);
            temp[i] = 1.0 / dsdE;
        }

        temp
    }
}

#[inline]
fn energy_to_bin(min: f64, max: f64, bin_width: f64, n_bins: usize, energy_value: f64) -> usize {
    let energy_value = energy_value.clamp(min, max);
    let i = ((energy_value - min) / bin_width) as usize;
    i.min(n_bins - 1)
}

#[inline]
fn bin_to_energy(min: f64, bin_width: f64, bin: usize) -> f64 {
    min + (bin as f64 + 0.5) * bin_width
}

/// Helper function to sum over ln values
fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + sum.ln()
}
