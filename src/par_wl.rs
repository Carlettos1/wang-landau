//! based on 2103.15028

use crate::*;

/// Uses ln(f) to get a rough estimate for a g_0(E), starts with the naive g_0(E)=1.
/// Then, uses 1/t to get a more precise g(E), starting from g_0(E) obtained before.
#[allow(clippy::too_many_arguments)]
pub fn par_wl<S>(
    target_time: f64,
    preliminary_sweeps: usize,
    preliminary_runs: usize,
    k_var: usize,
    mut params: S::Params,
    min_energy: f64,
    max_energy: f64,
    n_bins: usize,
) -> RawWangLandauData
where
    S: State + Randomizable,
{
    // uses 1/t approach
    let mut data: RawWangLandauData = RawWangLandauData::new(n_bins, min_energy, max_energy);
    let mut rng = rand::rng();
    let mut state = S::sample(&mut rng);
    let mut prev_energy = state.energy(&mut params);
    let mut prev_energy_bin = data.energy_to_bin(prev_energy);
    // using 2402.05653
    let t0 = n_bins as f64;
    // using 2402.05653 (t1 >= 10*t0)
    let t1 = 10.0 * t0;
    while let t = data.total_visits as f64 / data.bins.len() as f64
        && t < target_time
    {
        // using 2402.05653
        let gamma_t = t0 / (t1 + t);
        step(
            &mut state,
            &mut params,
            &mut data,
            &mut rng,
            &mut prev_energy,
            &mut prev_energy_bin,
            gamma_t,
        );
    }

    data
}

pub struct Window<R: Rng> {
    pub data: RawWangLandauData,
    pub rng: R,
}
