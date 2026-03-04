pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self, low: f32, high: f32) -> f32 {
        let unit = ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        low + (high - low) * unit as f32
    }
}

pub fn build_calibration_input_q(rows: usize, input_dim: usize, qmax: i32, seed: u64) -> Vec<i8> {
    let mut out = vec![0i8; rows * input_dim];
    let mut rng = XorShift64::new(seed);
    for value in &mut out {
        let sample = ((rng.next_u64() >> 8) as i32 % (2 * qmax + 1)) - qmax;
        *value = sample as i8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_build_calibration_input_q(
        rows: usize,
        input_dim: usize,
        qmax: i32,
        seed: u64,
    ) -> Vec<i8> {
        let mut out = vec![0i8; rows * input_dim];
        let mut state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        for value in &mut out {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sample = ((state >> 8) as i32 % (2 * qmax + 1)) - qmax;
            *value = sample as i8;
        }
        out
    }

    #[test]
    fn parity_calibration_rng_sequence() {
        let rows = 4;
        let input_dim = 9;
        let qmax = 31;
        let seed = 0x1234_5678_9abc_def0;
        let old = legacy_build_calibration_input_q(rows, input_dim, qmax, seed);
        let new = build_calibration_input_q(rows, input_dim, qmax, seed);
        assert_eq!(old, new);
    }
}
