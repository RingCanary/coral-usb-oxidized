#[path = "../examples/common/mod.rs"]
mod common;

use common::affine::{fit_affine, fit_affine_map, verify_against_affine, AffineMap};
use common::quant::{
    cpu_accumulator_reference, cpu_accumulator_reference_batch,
    cpu_accumulator_reference_seq_dmodel, quantize_symmetric_i8, symmetric_scale_for_qmax,
};
use common::rng::build_calibration_input_q;

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
fn parity_helpers_against_legacy_logic() {
    let f = vec![-2.25, -0.1, 0.0, 0.8, 1.6, 3.2];
    let scale_old = 3.2 / 16.0f32;
    let scale_new = symmetric_scale_for_qmax(&f, 16, 0.0);
    assert_eq!(scale_old.to_bits(), scale_new.to_bits());
    assert_eq!(
        quantize_symmetric_i8(&f, scale_old, 16),
        quantize_symmetric_i8(&f, scale_new, 16)
    );

    let input = vec![1, -2, 3, 0];
    let weights_row_major = vec![4, -5, 6, 7, -8, 9, -1, 2];
    assert_eq!(
        cpu_accumulator_reference(&input, &weights_row_major, 2),
        vec![-32, 8]
    );

    let batch_in = vec![1, 0, -1, 2, -2, 1];
    let batch_w = vec![2, 3, 4, 5, 6, 7];
    let old_rng = legacy_build_calibration_input_q(3, 4, 31, 42);
    let new_rng = build_calibration_input_q(3, 4, 31, 42);
    assert_eq!(old_rng, new_rng);

    let old_batch = cpu_accumulator_reference_batch(&batch_in, &batch_w, 3, 2).unwrap();

    let seq_inputs = vec![1, -1, 2, 0, 3, -2, 1, 1];
    let seq_weights = vec![1, 2, 3, 4, -1, -2, -3, -4, 5, 6, 7, 8, 0, 1, 0, 1];
    let seq = cpu_accumulator_reference_seq_dmodel(&seq_inputs, &seq_weights, 2, 4);
    assert_eq!(seq.len(), 8);

    let y: Vec<i8> = old_batch.iter().map(|v| (v / 2) as i8).collect();
    let fit = fit_affine(&old_batch, &y).unwrap();
    let map = fit_affine_map(&old_batch, &y).unwrap();
    assert_eq!(fit.alpha.to_bits(), map.alpha.to_bits());
    assert_eq!(fit.beta.to_bits(), map.beta.to_bits());

    let stats = verify_against_affine(
        &old_batch,
        &y,
        AffineMap {
            alpha: fit.alpha,
            beta: fit.beta,
        },
    );
    assert_eq!(stats.count, y.len());
}
