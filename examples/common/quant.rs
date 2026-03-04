use std::error::Error;

pub fn symmetric_scale_for_qmax(values: &[f32], qmax: i32, zero_threshold: f32) -> f32 {
    let mut max_abs = 0.0f32;
    for value in values {
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    if max_abs < zero_threshold {
        1.0
    } else {
        max_abs / qmax as f32
    }
}

pub fn quantize_symmetric_i8(values: &[f32], scale: f32, qmax: i32) -> Vec<i8> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let q = (*value / scale).round() as i32;
        out.push(q.clamp(-qmax, qmax) as i8);
    }
    out
}

pub fn cpu_accumulator_reference(
    input_q: &[i8],
    weights_row_major_q: &[i8],
    output_dim: usize,
) -> Vec<i32> {
    let input_dim = input_q.len();
    let mut out = vec![0i32; output_dim];
    for in_idx in 0..input_dim {
        let x = input_q[in_idx] as i32;
        if x == 0 {
            continue;
        }
        let row = &weights_row_major_q[in_idx * output_dim..(in_idx + 1) * output_dim];
        for out_idx in 0..output_dim {
            out[out_idx] += x * row[out_idx] as i32;
        }
    }
    out
}

pub fn cpu_accumulator_reference_batch(
    inputs_q: &[i8],
    weights_row_major_q: &[i8],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<i32>, Box<dyn Error>> {
    if inputs_q.len() % input_dim != 0 {
        return Err("inputs_q length mismatch for calibration".into());
    }
    if weights_row_major_q.len() != input_dim * output_dim {
        return Err("weights length mismatch for calibration".into());
    }

    let rows = inputs_q.len() / input_dim;
    let mut out = vec![0i32; rows * output_dim];

    for row in 0..rows {
        let x_row = &inputs_q[row * input_dim..(row + 1) * input_dim];
        let y_row = &mut out[row * output_dim..(row + 1) * output_dim];
        for in_idx in 0..input_dim {
            let x = x_row[in_idx] as i32;
            if x == 0 {
                continue;
            }
            let w_row = &weights_row_major_q[in_idx * output_dim..(in_idx + 1) * output_dim];
            for out_idx in 0..output_dim {
                y_row[out_idx] += x * w_row[out_idx] as i32;
            }
        }
    }

    Ok(out)
}

pub fn cpu_accumulator_reference_seq_dmodel(
    inputs_q: &[i8],
    weights_q: &[i8],
    seq_len: usize,
    d_model: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; seq_len * d_model];
    for row in 0..seq_len {
        let in_row = &inputs_q[row * d_model..(row + 1) * d_model];
        let out_row = &mut out[row * d_model..(row + 1) * d_model];
        for (k, &in_q) in in_row.iter().enumerate() {
            let x = in_q as i32;
            if x == 0 {
                continue;
            }
            let w_row = &weights_q[k * d_model..(k + 1) * d_model];
            for col in 0..d_model {
                out_row[col] += x * (w_row[col] as i32);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_symmetric_scale_for_qmax(values: &[f32], qmax: i32) -> f32 {
        let max_abs = values
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        if max_abs < 1e-12 {
            1.0
        } else {
            max_abs / qmax as f32
        }
    }

    fn legacy_quantize_symmetric_i8(values: &[f32], scale: f32, qmax: i32) -> Vec<i8> {
        values
            .iter()
            .map(|value| ((value / scale).round() as i32).clamp(-qmax, qmax) as i8)
            .collect()
    }

    #[test]
    fn parity_scale_and_quant() {
        let values = vec![-3.2, -0.5, 0.0, 0.49, 1.7, 2.9];
        let qmax = 16;
        let old_scale = legacy_symmetric_scale_for_qmax(&values, qmax);
        let new_scale = symmetric_scale_for_qmax(&values, qmax, 1e-12);
        assert_eq!(old_scale.to_bits(), new_scale.to_bits());

        let old_q = legacy_quantize_symmetric_i8(&values, old_scale, qmax);
        let new_q = quantize_symmetric_i8(&values, new_scale, qmax);
        assert_eq!(old_q, new_q);
    }

    fn legacy_cpu_accumulator_reference_batch(
        inputs_q: &[i8],
        weights_row_major_q: &[i8],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        if inputs_q.len() % input_dim != 0 {
            return Err("inputs_q length mismatch for calibration".into());
        }
        if weights_row_major_q.len() != input_dim * output_dim {
            return Err("weights length mismatch for calibration".into());
        }
        let rows = inputs_q.len() / input_dim;
        let mut out = vec![0i32; rows * output_dim];
        for row in 0..rows {
            let x_row = &inputs_q[row * input_dim..(row + 1) * input_dim];
            let y_row = &mut out[row * output_dim..(row + 1) * output_dim];
            for in_idx in 0..input_dim {
                let x = x_row[in_idx] as i32;
                if x == 0 {
                    continue;
                }
                let w_row = &weights_row_major_q[in_idx * output_dim..(in_idx + 1) * output_dim];
                for out_idx in 0..output_dim {
                    y_row[out_idx] += x * w_row[out_idx] as i32;
                }
            }
        }
        Ok(out)
    }

    #[test]
    fn parity_cpu_accumulator_batch() {
        let inputs = vec![1, -2, 3, 0, 4, -1];
        let weights = vec![2, -3, 4, 1, 5, -6, 7, -8, 9];
        let old = legacy_cpu_accumulator_reference_batch(&inputs, &weights, 3, 3).unwrap();
        let new = cpu_accumulator_reference_batch(&inputs, &weights, 3, 3).unwrap();
        assert_eq!(old, new);
    }
}
