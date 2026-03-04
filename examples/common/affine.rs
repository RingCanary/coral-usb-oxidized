use std::error::Error;

#[derive(Clone, Copy, Debug)]
pub struct AffineMap {
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct AffineFit {
    pub alpha: f64,
    pub beta: f64,
    pub corr: f64,
    pub mae: f64,
    pub rmse: f64,
}

#[derive(Default, Clone, Debug)]
pub struct VerifyStats {
    pub count: usize,
    pub mae: f64,
    pub rmse: f64,
    pub max_abs_delta: i32,
    pub mismatches_gt2: usize,
    pub corr: f64,
}

pub fn fit_affine(x: &[i32], y: &[i8]) -> Result<AffineFit, Box<dyn Error>> {
    if x.len() != y.len() || x.is_empty() {
        return Err("fit_affine expects equal non-empty slices".into());
    }

    let n = x.len() as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for idx in 0..x.len() {
        sum_x += x[idx] as f64;
        sum_y += y[idx] as f64;
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    let mut cov = 0.0f64;
    for idx in 0..x.len() {
        let dx = x[idx] as f64 - mean_x;
        let dy = y[idx] as f64 - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov += dx * dy;
    }

    let alpha = if var_x > 0.0 { cov / var_x } else { 0.0 };
    let beta = mean_y - alpha * mean_x;
    let corr = if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    };

    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for idx in 0..x.len() {
        let pred = alpha * x[idx] as f64 + beta;
        let err = y[idx] as f64 - pred;
        sum_abs += err.abs();
        sum_sq += err * err;
    }

    Ok(AffineFit {
        alpha,
        beta,
        corr,
        mae: sum_abs / n,
        rmse: (sum_sq / n).sqrt(),
    })
}

pub fn fit_affine_map(acc_i32: &[i32], tpu_q: &[i8]) -> Result<AffineMap, Box<dyn Error>> {
    let fit = fit_affine(acc_i32, tpu_q)?;
    Ok(AffineMap {
        alpha: fit.alpha,
        beta: fit.beta,
    })
}

fn clamp_i8_from_f64(value: f64) -> i8 {
    (value.round() as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

pub fn verify_against_affine(acc_i32: &[i32], tpu_q: &[i8], map: AffineMap) -> VerifyStats {
    if acc_i32.len() != tpu_q.len() || acc_i32.is_empty() {
        return VerifyStats::default();
    }

    let n = acc_i32.len() as f64;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut max_abs_delta = 0i32;
    let mut mismatches_gt2 = 0usize;

    let mut mean_pred = 0.0f64;
    let mut mean_tpu = 0.0f64;
    for i in 0..acc_i32.len() {
        let pred_q = clamp_i8_from_f64(map.alpha * acc_i32[i] as f64 + map.beta);
        mean_pred += pred_q as f64;
        mean_tpu += tpu_q[i] as f64;
    }
    mean_pred /= n;
    mean_tpu /= n;

    let mut cov = 0.0f64;
    let mut var_pred = 0.0f64;
    let mut var_tpu = 0.0f64;
    for i in 0..acc_i32.len() {
        let pred_q = clamp_i8_from_f64(map.alpha * acc_i32[i] as f64 + map.beta);
        let delta = pred_q as i32 - tpu_q[i] as i32;
        let abs_delta = delta.abs();
        abs_sum += abs_delta as f64;
        sq_sum += (delta * delta) as f64;
        if abs_delta > max_abs_delta {
            max_abs_delta = abs_delta;
        }
        if abs_delta > 2 {
            mismatches_gt2 += 1;
        }

        let dp = pred_q as f64 - mean_pred;
        let dt = tpu_q[i] as f64 - mean_tpu;
        cov += dp * dt;
        var_pred += dp * dp;
        var_tpu += dt * dt;
    }

    let corr = if var_pred > 0.0 && var_tpu > 0.0 {
        cov / (var_pred.sqrt() * var_tpu.sqrt())
    } else {
        0.0
    };

    VerifyStats {
        count: acc_i32.len(),
        mae: abs_sum / n,
        rmse: (sq_sum / n).sqrt(),
        max_abs_delta,
        mismatches_gt2,
        corr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_fit_affine(x: &[i32], y: &[i8]) -> (f64, f64, f64, f64, f64) {
        let n = x.len() as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for idx in 0..x.len() {
            sum_x += x[idx] as f64;
            sum_y += y[idx] as f64;
        }
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;
        let mut var_x = 0.0f64;
        let mut var_y = 0.0f64;
        let mut cov = 0.0f64;
        for idx in 0..x.len() {
            let dx = x[idx] as f64 - mean_x;
            let dy = y[idx] as f64 - mean_y;
            var_x += dx * dx;
            var_y += dy * dy;
            cov += dx * dy;
        }
        let alpha = if var_x > 0.0 { cov / var_x } else { 0.0 };
        let beta = mean_y - alpha * mean_x;
        let corr = if var_x > 0.0 && var_y > 0.0 {
            cov / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        };
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for idx in 0..x.len() {
            let pred = alpha * x[idx] as f64 + beta;
            let err = y[idx] as f64 - pred;
            sum_abs += err.abs();
            sum_sq += err * err;
        }
        (alpha, beta, corr, sum_abs / n, (sum_sq / n).sqrt())
    }

    #[test]
    fn parity_fit_affine() {
        let x = vec![-11, -3, 0, 4, 7, 21];
        let y = vec![-5, -1, 0, 2, 3, 9];
        let old = legacy_fit_affine(&x, &y);
        let new = fit_affine(&x, &y).unwrap();
        assert_eq!(old.0.to_bits(), new.alpha.to_bits());
        assert_eq!(old.1.to_bits(), new.beta.to_bits());
        assert_eq!(old.2.to_bits(), new.corr.to_bits());
        assert_eq!(old.3.to_bits(), new.mae.to_bits());
        assert_eq!(old.4.to_bits(), new.rmse.to_bits());
    }
}
