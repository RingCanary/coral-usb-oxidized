use coral_usb_oxidized::{version, CoralDevice, DenseGemmTemplate};
use std::env;
use std::error::Error;
use std::fs;
use std::time::Instant;

const DIM: usize = 2304;
const WEIGHT_COUNT: usize = DIM * DIM;

struct Cli {
    seq_len: usize,
    runs: usize,
    warmup: usize,
    calibration_rows: usize,
    seed: u64,
    input_qmax: i32,
    weight_qmax: i32,
    weights_f32_le_path: Option<String>,
    inputs_f32_le_path: Option<String>,
}

#[derive(Clone, Copy)]
struct AffineMap {
    alpha: f64,
    beta: f64,
}

#[derive(Default)]
struct VerifyStats {
    count: usize,
    mae: f64,
    rmse: f64,
    max_abs_delta: i32,
    mismatches_gt2: usize,
    mismatches_gt4: usize,
    corr: f64,
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self, low: f32, high: f32) -> f32 {
        let unit = ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        low + (high - low) * unit as f32
    }
}

fn usage(program: &str) {
    println!(
        "Usage: {program} [seq_len] [runs] [warmup] [calibration_rows] [--seed N] [--input-qmax N] [--weight-qmax N] [--weights-f32-le PATH] [--inputs-f32-le PATH]"
    );
    println!(
        "Defaults: seq_len=8 runs=3 warmup=1 calibration_rows=2 seed=1 input_qmax=32 weight_qmax=16"
    );
}

fn parse_args() -> Result<Cli, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "gemm_weight_load_verify".to_string());

    let mut positional: Vec<String> = Vec::new();
    let mut seed = 1u64;
    let mut input_qmax = 32i32;
    let mut weight_qmax = 16i32;
    let mut weights_f32_le_path = None;
    let mut inputs_f32_le_path = None;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                usage(&program);
                std::process::exit(0);
            }
            "--seed" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--seed requires a value".into());
                }
                seed = args[idx].parse::<u64>()?;
            }
            "--input-qmax" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--input-qmax requires a value".into());
                }
                input_qmax = args[idx].parse::<i32>()?;
            }
            "--weight-qmax" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--weight-qmax requires a value".into());
                }
                weight_qmax = args[idx].parse::<i32>()?;
            }
            "--weights-f32-le" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--weights-f32-le requires a path".into());
                }
                weights_f32_le_path = Some(args[idx].clone());
            }
            "--inputs-f32-le" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--inputs-f32-le requires a path".into());
                }
                inputs_f32_le_path = Some(args[idx].clone());
            }
            other => positional.push(other.to_string()),
        }
        idx += 1;
    }

    let seq_len = positional
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let runs = positional
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let warmup = positional
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let calibration_rows = positional
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);

    if seq_len == 0 {
        return Err("seq_len must be >= 1".into());
    }
    if runs == 0 {
        return Err("runs must be >= 1".into());
    }
    if calibration_rows == 0 {
        return Err("calibration_rows must be >= 1".into());
    }
    if !(1..=127).contains(&input_qmax) {
        return Err("input_qmax must be in [1, 127]".into());
    }
    if !(1..=127).contains(&weight_qmax) {
        return Err("weight_qmax must be in [1, 127]".into());
    }

    Ok(Cli {
        seq_len,
        runs,
        warmup,
        calibration_rows,
        seed,
        input_qmax,
        weight_qmax,
        weights_f32_le_path,
        inputs_f32_le_path,
    })
}

fn read_f32_le_file(path: &str, expected_count: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let expected_bytes = expected_count
        .checked_mul(4)
        .ok_or("f32 byte count overflow")?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "unexpected f32 byte size for {}: expected {}, got {}",
            path,
            expected_bytes,
            bytes.len()
        )
        .into());
    }

    let mut out = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn generate_weights_f32(seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed ^ 0xA5A5_5A5A_1234_5678);
    let mut out = vec![0.0f32; WEIGHT_COUNT];

    for row in 0..DIM {
        for col in 0..DIM {
            let idx = row * DIM + col;
            let base = rng.next_f32(-0.12, 0.12);
            let skip = if (row + 3 * col) % 11 == 0 { 0.18 } else { 0.0 };
            out[idx] = base + skip;
        }
    }
    out
}

fn generate_inputs_f32(seq_len: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed ^ 0xBADC_0FFE_EE11_D00D);
    let mut out = vec![0.0f32; seq_len * DIM];
    for row in 0..seq_len {
        for col in 0..DIM {
            let idx = row * DIM + col;
            let periodic = (((row * 13 + col) & 63) as f32 - 31.5) / 96.0;
            out[idx] = periodic + rng.next_f32(-0.04, 0.04);
        }
    }
    out
}

fn symmetric_scale_for_qmax(values: &[f32], qmax: i32) -> f32 {
    let max_abs = values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    if max_abs < 1e-12 {
        1.0
    } else {
        max_abs / qmax as f32
    }
}

fn quantize_symmetric_i8(values: &[f32], scale: f32) -> Vec<i8> {
    values
        .iter()
        .map(|value| ((value / scale).round() as i32).clamp(-127, 127) as i8)
        .collect()
}

fn cpu_accumulator_reference(inputs_q: &[i8], weights_q: &[i8], seq_len: usize) -> Vec<i32> {
    let mut out = vec![0i32; seq_len * DIM];
    for row in 0..seq_len {
        let in_row = &inputs_q[row * DIM..(row + 1) * DIM];
        let out_row = &mut out[row * DIM..(row + 1) * DIM];
        for (k, &in_q) in in_row.iter().enumerate() {
            let x = in_q as i32;
            if x == 0 {
                continue;
            }
            let w_row = &weights_q[k * DIM..(k + 1) * DIM];
            for col in 0..DIM {
                out_row[col] += x * (w_row[col] as i32);
            }
        }
    }
    out
}

fn fit_affine_map(acc_i32: &[i32], tpu_q: &[i8]) -> Result<AffineMap, Box<dyn Error>> {
    if acc_i32.len() != tpu_q.len() || acc_i32.is_empty() {
        return Err("fit_affine_map expects equal non-empty slices".into());
    }

    let n = acc_i32.len() as f64;
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for i in 0..acc_i32.len() {
        mean_x += acc_i32[i] as f64;
        mean_y += tpu_q[i] as f64;
    }
    mean_x /= n;
    mean_y /= n;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    for i in 0..acc_i32.len() {
        let dx = acc_i32[i] as f64 - mean_x;
        let dy = tpu_q[i] as f64 - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
    }

    let alpha = if var_x > 0.0 { cov / var_x } else { 0.0 };
    let beta = mean_y - alpha * mean_x;
    Ok(AffineMap { alpha, beta })
}

fn fit_affine_per_output(
    acc_i32: &[i32],
    tpu_q: &[i8],
    rows: usize,
) -> Result<Vec<AffineMap>, Box<dyn Error>> {
    if acc_i32.len() != tpu_q.len() {
        return Err("fit_affine_per_output size mismatch".into());
    }
    if acc_i32.len() != rows * DIM {
        return Err("fit_affine_per_output expects rows * DIM values".into());
    }
    if rows == 0 {
        return Err("fit_affine_per_output rows must be >= 1".into());
    }

    let mut maps = vec![
        AffineMap {
            alpha: 0.0,
            beta: 0.0
        };
        DIM
    ];
    for col in 0..DIM {
        let mut xs = Vec::with_capacity(rows);
        let mut ys = Vec::with_capacity(rows);
        for row in 0..rows {
            let idx = row * DIM + col;
            xs.push(acc_i32[idx]);
            ys.push(tpu_q[idx]);
        }
        maps[col] = fit_affine_map(&xs, &ys)?;
    }

    Ok(maps)
}

fn clamp_i8_from_f64(value: f64) -> i8 {
    (value.round() as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

fn verify_against_affine(acc_i32: &[i32], tpu_q: &[i8], map: AffineMap) -> VerifyStats {
    if acc_i32.len() != tpu_q.len() || acc_i32.is_empty() {
        return VerifyStats::default();
    }

    let n = acc_i32.len() as f64;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut max_abs_delta = 0i32;
    let mut mismatches_gt2 = 0usize;
    let mut mismatches_gt4 = 0usize;

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
        if abs_delta > 4 {
            mismatches_gt4 += 1;
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
        mismatches_gt4,
        corr,
    }
}

fn verify_against_per_output_affine(
    acc_i32: &[i32],
    tpu_q: &[i8],
    maps: &[AffineMap],
) -> VerifyStats {
    if acc_i32.len() != tpu_q.len() || acc_i32.is_empty() || maps.len() != DIM {
        return VerifyStats::default();
    }
    if !acc_i32.len().is_multiple_of(DIM) {
        return VerifyStats::default();
    }

    let rows = acc_i32.len() / DIM;
    let n = acc_i32.len() as f64;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut max_abs_delta = 0i32;
    let mut mismatches_gt2 = 0usize;
    let mut mismatches_gt4 = 0usize;

    let mut mean_pred = 0.0f64;
    let mut mean_tpu = 0.0f64;
    for row in 0..rows {
        for col in 0..DIM {
            let idx = row * DIM + col;
            let map = maps[col];
            let pred_q = clamp_i8_from_f64(map.alpha * acc_i32[idx] as f64 + map.beta);
            mean_pred += pred_q as f64;
            mean_tpu += tpu_q[idx] as f64;
        }
    }
    mean_pred /= n;
    mean_tpu /= n;

    let mut cov = 0.0f64;
    let mut var_pred = 0.0f64;
    let mut var_tpu = 0.0f64;
    for row in 0..rows {
        for col in 0..DIM {
            let idx = row * DIM + col;
            let map = maps[col];
            let pred_q = clamp_i8_from_f64(map.alpha * acc_i32[idx] as f64 + map.beta);
            let delta = pred_q as i32 - tpu_q[idx] as i32;
            let abs_delta = delta.abs();
            abs_sum += abs_delta as f64;
            sq_sum += (delta * delta) as f64;
            if abs_delta > max_abs_delta {
                max_abs_delta = abs_delta;
            }
            if abs_delta > 2 {
                mismatches_gt2 += 1;
            }
            if abs_delta > 4 {
                mismatches_gt4 += 1;
            }

            let dp = pred_q as f64 - mean_pred;
            let dt = tpu_q[idx] as f64 - mean_tpu;
            cov += dp * dt;
            var_pred += dp * dp;
            var_tpu += dt * dt;
        }
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
        mismatches_gt4,
        corr,
    }
}

fn checksum_i64(values: &[i8]) -> i64 {
    values.iter().map(|value| *value as i64).sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = parse_args()?;

    println!("EdgeTPU version: {}", version());
    println!("GEMM weight-load verifier (d_model=2304)");
    println!(
        "Config: seq_len={} runs={} warmup={} calibration_rows={} seed={} input_qmax={} weight_qmax={}",
        cli.seq_len,
        cli.runs,
        cli.warmup,
        cli.calibration_rows,
        cli.seed,
        cli.input_qmax,
        cli.weight_qmax
    );

    let weights_f32 = if let Some(path) = &cli.weights_f32_le_path {
        read_f32_le_file(path, WEIGHT_COUNT)?
    } else {
        generate_weights_f32(cli.seed)
    };
    let inputs_f32 = if let Some(path) = &cli.inputs_f32_le_path {
        read_f32_le_file(path, cli.seq_len * DIM)?
    } else {
        generate_inputs_f32(cli.seq_len, cli.seed)
    };

    let weight_scale = symmetric_scale_for_qmax(&weights_f32, cli.weight_qmax);
    let input_scale = symmetric_scale_for_qmax(&inputs_f32, cli.input_qmax);
    let weights_q = quantize_symmetric_i8(&weights_f32, weight_scale);
    let inputs_q = quantize_symmetric_i8(&inputs_f32, input_scale);

    println!(
        "Quantization scales: input_scale={:.8} weight_scale={:.8}",
        input_scale, weight_scale
    );

    let mut template = DenseGemmTemplate::from_bundled_2304()?;
    template.set_weights_from_slice(&weights_q)?;

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;
    let prepared = template.prepare(&delegate)?;

    for _ in 0..cli.warmup {
        let _ = prepared.execute_batch_rows(&inputs_q)?;
    }

    let mut tpu_output_q = Vec::new();
    let mut total_ms = 0.0f64;
    for run_idx in 0..cli.runs {
        let started = Instant::now();
        let current = prepared.execute_batch_rows(&inputs_q)?;
        total_ms += started.elapsed().as_secs_f64() * 1000.0;
        if run_idx + 1 == cli.runs {
            tpu_output_q = current;
        }
    }

    let avg_ms = total_ms / cli.runs as f64;
    let macs_per_run = (cli.seq_len as f64) * (DIM as f64) * (DIM as f64);
    let gmac_per_s = macs_per_run / (avg_ms * 1_000_000.0);
    println!(
        "EdgeTPU latency: avg_ms={:.3} total_ms={:.3} gmac_per_s={:.3}",
        avg_ms, total_ms, gmac_per_s
    );

    let started_cpu = Instant::now();
    let acc_i32 = cpu_accumulator_reference(&inputs_q, &weights_q, cli.seq_len);
    let cpu_ms = started_cpu.elapsed().as_secs_f64() * 1000.0;
    println!(
        "CPU reference accumulation: accum_count={} time_ms={:.3}",
        acc_i32.len(),
        cpu_ms
    );

    let cal_rows = cli.calibration_rows.min(cli.seq_len);
    let cal_count = cal_rows * DIM;
    let (acc_cal, acc_eval) = acc_i32.split_at(cal_count);
    let (tpu_cal, tpu_eval) = tpu_output_q.split_at(cal_count);
    let affine = fit_affine_map(acc_cal, tpu_cal)?;
    let affine_per_output = fit_affine_per_output(acc_cal, tpu_cal, cal_rows)?;

    let eval_acc = if acc_eval.is_empty() {
        acc_cal
    } else {
        acc_eval
    };
    let eval_tpu = if tpu_eval.is_empty() {
        tpu_cal
    } else {
        tpu_eval
    };
    let eval_stats_global = verify_against_affine(eval_acc, eval_tpu, affine);
    let all_stats_global = verify_against_affine(&acc_i32, &tpu_output_q, affine);
    let eval_stats_per_output =
        verify_against_per_output_affine(eval_acc, eval_tpu, &affine_per_output);
    let all_stats_per_output =
        verify_against_per_output_affine(&acc_i32, &tpu_output_q, &affine_per_output);

    println!(
        "Calibration: rows={} points={} alpha={:.10} beta={:.6}",
        cal_rows, cal_count, affine.alpha, affine.beta
    );
    println!(
        "Holdout verify (global affine): count={} mae={:.4} rmse={:.4} max_abs_delta={} mismatches_gt2={} mismatches_gt4={} corr={:.6}",
        eval_stats_global.count,
        eval_stats_global.mae,
        eval_stats_global.rmse,
        eval_stats_global.max_abs_delta,
        eval_stats_global.mismatches_gt2,
        eval_stats_global.mismatches_gt4,
        eval_stats_global.corr
    );
    println!(
        "Holdout verify (per-output affine): count={} mae={:.4} rmse={:.4} max_abs_delta={} mismatches_gt2={} mismatches_gt4={} corr={:.6}",
        eval_stats_per_output.count,
        eval_stats_per_output.mae,
        eval_stats_per_output.rmse,
        eval_stats_per_output.max_abs_delta,
        eval_stats_per_output.mismatches_gt2,
        eval_stats_per_output.mismatches_gt4,
        eval_stats_per_output.corr
    );
    println!(
        "All-points verify (global affine): count={} mae={:.4} rmse={:.4} max_abs_delta={} mismatches_gt2={} mismatches_gt4={} corr={:.6}",
        all_stats_global.count,
        all_stats_global.mae,
        all_stats_global.rmse,
        all_stats_global.max_abs_delta,
        all_stats_global.mismatches_gt2,
        all_stats_global.mismatches_gt4,
        all_stats_global.corr
    );
    println!(
        "All-points verify (per-output affine): count={} mae={:.4} rmse={:.4} max_abs_delta={} mismatches_gt2={} mismatches_gt4={} corr={:.6}",
        all_stats_per_output.count,
        all_stats_per_output.mae,
        all_stats_per_output.rmse,
        all_stats_per_output.max_abs_delta,
        all_stats_per_output.mismatches_gt2,
        all_stats_per_output.mismatches_gt4,
        all_stats_per_output.corr
    );
    println!(
        "Checksums: input_q={} weight_q={} tpu_output_q={}",
        checksum_i64(&inputs_q),
        checksum_i64(&weights_q),
        checksum_i64(&tpu_output_q)
    );

    Ok(())
}
