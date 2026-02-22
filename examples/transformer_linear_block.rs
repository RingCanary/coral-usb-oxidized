use coral_usb_oxidized::{
    version, CoralDevice, DenseGemmError, DenseGemmTemplate, PreparedDenseGemm,
};
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

const D_MODEL: usize = 2304;
const STAGE_COUNT: usize = 6;
const WEIGHT_COUNT: usize = D_MODEL * D_MODEL;

#[derive(Clone, Copy, Debug)]
enum StageMode {
    Identity,
    ShiftPlus1,
    ShiftMinus1,
}

#[derive(Clone, Copy, Debug)]
struct StageSpec {
    name: &'static str,
    mode: StageMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WeightSource {
    Pattern,
    F32,
}

struct Config {
    seq_len: usize,
    runs: usize,
    warmup: usize,
    use_attention: bool,
    weight_source: WeightSource,
    weights_dir: Option<String>,
    input_f32_le_path: Option<String>,
    seed: u64,
    input_qmax: i32,
    weight_qmax: i32,
    verify_calibration_rows: usize,
}

const STAGES: [StageSpec; STAGE_COUNT] = [
    StageSpec {
        name: "q_proj",
        mode: StageMode::Identity,
    },
    StageSpec {
        name: "k_proj",
        mode: StageMode::ShiftPlus1,
    },
    StageSpec {
        name: "v_proj",
        mode: StageMode::ShiftMinus1,
    },
    StageSpec {
        name: "o_proj",
        mode: StageMode::Identity,
    },
    StageSpec {
        name: "mlp_up",
        mode: StageMode::ShiftPlus1,
    },
    StageSpec {
        name: "mlp_down",
        mode: StageMode::ShiftMinus1,
    },
];

struct PreparedStage {
    spec: StageSpec,
    prepared: PreparedDenseGemm,
    source_label: String,
    weight_scale: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct StageSetupTiming {
    prepare_ms: f64,
    first_invoke_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct RunTiming {
    q_ms: f64,
    k_ms: f64,
    v_ms: f64,
    attn_ms: f64,
    o_ms: f64,
    up_ms: f64,
    down_ms: f64,
    total_ms: f64,
}

impl RunTiming {
    fn add_assign(&mut self, rhs: &Self) {
        self.q_ms += rhs.q_ms;
        self.k_ms += rhs.k_ms;
        self.v_ms += rhs.v_ms;
        self.attn_ms += rhs.attn_ms;
        self.o_ms += rhs.o_ms;
        self.up_ms += rhs.up_ms;
        self.down_ms += rhs.down_ms;
        self.total_ms += rhs.total_ms;
    }

    fn linear_ms(&self) -> f64 {
        self.q_ms + self.k_ms + self.v_ms + self.o_ms + self.up_ms + self.down_ms
    }
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
        "Usage: {program} [seq_len] [runs] [warmup] [--no-attention] [--weight-source pattern|f32] [--weights-dir DIR] [--input-f32-le PATH] [--seed N] [--input-qmax N] [--weight-qmax N] [--verify-calibration-rows N]"
    );
    println!(
        "Defaults: seq_len=8 runs=5 warmup=1 weight_source=pattern seed=1 input_qmax=32 weight_qmax=16 verify_calibration_rows=4"
    );
}

fn parse_weight_source(value: &str) -> Result<WeightSource, Box<dyn Error>> {
    match value {
        "pattern" => Ok(WeightSource::Pattern),
        "f32" => Ok(WeightSource::F32),
        _ => Err(format!("invalid --weight-source '{}': expected pattern|f32", value).into()),
    }
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "transformer_linear_block".to_string());

    let mut positional: Vec<String> = Vec::new();
    let mut use_attention = true;
    let mut weight_source = WeightSource::Pattern;
    let mut weights_dir: Option<String> = None;
    let mut input_f32_le_path: Option<String> = None;
    let mut seed: u64 = 1;
    let mut input_qmax: i32 = 32;
    let mut weight_qmax: i32 = 16;
    let mut verify_calibration_rows: usize = 4;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                usage(&program);
                std::process::exit(0);
            }
            "--no-attention" => use_attention = false,
            "--weight-source" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--weight-source requires a value".into());
                }
                weight_source = parse_weight_source(&args[idx])?;
            }
            "--weights-dir" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--weights-dir requires a path".into());
                }
                weights_dir = Some(args[idx].clone());
            }
            "--input-f32-le" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--input-f32-le requires a path".into());
                }
                input_f32_le_path = Some(args[idx].clone());
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
            "--verify-calibration-rows" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--verify-calibration-rows requires a value".into());
                }
                verify_calibration_rows = args[idx].parse::<usize>()?;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {}", other).into());
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
        .unwrap_or(5);
    let warmup = positional
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    if seq_len == 0 {
        return Err("seq_len must be >= 1".into());
    }
    if runs == 0 {
        return Err("runs must be >= 1".into());
    }
    if verify_calibration_rows == 0 {
        return Err("verify_calibration_rows must be >= 1".into());
    }
    if !(1..=127).contains(&input_qmax) {
        return Err("input_qmax must be in [1, 127]".into());
    }
    if !(1..=127).contains(&weight_qmax) {
        return Err("weight_qmax must be in [1, 127]".into());
    }
    if weight_source == WeightSource::Pattern
        && (weights_dir.is_some() || input_f32_le_path.is_some())
    {
        return Err(
            "--weights-dir/--input-f32-le require --weight-source f32 (or omit them)".into(),
        );
    }

    Ok(Config {
        seq_len,
        runs,
        warmup,
        use_attention,
        weight_source,
        weights_dir,
        input_f32_le_path,
        seed,
        input_qmax,
        weight_qmax,
        verify_calibration_rows,
    })
}

fn apply_stage_mode(
    template: &mut DenseGemmTemplate,
    mode: StageMode,
) -> Result<(), DenseGemmError> {
    match mode {
        StageMode::Identity => template.set_identity(127),
        StageMode::ShiftPlus1 => template.set_shift_plus1(127),
        StageMode::ShiftMinus1 => template.set_shift_minus1(127),
    }
}

fn build_prefill_input_pattern(seq_len: usize, dim: usize) -> Result<Vec<i8>, Box<dyn Error>> {
    let total = seq_len.checked_mul(dim).ok_or("sequence size overflow")?;
    let mut out = vec![0i8; total];
    for row in 0..seq_len {
        for col in 0..dim {
            let value = ((row + col) & 0xff) as i16 - 128;
            out[row * dim + col] = value as i8;
        }
    }
    Ok(out)
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

fn quantize_symmetric_i8(values: &[f32], scale: f32, qmax: i32) -> Vec<i8> {
    values
        .iter()
        .map(|value| ((value / scale).round() as i32).clamp(-qmax, qmax) as i8)
        .collect()
}

fn generate_stage_weights_f32(stage_index: usize, spec: StageSpec, seed: u64) -> Vec<f32> {
    let stage_mix = (stage_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = XorShift64::new(seed ^ stage_mix);
    let mut out = vec![0.0f32; WEIGHT_COUNT];
    let stage_stride = 3 + stage_index;
    for row in 0..D_MODEL {
        for col in 0..D_MODEL {
            let idx = row * D_MODEL + col;
            let base = rng.next_f32(-0.12, 0.12);
            let skip = if (row + stage_stride * col) % 11 == 0 {
                0.18
            } else {
                0.0
            };
            // Keep a weak mode imprint so stage classes still differ structurally.
            let mode_bias = match spec.mode {
                StageMode::Identity => {
                    if row == col {
                        0.02
                    } else {
                        0.0
                    }
                }
                StageMode::ShiftPlus1 => {
                    if row == (col + 1) % D_MODEL {
                        0.02
                    } else {
                        0.0
                    }
                }
                StageMode::ShiftMinus1 => {
                    if row == (col + D_MODEL - 1) % D_MODEL {
                        0.02
                    } else {
                        0.0
                    }
                }
            };
            let value = base + skip + mode_bias;
            out[idx] = value;
        }
    }
    out
}

fn load_or_generate_stage_weights_f32(
    config: &Config,
    stage_index: usize,
    spec: StageSpec,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if let Some(dir) = &config.weights_dir {
        let path = Path::new(dir).join(format!("{}.f32le", spec.name));
        return read_f32_le_file(path.to_str().ok_or("invalid UTF-8 path")?, WEIGHT_COUNT);
    }
    Ok(generate_stage_weights_f32(stage_index, spec, config.seed))
}

fn build_prefill_input_f32(config: &Config) -> Result<(Vec<i8>, f32), Box<dyn Error>> {
    let expected = config
        .seq_len
        .checked_mul(D_MODEL)
        .ok_or("sequence size overflow")?;
    let input_f32 = if let Some(path) = &config.input_f32_le_path {
        read_f32_le_file(path, expected)?
    } else {
        let mut rng = XorShift64::new(config.seed ^ 0xBADC_0FFE_EE11_D00D);
        let mut out = vec![0.0f32; expected];
        for row in 0..config.seq_len {
            for col in 0..D_MODEL {
                let idx = row * D_MODEL + col;
                let periodic = (((row * 13 + col) & 63) as f32 - 31.5) / 96.0;
                out[idx] = periodic + rng.next_f32(-0.04, 0.04);
            }
        }
        out
    };

    let input_scale = symmetric_scale_for_qmax(&input_f32, config.input_qmax);
    let input_q = quantize_symmetric_i8(&input_f32, input_scale, config.input_qmax);
    Ok((input_q, input_scale))
}

fn prepare_stage(
    config: &Config,
    spec: StageSpec,
    stage_index: usize,
    delegate: &coral_usb_oxidized::EdgeTPUDelegate,
    probe_input: &[i8],
) -> Result<(PreparedStage, StageSetupTiming), Box<dyn Error>> {
    let mut template = DenseGemmTemplate::from_bundled_2304()?;
    let (source_label, weight_scale) = match config.weight_source {
        WeightSource::Pattern => {
            apply_stage_mode(&mut template, spec.mode)?;
            (format!("pattern:{:?}", spec.mode), None)
        }
        WeightSource::F32 => {
            let weights_f32 = load_or_generate_stage_weights_f32(config, stage_index, spec)?;
            let scale = symmetric_scale_for_qmax(&weights_f32, config.weight_qmax);
            let weights_q = quantize_symmetric_i8(&weights_f32, scale, config.weight_qmax);
            template.set_weights_from_slice(&weights_q)?;
            let source_label = if config.weights_dir.is_some() {
                "f32:file".to_string()
            } else {
                "f32:generated".to_string()
            };
            (source_label, Some(scale))
        }
    };

    let started_prepare = Instant::now();
    let prepared = template.prepare(delegate)?;
    let prepare_ms = started_prepare.elapsed().as_secs_f64() * 1000.0;

    let started_first_invoke = Instant::now();
    let _ = prepared.execute_batch_rows(probe_input)?;
    let first_invoke_ms = started_first_invoke.elapsed().as_secs_f64() * 1000.0;

    Ok((
        PreparedStage {
            spec,
            prepared,
            source_label,
            weight_scale,
        },
        StageSetupTiming {
            prepare_ms,
            first_invoke_ms,
        },
    ))
}

fn clamp_i8(value: i32) -> i8 {
    value.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

fn cpu_single_head_attention(
    q: &[i8],
    k: &[i8],
    v: &[i8],
    seq_len: usize,
    dim: usize,
) -> Result<Vec<i8>, Box<dyn Error>> {
    if q.len() != seq_len * dim || k.len() != seq_len * dim || v.len() != seq_len * dim {
        return Err("attention tensor size mismatch".into());
    }

    let scale = 1.0f32 / (dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];

    for t in 0..seq_len {
        let q_row = &q[t * dim..(t + 1) * dim];
        let score_row = &mut scores[t * seq_len..(t + 1) * seq_len];
        for s in 0..seq_len {
            let k_row = &k[s * dim..(s + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += (q_row[d] as f32) * (k_row[d] as f32);
            }
            score_row[s] = dot * scale;
        }

        let max_val = score_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for value in score_row.iter_mut() {
            *value = (*value - max_val).exp();
            denom += *value;
        }
        let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        for value in score_row.iter_mut() {
            *value *= inv;
        }
    }

    let mut out = vec![0i8; seq_len * dim];
    for t in 0..seq_len {
        let score_row = &scores[t * seq_len..(t + 1) * seq_len];
        let out_row = &mut out[t * dim..(t + 1) * dim];
        for d in 0..dim {
            let mut acc = 0.0f32;
            for s in 0..seq_len {
                let v_row = &v[s * dim..(s + 1) * dim];
                acc += score_row[s] * (v_row[d] as f32);
            }
            out_row[d] = clamp_i8(acc.round() as i32);
        }
    }

    Ok(out)
}

fn run_block(
    stages: &[PreparedStage],
    input_rows: &[i8],
    seq_len: usize,
    use_attention: bool,
) -> Result<(Vec<i8>, Vec<i8>, RunTiming), Box<dyn Error>> {
    if stages.len() != STAGE_COUNT {
        return Err("expected six prepared stages".into());
    }

    let started_total = Instant::now();

    let started = Instant::now();
    let q = stages[0].prepared.execute_batch_rows(input_rows)?;
    let q_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let k = stages[1].prepared.execute_batch_rows(input_rows)?;
    let k_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let v = stages[2].prepared.execute_batch_rows(input_rows)?;
    let v_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let attn_out = if use_attention {
        cpu_single_head_attention(&q, &k, &v, seq_len, D_MODEL)?
    } else {
        q.clone()
    };
    let attn_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let o = stages[3].prepared.execute_batch_rows(&attn_out)?;
    let o_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let up = stages[4].prepared.execute_batch_rows(&o)?;
    let up_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let down = stages[5].prepared.execute_batch_rows(&up)?;
    let down_ms = started.elapsed().as_secs_f64() * 1000.0;

    let timing = RunTiming {
        q_ms,
        k_ms,
        v_ms,
        attn_ms,
        o_ms,
        up_ms,
        down_ms,
        total_ms: started_total.elapsed().as_secs_f64() * 1000.0,
    };

    Ok((q, down, timing))
}

fn run_same_stage_baseline(
    stage: &PreparedDenseGemm,
    input_rows: &[i8],
) -> Result<(Vec<i8>, f64), Box<dyn Error>> {
    let started = Instant::now();
    let mut current = input_rows.to_vec();
    for _ in 0..STAGE_COUNT {
        current = stage.execute_batch_rows(&current)?;
    }
    Ok((current, started.elapsed().as_secs_f64() * 1000.0))
}

fn cpu_accumulator_reference(inputs_q: &[i8], weights_q: &[i8], seq_len: usize) -> Vec<i32> {
    let mut out = vec![0i32; seq_len * D_MODEL];
    for row in 0..seq_len {
        let in_row = &inputs_q[row * D_MODEL..(row + 1) * D_MODEL];
        let out_row = &mut out[row * D_MODEL..(row + 1) * D_MODEL];
        for (k, &in_q) in in_row.iter().enumerate() {
            let x = in_q as i32;
            if x == 0 {
                continue;
            }
            let w_row = &weights_q[k * D_MODEL..(k + 1) * D_MODEL];
            for col in 0..D_MODEL {
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

fn checksum_i64(values: &[i8]) -> i64 {
    values.iter().map(|value| *value as i64).sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let (input_rows, input_scale) = match config.weight_source {
        WeightSource::Pattern => (build_prefill_input_pattern(config.seq_len, D_MODEL)?, None),
        WeightSource::F32 => {
            let (q, scale) = build_prefill_input_f32(&config)?;
            (q, Some(scale))
        }
    };

    println!("EdgeTPU version: {}", version());
    println!("Transformer linear block benchmark (d_model={D_MODEL}, six 2304x2304 GEMMs)");
    println!(
        "Config: seq_len={} runs={} warmup={} attention={} weight_source={:?}",
        config.seq_len,
        config.runs,
        config.warmup,
        if config.use_attention {
            "cpu_single_head"
        } else {
            "disabled"
        },
        config.weight_source
    );
    if let Some(dir) = &config.weights_dir {
        println!("Weights directory: {dir}");
    }
    if let Some(scale) = input_scale {
        println!(
            "Input quantization: input_qmax={} input_scale={:.8}",
            config.input_qmax, scale
        );
    }

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;

    let mut prepared_stages: Vec<PreparedStage> = Vec::with_capacity(STAGE_COUNT);
    let mut setup_timings: Vec<StageSetupTiming> = Vec::with_capacity(STAGE_COUNT);
    for (stage_index, spec) in STAGES.iter().copied().enumerate() {
        let (prepared, timing) = prepare_stage(&config, spec, stage_index, &delegate, &input_rows)?;
        prepared_stages.push(prepared);
        setup_timings.push(timing);
    }

    println!("Stage setup timings (includes model-specific prepare and first invoke):");
    for (stage, timing) in prepared_stages.iter().zip(setup_timings.iter()) {
        let scale_text = stage
            .weight_scale
            .map(|value| format!("{value:.8}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "  {:>8}: prepare_ms={:>8.3} first_invoke_ms={:>8.3} source={} weight_scale={}",
            stage.spec.name,
            timing.prepare_ms,
            timing.first_invoke_ms,
            stage.source_label,
            scale_text
        );
    }

    for _ in 0..config.warmup {
        let _ = run_block(
            &prepared_stages,
            &input_rows,
            config.seq_len,
            config.use_attention,
        )?;
        let _ = run_same_stage_baseline(&prepared_stages[0].prepared, &input_rows)?;
    }

    let mut summed = RunTiming::default();
    let mut same_stage_total_ms = 0.0f64;
    let mut final_q_output: Vec<i8> = Vec::new();
    let mut final_output: Vec<i8> = Vec::new();
    let mut final_baseline: Vec<i8> = Vec::new();

    for run_idx in 0..config.runs {
        let (q_out, output, timing) = run_block(
            &prepared_stages,
            &input_rows,
            config.seq_len,
            config.use_attention,
        )?;
        let (baseline_out, baseline_ms) =
            run_same_stage_baseline(&prepared_stages[0].prepared, &input_rows)?;
        summed.add_assign(&timing);
        same_stage_total_ms += baseline_ms;
        if run_idx + 1 == config.runs {
            final_q_output = q_out;
            final_output = output;
            final_baseline = baseline_out;
        }
    }

    let inv_runs = 1.0 / config.runs as f64;
    let avg = RunTiming {
        q_ms: summed.q_ms * inv_runs,
        k_ms: summed.k_ms * inv_runs,
        v_ms: summed.v_ms * inv_runs,
        attn_ms: summed.attn_ms * inv_runs,
        o_ms: summed.o_ms * inv_runs,
        up_ms: summed.up_ms * inv_runs,
        down_ms: summed.down_ms * inv_runs,
        total_ms: summed.total_ms * inv_runs,
    };
    let same_stage_avg_ms = same_stage_total_ms * inv_runs;

    let linear_macs =
        (STAGE_COUNT as f64) * (config.seq_len as f64) * (D_MODEL as f64) * (D_MODEL as f64);
    let linear_avg_ms = avg.linear_ms();
    let linear_gmac_per_s = linear_macs / (linear_avg_ms * 1_000_000.0);
    let end_to_end_gmac_per_s = linear_macs / (avg.total_ms * 1_000_000.0);
    let switch_penalty_ms = linear_avg_ms - same_stage_avg_ms;

    println!("Average stage timings over {} run(s):", config.runs);
    println!(
        "  q={:.3}ms k={:.3}ms v={:.3}ms attn_cpu={:.3}ms o={:.3}ms up={:.3}ms down={:.3}ms",
        avg.q_ms, avg.k_ms, avg.v_ms, avg.attn_ms, avg.o_ms, avg.up_ms, avg.down_ms
    );
    println!(
        "  linear_only_ms={:.3} total_ms={:.3} same_stage6_ms={:.3}",
        linear_avg_ms, avg.total_ms, same_stage_avg_ms
    );
    println!(
        "  linear_gmac_per_s={:.3} end_to_end_gmac_per_s={:.3} switch_penalty_ms={:.3}",
        linear_gmac_per_s, end_to_end_gmac_per_s, switch_penalty_ms
    );
    println!(
        "Output checksums: q={} pipeline={} same_stage6={}",
        checksum_i64(&final_q_output),
        checksum_i64(&final_output),
        checksum_i64(&final_baseline)
    );

    if config.weight_source == WeightSource::F32 {
        let q_spec = STAGES[0];
        let q_weights_f32 = load_or_generate_stage_weights_f32(&config, 0, q_spec)?;
        let q_weight_scale = symmetric_scale_for_qmax(&q_weights_f32, config.weight_qmax);
        let q_weights_q = quantize_symmetric_i8(&q_weights_f32, q_weight_scale, config.weight_qmax);

        let acc_i32 = cpu_accumulator_reference(&input_rows, &q_weights_q, config.seq_len);
        let cal_rows = config.verify_calibration_rows.min(config.seq_len);
        let cal_count = cal_rows * D_MODEL;
        let (acc_cal, acc_eval) = acc_i32.split_at(cal_count);
        let (q_cal, q_eval) = final_q_output.split_at(cal_count);
        let affine = fit_affine_map(acc_cal, q_cal)?;

        let eval_acc = if acc_eval.is_empty() {
            acc_cal
        } else {
            acc_eval
        };
        let eval_q = if q_eval.is_empty() { q_cal } else { q_eval };
        let holdout = verify_against_affine(eval_acc, eval_q, affine);
        let all = verify_against_affine(&acc_i32, &final_q_output, affine);

        println!(
            "q_proj verification: calibration_rows={} points={} alpha={:.10} beta={:.6}",
            cal_rows, cal_count, affine.alpha, affine.beta
        );
        println!(
            "  holdout: count={} mae={:.4} rmse={:.4} max_abs_delta={} mismatches_gt2={} corr={:.6}",
            holdout.count,
            holdout.mae,
            holdout.rmse,
            holdout.max_abs_delta,
            holdout.mismatches_gt2,
            holdout.corr
        );
        println!(
            "  all: count={} mae={:.4} rmse={:.4} max_abs_delta={} mismatches_gt2={} corr={:.6}",
            all.count, all.mae, all.rmse, all.max_abs_delta, all.mismatches_gt2, all.corr
        );
    }

    Ok(())
}
