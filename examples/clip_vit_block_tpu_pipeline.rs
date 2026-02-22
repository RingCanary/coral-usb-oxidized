use coral_usb_oxidized::{
    quantize_linear_out_in_to_row_major_qi8, version, ClipSafeTensorFile, ClipVitB32Dims,
    ClipVitLinearStageMeta, CoralDevice, DenseGemmTemplate, PreparedDenseGemm, QuantizationInfo,
};
use std::env;
use std::error::Error;
use std::fs;
use std::time::Instant;

const STAGE_COUNT: usize = 6;

struct Config {
    safetensors_path: String,
    template_768x768: String,
    template_768x3072: String,
    template_3072x768: String,
    layer_idx: usize,
    rows: usize,
    runs: usize,
    warmup: usize,
    qmax: i32,
    seed: u64,
    input_q_path: Option<String>,
}

struct PreparedStage {
    meta: ClipVitLinearStageMeta,
    template_path: String,
    weights_q: Vec<i8>,
    quant: QuantizationInfo,
    prepared: PreparedDenseGemm,
    prepare_ms: f64,
}

struct PipelineRun {
    stage_ms: Vec<f64>,
    total_ms: f64,
    final_output: Vec<i8>,
    stage_inputs: Vec<Vec<i8>>,
    stage_outputs: Vec<Vec<i8>>,
}

#[derive(Clone, Copy, Debug)]
struct AffineFit {
    alpha: f64,
    beta: f64,
    corr: f64,
    mae: f64,
    rmse: f64,
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
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
}

fn usage(program: &str) {
    println!(
        "Usage: {program} <model.safetensors> <template_768x768.tflite> <template_768x3072.tflite> <template_3072x768.tflite> [layer_idx] [rows] [runs] [warmup] [qmax] [--input-q PATH] [--seed N]"
    );
    println!(
        "Defaults: layer_idx=0 rows=8 runs=3 warmup=1 qmax=127 seed=1 (input defaults to synthetic int8 rows)"
    );
    println!(
        "Example: cargo run --example clip_vit_block_tpu_pipeline -- model.safetensors t768x768_edgetpu.tflite t768x3072_edgetpu.tflite t3072x768_edgetpu.tflite 0 8 3 1 127"
    );
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "clip_vit_block_tpu_pipeline".to_string());

    if args.len() <= 1 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage(&program);
        if args.len() <= 1 {
            std::process::exit(2);
        }
        std::process::exit(0);
    }

    let mut positional = Vec::new();
    let mut input_q_path: Option<String> = None;
    let mut seed: u64 = 1;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--input-q" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--input-q requires a path".into());
                }
                input_q_path = Some(args[idx].clone());
            }
            "--seed" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--seed requires a value".into());
                }
                seed = args[idx].parse::<u64>()?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option: {}", value).into());
            }
            value => positional.push(value.to_string()),
        }
        idx += 1;
    }

    if positional.len() < 4 {
        usage(&program);
        return Err("expected at least 4 positional args".into());
    }

    let layer_idx = positional
        .get(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let rows = positional
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let runs = positional
        .get(6)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let warmup = positional
        .get(7)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let qmax = positional
        .get(8)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(127);

    if rows == 0 {
        return Err("rows must be >= 1".into());
    }
    if runs == 0 {
        return Err("runs must be >= 1".into());
    }
    if !(1..=127).contains(&qmax) {
        return Err("qmax must be in [1, 127]".into());
    }

    Ok(Config {
        safetensors_path: positional[0].clone(),
        template_768x768: positional[1].clone(),
        template_768x3072: positional[2].clone(),
        template_3072x768: positional[3].clone(),
        layer_idx,
        rows,
        runs,
        warmup,
        qmax,
        seed,
        input_q_path,
    })
}

fn template_path_for_dims<'a>(
    config: &'a Config,
    input_dim: usize,
    output_dim: usize,
) -> Result<&'a str, Box<dyn Error>> {
    match (input_dim, output_dim) {
        (768, 768) => Ok(config.template_768x768.as_str()),
        (768, 3072) => Ok(config.template_768x3072.as_str()),
        (3072, 768) => Ok(config.template_3072x768.as_str()),
        _ => Err(format!(
            "unsupported stage dims {}x{} (expected one of 768x768, 768x3072, 3072x768)",
            input_dim, output_dim
        )
        .into()),
    }
}

fn load_input_rows(path: &str, rows: usize, input_dim: usize) -> Result<Vec<i8>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let expected = rows
        .checked_mul(input_dim)
        .ok_or("input size overflow for rows*input_dim")?;
    if bytes.len() != expected {
        return Err(format!(
            "input bytes mismatch for {}: expected {}, got {}",
            path,
            expected,
            bytes.len()
        )
        .into());
    }

    Ok(bytes.into_iter().map(|value| value as i8).collect())
}

fn build_synthetic_input(rows: usize, input_dim: usize, seed: u64) -> Vec<i8> {
    let mut rng = XorShift64::new(seed ^ 0xA5A5_5A5A_1234_5678);
    let mut out = vec![0i8; rows * input_dim];

    for row in 0..rows {
        for col in 0..input_dim {
            let idx = row * input_dim + col;
            let periodic = ((row * 29 + col * 7) % 129) as i32 - 64;
            let jitter = ((rng.next_u64() & 0x07) as i32) - 3;
            out[idx] = (periodic + jitter).clamp(-127, 127) as i8;
        }
    }

    out
}

fn cpu_accumulator_reference_batch(
    input_rows_q: &[i8],
    weights_row_major_q: &[i8],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<i32>, Box<dyn Error>> {
    if input_rows_q.len() % input_dim != 0 {
        return Err(format!(
            "input row-major length {} is not divisible by input_dim {}",
            input_rows_q.len(),
            input_dim
        )
        .into());
    }

    let expected_weights = input_dim
        .checked_mul(output_dim)
        .ok_or("weight dimension multiplication overflow")?;
    if weights_row_major_q.len() != expected_weights {
        return Err(format!(
            "weight length mismatch: expected {}, got {}",
            expected_weights,
            weights_row_major_q.len()
        )
        .into());
    }

    let rows = input_rows_q.len() / input_dim;
    let mut out = vec![0i32; rows * output_dim];

    for row in 0..rows {
        let input_row = &input_rows_q[row * input_dim..(row + 1) * input_dim];
        let out_row = &mut out[row * output_dim..(row + 1) * output_dim];
        for (in_idx, &x_q) in input_row.iter().enumerate() {
            let x = x_q as i32;
            if x == 0 {
                continue;
            }
            let weight_row = &weights_row_major_q[in_idx * output_dim..(in_idx + 1) * output_dim];
            for out_idx in 0..output_dim {
                out_row[out_idx] += x * weight_row[out_idx] as i32;
            }
        }
    }

    Ok(out)
}

fn fit_affine(cpu_acc: &[i32], tpu_output_q: &[i8]) -> Result<AffineFit, Box<dyn Error>> {
    if cpu_acc.len() != tpu_output_q.len() || cpu_acc.is_empty() {
        return Err("fit_affine expects equal non-empty slices".into());
    }

    let n = cpu_acc.len() as f64;
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for idx in 0..cpu_acc.len() {
        mean_x += cpu_acc[idx] as f64;
        mean_y += tpu_output_q[idx] as f64;
    }
    mean_x /= n;
    mean_y /= n;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    for idx in 0..cpu_acc.len() {
        let dx = cpu_acc[idx] as f64 - mean_x;
        let dy = tpu_output_q[idx] as f64 - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let alpha = if var_x > 0.0 { cov / var_x } else { 0.0 };
    let beta = mean_y - alpha * mean_x;
    let corr = if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    };

    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    for idx in 0..cpu_acc.len() {
        let pred = alpha * cpu_acc[idx] as f64 + beta;
        let err = tpu_output_q[idx] as f64 - pred;
        abs_sum += err.abs();
        sq_sum += err * err;
    }

    Ok(AffineFit {
        alpha,
        beta,
        corr,
        mae: abs_sum / n,
        rmse: (sq_sum / n).sqrt(),
    })
}

fn execute_pipeline_once(
    stages: &[PreparedStage],
    input_rows_q: &[i8],
    capture_trace: bool,
) -> Result<PipelineRun, Box<dyn Error>> {
    let mut stage_ms = vec![0.0f64; stages.len()];
    let mut stage_inputs = Vec::new();
    let mut stage_outputs = Vec::new();

    let started_total = Instant::now();
    let mut current = input_rows_q.to_vec();

    for (stage_idx, stage) in stages.iter().enumerate() {
        if capture_trace {
            stage_inputs.push(current.clone());
        }

        let started = Instant::now();
        let output = stage.prepared.execute_batch_rows(&current)?;
        stage_ms[stage_idx] = started.elapsed().as_secs_f64() * 1000.0;

        if capture_trace {
            stage_outputs.push(output.clone());
        }

        current = output;
    }

    Ok(PipelineRun {
        stage_ms,
        total_ms: started_total.elapsed().as_secs_f64() * 1000.0,
        final_output: current,
        stage_inputs,
        stage_outputs,
    })
}

fn checksum_i64(values: &[i8]) -> i64 {
    values.iter().map(|value| *value as i64).sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let dims = ClipVitB32Dims::default();

    let model = ClipSafeTensorFile::load(&config.safetensors_path)?;
    let stage_metas = model.clip_vit_layer_stage_metas(config.layer_idx, dims)?;

    let input_rows_q = if let Some(path) = &config.input_q_path {
        load_input_rows(path, config.rows, dims.d_model)?
    } else {
        build_synthetic_input(config.rows, dims.d_model, config.seed)
    };

    println!("EdgeTPU version: {}", version());
    println!("CLIP ViT-B/32 linear block TPU pipeline");
    println!(
        "Config: layer={} rows={} runs={} warmup={} qmax={} input_source={}",
        config.layer_idx,
        config.rows,
        config.runs,
        config.warmup,
        config.qmax,
        config.input_q_path.as_deref().unwrap_or("synthetic(seed)")
    );
    println!(
        "Templates: 768x768={} 768x3072={} 3072x768={}",
        config.template_768x768, config.template_768x3072, config.template_3072x768
    );

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;

    let mut prepared_stages: Vec<PreparedStage> = Vec::with_capacity(STAGE_COUNT);
    for meta in stage_metas {
        let weights_f32 = model.tensor_f32(&meta.tensor_name)?;
        let (weights_q, quant) = quantize_linear_out_in_to_row_major_qi8(
            &weights_f32,
            meta.input_dim,
            meta.output_dim,
            config.qmax,
        )?;

        let template_path = template_path_for_dims(&config, meta.input_dim, meta.output_dim)?;
        let mut template =
            DenseGemmTemplate::from_file_with_dims(template_path, meta.input_dim, meta.output_dim)?;
        template.set_weights_from_slice(&weights_q)?;

        let started_prepare = Instant::now();
        let prepared = template.prepare(&delegate)?;
        let prepare_ms = started_prepare.elapsed().as_secs_f64() * 1000.0;

        prepared_stages.push(PreparedStage {
            meta,
            template_path: template_path.to_string(),
            weights_q,
            quant,
            prepared,
            prepare_ms,
        });
    }

    println!("Stage setup:");
    for stage in &prepared_stages {
        println!(
            "  {:>4} dims={:>4}x{:<4} q_bytes={} scale={:.9} max_abs={:.9} prepare_ms={:.3} tensor={} template={}",
            stage.meta.stage,
            stage.meta.input_dim,
            stage.meta.output_dim,
            stage.weights_q.len(),
            stage.quant.scale,
            stage.quant.max_abs,
            stage.prepare_ms,
            stage.meta.tensor_name,
            stage.template_path
        );
    }

    for _ in 0..config.warmup {
        let _ = execute_pipeline_once(&prepared_stages, &input_rows_q, false)?;
    }

    let mut stage_totals_ms = vec![0.0f64; prepared_stages.len()];
    let mut total_ms = 0.0f64;
    let mut final_run: Option<PipelineRun> = None;

    for run_idx in 0..config.runs {
        let capture_trace = run_idx + 1 == config.runs;
        let run = execute_pipeline_once(&prepared_stages, &input_rows_q, capture_trace)?;
        for (stage_idx, elapsed) in run.stage_ms.iter().enumerate() {
            stage_totals_ms[stage_idx] += elapsed;
        }
        total_ms += run.total_ms;
        if capture_trace {
            final_run = Some(run);
        }
    }

    let final_run = final_run.ok_or("internal error: final run trace missing")?;

    println!("Stage timing (avg over {} measured run(s)):", config.runs);
    for (idx, stage) in prepared_stages.iter().enumerate() {
        println!(
            "  {:>4}: avg_ms={:.3}",
            stage.meta.stage,
            stage_totals_ms[idx] / config.runs as f64
        );
    }
    println!(
        "Pipeline timing: avg_total_ms={:.3} total_ms={:.3}",
        total_ms / config.runs as f64,
        total_ms
    );

    println!("Affine CPU accumulator vs TPU output (final measured run):");
    for (idx, stage) in prepared_stages.iter().enumerate() {
        let stage_input = &final_run.stage_inputs[idx];
        let stage_output = &final_run.stage_outputs[idx];
        let cpu_acc = cpu_accumulator_reference_batch(
            stage_input,
            &stage.weights_q,
            stage.meta.input_dim,
            stage.meta.output_dim,
        )?;
        let fit = fit_affine(&cpu_acc, stage_output)?;
        println!(
            "  {:>4}: alpha={:.9} beta={:.9} corr={:.6} mae={:.4} rmse={:.4} stage_ms={:.3}",
            stage.meta.stage,
            fit.alpha,
            fit.beta,
            fit.corr,
            fit.mae,
            fit.rmse,
            final_run.stage_ms[idx]
        );
    }

    println!(
        "Checksums: input_q={} final_output_q={}",
        checksum_i64(&input_rows_q),
        checksum_i64(&final_run.final_output)
    );

    Ok(())
}
