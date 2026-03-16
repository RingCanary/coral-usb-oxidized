#[path = "common/quant.rs"]
mod quant;

use quant::cpu_accumulator_reference_batch;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

type DynError = Box<dyn std::error::Error>;

struct Config {
    input_dim: usize,
    output_dim: usize,
    rows: usize,
    warmup: usize,
    repeats: usize,
    inputs_path: String,
    weights_path: String,
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --input-dim N --output-dim N --rows N --inputs-i8-file PATH --weights-row-major-i8-file PATH [--warmup N] [--repeats N]"
    )
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, DynError> {
    value
        .parse::<usize>()
        .map_err(|e| format!("invalid {} value '{}': {}", flag, value, e).into())
}

fn parse_args() -> Result<Config, DynError> {
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "cpu_gemm_baseline".to_string());

    let mut input_dim: Option<usize> = None;
    let mut output_dim: Option<usize> = None;
    let mut rows: Option<usize> = None;
    let mut warmup = 2usize;
    let mut repeats = 10usize;
    let mut inputs_path: Option<String> = None;
    let mut weights_path: Option<String> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input-dim" => {
                i += 1;
                input_dim = Some(parse_usize(
                    args.get(i).ok_or("--input-dim requires value")?,
                    "--input-dim",
                )?);
            }
            "--output-dim" => {
                i += 1;
                output_dim = Some(parse_usize(
                    args.get(i).ok_or("--output-dim requires value")?,
                    "--output-dim",
                )?);
            }
            "--rows" => {
                i += 1;
                rows = Some(parse_usize(
                    args.get(i).ok_or("--rows requires value")?,
                    "--rows",
                )?);
            }
            "--warmup" => {
                i += 1;
                warmup = parse_usize(args.get(i).ok_or("--warmup requires value")?, "--warmup")?;
            }
            "--repeats" => {
                i += 1;
                repeats = parse_usize(args.get(i).ok_or("--repeats requires value")?, "--repeats")?;
            }
            "--inputs-i8-file" => {
                i += 1;
                inputs_path = Some(
                    args.get(i)
                        .ok_or("--inputs-i8-file requires value")?
                        .to_string(),
                );
            }
            "--weights-row-major-i8-file" => {
                i += 1;
                weights_path = Some(
                    args.get(i)
                        .ok_or("--weights-row-major-i8-file requires value")?
                        .to_string(),
                );
            }
            "--help" | "-h" => {
                println!("{}", usage(&program));
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {}\n{}", other, usage(&program)).into()),
        }
        i += 1;
    }

    let input_dim = input_dim.ok_or_else(|| format!("missing --input-dim\n{}", usage(&program)))?;
    let output_dim =
        output_dim.ok_or_else(|| format!("missing --output-dim\n{}", usage(&program)))?;
    let rows = rows.ok_or_else(|| format!("missing --rows\n{}", usage(&program)))?;
    let inputs_path =
        inputs_path.ok_or_else(|| format!("missing --inputs-i8-file\n{}", usage(&program)))?;
    let weights_path = weights_path
        .ok_or_else(|| format!("missing --weights-row-major-i8-file\n{}", usage(&program)))?;

    if input_dim == 0 || output_dim == 0 || rows == 0 || repeats == 0 {
        return Err("input_dim/output_dim/rows/repeats must be >= 1".into());
    }

    Ok(Config {
        input_dim,
        output_dim,
        rows,
        warmup,
        repeats,
        inputs_path,
        weights_path,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn checksum_i64(values: &[i32]) -> i64 {
    values.iter().map(|&value| value as i64).sum()
}

fn load_i8_file(path: &str, expected_len: usize, label: &str) -> Result<Vec<i8>, DynError> {
    if !Path::new(path).is_file() {
        return Err(format!("{} path does not exist or is not a file: {}", label, path).into());
    }
    let bytes = fs::read(path)?;
    if bytes.len() != expected_len {
        return Err(format!(
            "{} length mismatch: expected {}, got {} ({})",
            label,
            expected_len,
            bytes.len(),
            path
        )
        .into());
    }
    Ok(bytes.into_iter().map(|byte| byte as i8).collect())
}

fn main() -> Result<(), DynError> {
    let config = parse_args()?;
    let expected_input_len = config
        .rows
        .checked_mul(config.input_dim)
        .ok_or("rows*input_dim overflow")?;
    let expected_weight_len = config
        .input_dim
        .checked_mul(config.output_dim)
        .ok_or("input_dim*output_dim overflow")?;

    let inputs = load_i8_file(&config.inputs_path, expected_input_len, "inputs")?;
    let weights = load_i8_file(
        &config.weights_path,
        expected_weight_len,
        "weights-row-major-i8-file",
    )?;

    println!(
        "CPU GEMM baseline: rows={} input_dim={} output_dim={} warmup={} repeats={}",
        config.rows, config.input_dim, config.output_dim, config.warmup, config.repeats
    );
    println!("Inputs: {}", config.inputs_path);
    println!("Weights: {}", config.weights_path);

    for _ in 0..config.warmup {
        let output = cpu_accumulator_reference_batch(
            &inputs,
            &weights,
            config.input_dim,
            config.output_dim,
        )?;
        black_box(checksum_i64(&output));
    }

    let mut repeat_ms = Vec::with_capacity(config.repeats);
    let mut last_checksum = 0i64;
    for repeat in 0..config.repeats {
        let started = Instant::now();
        let output = cpu_accumulator_reference_batch(
            &inputs,
            &weights,
            config.input_dim,
            config.output_dim,
        )?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        last_checksum = checksum_i64(&output);
        repeat_ms.push(elapsed_ms);
        println!(
            "Repeat {}: batch_ms={:.3} checksum_i64={}",
            repeat + 1,
            elapsed_ms,
            last_checksum
        );
    }

    let mut sorted = repeat_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean_ms = mean(&repeat_ms);
    let total_macs = (config.rows as f64) * (config.input_dim as f64) * (config.output_dim as f64);
    let effective_gmac_per_s = total_macs / (mean_ms * 1_000_000.0);
    println!(
        "Summary: repeats={} mean_ms={:.3} p50_ms={:.3} p95_ms={:.3} min_ms={:.3} max_ms={:.3} effective_gmac_per_s={:.3} checksum_i64={}",
        config.repeats,
        mean_ms,
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        sorted.first().copied().unwrap_or(0.0),
        sorted.last().copied().unwrap_or(0.0),
        effective_gmac_per_s,
        last_checksum
    );

    Ok(())
}
