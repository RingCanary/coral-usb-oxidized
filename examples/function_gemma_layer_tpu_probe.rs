use coral_usb_oxidized::{
    quantize_linear_out_in_to_row_major_qi8_with_config, version, DenseGemmTemplate,
    FunctionGemmaLinearStage, FunctionGemmaSafeTensorFile, LinearQuantConfig,
};
use std::env;
use std::error::Error;
use std::time::Instant;

#[derive(Debug)]
struct AffineFit {
    alpha: f64,
    beta: f64,
    corr: f64,
    mae: f64,
    rmse: f64,
}

fn usage(program: &str) {
    println!(
        "Usage: {program} <model.safetensors> <template_edgetpu.tflite> <layer_idx> <stage> [runs] [qmax] [clip_percentile]"
    );
    println!("Stages: q|k|v|o|gate|up|down");
    println!("Defaults: runs=20 qmax=32 clip_percentile=100");
}

fn build_input(input_dim: usize) -> Vec<i8> {
    let mut out = vec![0i8; input_dim];
    for (idx, value) in out.iter_mut().enumerate() {
        let centered = (idx as i32 % 129) - 64;
        *value = centered as i8;
    }
    out
}

fn cpu_accumulator_reference(
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

fn fit_affine(x: &[i32], y: &[i8]) -> AffineFit {
    let n = x.len().max(1) as f64;
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

    AffineFit {
        alpha,
        beta,
        corr,
        mae: sum_abs / n,
        rmse: (sum_sq / n).sqrt(),
    }
}

fn preview(label: &str, data: &[i8], count: usize) {
    let shown = count.min(data.len());
    let joined = data
        .iter()
        .take(shown)
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("{label} (first {shown}): {joined}");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "function_gemma_layer_tpu_probe".to_string());

    if args.len() < 5 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage(&program);
        if args.len() < 5 {
            std::process::exit(2);
        }
        return Ok(());
    }

    let safetensors_path = &args[1];
    let template_path = &args[2];
    let layer_idx = args[3].parse::<usize>()?;
    let stage = FunctionGemmaLinearStage::parse(&args[4])?;
    let runs = args
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    let qmax = args
        .get(6)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(32);
    let clip_percentile = args
        .get(7)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(100.0);

    if runs == 0 {
        return Err("runs must be >= 1".into());
    }

    let model = FunctionGemmaSafeTensorFile::load(safetensors_path)?;
    let dims = model.infer_layer_dims(layer_idx)?;
    let stage_meta = model
        .layer_stage_metas(layer_idx)?
        .into_iter()
        .find(|meta| meta.stage == stage)
        .ok_or("stage metadata not found")?;

    let weights_f32 = model.tensor_f32(&stage_meta.tensor_name)?;
    let (weights_q, quant_info) = quantize_linear_out_in_to_row_major_qi8_with_config(
        &weights_f32,
        stage_meta.input_dim,
        stage_meta.output_dim,
        LinearQuantConfig {
            qmax,
            clip_percentile,
        },
    )?;

    println!("EdgeTPU version: {}", version());
    println!("SafeTensors: {}", safetensors_path);
    println!("Template: {}", template_path);
    println!(
        "Layer: {} Stage: {} Tensor: {}",
        layer_idx, stage, stage_meta.tensor_name
    );
    println!(
        "Model dims: hidden={} q_out={} kv_out={} mlp_hidden={}",
        dims.hidden_size, dims.q_proj_out, dims.kv_proj_out, dims.mlp_hidden
    );
    println!(
        "Stage dims: {}x{} runs={} qmax={} clip_percentile={:.2}",
        stage_meta.input_dim, stage_meta.output_dim, runs, qmax, clip_percentile
    );
    println!(
        "Weight quant: scale={:.9} max_abs={:.9} clipped_max_abs={:.9} clipped_values={}",
        quant_info.scale, quant_info.max_abs, quant_info.clipped_max_abs, quant_info.clipped_values
    );

    let mut template = DenseGemmTemplate::from_file_with_dims(
        template_path,
        stage_meta.input_dim,
        stage_meta.output_dim,
    )?;
    template.set_weights_from_slice(&weights_q)?;
    let prepared = template.prepare_with_new_delegate()?;

    let input = build_input(stage_meta.input_dim);
    let mut output = vec![0i8; stage_meta.output_dim];
    let mut total_ms = 0.0f64;
    for run_idx in 0..runs {
        let started = Instant::now();
        let current = prepared.execute(&input)?;
        total_ms += started.elapsed().as_secs_f64() * 1000.0;
        if run_idx + 1 == runs {
            output = current;
        }
    }

    let cpu_acc = cpu_accumulator_reference(&input, &weights_q, stage_meta.output_dim);
    let fit = fit_affine(&cpu_acc, &output);

    println!(
        "Latency: avg_ms={:.3} total_ms={:.3}",
        total_ms / runs as f64,
        total_ms
    );
    preview("Input", &input, 32);
    preview("Output", &output, 32);
    println!(
        "CPU accumulator vs TPU output affine: alpha={:.9} beta={:.9} corr={:.6} mae={:.4} rmse={:.4}",
        fit.alpha, fit.beta, fit.corr, fit.mae, fit.rmse
    );

    Ok(())
}
