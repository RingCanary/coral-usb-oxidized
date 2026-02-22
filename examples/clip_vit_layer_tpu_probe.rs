use coral_usb_oxidized::{
    quantize_linear_out_in_to_row_major_qi8, version, ClipSafeTensorFile, ClipVitB32Dims,
    DenseGemmTemplate,
};
use std::env;
use std::error::Error;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum Stage {
    Q,
    K,
    V,
    O,
    Fc1,
    Fc2,
}

impl Stage {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "q" => Ok(Self::Q),
            "k" => Ok(Self::K),
            "v" => Ok(Self::V),
            "o" => Ok(Self::O),
            "fc1" => Ok(Self::Fc1),
            "fc2" => Ok(Self::Fc2),
            _ => Err(format!("unknown stage: {value} (expected q|k|v|o|fc1|fc2)").into()),
        }
    }

    fn dims(self, d_model: usize, mlp_hidden: usize) -> (usize, usize) {
        match self {
            Self::Q | Self::K | Self::V | Self::O => (d_model, d_model),
            Self::Fc1 => (d_model, mlp_hidden),
            Self::Fc2 => (mlp_hidden, d_model),
        }
    }
}

fn usage(program: &str) {
    println!(
        "Usage: {program} <model.safetensors> <template_edgetpu.tflite> <layer_idx> <stage> [runs] [qmax]"
    );
    println!("Defaults: runs=20 qmax=127");
}

fn build_input(input_dim: usize) -> Vec<i8> {
    let mut out = vec![0i8; input_dim];
    for (idx, value) in out.iter_mut().enumerate() {
        // Bounded signed ramp to avoid immediate saturating accumulators.
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
        let row = &weights_row_major_q[in_idx * output_dim..(in_idx + 1) * output_dim];
        for out_idx in 0..output_dim {
            out[out_idx] += x * row[out_idx] as i32;
        }
    }
    out
}

#[derive(Debug)]
struct AffineFit {
    alpha: f64,
    beta: f64,
    corr: f64,
    mae: f64,
    rmse: f64,
}

fn fit_affine(x: &[i32], y: &[i8]) -> AffineFit {
    let n = x.len().max(1) as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for i in 0..x.len() {
        sum_x += x[i] as f64;
        sum_y += y[i] as f64;
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    let mut cov = 0.0f64;
    for i in 0..x.len() {
        let dx = x[i] as f64 - mean_x;
        let dy = y[i] as f64 - mean_y;
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
    for i in 0..x.len() {
        let pred = alpha * x[i] as f64 + beta;
        let err = y[i] as f64 - pred;
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
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("{label} (first {shown}): {joined}");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "clip_vit_layer_tpu_probe".to_string());

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
    let stage = Stage::parse(&args[4])?;
    let runs = args
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    let qmax = args
        .get(6)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(127);
    if runs == 0 {
        return Err("runs must be >= 1".into());
    }

    let dims = ClipVitB32Dims::default();
    let (input_dim, output_dim) = stage.dims(dims.d_model, dims.mlp_hidden);

    let model = ClipSafeTensorFile::load(safetensors_path)?;
    let names = model.validate_clip_vit_layer_linears(layer_idx, dims)?;
    let tensor_name = match stage {
        Stage::Q => names.q_proj,
        Stage::K => names.k_proj,
        Stage::V => names.v_proj,
        Stage::O => names.o_proj,
        Stage::Fc1 => names.mlp_fc1,
        Stage::Fc2 => names.mlp_fc2,
    };

    let weights_f32 = model.tensor_f32(&tensor_name)?;
    let (weights_q, quant) =
        quantize_linear_out_in_to_row_major_qi8(&weights_f32, input_dim, output_dim, qmax)?;

    println!("EdgeTPU version: {}", version());
    println!("SafeTensors: {}", safetensors_path);
    println!("Template: {}", template_path);
    println!(
        "Layer: {} Stage: {:?} Tensor: {}",
        layer_idx, stage, tensor_name
    );
    println!(
        "Dims: {}x{} Runs: {} qmax={} weight_scale={:.9} weight_max_abs={:.9}",
        input_dim, output_dim, runs, qmax, quant.scale, quant.max_abs
    );

    let mut template =
        DenseGemmTemplate::from_file_with_dims(template_path, input_dim, output_dim)?;
    template.set_weights_from_slice(&weights_q)?;
    let prepared = template.prepare_with_new_delegate()?;

    let input = build_input(input_dim);
    let mut output = vec![0i8; output_dim];
    let mut total_ms = 0.0f64;
    for run_idx in 0..runs {
        let started = Instant::now();
        let current = prepared.execute(&input)?;
        total_ms += started.elapsed().as_secs_f64() * 1000.0;
        if run_idx + 1 == runs {
            output = current;
        }
    }

    let cpu_acc = cpu_accumulator_reference(&input, &weights_q, output_dim);
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
