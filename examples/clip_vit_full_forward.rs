use coral_usb_oxidized::{
    quantize_linear_out_in_to_row_major_qi8_with_config, version, ClipSafeTensorFile,
    ClipVitLayerLinearNames, CoralDevice, DenseGemmTemplate, LinearQuantConfig, PreparedDenseGemm,
    QuantizationInfo,
};
use std::env;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::time::Instant;

const IMAGE_H: usize = 224;
const IMAGE_W: usize = 224;
const IMAGE_C: usize = 3;
const PATCH: usize = 32;
const GRID: usize = IMAGE_H / PATCH;
const PATCH_COUNT: usize = GRID * GRID;
const TOKENS: usize = PATCH_COUNT + 1;
const D_MODEL: usize = 768;
const MLP_HIDDEN: usize = 3072;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const PROJ_DIM: usize = 512;
const LAYERS: usize = 12;
const LN_EPS: f32 = 1e-5;

struct Config {
    model_path: String,
    template_768x768: String,
    template_768x3072: String,
    template_3072x768: String,
    image_f32le_path: Option<String>,
    output_f32le_path: Option<String>,
    output_norm_f32le_path: Option<String>,
    reference_f32le_path: Option<String>,
    weight_qmax: i32,
    act_qmax: i32,
    clip_percentile: f32,
    calibration_rows: usize,
    max_layers: usize,
    seed: u64,
}

#[derive(Clone)]
struct LayerNormParams {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct VisionParams {
    patch_weight: Vec<f32>,
    class_embedding: Vec<f32>,
    pos_embedding: Vec<f32>,
    pre_ln: LayerNormParams,
    post_ln: LayerNormParams,
    visual_projection: Vec<f32>,
}

struct CoralLinearStage {
    prepared: PreparedDenseGemm,
    input_dim: usize,
    output_dim: usize,
    weight_scale: f32,
    bias: Vec<f32>,
    affine_alpha: f64,
    affine_beta: f64,
    fit_corr: f64,
    fit_rmse: f64,
    quant_info: QuantizationInfo,
}

struct TransformerBlock {
    ln1: LayerNormParams,
    ln2: LayerNormParams,
    q: CoralLinearStage,
    k: CoralLinearStage,
    v: CoralLinearStage,
    o: CoralLinearStage,
    fc1: CoralLinearStage,
    fc2: CoralLinearStage,
}

#[derive(Clone, Copy)]
struct AffineFit {
    alpha: f64,
    beta: f64,
    corr: f64,
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

    fn next_f32(&mut self, low: f32, high: f32) -> f32 {
        let unit = ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        low + (high - low) * unit as f32
    }
}

fn usage(program: &str) {
    println!(
        "Usage: {program} <model.safetensors> <template_768x768.tflite> <template_768x3072.tflite> <template_3072x768.tflite> [--image-f32le PATH] [--out-f32le PATH] [--out-norm-f32le PATH] [--reference-f32le PATH] [--weight-qmax N] [--act-qmax N] [--clip-percentile P] [--calibration-rows N] [--max-layers N] [--seed N]"
    );
    println!(
        "Defaults: weight-qmax=32 act-qmax=32 clip-percentile=100 calibration-rows=8 max-layers=12 seed=1"
    );
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "clip_vit_full_forward".to_string());

    if args.len() <= 1 || args.iter().any(|value| value == "--help" || value == "-h") {
        usage(&program);
        if args.len() <= 1 {
            std::process::exit(2);
        }
        std::process::exit(0);
    }

    let mut positional = Vec::new();
    let mut image_f32le_path = None;
    let mut output_f32le_path = None;
    let mut output_norm_f32le_path = None;
    let mut reference_f32le_path = None;
    let mut weight_qmax = 32i32;
    let mut act_qmax = 32i32;
    let mut clip_percentile = 100.0f32;
    let mut calibration_rows = 8usize;
    let mut max_layers = LAYERS;
    let mut seed = 1u64;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--image-f32le" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--image-f32le requires a path".into());
                }
                image_f32le_path = Some(args[idx].clone());
            }
            "--out-f32le" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--out-f32le requires a path".into());
                }
                output_f32le_path = Some(args[idx].clone());
            }
            "--out-norm-f32le" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--out-norm-f32le requires a path".into());
                }
                output_norm_f32le_path = Some(args[idx].clone());
            }
            "--reference-f32le" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--reference-f32le requires a path".into());
                }
                reference_f32le_path = Some(args[idx].clone());
            }
            "--weight-qmax" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--weight-qmax requires a value".into());
                }
                weight_qmax = args[idx].parse::<i32>()?;
            }
            "--act-qmax" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--act-qmax requires a value".into());
                }
                act_qmax = args[idx].parse::<i32>()?;
            }
            "--clip-percentile" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--clip-percentile requires a value".into());
                }
                clip_percentile = args[idx].parse::<f32>()?;
            }
            "--calibration-rows" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--calibration-rows requires a value".into());
                }
                calibration_rows = args[idx].parse::<usize>()?;
            }
            "--max-layers" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--max-layers requires a value".into());
                }
                max_layers = args[idx].parse::<usize>()?;
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

    if !(1..=127).contains(&weight_qmax) {
        return Err("weight-qmax must be in [1, 127]".into());
    }
    if !(1..=127).contains(&act_qmax) {
        return Err("act-qmax must be in [1, 127]".into());
    }
    if !(0.0..=100.0).contains(&clip_percentile) || clip_percentile == 0.0 {
        return Err("clip-percentile must be in (0, 100]".into());
    }
    if calibration_rows == 0 {
        return Err("calibration-rows must be >= 1".into());
    }
    if max_layers == 0 || max_layers > LAYERS {
        return Err(format!("max-layers must be in [1, {}]", LAYERS).into());
    }

    Ok(Config {
        model_path: positional[0].clone(),
        template_768x768: positional[1].clone(),
        template_768x3072: positional[2].clone(),
        template_3072x768: positional[3].clone(),
        image_f32le_path,
        output_f32le_path,
        output_norm_f32le_path,
        reference_f32le_path,
        weight_qmax,
        act_qmax,
        clip_percentile,
        calibration_rows,
        max_layers,
        seed,
    })
}

fn read_f32_le_file(path: &str, expected_count: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let expected_bytes = expected_count.checked_mul(4).ok_or("byte count overflow")?;
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

fn write_f32_le_file(path: &str, values: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    let mut file = fs::File::create(path)?;
    file.write_all(&out)?;
    Ok(())
}

fn load_or_generate_image(config: &Config) -> Result<Vec<f32>, Box<dyn Error>> {
    let expected = IMAGE_C * IMAGE_H * IMAGE_W;
    if let Some(path) = &config.image_f32le_path {
        return read_f32_le_file(path, expected);
    }

    let mut rng = XorShift64::new(config.seed ^ 0xD00D_BEEF_CAFE_1234);
    let mut image = vec![0.0f32; expected];
    for c in 0..IMAGE_C {
        for y in 0..IMAGE_H {
            for x in 0..IMAGE_W {
                let idx = (c * IMAGE_H + y) * IMAGE_W + x;
                let periodic = (((x * 13 + y * 17 + c * 7) & 255) as f32 - 127.5) / 127.5;
                let jitter = rng.next_f32(-0.05, 0.05);
                image[idx] = (periodic * 0.6 + jitter).clamp(-1.0, 1.0);
            }
        }
    }
    Ok(image)
}

fn tensor_f32_exact(
    model: &ClipSafeTensorFile,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let values = model.tensor_f32(name)?;
    if values.len() != expected_len {
        return Err(format!(
            "tensor {} length mismatch: expected {}, got {}",
            name,
            expected_len,
            values.len()
        )
        .into());
    }
    Ok(values)
}

fn load_layer_norm(
    model: &ClipSafeTensorFile,
    weight_name: &str,
    bias_name: &str,
    dim: usize,
) -> Result<LayerNormParams, Box<dyn Error>> {
    Ok(LayerNormParams {
        weight: tensor_f32_exact(model, weight_name, dim)?,
        bias: tensor_f32_exact(model, bias_name, dim)?,
    })
}

fn load_vision_params(model: &ClipSafeTensorFile) -> Result<VisionParams, Box<dyn Error>> {
    let patch_weight = tensor_f32_exact(
        model,
        "vision_model.embeddings.patch_embedding.weight",
        D_MODEL * IMAGE_C * PATCH * PATCH,
    )?;
    let class_embedding =
        tensor_f32_exact(model, "vision_model.embeddings.class_embedding", D_MODEL)?;
    let pos_embedding = tensor_f32_exact(
        model,
        "vision_model.embeddings.position_embedding.weight",
        TOKENS * D_MODEL,
    )?;

    let pre_ln = load_layer_norm(
        model,
        "vision_model.pre_layrnorm.weight",
        "vision_model.pre_layrnorm.bias",
        D_MODEL,
    )?;
    let post_ln = load_layer_norm(
        model,
        "vision_model.post_layernorm.weight",
        "vision_model.post_layernorm.bias",
        D_MODEL,
    )?;

    let visual_projection =
        tensor_f32_exact(model, "visual_projection.weight", PROJ_DIM * D_MODEL)?;

    Ok(VisionParams {
        patch_weight,
        class_embedding,
        pos_embedding,
        pre_ln,
        post_ln,
        visual_projection,
    })
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

fn build_calibration_input_q(rows: usize, input_dim: usize, qmax: i32, seed: u64) -> Vec<i8> {
    let mut rng = XorShift64::new(seed ^ 0xABCD_EF01_2345_6789);
    let mut out = vec![0i8; rows * input_dim];
    for row in 0..rows {
        for col in 0..input_dim {
            let idx = row * input_dim + col;
            let periodic = ((row * 19 + col * 7) % 129) as i32 - 64;
            let jitter = ((rng.next_u64() & 0x07) as i32) - 3;
            out[idx] = (periodic + jitter).clamp(-qmax, qmax) as i8;
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
        return Err("input rows are not aligned with input_dim".into());
    }
    if weights_row_major_q.len() != input_dim * output_dim {
        return Err("weight row-major length mismatch".into());
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

fn fit_affine(cpu_acc: &[i32], tpu_q: &[i8]) -> Result<AffineFit, Box<dyn Error>> {
    if cpu_acc.len() != tpu_q.len() || cpu_acc.is_empty() {
        return Err("fit_affine expects equal non-empty slices".into());
    }

    let n = cpu_acc.len() as f64;
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for idx in 0..cpu_acc.len() {
        mean_x += cpu_acc[idx] as f64;
        mean_y += tpu_q[idx] as f64;
    }
    mean_x /= n;
    mean_y /= n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for idx in 0..cpu_acc.len() {
        let dx = cpu_acc[idx] as f64 - mean_x;
        let dy = tpu_q[idx] as f64 - mean_y;
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

    let mut sq_sum = 0.0;
    for idx in 0..cpu_acc.len() {
        let pred = alpha * cpu_acc[idx] as f64 + beta;
        let err = tpu_q[idx] as f64 - pred;
        sq_sum += err * err;
    }

    Ok(AffineFit {
        alpha,
        beta,
        corr,
        rmse: (sq_sum / n).sqrt(),
    })
}

fn template_path_for_dims<'a>(
    config: &'a Config,
    input_dim: usize,
    output_dim: usize,
) -> Result<&'a str, Box<dyn Error>> {
    match (input_dim, output_dim) {
        (D_MODEL, D_MODEL) => Ok(config.template_768x768.as_str()),
        (D_MODEL, MLP_HIDDEN) => Ok(config.template_768x3072.as_str()),
        (MLP_HIDDEN, D_MODEL) => Ok(config.template_3072x768.as_str()),
        _ => Err(format!("unsupported stage dims {}x{}", input_dim, output_dim).into()),
    }
}

fn build_linear_stage(
    model: &ClipSafeTensorFile,
    weight_name: &str,
    bias_name: &str,
    input_dim: usize,
    output_dim: usize,
    template_path: &str,
    config: &Config,
    delegate: &coral_usb_oxidized::EdgeTPUDelegate,
    seed: u64,
) -> Result<CoralLinearStage, Box<dyn Error>> {
    let weights_f32 = tensor_f32_exact(model, weight_name, input_dim * output_dim)?;
    let bias = tensor_f32_exact(model, bias_name, output_dim)?;

    let (weights_q, quant_info) = quantize_linear_out_in_to_row_major_qi8_with_config(
        &weights_f32,
        input_dim,
        output_dim,
        LinearQuantConfig {
            qmax: config.weight_qmax,
            clip_percentile: config.clip_percentile,
        },
    )?;

    let mut template =
        DenseGemmTemplate::from_file_with_dims(template_path, input_dim, output_dim)?;
    template.set_weights_from_slice(&weights_q)?;
    let prepared = template.prepare(delegate)?;

    let calib_input_q =
        build_calibration_input_q(config.calibration_rows, input_dim, config.act_qmax, seed);
    let tpu_q = prepared.execute_batch_rows(&calib_input_q)?;
    let cpu_acc =
        cpu_accumulator_reference_batch(&calib_input_q, &weights_q, input_dim, output_dim)?;
    let fit = fit_affine(&cpu_acc, &tpu_q)?;

    Ok(CoralLinearStage {
        prepared,
        input_dim,
        output_dim,
        weight_scale: quant_info.scale,
        bias,
        affine_alpha: fit.alpha,
        affine_beta: fit.beta,
        fit_corr: fit.corr,
        fit_rmse: fit.rmse,
        quant_info,
    })
}

impl CoralLinearStage {
    fn forward_rows(
        &self,
        input_f32: &[f32],
        rows: usize,
        act_qmax: i32,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if input_f32.len() != rows * self.input_dim {
            return Err("linear input length mismatch".into());
        }
        let input_scale = symmetric_scale_for_qmax(input_f32, act_qmax);
        let input_q = quantize_symmetric_i8(input_f32, input_scale, act_qmax);
        let output_q = self.prepared.execute_batch_rows(&input_q)?;

        if output_q.len() != rows * self.output_dim {
            return Err("linear output length mismatch".into());
        }

        let mut out = vec![0.0f32; output_q.len()];
        let alpha = if self.affine_alpha.abs() < 1e-12 {
            1e-12
        } else {
            self.affine_alpha
        };

        for idx in 0..output_q.len() {
            let col = idx % self.output_dim;
            let acc_est = (output_q[idx] as f64 - self.affine_beta) / alpha;
            out[idx] = (acc_est as f32) * input_scale * self.weight_scale + self.bias[col];
        }
        Ok(out)
    }
}

fn layer_norm_rows(input: &[f32], rows: usize, dim: usize, params: &LayerNormParams) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for row in 0..rows {
        let row_start = row * dim;
        let row_end = row_start + dim;
        let row_slice = &input[row_start..row_end];

        let mean = row_slice.iter().sum::<f32>() / dim as f32;
        let var = row_slice
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / dim as f32;
        let inv = 1.0f32 / (var + LN_EPS).sqrt();

        for col in 0..dim {
            let normalized = (row_slice[col] - mean) * inv;
            out[row_start + col] = normalized * params.weight[col] + params.bias[col];
        }
    }
    out
}

fn quick_gelu_inplace(values: &mut [f32]) {
    for value in values.iter_mut() {
        let x = *value;
        *value = x / (1.0 + (-1.702 * x).exp());
    }
}

fn add_vectors(a: &[f32], b: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    if a.len() != b.len() {
        return Err("vector length mismatch in add".into());
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

fn attention_single_head(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * dim];
    let scale = 1.0f32 / (dim as f32).sqrt();

    for t in 0..seq {
        let q_row = &q[t * dim..(t + 1) * dim];
        let mut scores = vec![0.0f32; seq];
        for s in 0..seq {
            let k_row = &k[s * dim..(s + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q_row[d] * k_row[d];
            }
            scores[s] = dot * scale;
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };

        for d in 0..dim {
            let mut acc = 0.0f32;
            for s in 0..seq {
                let v_row = &v[s * dim..(s + 1) * dim];
                acc += scores[s] * inv * v_row[d];
            }
            out[t * dim + d] = acc;
        }
    }

    out
}

fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if q.len() != seq * D_MODEL || k.len() != seq * D_MODEL || v.len() != seq * D_MODEL {
        return Err("attention tensor length mismatch".into());
    }

    let mut out = vec![0.0f32; seq * D_MODEL];
    for head in 0..HEADS {
        let base = head * HEAD_DIM;
        let mut q_head = vec![0.0f32; seq * HEAD_DIM];
        let mut k_head = vec![0.0f32; seq * HEAD_DIM];
        let mut v_head = vec![0.0f32; seq * HEAD_DIM];

        for token in 0..seq {
            let src_base = token * D_MODEL + base;
            let dst_base = token * HEAD_DIM;
            q_head[dst_base..dst_base + HEAD_DIM]
                .copy_from_slice(&q[src_base..src_base + HEAD_DIM]);
            k_head[dst_base..dst_base + HEAD_DIM]
                .copy_from_slice(&k[src_base..src_base + HEAD_DIM]);
            v_head[dst_base..dst_base + HEAD_DIM]
                .copy_from_slice(&v[src_base..src_base + HEAD_DIM]);
        }

        let attn_head = attention_single_head(&q_head, &k_head, &v_head, seq, HEAD_DIM);
        for token in 0..seq {
            let src_base = token * HEAD_DIM;
            let dst_base = token * D_MODEL + base;
            out[dst_base..dst_base + HEAD_DIM]
                .copy_from_slice(&attn_head[src_base..src_base + HEAD_DIM]);
        }
    }

    Ok(out)
}

impl TransformerBlock {
    fn forward(&self, x: &[f32], seq: usize, act_qmax: i32) -> Result<Vec<f32>, Box<dyn Error>> {
        let x_ln1 = layer_norm_rows(x, seq, D_MODEL, &self.ln1);
        let q = self.q.forward_rows(&x_ln1, seq, act_qmax)?;
        let k = self.k.forward_rows(&x_ln1, seq, act_qmax)?;
        let v = self.v.forward_rows(&x_ln1, seq, act_qmax)?;

        let attn = multi_head_attention(&q, &k, &v, seq)?;
        let attn_out = self.o.forward_rows(&attn, seq, act_qmax)?;
        let x_res = add_vectors(x, &attn_out)?;

        let x_ln2 = layer_norm_rows(&x_res, seq, D_MODEL, &self.ln2);
        let mut hidden = self.fc1.forward_rows(&x_ln2, seq, act_qmax)?;
        quick_gelu_inplace(&mut hidden);
        let mlp_out = self.fc2.forward_rows(&hidden, seq, act_qmax)?;
        add_vectors(&x_res, &mlp_out)
    }
}

fn patch_embed_cpu(image_chw: &[f32], patch_weight: &[f32]) -> Vec<f32> {
    let mut patches = vec![0.0f32; PATCH_COUNT * D_MODEL];

    for py in 0..GRID {
        for px in 0..GRID {
            let token = py * GRID + px;
            let token_base = token * D_MODEL;
            for out_c in 0..D_MODEL {
                let mut acc = 0.0f32;
                let w_out_base = out_c * IMAGE_C * PATCH * PATCH;
                for in_c in 0..IMAGE_C {
                    let w_in_base = w_out_base + in_c * PATCH * PATCH;
                    for ky in 0..PATCH {
                        let iy = py * PATCH + ky;
                        let img_row_base = (in_c * IMAGE_H + iy) * IMAGE_W + px * PATCH;
                        let w_row_base = w_in_base + ky * PATCH;
                        for kx in 0..PATCH {
                            acc += image_chw[img_row_base + kx] * patch_weight[w_row_base + kx];
                        }
                    }
                }
                patches[token_base + out_c] = acc;
            }
        }
    }

    patches
}

fn apply_embeddings(
    patch_tokens: &[f32],
    class_embedding: &[f32],
    pos_embedding: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    if patch_tokens.len() != PATCH_COUNT * D_MODEL {
        return Err("patch token length mismatch".into());
    }
    if class_embedding.len() != D_MODEL || pos_embedding.len() != TOKENS * D_MODEL {
        return Err("embedding parameter length mismatch".into());
    }

    let mut hidden = vec![0.0f32; TOKENS * D_MODEL];
    for col in 0..D_MODEL {
        hidden[col] = class_embedding[col] + pos_embedding[col];
    }

    for patch in 0..PATCH_COUNT {
        let src_base = patch * D_MODEL;
        let dst_base = (patch + 1) * D_MODEL;
        let pos_base = dst_base;
        for col in 0..D_MODEL {
            hidden[dst_base + col] = patch_tokens[src_base + col] + pos_embedding[pos_base + col];
        }
    }

    Ok(hidden)
}

fn projection(weight_out_in: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for out_idx in 0..out_dim {
        let mut acc = 0.0f32;
        let row = &weight_out_in[out_idx * in_dim..(out_idx + 1) * in_dim];
        for in_idx in 0..in_dim {
            acc += row[in_idx] * input[in_idx];
        }
        out[out_idx] = acc;
    }
    out
}

fn l2_normalize(values: &[f32]) -> Vec<f32> {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    values.iter().map(|value| *value / norm).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for idx in 0..a.len() {
        dot += a[idx] * b[idx];
        na += a[idx] * a[idx];
        nb += b[idx] * b[idx];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn compare_embeddings(name: &str, produced: &[f32], reference: &[f32]) {
    let mut mae = 0.0f32;
    let mut mse = 0.0f32;
    let mut max_abs = 0.0f32;
    for idx in 0..produced.len() {
        let delta = (produced[idx] - reference[idx]).abs();
        mae += delta;
        mse += delta * delta;
        max_abs = max_abs.max(delta);
    }
    mae /= produced.len() as f32;
    mse /= produced.len() as f32;
    let rmse = mse.sqrt();
    let cos = cosine_similarity(produced, reference);
    println!(
        "Reference compare ({name}): cos={:.8} mae={:.8} rmse={:.8} max_abs={:.8}",
        cos, mae, rmse, max_abs
    );
}

fn build_transformer_block(
    model: &ClipSafeTensorFile,
    layer_idx: usize,
    config: &Config,
    delegate: &coral_usb_oxidized::EdgeTPUDelegate,
) -> Result<TransformerBlock, Box<dyn Error>> {
    let names = ClipVitLayerLinearNames::for_layer(layer_idx);

    let ln1 = load_layer_norm(
        model,
        &format!("vision_model.encoder.layers.{layer_idx}.layer_norm1.weight"),
        &format!("vision_model.encoder.layers.{layer_idx}.layer_norm1.bias"),
        D_MODEL,
    )?;
    let ln2 = load_layer_norm(
        model,
        &format!("vision_model.encoder.layers.{layer_idx}.layer_norm2.weight"),
        &format!("vision_model.encoder.layers.{layer_idx}.layer_norm2.bias"),
        D_MODEL,
    )?;

    let q = build_linear_stage(
        model,
        &names.q_proj,
        &format!("vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias"),
        D_MODEL,
        D_MODEL,
        template_path_for_dims(config, D_MODEL, D_MODEL)?,
        config,
        delegate,
        0x1000 + layer_idx as u64,
    )?;
    let k = build_linear_stage(
        model,
        &names.k_proj,
        &format!("vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias"),
        D_MODEL,
        D_MODEL,
        template_path_for_dims(config, D_MODEL, D_MODEL)?,
        config,
        delegate,
        0x2000 + layer_idx as u64,
    )?;
    let v = build_linear_stage(
        model,
        &names.v_proj,
        &format!("vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias"),
        D_MODEL,
        D_MODEL,
        template_path_for_dims(config, D_MODEL, D_MODEL)?,
        config,
        delegate,
        0x3000 + layer_idx as u64,
    )?;
    let o = build_linear_stage(
        model,
        &names.o_proj,
        &format!("vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias"),
        D_MODEL,
        D_MODEL,
        template_path_for_dims(config, D_MODEL, D_MODEL)?,
        config,
        delegate,
        0x4000 + layer_idx as u64,
    )?;
    let fc1 = build_linear_stage(
        model,
        &names.mlp_fc1,
        &format!("vision_model.encoder.layers.{layer_idx}.mlp.fc1.bias"),
        D_MODEL,
        MLP_HIDDEN,
        template_path_for_dims(config, D_MODEL, MLP_HIDDEN)?,
        config,
        delegate,
        0x5000 + layer_idx as u64,
    )?;
    let fc2 = build_linear_stage(
        model,
        &names.mlp_fc2,
        &format!("vision_model.encoder.layers.{layer_idx}.mlp.fc2.bias"),
        MLP_HIDDEN,
        D_MODEL,
        template_path_for_dims(config, MLP_HIDDEN, D_MODEL)?,
        config,
        delegate,
        0x6000 + layer_idx as u64,
    )?;

    Ok(TransformerBlock {
        ln1,
        ln2,
        q,
        k,
        v,
        o,
        fc1,
        fc2,
    })
}

fn print_stage_calibration(layer_idx: usize, name: &str, stage: &CoralLinearStage) {
    let clipped_pct = if stage.input_dim == 0 || stage.output_dim == 0 {
        0.0
    } else {
        100.0 * stage.quant_info.clipped_values as f64 / (stage.input_dim * stage.output_dim) as f64
    };
    println!(
        "L{:02} {:>4}: qmax={} scale={:.9} corr={:.6} rmse={:.4} clip={:.2}%",
        layer_idx,
        name,
        stage.quant_info.qmax,
        stage.quant_info.scale,
        stage.fit_corr,
        stage.fit_rmse,
        clipped_pct
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    println!("EdgeTPU version: {}", version());
    println!(
        "CLIP full forward config: layers={} weight_qmax={} act_qmax={} clip_percentile={} calibration_rows={}",
        config.max_layers, config.weight_qmax, config.act_qmax, config.clip_percentile, config.calibration_rows
    );

    let model = ClipSafeTensorFile::load(&config.model_path)?;
    let vision = load_vision_params(&model)?;

    let image = load_or_generate_image(&config)?;

    let started_prepare = Instant::now();
    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;

    let mut blocks = Vec::with_capacity(config.max_layers);
    for layer_idx in 0..config.max_layers {
        let block = build_transformer_block(&model, layer_idx, &config, &delegate)?;
        print_stage_calibration(layer_idx, "q", &block.q);
        print_stage_calibration(layer_idx, "k", &block.k);
        print_stage_calibration(layer_idx, "v", &block.v);
        print_stage_calibration(layer_idx, "o", &block.o);
        print_stage_calibration(layer_idx, "fc1", &block.fc1);
        print_stage_calibration(layer_idx, "fc2", &block.fc2);
        blocks.push(block);
    }
    let prepare_ms = started_prepare.elapsed().as_secs_f64() * 1000.0;

    let started_forward = Instant::now();

    let patch_tokens = patch_embed_cpu(&image, &vision.patch_weight);
    let mut hidden = apply_embeddings(
        &patch_tokens,
        &vision.class_embedding,
        &vision.pos_embedding,
    )?;
    hidden = layer_norm_rows(&hidden, TOKENS, D_MODEL, &vision.pre_ln);

    for (layer_idx, block) in blocks.iter().enumerate() {
        let started_layer = Instant::now();
        hidden = block.forward(&hidden, TOKENS, config.act_qmax)?;
        let layer_ms = started_layer.elapsed().as_secs_f64() * 1000.0;
        println!("layer {:02} forward_ms={:.3}", layer_idx, layer_ms);
    }

    hidden = layer_norm_rows(&hidden, TOKENS, D_MODEL, &vision.post_ln);
    let cls = &hidden[0..D_MODEL];
    let embedding = projection(&vision.visual_projection, cls, PROJ_DIM, D_MODEL);
    let embedding_norm = l2_normalize(&embedding);

    let forward_ms = started_forward.elapsed().as_secs_f64() * 1000.0;

    println!(
        "timing: prepare_ms={:.3} forward_ms={:.3} total_ms={:.3}",
        prepare_ms,
        forward_ms,
        prepare_ms + forward_ms
    );
    println!(
        "embedding raw first8: {}",
        embedding
            .iter()
            .take(8)
            .map(|value| format!("{:.6}", value))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "embedding norm first8: {}",
        embedding_norm
            .iter()
            .take(8)
            .map(|value| format!("{:.6}", value))
            .collect::<Vec<_>>()
            .join(",")
    );

    if let Some(path) = &config.output_f32le_path {
        write_f32_le_file(path, &embedding)?;
        println!("wrote raw embedding f32le: {}", path);
    }
    if let Some(path) = &config.output_norm_f32le_path {
        write_f32_le_file(path, &embedding_norm)?;
        println!("wrote normalized embedding f32le: {}", path);
    }

    if let Some(path) = &config.reference_f32le_path {
        let reference = read_f32_le_file(path, PROJ_DIM)?;
        compare_embeddings("raw", &embedding, &reference);
        let reference_norm = l2_normalize(&reference);
        compare_embeddings("normalized", &embedding_norm, &reference_norm);
    }

    Ok(())
}
