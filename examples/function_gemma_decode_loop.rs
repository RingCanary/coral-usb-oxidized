use coral_usb_oxidized::{
    quantize_linear_out_in_to_row_major_qi8_with_config, version, CoralDevice, DenseGemmTemplate,
    EdgeTPUDelegate, FunctionGemmaError, FunctionGemmaSafeTensorFile, LinearQuantConfig,
    PreparedDenseGemm,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LmHeadMode {
    Cpu,
    CoralTiled,
}

impl LmHeadMode {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "coral" | "coral-tiled" => Ok(Self::CoralTiled),
            other => Err(format!("invalid --lm-head '{}': expected cpu|coral", other).into()),
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    model_path: String,
    templates_dir: String,
    prompt_tokens: Vec<usize>,
    steps: usize,
    max_layers: Option<usize>,
    weight_qmax: i32,
    act_qmax: i32,
    clip_percentile: f32,
    calibration_rows: usize,
    head_dim: usize,
    rope_base: f32,
    rms_eps: f32,
    embed_scale: bool,
    topk: usize,
    lm_head_mode: LmHeadMode,
    lm_template_path: Option<String>,
    lm_tile_out_dim: usize,
}

#[derive(Clone, Copy, Debug)]
struct AffineFit {
    alpha: f64,
    beta: f64,
    corr: f64,
    rmse: f64,
}

struct CoralLinearStage {
    prepared: PreparedDenseGemm,
    input_dim: usize,
    output_dim: usize,
    weight_scale: f32,
    affine_alpha: f64,
    affine_beta: f64,
    fit_corr: f64,
}

struct CoralDecoderLayer {
    input_norm_weight: Vec<f32>,
    post_attn_norm_weight: Vec<f32>,
    q: CoralLinearStage,
    k: CoralLinearStage,
    v: CoralLinearStage,
    o: CoralLinearStage,
    gate: CoralLinearStage,
    up: CoralLinearStage,
    down: CoralLinearStage,
}

struct LayerKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

impl LayerKvCache {
    fn new(max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let len = max_seq
            .checked_mul(num_kv_heads)
            .and_then(|v| v.checked_mul(head_dim))
            .unwrap_or(0);
        Self {
            k: vec![0.0; len],
            v: vec![0.0; len],
        }
    }
}

struct CoralLmHeadTile {
    start_token: usize,
    valid_tokens: usize,
    stage: CoralLinearStage,
}

struct CoralTiledLmHead {
    hidden_dim: usize,
    tiles: Vec<CoralLmHeadTile>,
}

enum LmHeadBackend<'a> {
    Cpu {
        model: &'a FunctionGemmaSafeTensorFile,
    },
    Coral {
        tiled: CoralTiledLmHead,
    },
}

fn usage(program: &str) {
    println!(
        "Usage: {program} <model.safetensors> <templates_dir> <prompt_token_ids_csv> [options]"
    );
    println!("Options:");
    println!("  --steps N               Number of generated tokens (default: 8)");
    println!("  --max-layers N          Limit decoder layers (default: all detected)");
    println!("  --weight-qmax N         Weight quant qmax (default: 32)");
    println!("  --act-qmax N            Activation quant qmax (default: 32)");
    println!("  --clip-percentile P     Weight clipping percentile in (0,100] (default: 100)");
    println!("  --calibration-rows N    Rows used for stage affine calibration (default: 2)");
    println!("  --head-dim N            Attention head dim (default: 64)");
    println!("  --rope-base F           RoPE base theta (default: 10000)");
    println!("  --rms-eps F             RMSNorm epsilon (default: 1e-6)");
    println!("  --no-embed-scale        Disable sqrt(hidden) embedding scale");
    println!("  --topk N                Top-k logits printed per decode step (default: 5)");
    println!("  --lm-head MODE          cpu|coral (default: coral)");
    println!("  --lm-template PATH      Required when --lm-head coral");
    println!("  --lm-tile-out N         Coral LM-head tile output dim (default: 2624)");
    println!("Example:");
    println!(
        "  {program} /path/model.safetensors /path/templates 2,2516,29901 --steps 16 --lm-head coral --lm-template /path/dense_640x2624_quant_edgetpu.tflite"
    );
}

fn parse_csv_tokens(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut out = Vec::new();
    for item in value.split(',') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(trimmed.parse::<usize>()?);
    }
    if out.is_empty() {
        return Err("prompt_token_ids_csv must contain at least one token id".into());
    }
    Ok(out)
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "function_gemma_decode_loop".to_string());

    if args.len() < 4 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage(&program);
        if args.len() < 4 {
            std::process::exit(2);
        }
        std::process::exit(0);
    }

    let model_path = args[1].clone();
    let templates_dir = args[2].clone();
    let prompt_tokens = parse_csv_tokens(&args[3])?;

    let mut config = Config {
        model_path,
        templates_dir,
        prompt_tokens,
        steps: 8,
        max_layers: None,
        weight_qmax: 32,
        act_qmax: 32,
        clip_percentile: 100.0,
        calibration_rows: 2,
        head_dim: 64,
        rope_base: 10_000.0,
        rms_eps: 1e-6,
        embed_scale: true,
        topk: 5,
        lm_head_mode: LmHeadMode::CoralTiled,
        lm_template_path: None,
        lm_tile_out_dim: 2624,
    };

    let mut idx = 4usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--steps" => {
                idx += 1;
                config.steps = args.get(idx).ok_or("--steps requires a value")?.parse()?;
            }
            "--max-layers" => {
                idx += 1;
                config.max_layers = Some(
                    args.get(idx)
                        .ok_or("--max-layers requires a value")?
                        .parse()?,
                );
            }
            "--weight-qmax" => {
                idx += 1;
                config.weight_qmax = args
                    .get(idx)
                    .ok_or("--weight-qmax requires a value")?
                    .parse()?;
            }
            "--act-qmax" => {
                idx += 1;
                config.act_qmax = args
                    .get(idx)
                    .ok_or("--act-qmax requires a value")?
                    .parse()?;
            }
            "--clip-percentile" => {
                idx += 1;
                config.clip_percentile = args
                    .get(idx)
                    .ok_or("--clip-percentile requires a value")?
                    .parse()?;
            }
            "--calibration-rows" => {
                idx += 1;
                config.calibration_rows = args
                    .get(idx)
                    .ok_or("--calibration-rows requires a value")?
                    .parse()?;
            }
            "--head-dim" => {
                idx += 1;
                config.head_dim = args
                    .get(idx)
                    .ok_or("--head-dim requires a value")?
                    .parse()?;
            }
            "--rope-base" => {
                idx += 1;
                config.rope_base = args
                    .get(idx)
                    .ok_or("--rope-base requires a value")?
                    .parse()?;
            }
            "--rms-eps" => {
                idx += 1;
                config.rms_eps = args.get(idx).ok_or("--rms-eps requires a value")?.parse()?;
            }
            "--no-embed-scale" => {
                config.embed_scale = false;
            }
            "--topk" => {
                idx += 1;
                config.topk = args.get(idx).ok_or("--topk requires a value")?.parse()?;
            }
            "--lm-head" => {
                idx += 1;
                config.lm_head_mode =
                    LmHeadMode::parse(args.get(idx).ok_or("--lm-head requires a value")?)?;
            }
            "--lm-template" => {
                idx += 1;
                config.lm_template_path = Some(
                    args.get(idx)
                        .ok_or("--lm-template requires a value")?
                        .clone(),
                );
            }
            "--lm-tile-out" => {
                idx += 1;
                config.lm_tile_out_dim = args
                    .get(idx)
                    .ok_or("--lm-tile-out requires a value")?
                    .parse()?;
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        idx += 1;
    }

    if config.steps == 0 {
        return Err("--steps must be >= 1".into());
    }
    if config.calibration_rows == 0 {
        return Err("--calibration-rows must be >= 1".into());
    }
    if config.topk == 0 {
        return Err("--topk must be >= 1".into());
    }
    if config.lm_tile_out_dim == 0 || config.lm_tile_out_dim % 64 != 0 {
        return Err("--lm-tile-out must be a non-zero multiple of 64".into());
    }
    if !(1..=127).contains(&config.weight_qmax) {
        return Err("--weight-qmax must be in [1, 127]".into());
    }
    if !(1..=127).contains(&config.act_qmax) {
        return Err("--act-qmax must be in [1, 127]".into());
    }
    if !(0.0..=100.0).contains(&config.clip_percentile) || config.clip_percentile == 0.0 {
        return Err("--clip-percentile must be in (0, 100]".into());
    }
    if config.head_dim == 0 || config.head_dim % 2 != 0 {
        return Err("--head-dim must be a positive even integer".into());
    }

    if config.lm_head_mode == LmHeadMode::CoralTiled && config.lm_template_path.is_none() {
        return Err("--lm-template is required when --lm-head coral".into());
    }

    Ok(config)
}

fn template_path_for(templates_dir: &str, input_dim: usize, output_dim: usize) -> PathBuf {
    Path::new(templates_dir).join(format!(
        "dense_{}x{}_quant_edgetpu.tflite",
        input_dim, output_dim
    ))
}

fn detect_layer_count(model: &FunctionGemmaSafeTensorFile) -> Result<usize, Box<dyn Error>> {
    let names = model.tensor_names()?;
    let mut max_layer: Option<usize> = None;
    for name in names {
        let Some(rest) = name.strip_prefix("model.layers.") else {
            continue;
        };
        let Some((idx_str, _)) = rest.split_once('.') else {
            continue;
        };
        let Ok(idx) = idx_str.parse::<usize>() else {
            continue;
        };
        max_layer = Some(max_layer.map_or(idx, |current| current.max(idx)));
    }
    max_layer
        .map(|value| value + 1)
        .ok_or_else(|| "could not detect any layer under model.layers.*".into())
}

fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, Box<dyn Error>> {
    if input.len() != weight.len() {
        return Err("rms_norm: input/weight length mismatch".into());
    }
    let mean_sq = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    let mut out = Vec::with_capacity(input.len());
    for idx in 0..input.len() {
        out.push(input[idx] * inv_rms * weight[idx]);
    }
    Ok(out)
}

fn add_inplace(dst: &mut [f32], src: &[f32]) -> Result<(), Box<dyn Error>> {
    if dst.len() != src.len() {
        return Err("add_inplace: length mismatch".into());
    }
    for idx in 0..dst.len() {
        dst[idx] += src[idx];
    }
    Ok(())
}

fn silu_inplace(values: &mut [f32]) {
    for value in values.iter_mut() {
        let x = *value;
        *value = x / (1.0 + (-x).exp());
    }
}

fn apply_rope_inplace(
    vector: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    rope_base: f32,
) {
    for head in 0..num_heads {
        let head_start = head * head_dim;
        for pair in (0..head_dim).step_by(2) {
            let i = pair as f32;
            let theta = rope_base.powf(i / head_dim as f32);
            let angle = position as f32 / theta;
            let cos = angle.cos();
            let sin = angle.sin();

            let even_idx = head_start + pair;
            let odd_idx = even_idx + 1;
            let even = vector[even_idx];
            let odd = vector[odd_idx];
            vector[even_idx] = even * cos - odd * sin;
            vector[odd_idx] = even * sin + odd * cos;
        }
    }
}

fn kv_index(
    position: usize,
    head: usize,
    dim: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> usize {
    (position * num_kv_heads + head) * head_dim + dim
}

fn store_kv(
    cache: &mut LayerKvCache,
    position: usize,
    k: &[f32],
    v: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
) {
    for head in 0..num_kv_heads {
        for dim in 0..head_dim {
            let src = head * head_dim + dim;
            let dst = kv_index(position, head, dim, num_kv_heads, head_dim);
            cache.k[dst] = k[src];
            cache.v[dst] = v[src];
        }
    }
}

fn softmax_inplace(values: &mut [f32]) {
    let mut max_value = f32::NEG_INFINITY;
    for value in values.iter() {
        if *value > max_value {
            max_value = *value;
        }
    }

    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max_value).exp();
        sum += *value;
    }

    if sum > 0.0 {
        let inv = 1.0 / sum;
        for value in values.iter_mut() {
            *value *= inv;
        }
    }
}

fn gqa_attention_single_step(
    q: &[f32],
    cache: &LayerKvCache,
    position: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if q.len() != num_q_heads * head_dim {
        return Err("q length mismatch for attention".into());
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err("num_q_heads must be divisible by num_kv_heads".into());
    }

    let mut out = vec![0.0f32; q.len()];
    let q_per_kv = num_q_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; position + 1];

    for q_head in 0..num_q_heads {
        let kv_head = q_head / q_per_kv;
        let q_start = q_head * head_dim;
        let q_slice = &q[q_start..q_start + head_dim];

        for (t, score_slot) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for dim in 0..head_dim {
                let idx = kv_index(t, kv_head, dim, num_kv_heads, head_dim);
                dot += q_slice[dim] * cache.k[idx];
            }
            *score_slot = dot * scale;
        }

        softmax_inplace(&mut scores);

        for dim in 0..head_dim {
            let mut acc = 0.0f32;
            for (t, weight) in scores.iter().enumerate() {
                let idx = kv_index(t, kv_head, dim, num_kv_heads, head_dim);
                acc += *weight * cache.v[idx];
            }
            out[q_start + dim] = acc;
        }
    }

    Ok(out)
}

fn symmetric_scale_for_qmax(values: &[f32], qmax: i32) -> f32 {
    let mut max_abs = 0.0f32;
    for value in values {
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    if max_abs > 0.0 {
        max_abs / qmax as f32
    } else {
        1.0
    }
}

fn quantize_symmetric_i8(values: &[f32], scale: f32, qmax: i32) -> Vec<i8> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let q = (*value / scale).round() as i32;
        out.push(q.clamp(-qmax, qmax) as i8);
    }
    out
}

fn build_calibration_input_q(rows: usize, input_dim: usize, qmax: i32, seed: u64) -> Vec<i8> {
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

fn cpu_accumulator_reference_batch(
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

fn fit_affine(cpu_acc: &[i32], tpu_q: &[i8]) -> Result<AffineFit, Box<dyn Error>> {
    if cpu_acc.len() != tpu_q.len() || cpu_acc.is_empty() {
        return Err("fit_affine expects equal non-empty slices".into());
    }

    let n = cpu_acc.len() as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for idx in 0..cpu_acc.len() {
        sum_x += cpu_acc[idx] as f64;
        sum_y += tpu_q[idx] as f64;
    }

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    let mut cov = 0.0f64;
    for idx in 0..cpu_acc.len() {
        let dx = cpu_acc[idx] as f64 - mean_x;
        let dy = tpu_q[idx] as f64 - mean_y;
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

    let mut sum_sq = 0.0f64;
    for idx in 0..cpu_acc.len() {
        let pred = alpha * cpu_acc[idx] as f64 + beta;
        let err = tpu_q[idx] as f64 - pred;
        sum_sq += err * err;
    }

    Ok(AffineFit {
        alpha,
        beta,
        corr,
        rmse: (sum_sq / n).sqrt(),
    })
}

impl CoralLinearStage {
    fn forward_row(&self, input_f32: &[f32], act_qmax: i32) -> Result<Vec<f32>, Box<dyn Error>> {
        if input_f32.len() != self.input_dim {
            return Err(format!(
                "stage input mismatch: expected {}, got {}",
                self.input_dim,
                input_f32.len()
            )
            .into());
        }
        let input_scale = symmetric_scale_for_qmax(input_f32, act_qmax);
        let input_q = quantize_symmetric_i8(input_f32, input_scale, act_qmax);
        self.forward_from_quantized(&input_q, input_scale)
    }

    fn forward_from_quantized(
        &self,
        input_q: &[i8],
        input_scale: f32,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let output_q = self.prepared.execute(input_q)?;
        if output_q.len() != self.output_dim {
            return Err("stage output length mismatch".into());
        }

        let alpha = if self.affine_alpha.abs() < 1e-12 {
            1e-12
        } else {
            self.affine_alpha
        };

        let mut out = vec![0.0f32; self.output_dim];
        for idx in 0..self.output_dim {
            let acc_est = (output_q[idx] as f64 - self.affine_beta) / alpha;
            out[idx] = (acc_est as f32) * input_scale * self.weight_scale;
        }
        Ok(out)
    }
}

fn build_stage_from_weights(
    weights_f32: &[f32],
    input_dim: usize,
    output_dim: usize,
    template_bytes: &[u8],
    delegate: &EdgeTPUDelegate,
    config: &Config,
    seed: u64,
) -> Result<CoralLinearStage, Box<dyn Error>> {
    let (weights_q, quant_info) = quantize_linear_out_in_to_row_major_qi8_with_config(
        weights_f32,
        input_dim,
        output_dim,
        LinearQuantConfig {
            qmax: config.weight_qmax,
            clip_percentile: config.clip_percentile,
        },
    )?;

    let mut template =
        DenseGemmTemplate::from_bytes_with_dims(template_bytes, input_dim, output_dim)?;
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
        affine_alpha: fit.alpha,
        affine_beta: fit.beta,
        fit_corr: fit.corr,
    })
}

fn load_first_tensor_by_names(
    model: &FunctionGemmaSafeTensorFile,
    candidates: &[String],
) -> Result<(String, Vec<f32>), Box<dyn Error>> {
    let mut last_missing = None::<String>;
    for name in candidates {
        match model.tensor_f32(name) {
            Ok(tensor) => return Ok((name.clone(), tensor)),
            Err(FunctionGemmaError::MissingTensor(_)) => {
                last_missing = Some(name.clone());
            }
            Err(err) => return Err(Box::new(err)),
        }
    }
    Err(format!(
        "none of candidate tensors exist (last checked: {:?})",
        last_missing
    )
    .into())
}

fn build_decoder_layers(
    model: &FunctionGemmaSafeTensorFile,
    delegate: &EdgeTPUDelegate,
    config: &Config,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    mlp_dim: usize,
    layer_count: usize,
) -> Result<Vec<CoralDecoderLayer>, Box<dyn Error>> {
    let mut template_cache: BTreeMap<(usize, usize), Vec<u8>> = BTreeMap::new();
    let stage_dims = [
        (hidden_dim, q_dim),
        (hidden_dim, kv_dim),
        (q_dim, hidden_dim),
        (hidden_dim, mlp_dim),
        (mlp_dim, hidden_dim),
    ];

    for (input_dim, output_dim) in stage_dims {
        let path = template_path_for(&config.templates_dir, input_dim, output_dim);
        let bytes = std::fs::read(&path).map_err(|err| {
            format!(
                "failed to read template {} for {}x{}: {}",
                path.display(),
                input_dim,
                output_dim,
                err
            )
        })?;
        template_cache.insert((input_dim, output_dim), bytes);
    }

    let mut layers = Vec::with_capacity(layer_count);

    for layer_idx in 0..layer_count {
        let names = coral_usb_oxidized::FunctionGemmaLayerLinearNames::for_layer(layer_idx);

        let (input_norm_name, input_norm_weight) = load_first_tensor_by_names(
            model,
            &[format!("model.layers.{}.input_layernorm.weight", layer_idx)],
        )?;
        let (post_norm_name, post_attn_norm_weight) = load_first_tensor_by_names(
            model,
            &[
                format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
                format!(
                    "model.layers.{}.pre_feedforward_layernorm.weight",
                    layer_idx
                ),
            ],
        )?;

        if input_norm_weight.len() != hidden_dim {
            return Err(format!(
                "{} length mismatch: expected {}, got {}",
                input_norm_name,
                hidden_dim,
                input_norm_weight.len()
            )
            .into());
        }
        if post_attn_norm_weight.len() != hidden_dim {
            return Err(format!(
                "{} length mismatch: expected {}, got {}",
                post_norm_name,
                hidden_dim,
                post_attn_norm_weight.len()
            )
            .into());
        }

        let weights_q = model.tensor_f32(&names.q_proj)?;
        let weights_k = model.tensor_f32(&names.k_proj)?;
        let weights_v = model.tensor_f32(&names.v_proj)?;
        let weights_o = model.tensor_f32(&names.o_proj)?;
        let weights_gate = model.tensor_f32(&names.gate_proj)?;
        let weights_up = model.tensor_f32(&names.up_proj)?;
        let weights_down = model.tensor_f32(&names.down_proj)?;

        let q = build_stage_from_weights(
            &weights_q,
            hidden_dim,
            q_dim,
            template_cache
                .get(&(hidden_dim, q_dim))
                .ok_or("missing q template in cache")?,
            delegate,
            config,
            10_000 + layer_idx as u64 * 100 + 1,
        )?;
        let k = build_stage_from_weights(
            &weights_k,
            hidden_dim,
            kv_dim,
            template_cache
                .get(&(hidden_dim, kv_dim))
                .ok_or("missing k template in cache")?,
            delegate,
            config,
            10_000 + layer_idx as u64 * 100 + 2,
        )?;
        let v = build_stage_from_weights(
            &weights_v,
            hidden_dim,
            kv_dim,
            template_cache
                .get(&(hidden_dim, kv_dim))
                .ok_or("missing v template in cache")?,
            delegate,
            config,
            10_000 + layer_idx as u64 * 100 + 3,
        )?;
        let o = build_stage_from_weights(
            &weights_o,
            q_dim,
            hidden_dim,
            template_cache
                .get(&(q_dim, hidden_dim))
                .ok_or("missing o template in cache")?,
            delegate,
            config,
            10_000 + layer_idx as u64 * 100 + 4,
        )?;
        let gate = build_stage_from_weights(
            &weights_gate,
            hidden_dim,
            mlp_dim,
            template_cache
                .get(&(hidden_dim, mlp_dim))
                .ok_or("missing gate template in cache")?,
            delegate,
            config,
            10_000 + layer_idx as u64 * 100 + 5,
        )?;
        let up = build_stage_from_weights(
            &weights_up,
            hidden_dim,
            mlp_dim,
            template_cache
                .get(&(hidden_dim, mlp_dim))
                .ok_or("missing up template in cache")?,
            delegate,
            config,
            10_000 + layer_idx as u64 * 100 + 6,
        )?;
        let down = build_stage_from_weights(
            &weights_down,
            mlp_dim,
            hidden_dim,
            template_cache
                .get(&(mlp_dim, hidden_dim))
                .ok_or("missing down template in cache")?,
            delegate,
            config,
            10_000 + layer_idx as u64 * 100 + 7,
        )?;

        println!(
            "Prepared layer {:02}: q(corr={:.5}) k(corr={:.5}) v(corr={:.5}) o(corr={:.5}) gate(corr={:.5}) up(corr={:.5}) down(corr={:.5})",
            layer_idx,
            q.fit_corr,
            k.fit_corr,
            v.fit_corr,
            o.fit_corr,
            gate.fit_corr,
            up.fit_corr,
            down.fit_corr
        );

        layers.push(CoralDecoderLayer {
            input_norm_weight,
            post_attn_norm_weight,
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
        });
    }

    Ok(layers)
}

fn push_topk(best: &mut Vec<(usize, f32)>, candidate: (usize, f32), topk: usize) {
    if best.len() < topk {
        best.push(candidate);
        return;
    }

    let mut worst_idx = 0usize;
    let mut worst_val = best[0].1;
    for (idx, (_, value)) in best.iter().enumerate().skip(1) {
        if *value < worst_val {
            worst_idx = idx;
            worst_val = *value;
        }
    }
    if candidate.1 > worst_val {
        best[worst_idx] = candidate;
    }
}

fn build_coral_tiled_lm_head(
    model: &FunctionGemmaSafeTensorFile,
    delegate: &EdgeTPUDelegate,
    config: &Config,
    hidden_dim: usize,
) -> Result<CoralTiledLmHead, Box<dyn Error>> {
    let lm_template_path = config
        .lm_template_path
        .as_ref()
        .ok_or("missing lm template path")?;
    let lm_template_bytes = std::fs::read(lm_template_path)
        .map_err(|err| format!("failed to read LM template {}: {}", lm_template_path, err))?;

    let (vocab, embed_hidden) = model.embedding_dims()?;
    if embed_hidden != hidden_dim {
        return Err(format!(
            "embedding hidden dim mismatch: expected {}, got {}",
            hidden_dim, embed_hidden
        )
        .into());
    }

    let tile_out = config.lm_tile_out_dim;
    let tile_count = vocab.div_ceil(tile_out);
    let mut tiles = Vec::with_capacity(tile_count);

    for tile_idx in 0..tile_count {
        let start_token = tile_idx * tile_out;
        let valid_tokens = (vocab - start_token).min(tile_out);

        let mut weights_out_by_in = model.embedding_rows_f32(start_token, valid_tokens)?;
        if valid_tokens < tile_out {
            weights_out_by_in.resize(tile_out * hidden_dim, 0.0);
        }

        let (weights_q, quant_info) = quantize_linear_out_in_to_row_major_qi8_with_config(
            &weights_out_by_in,
            hidden_dim,
            tile_out,
            LinearQuantConfig {
                qmax: config.weight_qmax,
                clip_percentile: config.clip_percentile,
            },
        )?;

        let mut template =
            DenseGemmTemplate::from_bytes_with_dims(&lm_template_bytes, hidden_dim, tile_out)?;
        template.set_weights_from_slice(&weights_q)?;
        let prepared = template.prepare(delegate)?;

        let calib_input_q = build_calibration_input_q(
            config.calibration_rows,
            hidden_dim,
            config.act_qmax,
            900_000 + tile_idx as u64,
        );
        let tpu_q = prepared.execute_batch_rows(&calib_input_q)?;
        let cpu_acc =
            cpu_accumulator_reference_batch(&calib_input_q, &weights_q, hidden_dim, tile_out)?;
        let fit = fit_affine(&cpu_acc, &tpu_q)?;

        let stage = CoralLinearStage {
            prepared,
            input_dim: hidden_dim,
            output_dim: tile_out,
            weight_scale: quant_info.scale,
            affine_alpha: fit.alpha,
            affine_beta: fit.beta,
            fit_corr: fit.corr,
        };

        if tile_idx % 10 == 0 || tile_idx + 1 == tile_count {
            println!(
                "Prepared LM tile {:03}/{:03}: start={} valid={} corr={:.5} rmse={:.5}",
                tile_idx + 1,
                tile_count,
                start_token,
                valid_tokens,
                stage.fit_corr,
                fit.rmse,
            );
        }

        tiles.push(CoralLmHeadTile {
            start_token,
            valid_tokens,
            stage,
        });
    }

    Ok(CoralTiledLmHead { hidden_dim, tiles })
}

impl CoralTiledLmHead {
    fn topk(
        &self,
        hidden_state: &[f32],
        act_qmax: i32,
        topk: usize,
    ) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
        if hidden_state.len() != self.hidden_dim {
            return Err("hidden length mismatch in LM head".into());
        }

        let input_scale = symmetric_scale_for_qmax(hidden_state, act_qmax);
        let input_q = quantize_symmetric_i8(hidden_state, input_scale, act_qmax);

        let mut best: Vec<(usize, f32)> = Vec::with_capacity(topk);
        for tile in &self.tiles {
            let logits = tile.stage.forward_from_quantized(&input_q, input_scale)?;
            for (local_idx, logit) in logits.iter().take(tile.valid_tokens).enumerate() {
                let token_id = tile.start_token + local_idx;
                push_topk(&mut best, (token_id, *logit), topk);
            }
        }

        best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(best)
    }
}

struct ForwardContext<'a> {
    layers: &'a [CoralDecoderLayer],
    caches: &'a mut [LayerKvCache],
    model: &'a FunctionGemmaSafeTensorFile,
    final_norm_weight: &'a [f32],
    hidden_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: &'a Config,
    lm_backend: &'a LmHeadBackend<'a>,
}

fn forward_single_token(
    ctx: &mut ForwardContext<'_>,
    token_id: usize,
    position: usize,
) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
    let mut hidden = ctx.model.token_embedding_row_f32(token_id)?;
    if hidden.len() != ctx.hidden_dim {
        return Err(format!(
            "embedding hidden mismatch: expected {}, got {}",
            ctx.hidden_dim,
            hidden.len()
        )
        .into());
    }

    if ctx.config.embed_scale {
        let scale = (ctx.hidden_dim as f32).sqrt();
        for value in hidden.iter_mut() {
            *value *= scale;
        }
    }

    for (layer_idx, layer) in ctx.layers.iter().enumerate() {
        let cache = &mut ctx.caches[layer_idx];

        let x_norm = rms_norm(&hidden, &layer.input_norm_weight, ctx.config.rms_eps)?;

        let mut q = layer.q.forward_row(&x_norm, ctx.config.act_qmax)?;
        let mut k = layer.k.forward_row(&x_norm, ctx.config.act_qmax)?;
        let v = layer.v.forward_row(&x_norm, ctx.config.act_qmax)?;

        apply_rope_inplace(
            &mut q,
            ctx.num_q_heads,
            ctx.head_dim,
            position,
            ctx.config.rope_base,
        );
        apply_rope_inplace(
            &mut k,
            ctx.num_kv_heads,
            ctx.head_dim,
            position,
            ctx.config.rope_base,
        );

        store_kv(cache, position, &k, &v, ctx.num_kv_heads, ctx.head_dim);
        let attn = gqa_attention_single_step(
            &q,
            cache,
            position,
            ctx.num_q_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
        )?;
        let attn_out = layer.o.forward_row(&attn, ctx.config.act_qmax)?;
        add_inplace(&mut hidden, &attn_out)?;

        let x_norm2 = rms_norm(&hidden, &layer.post_attn_norm_weight, ctx.config.rms_eps)?;
        let mut gate = layer.gate.forward_row(&x_norm2, ctx.config.act_qmax)?;
        let up = layer.up.forward_row(&x_norm2, ctx.config.act_qmax)?;
        silu_inplace(&mut gate);
        for idx in 0..gate.len() {
            gate[idx] *= up[idx];
        }

        let mlp_out = layer.down.forward_row(&gate, ctx.config.act_qmax)?;
        add_inplace(&mut hidden, &mlp_out)?;
    }

    hidden = rms_norm(&hidden, ctx.final_norm_weight, ctx.config.rms_eps)?;

    match ctx.lm_backend {
        LmHeadBackend::Cpu { model } => {
            Ok(model.lm_head_topk_from_hidden(&hidden, ctx.config.topk)?)
        }
        LmHeadBackend::Coral { tiled } => tiled.topk(&hidden, ctx.config.act_qmax, ctx.config.topk),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;

    println!("EdgeTPU version: {}", version());
    println!("Model: {}", config.model_path);
    println!("Templates dir: {}", config.templates_dir);
    println!(
        "Prompt tokens: {}",
        config
            .prompt_tokens
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "Decode config: steps={} weight_qmax={} act_qmax={} clip_percentile={} calibration_rows={} lm_head={:?}",
        config.steps,
        config.weight_qmax,
        config.act_qmax,
        config.clip_percentile,
        config.calibration_rows,
        config.lm_head_mode
    );

    let setup_started = Instant::now();

    let model = FunctionGemmaSafeTensorFile::load(&config.model_path)?;
    println!("SafeTensors storage: {}", model.storage_kind());

    let detected_layers = detect_layer_count(&model)?;
    let layer_count = config
        .max_layers
        .map(|limit| limit.min(detected_layers))
        .unwrap_or(detected_layers);
    if layer_count == 0 {
        return Err("layer_count resolved to 0".into());
    }

    let dims = model.infer_layer_dims(0)?;
    let hidden_dim = dims.hidden_size;
    let q_dim = dims.q_proj_out;
    let kv_dim = dims.kv_proj_out;
    let mlp_dim = dims.mlp_hidden;

    if q_dim % config.head_dim != 0 || kv_dim % config.head_dim != 0 {
        return Err(format!(
            "head_dim {} does not divide q_dim {} and kv_dim {}",
            config.head_dim, q_dim, kv_dim
        )
        .into());
    }

    let num_q_heads = q_dim / config.head_dim;
    let num_kv_heads = kv_dim / config.head_dim;
    if num_q_heads % num_kv_heads != 0 {
        return Err(format!(
            "num_q_heads {} not divisible by num_kv_heads {}",
            num_q_heads, num_kv_heads
        )
        .into());
    }

    println!(
        "Model dims: layers={} hidden={} q_dim={} kv_dim={} mlp_dim={} q_heads={} kv_heads={} head_dim={}",
        layer_count,
        hidden_dim,
        q_dim,
        kv_dim,
        mlp_dim,
        num_q_heads,
        num_kv_heads,
        config.head_dim
    );

    let final_norm_candidates = vec![
        "model.norm.weight".to_string(),
        "model.final_layernorm.weight".to_string(),
    ];
    let (final_norm_name, final_norm_weight) =
        load_first_tensor_by_names(&model, &final_norm_candidates)?;
    if final_norm_weight.len() != hidden_dim {
        return Err(format!(
            "{} length mismatch: expected {}, got {}",
            final_norm_name,
            hidden_dim,
            final_norm_weight.len()
        )
        .into());
    }

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;

    let layers = build_decoder_layers(
        &model,
        &delegate,
        &config,
        hidden_dim,
        q_dim,
        kv_dim,
        mlp_dim,
        layer_count,
    )?;

    let lm_backend = match config.lm_head_mode {
        LmHeadMode::Cpu => LmHeadBackend::Cpu { model: &model },
        LmHeadMode::CoralTiled => LmHeadBackend::Coral {
            tiled: build_coral_tiled_lm_head(&model, &delegate, &config, hidden_dim)?,
        },
    };

    let max_seq = config.prompt_tokens.len() + config.steps + 2;
    let mut caches: Vec<LayerKvCache> = (0..layer_count)
        .map(|_| LayerKvCache::new(max_seq, num_kv_heads, config.head_dim))
        .collect();

    let setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;
    println!("Setup complete: {:.3} ms", setup_ms);

    let mut forward_ctx = ForwardContext {
        layers: &layers,
        caches: &mut caches,
        model: &model,
        final_norm_weight: &final_norm_weight,
        hidden_dim,
        num_q_heads,
        num_kv_heads,
        head_dim: config.head_dim,
        config: &config,
        lm_backend: &lm_backend,
    };

    let prefill_started = Instant::now();
    for (pos, token) in config
        .prompt_tokens
        .iter()
        .take(config.prompt_tokens.len().saturating_sub(1))
        .enumerate()
    {
        let _ = forward_single_token(&mut forward_ctx, *token, pos)?;
    }
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;

    let mut current_token = *config
        .prompt_tokens
        .last()
        .ok_or("prompt token list unexpectedly empty")?;
    let mut position = config.prompt_tokens.len() - 1;

    let decode_started = Instant::now();
    let mut generated_tokens = Vec::with_capacity(config.steps);

    for step_idx in 0..config.steps {
        let step_started = Instant::now();
        let topk = forward_single_token(&mut forward_ctx, current_token, position)?;
        let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;

        let next_token = topk.first().ok_or("topk result unexpectedly empty")?.0;
        generated_tokens.push(next_token);

        let topk_preview = topk
            .iter()
            .map(|(token, logit)| format!("{}:{:.4}", token, logit))
            .collect::<Vec<_>>()
            .join(" ");

        println!(
            "STEP {:03} pos={} in_token={} next_token={} step_ms={:.3} topk=[{}]",
            step_idx, position, current_token, next_token, step_ms, topk_preview
        );

        current_token = next_token;
        position += 1;
    }

    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;

    println!(
        "RESULT layers={} prompt_len={} generated={} prefill_ms={:.3} decode_ms={:.3} ms_per_token={:.3}",
        layer_count,
        config.prompt_tokens.len(),
        generated_tokens.len(),
        prefill_ms,
        decode_ms,
        decode_ms / generated_tokens.len() as f64
    );
    println!(
        "Generated token ids: {}",
        generated_tokens
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    Ok(())
}
