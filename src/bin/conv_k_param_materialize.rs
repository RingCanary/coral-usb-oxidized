use coral_usb_oxidized::{extract_tflite_conv_quantized_weights, DenseGemmError};
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct GeneratedConvMetadata {
    model_name: String,
    height: usize,
    width: usize,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --model MODEL.tflite --metadata MODEL.metadata.json --out OUT.bin [--verify-against STREAM.bin]"
    );
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{flag} requires a value"));
    }
    Ok(args[*idx].clone())
}

fn expand_weight_scales(out_channels: usize, weight_scales: &[f32]) -> Result<Vec<f32>, DenseGemmError> {
    if weight_scales.len() == out_channels {
        return Ok(weight_scales.to_vec());
    }
    if weight_scales.len() == 1 {
        return Ok(vec![weight_scales[0]; out_channels]);
    }
    Err(DenseGemmError::InvalidTemplate(format!(
        "weight_scales len must be 1 or out_channels (got {}, out_channels={})",
        weight_scales.len(),
        out_channels
    )))
}

fn expand_weight_zero_points(out_channels: usize, weight_zero_points: &[i64]) -> Result<Vec<i64>, DenseGemmError> {
    if weight_zero_points.is_empty() {
        return Ok(vec![0; out_channels]);
    }
    if weight_zero_points.len() == out_channels {
        return Ok(weight_zero_points.to_vec());
    }
    if weight_zero_points.len() == 1 {
        return Ok(vec![weight_zero_points[0]; out_channels]);
    }
    Err(DenseGemmError::InvalidTemplate(format!(
        "weight_zero_points len must be 0, 1, or out_channels (got {}, out_channels={})",
        weight_zero_points.len(),
        out_channels
    )))
}

fn block_widths(out_channels: usize) -> Vec<usize> {
    let mut remaining = out_channels;
    let mut blocks = Vec::new();
    while remaining > 64 {
        blocks.push(64);
        remaining -= 64;
    }
    blocks.push(remaining);
    blocks
}

fn materialize_param_stream(
    height: usize,
    width: usize,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    input_scale: f32,
    output_scale: f32,
    weight_scales: &[f32],
    weight_zero_points: &[i64],
    weight_bytes_oc_kh_kw_ic: &[u8],
) -> Result<Vec<u8>, Box<dyn Error>> {
    if kernel_size == 0 || in_channels == 0 || out_channels == 0 {
        return Err("kernel_size/in_channels/out_channels must be non-zero".into());
    }
    if (in_channels % 4) != 0 {
        return Err("in_channels must be a multiple of 4".into());
    }
    if (out_channels % 32) != 0 {
        return Err("out_channels must be a multiple of 32".into());
    }
    if output_scale == 0.0 {
        return Err("output_scale must be non-zero".into());
    }

    let expected_weight_len = kernel_size
        .checked_mul(kernel_size)
        .and_then(|v| v.checked_mul(in_channels))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or("weight length overflow")?;
    if weight_bytes_oc_kh_kw_ic.len() != expected_weight_len {
        return Err(format!(
            "weight byte length mismatch: expected {}, got {}",
            expected_weight_len,
            weight_bytes_oc_kh_kw_ic.len()
        )
        .into());
    }

    let weight_scales = expand_weight_scales(out_channels, weight_scales)?;
    let weight_zero_points = expand_weight_zero_points(out_channels, weight_zero_points)?;
    let output_recip = 1.0f32 / output_scale;
    let prefix_total = out_channels
        .checked_mul(8)
        .ok_or("prefix_total overflow")?;
    let total_len = prefix_total
        .checked_add(expected_weight_len)
        .ok_or("stream length overflow")?;
    let mut out = vec![0u8; total_len];

    let ic_group_count = in_channels / 4;
    let use_kernel_major_groups = kernel_size == 3
        && in_channels == 32
        && out_channels == 32
        && height == 12
        && width >= 176;

    let mut block_start = 0usize;
    let mut oc_base = 0usize;
    for bw in block_widths(out_channels) {
        for local_oc in 0..bw {
            let oc = oc_base + local_oc;
            let eff = (input_scale * weight_scales[oc]) * output_recip;
            let scale_off = block_start + local_oc * 4;
            out[scale_off..scale_off + 4].copy_from_slice(&eff.to_le_bytes());
        }
        let zp_start = block_start + bw * 4;
        for local_oc in 0..bw {
            let oc = oc_base + local_oc;
            let stored_zp = u32::try_from(weight_zero_points[oc] + 128)
                .map_err(|_| format!("stored zero point out of range for oc={oc}"))?;
            let zp_off = zp_start + local_oc * 4;
            out[zp_off..zp_off + 4].copy_from_slice(&stored_zp.to_le_bytes());
        }

        let weight_start = block_start + bw * 8;
        for local_oc in 0..bw {
            let oc = oc_base + local_oc;
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    for ic in 0..in_channels {
                        let src_idx =
                            (((oc * kernel_size + ky) * kernel_size + kx) * in_channels) + ic;
                        let kernel_pos = (ky * kernel_size) + kx;
                        let ic_group = ic / 4;
                        let group_index = if use_kernel_major_groups {
                            (kernel_pos * ic_group_count) + ic_group
                        } else {
                            (ic_group * kernel_size * kernel_size) + kernel_pos
                        };
                        let group_base = group_index * (bw * 4);
                        let dst_idx = weight_start + group_base + local_oc * 4 + (ic % 4);
                        out[dst_idx] = weight_bytes_oc_kh_kw_ic[src_idx].wrapping_add(128);
                    }
                }
            }
        }

        block_start += bw * (8 + in_channels * kernel_size * kernel_size);
        oc_base += bw;
    }

    Ok(out)
}

fn mismatch_count(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "conv_k_param_materialize".to_string());
    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut model_path: Option<PathBuf> = None;
    let mut metadata_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut verify_against: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--model" => model_path = Some(PathBuf::from(next_arg(&args, &mut i, "--model")?)),
            "--metadata" => {
                metadata_path = Some(PathBuf::from(next_arg(&args, &mut i, "--metadata")?))
            }
            "--out" => out_path = Some(PathBuf::from(next_arg(&args, &mut i, "--out")?)),
            "--verify-against" => {
                verify_against = Some(PathBuf::from(next_arg(&args, &mut i, "--verify-against")?))
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let model_path = model_path.ok_or("missing --model")?;
    let metadata_path = metadata_path.ok_or("missing --metadata")?;
    let out_path = out_path.ok_or("missing --out")?;

    let metadata: GeneratedConvMetadata =
        serde_json::from_slice(&fs::read(&metadata_path)?)?;
    let model_bytes = fs::read(&model_path)?;
    let quant = extract_tflite_conv_quantized_weights(
        &model_bytes,
        0,
        metadata.kernel_size,
        metadata.in_channels,
        metadata.out_channels,
    )?;
    let stream = materialize_param_stream(
        metadata.height,
        metadata.width,
        metadata.kernel_size,
        metadata.in_channels,
        metadata.out_channels,
        quant.input_scale,
        quant.output_scale,
        &quant.weight_scales,
        &quant.weight_zero_points,
        &quant.weight_bytes_oc_kh_kw_ic,
    )?;

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, &stream)?;

    let mut summary = format!(
        "model={} height={} width={} kernel_size={} in_channels={} out_channels={} weight_tensor_index={} stored_shape={:?} stream_len={}",
        metadata.model_name,
        metadata.height,
        metadata.width,
        metadata.kernel_size,
        metadata.in_channels,
        metadata.out_channels,
        quant.weight_tensor_index,
        quant.stored_shape,
        stream.len()
    );
    if let Some(verify_path) = verify_against.as_ref() {
        let expected = fs::read(verify_path)?;
        if expected.len() != stream.len() {
            return Err(format!(
                "verify-against length mismatch: expected {}, got {} ({})",
                expected.len(),
                stream.len(),
                verify_path.display()
            )
            .into());
        }
        let mismatch = mismatch_count(&stream, &expected);
        summary.push_str(&format!(
            " byte_equal={} mismatch_count={} verify_against={}",
            mismatch == 0,
            mismatch,
            verify_path.display()
        ));
    }
    summary.push_str(&format!(" out={}", out_path.display()));
    println!("{summary}");
    Ok(())
}
