use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct GeneratedConvMetadata {
    model_name: String,
    height: usize,
    width: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    input_tensor: QuantizedTensor,
    output_tensor: QuantizedTensor,
    kernel_sha256: String,
}

#[derive(Debug, Deserialize)]
struct QuantizedTensor {
    quantization: (f32, i64),
}

#[derive(Debug, Deserialize)]
struct PrepSummary {
    param_equal: bool,
    eo_rule_count: usize,
    pc_rule_count: usize,
}

#[derive(Debug, Serialize)]
struct PairReport {
    pair_id: String,
    anchor_model_name: String,
    target_model_name: String,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    anchor_hw: (usize, usize),
    target_hw: (usize, usize),
    kernel_sha256_equal: bool,
    input_scale_equal: bool,
    output_scale_equal: bool,
    anchor_output_scale: f32,
    target_output_scale: f32,
    param_len: usize,
    expected_weight_bytes: usize,
    inferred_prefix_total_bytes: usize,
    inferred_blocks: Vec<BlockReport>,
    diff_byte_count: usize,
    diff_span_count: usize,
    diff_spans: Vec<(usize, usize)>,
    diff_bytes_in_scale_prefix: usize,
    diff_bytes_in_zero_point_prefix: usize,
    diff_bytes_in_weight_region: usize,
    weights_equal: bool,
    zero_points_equal: bool,
    all_diffs_confined_to_scale_prefix: bool,
    conclusion: String,
    prep_param_equal: bool,
    eo_rule_count: usize,
    pc_rule_count: usize,
}

#[derive(Debug, Serialize)]
struct BlockReport {
    block_index: usize,
    out_channel_start: usize,
    out_channel_count: usize,
    block_start: usize,
    scale_prefix_start: usize,
    scale_prefix_end: usize,
    zero_point_prefix_start: usize,
    zero_point_prefix_end: usize,
    weight_start: usize,
    weight_end: usize,
    diff_bytes_in_scale_prefix: usize,
    diff_bytes_in_zero_point_prefix: usize,
    diff_bytes_in_weight_region: usize,
}

fn usage(program: &str) {
    eprintln!("Usage: {program} --run-dir PATH");
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{flag} requires a value"));
    }
    Ok(args[*idx].clone())
}

fn find_metadata(dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let mut matches = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json")
            && path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.ends_with(".metadata.json"))
                .unwrap_or(false)
        {
            matches.push(path);
        }
    }
    matches.sort();
    matches
        .into_iter()
        .next()
        .ok_or_else(|| format!("no *.metadata.json found in {}", dir.display()).into())
}

fn count_diffs_in_range(diffs: &[usize], start: usize, end_exclusive: usize) -> usize {
    diffs.iter()
        .filter(|&&off| start <= off && off < end_exclusive)
        .count()
}

fn diff_spans(anchor: &[u8], target: &[u8]) -> Vec<(usize, usize)> {
    let mut diffs = anchor
        .iter()
        .zip(target.iter())
        .enumerate()
        .filter_map(|(idx, (a, b))| if a != b { Some(idx) } else { None })
        .peekable();
    let mut spans = Vec::new();
    while let Some(start) = diffs.next() {
        let mut end = start;
        while let Some(next) = diffs.peek() {
            if *next == end + 1 {
                end = *next;
                diffs.next();
            } else {
                break;
            }
        }
        spans.push((start, end));
    }
    spans
}

fn analyze_pair(pair_dir: &Path) -> Result<PairReport, Box<dyn Error>> {
    let pair_id = pair_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("invalid pair dir name")?
        .to_string();
    let prep: PrepSummary =
        serde_json::from_slice(&fs::read(pair_dir.join("PREP_SUMMARY.json"))?)?;
    let anchor_meta_path = find_metadata(&pair_dir.join("anchor"))?;
    let target_meta_path = find_metadata(&pair_dir.join("target"))?;
    let anchor_meta: GeneratedConvMetadata = serde_json::from_slice(&fs::read(&anchor_meta_path)?)?;
    let target_meta: GeneratedConvMetadata = serde_json::from_slice(&fs::read(&target_meta_path)?)?;
    let anchor_stream = fs::read(pair_dir.join("anchor_param_stream.bin"))?;
    let target_stream = fs::read(pair_dir.join("target_param_stream.bin"))?;
    if anchor_stream.len() != target_stream.len() {
        return Err(format!(
            "param stream length mismatch in {}: {} vs {}",
            pair_id,
            anchor_stream.len(),
            target_stream.len()
        )
        .into());
    }

    let kernel_size = anchor_meta.kernel_size;
    let in_channels = anchor_meta.in_channels;
    let out_channels = anchor_meta.out_channels;
    let weight_bytes = kernel_size
        .checked_mul(kernel_size)
        .and_then(|v| v.checked_mul(in_channels))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or("weight size overflow")?;
    let prefix_total = anchor_stream
        .len()
        .checked_sub(weight_bytes)
        .ok_or("param stream smaller than inferred weight bytes")?;
    let diff_offsets: Vec<usize> = anchor_stream
        .iter()
        .zip(target_stream.iter())
        .enumerate()
        .filter_map(|(idx, (a, b))| if a != b { Some(idx) } else { None })
        .collect();

    let mut blocks = Vec::new();
    let mut block_start = 0usize;
    let mut oc_base = 0usize;
    let mut total_scale_diffs = 0usize;
    let mut total_zp_diffs = 0usize;
    let mut total_weight_diffs = 0usize;
    while oc_base < out_channels {
        let bw = (out_channels - oc_base).min(64);
        let prefix_bytes = bw
            .checked_mul(8)
            .ok_or("prefix bytes overflow")?;
        let weight_region_bytes = bw
            .checked_mul(in_channels)
            .and_then(|v| v.checked_mul(kernel_size))
            .and_then(|v| v.checked_mul(kernel_size))
            .ok_or("block weight bytes overflow")?;
        let scale_start = block_start;
        let scale_end = scale_start + bw * 4;
        let zp_start = scale_end;
        let zp_end = zp_start + bw * 4;
        let weight_start = zp_end;
        let weight_end = weight_start + weight_region_bytes;
        let diff_scale = count_diffs_in_range(&diff_offsets, scale_start, scale_end);
        let diff_zp = count_diffs_in_range(&diff_offsets, zp_start, zp_end);
        let diff_weight = count_diffs_in_range(&diff_offsets, weight_start, weight_end);
        total_scale_diffs += diff_scale;
        total_zp_diffs += diff_zp;
        total_weight_diffs += diff_weight;
        blocks.push(BlockReport {
            block_index: blocks.len(),
            out_channel_start: oc_base,
            out_channel_count: bw,
            block_start,
            scale_prefix_start: scale_start,
            scale_prefix_end: scale_end,
            zero_point_prefix_start: zp_start,
            zero_point_prefix_end: zp_end,
            weight_start,
            weight_end,
            diff_bytes_in_scale_prefix: diff_scale,
            diff_bytes_in_zero_point_prefix: diff_zp,
            diff_bytes_in_weight_region: diff_weight,
        });
        block_start += prefix_bytes + weight_region_bytes;
        oc_base += bw;
    }
    if block_start != anchor_stream.len() {
        return Err(format!(
            "block layout did not cover stream for {}: covered {} of {}",
            pair_id,
            block_start,
            anchor_stream.len()
        )
        .into());
    }

    let all_diffs_confined_to_scale_prefix = total_zp_diffs == 0 && total_weight_diffs == 0;
    let conclusion = if anchor_meta.kernel_sha256 == target_meta.kernel_sha256
        && all_diffs_confined_to_scale_prefix
    {
        "Parameter delta is confined to blockwise effective-scale prefix bytes; stored zero-point bytes and weight region are unchanged.".to_string()
    } else if total_weight_diffs == 0 && total_zp_diffs == 0 {
        "Parameter delta is prefix-only, but not proven confined to effective-scale bytes alone.".to_string()
    } else if total_weight_diffs == 0 {
        "Parameter delta touches prefix/meta regions only; weight region is unchanged.".to_string()
    } else {
        "Parameter delta reaches weight bytes; layout/weight mechanics remain unsolved.".to_string()
    };

    Ok(PairReport {
        pair_id,
        anchor_model_name: anchor_meta.model_name,
        target_model_name: target_meta.model_name,
        kernel_size,
        in_channels,
        out_channels,
        anchor_hw: (anchor_meta.height, anchor_meta.width),
        target_hw: (target_meta.height, target_meta.width),
        kernel_sha256_equal: anchor_meta.kernel_sha256 == target_meta.kernel_sha256,
        input_scale_equal: anchor_meta.input_tensor.quantization.0
            == target_meta.input_tensor.quantization.0,
        output_scale_equal: anchor_meta.output_tensor.quantization.0
            == target_meta.output_tensor.quantization.0,
        anchor_output_scale: anchor_meta.output_tensor.quantization.0,
        target_output_scale: target_meta.output_tensor.quantization.0,
        param_len: anchor_stream.len(),
        expected_weight_bytes: weight_bytes,
        inferred_prefix_total_bytes: prefix_total,
        inferred_blocks: blocks,
        diff_byte_count: diff_offsets.len(),
        diff_span_count: diff_spans(&anchor_stream, &target_stream).len(),
        diff_spans: diff_spans(&anchor_stream, &target_stream),
        diff_bytes_in_scale_prefix: total_scale_diffs,
        diff_bytes_in_zero_point_prefix: total_zp_diffs,
        diff_bytes_in_weight_region: total_weight_diffs,
        weights_equal: total_weight_diffs == 0,
        zero_points_equal: total_zp_diffs == 0,
        all_diffs_confined_to_scale_prefix,
        conclusion,
        prep_param_equal: prep.param_equal,
        eo_rule_count: prep.eo_rule_count,
        pc_rule_count: prep.pc_rule_count,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "conv_k3_param_anatomy".to_string());
    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut run_dir: Option<PathBuf> = None;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--run-dir" => {
                run_dir = Some(PathBuf::from(next_arg(&args, &mut i, "--run-dir")?));
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let run_dir = run_dir.ok_or("missing --run-dir")?;
    let mut pair_dirs: Vec<PathBuf> = fs::read_dir(&run_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir() && path.join("PREP_SUMMARY.json").exists())
        .collect();
    pair_dirs.sort();
    if pair_dirs.is_empty() {
        return Err(format!("no pair dirs found in {}", run_dir.display()).into());
    }

    let mut root_lines = vec![format!("run_dir={}", run_dir.display())];
    for pair_dir in pair_dirs {
        let report = analyze_pair(&pair_dir)?;
        let json_path = pair_dir.join("PARAM_ANATOMY.json");
        let txt_path = pair_dir.join("PARAM_ANATOMY.txt");
        fs::write(&json_path, serde_json::to_vec_pretty(&report)?)?;

        let mut lines = Vec::new();
        lines.push(format!(
            "pair={} anchor_hw={}x{} target_hw={}x{} kernel_size={} in_channels={} out_channels={}",
            report.pair_id,
            report.anchor_hw.0,
            report.anchor_hw.1,
            report.target_hw.0,
            report.target_hw.1,
            report.kernel_size,
            report.in_channels,
            report.out_channels
        ));
        lines.push(format!(
            "kernel_sha256_equal={} input_scale_equal={} output_scale_equal={} anchor_output_scale={} target_output_scale={}",
            report.kernel_sha256_equal,
            report.input_scale_equal,
            report.output_scale_equal,
            report.anchor_output_scale,
            report.target_output_scale
        ));
        lines.push(format!(
            "param_len={} expected_weight_bytes={} inferred_prefix_total_bytes={} diff_byte_count={} diff_span_count={}",
            report.param_len,
            report.expected_weight_bytes,
            report.inferred_prefix_total_bytes,
            report.diff_byte_count,
            report.diff_span_count
        ));
        lines.push(format!(
            "diff_bytes_in_scale_prefix={} diff_bytes_in_zero_point_prefix={} diff_bytes_in_weight_region={} weights_equal={} zero_points_equal={} all_diffs_confined_to_scale_prefix={}",
            report.diff_bytes_in_scale_prefix,
            report.diff_bytes_in_zero_point_prefix,
            report.diff_bytes_in_weight_region,
            report.weights_equal,
            report.zero_points_equal,
            report.all_diffs_confined_to_scale_prefix
        ));
        for block in &report.inferred_blocks {
            lines.push(format!(
                "block{}: oc={}..{} scale={}..{} zp={}..{} weight={}..{} diff_scale={} diff_zp={} diff_weight={}",
                block.block_index,
                block.out_channel_start,
                block.out_channel_start + block.out_channel_count - 1,
                block.scale_prefix_start,
                block.scale_prefix_end.saturating_sub(1),
                block.zero_point_prefix_start,
                block.zero_point_prefix_end.saturating_sub(1),
                block.weight_start,
                block.weight_end.saturating_sub(1),
                block.diff_bytes_in_scale_prefix,
                block.diff_bytes_in_zero_point_prefix,
                block.diff_bytes_in_weight_region
            ));
        }
        lines.push(format!("conclusion={}", report.conclusion));
        fs::write(&txt_path, lines.join("\n") + "\n")?;
        root_lines.push(format!(
            "[{}] diff_bytes={} scale_only={} weights_equal={} conclusion={}",
            report.pair_id,
            report.diff_byte_count,
            report.all_diffs_confined_to_scale_prefix,
            report.weights_equal,
            report.conclusion
        ));
    }

    fs::write(run_dir.join("PARAM_ANATOMY_SUMMARY.txt"), root_lines.join("\n") + "\n")?;
    println!("{}", run_dir.join("PARAM_ANATOMY_SUMMARY.txt").display());
    Ok(())
}
