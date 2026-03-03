use coral_usb_oxidized::{executable_type_name, extract_serialized_executables_from_tflite};
use serde::Serialize;
use std::cmp::min;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

type DynError = Box<dyn std::error::Error>;

#[derive(Debug, Clone)]
struct ModelSpec {
    name: String,
    path: PathBuf,
}

#[derive(Debug, Clone)]
struct LoadedModel {
    info: ModelInfo,
    param_stream: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelInfo {
    name: String,
    path: String,
    package_index: usize,
    executable_index: usize,
    executable_type: i16,
    executable_type_name: String,
    payload_len: usize,
    parameter_region: Option<(usize, usize)>,
    param_len: usize,
    param_fnv1a64_hex: String,
    instruction_chunk_lens: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct DiffSample {
    offset: usize,
    lhs: u8,
    rhs: u8,
}

#[derive(Debug, Clone, Serialize)]
struct ByteTransition {
    lhs: u8,
    rhs: u8,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonReport {
    lhs: String,
    rhs: String,
    lhs_len: usize,
    rhs_len: usize,
    overlap_len: usize,
    equal_in_overlap: usize,
    changed_in_overlap: usize,
    changed_fraction_in_overlap: f64,
    extra_bytes_lhs: usize,
    extra_bytes_rhs: usize,
    equal_prefix_len: usize,
    equal_suffix_len: usize,
    same_byte_histogram: bool,
    histogram_l1_distance: usize,
    equal_chunk_count_4k: usize,
    changed_chunk_count_4k: usize,
    sample_diffs: Vec<DiffSample>,
    top_byte_transitions: Vec<ByteTransition>,
}

#[derive(Debug, Clone, Serialize)]
struct MultiInvariantReport {
    compared_model_count: usize,
    common_prefix_len: usize,
    invariant_offset_count: usize,
    invariant_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    generated_utc: String,
    max_samples: usize,
    models: Vec<ModelInfo>,
    comparisons: Vec<ComparisonReport>,
    multi_model_invariants: Option<MultiInvariantReport>,
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} --model <name>=<path> [--model <name>=<path> ...] [options]\n\n\
Options:\n\
  --model NAME=PATH        Model label and .tflite path (repeatable; minimum 2)\n\
  --compare A:B            Explicit pair to compare (repeatable). Default: all pairs\n\
  --out-json PATH          Write JSON report to file\n\
  --max-samples N          Max sampled byte diffs and transitions per pair (default: 24)\n\
  -h, --help               Show this help\n"
    );
}

fn parse_model_spec(raw: &str) -> Result<ModelSpec, DynError> {
    let mut parts = raw.splitn(2, '=');
    let name = parts
        .next()
        .ok_or("missing model name")?
        .trim()
        .to_string();
    let path = parts
        .next()
        .ok_or("missing model path (expected NAME=PATH)")?
        .trim();
    if name.is_empty() {
        return Err("model name must be non-empty".into());
    }
    if path.is_empty() {
        return Err("model path must be non-empty".into());
    }
    Ok(ModelSpec {
        name,
        path: PathBuf::from(path),
    })
}

fn parse_compare_pair(raw: &str) -> Result<(String, String), DynError> {
    let mut parts = raw.splitn(2, ':');
    let lhs = parts
        .next()
        .ok_or("missing compare lhs")?
        .trim()
        .to_string();
    let rhs = parts
        .next()
        .ok_or("missing compare rhs (expected A:B)")?
        .trim()
        .to_string();
    if lhs.is_empty() || rhs.is_empty() {
        return Err("compare labels must be non-empty".into());
    }
    Ok((lhs, rhs))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn now_utc_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn load_model(spec: &ModelSpec) -> Result<LoadedModel, DynError> {
    let model_bytes = fs::read(&spec.path)
        .map_err(|e| format!("failed to read model {}: {e}", spec.path.display()))?;
    let executables = extract_serialized_executables_from_tflite(&model_bytes)
        .map_err(|e| format!("failed to extract executables from {}: {e}", spec.path.display()))?;

    let selected = executables
        .iter()
        .find(|e| {
            executable_type_name(e.executable_type) == "PARAMETER_CACHING"
                && !e.parameters_stream.is_empty()
        })
        .or_else(|| executables.iter().find(|e| !e.parameters_stream.is_empty()))
        .ok_or_else(|| {
            format!(
                "no executable with non-empty parameter stream found in {}",
                spec.path.display()
            )
        })?;

    let info = ModelInfo {
        name: spec.name.clone(),
        path: spec.path.display().to_string(),
        package_index: selected.package_index,
        executable_index: selected.executable_index,
        executable_type: selected.executable_type,
        executable_type_name: executable_type_name(selected.executable_type).to_string(),
        payload_len: selected.payload.len(),
        parameter_region: selected.parameter_region,
        param_len: selected.parameters_stream.len(),
        param_fnv1a64_hex: format!("0x{:016x}", fnv1a64(&selected.parameters_stream)),
        instruction_chunk_lens: selected
            .instruction_bitstreams
            .iter()
            .map(|x| x.len())
            .collect(),
    };

    Ok(LoadedModel {
        info,
        param_stream: selected.parameters_stream.clone(),
    })
}

fn compare_streams(
    lhs_name: &str,
    lhs: &[u8],
    rhs_name: &str,
    rhs: &[u8],
    max_samples: usize,
) -> ComparisonReport {
    let lhs_len = lhs.len();
    let rhs_len = rhs.len();
    let overlap = min(lhs_len, rhs_len);

    let mut equal_prefix_len = 0usize;
    while equal_prefix_len < overlap && lhs[equal_prefix_len] == rhs[equal_prefix_len] {
        equal_prefix_len += 1;
    }

    let mut equal_suffix_len = 0usize;
    while equal_suffix_len < overlap.saturating_sub(equal_prefix_len)
        && lhs[lhs_len - 1 - equal_suffix_len] == rhs[rhs_len - 1 - equal_suffix_len]
    {
        equal_suffix_len += 1;
    }

    let mut equal_in_overlap = 0usize;
    let mut changed_in_overlap = 0usize;
    let mut sample_diffs = Vec::new();
    let mut transition_counts: HashMap<(u8, u8), usize> = HashMap::new();

    for i in 0..overlap {
        if lhs[i] == rhs[i] {
            equal_in_overlap += 1;
        } else {
            changed_in_overlap += 1;
            if sample_diffs.len() < max_samples {
                sample_diffs.push(DiffSample {
                    offset: i,
                    lhs: lhs[i],
                    rhs: rhs[i],
                });
            }
            *transition_counts.entry((lhs[i], rhs[i])).or_insert(0) += 1;
        }
    }

    let mut hist_lhs = [0usize; 256];
    let mut hist_rhs = [0usize; 256];
    for &b in lhs {
        hist_lhs[b as usize] += 1;
    }
    for &b in rhs {
        hist_rhs[b as usize] += 1;
    }

    let same_byte_histogram = lhs_len == rhs_len && hist_lhs == hist_rhs;
    let histogram_l1_distance = (0..256)
        .map(|i| hist_lhs[i].abs_diff(hist_rhs[i]))
        .sum::<usize>();

    let chunk_size = 4096usize;
    let mut equal_chunk_count_4k = 0usize;
    let mut changed_chunk_count_4k = 0usize;
    if overlap > 0 {
        let chunk_count = overlap.div_ceil(chunk_size);
        for chunk_idx in 0..chunk_count {
            let start = chunk_idx * chunk_size;
            let end = min(start + chunk_size, overlap);
            if lhs[start..end] == rhs[start..end] {
                equal_chunk_count_4k += 1;
            } else {
                changed_chunk_count_4k += 1;
            }
        }
    }

    let mut top_byte_transitions = transition_counts
        .into_iter()
        .map(|((l, r), c)| ByteTransition {
            lhs: l,
            rhs: r,
            count: c,
        })
        .collect::<Vec<_>>();
    top_byte_transitions.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.lhs.cmp(&b.lhs)));
    top_byte_transitions.truncate(max_samples);

    ComparisonReport {
        lhs: lhs_name.to_string(),
        rhs: rhs_name.to_string(),
        lhs_len,
        rhs_len,
        overlap_len: overlap,
        equal_in_overlap,
        changed_in_overlap,
        changed_fraction_in_overlap: if overlap == 0 {
            0.0
        } else {
            changed_in_overlap as f64 / overlap as f64
        },
        extra_bytes_lhs: lhs_len.saturating_sub(overlap),
        extra_bytes_rhs: rhs_len.saturating_sub(overlap),
        equal_prefix_len,
        equal_suffix_len,
        same_byte_histogram,
        histogram_l1_distance,
        equal_chunk_count_4k,
        changed_chunk_count_4k,
        sample_diffs,
        top_byte_transitions,
    }
}

fn compute_multi_model_invariants(models: &[LoadedModel]) -> Option<MultiInvariantReport> {
    if models.len() < 2 {
        return None;
    }
    let common_len = models
        .iter()
        .map(|m| m.param_stream.len())
        .min()
        .unwrap_or(0usize);
    if common_len == 0 {
        return Some(MultiInvariantReport {
            compared_model_count: models.len(),
            common_prefix_len: 0,
            invariant_offset_count: 0,
            invariant_fraction: 0.0,
        });
    }

    let mut invariant_count = 0usize;
    for i in 0..common_len {
        let b0 = models[0].param_stream[i];
        if models.iter().all(|m| m.param_stream[i] == b0) {
            invariant_count += 1;
        }
    }

    Some(MultiInvariantReport {
        compared_model_count: models.len(),
        common_prefix_len: common_len,
        invariant_offset_count: invariant_count,
        invariant_fraction: invariant_count as f64 / common_len as f64,
    })
}

fn main() -> Result<(), DynError> {
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "param_stream_diff".to_string());

    let mut model_specs: Vec<ModelSpec> = Vec::new();
    let mut compare_pairs: Vec<(String, String)> = Vec::new();
    let mut out_json: Option<PathBuf> = None;
    let mut max_samples = 24usize;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                let raw = args.get(i).ok_or("--model requires NAME=PATH")?;
                model_specs.push(parse_model_spec(raw)?);
            }
            "--compare" => {
                i += 1;
                let raw = args.get(i).ok_or("--compare requires A:B")?;
                compare_pairs.push(parse_compare_pair(raw)?);
            }
            "--out-json" => {
                i += 1;
                let raw = args.get(i).ok_or("--out-json requires PATH")?;
                out_json = Some(PathBuf::from(raw));
            }
            "--max-samples" => {
                i += 1;
                let raw = args.get(i).ok_or("--max-samples requires value")?;
                max_samples = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --max-samples value '{}': {e}", raw))?;
                if max_samples == 0 {
                    return Err("--max-samples must be >= 1".into());
                }
            }
            "-h" | "--help" => {
                print_usage(&program);
                return Ok(());
            }
            other => {
                return Err(format!("unknown arg: {other} (use --help)").into());
            }
        }
        i += 1;
    }

    if model_specs.len() < 2 {
        print_usage(&program);
        return Err("at least two --model entries are required".into());
    }

    let mut seen = HashMap::<String, usize>::new();
    for spec in &model_specs {
        if let Some(prev) = seen.insert(spec.name.clone(), 1usize) {
            let _ = prev;
            return Err(format!("duplicate model name: {}", spec.name).into());
        }
    }

    let mut loaded_models = Vec::new();
    for spec in &model_specs {
        loaded_models.push(load_model(spec)?);
    }

    let by_name: HashMap<String, usize> = loaded_models
        .iter()
        .enumerate()
        .map(|(idx, m)| (m.info.name.clone(), idx))
        .collect();

    let mut comparisons = Vec::new();
    if compare_pairs.is_empty() {
        for lhs_idx in 0..loaded_models.len() {
            for rhs_idx in (lhs_idx + 1)..loaded_models.len() {
                let lhs = &loaded_models[lhs_idx];
                let rhs = &loaded_models[rhs_idx];
                comparisons.push(compare_streams(
                    &lhs.info.name,
                    &lhs.param_stream,
                    &rhs.info.name,
                    &rhs.param_stream,
                    max_samples,
                ));
            }
        }
    } else {
        for (lhs_name, rhs_name) in compare_pairs {
            let lhs_idx = *by_name
                .get(&lhs_name)
                .ok_or_else(|| format!("unknown model label in --compare: {}", lhs_name))?;
            let rhs_idx = *by_name
                .get(&rhs_name)
                .ok_or_else(|| format!("unknown model label in --compare: {}", rhs_name))?;
            let lhs = &loaded_models[lhs_idx];
            let rhs = &loaded_models[rhs_idx];
            comparisons.push(compare_streams(
                &lhs.info.name,
                &lhs.param_stream,
                &rhs.info.name,
                &rhs.param_stream,
                max_samples,
            ));
        }
    }

    let report = Report {
        generated_utc: now_utc_rfc3339(),
        max_samples,
        models: loaded_models.iter().map(|m| m.info.clone()).collect(),
        comparisons,
        multi_model_invariants: compute_multi_model_invariants(&loaded_models),
    };

    let report_json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = out_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, report_json.as_bytes())?;
        eprintln!("wrote {}", path.display());
    } else {
        println!("{report_json}");
    }

    Ok(())
}
