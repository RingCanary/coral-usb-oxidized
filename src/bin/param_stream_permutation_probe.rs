use coral_usb_oxidized::{executable_type_name, extract_serialized_executables_from_tflite};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

type DynError = Box<dyn std::error::Error>;

#[derive(Debug, Clone, Serialize)]
struct DiffCount {
    byte: u8,
    expected: usize,
    actual: usize,
    delta: isize,
}

#[derive(Debug, Clone, Serialize)]
struct MappingSample {
    stream_index: usize,
    source_index: usize,
}

#[derive(Debug, Clone, Serialize)]
struct MappingStats {
    fixed_points: usize,
    fixed_point_fraction: f64,
    identity_match_count: usize,
    identity_match_fraction: f64,
    transpose_match_count: Option<usize>,
    transpose_match_fraction: Option<f64>,
    longest_consecutive_source_run: usize,
    sample: Vec<MappingSample>,
    non_unique_value_note: String,
}

#[derive(Debug, Clone, Serialize)]
struct ExecutableMeta {
    package_index: usize,
    executable_index: usize,
    executable_type: i16,
    executable_type_name: String,
    payload_len: usize,
    parameter_region: Option<(usize, usize)>,
    instruction_chunk_lens: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct FormulaCheck {
    name: String,
    mismatch_count: usize,
    mismatch_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    generated_utc: String,
    compiled_model: String,
    modulus: usize,
    expected_mode: String,
    dim: Option<usize>,
    param_len: usize,
    param_fnv1a64_hex: String,
    expected_fnv1a64_hex: String,
    is_permutation: bool,
    histogram_l1_distance: usize,
    top_histogram_deltas: Vec<DiffCount>,
    executable: ExecutableMeta,
    formula_checks: Vec<FormulaCheck>,
    mapping: Option<MappingStats>,
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --compiled PATH [options]\n\n\
Options:\n\
  --compiled PATH         Compiled *_edgetpu.tflite model\n\
  --modulus N             Expected pattern modulus for source bytes i % N (default: 251)\n\
  --expected-mode MODE    raw_mod|signed_reinterpret (default: raw_mod)\n\
  --dim N                 Optional dense dim (enables transpose-match statistic when len == N*N)\n\
  --sample N              Mapping sample size (default: 64)\n\
  --out-json PATH         Write JSON report to file\n\
  --out-map-bin PATH      Optional mapping witness dump (u32 LE source indices)\n\
  -h, --help              Show this help\n"
    );
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

fn select_param_executable(
    model_bytes: &[u8],
) -> Result<(ExecutableMeta, Vec<u8>), coral_usb_oxidized::DenseGemmError> {
    let executables = extract_serialized_executables_from_tflite(model_bytes)?;
    let selected = executables
        .iter()
        .find(|e| {
            executable_type_name(e.executable_type) == "PARAMETER_CACHING"
                && !e.parameters_stream.is_empty()
        })
        .or_else(|| executables.iter().find(|e| !e.parameters_stream.is_empty()))
        .ok_or_else(|| {
            coral_usb_oxidized::DenseGemmError::InvalidTemplate(
                "no executable with non-empty parameter stream found".to_string(),
            )
        })?;

    let meta = ExecutableMeta {
        package_index: selected.package_index,
        executable_index: selected.executable_index,
        executable_type: selected.executable_type,
        executable_type_name: executable_type_name(selected.executable_type).to_string(),
        payload_len: selected.payload.len(),
        parameter_region: selected.parameter_region,
        instruction_chunk_lens: selected
            .instruction_bitstreams
            .iter()
            .map(|x| x.len())
            .collect(),
    };

    Ok((meta, selected.parameters_stream.clone()))
}

fn main() -> Result<(), DynError> {
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "param_stream_permutation_probe".to_string());

    let mut compiled: Option<PathBuf> = None;
    let mut modulus: usize = 251;
    let mut expected_mode = "raw_mod".to_string();
    let mut dim: Option<usize> = None;
    let mut sample_n: usize = 64;
    let mut out_json: Option<PathBuf> = None;
    let mut out_map_bin: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--compiled" => {
                i += 1;
                compiled = Some(PathBuf::from(
                    args.get(i).ok_or("--compiled requires PATH")?,
                ));
            }
            "--modulus" => {
                i += 1;
                let raw = args.get(i).ok_or("--modulus requires value")?;
                modulus = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --modulus '{}': {e}", raw))?;
            }
            "--expected-mode" => {
                i += 1;
                let raw = args.get(i).ok_or("--expected-mode requires value")?;
                match raw.as_str() {
                    "raw_mod" | "signed_reinterpret" => {
                        expected_mode = raw.to_string();
                    }
                    _ => {
                        return Err(format!(
                            "invalid --expected-mode '{}'; expected raw_mod|signed_reinterpret",
                            raw
                        )
                        .into())
                    }
                }
            }
            "--dim" => {
                i += 1;
                let raw = args.get(i).ok_or("--dim requires value")?;
                dim = Some(
                    raw.parse::<usize>()
                        .map_err(|e| format!("invalid --dim '{}': {e}", raw))?,
                );
            }
            "--sample" => {
                i += 1;
                let raw = args.get(i).ok_or("--sample requires value")?;
                sample_n = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --sample '{}': {e}", raw))?;
            }
            "--out-json" => {
                i += 1;
                out_json = Some(PathBuf::from(
                    args.get(i).ok_or("--out-json requires PATH")?,
                ));
            }
            "--out-map-bin" => {
                i += 1;
                out_map_bin = Some(PathBuf::from(
                    args.get(i).ok_or("--out-map-bin requires PATH")?,
                ));
            }
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            other => {
                return Err(format!("unknown arg: {other} (use --help)").into());
            }
        }
        i += 1;
    }

    let compiled = compiled.ok_or("--compiled is required")?;
    if modulus == 0 || modulus > 256 {
        return Err("--modulus must be in [1,256]".into());
    }

    let model_bytes = fs::read(&compiled)
        .map_err(|e| format!("failed to read compiled model {}: {e}", compiled.display()))?;

    let (exec_meta, actual) = select_param_executable(&model_bytes)
        .map_err(|e| format!("failed to extract param stream: {e}"))?;

    let expected: Vec<u8> = match expected_mode.as_str() {
        "raw_mod" => (0..actual.len()).map(|idx| (idx % modulus) as u8).collect(),
        "signed_reinterpret" => (0..actual.len())
            .map(|idx| {
                let v = (idx % modulus) as i16;
                ((v - 128).rem_euclid(256)) as u8
            })
            .collect(),
        _ => unreachable!("validated by arg parser"),
    };

    let mut hist_actual = [0usize; 256];
    let mut hist_expected = [0usize; 256];
    for &b in &actual {
        hist_actual[b as usize] += 1;
    }
    for &b in &expected {
        hist_expected[b as usize] += 1;
    }

    let is_permutation = hist_actual == hist_expected;
    let histogram_l1_distance = (0..256)
        .map(|k| hist_actual[k].abs_diff(hist_expected[k]))
        .sum::<usize>();

    let mut top_histogram_deltas = (0..256)
        .filter_map(|k| {
            let d = hist_actual[k] as isize - hist_expected[k] as isize;
            if d == 0 {
                None
            } else {
                Some(DiffCount {
                    byte: k as u8,
                    expected: hist_expected[k],
                    actual: hist_actual[k],
                    delta: d,
                })
            }
        })
        .collect::<Vec<_>>();
    top_histogram_deltas.sort_by(|a, b| {
        b.delta
            .abs()
            .cmp(&a.delta.abs())
            .then_with(|| a.byte.cmp(&b.byte))
    });
    top_histogram_deltas.truncate(16);

    let mut formula_checks = Vec::<FormulaCheck>::new();
    if let Some(d) = dim {
        if d > 0 && actual.len() == d * d {
            let tile = 64usize;
            if d % tile == 0 {
                let tile_cols = d / tile;
                let tile_rows = d / tile;
                let tile_bytes = tile * tile;

                let mut pred_rowmajor_cr4 = vec![0u8; actual.len()];
                let mut pred_colmajor_rc4 = vec![0u8; actual.len()];

                for r in 0..d {
                    for c in 0..d {
                        let src_idx = r * d + c;

                        let off_rowmajor_cr4 = (r / tile) * (tile_cols * tile_bytes)
                            + (c / tile) * tile_bytes
                            + ((c % tile) / 4) * (tile * 4)
                            + (r % tile) * 4
                            + (c % 4);
                        pred_rowmajor_cr4[off_rowmajor_cr4] = expected[src_idx];

                        let off_colmajor_rc4 = (c / tile) * (tile_rows * tile_bytes)
                            + (r / tile) * tile_bytes
                            + ((r % tile) / 4) * (tile * 4)
                            + (c % tile) * 4
                            + (r % 4);
                        pred_colmajor_rc4[off_colmajor_rc4] = expected[src_idx];
                    }
                }

                let mismatch_rowmajor_cr4 = actual
                    .iter()
                    .zip(pred_rowmajor_cr4.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                let mismatch_colmajor_rc4 = actual
                    .iter()
                    .zip(pred_colmajor_rc4.iter())
                    .filter(|(a, b)| a != b)
                    .count();

                let n = actual.len() as f64;
                formula_checks.push(FormulaCheck {
                    name: "tile64_rowmajor_tiles_local_cr4".to_string(),
                    mismatch_count: mismatch_rowmajor_cr4,
                    mismatch_fraction: mismatch_rowmajor_cr4 as f64 / n,
                });
                formula_checks.push(FormulaCheck {
                    name: "tile64_colmajor_tiles_local_rc4".to_string(),
                    mismatch_count: mismatch_colmajor_rc4,
                    mismatch_fraction: mismatch_colmajor_rc4 as f64 / n,
                });
            }
        }
    }

    let mut mapping_report = None;

    if is_permutation {
        let mut positions_by_value = vec![Vec::<usize>::new(); 256];
        for (src_idx, &b) in expected.iter().enumerate() {
            positions_by_value[b as usize].push(src_idx);
        }
        let mut next_pos = [0usize; 256];
        let mut mapping = vec![0u32; actual.len()];

        for (stream_idx, &b) in actual.iter().enumerate() {
            let bi = b as usize;
            let pos_idx = next_pos[bi];
            let src = positions_by_value[bi]
                .get(pos_idx)
                .ok_or("internal mismatch while constructing mapping witness")?;
            mapping[stream_idx] = *src as u32;
            next_pos[bi] += 1;
        }

        if let Some(path) = out_map_bin.as_ref() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out = Vec::<u8>::with_capacity(mapping.len() * 4);
            for &v in &mapping {
                out.extend_from_slice(&v.to_le_bytes());
            }
            fs::write(path, out)?;
        }

        let fixed_points = mapping
            .iter()
            .enumerate()
            .filter(|(i, src)| **src as usize == *i)
            .count();
        let identity_match_count = fixed_points;

        let mut longest_consecutive_source_run = 1usize;
        let mut cur_run = 1usize;
        for idx in 1..mapping.len() {
            if mapping[idx] == mapping[idx - 1].wrapping_add(1) {
                cur_run += 1;
                if cur_run > longest_consecutive_source_run {
                    longest_consecutive_source_run = cur_run;
                }
            } else {
                cur_run = 1;
            }
        }

        let (transpose_match_count, transpose_match_fraction) = if let Some(d) = dim {
            if d > 0 && mapping.len() == d * d {
                let mut count = 0usize;
                for (stream_idx, &src_u32) in mapping.iter().enumerate() {
                    let src = src_u32 as usize;
                    let expected_src = (stream_idx % d) * d + (stream_idx / d);
                    if src == expected_src {
                        count += 1;
                    }
                }
                (Some(count), Some(count as f64 / mapping.len() as f64))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let sample = mapping
            .iter()
            .take(sample_n)
            .enumerate()
            .map(|(stream_index, &source_index)| MappingSample {
                stream_index,
                source_index: source_index as usize,
            })
            .collect::<Vec<_>>();

        mapping_report = Some(MappingStats {
            fixed_points,
            fixed_point_fraction: if mapping.is_empty() {
                0.0
            } else {
                fixed_points as f64 / mapping.len() as f64
            },
            identity_match_count,
            identity_match_fraction: if mapping.is_empty() {
                0.0
            } else {
                identity_match_count as f64 / mapping.len() as f64
            },
            transpose_match_count,
            transpose_match_fraction,
            longest_consecutive_source_run,
            sample,
            non_unique_value_note: "Mapping is a deterministic witness built by stable queue assignment per byte value; with repeated values (e.g. modulus < len), mapping is not unique.".to_string(),
        });
    }

    let report = Report {
        generated_utc: now_utc_rfc3339(),
        compiled_model: compiled.display().to_string(),
        modulus,
        expected_mode,
        dim,
        param_len: actual.len(),
        param_fnv1a64_hex: format!("0x{:016x}", fnv1a64(&actual)),
        expected_fnv1a64_hex: format!("0x{:016x}", fnv1a64(&expected)),
        is_permutation,
        histogram_l1_distance,
        top_histogram_deltas,
        executable: exec_meta,
        formula_checks,
        mapping: mapping_report,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = out_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, json.as_bytes())?;
        eprintln!("wrote {}", path.display());
    } else {
        println!("{json}");
    }

    Ok(())
}
