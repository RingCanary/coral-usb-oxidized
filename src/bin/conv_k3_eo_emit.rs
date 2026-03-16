use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct FamilySpec {
    schema_version: u32,
    family_id: String,
    same_product: usize,
    anchor_height: usize,
    anchor_width: usize,
    eo_payload_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: String,
    bias: bool,
    asset_root: String,
    regimes: Vec<RegimeSpec>,
}

#[derive(Debug, Deserialize)]
struct RegimeSpec {
    name: String,
    channels: usize,
    anchor_compiled_model: String,
    anchor_uncompiled_model: String,
    anchor_metadata: String,
    targets: Vec<TargetSpec>,
}

#[derive(Debug, Deserialize)]
struct TargetSpec {
    height: usize,
    width: usize,
    target_model: String,
    target_compiled_model: String,
    target_metadata: String,
    source_kind: String,
    rules: Vec<[usize; 2]>,
}

#[derive(Debug, Serialize)]
struct EmitReport {
    family_id: String,
    asset_root: String,
    regime_name: String,
    channels: usize,
    same_product: usize,
    anchor_height: usize,
    anchor_width: usize,
    target_height: usize,
    target_width: usize,
    eo_payload_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: String,
    bias: bool,
    source_kind: String,
    rule_count: usize,
    anchor_compiled_model: String,
    anchor_uncompiled_model: String,
    anchor_metadata: String,
    target_model: String,
    target_compiled_model: String,
    target_metadata: String,
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --family-spec PATH --channels N --target-height N [--out-patchspec PATH] [--out-report PATH]"
    );
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{flag} requires a value"));
    }
    Ok(args[*idx].clone())
}

fn resolve_relative(base: &Path, rel: &str) -> String {
    base.join(rel).display().to_string()
}

fn write_patchspec(path: &Path, payload_len: usize, rules: &[[usize; 2]]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if rules.is_empty() {
        fs::write(path, "# no EO patch required for this target\n")?;
        return Ok(());
    }
    let mut lines = Vec::with_capacity(rules.len() + 1);
    lines.push("# emitted by conv_k3_eo_emit".to_string());
    for [offset, value] in rules {
        lines.push(format!("{payload_len} {offset} {value}"));
    }
    fs::write(path, lines.join("\n") + "\n")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "conv_k3_eo_emit".to_string());
    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut family_spec_path: Option<PathBuf> = None;
    let mut channels: Option<usize> = None;
    let mut target_height: Option<usize> = None;
    let mut out_patchspec: Option<PathBuf> = None;
    let mut out_report: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--family-spec" => {
                family_spec_path = Some(PathBuf::from(next_arg(&args, &mut i, "--family-spec")?))
            }
            "--channels" => {
                channels = Some(
                    next_arg(&args, &mut i, "--channels")?
                        .parse::<usize>()
                        .map_err(|e| format!("--channels invalid integer: {e}"))?,
                )
            }
            "--target-height" => {
                target_height = Some(
                    next_arg(&args, &mut i, "--target-height")?
                        .parse::<usize>()
                        .map_err(|e| format!("--target-height invalid integer: {e}"))?,
                )
            }
            "--out-patchspec" => {
                out_patchspec = Some(PathBuf::from(next_arg(&args, &mut i, "--out-patchspec")?))
            }
            "--out-report" => {
                out_report = Some(PathBuf::from(next_arg(&args, &mut i, "--out-report")?))
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let family_spec_path = family_spec_path.ok_or("missing --family-spec")?;
    let channels = channels.ok_or("missing --channels")?;
    let target_height = target_height.ok_or("missing --target-height")?;
    let spec_dir = family_spec_path
        .parent()
        .ok_or("family spec path must have a parent directory")?;
    let spec: FamilySpec = serde_json::from_slice(&fs::read(&family_spec_path)?)?;
    if spec.schema_version != 1 {
        return Err(format!("unsupported schema_version {}", spec.schema_version).into());
    }

    let regime = spec
        .regimes
        .iter()
        .find(|r| r.channels == channels)
        .ok_or_else(|| format!("unsupported channels={} in {}", channels, family_spec_path.display()))?;
    let target = regime
        .targets
        .iter()
        .find(|t| t.height == target_height)
        .ok_or_else(|| format!("unsupported target height={} for regime {}", target_height, regime.name))?;

    if target.height
        .checked_mul(target.width)
        .ok_or("target area overflow")?
        != spec.same_product
    {
        return Err(format!(
            "target area mismatch for {}: {}x{} != same_product {}",
            regime.name, target.height, target.width, spec.same_product
        )
        .into());
    }

    if let Some(path) = out_patchspec.as_ref() {
        write_patchspec(path, spec.eo_payload_len, &target.rules)?;
    }

    let report = EmitReport {
        family_id: spec.family_id,
        asset_root: spec.asset_root,
        regime_name: regime.name.clone(),
        channels,
        same_product: spec.same_product,
        anchor_height: spec.anchor_height,
        anchor_width: spec.anchor_width,
        target_height: target.height,
        target_width: target.width,
        eo_payload_len: spec.eo_payload_len,
        kernel_size: spec.kernel_size,
        stride: spec.stride,
        padding: spec.padding,
        bias: spec.bias,
        source_kind: target.source_kind.clone(),
        rule_count: target.rules.len(),
        anchor_compiled_model: resolve_relative(spec_dir, &regime.anchor_compiled_model),
        anchor_uncompiled_model: resolve_relative(spec_dir, &regime.anchor_uncompiled_model),
        anchor_metadata: resolve_relative(spec_dir, &regime.anchor_metadata),
        target_model: resolve_relative(spec_dir, &target.target_model),
        target_compiled_model: resolve_relative(spec_dir, &target.target_compiled_model),
        target_metadata: resolve_relative(spec_dir, &target.target_metadata),
    };

    if let Some(path) = out_report.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }

    println!(
        "family_id={} regime={} channels={} target={}x{} rule_count={} source={}",
        report.family_id,
        report.regime_name,
        report.channels,
        report.target_height,
        report.target_width,
        report.rule_count,
        report.source_kind
    );
    if report.rule_count == 0 {
        println!("note=no_eo_patch_needed");
    }

    Ok(())
}
