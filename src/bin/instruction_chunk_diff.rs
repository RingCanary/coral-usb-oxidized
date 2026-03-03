use coral_usb_oxidized::extract_instruction_chunk_from_serialized_executable;
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Debug)]
struct VariantArg {
    name: String,
    path: PathBuf,
}

#[derive(Debug, Serialize)]
struct VariantDiff {
    name: String,
    path: String,
    payload_len: usize,
    changed_count: usize,
    changed_offsets: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct DiffReport {
    baseline_path: String,
    chunk_index: usize,
    payload_len: usize,
    variants: Vec<VariantDiff>,
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --baseline PATH --variant NAME:PATH [--variant NAME:PATH ...] --out-json PATH [--chunk-index N]"
    );
}

fn parse_variant_arg(raw: &str) -> Result<VariantArg, String> {
    let Some((name, path)) = raw.split_once(':') else {
        return Err(format!("invalid --variant '{}': expected NAME:PATH", raw));
    };
    if name.trim().is_empty() {
        return Err(format!("invalid --variant '{}': empty NAME", raw));
    }
    if path.trim().is_empty() {
        return Err(format!("invalid --variant '{}': empty PATH", raw));
    }
    Ok(VariantArg {
        name: name.to_string(),
        path: PathBuf::from(path),
    })
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{} requires a value", flag));
    }
    Ok(args[*idx].clone())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "instruction_chunk_diff".to_string());

    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut baseline_path: Option<PathBuf> = None;
    let mut variants: Vec<VariantArg> = Vec::new();
    let mut chunk_index: usize = 0;
    let mut out_json: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--baseline" => {
                baseline_path = Some(PathBuf::from(next_arg(&args, &mut i, "--baseline")?));
            }
            "--variant" => {
                let raw = next_arg(&args, &mut i, "--variant")?;
                variants.push(parse_variant_arg(&raw)?);
            }
            "--chunk-index" => {
                let raw = next_arg(&args, &mut i, "--chunk-index")?;
                chunk_index = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --chunk-index '{}': {}", raw, e))?;
            }
            "--out-json" => {
                out_json = Some(PathBuf::from(next_arg(&args, &mut i, "--out-json")?));
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        i += 1;
    }

    let baseline_path = baseline_path.ok_or("missing --baseline")?;
    let out_json = out_json.ok_or("missing --out-json")?;
    if variants.is_empty() {
        return Err("at least one --variant is required".into());
    }

    let baseline_blob = fs::read(&baseline_path)?;
    let baseline_chunk =
        extract_instruction_chunk_from_serialized_executable(&baseline_blob, chunk_index)?;

    let mut reports = Vec::with_capacity(variants.len());
    for var in variants {
        let blob = fs::read(&var.path)?;
        let chunk = extract_instruction_chunk_from_serialized_executable(&blob, chunk_index)?;
        if chunk.len() != baseline_chunk.len() {
            return Err(format!(
                "payload len mismatch for variant '{}' baseline={} variant={}",
                var.name,
                baseline_chunk.len(),
                chunk.len()
            )
            .into());
        }
        let mut changed_offsets = Vec::new();
        for (idx, (&b0, &b1)) in baseline_chunk.iter().zip(chunk.iter()).enumerate() {
            if b0 != b1 {
                changed_offsets.push(idx);
            }
        }
        reports.push(VariantDiff {
            name: var.name,
            path: var.path.to_string_lossy().into_owned(),
            payload_len: chunk.len(),
            changed_count: changed_offsets.len(),
            changed_offsets,
        });
    }

    let report = DiffReport {
        baseline_path: baseline_path.to_string_lossy().into_owned(),
        chunk_index,
        payload_len: baseline_chunk.len(),
        variants: reports,
    };

    if let Some(parent) = out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_json, serde_json::to_string_pretty(&report)?)?;
    println!(
        "wrote diff report: payload_len={} variants={} out={}",
        report.payload_len,
        report.variants.len(),
        out_json.display()
    );

    Ok(())
}
