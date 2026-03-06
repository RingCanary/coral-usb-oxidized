use coral_usb_oxidized::{executable_type_name, extract_serialized_executables_from_tflite};
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct DumpMetadata {
    model_path: String,
    out_path: String,
    executable_index: usize,
    executable_type: i16,
    executable_type_name: String,
    param_len: usize,
    param_fnv1a64_hex: String,
}

fn usage(program: &str) {
    eprintln!("Usage: {program} --model PATH --out PATH [--exec-index N] [--metadata-out PATH]");
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{} requires a value", flag));
    }
    Ok(args[*idx].clone())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "model_param_stream_dump".to_string());

    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut model_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut metadata_out: Option<PathBuf> = None;
    let mut exec_index: Option<usize> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--model" => {
                model_path = Some(PathBuf::from(next_arg(&args, &mut i, "--model")?));
            }
            "--out" => {
                out_path = Some(PathBuf::from(next_arg(&args, &mut i, "--out")?));
            }
            "--metadata-out" => {
                metadata_out = Some(PathBuf::from(next_arg(&args, &mut i, "--metadata-out")?));
            }
            "--exec-index" => {
                let raw = next_arg(&args, &mut i, "--exec-index")?;
                exec_index = Some(
                    raw.parse::<usize>()
                        .map_err(|e| format!("invalid --exec-index '{}': {}", raw, e))?,
                );
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        i += 1;
    }

    let model_path = model_path.ok_or("missing --model")?;
    let out_path = out_path.ok_or("missing --out")?;

    let model_bytes = fs::read(&model_path)?;
    let executables = extract_serialized_executables_from_tflite(&model_bytes)?;

    let selected = if let Some(idx) = exec_index {
        executables
            .iter()
            .find(|e| e.executable_index == idx)
            .ok_or_else(|| format!("no executable with index {}", idx))?
    } else {
        executables
            .iter()
            .find(|e| e.executable_type == 1 && !e.parameters_stream.is_empty())
            .or_else(|| executables.iter().find(|e| !e.parameters_stream.is_empty()))
            .ok_or("no executable with parameters_stream found")?
    };

    if selected.parameters_stream.is_empty() {
        return Err(format!(
            "selected executable idx={} type={}({}) has empty parameters_stream",
            selected.executable_index,
            selected.executable_type,
            executable_type_name(selected.executable_type)
        )
        .into());
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, &selected.parameters_stream)?;

    let meta = DumpMetadata {
        model_path: model_path.to_string_lossy().into_owned(),
        out_path: out_path.to_string_lossy().into_owned(),
        executable_index: selected.executable_index,
        executable_type: selected.executable_type,
        executable_type_name: executable_type_name(selected.executable_type).to_string(),
        param_len: selected.parameters_stream.len(),
        param_fnv1a64_hex: format!("0x{:016x}", fnv1a64(&selected.parameters_stream)),
    };

    if let Some(path) = metadata_out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(&meta)?)?;
    }

    println!(
        "wrote param stream: exec_idx={} type={}({}) len={} fnv={} out={}",
        meta.executable_index,
        meta.executable_type,
        meta.executable_type_name,
        meta.param_len,
        meta.param_fnv1a64_hex,
        out_path.display()
    );

    Ok(())
}
