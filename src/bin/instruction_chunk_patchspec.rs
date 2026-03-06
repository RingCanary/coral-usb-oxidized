use coral_usb_oxidized::extract_instruction_chunk_from_serialized_executable;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --base-exec PATH --target-exec PATH --out-patchspec PATH [--chunk-index N]"
    );
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
        .unwrap_or_else(|| "instruction_chunk_patchspec".to_string());

    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut base_exec: Option<PathBuf> = None;
    let mut target_exec: Option<PathBuf> = None;
    let mut out_patchspec: Option<PathBuf> = None;
    let mut chunk_index: usize = 0;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--base-exec" => {
                base_exec = Some(PathBuf::from(next_arg(&args, &mut i, "--base-exec")?));
            }
            "--target-exec" => {
                target_exec = Some(PathBuf::from(next_arg(&args, &mut i, "--target-exec")?));
            }
            "--out-patchspec" => {
                out_patchspec = Some(PathBuf::from(next_arg(&args, &mut i, "--out-patchspec")?));
            }
            "--chunk-index" => {
                let raw = next_arg(&args, &mut i, "--chunk-index")?;
                chunk_index = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --chunk-index '{}': {}", raw, e))?;
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        i += 1;
    }

    let base_exec = base_exec.ok_or("missing --base-exec")?;
    let target_exec = target_exec.ok_or("missing --target-exec")?;
    let out_patchspec = out_patchspec.ok_or("missing --out-patchspec")?;

    let base_blob = fs::read(&base_exec)?;
    let target_blob = fs::read(&target_exec)?;
    let base_chunk = extract_instruction_chunk_from_serialized_executable(&base_blob, chunk_index)?;
    let target_chunk =
        extract_instruction_chunk_from_serialized_executable(&target_blob, chunk_index)?;

    if base_chunk.len() != target_chunk.len() {
        return Err(format!(
            "chunk size mismatch: base={} target={}",
            base_chunk.len(),
            target_chunk.len()
        )
        .into());
    }

    let payload_len = base_chunk.len();
    let mut changed = 0usize;
    let mut lines = vec![format!(
        "# exact diff patchspec generated from base={} target={} chunk_index={}",
        base_exec.display(),
        target_exec.display(),
        chunk_index
    )];
    for (offset, (&b0, &b1)) in base_chunk.iter().zip(target_chunk.iter()).enumerate() {
        if b0 != b1 {
            lines.push(format!("{} {} 0x{:02x}", payload_len, offset, b1));
            changed += 1;
        }
    }

    if let Some(parent) = out_patchspec.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_patchspec, lines.join("\n") + "\n")?;
    println!(
        "wrote patchspec: payload_len={} changed={} out={}",
        payload_len,
        changed,
        out_patchspec.display()
    );

    Ok(())
}
