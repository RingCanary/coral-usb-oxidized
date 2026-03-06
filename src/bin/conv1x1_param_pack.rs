use coral_usb_oxidized::{
    conv1x1_param_stream_len, pack_conv1x1_row_major_i8_to_stream,
    pack_conv1x1_row_major_u8_to_stream,
};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --in-channels N --out-channels N (--stored-u8 PATH | --stored-i8 PATH) --out PATH"
    );
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
        .unwrap_or_else(|| "conv1x1_param_pack".to_string());

    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut in_channels: Option<usize> = None;
    let mut out_channels: Option<usize> = None;
    let mut stored_u8: Option<PathBuf> = None;
    let mut stored_i8: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--in-channels" => {
                let raw = next_arg(&args, &mut i, "--in-channels")?;
                in_channels = Some(raw.parse()?);
            }
            "--out-channels" => {
                let raw = next_arg(&args, &mut i, "--out-channels")?;
                out_channels = Some(raw.parse()?);
            }
            "--stored-u8" => {
                stored_u8 = Some(PathBuf::from(next_arg(&args, &mut i, "--stored-u8")?));
            }
            "--stored-i8" => {
                stored_i8 = Some(PathBuf::from(next_arg(&args, &mut i, "--stored-i8")?));
            }
            "--out" => {
                out_path = Some(PathBuf::from(next_arg(&args, &mut i, "--out")?));
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        i += 1;
    }

    let in_channels = in_channels.ok_or("missing --in-channels")?;
    let out_channels = out_channels.ok_or("missing --out-channels")?;
    let out_path = out_path.ok_or("missing --out")?;

    let weight_source_count = (stored_u8.is_some() as usize) + (stored_i8.is_some() as usize);
    if weight_source_count != 1 {
        return Err("choose exactly one of --stored-u8 or --stored-i8".into());
    }

    let expected_row_major_len = in_channels
        .checked_mul(out_channels)
        .ok_or("overflow computing row-major length")?;
    let stream = if let Some(path) = stored_u8.as_ref() {
        let bytes = fs::read(path)?;
        if bytes.len() != expected_row_major_len {
            return Err(format!(
                "stored u8 length mismatch: expected {}, got {} ({})",
                expected_row_major_len,
                bytes.len(),
                path.display()
            )
            .into());
        }
        pack_conv1x1_row_major_u8_to_stream(in_channels, out_channels, &bytes)?
    } else {
        let bytes = fs::read(stored_i8.as_ref().unwrap())?;
        if bytes.len() != expected_row_major_len {
            return Err(format!(
                "stored i8 length mismatch: expected {}, got {} ({})",
                expected_row_major_len,
                bytes.len(),
                stored_i8.as_ref().unwrap().display()
            )
            .into());
        }
        let vals: Vec<i8> = bytes.iter().map(|&v| v as i8).collect();
        pack_conv1x1_row_major_i8_to_stream(in_channels, out_channels, &vals)?
    };

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, &stream)?;
    println!(
        "wrote conv1x1 param stream: in_channels={} out_channels={} stored_len={} stream_len={} expected_stream_len={} fnv=0x{:016x} out={}",
        in_channels,
        out_channels,
        expected_row_major_len,
        stream.len(),
        conv1x1_param_stream_len(in_channels, out_channels)?,
        fnv1a64(&stream),
        out_path.display()
    );

    Ok(())
}
