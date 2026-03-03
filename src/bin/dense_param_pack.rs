use coral_usb_oxidized::{
    dense_param_stream_len, pack_dense_row_major_i8_to_stream, pack_dense_row_major_u8_to_stream,
    unpack_dense_stream_to_row_major_u8,
};
use std::fs;
use std::path::PathBuf;

type DynError = Box<dyn std::error::Error>;

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --rows N --cols N --out PATH [source options]\n\n\
Source options (choose one):\n\
  --row-major-u8-file PATH        Read row-major raw u8 bytes (len=rows*cols)\n\
  --row-major-i8-file PATH        Read row-major raw i8 bytes (len=rows*cols)\n\
  --pattern-index-mod             Generate row-major pattern value=i%modulus\n\
\
Pattern options:\n\
  --modulus N                     Modulus for --pattern-index-mod (default: 251)\n\
  --signed-reinterpret            Pattern bytes as signed reinterpret: ((i%modulus)-128) mod 256\n\
\
Other options:\n\
  --out-row-major-u8 PATH         Optionally dump generated/loaded row-major u8 bytes\n\
  --verify-roundtrip              Verify unpack(pack(src)) == src\n\
  -h, --help                      Show this help\n"
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

fn parse_usize(value: &str, flag: &str) -> Result<usize, DynError> {
    value
        .parse::<usize>()
        .map_err(|e| format!("invalid {} value '{}': {e}", flag, value).into())
}

fn main() -> Result<(), DynError> {
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "dense_param_pack".to_string());

    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage(&program);
        return Ok(());
    }

    let mut rows: Option<usize> = None;
    let mut cols: Option<usize> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut row_major_u8_file: Option<PathBuf> = None;
    let mut row_major_i8_file: Option<PathBuf> = None;
    let mut pattern_index_mod = false;
    let mut modulus = 251usize;
    let mut signed_reinterpret = false;
    let mut out_row_major_u8: Option<PathBuf> = None;
    let mut verify_roundtrip = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--rows" => {
                i += 1;
                rows = Some(parse_usize(
                    args.get(i).ok_or("--rows requires value")?,
                    "--rows",
                )?);
            }
            "--cols" => {
                i += 1;
                cols = Some(parse_usize(
                    args.get(i).ok_or("--cols requires value")?,
                    "--cols",
                )?);
            }
            "--out" => {
                i += 1;
                out_path = Some(PathBuf::from(args.get(i).ok_or("--out requires value")?));
            }
            "--row-major-u8-file" => {
                i += 1;
                row_major_u8_file = Some(PathBuf::from(
                    args.get(i).ok_or("--row-major-u8-file requires value")?,
                ));
            }
            "--row-major-i8-file" => {
                i += 1;
                row_major_i8_file = Some(PathBuf::from(
                    args.get(i).ok_or("--row-major-i8-file requires value")?,
                ));
            }
            "--pattern-index-mod" => {
                pattern_index_mod = true;
            }
            "--modulus" => {
                i += 1;
                modulus = parse_usize(args.get(i).ok_or("--modulus requires value")?, "--modulus")?;
            }
            "--signed-reinterpret" => {
                signed_reinterpret = true;
            }
            "--out-row-major-u8" => {
                i += 1;
                out_row_major_u8 = Some(PathBuf::from(
                    args.get(i).ok_or("--out-row-major-u8 requires value")?,
                ));
            }
            "--verify-roundtrip" => {
                verify_roundtrip = true;
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
        i += 1;
    }

    let rows = rows.ok_or("--rows is required")?;
    let cols = cols.ok_or("--cols is required")?;
    let out_path = out_path.ok_or("--out is required")?;

    if modulus == 0 || modulus > 256 {
        return Err("--modulus must be in [1,256]".into());
    }

    let source_i8 = row_major_i8_file.is_some();

    let source_count = (row_major_u8_file.is_some() as usize)
        + (source_i8 as usize)
        + (pattern_index_mod as usize);
    if source_count != 1 {
        return Err(
            "choose exactly one source: --row-major-u8-file | --row-major-i8-file | --pattern-index-mod"
                .into(),
        );
    }

    let expected_len = dense_param_stream_len(rows, cols)?;

    let row_major_u8: Vec<u8> = if let Some(path) = row_major_u8_file {
        let bytes = fs::read(&path)
            .map_err(|e| format!("failed to read row-major-u8 file {}: {e}", path.display()))?;
        if bytes.len() != expected_len {
            return Err(format!(
                "row-major-u8 length mismatch: expected {}, got {} ({})",
                expected_len,
                bytes.len(),
                path.display()
            )
            .into());
        }
        bytes
    } else if let Some(path) = row_major_i8_file {
        let bytes = fs::read(&path)
            .map_err(|e| format!("failed to read row-major-i8 file {}: {e}", path.display()))?;
        if bytes.len() != expected_len {
            return Err(format!(
                "row-major-i8 length mismatch: expected {}, got {} ({})",
                expected_len,
                bytes.len(),
                path.display()
            )
            .into());
        }
        bytes
    } else {
        (0..expected_len)
            .map(|idx| {
                let v = (idx % modulus) as i16;
                if signed_reinterpret {
                    ((v - 128).rem_euclid(256)) as u8
                } else {
                    (v % 256) as u8
                }
            })
            .collect()
    };

    if let Some(path) = out_row_major_u8.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &row_major_u8)?;
    }

    let stream = if source_i8 {
        let as_i8: Vec<i8> = row_major_u8.iter().map(|&v| v as i8).collect();
        pack_dense_row_major_i8_to_stream(rows, cols, &as_i8)?
    } else {
        pack_dense_row_major_u8_to_stream(rows, cols, &row_major_u8)?
    };

    if verify_roundtrip {
        let restored = unpack_dense_stream_to_row_major_u8(rows, cols, &stream)?;
        if restored != row_major_u8 {
            return Err("roundtrip verify failed: unpack(pack(src)) != src".into());
        }
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, &stream)?;

    println!("rows={} cols={} len={}", rows, cols, expected_len);
    println!("row_major_fnv1a64=0x{:016x}", fnv1a64(&row_major_u8));
    println!("stream_fnv1a64=0x{:016x}", fnv1a64(&stream));
    println!("out={}", out_path.display());

    Ok(())
}
