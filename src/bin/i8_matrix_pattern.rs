use std::fs;
use std::path::PathBuf;

type DynError = Box<dyn std::error::Error>;

#[derive(Clone, Copy, Debug)]
enum Mode {
    IdentityCycle,
    ShiftPlus1Cycle,
    IndexMod,
    RowIndexMod,
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --rows N --cols N --mode MODE --out PATH [options]\n\n\
Modes:\n\
  identity_cycle      Square matrix: one hot at (row, row)\n\
  shift_plus1_cycle   Square matrix: one hot at (row, (row+1)%cols)\n\
  index_mod           Dense pattern by flat index modulo --modulus\n\
  row_index_mod       Dense pattern by row/col with row offset --row-step\n\n\
Options:\n\
  --rows N            Row count\n\
  --cols N            Column count\n\
  --mode MODE         Pattern mode\n\
  --out PATH          Output raw i8 bytes\n\
  --active-q N        Active q value for cycle modes (default: 127)\n\
  --inactive-q N      Inactive q value for cycle modes (default: 0)\n\
  --modulus N         Modulus for *_mod modes (default: 251)\n\
  --row-step N        Row step for row_index_mod (default: 17)\n\
  --signed-center N   Center values around zero with offset N (default: 128)\n\
  -h, --help          Show this help\n"
    );
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, DynError> {
    value
        .parse::<usize>()
        .map_err(|e| format!("invalid {} value '{}': {}", flag, value, e).into())
}

fn parse_i8(value: &str, flag: &str) -> Result<i8, DynError> {
    value
        .parse::<i16>()
        .map_err(|e| format!("invalid {} value '{}': {}", flag, value, e))?
        .try_into()
        .map_err(|_| format!("{} out of i8 range: {}", flag, value).into())
}

fn parse_mode(value: &str) -> Result<Mode, DynError> {
    match value {
        "identity_cycle" => Ok(Mode::IdentityCycle),
        "shift_plus1_cycle" => Ok(Mode::ShiftPlus1Cycle),
        "index_mod" => Ok(Mode::IndexMod),
        "row_index_mod" => Ok(Mode::RowIndexMod),
        _ => Err(format!(
            "unknown --mode '{}': expected identity_cycle|shift_plus1_cycle|index_mod|row_index_mod",
            value
        )
        .into()),
    }
}

fn encode_signed_mod(value: usize, modulus: usize, signed_center: i16) -> i8 {
    let reduced = (value % modulus) as i16;
    ((reduced - signed_center).rem_euclid(256) as u8) as i8
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn main() -> Result<(), DynError> {
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "i8_matrix_pattern".to_string());

    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        usage(&program);
        return Ok(());
    }

    let mut rows: Option<usize> = None;
    let mut cols: Option<usize> = None;
    let mut mode: Option<Mode> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut active_q: i8 = 127;
    let mut inactive_q: i8 = 0;
    let mut modulus: usize = 251;
    let mut row_step: usize = 17;
    let mut signed_center: i16 = 128;

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
            "--mode" => {
                i += 1;
                mode = Some(parse_mode(args.get(i).ok_or("--mode requires value")?)?);
            }
            "--out" => {
                i += 1;
                out_path = Some(PathBuf::from(args.get(i).ok_or("--out requires value")?));
            }
            "--active-q" => {
                i += 1;
                active_q = parse_i8(
                    args.get(i).ok_or("--active-q requires value")?,
                    "--active-q",
                )?;
            }
            "--inactive-q" => {
                i += 1;
                inactive_q = parse_i8(
                    args.get(i).ok_or("--inactive-q requires value")?,
                    "--inactive-q",
                )?;
            }
            "--modulus" => {
                i += 1;
                modulus = parse_usize(args.get(i).ok_or("--modulus requires value")?, "--modulus")?;
            }
            "--row-step" => {
                i += 1;
                row_step = parse_usize(
                    args.get(i).ok_or("--row-step requires value")?,
                    "--row-step",
                )?;
            }
            "--signed-center" => {
                i += 1;
                signed_center = args
                    .get(i)
                    .ok_or("--signed-center requires value")?
                    .parse::<i16>()
                    .map_err(|e| format!("invalid --signed-center value: {}", e))?;
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        i += 1;
    }

    let rows = rows.ok_or("--rows is required")?;
    let cols = cols.ok_or("--cols is required")?;
    let mode = mode.ok_or("--mode is required")?;
    let out_path = out_path.ok_or("--out is required")?;

    if rows == 0 || cols == 0 {
        return Err("--rows/--cols must be >= 1".into());
    }
    if modulus == 0 || modulus > 256 {
        return Err("--modulus must be in [1,256]".into());
    }
    if !(-255..=255).contains(&signed_center) {
        return Err("--signed-center must be in [-255,255]".into());
    }
    if matches!(mode, Mode::IdentityCycle | Mode::ShiftPlus1Cycle) && rows != cols {
        return Err("cycle modes require --rows == --cols".into());
    }

    let mut out = vec![inactive_q; rows.checked_mul(cols).ok_or("rows*cols overflow")?];
    match mode {
        Mode::IdentityCycle => {
            out.fill(inactive_q);
            for row in 0..rows {
                out[row * cols + row] = active_q;
            }
        }
        Mode::ShiftPlus1Cycle => {
            out.fill(inactive_q);
            for row in 0..rows {
                out[row * cols + ((row + 1) % cols)] = active_q;
            }
        }
        Mode::IndexMod => {
            for (idx, value) in out.iter_mut().enumerate() {
                *value = encode_signed_mod(idx, modulus, signed_center);
            }
        }
        Mode::RowIndexMod => {
            for row in 0..rows {
                for col in 0..cols {
                    let idx = row * cols + col;
                    *out.get_mut(idx).ok_or("pattern index out of range")? =
                        encode_signed_mod(row * row_step + col, modulus, signed_center);
                }
            }
        }
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let out_bytes: Vec<u8> = out.into_iter().map(|value| value as u8).collect();
    fs::write(&out_path, &out_bytes)?;

    println!("rows={} cols={} mode={:?}", rows, cols, mode);
    println!("fnv1a64=0x{:016x}", fnv1a64(&out_bytes));
    println!("out={}", out_path.display());

    Ok(())
}
