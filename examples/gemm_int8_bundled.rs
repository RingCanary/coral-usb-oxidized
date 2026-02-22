use coral_usb_oxidized::{version, CoralDevice, DenseGemmTemplate};
use std::env;
use std::error::Error;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum MatrixMode {
    Identity,
    ShiftPlus1,
    ShiftMinus1,
}

fn parse_mode(value: &str) -> Result<MatrixMode, Box<dyn Error>> {
    match value {
        "identity" => Ok(MatrixMode::Identity),
        "shift_plus1" => Ok(MatrixMode::ShiftPlus1),
        "shift_minus1" => Ok(MatrixMode::ShiftMinus1),
        _ => Err(
            format!("unknown mode: {value} (expected identity|shift_plus1|shift_minus1)").into(),
        ),
    }
}

fn load_bundled_template(dim: usize) -> Result<DenseGemmTemplate, Box<dyn Error>> {
    let template = match dim {
        2048 => DenseGemmTemplate::from_bundled_2048()?,
        2304 => DenseGemmTemplate::from_bundled_2304()?,
        2688 => DenseGemmTemplate::from_bundled_2688()?,
        _ => return Err(format!("unsupported bundled dimension: {dim}").into()),
    };

    Ok(template)
}

fn build_ramp(dim: usize) -> Vec<i8> {
    let mut out = vec![0i8; dim];
    for (idx, value) in out.iter_mut().enumerate() {
        *value = idx as i8;
    }
    out
}

fn preview(label: &str, data: &[i8], count: usize) {
    let shown = count.min(data.len());
    let joined = data
        .iter()
        .take(shown)
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("{label} (first {shown}): {joined}");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let dim = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2688);
    let mode = parse_mode(args.get(2).map(String::as_str).unwrap_or("identity"))?;
    let runs = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(30);
    if runs == 0 {
        return Err("runs must be >= 1".into());
    }

    let mut template = load_bundled_template(dim)?;
    match mode {
        MatrixMode::Identity => template.set_identity(127)?,
        MatrixMode::ShiftPlus1 => template.set_shift_plus1(127)?,
        MatrixMode::ShiftMinus1 => template.set_shift_minus1(127)?,
    }

    let input = build_ramp(dim);

    println!("EdgeTPU version: {}", version());
    println!("Bundled template dim: {}", dim);
    println!("Mode: {:?}", mode);
    println!("Runs: {}", runs);

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;
    let prepared = template.prepare(&delegate)?;

    let mut output = vec![0i8; dim];
    let mut total_ms = 0.0f64;
    for run_idx in 0..runs {
        let started = Instant::now();
        let current = prepared.execute(&input)?;
        total_ms += started.elapsed().as_secs_f64() * 1000.0;
        if run_idx + 1 == runs {
            output = current;
        }
    }

    println!(
        "Latency: avg_ms={:.3} total_ms={:.3}",
        total_ms / runs as f64,
        total_ms
    );
    preview("Input", &input, 32);
    preview("Output", &output, 32);

    Ok(())
}
