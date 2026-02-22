use coral_usb_oxidized::{version, CoralDevice, DenseGemmTemplate};
use std::env;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum MatrixMode {
    Identity,
    ShiftPlus1,
    ShiftMinus1,
    Zero,
}

#[derive(Clone, Copy, Debug)]
enum InputMode {
    Ramp,
    Ones,
}

fn parse_matrix_mode(value: &str) -> Result<MatrixMode, Box<dyn Error>> {
    match value {
        "identity" => Ok(MatrixMode::Identity),
        "shift_plus1" => Ok(MatrixMode::ShiftPlus1),
        "shift_minus1" => Ok(MatrixMode::ShiftMinus1),
        "zero" => Ok(MatrixMode::Zero),
        _ => Err(format!(
            "unknown matrix mode: {value} (expected identity|shift_plus1|shift_minus1|zero)"
        )
        .into()),
    }
}

fn parse_input_mode(value: &str) -> Result<InputMode, Box<dyn Error>> {
    match value {
        "ramp" => Ok(InputMode::Ramp),
        "ones" => Ok(InputMode::Ones),
        _ => Err(format!("unknown input mode: {value} (expected ramp|ones)").into()),
    }
}

fn build_input(input_dim: usize, mode: InputMode) -> Vec<i8> {
    let mut out = vec![0i8; input_dim];
    match mode {
        InputMode::Ramp => {
            for (idx, value) in out.iter_mut().enumerate() {
                *value = idx as i8;
            }
        }
        InputMode::Ones => out.fill(1),
    }
    out
}

fn apply_mode(template: &mut DenseGemmTemplate, mode: MatrixMode) -> Result<(), Box<dyn Error>> {
    match mode {
        MatrixMode::Identity => template.set_identity(127)?,
        MatrixMode::ShiftPlus1 => template.set_shift_plus1(127)?,
        MatrixMode::ShiftMinus1 => template.set_shift_minus1(127)?,
        MatrixMode::Zero => template.fill_matrix_qi8(0)?,
    }
    Ok(())
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
    if args.len() < 4 {
        eprintln!(
            "Usage: cargo run --example gemm_int8_dynamic -- <template.tflite> <input_dim> <output_dim> [matrix_mode] [input_mode] [runs]"
        );
        std::process::exit(2);
    }

    let model_path = &args[1];
    let input_dim = args[2].parse::<usize>()?;
    let output_dim = args[3].parse::<usize>()?;
    let matrix_mode = parse_matrix_mode(args.get(4).map(String::as_str).unwrap_or("identity"))?;
    let input_mode = parse_input_mode(args.get(5).map(String::as_str).unwrap_or("ramp"))?;
    let runs = args
        .get(6)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);

    if !Path::new(model_path).exists() {
        return Err(format!("model not found: {model_path}").into());
    }

    println!("EdgeTPU version: {}", version());
    println!("Model template: {}", model_path);
    println!("Dims: {}x{}", input_dim, output_dim);
    println!("Matrix mode: {:?}", matrix_mode);
    println!("Input mode: {:?}", input_mode);
    println!("Runs: {}", runs);

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;

    let mut template = DenseGemmTemplate::from_file_with_dims(model_path, input_dim, output_dim)?;
    apply_mode(&mut template, matrix_mode)?;
    let prepared = template.prepare(&delegate)?;

    let input = build_input(input_dim, input_mode);
    let mut output = vec![0i8; output_dim];
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
