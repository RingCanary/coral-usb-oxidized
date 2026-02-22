use coral_usb_oxidized::{
    version, CoralDevice, DenseGemm256Template, GemmTemplate256, DENSE_GEMM256_DIM,
};
use std::env;
use std::error::Error;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
enum MatrixMode {
    Identity,
    ShiftPlus1,
    ShiftMinus1,
    DiagonalRamp,
}

#[derive(Clone, Copy, Debug)]
enum InputMode {
    Ramp,
    Impulse,
    Alternating,
    Ones,
}

fn parse_matrix_mode(value: &str) -> Result<MatrixMode, Box<dyn Error>> {
    match value {
        "identity" => Ok(MatrixMode::Identity),
        "shift_plus1" => Ok(MatrixMode::ShiftPlus1),
        "shift_minus1" => Ok(MatrixMode::ShiftMinus1),
        "diag_ramp" => Ok(MatrixMode::DiagonalRamp),
        _ => Err(format!(
            "unknown matrix mode: {} (expected identity|shift_plus1|shift_minus1|diag_ramp)",
            value
        )
        .into()),
    }
}

fn parse_input_mode(value: &str) -> Result<InputMode, Box<dyn Error>> {
    match value {
        "ramp" => Ok(InputMode::Ramp),
        "impulse" => Ok(InputMode::Impulse),
        "alternating" => Ok(InputMode::Alternating),
        "ones" => Ok(InputMode::Ones),
        _ => Err(format!(
            "unknown input mode: {} (expected ramp|impulse|alternating|ones)",
            value
        )
        .into()),
    }
}

fn build_input(mode: InputMode) -> [i8; DENSE_GEMM256_DIM] {
    let mut out = [0i8; DENSE_GEMM256_DIM];
    match mode {
        InputMode::Ramp => {
            for (idx, value) in out.iter_mut().enumerate() {
                *value = idx as i8;
            }
        }
        InputMode::Impulse => {
            out[0] = 127;
        }
        InputMode::Alternating => {
            for (idx, value) in out.iter_mut().enumerate() {
                *value = if idx % 2 == 0 { 127 } else { -128 };
            }
        }
        InputMode::Ones => {
            out.fill(1);
        }
    }
    out
}

fn apply_matrix_mode(
    template: &mut DenseGemm256Template,
    mode: MatrixMode,
) -> Result<(), Box<dyn Error>> {
    match mode {
        MatrixMode::Identity => template.set_identity(127)?,
        MatrixMode::ShiftPlus1 => template.set_shift_plus1(127)?,
        MatrixMode::ShiftMinus1 => template.set_shift_minus1(127)?,
        MatrixMode::DiagonalRamp => {
            let mut diagonal = [0i8; DENSE_GEMM256_DIM];
            for (idx, value) in diagonal.iter_mut().enumerate() {
                *value = (idx as i16 - 128) as i8;
            }
            template.set_diagonal(&diagonal)?;
        }
    }
    Ok(())
}

fn expected_output(
    mode: MatrixMode,
    input: &[i8; DENSE_GEMM256_DIM],
) -> Option<[i8; DENSE_GEMM256_DIM]> {
    let mut expected = [0i8; DENSE_GEMM256_DIM];
    match mode {
        MatrixMode::Identity => {
            expected.copy_from_slice(input);
            Some(expected)
        }
        MatrixMode::ShiftPlus1 => {
            for idx in 0..DENSE_GEMM256_DIM {
                expected[idx] = input[(idx + 1) % DENSE_GEMM256_DIM];
            }
            Some(expected)
        }
        MatrixMode::ShiftMinus1 => {
            for idx in 0..DENSE_GEMM256_DIM {
                expected[idx] = input[(idx + DENSE_GEMM256_DIM - 1) % DENSE_GEMM256_DIM];
            }
            Some(expected)
        }
        MatrixMode::DiagonalRamp => None,
    }
}

fn summarize_delta(
    output: &[i8; DENSE_GEMM256_DIM],
    expected: &[i8; DENSE_GEMM256_DIM],
) -> (usize, i16) {
    let mut mismatches = 0usize;
    let mut max_abs_delta = 0i16;
    for (got, want) in output.iter().zip(expected.iter()) {
        let delta = (*got as i16 - *want as i16).abs();
        if delta > 1 {
            mismatches += 1;
        }
        if delta > max_abs_delta {
            max_abs_delta = delta;
        }
    }
    (mismatches, max_abs_delta)
}

fn preview(label: &str, data: &[i8; DENSE_GEMM256_DIM], count: usize) {
    let shown = count.min(DENSE_GEMM256_DIM);
    let joined = data
        .iter()
        .take(shown)
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("{} (first {}): {}", label, shown, joined);
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: cargo run --example gemm_int8 -- <dense_template_edgetpu.tflite> [matrix_mode] [input_mode]"
        );
        std::process::exit(2);
    }

    let model_path = &args[1];
    let matrix_mode = parse_matrix_mode(args.get(2).map(String::as_str).unwrap_or("shift_plus1"))?;
    let input_mode = parse_input_mode(args.get(3).map(String::as_str).unwrap_or("ramp"))?;

    if !Path::new(model_path).exists() {
        return Err(format!("model not found: {}", model_path).into());
    }

    println!("EdgeTPU version: {}", version());
    println!("Model template: {}", model_path);
    println!("Matrix mode: {:?}", matrix_mode);
    println!("Input mode: {:?}", input_mode);

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;
    let mut gemm = GemmTemplate256::from_compiled_template_file(model_path)?;
    apply_matrix_mode(gemm.template_mut(), matrix_mode)?;

    let input = build_input(input_mode);
    let output = gemm.execute(&delegate, &input)?;
    preview("Input", &input, 32);
    preview("Output", &output, 32);

    if let Some(expected) = expected_output(matrix_mode, &input) {
        preview("Expected", &expected, 32);
        let (mismatches, max_abs_delta) = summarize_delta(&output, &expected);
        println!(
            "Verification: mismatches(|delta|>1)={} max_abs_delta={}",
            mismatches, max_abs_delta
        );
    } else {
        println!(
            "Verification: no closed-form check for mode {:?}",
            matrix_mode
        );
    }

    Ok(())
}
