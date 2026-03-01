use coral_usb_oxidized::{compile_dense_template_with_uv, DenseTemplateCompileRequest};
use std::env;
use std::error::Error;
use std::path::PathBuf;

fn usage(program: &str) {
    println!("Usage: {program} [options]");
    println!("Options:");
    println!("  --out-dir PATH           Output directory (default: traces/rust-dense-template)");
    println!("  --python-version VER     uv Python version (default: 3.9)");
    println!("  --tf-package NAME        TensorFlow package (default: arch-aware)");
    println!("  --tf-version VER         TensorFlow version (default: package-aware)");
    println!("  --numpy-version VER      NumPy version (default: package-aware)");
    println!("  --compiler PATH          edgetpu_compiler path (default: edgetpu_compiler)");
    println!("  --batch-size N           Dense batch size (default: 1)");
    println!("  --input-dim N            Dense input dim (default: 256)");
    println!("  --output-dim N           Dense output dim (default: 256)");
    println!("  --init-mode MODE         identity|permutation|ones|zero|single_hot|random_uniform");
    println!("  --diag-scale F           Diagonal/fill scale (default: 1.0)");
    println!("  --seed N                 RNG seed (default: 1337)");
    println!("  --rep-samples N          Representative sample count (default: 256)");
    println!("  --rep-range F            Representative value range (default: 1.0)");
    println!("  --no-extract             Skip DWN1 extraction");
    println!("  --no-parse               Skip executable parser report");
    println!("  -h, --help               Show this help");
    println!();
    println!("Example:");
    println!("  {program} --input-dim 2048 --output-dim 2048 --out-dir traces/rust-compile-2048");
}

fn parse_usize(v: &str, flag: &str) -> Result<usize, Box<dyn Error>> {
    Ok(v.parse::<usize>()
        .map_err(|e| format!("invalid {} value '{}': {}", flag, v, e))?)
}

fn parse_u64(v: &str, flag: &str) -> Result<u64, Box<dyn Error>> {
    Ok(v.parse::<u64>()
        .map_err(|e| format!("invalid {} value '{}': {}", flag, v, e))?)
}

fn parse_f32(v: &str, flag: &str) -> Result<f32, Box<dyn Error>> {
    Ok(v.parse::<f32>()
        .map_err(|e| format!("invalid {} value '{}': {}", flag, v, e))?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "rust_dense_template_compile".to_string());

    let mut req = DenseTemplateCompileRequest::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--out-dir" => {
                i += 1;
                let v = args.get(i).ok_or("--out-dir requires a value")?;
                req.out_dir = PathBuf::from(v);
            }
            "--python-version" => {
                i += 1;
                let v = args.get(i).ok_or("--python-version requires a value")?;
                req.python_version = v.to_string();
            }
            "--tf-package" => {
                i += 1;
                let v = args.get(i).ok_or("--tf-package requires a value")?;
                req.tf_package = Some(v.to_string());
            }
            "--tf-version" => {
                i += 1;
                let v = args.get(i).ok_or("--tf-version requires a value")?;
                req.tf_version = Some(v.to_string());
            }
            "--numpy-version" => {
                i += 1;
                let v = args.get(i).ok_or("--numpy-version requires a value")?;
                req.numpy_version = Some(v.to_string());
            }
            "--compiler" => {
                i += 1;
                let v = args.get(i).ok_or("--compiler requires a value")?;
                req.compiler_path = Some(PathBuf::from(v));
            }
            "--batch-size" => {
                i += 1;
                let v = args.get(i).ok_or("--batch-size requires a value")?;
                req.batch_size = parse_usize(v, "--batch-size")?;
            }
            "--input-dim" => {
                i += 1;
                let v = args.get(i).ok_or("--input-dim requires a value")?;
                req.input_dim = parse_usize(v, "--input-dim")?;
            }
            "--output-dim" => {
                i += 1;
                let v = args.get(i).ok_or("--output-dim requires a value")?;
                req.output_dim = parse_usize(v, "--output-dim")?;
            }
            "--init-mode" => {
                i += 1;
                let v = args.get(i).ok_or("--init-mode requires a value")?;
                req.init_mode = v.to_string();
            }
            "--diag-scale" => {
                i += 1;
                let v = args.get(i).ok_or("--diag-scale requires a value")?;
                req.diag_scale = parse_f32(v, "--diag-scale")?;
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or("--seed requires a value")?;
                req.seed = parse_u64(v, "--seed")?;
            }
            "--rep-samples" => {
                i += 1;
                let v = args.get(i).ok_or("--rep-samples requires a value")?;
                req.rep_samples = parse_usize(v, "--rep-samples")?;
            }
            "--rep-range" => {
                i += 1;
                let v = args.get(i).ok_or("--rep-range requires a value")?;
                req.rep_range = parse_f32(v, "--rep-range")?;
            }
            "--no-extract" => req.run_extract = false,
            "--no-parse" => req.run_parse = false,
            other => {
                return Err(format!("unknown argument: {}", other).into());
            }
        }
        i += 1;
    }

    println!("Running Rust toolchain compile pipeline...");
    let artifacts = compile_dense_template_with_uv(&req)?;
    println!("Done.");
    println!("  out_dir:       {}", artifacts.out_dir.display());
    println!("  quant_model:   {}", artifacts.quant_model.display());
    println!("  quant_meta:    {}", artifacts.quant_metadata.display());
    println!("  compiled:      {}", artifacts.compiled_model.display());
    println!("  compile_log:   {}", artifacts.compile_log.display());
    if let Some(extract) = artifacts.extract_dir {
        println!("  extract_dir:   {}", extract.display());
    }
    if let Some(parse) = artifacts.parse_report {
        println!("  parse_report:  {}", parse.display());
    }
    Ok(())
}
