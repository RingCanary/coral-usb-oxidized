use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug)]
pub enum ToolchainError {
    InvalidArgument(String),
    Io(std::io::Error),
    CommandFailed {
        program: String,
        args: Vec<String>,
        status: Option<i32>,
        stderr: String,
    },
}

impl fmt::Display for ToolchainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolchainError::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
            ToolchainError::Io(err) => write!(f, "I/O error: {}", err),
            ToolchainError::CommandFailed {
                program,
                args,
                status,
                stderr,
            } => write!(
                f,
                "command failed: {} {} (status: {:?}){}{}",
                program,
                args.join(" "),
                status,
                if stderr.is_empty() { "" } else { "\nstderr:\n" },
                stderr
            ),
        }
    }
}

impl std::error::Error for ToolchainError {}

impl From<std::io::Error> for ToolchainError {
    fn from(value: std::io::Error) -> Self {
        ToolchainError::Io(value)
    }
}

#[derive(Debug, Clone)]
pub struct DenseTemplateCompileRequest {
    pub out_dir: PathBuf,
    pub python_version: String,
    pub tf_package: Option<String>,
    pub tf_version: Option<String>,
    pub numpy_version: Option<String>,
    pub compiler_path: Option<PathBuf>,
    pub batch_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub init_mode: String,
    pub diag_scale: f32,
    pub seed: u64,
    pub rep_samples: usize,
    pub rep_range: f32,
    pub run_extract: bool,
    pub run_parse: bool,
}

impl Default for DenseTemplateCompileRequest {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("traces/rust-dense-template"),
            python_version: "3.9".to_string(),
            tf_package: None,
            tf_version: None,
            numpy_version: None,
            compiler_path: None,
            batch_size: 1,
            input_dim: 256,
            output_dim: 256,
            init_mode: "identity".to_string(),
            diag_scale: 1.0,
            seed: 1337,
            rep_samples: 256,
            rep_range: 1.0,
            run_extract: true,
            run_parse: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DenseTemplateCompileArtifacts {
    pub out_dir: PathBuf,
    pub quant_model: PathBuf,
    pub quant_metadata: PathBuf,
    pub compiled_model: PathBuf,
    pub compile_log: PathBuf,
    pub extract_dir: Option<PathBuf>,
    pub parse_report: Option<PathBuf>,
}

fn tool_paths() -> (PathBuf, PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    (
        root.join("tools/generate_dense_quant_tflite.py"),
        root.join("tools/extract_edgetpu_package.py"),
        root.join("tools/parse_edgetpu_executable.py"),
    )
}

fn default_tf_stack() -> (&'static str, &'static str, &'static str) {
    match std::env::consts::ARCH {
        "aarch64" | "arm" => ("tensorflow", "2.19.0", "1.26.4"),
        _ => ("tensorflow-cpu", "2.10.1", "1.23.5"),
    }
}

fn model_basename(input_dim: usize, output_dim: usize, batch_size: usize) -> String {
    if batch_size == 1 {
        format!("dense_{}x{}_quant", input_dim, output_dim)
    } else {
        format!("dense_{}x{}_quant_b{}", input_dim, output_dim, batch_size)
    }
}

fn run_checked(program: &str, args: &[String], cwd: &Path) -> Result<(), ToolchainError> {
    let status = Command::new(program).args(args).current_dir(cwd).status()?;
    if status.success() {
        return Ok(());
    }
    Err(ToolchainError::CommandFailed {
        program: program.to_string(),
        args: args.to_vec(),
        status: status.code(),
        stderr: String::new(),
    })
}

fn run_capture(
    program: &str,
    args: &[String],
    cwd: &Path,
) -> Result<std::process::Output, ToolchainError> {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .output()?;
    if output.status.success() {
        return Ok(output);
    }
    Err(ToolchainError::CommandFailed {
        program: program.to_string(),
        args: args.to_vec(),
        status: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

pub fn compile_dense_template_with_uv(
    request: &DenseTemplateCompileRequest,
) -> Result<DenseTemplateCompileArtifacts, ToolchainError> {
    if request.batch_size == 0 {
        return Err(ToolchainError::InvalidArgument(
            "batch_size must be >= 1".to_string(),
        ));
    }
    if request.input_dim == 0 || request.output_dim == 0 {
        return Err(ToolchainError::InvalidArgument(
            "input_dim/output_dim must be >= 1".to_string(),
        ));
    }

    std::fs::create_dir_all(&request.out_dir)?;
    let (generate_script, extract_script, parse_script) = tool_paths();
    let (default_tf_pkg, default_tf_ver, default_np_ver) = default_tf_stack();
    let tf_package = request
        .tf_package
        .clone()
        .unwrap_or_else(|| default_tf_pkg.to_string());
    let tf_version = request
        .tf_version
        .clone()
        .unwrap_or_else(|| default_tf_ver.to_string());
    let numpy_version = request
        .numpy_version
        .clone()
        .unwrap_or_else(|| default_np_ver.to_string());
    let model_name = model_basename(request.input_dim, request.output_dim, request.batch_size);

    let quant_model = request.out_dir.join(format!("{}.tflite", model_name));
    let quant_metadata = request
        .out_dir
        .join(format!("{}.metadata.json", model_name));
    let compile_log = request.out_dir.join("edgetpu_compile.log");
    let compiled_model = request
        .out_dir
        .join(format!("{}_edgetpu.tflite", model_name));

    run_checked(
        "uv",
        &[
            "python".to_string(),
            "install".to_string(),
            request.python_version.clone(),
        ],
        &request.out_dir,
    )?;

    run_checked(
        "uv",
        &[
            "run".to_string(),
            "--python".to_string(),
            request.python_version.clone(),
            "--with".to_string(),
            format!("{}=={}", tf_package, tf_version),
            "--with".to_string(),
            format!("numpy=={}", numpy_version),
            "python3".to_string(),
            generate_script.to_string_lossy().into_owned(),
            "--output".to_string(),
            quant_model.to_string_lossy().into_owned(),
            "--metadata-out".to_string(),
            quant_metadata.to_string_lossy().into_owned(),
            "--batch-size".to_string(),
            request.batch_size.to_string(),
            "--input-dim".to_string(),
            request.input_dim.to_string(),
            "--output-dim".to_string(),
            request.output_dim.to_string(),
            "--init-mode".to_string(),
            request.init_mode.clone(),
            "--diag-scale".to_string(),
            request.diag_scale.to_string(),
            "--seed".to_string(),
            request.seed.to_string(),
            "--rep-samples".to_string(),
            request.rep_samples.to_string(),
            "--rep-range".to_string(),
            request.rep_range.to_string(),
        ],
        &request.out_dir,
    )?;

    let compiler = request
        .compiler_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("edgetpu_compiler"));
    let compiler_args = vec![
        "-s".to_string(),
        "-o".to_string(),
        request.out_dir.to_string_lossy().into_owned(),
        quant_model.to_string_lossy().into_owned(),
    ];
    let compile_out = run_capture(
        &compiler.to_string_lossy(),
        &compiler_args,
        &request.out_dir,
    )?;
    std::fs::write(&compile_log, &compile_out.stdout)?;
    if !compiled_model.exists() {
        return Err(ToolchainError::InvalidArgument(format!(
            "compiled model missing at {} (check {})",
            compiled_model.display(),
            compile_log.display()
        )));
    }

    let mut artifacts = DenseTemplateCompileArtifacts {
        out_dir: request.out_dir.clone(),
        quant_model,
        quant_metadata,
        compiled_model: compiled_model.clone(),
        compile_log,
        extract_dir: None,
        parse_report: None,
    };

    if request.run_extract {
        let extract_dir = request.out_dir.join("extract");
        run_checked(
            "python3",
            &[
                extract_script.to_string_lossy().into_owned(),
                "extract".to_string(),
                compiled_model.to_string_lossy().into_owned(),
                "--out".to_string(),
                extract_dir.to_string_lossy().into_owned(),
                "--overwrite".to_string(),
            ],
            &request.out_dir,
        )?;
        artifacts.extract_dir = Some(extract_dir.clone());

        if request.run_parse {
            let parse_target = extract_dir.join("package_000");
            let parse_output_path = request.out_dir.join("exec_parse.txt");
            let parse_output = run_capture(
                "python3",
                &[
                    parse_script.to_string_lossy().into_owned(),
                    parse_target.to_string_lossy().into_owned(),
                ],
                &request.out_dir,
            )?;
            std::fs::write(&parse_output_path, &parse_output.stdout)?;
            artifacts.parse_report = Some(parse_output_path);
        }
    }

    Ok(artifacts)
}
