use crate::model::PredictMode;

#[derive(Debug, Clone)]
pub struct Config {
    pub analysis_json: String,
    pub base_exec: String,
    pub target_exec: String,
    pub chunk_index: usize,
    pub low_dim: i64,
    pub high_dim: i64,
    pub target_dim: i64,
    pub mid_dim: Option<i64>,
    pub mid_exec: Option<String>,
    pub tile_size: i64,
    pub predict_mode: PredictMode,
    pub lane_priority: String,
    pub out_spec: String,
    pub out_report: Option<String>,
    pub out_patchspec: Option<String>,
    pub out_patchspec_safe: Option<String>,
    pub out_patchspec_discrete: Option<String>,
}

pub fn parse_i64_flag(value: &str, flag: &str) -> Result<i64, String> {
    value
        .parse::<i64>()
        .map_err(|e| format!("{} invalid integer '{}': {}", flag, value, e))
}

pub fn parse_usize_flag(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|e| format!("{} invalid integer '{}': {}", flag, value, e))
}

pub fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{} requires a value", flag));
    }
    Ok(args[*idx].clone())
}

pub fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        return Err(
            "missing args: --analysis-json --base-exec --target-exec --low-dim --high-dim --target-dim --out-spec"
                .to_string(),
        );
    }

    let mut analysis_json: Option<String> = None;
    let mut base_exec: Option<String> = None;
    let mut target_exec: Option<String> = None;
    let mut chunk_index: usize = 0;
    let mut low_dim: Option<i64> = None;
    let mut high_dim: Option<i64> = None;
    let mut target_dim: Option<i64> = None;
    let mut mid_dim: Option<i64> = None;
    let mut mid_exec: Option<String> = None;
    let mut tile_size: i64 = 64;
    let mut predict_mode = PredictMode::Endpoint;
    let mut lane_priority = "lane32,lane16".to_string();
    let mut out_spec: Option<String> = None;
    let mut out_report: Option<String> = None;
    let mut out_patchspec: Option<String> = None;
    let mut out_patchspec_safe: Option<String> = None;
    let mut out_patchspec_discrete: Option<String> = None;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--analysis-json" => {
                analysis_json = Some(next_arg(&args, &mut idx, "--analysis-json")?);
            }
            "--base-exec" => {
                base_exec = Some(next_arg(&args, &mut idx, "--base-exec")?);
            }
            "--target-exec" => {
                target_exec = Some(next_arg(&args, &mut idx, "--target-exec")?);
            }
            "--chunk-index" => {
                chunk_index = parse_usize_flag(
                    &next_arg(&args, &mut idx, "--chunk-index")?,
                    "--chunk-index",
                )?;
            }
            "--low-dim" => {
                low_dim = Some(parse_i64_flag(
                    &next_arg(&args, &mut idx, "--low-dim")?,
                    "--low-dim",
                )?);
            }
            "--high-dim" => {
                high_dim = Some(parse_i64_flag(
                    &next_arg(&args, &mut idx, "--high-dim")?,
                    "--high-dim",
                )?);
            }
            "--target-dim" => {
                target_dim = Some(parse_i64_flag(
                    &next_arg(&args, &mut idx, "--target-dim")?,
                    "--target-dim",
                )?);
            }
            "--mid-dim" => {
                mid_dim = Some(parse_i64_flag(
                    &next_arg(&args, &mut idx, "--mid-dim")?,
                    "--mid-dim",
                )?);
            }
            "--mid-exec" => {
                mid_exec = Some(next_arg(&args, &mut idx, "--mid-exec")?);
            }
            "--tile-size" => {
                tile_size =
                    parse_i64_flag(&next_arg(&args, &mut idx, "--tile-size")?, "--tile-size")?;
            }
            "--predict-mode" => {
                predict_mode = PredictMode::parse(&next_arg(&args, &mut idx, "--predict-mode")?)?;
            }
            "--lane-priority" => {
                lane_priority = next_arg(&args, &mut idx, "--lane-priority")?;
            }
            "--out-spec" => {
                out_spec = Some(next_arg(&args, &mut idx, "--out-spec")?);
            }
            "--out-report" => {
                out_report = Some(next_arg(&args, &mut idx, "--out-report")?);
            }
            "--out-patchspec" => {
                out_patchspec = Some(next_arg(&args, &mut idx, "--out-patchspec")?);
            }
            "--out-patchspec-safe" => {
                out_patchspec_safe = Some(next_arg(&args, &mut idx, "--out-patchspec-safe")?);
            }
            "--out-patchspec-discrete" => {
                out_patchspec_discrete =
                    Some(next_arg(&args, &mut idx, "--out-patchspec-discrete")?);
            }
            other => return Err(format!("unknown flag: {}", other)),
        }
        idx += 1;
    }

    let Some(analysis_json) = analysis_json else {
        return Err("missing --analysis-json".to_string());
    };
    let Some(base_exec) = base_exec else {
        return Err("missing --base-exec".to_string());
    };
    let Some(target_exec) = target_exec else {
        return Err("missing --target-exec".to_string());
    };
    let Some(low_dim) = low_dim else {
        return Err("missing --low-dim".to_string());
    };
    let Some(high_dim) = high_dim else {
        return Err("missing --high-dim".to_string());
    };
    let Some(target_dim) = target_dim else {
        return Err("missing --target-dim".to_string());
    };
    let Some(out_spec) = out_spec else {
        return Err("missing --out-spec".to_string());
    };

    if tile_size <= 0 {
        return Err("--tile-size must be > 0".to_string());
    }
    if mid_exec.is_some() && mid_dim.is_none() {
        return Err("--mid-exec requires --mid-dim".to_string());
    }

    Ok(Config {
        analysis_json,
        base_exec,
        target_exec,
        chunk_index,
        low_dim,
        high_dim,
        target_dim,
        mid_dim,
        mid_exec,
        tile_size,
        predict_mode,
        lane_priority,
        out_spec,
        out_report,
        out_patchspec,
        out_patchspec_safe,
        out_patchspec_discrete,
    })
}
