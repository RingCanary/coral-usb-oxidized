use coral_usb_oxidized::{
    version, CoralDevice, DenseGemmError, DenseGemmTemplate, PreparedDenseGemm,
};
use std::env;
use std::error::Error;
use std::time::Instant;

const D_MODEL: usize = 2304;
const STAGE_COUNT: usize = 6;

#[derive(Clone, Copy, Debug)]
enum StageMode {
    Identity,
    ShiftPlus1,
    ShiftMinus1,
}

#[derive(Clone, Copy, Debug)]
struct StageSpec {
    name: &'static str,
    mode: StageMode,
}

const STAGES: [StageSpec; STAGE_COUNT] = [
    StageSpec {
        name: "q_proj",
        mode: StageMode::Identity,
    },
    StageSpec {
        name: "k_proj",
        mode: StageMode::ShiftPlus1,
    },
    StageSpec {
        name: "v_proj",
        mode: StageMode::ShiftMinus1,
    },
    StageSpec {
        name: "o_proj",
        mode: StageMode::Identity,
    },
    StageSpec {
        name: "mlp_up",
        mode: StageMode::ShiftPlus1,
    },
    StageSpec {
        name: "mlp_down",
        mode: StageMode::ShiftMinus1,
    },
];

struct PreparedStage {
    spec: StageSpec,
    prepared: PreparedDenseGemm,
}

#[derive(Clone, Copy, Debug, Default)]
struct StageSetupTiming {
    prepare_ms: f64,
    first_invoke_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct RunTiming {
    q_ms: f64,
    k_ms: f64,
    v_ms: f64,
    attn_ms: f64,
    o_ms: f64,
    up_ms: f64,
    down_ms: f64,
    total_ms: f64,
}

impl RunTiming {
    fn add_assign(&mut self, rhs: &Self) {
        self.q_ms += rhs.q_ms;
        self.k_ms += rhs.k_ms;
        self.v_ms += rhs.v_ms;
        self.attn_ms += rhs.attn_ms;
        self.o_ms += rhs.o_ms;
        self.up_ms += rhs.up_ms;
        self.down_ms += rhs.down_ms;
        self.total_ms += rhs.total_ms;
    }

    fn linear_ms(&self) -> f64 {
        self.q_ms + self.k_ms + self.v_ms + self.o_ms + self.up_ms + self.down_ms
    }
}

fn usage(program: &str) {
    println!("Usage: {program} [seq_len] [runs] [warmup] [--no-attention]");
    println!("Defaults: seq_len=8 runs=5 warmup=1");
}

fn parse_args() -> Result<(usize, usize, usize, bool), Box<dyn Error>> {
    let program = env::args()
        .next()
        .unwrap_or_else(|| "transformer_linear_block".to_string());
    let mut positional: Vec<String> = Vec::new();
    let mut use_attention = true;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--help" | "-h" => {
                usage(&program);
                std::process::exit(0);
            }
            "--no-attention" => use_attention = false,
            _ => positional.push(arg),
        }
    }

    let seq_len = positional
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let runs = positional
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5);
    let warmup = positional
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    if seq_len == 0 {
        return Err("seq_len must be >= 1".into());
    }
    if runs == 0 {
        return Err("runs must be >= 1".into());
    }

    Ok((seq_len, runs, warmup, use_attention))
}

fn apply_stage_mode(
    template: &mut DenseGemmTemplate,
    mode: StageMode,
) -> Result<(), DenseGemmError> {
    match mode {
        StageMode::Identity => template.set_identity(127),
        StageMode::ShiftPlus1 => template.set_shift_plus1(127),
        StageMode::ShiftMinus1 => template.set_shift_minus1(127),
    }
}

fn build_prefill_input(seq_len: usize, dim: usize) -> Result<Vec<i8>, Box<dyn Error>> {
    let total = seq_len.checked_mul(dim).ok_or("sequence size overflow")?;
    let mut out = vec![0i8; total];
    for row in 0..seq_len {
        for col in 0..dim {
            let value = ((row + col) & 0xff) as i16 - 128;
            out[row * dim + col] = value as i8;
        }
    }
    Ok(out)
}

fn prepare_stage(
    spec: StageSpec,
    delegate: &coral_usb_oxidized::EdgeTPUDelegate,
    probe_input: &[i8],
) -> Result<(PreparedStage, StageSetupTiming), Box<dyn Error>> {
    let mut template = DenseGemmTemplate::from_bundled_2304()?;
    apply_stage_mode(&mut template, spec.mode)?;

    let started_prepare = Instant::now();
    let prepared = template.prepare(delegate)?;
    let prepare_ms = started_prepare.elapsed().as_secs_f64() * 1000.0;

    let started_first_invoke = Instant::now();
    let _ = prepared.execute_batch_rows(probe_input)?;
    let first_invoke_ms = started_first_invoke.elapsed().as_secs_f64() * 1000.0;

    Ok((
        PreparedStage { spec, prepared },
        StageSetupTiming {
            prepare_ms,
            first_invoke_ms,
        },
    ))
}

fn clamp_i8(value: i32) -> i8 {
    value.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

fn cpu_single_head_attention(
    q: &[i8],
    k: &[i8],
    v: &[i8],
    seq_len: usize,
    dim: usize,
) -> Result<Vec<i8>, Box<dyn Error>> {
    if q.len() != seq_len * dim || k.len() != seq_len * dim || v.len() != seq_len * dim {
        return Err("attention tensor size mismatch".into());
    }

    let scale = 1.0f32 / (dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];

    for t in 0..seq_len {
        let q_row = &q[t * dim..(t + 1) * dim];
        let score_row = &mut scores[t * seq_len..(t + 1) * seq_len];
        for s in 0..seq_len {
            let k_row = &k[s * dim..(s + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += (q_row[d] as f32) * (k_row[d] as f32);
            }
            score_row[s] = dot * scale;
        }

        let max_val = score_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for value in score_row.iter_mut() {
            *value = (*value - max_val).exp();
            denom += *value;
        }
        let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        for value in score_row.iter_mut() {
            *value *= inv;
        }
    }

    let mut out = vec![0i8; seq_len * dim];
    for t in 0..seq_len {
        let score_row = &scores[t * seq_len..(t + 1) * seq_len];
        let out_row = &mut out[t * dim..(t + 1) * dim];
        for d in 0..dim {
            let mut acc = 0.0f32;
            for s in 0..seq_len {
                let v_row = &v[s * dim..(s + 1) * dim];
                acc += score_row[s] * (v_row[d] as f32);
            }
            out_row[d] = clamp_i8(acc.round() as i32);
        }
    }

    Ok(out)
}

fn run_block(
    stages: &[PreparedStage],
    input_rows: &[i8],
    seq_len: usize,
    use_attention: bool,
) -> Result<(Vec<i8>, RunTiming), Box<dyn Error>> {
    if stages.len() != STAGE_COUNT {
        return Err("expected six prepared stages".into());
    }

    let started_total = Instant::now();

    let started = Instant::now();
    let q = stages[0].prepared.execute_batch_rows(input_rows)?;
    let q_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let k = stages[1].prepared.execute_batch_rows(input_rows)?;
    let k_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let v = stages[2].prepared.execute_batch_rows(input_rows)?;
    let v_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let attn_out = if use_attention {
        cpu_single_head_attention(&q, &k, &v, seq_len, D_MODEL)?
    } else {
        q.clone()
    };
    let attn_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let o = stages[3].prepared.execute_batch_rows(&attn_out)?;
    let o_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let up = stages[4].prepared.execute_batch_rows(&o)?;
    let up_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let down = stages[5].prepared.execute_batch_rows(&up)?;
    let down_ms = started.elapsed().as_secs_f64() * 1000.0;

    let timing = RunTiming {
        q_ms,
        k_ms,
        v_ms,
        attn_ms,
        o_ms,
        up_ms,
        down_ms,
        total_ms: started_total.elapsed().as_secs_f64() * 1000.0,
    };

    Ok((down, timing))
}

fn run_same_stage_baseline(
    stage: &PreparedDenseGemm,
    input_rows: &[i8],
) -> Result<(Vec<i8>, f64), Box<dyn Error>> {
    let started = Instant::now();
    let mut current = input_rows.to_vec();
    for _ in 0..STAGE_COUNT {
        current = stage.execute_batch_rows(&current)?;
    }
    Ok((current, started.elapsed().as_secs_f64() * 1000.0))
}

fn checksum_i64(values: &[i8]) -> i64 {
    values.iter().map(|value| *value as i64).sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let (seq_len, runs, warmup, use_attention) = parse_args()?;
    let input_rows = build_prefill_input(seq_len, D_MODEL)?;

    println!("EdgeTPU version: {}", version());
    println!("Transformer linear block benchmark (d_model={D_MODEL}, six 2304x2304 GEMMs)");
    println!(
        "Config: seq_len={} runs={} warmup={} attention={}",
        seq_len,
        runs,
        warmup,
        if use_attention {
            "cpu_single_head"
        } else {
            "disabled"
        }
    );

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;

    let mut prepared_stages: Vec<PreparedStage> = Vec::with_capacity(STAGE_COUNT);
    let mut setup_timings: Vec<StageSetupTiming> = Vec::with_capacity(STAGE_COUNT);
    for spec in STAGES {
        let (prepared, timing) = prepare_stage(spec, &delegate, &input_rows)?;
        prepared_stages.push(prepared);
        setup_timings.push(timing);
    }

    println!("Stage setup timings (includes model-specific prepare and first invoke):");
    for (stage, timing) in prepared_stages.iter().zip(setup_timings.iter()) {
        println!(
            "  {:>8}: prepare_ms={:>8.3} first_invoke_ms={:>8.3} mode={:?}",
            stage.spec.name, timing.prepare_ms, timing.first_invoke_ms, stage.spec.mode
        );
    }

    for _ in 0..warmup {
        let _ = run_block(&prepared_stages, &input_rows, seq_len, use_attention)?;
        let _ = run_same_stage_baseline(&prepared_stages[0].prepared, &input_rows)?;
    }

    let mut summed = RunTiming::default();
    let mut same_stage_total_ms = 0.0f64;
    let mut final_output: Vec<i8> = Vec::new();
    let mut final_baseline: Vec<i8> = Vec::new();

    for run_idx in 0..runs {
        let (output, timing) = run_block(&prepared_stages, &input_rows, seq_len, use_attention)?;
        let (baseline_out, baseline_ms) =
            run_same_stage_baseline(&prepared_stages[0].prepared, &input_rows)?;
        summed.add_assign(&timing);
        same_stage_total_ms += baseline_ms;
        if run_idx + 1 == runs {
            final_output = output;
            final_baseline = baseline_out;
        }
    }

    let inv_runs = 1.0 / runs as f64;
    let avg = RunTiming {
        q_ms: summed.q_ms * inv_runs,
        k_ms: summed.k_ms * inv_runs,
        v_ms: summed.v_ms * inv_runs,
        attn_ms: summed.attn_ms * inv_runs,
        o_ms: summed.o_ms * inv_runs,
        up_ms: summed.up_ms * inv_runs,
        down_ms: summed.down_ms * inv_runs,
        total_ms: summed.total_ms * inv_runs,
    };
    let same_stage_avg_ms = same_stage_total_ms * inv_runs;

    let linear_macs = (STAGE_COUNT as f64) * (seq_len as f64) * (D_MODEL as f64) * (D_MODEL as f64);
    let linear_avg_ms = avg.linear_ms();
    let linear_gmac_per_s = linear_macs / (linear_avg_ms * 1_000_000.0);
    let end_to_end_gmac_per_s = linear_macs / (avg.total_ms * 1_000_000.0);
    let switch_penalty_ms = linear_avg_ms - same_stage_avg_ms;

    println!("Average stage timings over {} run(s):", runs);
    println!(
        "  q={:.3}ms k={:.3}ms v={:.3}ms attn_cpu={:.3}ms o={:.3}ms up={:.3}ms down={:.3}ms",
        avg.q_ms, avg.k_ms, avg.v_ms, avg.attn_ms, avg.o_ms, avg.up_ms, avg.down_ms
    );
    println!(
        "  linear_only_ms={:.3} total_ms={:.3} same_stage6_ms={:.3}",
        linear_avg_ms, avg.total_ms, same_stage_avg_ms
    );
    println!(
        "  linear_gmac_per_s={:.3} end_to_end_gmac_per_s={:.3} switch_penalty_ms={:.3}",
        linear_gmac_per_s, end_to_end_gmac_per_s, switch_penalty_ms
    );
    println!(
        "Output checksums: pipeline={} same_stage6={}",
        checksum_i64(&final_output),
        checksum_i64(&final_baseline)
    );

    Ok(())
}
