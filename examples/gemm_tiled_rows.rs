use coral_usb_oxidized::{version, CoralDevice, DenseGemmTemplate};
use std::env;
use std::error::Error;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum RowMode {
    IdentityCycle,
    ShiftPlus1Cycle,
}

fn parse_mode(value: &str) -> Result<RowMode, Box<dyn Error>> {
    match value {
        "identity_cycle" => Ok(RowMode::IdentityCycle),
        "shift_plus1_cycle" => Ok(RowMode::ShiftPlus1Cycle),
        _ => {
            Err(format!("unknown mode: {value} (expected identity_cycle|shift_plus1_cycle)").into())
        }
    }
}

fn selected_input_row(global_output_row: usize, tile_dim: usize, mode: RowMode) -> usize {
    match mode {
        RowMode::IdentityCycle => global_output_row % tile_dim,
        RowMode::ShiftPlus1Cycle => (global_output_row + 1) % tile_dim,
    }
}

fn build_input(tile_dim: usize) -> Vec<i8> {
    let mut input = vec![0i8; tile_dim];
    for (idx, value) in input.iter_mut().enumerate() {
        *value = idx as i8;
    }
    input
}

fn tiled_rows_execute(
    rows_total: usize,
    mode: RowMode,
    input: &[i8],
    delegate: &coral_usb_oxidized::EdgeTPUDelegate,
) -> Result<Vec<i8>, Box<dyn Error>> {
    let base = DenseGemmTemplate::from_bundled_2688()?;
    let (tile_input_dim, tile_output_dim) = base.dims();
    if tile_input_dim != tile_output_dim {
        return Err("bundled 2688 template is expected to be square".into());
    }
    if input.len() != tile_input_dim {
        return Err(format!(
            "input length mismatch: expected {}, got {}",
            tile_input_dim,
            input.len()
        )
        .into());
    }

    let mut output = vec![0i8; rows_total];
    for row_base in (0..rows_total).step_by(tile_output_dim) {
        let row_block = (rows_total - row_base).min(tile_output_dim);
        let mut tile = base.clone();
        tile.fill_matrix_qi8(0)?;

        for local_row in 0..row_block {
            let global_output_row = row_base + local_row;
            let input_row = selected_input_row(global_output_row, tile_input_dim, mode);
            // set_weight_qi8(row, col): row=input index, col=output index.
            tile.set_weight_qi8(input_row, local_row, 127)?;
        }

        let prepared = tile.prepare(delegate)?;
        let tile_output = prepared.execute(input)?;
        output[row_base..row_base + row_block].copy_from_slice(&tile_output[..row_block]);
    }

    Ok(output)
}

fn verify(rows_total: usize, mode: RowMode, input: &[i8], output: &[i8]) -> (usize, i16) {
    let mut mismatches = 0usize;
    let mut max_abs_delta = 0i16;
    let tile_dim = input.len();

    for row in 0..rows_total {
        let expected = input[selected_input_row(row, tile_dim, mode)];
        let got = output[row];
        let delta = (got as i16 - expected as i16).abs();
        if delta > 1 {
            mismatches += 1;
        }
        if delta > max_abs_delta {
            max_abs_delta = delta;
        }
    }

    (mismatches, max_abs_delta)
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
    let rows_total = args
        .get(1)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192);
    let mode = parse_mode(args.get(2).map(String::as_str).unwrap_or("identity_cycle"))?;
    let runs = args
        .get(3)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);

    if rows_total == 0 {
        return Err("rows_total must be >= 1".into());
    }
    if runs == 0 {
        return Err("runs must be >= 1".into());
    }

    let probe = DenseGemmTemplate::from_bundled_2688()?;
    let (tile_dim, _) = probe.dims();
    let input = build_input(tile_dim);

    println!("EdgeTPU version: {}", version());
    println!("Mode: {:?}", mode);
    println!("Rows total: {}", rows_total);
    println!("Tile dim: {}", tile_dim);
    println!("Runs: {}", runs);

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;

    let mut output = vec![0i8; rows_total];
    let mut total_ms = 0.0f64;
    for run_idx in 0..runs {
        let started = Instant::now();
        let current = tiled_rows_execute(rows_total, mode, &input, &delegate)?;
        total_ms += started.elapsed().as_secs_f64() * 1000.0;
        if run_idx + 1 == runs {
            output = current;
        }
    }

    let avg_ms = total_ms / runs as f64;
    let macs_per_run = (rows_total as f64) * (tile_dim as f64);
    let effective_gmac_per_s = macs_per_run / (avg_ms * 1_000_000.0);
    println!(
        "Latency: avg_ms={:.3} total_ms={:.3} effective_gmac_per_s={:.3}",
        avg_ms, total_ms, effective_gmac_per_s
    );

    preview("Input", &input, 32);
    preview("Output", &output, 32);

    let (mismatches, max_abs_delta) = verify(rows_total, mode, &input, &output);
    println!(
        "Verification: mismatches(|delta|>1)={} max_abs_delta={}",
        mismatches, max_abs_delta
    );
    println!(
        "Tiling note: this executes row tiles of 2688x2688 to emulate matrices larger than on-chip parameter capacity."
    );

    Ok(())
}
