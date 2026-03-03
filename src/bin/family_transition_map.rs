use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize)]
struct FamilyRecord {
    dim: usize,
    eo_instr_bytes: usize,
    pc_instr_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct FamilySummary {
    family_id: String,
    eo_instr_bytes: usize,
    pc_instr_bytes: usize,
    dims: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct FamilySpan {
    start_dim: usize,
    end_dim: usize,
    family_id: String,
    eo_instr_bytes: usize,
    pc_instr_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct TransitionEdge {
    from_dim: usize,
    to_dim: usize,
    from_family_id: String,
    to_family_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct TransitionReport {
    input_tsv: String,
    records: Vec<FamilyRecord>,
    family_count: usize,
    recurrent_family_count: usize,
    families: Vec<FamilySummary>,
    recurrent_families: Vec<FamilySummary>,
    spans: Vec<FamilySpan>,
    transitions: Vec<TransitionEdge>,
}

fn usage(program: &str) {
    eprintln!("Usage: {program} --input PATH --out-json PATH [--out-md PATH]");
}

fn parse_args() -> Result<(PathBuf, PathBuf, Option<PathBuf>), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "family_transition_map".to_string());

    let mut input: Option<PathBuf> = None;
    let mut out_json: Option<PathBuf> = None;
    let mut out_md: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                std::process::exit(0);
            }
            "--input" => {
                i += 1;
                input = Some(PathBuf::from(args.get(i).ok_or("--input requires value")?));
            }
            "--out-json" => {
                i += 1;
                out_json = Some(PathBuf::from(
                    args.get(i).ok_or("--out-json requires value")?,
                ));
            }
            "--out-md" => {
                i += 1;
                out_md = Some(PathBuf::from(args.get(i).ok_or("--out-md requires value")?));
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let input = input.ok_or("missing --input")?;
    let out_json = out_json.ok_or("missing --out-json")?;
    Ok((input, out_json, out_md))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (input_path, out_json_path, out_md_path) = parse_args()?;

    let text = fs::read_to_string(&input_path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty TSV")?;
    let columns: Vec<&str> = header.split('\t').collect();

    let col_dim = columns
        .iter()
        .position(|c| *c == "dim")
        .ok_or("missing dim column")?;
    let col_eo = columns
        .iter()
        .position(|c| *c == "eo_instr_bytes")
        .ok_or("missing eo_instr_bytes column")?;
    let col_pc = columns
        .iter()
        .position(|c| *c == "pc_instr_bytes")
        .ok_or("missing pc_instr_bytes column")?;

    let mut record_set: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() <= col_pc {
            continue;
        }
        let dim = match cells[col_dim].trim().parse::<usize>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let eo = match cells[col_eo].trim().parse::<usize>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let pc = match cells[col_pc].trim().parse::<usize>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        record_set.insert((dim, eo, pc));
    }

    let mut records: Vec<FamilyRecord> = record_set
        .into_iter()
        .map(|(dim, eo, pc)| FamilyRecord {
            dim,
            eo_instr_bytes: eo,
            pc_instr_bytes: pc,
        })
        .collect();
    records.sort_by_key(|r| r.dim);

    let mut families_map: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for r in &records {
        families_map
            .entry((r.eo_instr_bytes, r.pc_instr_bytes))
            .or_default()
            .push(r.dim);
    }

    let families: Vec<FamilySummary> = families_map
        .iter()
        .map(|((eo, pc), dims)| FamilySummary {
            family_id: format!("eo{}_pc{}", eo, pc),
            eo_instr_bytes: *eo,
            pc_instr_bytes: *pc,
            dims: dims.clone(),
        })
        .collect();

    let mut spans: Vec<FamilySpan> = Vec::new();
    let mut transitions: Vec<TransitionEdge> = Vec::new();

    if !records.is_empty() {
        let mut span_start = records[0].dim;
        let mut span_end = records[0].dim;
        let mut cur_eo = records[0].eo_instr_bytes;
        let mut cur_pc = records[0].pc_instr_bytes;

        for pair in records.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if b.eo_instr_bytes == cur_eo && b.pc_instr_bytes == cur_pc {
                span_end = b.dim;
            } else {
                spans.push(FamilySpan {
                    start_dim: span_start,
                    end_dim: span_end,
                    family_id: format!("eo{}_pc{}", cur_eo, cur_pc),
                    eo_instr_bytes: cur_eo,
                    pc_instr_bytes: cur_pc,
                });
                transitions.push(TransitionEdge {
                    from_dim: a.dim,
                    to_dim: b.dim,
                    from_family_id: format!("eo{}_pc{}", a.eo_instr_bytes, a.pc_instr_bytes),
                    to_family_id: format!("eo{}_pc{}", b.eo_instr_bytes, b.pc_instr_bytes),
                });
                span_start = b.dim;
                span_end = b.dim;
                cur_eo = b.eo_instr_bytes;
                cur_pc = b.pc_instr_bytes;
            }
        }

        spans.push(FamilySpan {
            start_dim: span_start,
            end_dim: span_end,
            family_id: format!("eo{}_pc{}", cur_eo, cur_pc),
            eo_instr_bytes: cur_eo,
            pc_instr_bytes: cur_pc,
        });
    }

    let recurrent_families: Vec<FamilySummary> = families
        .iter()
        .filter(|f| f.dims.len() >= 2)
        .cloned()
        .collect();

    let report = TransitionReport {
        input_tsv: input_path.to_string_lossy().into_owned(),
        records,
        family_count: families.len(),
        recurrent_family_count: recurrent_families.len(),
        families,
        recurrent_families,
        spans,
        transitions,
    };

    if let Some(parent) = out_json_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_json_path, serde_json::to_string_pretty(&report)?)?;

    if let Some(md_path) = out_md_path {
        if let Some(parent) = md_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut md = String::new();
        md.push_str("# Dense instruction family transition map\n\n");
        md.push_str(&format!("Source TSV: `{}`\n\n", report.input_tsv));
        md.push_str(&format!(
            "Known paired families: **{}**\n",
            report.family_count
        ));
        md.push_str(&format!(
            "Recurrent families (>=2 sampled dims): **{}**\n\n",
            report.recurrent_family_count
        ));
        md.push_str("## Families\n");
        for f in &report.families {
            md.push_str(&format!(
                "- `{}` (EO={}, PC={}): dims {:?}\n",
                f.family_id, f.eo_instr_bytes, f.pc_instr_bytes, f.dims
            ));
        }
        md.push_str("\n## Recurrent families\n");
        for f in &report.recurrent_families {
            md.push_str(&format!("- `{}` dims {:?}\n", f.family_id, f.dims));
        }
        md.push_str("\n## Observed transitions in sampled dims\n");
        if report.transitions.is_empty() {
            md.push_str("- none\n");
        } else {
            for t in &report.transitions {
                md.push_str(&format!(
                    "- {}@{} -> {}@{}\n",
                    t.from_family_id, t.from_dim, t.to_family_id, t.to_dim
                ));
            }
        }
        fs::write(md_path, md)?;
    }

    println!(
        "wrote transition map: families={} records={} json={}",
        report.family_count,
        report.records.len(),
        out_json_path.display()
    );
    Ok(())
}
