use super::*;
use crate::cli::{parse_u8_auto, parse_usize_auto};

#[derive(Debug, Clone)]
pub(crate) struct InstructionPatchRule {
    payload_len: usize,
    offset: usize,
    value: u8,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct InstructionPatchSpec {
    pub(crate) by_payload_len: HashMap<usize, Vec<(usize, u8)>>,
}

impl InstructionPatchSpec {
    pub(crate) fn rule_count(&self) -> usize {
        self.by_payload_len.values().map(|v| v.len()).sum()
    }
}

pub(crate) fn load_instruction_patch_spec(
    path: &str,
) -> Result<InstructionPatchSpec, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut rules = Vec::<InstructionPatchRule>::new();
    for (line_idx, raw) in text.lines().enumerate() {
        let clean = raw.split('#').next().unwrap_or("").replace(',', " ");
        let trimmed = clean.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(format!(
                "{}:{} invalid patch rule (expected '<len> <offset> <value>'): {}",
                path,
                line_idx + 1,
                raw
            )
            .into());
        }
        let payload_len = parse_usize_auto(fields[0])?;
        let offset = parse_usize_auto(fields[1])?;
        let value = parse_u8_auto(fields[2], "--instruction-patch-spec value")?;
        rules.push(InstructionPatchRule {
            payload_len,
            offset,
            value,
        });
    }

    if rules.is_empty() {
        return Err(format!(
            "instruction patch spec '{}' has no rules (add '<len> <offset> <value>' lines)",
            path
        )
        .into());
    }

    let mut by_payload_len: HashMap<usize, HashMap<usize, u8>> = HashMap::new();
    for rule in rules {
        by_payload_len
            .entry(rule.payload_len)
            .or_default()
            .insert(rule.offset, rule.value);
    }

    let mut spec = InstructionPatchSpec::default();
    for (payload_len, offsets) in by_payload_len {
        let mut entries: Vec<(usize, u8)> = offsets.into_iter().collect();
        entries.sort_by_key(|(offset, _)| *offset);
        spec.by_payload_len.insert(payload_len, entries);
    }
    Ok(spec)
}

pub(crate) fn merge_instruction_patch_specs(
    paths: &[String],
) -> Result<InstructionPatchSpec, Box<dyn Error>> {
    if paths.is_empty() {
        return Err("merge_instruction_patch_specs called with empty paths".into());
    }

    let mut by_payload_len: HashMap<usize, HashMap<usize, u8>> = HashMap::new();

    for path in paths {
        let spec = load_instruction_patch_spec(path)?;
        for (payload_len, entries) in spec.by_payload_len {
            let table = by_payload_len.entry(payload_len).or_default();
            for (offset, value) in entries {
                if let Some(prev) = table.get(&offset).copied() {
                    if prev != value {
                        return Err(format!(
                            "instruction patch conflict at payload_len={} offset={} existing=0x{:02x} new=0x{:02x} source={}",
                            payload_len,
                            offset,
                            prev,
                            value,
                            path
                        )
                        .into());
                    }
                } else {
                    table.insert(offset, value);
                }
            }
        }
    }

    let mut merged = InstructionPatchSpec::default();
    for (payload_len, offsets) in by_payload_len {
        let mut entries: Vec<(usize, u8)> = offsets.into_iter().collect();
        entries.sort_by_key(|(offset, _)| *offset);
        merged.by_payload_len.insert(payload_len, entries);
    }
    Ok(merged)
}

pub(crate) fn validate_instruction_patch_spec_against_executables(
    spec: &InstructionPatchSpec,
    executables: &[SerializedExecutableBlob],
) -> Result<(), Box<dyn Error>> {
    let mut available_chunk_lens: HashMap<usize, usize> = HashMap::new();
    for exe in executables {
        for chunk in &exe.instruction_bitstreams {
            *available_chunk_lens.entry(chunk.len()).or_insert(0) += 1;
        }
    }

    if available_chunk_lens.is_empty() && spec.rule_count() > 0 {
        return Err("instruction patch spec provided but model has no instruction chunks".into());
    }

    let mut available: Vec<usize> = available_chunk_lens.keys().copied().collect();
    available.sort_unstable();

    for (payload_len, entries) in &spec.by_payload_len {
        if !available_chunk_lens.contains_key(payload_len) {
            return Err(format!(
                "instruction patch payload_len={} not present in model instruction chunks {:?}",
                payload_len, available
            )
            .into());
        }
        for (offset, _) in entries {
            if *offset >= *payload_len {
                return Err(format!(
                    "instruction patch offset out of range: payload_len={} offset={}",
                    payload_len, offset
                )
                .into());
            }
        }
    }

    Ok(())
}

pub(crate) fn descriptor_tag_name(tag: u32) -> &'static str {
    match tag {
        0 => "Instructions",
        1 => "InputActivations",
        2 => "Parameters",
        3 => "OutputActivations",
        4 => "Interrupt0",
        5 => "Interrupt1",
        6 => "Interrupt2",
        7 => "Interrupt3",
        _ => "Custom",
    }
}
