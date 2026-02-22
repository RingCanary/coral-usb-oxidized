use crate::error::DenseGemmError;

const DWN1_IDENTIFIER: &[u8; 4] = b"DWN1";
const EXECUTABLE_TYPE_PARAMETER_CACHING: i16 = 1;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Region {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl Region {
    pub(crate) fn size(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

struct FlatTable<'a> {
    data: &'a [u8],
    table_offset: usize,
    vtable_offset: usize,
    vtable_len: usize,
}

impl FlatTable<'_> {
    fn field_offset(&self, field_id: usize) -> Result<Option<usize>, DenseGemmError> {
        let entry = self
            .vtable_offset
            .checked_add(4)
            .and_then(|v| v.checked_add(field_id.saturating_mul(2)))
            .ok_or_else(|| DenseGemmError::InvalidTemplate("vtable entry overflow".to_string()))?;
        if entry + 2 > self.vtable_offset + self.vtable_len {
            return Ok(None);
        }

        let rel = read_u16(self.data, entry)? as usize;
        if rel == 0 {
            return Ok(None);
        }

        let abs = self
            .table_offset
            .checked_add(rel)
            .ok_or_else(|| DenseGemmError::InvalidTemplate("field offset overflow".to_string()))?;
        if abs > self.data.len() {
            return Err(DenseGemmError::InvalidTemplate(format!(
                "field {} offset {} out of range",
                field_id, abs
            )));
        }
        Ok(Some(abs))
    }
}

#[derive(Debug)]
struct ExecutableView {
    type_value: i16,
    parameter_region: Option<Region>,
}

#[derive(Debug)]
pub(crate) struct PackageView {
    executables: Vec<ExecutableView>,
}

fn invalid_template(message: impl Into<String>) -> DenseGemmError {
    DenseGemmError::InvalidTemplate(message.into())
}

fn checked_slice<'a>(
    data: &'a [u8],
    start: usize,
    len: usize,
    what: &str,
) -> Result<&'a [u8], DenseGemmError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid_template(format!("{} range overflow", what)))?;
    if end > data.len() {
        return Err(invalid_template(format!(
            "{} out of bounds: start={} len={} data_len={}",
            what,
            start,
            len,
            data.len()
        )));
    }
    Ok(&data[start..end])
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, DenseGemmError> {
    let bytes = checked_slice(data, offset, 2, "u16 read")?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_i16(data: &[u8], offset: usize) -> Result<i16, DenseGemmError> {
    let bytes = checked_slice(data, offset, 2, "i16 read")?;
    Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, DenseGemmError> {
    let bytes = checked_slice(data, offset, 4, "u32 read")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_i32(data: &[u8], offset: usize) -> Result<i32, DenseGemmError> {
    let bytes = checked_slice(data, offset, 4, "i32 read")?;
    Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn parse_root_table<'a>(
    data: &'a [u8],
    root_offset: usize,
    file_identifier: Option<&[u8; 4]>,
) -> Result<FlatTable<'a>, DenseGemmError> {
    checked_slice(data, root_offset, 4, "root table")?;

    if let Some(id) = file_identifier {
        let got = checked_slice(data, root_offset + 4, 4, "file identifier")?;
        if got != id {
            return Err(invalid_template(format!(
                "identifier mismatch: expected {:?}, got {:?}",
                id, got
            )));
        }
    }

    let table_rel = read_u32(data, root_offset)? as usize;
    let table_offset = root_offset
        .checked_add(table_rel)
        .ok_or_else(|| invalid_template("table pointer overflow"))?;
    checked_slice(data, table_offset, 4, "table pointer")?;

    let vtable_rel = read_i32(data, table_offset)?;
    if vtable_rel == 0 {
        return Err(invalid_template("invalid vtable relative offset 0"));
    }

    let vtable_offset = if vtable_rel > 0 {
        table_offset
            .checked_sub(vtable_rel as usize)
            .ok_or_else(|| invalid_template("vtable underflow"))?
    } else {
        table_offset
            .checked_add((-vtable_rel) as usize)
            .ok_or_else(|| invalid_template("vtable overflow"))?
    };

    let vtable_len = read_u16(data, vtable_offset)? as usize;
    let object_len = read_u16(data, vtable_offset + 2)? as usize;
    if vtable_len < 4 || vtable_len % 2 != 0 {
        return Err(invalid_template(format!(
            "invalid vtable length {}",
            vtable_len
        )));
    }
    if object_len < 4 {
        return Err(invalid_template(format!(
            "invalid object length {}",
            object_len
        )));
    }
    checked_slice(data, vtable_offset, vtable_len, "vtable bounds")?;
    checked_slice(data, table_offset, object_len, "table bounds")?;

    Ok(FlatTable {
        data,
        table_offset,
        vtable_offset,
        vtable_len,
    })
}

fn read_offset_object(
    table: &FlatTable<'_>,
    field_id: usize,
) -> Result<Option<usize>, DenseGemmError> {
    let Some(off) = table.field_offset(field_id)? else {
        return Ok(None);
    };

    let rel = read_u32(table.data, off)? as usize;
    if rel == 0 {
        return Ok(None);
    }

    let target = off
        .checked_add(rel)
        .ok_or_else(|| invalid_template("offset-object overflow"))?;
    checked_slice(table.data, target, 4, "offset-object target")?;
    Ok(Some(target))
}

fn read_vector_region(
    table: &FlatTable<'_>,
    field_id: usize,
) -> Result<Option<Region>, DenseGemmError> {
    let Some(target) = read_offset_object(table, field_id)? else {
        return Ok(None);
    };

    let vlen = read_u32(table.data, target)? as usize;
    let start = target
        .checked_add(4)
        .ok_or_else(|| invalid_template("vector start overflow"))?;
    let end = start
        .checked_add(vlen)
        .ok_or_else(|| invalid_template("vector end overflow"))?;
    checked_slice(table.data, start, vlen, "vector data")?;
    Ok(Some(Region { start, end }))
}

fn read_i16_field(
    table: &FlatTable<'_>,
    field_id: usize,
    default: i16,
) -> Result<i16, DenseGemmError> {
    let Some(off) = table.field_offset(field_id)? else {
        return Ok(default);
    };
    read_i16(table.data, off)
}

fn scan_dwn1_candidates(data: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    if data.len() < 8 {
        return out;
    }

    for idx in 0..=(data.len() - 4) {
        if &data[idx..idx + 4] == DWN1_IDENTIFIER && idx >= 4 {
            let root = idx - 4;
            if !out.contains(&root) {
                out.push(root);
            }
        }
    }
    out
}

fn parse_multi_executable_layout(multi_bytes: &[u8]) -> Result<Vec<Region>, DenseGemmError> {
    let table = parse_root_table(multi_bytes, 0, None)?;
    let Some(vec_target) = read_offset_object(&table, 0)? else {
        return Err(invalid_template(
            "MultiExecutable.serialized_executables missing",
        ));
    };

    let length = read_u32(multi_bytes, vec_target)? as usize;
    let vec_start = vec_target
        .checked_add(4)
        .ok_or_else(|| invalid_template("serialized_executables vector start overflow"))?;
    let vec_bytes = length
        .checked_mul(4)
        .ok_or_else(|| invalid_template("serialized_executables vector length overflow"))?;
    checked_slice(
        multi_bytes,
        vec_start,
        vec_bytes,
        "serialized_executables vector",
    )?;

    let mut regions = Vec::with_capacity(length);
    for i in 0..length {
        let slot = vec_start + i * 4;
        let rel = read_u32(multi_bytes, slot)? as usize;
        let str_off = slot
            .checked_add(rel)
            .ok_or_else(|| invalid_template("serialized executable string offset overflow"))?;
        let str_len = read_u32(multi_bytes, str_off)? as usize;
        let str_start = str_off
            .checked_add(4)
            .ok_or_else(|| invalid_template("serialized executable string start overflow"))?;
        checked_slice(
            multi_bytes,
            str_start,
            str_len,
            "serialized executable string data",
        )?;
        regions.push(Region {
            start: str_start,
            end: str_start + str_len,
        });
    }

    Ok(regions)
}

pub(crate) fn inspect_packages(blob: &[u8]) -> Vec<PackageView> {
    let mut packages = Vec::new();

    for root_offset in scan_dwn1_candidates(blob) {
        let pkg = (|| -> Result<PackageView, DenseGemmError> {
            let package_table = parse_root_table(blob, root_offset, Some(DWN1_IDENTIFIER))?;
            let Some(multi_region) = read_vector_region(&package_table, 1)? else {
                return Err(invalid_template("package missing multi_executable"));
            };
            let multi_bytes = &blob[multi_region.start..multi_region.end];
            let executable_regions = parse_multi_executable_layout(multi_bytes)?;

            let mut executables = Vec::with_capacity(executable_regions.len());
            for executable_region in executable_regions {
                let abs_start = multi_region
                    .start
                    .checked_add(executable_region.start)
                    .ok_or_else(|| invalid_template("executable region start overflow"))?;
                let abs_end = multi_region
                    .start
                    .checked_add(executable_region.end)
                    .ok_or_else(|| invalid_template("executable region end overflow"))?;
                if abs_end > blob.len() || abs_start >= abs_end {
                    return Err(invalid_template("executable region out of bounds"));
                }

                let executable_blob = &blob[abs_start..abs_end];
                let executable_table = parse_root_table(executable_blob, 0, None)?;
                let type_value = read_i16_field(&executable_table, 13, 0)?;
                let parameter_region = read_vector_region(&executable_table, 6)?;
                let parameter_region = parameter_region.map(|region| Region {
                    start: abs_start + region.start,
                    end: abs_start + region.end,
                });

                executables.push(ExecutableView {
                    type_value,
                    parameter_region,
                });
            }

            Ok(PackageView { executables })
        })();

        if let Ok(parsed) = pkg {
            packages.push(parsed);
        }
    }

    packages
}

pub(crate) fn select_dense_parameter_region(
    packages: &[PackageView],
) -> Result<Region, DenseGemmError> {
    let mut first_nonempty = None;
    for package in packages {
        for executable in &package.executables {
            let Some(region) = executable.parameter_region else {
                continue;
            };
            if region.size() == 0 {
                continue;
            }
            if first_nonempty.is_none() {
                first_nonempty = Some(region);
            }
            if executable.type_value == EXECUTABLE_TYPE_PARAMETER_CACHING {
                return Ok(region);
            }
        }
    }
    first_nonempty.ok_or(DenseGemmError::ParameterRegionNotFound)
}
