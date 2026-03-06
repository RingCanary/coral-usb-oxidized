use crate::error::DenseGemmError;

const DWN1_IDENTIFIER: &[u8; 4] = b"DWN1";
const TFL3_IDENTIFIER: &[u8; 4] = b"TFL3";
const EXECUTABLE_TYPE_STAND_ALONE: i16 = 0;
const EXECUTABLE_TYPE_PARAMETER_CACHING: i16 = 1;
const EXECUTABLE_TYPE_EXECUTION_ONLY: i16 = 2;

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

#[derive(Debug, Clone)]
pub struct SerializedExecutableBlob {
    pub package_index: usize,
    pub executable_index: usize,
    pub executable_type: i16,
    pub payload: Vec<u8>,
    pub instruction_bitstreams: Vec<Vec<u8>>,
    pub parameters_stream: Vec<u8>,
    // Parameter region offset range relative to payload bytes.
    pub parameter_region: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct Conv1x1QuantParams {
    pub subgraph_index: usize,
    pub weight_tensor_index: usize,
    pub weight_buffer_index: usize,
    pub stored_shape: Vec<i32>,
    pub quantized_dimension: i32,
    pub input_scale: f32,
    pub output_scale: f32,
    pub weight_scales: Vec<f32>,
    pub weight_zero_points: Vec<i64>,
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

fn read_i64(data: &[u8], offset: usize) -> Result<i64, DenseGemmError> {
    let bytes = checked_slice(data, offset, 8, "i64 read")?;
    Ok(i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_f32(data: &[u8], offset: usize) -> Result<f32, DenseGemmError> {
    let bytes = checked_slice(data, offset, 4, "f32 read")?;
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
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
    if vtable_len < 4 || (vtable_len % 2) != 0 {
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

fn parse_table_at(data: &[u8], table_offset: usize) -> Result<FlatTable<'_>, DenseGemmError> {
    checked_slice(data, table_offset, 4, "nested table header")?;

    let vtable_rel = read_i32(data, table_offset)?;
    if vtable_rel == 0 {
        return Err(invalid_template("invalid nested vtable relative offset 0"));
    }

    let vtable_offset = if vtable_rel > 0 {
        table_offset
            .checked_sub(vtable_rel as usize)
            .ok_or_else(|| invalid_template("nested vtable underflow"))?
    } else {
        table_offset
            .checked_add((-vtable_rel) as usize)
            .ok_or_else(|| invalid_template("nested vtable overflow"))?
    };

    let vtable_len = read_u16(data, vtable_offset)? as usize;
    let object_len = read_u16(data, vtable_offset + 2)? as usize;
    if vtable_len < 4 || (vtable_len % 2) != 0 {
        return Err(invalid_template(format!(
            "invalid nested vtable length {}",
            vtable_len
        )));
    }
    if object_len < 4 {
        return Err(invalid_template(format!(
            "invalid nested object length {}",
            object_len
        )));
    }
    checked_slice(data, vtable_offset, vtable_len, "nested vtable bounds")?;
    checked_slice(data, table_offset, object_len, "nested table bounds")?;

    Ok(FlatTable {
        data,
        table_offset,
        vtable_offset,
        vtable_len,
    })
}

fn read_vector_table_offsets(
    table: &FlatTable<'_>,
    field_id: usize,
) -> Result<Vec<usize>, DenseGemmError> {
    let Some(target) = read_offset_object(table, field_id)? else {
        return Ok(Vec::new());
    };

    let length = read_u32(table.data, target)? as usize;
    let vec_start = target
        .checked_add(4)
        .ok_or_else(|| invalid_template("table vector start overflow"))?;
    let vec_bytes = length
        .checked_mul(4)
        .ok_or_else(|| invalid_template("table vector length overflow"))?;
    checked_slice(table.data, vec_start, vec_bytes, "table vector bounds")?;

    let mut out = Vec::with_capacity(length);
    for i in 0..length {
        let slot = vec_start + i * 4;
        let rel = read_u32(table.data, slot)? as usize;
        if rel == 0 {
            continue;
        }
        let obj = slot
            .checked_add(rel)
            .ok_or_else(|| invalid_template("table vector object overflow"))?;
        checked_slice(table.data, obj, 4, "table vector object bounds")?;
        out.push(obj);
    }

    Ok(out)
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

fn read_i32_field(
    table: &FlatTable<'_>,
    field_id: usize,
    default: i32,
) -> Result<i32, DenseGemmError> {
    let Some(off) = table.field_offset(field_id)? else {
        return Ok(default);
    };
    read_i32(table.data, off)
}

fn read_u32_field(
    table: &FlatTable<'_>,
    field_id: usize,
    default: u32,
) -> Result<u32, DenseGemmError> {
    let Some(off) = table.field_offset(field_id)? else {
        return Ok(default);
    };
    read_u32(table.data, off)
}

fn read_vector_i32(table: &FlatTable<'_>, field_id: usize) -> Result<Vec<i32>, DenseGemmError> {
    let Some(target) = read_offset_object(table, field_id)? else {
        return Ok(Vec::new());
    };
    let len = read_u32(table.data, target)? as usize;
    let start = target
        .checked_add(4)
        .ok_or_else(|| invalid_template("i32 vector start overflow"))?;
    let byte_len = len
        .checked_mul(4)
        .ok_or_else(|| invalid_template("i32 vector byte_len overflow"))?;
    checked_slice(table.data, start, byte_len, "i32 vector bounds")?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(read_i32(table.data, start + i * 4)?);
    }
    Ok(out)
}

fn read_vector_i64(table: &FlatTable<'_>, field_id: usize) -> Result<Vec<i64>, DenseGemmError> {
    let Some(target) = read_offset_object(table, field_id)? else {
        return Ok(Vec::new());
    };
    let len = read_u32(table.data, target)? as usize;
    let start = target
        .checked_add(4)
        .ok_or_else(|| invalid_template("i64 vector start overflow"))?;
    let byte_len = len
        .checked_mul(8)
        .ok_or_else(|| invalid_template("i64 vector byte_len overflow"))?;
    checked_slice(table.data, start, byte_len, "i64 vector bounds")?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(read_i64(table.data, start + i * 8)?);
    }
    Ok(out)
}

fn read_vector_f32(table: &FlatTable<'_>, field_id: usize) -> Result<Vec<f32>, DenseGemmError> {
    let Some(target) = read_offset_object(table, field_id)? else {
        return Ok(Vec::new());
    };
    let len = read_u32(table.data, target)? as usize;
    let start = target
        .checked_add(4)
        .ok_or_else(|| invalid_template("f32 vector start overflow"))?;
    let byte_len = len
        .checked_mul(4)
        .ok_or_else(|| invalid_template("f32 vector byte_len overflow"))?;
    checked_slice(table.data, start, byte_len, "f32 vector bounds")?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(read_f32(table.data, start + i * 4)?);
    }
    Ok(out)
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

fn tensor_scale_from_offset(
    blob: &[u8],
    tensor_table_offset: usize,
) -> Result<f32, DenseGemmError> {
    let tensor_table = parse_table_at(blob, tensor_table_offset)?;
    let quant_off = read_offset_object(&tensor_table, 4)?
        .ok_or_else(|| invalid_template("tensor missing quantization table"))?;
    let quant_table = parse_table_at(blob, quant_off)?;
    let scales = read_vector_f32(&quant_table, 2)?;
    scales
        .first()
        .copied()
        .ok_or_else(|| invalid_template("tensor quantization scales missing"))
}

pub fn extract_tflite_conv1x1_quant_params(
    blob: &[u8],
    subgraph_index: usize,
    in_channels: usize,
    out_channels: usize,
) -> Result<Conv1x1QuantParams, DenseGemmError> {
    let model_table = parse_root_table(blob, 0, Some(TFL3_IDENTIFIER))?;
    let subgraphs = read_vector_table_offsets(&model_table, 2)?;
    if subgraph_index >= subgraphs.len() {
        return Err(invalid_template(format!(
            "subgraph index {} out of range (subgraphs={})",
            subgraph_index,
            subgraphs.len()
        )));
    }
    let buffers = read_vector_table_offsets(&model_table, 4)?;
    let subgraph_table = parse_table_at(blob, subgraphs[subgraph_index])?;
    let tensors = read_vector_table_offsets(&subgraph_table, 0)?;
    let inputs = read_vector_i32(&subgraph_table, 1)?;
    let outputs = read_vector_i32(&subgraph_table, 2)?;
    let input_tensor_index = (*inputs
        .first()
        .ok_or_else(|| invalid_template("subgraph missing input tensor indexes"))?)
        as usize;
    let output_tensor_index = (*outputs
        .first()
        .ok_or_else(|| invalid_template("subgraph missing output tensor indexes"))?)
        as usize;
    if input_tensor_index >= tensors.len() || output_tensor_index >= tensors.len() {
        return Err(invalid_template(format!(
            "input/output tensor index out of range (input={} output={} tensors={})",
            input_tensor_index,
            output_tensor_index,
            tensors.len()
        )));
    }

    let input_scale = tensor_scale_from_offset(blob, tensors[input_tensor_index])?;
    let output_scale = tensor_scale_from_offset(blob, tensors[output_tensor_index])?;
    let expected_len = in_channels
        .checked_mul(out_channels)
        .ok_or_else(|| invalid_template("conv1x1 expected_len overflow"))?;
    let accepted_shapes = [
        vec![out_channels as i32, 1, 1, in_channels as i32],
        vec![1, 1, in_channels as i32, out_channels as i32],
    ];

    for (tensor_index, tensor_table_offset) in tensors.iter().copied().enumerate() {
        let tensor_table = parse_table_at(blob, tensor_table_offset)?;
        let shape = read_vector_i32(&tensor_table, 0)?;
        if !accepted_shapes.iter().any(|s| s == &shape) {
            continue;
        }
        let buffer_index = read_u32_field(&tensor_table, 2, 0)? as usize;
        if buffer_index >= buffers.len() {
            continue;
        }
        let buffer_table = parse_table_at(blob, buffers[buffer_index])?;
        let Some(buffer_region) = read_vector_region(&buffer_table, 0)? else {
            continue;
        };
        let buffer_len = buffer_region.end.saturating_sub(buffer_region.start);
        if buffer_len != expected_len {
            continue;
        }
        let Some(quant_off) = read_offset_object(&tensor_table, 4)? else {
            continue;
        };
        let quant_table = parse_table_at(blob, quant_off)?;
        let weight_scales = read_vector_f32(&quant_table, 2)?;
        let weight_zero_points = read_vector_i64(&quant_table, 3)?;
        let quantized_dimension = read_i32_field(&quant_table, 5, 0)?;
        if !(weight_scales.len() == 1 || weight_scales.len() == out_channels) {
            continue;
        }
        if !weight_zero_points.is_empty()
            && !(weight_zero_points.len() == 1 || weight_zero_points.len() == out_channels)
        {
            continue;
        }
        return Ok(Conv1x1QuantParams {
            subgraph_index,
            weight_tensor_index: tensor_index,
            weight_buffer_index: buffer_index,
            stored_shape: shape,
            quantized_dimension,
            input_scale,
            output_scale,
            weight_scales,
            weight_zero_points,
        });
    }

    Err(invalid_template(format!(
        "no candidate 1x1 Conv2D weight tensor found with shapes {:?} and buffer_size={}",
        accepted_shapes, expected_len
    )))
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

pub fn executable_type_name(type_value: i16) -> &'static str {
    match type_value {
        EXECUTABLE_TYPE_STAND_ALONE => "STAND_ALONE",
        EXECUTABLE_TYPE_PARAMETER_CACHING => "PARAMETER_CACHING",
        EXECUTABLE_TYPE_EXECUTION_ONLY => "EXECUTION_ONLY",
        _ => "UNKNOWN",
    }
}

fn extract_instruction_bitstreams_from_payload(
    payload: &[u8],
) -> Result<Vec<Vec<u8>>, DenseGemmError> {
    let executable_table = parse_root_table(payload, 0, None)?;
    let mut instruction_bitstreams = Vec::new();
    for bitstream_table_offset in read_vector_table_offsets(&executable_table, 5)? {
        let bitstream_table = parse_table_at(payload, bitstream_table_offset)?;
        if let Some(bitstream_region) = read_vector_region(&bitstream_table, 0)? {
            let bitstream = payload[bitstream_region.start..bitstream_region.end].to_vec();
            if !bitstream.is_empty() {
                instruction_bitstreams.push(bitstream);
            }
        }
    }
    Ok(instruction_bitstreams)
}

pub fn extract_instruction_chunk_from_serialized_executable(
    payload: &[u8],
    chunk_index: usize,
) -> Result<Vec<u8>, DenseGemmError> {
    let chunks = extract_instruction_bitstreams_from_payload(payload)?;
    if chunk_index >= chunks.len() {
        return Err(invalid_template(format!(
            "instruction chunk index {} out of range (count={})",
            chunk_index,
            chunks.len()
        )));
    }
    Ok(chunks[chunk_index].clone())
}

pub fn extract_serialized_executables_from_tflite(
    blob: &[u8],
) -> Result<Vec<SerializedExecutableBlob>, DenseGemmError> {
    let mut executables = Vec::new();

    for (package_index, root_offset) in scan_dwn1_candidates(blob).into_iter().enumerate() {
        let package_table = match parse_root_table(blob, root_offset, Some(DWN1_IDENTIFIER)) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(multi_region) = read_vector_region(&package_table, 1)? else {
            continue;
        };

        let multi_bytes = &blob[multi_region.start..multi_region.end];
        let regions = parse_multi_executable_layout(multi_bytes)?;

        for (executable_index, region) in regions.iter().enumerate() {
            let abs_start = multi_region
                .start
                .checked_add(region.start)
                .ok_or_else(|| invalid_template("serialized executable start overflow"))?;
            let abs_end = multi_region
                .start
                .checked_add(region.end)
                .ok_or_else(|| invalid_template("serialized executable end overflow"))?;
            if abs_end > blob.len() || abs_start >= abs_end {
                return Err(invalid_template("serialized executable out of bounds"));
            }

            let payload = &blob[abs_start..abs_end];
            let executable_table = parse_root_table(payload, 0, None)?;
            let executable_type = read_i16_field(&executable_table, 13, 0)?;
            let parameter_region =
                read_vector_region(&executable_table, 6)?.map(|r| (r.start, r.end));

            let instruction_bitstreams = extract_instruction_bitstreams_from_payload(payload)?;

            let parameters_stream = match parameter_region {
                Some((start, end)) if end > start && end <= payload.len() => {
                    payload[start..end].to_vec()
                }
                _ => Vec::new(),
            };

            executables.push(SerializedExecutableBlob {
                package_index,
                executable_index,
                executable_type,
                payload: payload.to_vec(),
                instruction_bitstreams,
                parameters_stream,
                parameter_region,
            });
        }
    }

    if executables.is_empty() {
        return Err(invalid_template(
            "no serialized executables found in any DWN1 package",
        ));
    }

    Ok(executables)
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
