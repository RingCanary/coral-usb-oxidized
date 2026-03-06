use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseParamPackError {
    InvalidDims {
        rows: usize,
        cols: usize,
        reason: &'static str,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
        what: &'static str,
    },
    IndexOutOfRange {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    Overflow(&'static str),
}

impl fmt::Display for DenseParamPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenseParamPackError::InvalidDims { rows, cols, reason } => {
                write!(
                    f,
                    "invalid dense dims rows={} cols={}: {}",
                    rows, cols, reason
                )
            }
            DenseParamPackError::LengthMismatch {
                expected,
                actual,
                what,
            } => write!(
                f,
                "length mismatch for {}: expected {}, got {}",
                what, expected, actual
            ),
            DenseParamPackError::IndexOutOfRange {
                row,
                col,
                rows,
                cols,
            } => write!(
                f,
                "index out of range row={} col={} for rows={} cols={}",
                row, col, rows, cols
            ),
            DenseParamPackError::Overflow(what) => write!(f, "overflow while computing {}", what),
        }
    }
}

impl std::error::Error for DenseParamPackError {}

fn validate_dims(rows: usize, cols: usize) -> Result<(), DenseParamPackError> {
    if rows == 0 || cols == 0 {
        return Err(DenseParamPackError::InvalidDims {
            rows,
            cols,
            reason: "rows/cols must be >= 1",
        });
    }
    if (rows % 64) != 0 || (cols % 64) != 0 {
        return Err(DenseParamPackError::InvalidDims {
            rows,
            cols,
            reason: "rows/cols must be multiples of 64 for recovered dense packing",
        });
    }
    Ok(())
}

pub fn dense_param_stream_len(rows: usize, cols: usize) -> Result<usize, DenseParamPackError> {
    validate_dims(rows, cols)?;
    rows.checked_mul(cols)
        .ok_or(DenseParamPackError::Overflow("rows*cols"))
}

/// Compute packed parameter stream offset for stored weight tensor coordinates (row, col).
///
/// Recovered layout (tile size 64x64, row-major tiles, local col-group-of-4):
/// off = (row/64)*(cols/64*4096) + (col/64)*4096 + ((col%64)/4)*256 + (row%64)*4 + (col%4)
pub fn dense_param_stream_offset(
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) -> Result<usize, DenseParamPackError> {
    validate_dims(rows, cols)?;
    if row >= rows || col >= cols {
        return Err(DenseParamPackError::IndexOutOfRange {
            row,
            col,
            rows,
            cols,
        });
    }

    let tile_cols = cols / 64;
    let tile_row = row / 64;
    let tile_col = col / 64;

    let tile_strip_stride = tile_cols
        .checked_mul(4096)
        .ok_or(DenseParamPackError::Overflow("tile_cols*4096"))?;

    let base_tile_strip = tile_row
        .checked_mul(tile_strip_stride)
        .ok_or(DenseParamPackError::Overflow("tile_row*tile_strip_stride"))?;
    let base_tile = tile_col
        .checked_mul(4096)
        .ok_or(DenseParamPackError::Overflow("tile_col*4096"))?;
    let local_col_group = (col % 64) / 4;
    let local_col_group_base = local_col_group
        .checked_mul(256)
        .ok_or(DenseParamPackError::Overflow("local_col_group*256"))?;
    let local_row_base = (row % 64)
        .checked_mul(4)
        .ok_or(DenseParamPackError::Overflow("local_row*4"))?;

    let off = base_tile_strip
        .checked_add(base_tile)
        .and_then(|v| v.checked_add(local_col_group_base))
        .and_then(|v| v.checked_add(local_row_base))
        .and_then(|v| v.checked_add(col % 4))
        .ok_or(DenseParamPackError::Overflow("final offset accumulation"))?;

    let len = dense_param_stream_len(rows, cols)?;
    if off >= len {
        return Err(DenseParamPackError::Overflow("offset >= stream_len"));
    }

    Ok(off)
}

pub fn pack_dense_row_major_u8_to_stream(
    rows: usize,
    cols: usize,
    row_major_u8: &[u8],
) -> Result<Vec<u8>, DenseParamPackError> {
    let len = dense_param_stream_len(rows, cols)?;
    if row_major_u8.len() != len {
        return Err(DenseParamPackError::LengthMismatch {
            expected: len,
            actual: row_major_u8.len(),
            what: "row_major_u8",
        });
    }

    let mut out = vec![0u8; len];
    for row in 0..rows {
        let row_base = row
            .checked_mul(cols)
            .ok_or(DenseParamPackError::Overflow("row*cols"))?;
        for col in 0..cols {
            let src_idx = row_base
                .checked_add(col)
                .ok_or(DenseParamPackError::Overflow("row_base+col"))?;
            let dst_idx = dense_param_stream_offset(rows, cols, row, col)?;
            out[dst_idx] = row_major_u8[src_idx];
        }
    }
    Ok(out)
}

pub fn pack_dense_row_major_i8_to_stream(
    rows: usize,
    cols: usize,
    row_major_i8: &[i8],
) -> Result<Vec<u8>, DenseParamPackError> {
    let len = dense_param_stream_len(rows, cols)?;
    if row_major_i8.len() != len {
        return Err(DenseParamPackError::LengthMismatch {
            expected: len,
            actual: row_major_i8.len(),
            what: "row_major_i8",
        });
    }
    let as_u8: Vec<u8> = row_major_i8.iter().map(|&v| v as u8).collect();
    pack_dense_row_major_u8_to_stream(rows, cols, &as_u8)
}

pub fn unpack_dense_stream_to_row_major_u8(
    rows: usize,
    cols: usize,
    stream: &[u8],
) -> Result<Vec<u8>, DenseParamPackError> {
    let len = dense_param_stream_len(rows, cols)?;
    if stream.len() != len {
        return Err(DenseParamPackError::LengthMismatch {
            expected: len,
            actual: stream.len(),
            what: "stream",
        });
    }

    let mut out = vec![0u8; len];
    for row in 0..rows {
        let row_base = row
            .checked_mul(cols)
            .ok_or(DenseParamPackError::Overflow("row*cols"))?;
        for col in 0..cols {
            let src_idx = dense_param_stream_offset(rows, cols, row, col)?;
            let dst_idx = row_base
                .checked_add(col)
                .ok_or(DenseParamPackError::Overflow("row_base+col"))?;
            out[dst_idx] = stream[src_idx];
        }
    }
    Ok(out)
}

pub fn unpack_dense_stream_to_row_major_i8(
    rows: usize,
    cols: usize,
    stream: &[u8],
) -> Result<Vec<i8>, DenseParamPackError> {
    let row_major_u8 = unpack_dense_stream_to_row_major_u8(rows, cols, stream)?;
    Ok(row_major_u8.into_iter().map(|v| v as i8).collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conv1x1ParamPackError {
    InvalidDims {
        in_channels: usize,
        out_channels: usize,
        reason: &'static str,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
        what: &'static str,
    },
    IndexOutOfRange {
        in_channel: usize,
        out_channel: usize,
        in_channels: usize,
        out_channels: usize,
    },
    InvalidQuantization(&'static str),
    StoredZeroPointOutOfRange(i64),
    Overflow(&'static str),
}

impl fmt::Display for Conv1x1ParamPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Conv1x1ParamPackError::InvalidDims {
                in_channels,
                out_channels,
                reason,
            } => write!(
                f,
                "invalid conv1x1 dims in_channels={} out_channels={}: {}",
                in_channels, out_channels, reason
            ),
            Conv1x1ParamPackError::LengthMismatch {
                expected,
                actual,
                what,
            } => write!(
                f,
                "length mismatch for {}: expected {}, got {}",
                what, expected, actual
            ),
            Conv1x1ParamPackError::IndexOutOfRange {
                in_channel,
                out_channel,
                in_channels,
                out_channels,
            } => write!(
                f,
                "index out of range in_channel={} out_channel={} for in_channels={} out_channels={}",
                in_channel, out_channel, in_channels, out_channels
            ),
            Conv1x1ParamPackError::InvalidQuantization(reason) => {
                write!(f, "invalid conv1x1 quantization: {}", reason)
            }
            Conv1x1ParamPackError::StoredZeroPointOutOfRange(value) => {
                write!(f, "stored zero point out of range after +128 bias: {}", value)
            }
            Conv1x1ParamPackError::Overflow(what) => {
                write!(f, "overflow while computing {}", what)
            }
        }
    }
}

impl std::error::Error for Conv1x1ParamPackError {}

fn validate_conv1x1_dims(
    in_channels: usize,
    out_channels: usize,
) -> Result<(), Conv1x1ParamPackError> {
    if in_channels == 0 || out_channels == 0 {
        return Err(Conv1x1ParamPackError::InvalidDims {
            in_channels,
            out_channels,
            reason: "in/out channels must be >= 1",
        });
    }
    if (in_channels % 4) != 0 {
        return Err(Conv1x1ParamPackError::InvalidDims {
            in_channels,
            out_channels,
            reason: "in_channels must be a multiple of 4 for recovered 1x1 Conv2D packing",
        });
    }
    if (out_channels % 32) != 0 {
        return Err(Conv1x1ParamPackError::InvalidDims {
            in_channels,
            out_channels,
            reason: "out_channels must be a multiple of 32 for recovered 1x1 Conv2D packing",
        });
    }
    Ok(())
}

pub fn conv1x1_param_stream_prefix_len(
    out_channels: usize,
) -> Result<usize, Conv1x1ParamPackError> {
    if out_channels == 0 || (out_channels % 32) != 0 {
        return Err(Conv1x1ParamPackError::InvalidDims {
            in_channels: 0,
            out_channels,
            reason:
                "out_channels must be a non-zero multiple of 32 for recovered 1x1 Conv2D packing",
        });
    }
    out_channels
        .checked_mul(8)
        .ok_or(Conv1x1ParamPackError::Overflow(
            "out_channels*8 total_prefix",
        ))
}

pub fn conv1x1_param_stream_len(
    in_channels: usize,
    out_channels: usize,
) -> Result<usize, Conv1x1ParamPackError> {
    validate_conv1x1_dims(in_channels, out_channels)?;
    let prefix_total = conv1x1_param_stream_prefix_len(out_channels)?;
    let weights = in_channels
        .checked_mul(out_channels)
        .ok_or(Conv1x1ParamPackError::Overflow("in_channels*out_channels"))?;
    prefix_total
        .checked_add(weights)
        .ok_or(Conv1x1ParamPackError::Overflow("prefix_total+weights"))
}

fn conv1x1_block_widths(out_channels: usize) -> Result<Vec<usize>, Conv1x1ParamPackError> {
    if out_channels == 0 || (out_channels % 32) != 0 {
        return Err(Conv1x1ParamPackError::InvalidDims {
            in_channels: 0,
            out_channels,
            reason:
                "out_channels must be a non-zero multiple of 32 for recovered 1x1 Conv2D packing",
        });
    }
    let mut remaining = out_channels;
    let mut blocks = Vec::new();
    while remaining > 64 {
        blocks.push(64);
        remaining -= 64;
    }
    blocks.push(remaining);
    Ok(blocks)
}

fn expand_conv1x1_weight_scales(
    out_channels: usize,
    weight_scales: &[f32],
) -> Result<Vec<f32>, Conv1x1ParamPackError> {
    if weight_scales.len() == out_channels {
        return Ok(weight_scales.to_vec());
    }
    if weight_scales.len() == 1 {
        return Ok(vec![weight_scales[0]; out_channels]);
    }
    Err(Conv1x1ParamPackError::LengthMismatch {
        expected: out_channels,
        actual: weight_scales.len(),
        what: "weight_scales (must have len 1 or out_channels)",
    })
}

fn expand_conv1x1_weight_zero_points(
    out_channels: usize,
    weight_zero_points: &[i64],
) -> Result<Vec<i64>, Conv1x1ParamPackError> {
    if weight_zero_points.is_empty() {
        return Ok(vec![0; out_channels]);
    }
    if weight_zero_points.len() == out_channels {
        return Ok(weight_zero_points.to_vec());
    }
    if weight_zero_points.len() == 1 {
        return Ok(vec![weight_zero_points[0]; out_channels]);
    }
    Err(Conv1x1ParamPackError::LengthMismatch {
        expected: out_channels,
        actual: weight_zero_points.len(),
        what: "weight_zero_points (must have len 0, 1, or out_channels)",
    })
}

fn fill_conv1x1_quant_prefix(
    in_channels: usize,
    out_channels: usize,
    effective_scales: &[f32],
    stored_zero_points: &[u32],
    out: &mut [u8],
) -> Result<(), Conv1x1ParamPackError> {
    validate_conv1x1_dims(in_channels, out_channels)?;
    if effective_scales.len() != out_channels {
        return Err(Conv1x1ParamPackError::LengthMismatch {
            expected: out_channels,
            actual: effective_scales.len(),
            what: "effective_scales",
        });
    }
    if stored_zero_points.len() != out_channels {
        return Err(Conv1x1ParamPackError::LengthMismatch {
            expected: out_channels,
            actual: stored_zero_points.len(),
            what: "stored_zero_points",
        });
    }
    if out.len() != conv1x1_param_stream_len(in_channels, out_channels)? {
        return Err(Conv1x1ParamPackError::LengthMismatch {
            expected: conv1x1_param_stream_len(in_channels, out_channels)?,
            actual: out.len(),
            what: "out stream",
        });
    }

    let blocks = conv1x1_block_widths(out_channels)?;
    let mut block_stream_start = 0usize;
    let mut oc_base = 0usize;
    for bw in blocks {
        for local_oc in 0..bw {
            let scale_off = block_stream_start
                .checked_add(
                    local_oc
                        .checked_mul(4)
                        .ok_or(Conv1x1ParamPackError::Overflow("local_oc*4 scale_off"))?,
                )
                .ok_or(Conv1x1ParamPackError::Overflow("scale_off accumulation"))?;
            out[scale_off..scale_off + 4]
                .copy_from_slice(&effective_scales[oc_base + local_oc].to_le_bytes());
        }
        let zp_start = block_stream_start
            .checked_add(
                bw.checked_mul(4)
                    .ok_or(Conv1x1ParamPackError::Overflow("bw*4 zp_start"))?,
            )
            .ok_or(Conv1x1ParamPackError::Overflow("zp_start accumulation"))?;
        for local_oc in 0..bw {
            let zp_off = zp_start
                .checked_add(
                    local_oc
                        .checked_mul(4)
                        .ok_or(Conv1x1ParamPackError::Overflow("local_oc*4 zp_off"))?,
                )
                .ok_or(Conv1x1ParamPackError::Overflow("zp_off accumulation"))?;
            out[zp_off..zp_off + 4]
                .copy_from_slice(&stored_zero_points[oc_base + local_oc].to_le_bytes());
        }
        let block_stride =
            bw.checked_mul(8 + in_channels)
                .ok_or(Conv1x1ParamPackError::Overflow(
                    "bw*(8+in_channels) block_stride",
                ))?;
        block_stream_start =
            block_stream_start
                .checked_add(block_stride)
                .ok_or(Conv1x1ParamPackError::Overflow(
                    "block_stream_start accumulation",
                ))?;
        oc_base += bw;
    }
    Ok(())
}

pub fn conv1x1_effective_scales_from_quant_params(
    out_channels: usize,
    input_scale: f32,
    output_scale: f32,
    weight_scales: &[f32],
) -> Result<Vec<f32>, Conv1x1ParamPackError> {
    if output_scale == 0.0 {
        return Err(Conv1x1ParamPackError::InvalidQuantization(
            "output_scale must be non-zero",
        ));
    }
    let output_recip = 1.0f32 / output_scale;
    let expanded = expand_conv1x1_weight_scales(out_channels, weight_scales)?;
    Ok(expanded
        .into_iter()
        .map(|ws| (input_scale * ws) * output_recip)
        .collect())
}

pub fn conv1x1_stored_zero_points_from_quant_params(
    out_channels: usize,
    weight_zero_points: &[i64],
) -> Result<Vec<u32>, Conv1x1ParamPackError> {
    let expanded = expand_conv1x1_weight_zero_points(out_channels, weight_zero_points)?;
    expanded
        .into_iter()
        .map(|zp| {
            let stored = zp + 128;
            u32::try_from(stored)
                .map_err(|_| Conv1x1ParamPackError::StoredZeroPointOutOfRange(stored))
        })
        .collect()
}

/// Compute packed parameter stream offset for 1x1 Conv2D stored weight coordinates
/// `(in_channel, out_channel)`.
///
/// Current recovered model for tested Conv2D 1x1 cases:
/// - the TFLite constant tensor is stored in `[out_channel, 1, 1, in_channel]` order,
/// - output channels are partitioned into blocks of up to 64 channels,
/// - each block contributes `block_width * 8` prefix bytes,
/// - block-local weight layout is `((in_channel / 4) * (block_width * 4)) + ((out_channel % block_width) * 4) + (in_channel % 4)`.
pub fn conv1x1_param_stream_offset(
    in_channels: usize,
    out_channels: usize,
    in_channel: usize,
    out_channel: usize,
) -> Result<usize, Conv1x1ParamPackError> {
    validate_conv1x1_dims(in_channels, out_channels)?;
    if in_channel >= in_channels || out_channel >= out_channels {
        return Err(Conv1x1ParamPackError::IndexOutOfRange {
            in_channel,
            out_channel,
            in_channels,
            out_channels,
        });
    }

    let blocks = conv1x1_block_widths(out_channels)?;
    let mut block_start = 0usize;
    let mut remaining_out = out_channel;
    let mut block_width = None;
    for bw in blocks {
        if remaining_out < bw {
            block_width = Some(bw);
            break;
        }
        let stride = bw
            .checked_mul(8 + in_channels)
            .ok_or(Conv1x1ParamPackError::Overflow(
                "block stride bw*(8+in_channels)",
            ))?;
        block_start = block_start
            .checked_add(stride)
            .ok_or(Conv1x1ParamPackError::Overflow("accumulated block_start"))?;
        remaining_out -= bw;
    }
    let block_width = block_width.ok_or(Conv1x1ParamPackError::Overflow(
        "failed to resolve output block",
    ))?;

    let block_prefix = block_width
        .checked_mul(8)
        .ok_or(Conv1x1ParamPackError::Overflow(
            "block_width*8 block_prefix",
        ))?;
    let ic_group_stride = block_width
        .checked_mul(4)
        .ok_or(Conv1x1ParamPackError::Overflow(
            "block_width*4 ic_group_stride",
        ))?;
    let group_base =
        (in_channel / 4)
            .checked_mul(ic_group_stride)
            .ok_or(Conv1x1ParamPackError::Overflow(
                "(in_channel/4)*ic_group_stride",
            ))?;
    let out_base = remaining_out
        .checked_mul(4)
        .ok_or(Conv1x1ParamPackError::Overflow("remaining_out*4"))?;
    let off = block_start
        .checked_add(block_prefix)
        .and_then(|v| v.checked_add(group_base))
        .and_then(|v| v.checked_add(out_base))
        .and_then(|v| v.checked_add(in_channel % 4))
        .ok_or(Conv1x1ParamPackError::Overflow("final offset accumulation"))?;

    let len = conv1x1_param_stream_len(in_channels, out_channels)?;
    if off >= len {
        return Err(Conv1x1ParamPackError::Overflow("offset >= stream_len"));
    }
    Ok(off)
}

pub fn pack_conv1x1_row_major_u8_to_stream(
    in_channels: usize,
    out_channels: usize,
    row_major_u8: &[u8],
) -> Result<Vec<u8>, Conv1x1ParamPackError> {
    validate_conv1x1_dims(in_channels, out_channels)?;
    let weights_len = in_channels
        .checked_mul(out_channels)
        .ok_or(Conv1x1ParamPackError::Overflow("in_channels*out_channels"))?;
    if row_major_u8.len() != weights_len {
        return Err(Conv1x1ParamPackError::LengthMismatch {
            expected: weights_len,
            actual: row_major_u8.len(),
            what: "row_major_u8",
        });
    }

    let mut out = vec![0u8; conv1x1_param_stream_len(in_channels, out_channels)?];
    for oc in 0..out_channels {
        let src_base = oc
            .checked_mul(in_channels)
            .ok_or(Conv1x1ParamPackError::Overflow("oc*in_channels"))?;
        for ic in 0..in_channels {
            let src_idx = src_base
                .checked_add(ic)
                .ok_or(Conv1x1ParamPackError::Overflow("src_base+ic"))?;
            let dst_idx = conv1x1_param_stream_offset(in_channels, out_channels, ic, oc)?;
            out[dst_idx] = row_major_u8[src_idx].wrapping_add(128);
        }
    }
    Ok(out)
}

pub fn pack_conv1x1_row_major_i8_to_stream(
    in_channels: usize,
    out_channels: usize,
    row_major_i8: &[i8],
) -> Result<Vec<u8>, Conv1x1ParamPackError> {
    validate_conv1x1_dims(in_channels, out_channels)?;
    let weights_len = in_channels
        .checked_mul(out_channels)
        .ok_or(Conv1x1ParamPackError::Overflow("in_channels*out_channels"))?;
    if row_major_i8.len() != weights_len {
        return Err(Conv1x1ParamPackError::LengthMismatch {
            expected: weights_len,
            actual: row_major_i8.len(),
            what: "row_major_i8",
        });
    }
    let mut out = vec![0u8; conv1x1_param_stream_len(in_channels, out_channels)?];
    for oc in 0..out_channels {
        let src_base = oc
            .checked_mul(in_channels)
            .ok_or(Conv1x1ParamPackError::Overflow("oc*in_channels"))?;
        for ic in 0..in_channels {
            let src_idx = src_base
                .checked_add(ic)
                .ok_or(Conv1x1ParamPackError::Overflow("src_base+ic"))?;
            let dst_idx = conv1x1_param_stream_offset(in_channels, out_channels, ic, oc)?;
            out[dst_idx] = ((row_major_i8[src_idx] as i16) + 128) as u8;
        }
    }
    Ok(out)
}

pub fn pack_conv1x1_row_major_u8_to_stream_with_quant_params(
    in_channels: usize,
    out_channels: usize,
    row_major_u8: &[u8],
    input_scale: f32,
    output_scale: f32,
    weight_scales: &[f32],
    weight_zero_points: &[i64],
) -> Result<Vec<u8>, Conv1x1ParamPackError> {
    let mut out = pack_conv1x1_row_major_u8_to_stream(in_channels, out_channels, row_major_u8)?;
    let effective_scales = conv1x1_effective_scales_from_quant_params(
        out_channels,
        input_scale,
        output_scale,
        weight_scales,
    )?;
    let stored_zero_points =
        conv1x1_stored_zero_points_from_quant_params(out_channels, weight_zero_points)?;
    fill_conv1x1_quant_prefix(
        in_channels,
        out_channels,
        &effective_scales,
        &stored_zero_points,
        &mut out,
    )?;
    Ok(out)
}

pub fn pack_conv1x1_row_major_i8_to_stream_with_quant_params(
    in_channels: usize,
    out_channels: usize,
    row_major_i8: &[i8],
    input_scale: f32,
    output_scale: f32,
    weight_scales: &[f32],
    weight_zero_points: &[i64],
) -> Result<Vec<u8>, Conv1x1ParamPackError> {
    let mut out = pack_conv1x1_row_major_i8_to_stream(in_channels, out_channels, row_major_i8)?;
    let effective_scales = conv1x1_effective_scales_from_quant_params(
        out_channels,
        input_scale,
        output_scale,
        weight_scales,
    )?;
    let stored_zero_points =
        conv1x1_stored_zero_points_from_quant_params(out_channels, weight_zero_points)?;
    fill_conv1x1_quant_prefix(
        in_channels,
        out_channels,
        &effective_scales,
        &stored_zero_points,
        &mut out,
    )?;
    Ok(out)
}

pub fn unpack_conv1x1_stream_to_row_major_u8(
    in_channels: usize,
    out_channels: usize,
    stream: &[u8],
) -> Result<Vec<u8>, Conv1x1ParamPackError> {
    let len = conv1x1_param_stream_len(in_channels, out_channels)?;
    if stream.len() != len {
        return Err(Conv1x1ParamPackError::LengthMismatch {
            expected: len,
            actual: stream.len(),
            what: "stream",
        });
    }

    let mut out = vec![0u8; in_channels * out_channels];
    for oc in 0..out_channels {
        let dst_base = oc
            .checked_mul(in_channels)
            .ok_or(Conv1x1ParamPackError::Overflow("oc*in_channels"))?;
        for ic in 0..in_channels {
            let src_idx = conv1x1_param_stream_offset(in_channels, out_channels, ic, oc)?;
            let dst_idx = dst_base
                .checked_add(ic)
                .ok_or(Conv1x1ParamPackError::Overflow("dst_base+ic"))?;
            out[dst_idx] = stream[src_idx].wrapping_sub(128);
        }
    }
    Ok(out)
}

pub fn unpack_conv1x1_stream_to_row_major_i8(
    in_channels: usize,
    out_channels: usize,
    stream: &[u8],
) -> Result<Vec<i8>, Conv1x1ParamPackError> {
    let row_major_u8 = unpack_conv1x1_stream_to_row_major_u8(in_channels, out_channels, stream)?;
    Ok(row_major_u8.into_iter().map(|v| v as i8).collect())
}

#[cfg(test)]
mod tests {
    use super::{
        conv1x1_effective_scales_from_quant_params, conv1x1_param_stream_len,
        conv1x1_param_stream_offset, dense_param_stream_len, dense_param_stream_offset,
        pack_conv1x1_row_major_i8_to_stream, pack_conv1x1_row_major_i8_to_stream_with_quant_params,
        pack_conv1x1_row_major_u8_to_stream, pack_dense_row_major_i8_to_stream,
        pack_dense_row_major_u8_to_stream, unpack_conv1x1_stream_to_row_major_i8,
        unpack_conv1x1_stream_to_row_major_u8, unpack_dense_stream_to_row_major_i8,
        unpack_dense_stream_to_row_major_u8, Conv1x1ParamPackError, DenseParamPackError,
    };

    #[test]
    fn offset_known_points() {
        assert_eq!(dense_param_stream_offset(64, 64, 0, 0).unwrap(), 0);
        assert_eq!(dense_param_stream_offset(64, 64, 0, 1).unwrap(), 1);
        assert_eq!(dense_param_stream_offset(64, 64, 1, 0).unwrap(), 4);
        assert_eq!(dense_param_stream_offset(64, 64, 0, 4).unwrap(), 256);
        assert_eq!(dense_param_stream_offset(64, 64, 63, 63).unwrap(), 4095);

        assert_eq!(dense_param_stream_offset(128, 64, 64, 0).unwrap(), 4096);
        assert_eq!(dense_param_stream_offset(64, 128, 0, 64).unwrap(), 4096);
    }

    #[test]
    fn offsets_form_bijection_for_128x128() {
        let rows = 128usize;
        let cols = 128usize;
        let len = dense_param_stream_len(rows, cols).unwrap();
        let mut seen = vec![false; len];

        for r in 0..rows {
            for c in 0..cols {
                let off = dense_param_stream_offset(rows, cols, r, c).unwrap();
                assert!(!seen[off], "duplicate offset {off} at r={r} c={c}");
                seen[off] = true;
            }
        }
        assert!(seen.into_iter().all(|v| v));
    }

    #[test]
    fn roundtrip_u8_pattern_896() {
        let rows = 896usize;
        let cols = 896usize;
        let len = rows * cols;
        let src: Vec<u8> = (0..len)
            .map(|i| (((i % 251) as i16 - 128).rem_euclid(256)) as u8)
            .collect();

        let stream = pack_dense_row_major_u8_to_stream(rows, cols, &src).unwrap();
        let restored = unpack_dense_stream_to_row_major_u8(rows, cols, &stream).unwrap();
        assert_eq!(src, restored);
    }

    #[test]
    fn roundtrip_i8_pattern_896() {
        let rows = 896usize;
        let cols = 896usize;
        let len = rows * cols;
        let src_i8: Vec<i8> = (0..len).map(|i| ((i % 251) as i16 - 128) as i8).collect();

        let stream = pack_dense_row_major_i8_to_stream(rows, cols, &src_i8).unwrap();
        let restored_i8 = unpack_dense_stream_to_row_major_i8(rows, cols, &stream).unwrap();
        assert_eq!(src_i8, restored_i8);
    }

    #[test]
    fn rejects_invalid_dims() {
        let err = dense_param_stream_len(63, 64).unwrap_err();
        assert!(matches!(
            err,
            DenseParamPackError::InvalidDims {
                rows: 63,
                cols: 64,
                ..
            }
        ));
    }

    #[test]
    fn rejects_length_mismatch() {
        let err = pack_dense_row_major_u8_to_stream(64, 64, &[0u8; 3]).unwrap_err();
        assert!(matches!(
            err,
            DenseParamPackError::LengthMismatch {
                expected: 4096,
                actual: 3,
                ..
            }
        ));
    }

    #[test]
    fn conv1x1_prefix_and_known_points() {
        assert_eq!(conv1x1_param_stream_len(32, 32).unwrap(), 1280);
        assert_eq!(conv1x1_param_stream_len(64, 64).unwrap(), 4608);
        assert_eq!(conv1x1_param_stream_len(64, 128).unwrap(), 9216);

        assert_eq!(conv1x1_param_stream_offset(64, 64, 0, 0).unwrap(), 512);
        assert_eq!(conv1x1_param_stream_offset(64, 64, 0, 1).unwrap(), 516);
        assert_eq!(conv1x1_param_stream_offset(64, 64, 1, 0).unwrap(), 513);
        assert_eq!(conv1x1_param_stream_offset(64, 64, 31, 31).unwrap(), 2431);
        assert_eq!(conv1x1_param_stream_offset(64, 64, 63, 63).unwrap(), 4607);

        assert_eq!(conv1x1_param_stream_offset(32, 32, 0, 1).unwrap(), 260);
        assert_eq!(conv1x1_param_stream_offset(64, 128, 0, 1).unwrap(), 516);
        assert_eq!(conv1x1_param_stream_offset(64, 128, 63, 127).unwrap(), 9215);
        assert_eq!(
            conv1x1_param_stream_offset(128, 128, 127, 127).unwrap(),
            17407
        );
    }

    #[test]
    fn conv1x1_offsets_form_bijection_for_64x128_weights_region() {
        let in_channels = 64usize;
        let out_channels = 128usize;
        let len = conv1x1_param_stream_len(in_channels, out_channels).unwrap();
        let mut seen = vec![false; len];

        for ic in 0..in_channels {
            for oc in 0..out_channels {
                let off = conv1x1_param_stream_offset(in_channels, out_channels, ic, oc).unwrap();
                assert!(!seen[off], "duplicate offset {off} at ic={ic} oc={oc}");
                seen[off] = true;
            }
        }
        let unused = seen.iter().enumerate().filter(|(_, v)| !**v).count();
        assert_eq!(unused, out_channels * 8);
    }

    #[test]
    fn conv1x1_roundtrip_u8_pattern() {
        let in_channels = 128usize;
        let out_channels = 64usize;
        let len = in_channels * out_channels;
        let src: Vec<u8> = (0..len)
            .map(|i| (((i % 251) as i16 - 128).rem_euclid(256)) as u8)
            .collect();
        let stream = pack_conv1x1_row_major_u8_to_stream(in_channels, out_channels, &src).unwrap();
        let restored =
            unpack_conv1x1_stream_to_row_major_u8(in_channels, out_channels, &stream).unwrap();
        assert_eq!(src, restored);
    }

    #[test]
    fn conv1x1_roundtrip_i8_pattern() {
        let in_channels = 64usize;
        let out_channels = 128usize;
        let len = in_channels * out_channels;
        let src_i8: Vec<i8> = (0..len).map(|i| ((i % 251) as i16 - 128) as i8).collect();
        let stream =
            pack_conv1x1_row_major_i8_to_stream(in_channels, out_channels, &src_i8).unwrap();
        let restored =
            unpack_conv1x1_stream_to_row_major_i8(in_channels, out_channels, &stream).unwrap();
        assert_eq!(src_i8, restored);
    }

    #[test]
    fn conv1x1_quant_prefix_matches_blockwise_layout() {
        let in_channels = 64usize;
        let out_channels = 128usize;
        let len = in_channels * out_channels;
        let weights: Vec<i8> = (0..len).map(|i| ((i % 251) as i16 - 128) as i8).collect();
        let weight_scales: Vec<f32> = (0..out_channels)
            .map(|oc| 0.25f32 + (oc as f32) * 0.001f32)
            .collect();
        let zero_points = vec![0i64; out_channels];
        let input_scale = 0.125f32;
        let output_scale = 0.5f32;
        let effective = conv1x1_effective_scales_from_quant_params(
            out_channels,
            input_scale,
            output_scale,
            &weight_scales,
        )
        .unwrap();
        let stream = pack_conv1x1_row_major_i8_to_stream_with_quant_params(
            in_channels,
            out_channels,
            &weights,
            input_scale,
            output_scale,
            &weight_scales,
            &zero_points,
        )
        .unwrap();

        assert_eq!(
            f32::from_le_bytes(stream[0..4].try_into().unwrap()),
            effective[0]
        );
        assert_eq!(
            f32::from_le_bytes(stream[252..256].try_into().unwrap()),
            effective[63]
        );
        assert_eq!(
            u32::from_le_bytes(stream[256..260].try_into().unwrap()),
            128
        );
        assert_eq!(
            u32::from_le_bytes(stream[508..512].try_into().unwrap()),
            128
        );
        assert_eq!(
            f32::from_le_bytes(stream[4608..4612].try_into().unwrap()),
            effective[64]
        );
        assert_eq!(
            u32::from_le_bytes(stream[4864..4868].try_into().unwrap()),
            128
        );

        let off0 = conv1x1_param_stream_offset(in_channels, out_channels, 0, 0).unwrap();
        let off1 = conv1x1_param_stream_offset(in_channels, out_channels, 63, 127).unwrap();
        assert_eq!(stream[off0], ((weights[0] as i16) + 128) as u8);
        assert_eq!(stream[off1], ((weights[len - 1] as i16) + 128) as u8);
    }

    #[test]
    fn conv1x1_rejects_invalid_dims() {
        let err = conv1x1_param_stream_len(30, 32).unwrap_err();
        assert!(matches!(
            err,
            Conv1x1ParamPackError::InvalidDims {
                in_channels: 30,
                out_channels: 32,
                ..
            }
        ));
    }
}
