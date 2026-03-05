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
    if !rows.is_multiple_of(64) || !cols.is_multiple_of(64) {
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

#[cfg(test)]
mod tests {
    use super::{
        dense_param_stream_len, dense_param_stream_offset, pack_dense_row_major_i8_to_stream,
        pack_dense_row_major_u8_to_stream, unpack_dense_stream_to_row_major_i8,
        unpack_dense_stream_to_row_major_u8, DenseParamPackError,
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
}
