use half::{bf16, f16};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};

pub fn tensor_from_parsed<'data, E>(
    parsed: &SafeTensors<'data>,
    name: &str,
    missing_tensor: impl FnOnce(String) -> E,
) -> Result<TensorView<'data>, E> {
    parsed
        .tensor(name)
        .map_err(|_| missing_tensor(name.to_string()))
}

pub fn ensure_dtype<E>(
    name: &str,
    actual: Dtype,
    expected: Dtype,
    dtype_mismatch: impl FnOnce(String, Dtype, Dtype) -> E,
) -> Result<(), E> {
    if actual != expected {
        return Err(dtype_mismatch(name.to_string(), expected, actual));
    }
    Ok(())
}

pub fn ensure_exact_shape<E>(
    actual: &[usize],
    expected: &[usize],
    name: &str,
    shape_mismatch: impl FnOnce(String, Vec<usize>, Vec<usize>) -> E,
) -> Result<(), E> {
    if actual != expected {
        return Err(shape_mismatch(
            name.to_string(),
            expected.to_vec(),
            actual.to_vec(),
        ));
    }
    Ok(())
}

pub fn ensure_rank<E>(
    actual: &[usize],
    expected_rank: usize,
    invalid_model: impl FnOnce(String) -> E,
    context: &str,
) -> Result<(), E> {
    if actual.len() != expected_rank {
        return Err(invalid_model(format!(
            "expected {} to be rank-{}, got {:?}",
            context, expected_rank, actual
        )));
    }
    Ok(())
}

pub fn dtype_elem_size(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::F32 => Some(4),
        Dtype::F16 | Dtype::BF16 => Some(2),
        _ => None,
    }
}

pub fn decode_to_f32<E>(
    data: &[u8],
    dtype: Dtype,
    expected_values: usize,
    invalid_model: impl Fn(String) -> E,
    context: &str,
) -> Result<Vec<f32>, E> {
    let mut out = Vec::with_capacity(expected_values);
    match dtype {
        Dtype::F32 => {
            ensure_byte_len(data, expected_values * 4, "f32", invalid_model, context)?;
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Dtype::F16 => {
            ensure_byte_len(data, expected_values * 2, "f16", invalid_model, context)?;
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16::from_bits(bits).to_f32());
            }
        }
        Dtype::BF16 => {
            ensure_byte_len(data, expected_values * 2, "bf16", invalid_model, context)?;
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16::from_bits(bits).to_f32());
            }
        }
        other => {
            return Err(invalid_model(format!(
                "unsupported {} dtype: {:?}",
                context, other
            )));
        }
    }
    Ok(out)
}

pub fn invalid_multiple_of_error(dtype: &str, name: &str, len: usize, multiple: usize) -> String {
    format!(
        "{} tensor {} has non-multiple-of-{} byte length {}",
        dtype, name, multiple, len
    )
}

pub fn row_byte_len_mismatch(context: &str, dtype: &str, expected: usize, actual: usize) -> String {
    format!(
        "{} {} byte length mismatch: expected {}, got {}",
        dtype, context, expected, actual
    )
}

fn ensure_byte_len<E>(
    data: &[u8],
    expected: usize,
    dtype_label: &str,
    invalid_model: impl Fn(String) -> E,
    context: &str,
) -> Result<(), E> {
    if data.len() != expected {
        return Err(invalid_model(row_byte_len_mismatch(
            context,
            dtype_label,
            expected,
            data.len(),
        )));
    }
    Ok(())
}
