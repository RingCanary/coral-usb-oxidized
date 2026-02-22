use crate::delegate::EdgeTPUDelegate;
use crate::device::CoralDevice;
use crate::error::DenseGemmError;
use crate::flatbuffer::{inspect_packages, select_dense_parameter_region, Region};
use crate::interpreter::CoralInterpreter;

pub const DENSE_GEMM256_DIM: usize = 256;
pub const DENSE_GEMM256_WEIGHT_COUNT: usize = DENSE_GEMM256_DIM * DENSE_GEMM256_DIM;
pub const DENSE_GEMM256_WEIGHT_BYTES: usize = DENSE_GEMM256_WEIGHT_COUNT;
pub const DENSE_GEMM256_ZERO_POINT: i16 = 128;

pub const TEMPLATE_2048: &[u8] =
    include_bytes!("../templates/dense_2048x2048_quant_edgetpu.tflite");
pub const TEMPLATE_2304: &[u8] =
    include_bytes!("../templates/dense_2304x2304_quant_edgetpu.tflite");
pub const TEMPLATE_2688: &[u8] =
    include_bytes!("../templates/dense_2688x2688_quant_edgetpu.tflite");

fn invalid_template(message: impl Into<String>) -> DenseGemmError {
    DenseGemmError::InvalidTemplate(message.into())
}

fn encode_row_major_weights_into_payload(
    input_dim: usize,
    output_dim: usize,
    weights_row_major_q: &[i8],
    payload: &mut [u8],
) {
    let row_tile_count = input_dim / 64;
    let col_tile_count = output_dim / 64;

    // Re-stride row-major weights into recovered 64x64 tile / 4-lane payload layout.
    for col_tile in 0..col_tile_count {
        let col_base = col_tile * 64;
        for row_tile in 0..row_tile_count {
            let row_base = row_tile * 64;
            let tile_base = (col_tile * row_tile_count + row_tile) * 4096;

            for row_group in 0..16 {
                let row_group_base = row_base + row_group * 4;
                let group_base = tile_base + row_group * 256;

                for col_local in 0..64 {
                    let src_col = col_base + col_local;
                    let lane_base = group_base + col_local * 4;

                    for lane in 0..4 {
                        let src_row = row_group_base + lane;
                        let source_index = src_row * output_dim + src_col;
                        payload[lane_base + lane] = DenseGemmTemplate::payload_byte_from_qi8(
                            weights_row_major_q[source_index],
                        );
                    }
                }
            }
        }
    }
}

pub fn dense_param_offset(
    input_dim: usize,
    output_dim: usize,
    row: usize,
    col: usize,
) -> Result<usize, DenseGemmError> {
    if input_dim == 0 || output_dim == 0 {
        return Err(DenseGemmError::UnsupportedDimensions {
            input_dim,
            output_dim,
            reason: "dimensions must be non-zero",
        });
    }
    if input_dim % 64 != 0 || output_dim % 64 != 0 {
        return Err(DenseGemmError::UnsupportedDimensions {
            input_dim,
            output_dim,
            reason: "expected multiples of 64 for recovered layout mapping",
        });
    }
    if input_dim % 4 != 0 {
        return Err(DenseGemmError::UnsupportedDimensions {
            input_dim,
            output_dim,
            reason: "input dimension must be a multiple of 4",
        });
    }
    if row >= input_dim || col >= output_dim {
        return Err(invalid_template(format!(
            "row/col out of range: row={} col={} (expected row < {}, col < {})",
            row, col, input_dim, output_dim
        )));
    }

    let row_tile_count = input_dim / 64;
    let base = (col / 64) * row_tile_count * 4096 + (row / 64) * 4096;
    let inner = ((row % 64) / 4) * 256 + (col % 64) * 4 + (row % 4);
    Ok(base + inner)
}

pub fn dense_256_param_offset(row: usize, col: usize) -> Result<usize, DenseGemmError> {
    dense_param_offset(DENSE_GEMM256_DIM, DENSE_GEMM256_DIM, row, col)
}

#[derive(Clone)]
pub struct DenseGemmTemplate {
    model_bytes: Vec<u8>,
    parameter_region: Region,
    input_dim: usize,
    output_dim: usize,
}

impl DenseGemmTemplate {
    pub fn from_file_with_dims(
        model_path: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DenseGemmError> {
        let model_bytes = std::fs::read(model_path)?;
        Self::from_bytes_with_dims(&model_bytes, input_dim, output_dim)
    }

    pub fn from_bytes_with_dims(
        model_bytes: &[u8],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DenseGemmError> {
        let _ = dense_param_offset(input_dim, output_dim, 0, 0)?;

        let packages = inspect_packages(model_bytes);
        if packages.is_empty() {
            return Err(invalid_template("no valid DWN1 package found"));
        }

        let parameter_region = select_dense_parameter_region(&packages)?;
        let expected_size = input_dim
            .checked_mul(output_dim)
            .ok_or_else(|| invalid_template("dimension multiplication overflow"))?;
        let actual_size = parameter_region.size();
        if actual_size != expected_size {
            return Err(DenseGemmError::InvalidParameterRegionSize {
                expected: expected_size,
                actual: actual_size,
            });
        }

        Ok(Self {
            model_bytes: model_bytes.to_vec(),
            parameter_region,
            input_dim,
            output_dim,
        })
    }

    pub fn from_bundled_2048() -> Result<Self, DenseGemmError> {
        Self::from_bytes_with_dims(TEMPLATE_2048, 2048, 2048)
    }

    pub fn from_bundled_2304() -> Result<Self, DenseGemmError> {
        Self::from_bytes_with_dims(TEMPLATE_2304, 2304, 2304)
    }

    pub fn from_bundled_2688() -> Result<Self, DenseGemmError> {
        Self::from_bytes_with_dims(TEMPLATE_2688, 2688, 2688)
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }

    pub fn model_bytes(&self) -> &[u8] {
        &self.model_bytes
    }

    pub fn parameter_region(&self) -> (usize, usize) {
        (self.parameter_region.start, self.parameter_region.end)
    }

    pub fn payload_bytes(&self) -> &[u8] {
        &self.model_bytes[self.parameter_region.start..self.parameter_region.end]
    }

    pub fn payload_byte_from_qi8(weight_q: i8) -> u8 {
        ((weight_q as i16) + DENSE_GEMM256_ZERO_POINT) as u8
    }

    pub fn qi8_from_payload_byte(payload: u8) -> i8 {
        (payload as i16 - DENSE_GEMM256_ZERO_POINT) as i8
    }

    fn payload_bytes_mut(&mut self) -> Result<&mut [u8], DenseGemmError> {
        let end = self.parameter_region.end;
        if end > self.model_bytes.len() || self.parameter_region.start >= end {
            return Err(invalid_template("parameter payload region out of bounds"));
        }
        Ok(&mut self.model_bytes[self.parameter_region.start..end])
    }

    pub fn fill_matrix_qi8(&mut self, value_q: i8) -> Result<(), DenseGemmError> {
        let encoded = Self::payload_byte_from_qi8(value_q);
        self.payload_bytes_mut()?.fill(encoded);
        Ok(())
    }

    pub fn set_weight_qi8(
        &mut self,
        row: usize,
        col: usize,
        weight_q: i8,
    ) -> Result<(), DenseGemmError> {
        let offset = dense_param_offset(self.input_dim, self.output_dim, row, col)?;
        let payload = self.payload_bytes_mut()?;
        payload[offset] = Self::payload_byte_from_qi8(weight_q);
        Ok(())
    }

    pub fn set_weights_from_slice(
        &mut self,
        weights_row_major_q: &[i8],
    ) -> Result<(), DenseGemmError> {
        let expected_size = self
            .input_dim
            .checked_mul(self.output_dim)
            .ok_or_else(|| invalid_template("dimension multiplication overflow"))?;
        if weights_row_major_q.len() != expected_size {
            return Err(DenseGemmError::WeightSizeMismatch {
                expected: expected_size,
                actual: weights_row_major_q.len(),
            });
        }

        self.fill_matrix_qi8(0)?;
        let input_dim = self.input_dim;
        let output_dim = self.output_dim;
        let payload = self.payload_bytes_mut()?;
        encode_row_major_weights_into_payload(input_dim, output_dim, weights_row_major_q, payload);
        Ok(())
    }

    pub fn set_identity(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        if self.input_dim != self.output_dim {
            return Err(DenseGemmError::UnsupportedDimensions {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                reason: "identity requires square matrix",
            });
        }

        self.fill_matrix_qi8(0)?;
        for i in 0..self.input_dim {
            self.set_weight_qi8(i, i, active_q)?;
        }
        Ok(())
    }

    pub fn set_diagonal(&mut self, diagonal_q: &[i8]) -> Result<(), DenseGemmError> {
        if self.input_dim != self.output_dim {
            return Err(DenseGemmError::UnsupportedDimensions {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                reason: "diagonal mode requires square matrix",
            });
        }

        if diagonal_q.len() != self.input_dim {
            return Err(DenseGemmError::WeightSizeMismatch {
                expected: self.input_dim,
                actual: diagonal_q.len(),
            });
        }

        self.fill_matrix_qi8(0)?;
        for (idx, value_q) in diagonal_q.iter().enumerate() {
            self.set_weight_qi8(idx, idx, *value_q)?;
        }

        Ok(())
    }

    pub fn set_shift_plus1(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        if self.input_dim != self.output_dim {
            return Err(DenseGemmError::UnsupportedDimensions {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                reason: "shift modes require square matrix",
            });
        }

        self.fill_matrix_qi8(0)?;
        for col in 0..self.output_dim {
            let row = (col + 1) % self.input_dim;
            self.set_weight_qi8(row, col, active_q)?;
        }
        Ok(())
    }

    pub fn set_shift_minus1(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        if self.input_dim != self.output_dim {
            return Err(DenseGemmError::UnsupportedDimensions {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                reason: "shift modes require square matrix",
            });
        }

        self.fill_matrix_qi8(0)?;
        for col in 0..self.output_dim {
            let row = (col + self.input_dim - 1) % self.input_dim;
            self.set_weight_qi8(row, col, active_q)?;
        }
        Ok(())
    }

    pub fn prepare(&self, delegate: &EdgeTPUDelegate) -> Result<PreparedDenseGemm, DenseGemmError> {
        PreparedDenseGemm::new(
            self.model_bytes(),
            delegate,
            self.input_dim,
            self.output_dim,
        )
    }

    pub fn prepare_with_new_delegate(&self) -> Result<PreparedDenseGemm, DenseGemmError> {
        let device = CoralDevice::new()?;
        let delegate = device.create_delegate()?;
        self.prepare(&delegate)
    }

    pub fn execute(
        &self,
        delegate: &EdgeTPUDelegate,
        input: &[i8],
    ) -> Result<Vec<i8>, DenseGemmError> {
        let prepared = self.prepare(delegate)?;
        prepared.execute(input)
    }
}

pub struct PreparedDenseGemm {
    interpreter: CoralInterpreter,
    input_dim: usize,
    output_dim: usize,
    max_batch_rows: usize,
}

impl PreparedDenseGemm {
    fn new(
        model_bytes: &[u8],
        delegate: &EdgeTPUDelegate,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DenseGemmError> {
        let interpreter = CoralInterpreter::new_from_memory(model_bytes, delegate)?;
        let input_size = interpreter.input_tensor_byte_size(0)?;
        if input_size < input_dim || input_size % input_dim != 0 {
            return Err(DenseGemmError::InputSizeMismatch {
                expected: input_dim,
                actual: input_size,
            });
        }
        let max_batch_rows = input_size / input_dim;

        let output_size = interpreter.output_tensor_byte_size(0)?;
        let expected_output_size = output_dim
            .checked_mul(max_batch_rows)
            .ok_or_else(|| invalid_template("output byte size overflow"))?;
        if output_size != expected_output_size {
            return Err(DenseGemmError::OutputSizeMismatch {
                expected: expected_output_size,
                actual: output_size,
            });
        }
        Ok(Self {
            interpreter,
            input_dim,
            output_dim,
            max_batch_rows,
        })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }

    pub fn max_batch_rows(&self) -> usize {
        self.max_batch_rows
    }

    pub fn execute(&self, input: &[i8]) -> Result<Vec<i8>, DenseGemmError> {
        let outputs = self.execute_batch_rows(input)?;
        if outputs.len() != self.output_dim {
            return Err(DenseGemmError::OutputSizeMismatch {
                expected: self.output_dim,
                actual: outputs.len(),
            });
        }
        Ok(outputs)
    }

    pub fn execute_batch_rows(&self, inputs_row_major_q: &[i8]) -> Result<Vec<i8>, DenseGemmError> {
        if inputs_row_major_q.len() % self.input_dim != 0 {
            return Err(DenseGemmError::BatchInputSizeMismatch {
                input_dim: self.input_dim,
                actual: inputs_row_major_q.len(),
            });
        }

        let batch = inputs_row_major_q.len() / self.input_dim;
        let mut outputs = Vec::with_capacity(batch * self.output_dim);
        if batch == 0 {
            return Ok(outputs);
        }

        let rows_per_invoke = self.max_batch_rows.max(1);
        let input_bytes_per_invoke = rows_per_invoke
            .checked_mul(self.input_dim)
            .ok_or_else(|| invalid_template("input invoke byte size overflow"))?;
        let output_bytes_per_invoke = rows_per_invoke
            .checked_mul(self.output_dim)
            .ok_or_else(|| invalid_template("output invoke byte size overflow"))?;
        let mut input_bytes = vec![0u8; input_bytes_per_invoke];
        let mut output_bytes = vec![0u8; output_bytes_per_invoke];

        for chunk in inputs_row_major_q.chunks(rows_per_invoke * self.input_dim) {
            let rows_in_chunk = chunk.len() / self.input_dim;
            input_bytes.fill(0);
            for (idx, value) in chunk.iter().enumerate() {
                input_bytes[idx] = *value as u8;
            }

            self.interpreter.copy_to_input_tensor(0, &input_bytes)?;
            self.interpreter.run()?;

            self.interpreter
                .copy_from_output_tensor(0, &mut output_bytes)?;

            let used = rows_in_chunk
                .checked_mul(self.output_dim)
                .ok_or_else(|| invalid_template("output slice size overflow"))?;
            outputs.extend(output_bytes.iter().take(used).map(|value| *value as i8));
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_offset_formula_matches_known_points() {
        assert_eq!(dense_256_param_offset(0, 0).unwrap(), 0);
        assert_eq!(dense_256_param_offset(1, 0).unwrap(), 1);
        assert_eq!(dense_256_param_offset(0, 1).unwrap(), 4);
        assert_eq!(dense_256_param_offset(4, 0).unwrap(), 256);
        assert_eq!(dense_256_param_offset(64, 0).unwrap(), 4096);
        assert_eq!(dense_256_param_offset(0, 64).unwrap(), 16384);
        assert_eq!(dense_256_param_offset(255, 255).unwrap(), 65535);
    }

    #[test]
    fn dense_offset_formula_generalizes_with_dimension() {
        assert_eq!(dense_param_offset(512, 512, 0, 64).unwrap(), 32768);
        assert_eq!(dense_param_offset(512, 512, 511, 511).unwrap(), 262143);
        assert_eq!(dense_param_offset(1024, 1024, 0, 64).unwrap(), 65536);
        assert_eq!(dense_param_offset(1024, 1024, 1023, 1023).unwrap(), 1048575);
    }

    #[test]
    fn dense_offset_rejects_unsupported_dimensions() {
        assert!(matches!(
            dense_param_offset(250, 256, 0, 0),
            Err(DenseGemmError::UnsupportedDimensions { .. })
        ));
    }

    #[test]
    fn dense_payload_encoding_round_trips() {
        let values = [-128i8, -127, -1, 0, 1, 63, 127];
        for value in values {
            let encoded = DenseGemmTemplate::payload_byte_from_qi8(value);
            let decoded = DenseGemmTemplate::qi8_from_payload_byte(encoded);
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn fast_restride_matches_formula_mapping() {
        let input_dim = 128usize;
        let output_dim = 128usize;
        let count = input_dim * output_dim;
        let mut row_major = vec![0i8; count];
        for (idx, value) in row_major.iter_mut().enumerate() {
            *value = ((idx % 251) as i16 - 125) as i8;
        }

        let mut payload = vec![DenseGemmTemplate::payload_byte_from_qi8(0); count];
        encode_row_major_weights_into_payload(input_dim, output_dim, &row_major, &mut payload);

        for row in 0..input_dim {
            for col in 0..output_dim {
                let off = dense_param_offset(input_dim, output_dim, row, col).unwrap();
                let expected =
                    DenseGemmTemplate::payload_byte_from_qi8(row_major[row * output_dim + col]);
                assert_eq!(payload[off], expected);
            }
        }
    }
}
