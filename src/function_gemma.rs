use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use std::fmt;
use std::fs::File;
use std::path::Path;

#[derive(Debug)]
pub enum FunctionGemmaError {
    Io(std::io::Error),
    SafeTensors(safetensors::SafeTensorError),
    InvalidModel(String),
    MissingTensor(String),
    DtypeMismatch {
        name: String,
        expected: Dtype,
        actual: Dtype,
    },
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidArgument(String),
}

impl fmt::Display for FunctionGemmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionGemmaError::Io(err) => write!(f, "I/O error: {}", err),
            FunctionGemmaError::SafeTensors(err) => write!(f, "SafeTensors error: {}", err),
            FunctionGemmaError::InvalidModel(msg) => {
                write!(f, "Invalid Function Gemma model: {}", msg)
            }
            FunctionGemmaError::MissingTensor(name) => write!(f, "Missing tensor: {}", name),
            FunctionGemmaError::DtypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "Tensor dtype mismatch for {}: expected {:?}, got {:?}",
                name, expected, actual
            ),
            FunctionGemmaError::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "Tensor shape mismatch for {}: expected {:?}, got {:?}",
                name, expected, actual
            ),
            FunctionGemmaError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
        }
    }
}

impl std::error::Error for FunctionGemmaError {}

impl From<std::io::Error> for FunctionGemmaError {
    fn from(value: std::io::Error) -> Self {
        FunctionGemmaError::Io(value)
    }
}

impl From<safetensors::SafeTensorError> for FunctionGemmaError {
    fn from(value: safetensors::SafeTensorError) -> Self {
        FunctionGemmaError::SafeTensors(value)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FunctionGemmaLinearStage {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl FunctionGemmaLinearStage {
    pub const ALL: [Self; 7] = [
        Self::Q,
        Self::K,
        Self::V,
        Self::O,
        Self::Gate,
        Self::Up,
        Self::Down,
    ];

    pub fn short_name(self) -> &'static str {
        match self {
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::O => "o",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    pub fn parse(value: &str) -> Result<Self, FunctionGemmaError> {
        match value {
            "q" => Ok(Self::Q),
            "k" => Ok(Self::K),
            "v" => Ok(Self::V),
            "o" => Ok(Self::O),
            "gate" => Ok(Self::Gate),
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            _ => Err(FunctionGemmaError::InvalidArgument(format!(
                "unknown stage '{}': expected q|k|v|o|gate|up|down",
                value
            ))),
        }
    }
}

impl fmt::Display for FunctionGemmaLinearStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FunctionGemmaLayerLinearNames {
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

impl FunctionGemmaLayerLinearNames {
    pub fn for_layer(layer_idx: usize) -> Self {
        let base = format!("model.layers.{}", layer_idx);
        Self {
            q_proj: format!("{}.self_attn.q_proj.weight", base),
            k_proj: format!("{}.self_attn.k_proj.weight", base),
            v_proj: format!("{}.self_attn.v_proj.weight", base),
            o_proj: format!("{}.self_attn.o_proj.weight", base),
            gate_proj: format!("{}.mlp.gate_proj.weight", base),
            up_proj: format!("{}.mlp.up_proj.weight", base),
            down_proj: format!("{}.mlp.down_proj.weight", base),
        }
    }

    pub fn tensor_name_for_stage(&self, stage: FunctionGemmaLinearStage) -> &str {
        match stage {
            FunctionGemmaLinearStage::Q => &self.q_proj,
            FunctionGemmaLinearStage::K => &self.k_proj,
            FunctionGemmaLinearStage::V => &self.v_proj,
            FunctionGemmaLinearStage::O => &self.o_proj,
            FunctionGemmaLinearStage::Gate => &self.gate_proj,
            FunctionGemmaLinearStage::Up => &self.up_proj,
            FunctionGemmaLinearStage::Down => &self.down_proj,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct FunctionGemmaDims {
    pub hidden_size: usize,
    pub q_proj_out: usize,
    pub kv_proj_out: usize,
    pub mlp_hidden: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FunctionGemmaLinearStageMeta {
    pub stage: FunctionGemmaLinearStage,
    pub tensor_name: String,
    pub input_dim: usize,
    pub output_dim: usize,
}

enum FunctionGemmaStorage {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

pub struct FunctionGemmaSafeTensorFile {
    storage: FunctionGemmaStorage,
}

impl FunctionGemmaSafeTensorFile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FunctionGemmaError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // Prefer mmap to avoid copying whole checkpoints into process heap.
        let mapped = unsafe { MmapOptions::new().map(&file) };
        match mapped {
            Ok(mmap) => Ok(Self {
                storage: FunctionGemmaStorage::Mapped(mmap),
            }),
            Err(_) => {
                let bytes = std::fs::read(path)?;
                Ok(Self {
                    storage: FunctionGemmaStorage::Owned(bytes),
                })
            }
        }
    }

    pub fn storage_kind(&self) -> &'static str {
        match self.storage {
            FunctionGemmaStorage::Mapped(_) => "mmap",
            FunctionGemmaStorage::Owned(_) => "owned",
        }
    }

    fn bytes(&self) -> &[u8] {
        match &self.storage {
            FunctionGemmaStorage::Owned(bytes) => bytes.as_slice(),
            FunctionGemmaStorage::Mapped(mmap) => mmap.as_ref(),
        }
    }

    fn parsed(&self) -> Result<SafeTensors<'_>, FunctionGemmaError> {
        Ok(SafeTensors::deserialize(self.bytes())?)
    }

    pub fn tensor_names(&self) -> Result<Vec<String>, FunctionGemmaError> {
        let parsed = self.parsed()?;
        Ok(parsed.names().into_iter().map(str::to_string).collect())
    }

    fn tensor_from_parsed<'data>(
        parsed: &SafeTensors<'data>,
        name: &str,
    ) -> Result<safetensors::tensor::TensorView<'data>, FunctionGemmaError> {
        parsed
            .tensor(name)
            .map_err(|_| FunctionGemmaError::MissingTensor(name.to_string()))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, FunctionGemmaError> {
        let parsed = self.parsed()?;
        let tensor = Self::tensor_from_parsed(&parsed, name)?;
        let data = tensor.data();
        match tensor.dtype() {
            Dtype::F32 => {
                if data.len() % 4 != 0 {
                    return Err(FunctionGemmaError::InvalidModel(format!(
                        "f32 tensor {} has non-multiple-of-4 byte length {}",
                        name,
                        data.len()
                    )));
                }
                let mut out = Vec::with_capacity(data.len() / 4);
                for chunk in data.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(out)
            }
            Dtype::F16 => {
                if data.len() % 2 != 0 {
                    return Err(FunctionGemmaError::InvalidModel(format!(
                        "f16 tensor {} has non-multiple-of-2 byte length {}",
                        name,
                        data.len()
                    )));
                }
                let mut out = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f16::from_bits(bits).to_f32());
                }
                Ok(out)
            }
            Dtype::BF16 => {
                if data.len() % 2 != 0 {
                    return Err(FunctionGemmaError::InvalidModel(format!(
                        "bf16 tensor {} has non-multiple-of-2 byte length {}",
                        name,
                        data.len()
                    )));
                }
                let mut out = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(bf16::from_bits(bits).to_f32());
                }
                Ok(out)
            }
            other => Err(FunctionGemmaError::DtypeMismatch {
                name: name.to_string(),
                expected: Dtype::BF16,
                actual: other,
            }),
        }
    }

    pub fn infer_layer_dims(
        &self,
        layer_idx: usize,
    ) -> Result<FunctionGemmaDims, FunctionGemmaError> {
        let names = FunctionGemmaLayerLinearNames::for_layer(layer_idx);
        let parsed = self.parsed()?;
        let q = Self::tensor_from_parsed(&parsed, &names.q_proj)?;
        let k = Self::tensor_from_parsed(&parsed, &names.k_proj)?;
        let v = Self::tensor_from_parsed(&parsed, &names.v_proj)?;
        let o = Self::tensor_from_parsed(&parsed, &names.o_proj)?;
        let gate = Self::tensor_from_parsed(&parsed, &names.gate_proj)?;
        let up = Self::tensor_from_parsed(&parsed, &names.up_proj)?;
        let down = Self::tensor_from_parsed(&parsed, &names.down_proj)?;

        if q.shape().len() != 2 {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "expected {} to be rank-2, got {:?}",
                names.q_proj,
                q.shape()
            )));
        }
        let q_out = q.shape()[0];
        let hidden = q.shape()[1];
        let kv_out = expect_shape_2d(k.shape(), &names.k_proj, hidden)?;
        let v_out = expect_shape_2d(v.shape(), &names.v_proj, hidden)?;
        if kv_out != v_out {
            return Err(FunctionGemmaError::ShapeMismatch {
                name: names.v_proj,
                expected: vec![kv_out, hidden],
                actual: v.shape().to_vec(),
            });
        }

        expect_exact_shape(o.shape(), &[hidden, q_out], &names.o_proj)?;
        let mlp_hidden = expect_shape_2d(gate.shape(), &names.gate_proj, hidden)?;
        expect_exact_shape(up.shape(), &[mlp_hidden, hidden], &names.up_proj)?;
        expect_exact_shape(down.shape(), &[hidden, mlp_hidden], &names.down_proj)?;

        Ok(FunctionGemmaDims {
            hidden_size: hidden,
            q_proj_out: q_out,
            kv_proj_out: kv_out,
            mlp_hidden,
        })
    }

    pub fn layer_stage_metas(
        &self,
        layer_idx: usize,
    ) -> Result<Vec<FunctionGemmaLinearStageMeta>, FunctionGemmaError> {
        let names = FunctionGemmaLayerLinearNames::for_layer(layer_idx);
        let dims = self.infer_layer_dims(layer_idx)?;
        Ok(vec![
            FunctionGemmaLinearStageMeta {
                stage: FunctionGemmaLinearStage::Q,
                tensor_name: names.q_proj,
                input_dim: dims.hidden_size,
                output_dim: dims.q_proj_out,
            },
            FunctionGemmaLinearStageMeta {
                stage: FunctionGemmaLinearStage::K,
                tensor_name: names.k_proj,
                input_dim: dims.hidden_size,
                output_dim: dims.kv_proj_out,
            },
            FunctionGemmaLinearStageMeta {
                stage: FunctionGemmaLinearStage::V,
                tensor_name: names.v_proj,
                input_dim: dims.hidden_size,
                output_dim: dims.kv_proj_out,
            },
            FunctionGemmaLinearStageMeta {
                stage: FunctionGemmaLinearStage::O,
                tensor_name: names.o_proj,
                input_dim: dims.q_proj_out,
                output_dim: dims.hidden_size,
            },
            FunctionGemmaLinearStageMeta {
                stage: FunctionGemmaLinearStage::Gate,
                tensor_name: names.gate_proj,
                input_dim: dims.hidden_size,
                output_dim: dims.mlp_hidden,
            },
            FunctionGemmaLinearStageMeta {
                stage: FunctionGemmaLinearStage::Up,
                tensor_name: names.up_proj,
                input_dim: dims.hidden_size,
                output_dim: dims.mlp_hidden,
            },
            FunctionGemmaLinearStageMeta {
                stage: FunctionGemmaLinearStage::Down,
                tensor_name: names.down_proj,
                input_dim: dims.mlp_hidden,
                output_dim: dims.hidden_size,
            },
        ])
    }

    pub fn embedding_dims(&self) -> Result<(usize, usize), FunctionGemmaError> {
        let parsed = self.parsed()?;
        let tensor = Self::tensor_from_parsed(&parsed, "model.embed_tokens.weight")?;
        if tensor.shape().len() != 2 {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "expected model.embed_tokens.weight to be rank-2, got {:?}",
                tensor.shape()
            )));
        }
        Ok((tensor.shape()[0], tensor.shape()[1]))
    }

    pub fn token_embedding_row_f32(&self, token_id: usize) -> Result<Vec<f32>, FunctionGemmaError> {
        let parsed = self.parsed()?;
        let tensor = Self::tensor_from_parsed(&parsed, "model.embed_tokens.weight")?;
        if tensor.shape().len() != 2 {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "expected model.embed_tokens.weight to be rank-2, got {:?}",
                tensor.shape()
            )));
        }
        let vocab = tensor.shape()[0];
        let hidden = tensor.shape()[1];
        if token_id >= vocab {
            return Err(FunctionGemmaError::InvalidArgument(format!(
                "token id {} out of range [0, {})",
                token_id, vocab
            )));
        }
        let stride = hidden
            .checked_mul(dtype_elem_size(tensor.dtype())?)
            .ok_or_else(|| {
                FunctionGemmaError::InvalidModel("embedding stride overflow".to_string())
            })?;
        let start = token_id.checked_mul(stride).ok_or_else(|| {
            FunctionGemmaError::InvalidModel("embedding start overflow".to_string())
        })?;
        let end = start + stride;
        let data = tensor.data();
        if end > data.len() {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "embedding row slice out of bounds: {}..{} > {}",
                start,
                end,
                data.len()
            )));
        }
        decode_row_to_f32(&data[start..end], tensor.dtype(), hidden)
    }

    pub fn embedding_rows_f32(
        &self,
        token_start: usize,
        token_count: usize,
    ) -> Result<Vec<f32>, FunctionGemmaError> {
        if token_count == 0 {
            return Ok(Vec::new());
        }

        let parsed = self.parsed()?;
        let tensor = Self::tensor_from_parsed(&parsed, "model.embed_tokens.weight")?;
        if tensor.shape().len() != 2 {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "expected model.embed_tokens.weight to be rank-2, got {:?}",
                tensor.shape()
            )));
        }
        let vocab = tensor.shape()[0];
        let hidden = tensor.shape()[1];
        if token_start >= vocab {
            return Err(FunctionGemmaError::InvalidArgument(format!(
                "token_start {} out of range [0, {})",
                token_start, vocab
            )));
        }
        let token_end = token_start
            .checked_add(token_count)
            .ok_or_else(|| FunctionGemmaError::InvalidModel("token range overflow".to_string()))?;
        if token_end > vocab {
            return Err(FunctionGemmaError::InvalidArgument(format!(
                "token range {}..{} out of bounds for vocab {}",
                token_start, token_end, vocab
            )));
        }

        let stride = hidden
            .checked_mul(dtype_elem_size(tensor.dtype())?)
            .ok_or_else(|| {
                FunctionGemmaError::InvalidModel("embedding stride overflow".to_string())
            })?;
        let start = token_start.checked_mul(stride).ok_or_else(|| {
            FunctionGemmaError::InvalidModel("embedding start overflow".to_string())
        })?;
        let end = token_end.checked_mul(stride).ok_or_else(|| {
            FunctionGemmaError::InvalidModel("embedding end overflow".to_string())
        })?;
        let data = tensor.data();
        if end > data.len() || start > end {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "embedding block slice out of bounds: {}..{} > {}",
                start,
                end,
                data.len()
            )));
        }
        decode_slice_to_f32(&data[start..end], tensor.dtype(), token_count * hidden)
    }

    pub fn lm_head_topk_from_hidden(
        &self,
        hidden_state: &[f32],
        topk: usize,
    ) -> Result<Vec<(usize, f32)>, FunctionGemmaError> {
        if topk == 0 {
            return Err(FunctionGemmaError::InvalidArgument(
                "topk must be >= 1".to_string(),
            ));
        }

        let parsed = self.parsed()?;
        let tensor = Self::tensor_from_parsed(&parsed, "model.embed_tokens.weight")?;
        if tensor.shape().len() != 2 {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "expected model.embed_tokens.weight to be rank-2, got {:?}",
                tensor.shape()
            )));
        }
        let vocab = tensor.shape()[0];
        let hidden = tensor.shape()[1];
        if hidden_state.len() != hidden {
            return Err(FunctionGemmaError::InvalidArgument(format!(
                "hidden_state length mismatch: expected {}, got {}",
                hidden,
                hidden_state.len()
            )));
        }

        let elem_size = dtype_elem_size(tensor.dtype())?;
        let row_stride = hidden
            .checked_mul(elem_size)
            .ok_or_else(|| FunctionGemmaError::InvalidModel("row stride overflow".to_string()))?;
        let data = tensor.data();
        let expected_bytes = vocab.checked_mul(row_stride).ok_or_else(|| {
            FunctionGemmaError::InvalidModel("embedding size overflow".to_string())
        })?;
        if data.len() != expected_bytes {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "embedding byte length mismatch: expected {}, got {}",
                expected_bytes,
                data.len()
            )));
        }

        let mut best: Vec<(usize, f32)> = Vec::with_capacity(topk);
        for token_id in 0..vocab {
            let start = token_id * row_stride;
            let end = start + row_stride;
            let logit = dot_decoded_row(hidden_state, &data[start..end], tensor.dtype(), hidden)?;
            push_topk(&mut best, (token_id, logit), topk);
        }
        best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(best)
    }
}

fn expect_shape_2d(
    actual: &[usize],
    name: &str,
    expected_dim1: usize,
) -> Result<usize, FunctionGemmaError> {
    if actual.len() != 2 {
        return Err(FunctionGemmaError::InvalidModel(format!(
            "expected {} to be rank-2, got {:?}",
            name, actual
        )));
    }
    if actual[1] != expected_dim1 {
        return Err(FunctionGemmaError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![actual[0], expected_dim1],
            actual: actual.to_vec(),
        });
    }
    Ok(actual[0])
}

fn expect_exact_shape(
    actual: &[usize],
    expected: &[usize],
    name: &str,
) -> Result<(), FunctionGemmaError> {
    if actual != expected {
        return Err(FunctionGemmaError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

fn dtype_elem_size(dtype: Dtype) -> Result<usize, FunctionGemmaError> {
    match dtype {
        Dtype::F32 => Ok(4),
        Dtype::F16 | Dtype::BF16 => Ok(2),
        other => Err(FunctionGemmaError::InvalidModel(format!(
            "unsupported embedding dtype: {:?}",
            other
        ))),
    }
}

fn decode_row_to_f32(
    data: &[u8],
    dtype: Dtype,
    hidden: usize,
) -> Result<Vec<f32>, FunctionGemmaError> {
    decode_slice_to_f32(data, dtype, hidden)
}

fn decode_slice_to_f32(
    data: &[u8],
    dtype: Dtype,
    expected_values: usize,
) -> Result<Vec<f32>, FunctionGemmaError> {
    let mut out = Vec::with_capacity(expected_values);
    match dtype {
        Dtype::F32 => {
            if data.len() != expected_values * 4 {
                return Err(FunctionGemmaError::InvalidModel(format!(
                    "f32 row byte length mismatch: expected {}, got {}",
                    expected_values * 4,
                    data.len()
                )));
            }
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Dtype::F16 => {
            if data.len() != expected_values * 2 {
                return Err(FunctionGemmaError::InvalidModel(format!(
                    "f16 row byte length mismatch: expected {}, got {}",
                    expected_values * 2,
                    data.len()
                )));
            }
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16::from_bits(bits).to_f32());
            }
        }
        Dtype::BF16 => {
            if data.len() != expected_values * 2 {
                return Err(FunctionGemmaError::InvalidModel(format!(
                    "bf16 row byte length mismatch: expected {}, got {}",
                    expected_values * 2,
                    data.len()
                )));
            }
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16::from_bits(bits).to_f32());
            }
        }
        other => {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "unsupported row dtype: {:?}",
                other
            )));
        }
    }
    Ok(out)
}

fn dot_decoded_row(
    hidden_state: &[f32],
    row_bytes: &[u8],
    dtype: Dtype,
    hidden: usize,
) -> Result<f32, FunctionGemmaError> {
    let mut sum = 0.0f32;
    match dtype {
        Dtype::F32 => {
            if row_bytes.len() != hidden * 4 {
                return Err(FunctionGemmaError::InvalidModel(format!(
                    "f32 row byte length mismatch: expected {}, got {}",
                    hidden * 4,
                    row_bytes.len()
                )));
            }
            for (idx, chunk) in row_bytes.chunks_exact(4).enumerate() {
                let w = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                sum += hidden_state[idx] * w;
            }
        }
        Dtype::F16 => {
            if row_bytes.len() != hidden * 2 {
                return Err(FunctionGemmaError::InvalidModel(format!(
                    "f16 row byte length mismatch: expected {}, got {}",
                    hidden * 2,
                    row_bytes.len()
                )));
            }
            for (idx, chunk) in row_bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let w = f16::from_bits(bits).to_f32();
                sum += hidden_state[idx] * w;
            }
        }
        Dtype::BF16 => {
            if row_bytes.len() != hidden * 2 {
                return Err(FunctionGemmaError::InvalidModel(format!(
                    "bf16 row byte length mismatch: expected {}, got {}",
                    hidden * 2,
                    row_bytes.len()
                )));
            }
            for (idx, chunk) in row_bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let w = bf16::from_bits(bits).to_f32();
                sum += hidden_state[idx] * w;
            }
        }
        other => {
            return Err(FunctionGemmaError::InvalidModel(format!(
                "unsupported row dtype: {:?}",
                other
            )));
        }
    }
    Ok(sum)
}

fn push_topk(best: &mut Vec<(usize, f32)>, candidate: (usize, f32), topk: usize) {
    if best.len() < topk {
        best.push(candidate);
        return;
    }
    let mut worst_idx = 0usize;
    let mut worst_val = best[0].1;
    for (idx, (_, value)) in best.iter().enumerate().skip(1) {
        if *value < worst_val {
            worst_idx = idx;
            worst_val = *value;
        }
    }
    if candidate.1 > worst_val {
        best[worst_idx] = candidate;
    }
}
