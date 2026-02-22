use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::fmt;
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

pub struct FunctionGemmaSafeTensorFile {
    bytes: Vec<u8>,
}

impl FunctionGemmaSafeTensorFile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FunctionGemmaError> {
        let bytes = std::fs::read(path)?;
        Ok(Self { bytes })
    }

    fn parsed(&self) -> Result<SafeTensors<'_>, FunctionGemmaError> {
        Ok(SafeTensors::deserialize(&self.bytes)?)
    }

    fn tensor(
        &self,
        name: &str,
    ) -> Result<safetensors::tensor::TensorView<'_>, FunctionGemmaError> {
        let parsed = self.parsed()?;
        parsed
            .tensor(name)
            .map_err(|_| FunctionGemmaError::MissingTensor(name.to_string()))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, FunctionGemmaError> {
        let tensor = self.tensor(name)?;
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
        let q = self.tensor(&names.q_proj)?;
        let k = self.tensor(&names.k_proj)?;
        let v = self.tensor(&names.v_proj)?;
        let o = self.tensor(&names.o_proj)?;
        let gate = self.tensor(&names.gate_proj)?;
        let up = self.tensor(&names.up_proj)?;
        let down = self.tensor(&names.down_proj)?;

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
