use safetensors::{Dtype, SafeTensors};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum ClipError {
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

impl fmt::Display for ClipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClipError::Io(err) => write!(f, "I/O error: {}", err),
            ClipError::SafeTensors(err) => write!(f, "SafeTensors error: {}", err),
            ClipError::InvalidModel(msg) => write!(f, "Invalid CLIP model: {}", msg),
            ClipError::MissingTensor(name) => write!(f, "Missing tensor: {}", name),
            ClipError::DtypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "Tensor dtype mismatch for {}: expected {:?}, got {:?}",
                name, expected, actual
            ),
            ClipError::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "Tensor shape mismatch for {}: expected {:?}, got {:?}",
                name, expected, actual
            ),
            ClipError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
        }
    }
}

impl std::error::Error for ClipError {}

impl From<std::io::Error> for ClipError {
    fn from(value: std::io::Error) -> Self {
        ClipError::Io(value)
    }
}

impl From<safetensors::SafeTensorError> for ClipError {
    fn from(value: safetensors::SafeTensorError) -> Self {
        ClipError::SafeTensors(value)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ClipTensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ClipVitLayerLinearNames {
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub mlp_fc1: String,
    pub mlp_fc2: String,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ClipVitLinearStage {
    Q,
    K,
    V,
    O,
    Fc1,
    Fc2,
}

impl ClipVitLinearStage {
    pub const ALL: [Self; 6] = [Self::Q, Self::K, Self::V, Self::O, Self::Fc1, Self::Fc2];

    pub fn short_name(self) -> &'static str {
        match self {
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::O => "o",
            Self::Fc1 => "fc1",
            Self::Fc2 => "fc2",
        }
    }

    pub fn io_dims(self, dims: ClipVitB32Dims) -> (usize, usize) {
        match self {
            Self::Q | Self::K | Self::V | Self::O => (dims.d_model, dims.d_model),
            Self::Fc1 => (dims.d_model, dims.mlp_hidden),
            Self::Fc2 => (dims.mlp_hidden, dims.d_model),
        }
    }
}

impl fmt::Display for ClipVitLinearStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ClipVitLinearStageMeta {
    pub stage: ClipVitLinearStage,
    pub tensor_name: String,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl ClipVitLinearStageMeta {
    fn new(stage: ClipVitLinearStage, tensor_name: String, dims: ClipVitB32Dims) -> Self {
        let (input_dim, output_dim) = stage.io_dims(dims);
        Self {
            stage,
            tensor_name,
            input_dim,
            output_dim,
        }
    }
}

impl ClipVitLayerLinearNames {
    pub fn for_layer(layer_idx: usize) -> Self {
        let base = format!("vision_model.encoder.layers.{}", layer_idx);
        Self {
            q_proj: format!("{}.self_attn.q_proj.weight", base),
            k_proj: format!("{}.self_attn.k_proj.weight", base),
            v_proj: format!("{}.self_attn.v_proj.weight", base),
            o_proj: format!("{}.self_attn.out_proj.weight", base),
            mlp_fc1: format!("{}.mlp.fc1.weight", base),
            mlp_fc2: format!("{}.mlp.fc2.weight", base),
        }
    }

    pub fn all(&self) -> [&str; 6] {
        [
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &self.mlp_fc1,
            &self.mlp_fc2,
        ]
    }

    pub fn tensor_name_for_stage(&self, stage: ClipVitLinearStage) -> &str {
        match stage {
            ClipVitLinearStage::Q => &self.q_proj,
            ClipVitLinearStage::K => &self.k_proj,
            ClipVitLinearStage::V => &self.v_proj,
            ClipVitLinearStage::O => &self.o_proj,
            ClipVitLinearStage::Fc1 => &self.mlp_fc1,
            ClipVitLinearStage::Fc2 => &self.mlp_fc2,
        }
    }

    pub fn stage_metas(&self, dims: ClipVitB32Dims) -> [ClipVitLinearStageMeta; 6] {
        [
            ClipVitLinearStageMeta::new(ClipVitLinearStage::Q, self.q_proj.clone(), dims),
            ClipVitLinearStageMeta::new(ClipVitLinearStage::K, self.k_proj.clone(), dims),
            ClipVitLinearStageMeta::new(ClipVitLinearStage::V, self.v_proj.clone(), dims),
            ClipVitLinearStageMeta::new(ClipVitLinearStage::O, self.o_proj.clone(), dims),
            ClipVitLinearStageMeta::new(ClipVitLinearStage::Fc1, self.mlp_fc1.clone(), dims),
            ClipVitLinearStageMeta::new(ClipVitLinearStage::Fc2, self.mlp_fc2.clone(), dims),
        ]
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ClipVitB32Dims {
    pub d_model: usize,
    pub mlp_hidden: usize,
}

impl Default for ClipVitB32Dims {
    fn default() -> Self {
        Self {
            d_model: 768,
            mlp_hidden: 3072,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizationInfo {
    pub max_abs: f32,
    pub clipped_max_abs: f32,
    pub scale: f32,
    pub qmax: i32,
    pub clip_percentile: f32,
    pub clipped_values: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearQuantConfig {
    pub qmax: i32,
    pub clip_percentile: f32,
}

impl Default for LinearQuantConfig {
    fn default() -> Self {
        Self {
            qmax: 127,
            clip_percentile: 100.0,
        }
    }
}

pub struct ClipSafeTensorFile {
    bytes: Vec<u8>,
}

impl ClipSafeTensorFile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ClipError> {
        let bytes = std::fs::read(path)?;
        Ok(Self { bytes })
    }

    fn parsed(&self) -> Result<SafeTensors<'_>, ClipError> {
        Ok(SafeTensors::deserialize(&self.bytes)?)
    }

    pub fn tensor_count(&self) -> Result<usize, ClipError> {
        Ok(self.parsed()?.names().len())
    }

    pub fn tensor_names(&self) -> Result<Vec<String>, ClipError> {
        let parsed = self.parsed()?;
        Ok(parsed.names().into_iter().map(str::to_owned).collect())
    }

    pub fn tensor_info(&self, name: &str) -> Result<ClipTensorInfo, ClipError> {
        let parsed = self.parsed()?;
        let tensor = parsed
            .tensor(name)
            .map_err(|_| ClipError::MissingTensor(name.to_string()))?;
        Ok(ClipTensorInfo {
            name: name.to_string(),
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            bytes: tensor.data().len(),
        })
    }

    pub fn tensor_infos(&self) -> Result<Vec<ClipTensorInfo>, ClipError> {
        let parsed = self.parsed()?;
        let mut infos = Vec::with_capacity(parsed.names().len());
        for name in parsed.names() {
            let tensor = parsed
                .tensor(name)
                .map_err(|_| ClipError::MissingTensor(name.to_string()))?;
            infos.push(ClipTensorInfo {
                name: name.to_string(),
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                bytes: tensor.data().len(),
            });
        }
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, ClipError> {
        let parsed = self.parsed()?;
        let tensor = parsed
            .tensor(name)
            .map_err(|_| ClipError::MissingTensor(name.to_string()))?;
        if tensor.dtype() != Dtype::F32 {
            return Err(ClipError::DtypeMismatch {
                name: name.to_string(),
                expected: Dtype::F32,
                actual: tensor.dtype(),
            });
        }

        let data = tensor.data();
        if data.len() % 4 != 0 {
            return Err(ClipError::InvalidModel(format!(
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

    pub fn discover_clip_vit_encoder_layers(&self) -> Result<Vec<usize>, ClipError> {
        let names = self.tensor_names()?;
        let mut layer_indices = BTreeSet::new();
        let suffix = ".self_attn.q_proj.weight";

        for name in names {
            if let Some(rest) = name.strip_prefix("vision_model.encoder.layers.") {
                if let Some(stripped) = rest.strip_suffix(suffix) {
                    if !stripped.is_empty() && stripped.chars().all(|ch| ch.is_ascii_digit()) {
                        let idx = stripped.parse::<usize>().map_err(|err| {
                            ClipError::InvalidModel(format!(
                                "invalid layer index in {}: {}",
                                name, err
                            ))
                        })?;
                        layer_indices.insert(idx);
                    }
                }
            }
        }

        Ok(layer_indices.into_iter().collect())
    }

    pub fn validate_clip_vit_layer_linears(
        &self,
        layer_idx: usize,
        dims: ClipVitB32Dims,
    ) -> Result<ClipVitLayerLinearNames, ClipError> {
        let names = ClipVitLayerLinearNames::for_layer(layer_idx);
        let expected = [
            (&names.q_proj, vec![dims.d_model, dims.d_model]),
            (&names.k_proj, vec![dims.d_model, dims.d_model]),
            (&names.v_proj, vec![dims.d_model, dims.d_model]),
            (&names.o_proj, vec![dims.d_model, dims.d_model]),
            (&names.mlp_fc1, vec![dims.mlp_hidden, dims.d_model]),
            (&names.mlp_fc2, vec![dims.d_model, dims.mlp_hidden]),
        ];

        for (name, expected_shape) in expected {
            let info = self.tensor_info(name)?;
            if info.dtype != Dtype::F32 {
                return Err(ClipError::DtypeMismatch {
                    name: info.name,
                    expected: Dtype::F32,
                    actual: info.dtype,
                });
            }
            if info.shape != expected_shape {
                return Err(ClipError::ShapeMismatch {
                    name: info.name,
                    expected: expected_shape,
                    actual: info.shape,
                });
            }
        }

        Ok(names)
    }

    pub fn clip_vit_layer_stage_metas(
        &self,
        layer_idx: usize,
        dims: ClipVitB32Dims,
    ) -> Result<[ClipVitLinearStageMeta; 6], ClipError> {
        let names = self.validate_clip_vit_layer_linears(layer_idx, dims)?;
        Ok(names.stage_metas(dims))
    }
}

pub fn quantize_linear_out_in_to_row_major_qi8(
    weights_out_by_in_f32: &[f32],
    input_dim: usize,
    output_dim: usize,
    qmax: i32,
) -> Result<(Vec<i8>, QuantizationInfo), ClipError> {
    quantize_linear_out_in_to_row_major_qi8_with_config(
        weights_out_by_in_f32,
        input_dim,
        output_dim,
        LinearQuantConfig {
            qmax,
            clip_percentile: 100.0,
        },
    )
}

fn absolute_percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut abs_values: Vec<f32> = values.iter().map(|value| value.abs()).collect();
    let max_index = abs_values.len() - 1;
    let rank = (((percentile / 100.0) * max_index as f32).round() as usize).min(max_index);
    let (_, nth, _) =
        abs_values.select_nth_unstable_by(rank, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    *nth
}

pub fn quantize_linear_out_in_to_row_major_qi8_with_config(
    weights_out_by_in_f32: &[f32],
    input_dim: usize,
    output_dim: usize,
    config: LinearQuantConfig,
) -> Result<(Vec<i8>, QuantizationInfo), ClipError> {
    if input_dim == 0 || output_dim == 0 {
        return Err(ClipError::InvalidArgument(
            "input_dim and output_dim must be non-zero".to_string(),
        ));
    }
    if !(1..=127).contains(&config.qmax) {
        return Err(ClipError::InvalidArgument(format!(
            "qmax must be in [1, 127], got {}",
            config.qmax
        )));
    }
    if !(0.0..=100.0).contains(&config.clip_percentile) || config.clip_percentile == 0.0 {
        return Err(ClipError::InvalidArgument(format!(
            "clip_percentile must be in (0, 100], got {}",
            config.clip_percentile
        )));
    }

    let expected = input_dim.checked_mul(output_dim).ok_or_else(|| {
        ClipError::InvalidArgument("dimension multiplication overflow".to_string())
    })?;
    if weights_out_by_in_f32.len() != expected {
        return Err(ClipError::InvalidArgument(format!(
            "weight length mismatch: expected {}, got {}",
            expected,
            weights_out_by_in_f32.len()
        )));
    }

    let mut max_abs = 0.0f32;
    for &value in weights_out_by_in_f32 {
        if !value.is_finite() {
            return Err(ClipError::InvalidModel(
                "weight tensor contains non-finite values".to_string(),
            ));
        }
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    let clipped_max_abs = if config.clip_percentile >= 100.0 {
        max_abs
    } else {
        absolute_percentile(weights_out_by_in_f32, config.clip_percentile)
    };

    let scale = if clipped_max_abs > 0.0 {
        clipped_max_abs / config.qmax as f32
    } else {
        1.0
    };

    let mut out_row_major = vec![0i8; expected];
    let mut clipped_values = 0usize;
    for out_idx in 0..output_dim {
        for in_idx in 0..input_dim {
            let source = weights_out_by_in_f32[out_idx * input_dim + in_idx];
            let q = (source / scale).round() as i32;
            if q < -config.qmax || q > config.qmax {
                clipped_values += 1;
            }
            let clamped = q.clamp(-config.qmax, config.qmax) as i8;
            out_row_major[in_idx * output_dim + out_idx] = clamped;
        }
    }

    Ok((
        out_row_major,
        QuantizationInfo {
            max_abs,
            clipped_max_abs,
            scale,
            qmax: config.qmax,
            clip_percentile: config.clip_percentile,
            clipped_values,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_name_builder_is_stable() {
        let names = ClipVitLayerLinearNames::for_layer(3);
        assert_eq!(
            names.q_proj,
            "vision_model.encoder.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            names.mlp_fc2,
            "vision_model.encoder.layers.3.mlp.fc2.weight"
        );
    }

    #[test]
    fn stage_enum_order_and_labels_are_stable() {
        let labels = ClipVitLinearStage::ALL
            .iter()
            .map(|stage| stage.short_name())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["q", "k", "v", "o", "fc1", "fc2"]);
    }

    #[test]
    fn layer_stage_metas_match_expected_dims() {
        let dims = ClipVitB32Dims::default();
        let names = ClipVitLayerLinearNames::for_layer(2);
        let metas = names.stage_metas(dims);

        assert_eq!(metas[0].stage, ClipVitLinearStage::Q);
        assert_eq!(metas[0].tensor_name, names.q_proj.as_str());
        assert_eq!((metas[0].input_dim, metas[0].output_dim), (768, 768));

        assert_eq!(metas[4].stage, ClipVitLinearStage::Fc1);
        assert_eq!(metas[4].tensor_name, names.mlp_fc1.as_str());
        assert_eq!((metas[4].input_dim, metas[4].output_dim), (768, 3072));

        assert_eq!(metas[5].stage, ClipVitLinearStage::Fc2);
        assert_eq!(metas[5].tensor_name, names.mlp_fc2.as_str());
        assert_eq!((metas[5].input_dim, metas[5].output_dim), (3072, 768));
    }

    #[test]
    fn quantizer_transposes_out_by_in_to_row_major() {
        // out_dim=2, in_dim=3
        // [ [1,2,3],
        //   [4,5,6] ]
        let source = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (row_major, info) =
            quantize_linear_out_in_to_row_major_qi8(&source, 3, 2, 127).unwrap();
        assert_eq!(row_major.len(), 6);
        assert!(info.scale > 0.0);

        // row_major shape: in_dim x out_dim
        // [ [1,4],
        //   [2,5],
        //   [3,6] ] (up to scale+rounding)
        assert!(row_major[0] <= row_major[1]);
        assert!(row_major[2] <= row_major[3]);
        assert!(row_major[4] <= row_major[5]);
    }

    #[test]
    fn quantizer_respects_percentile_clipping() {
        let source = vec![0.0f32, 1.0, 2.0, 100.0];
        let (_row_major, info) = quantize_linear_out_in_to_row_major_qi8_with_config(
            &source,
            2,
            2,
            LinearQuantConfig {
                qmax: 127,
                clip_percentile: 75.0,
            },
        )
        .unwrap();
        assert!(info.clipped_max_abs < info.max_abs);
        assert!(info.clipped_values > 0);
    }
}
