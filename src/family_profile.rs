use crate::dense_param_stream_len;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

const DENSE_FAMILY_PROFILE_SCHEMA_V1: u32 = 1;

fn default_schema_version() -> u32 {
    DENSE_FAMILY_PROFILE_SCHEMA_V1
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct DenseFamilyReplayDefaults {
    #[serde(default)]
    pub input_bytes: Option<usize>,
    #[serde(default)]
    pub output_bytes: Option<usize>,
    #[serde(default)]
    pub bootstrap_known_good_order: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DenseFamilyProfile {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub profile_id: String,
    pub anchor_model: String,
    #[serde(default)]
    pub instruction_patch_spec: Option<String>,
    pub stored_weight_shape: [usize; 2],
    #[serde(default)]
    pub expected_param_stream_len: Option<usize>,
    #[serde(default)]
    pub replay_defaults: DenseFamilyReplayDefaults,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug)]
pub enum DenseFamilyProfileError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedSchemaVersion(u32),
    InvalidField { field: &'static str, reason: String },
}

impl fmt::Display for DenseFamilyProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenseFamilyProfileError::Io(err) => write!(f, "I/O error: {err}"),
            DenseFamilyProfileError::Json(err) => write!(f, "JSON parse error: {err}"),
            DenseFamilyProfileError::UnsupportedSchemaVersion(v) => write!(
                f,
                "unsupported dense family profile schema_version={v} (expected {})",
                DENSE_FAMILY_PROFILE_SCHEMA_V1
            ),
            DenseFamilyProfileError::InvalidField { field, reason } => {
                write!(f, "invalid field '{field}': {reason}")
            }
        }
    }
}

impl std::error::Error for DenseFamilyProfileError {}

impl From<std::io::Error> for DenseFamilyProfileError {
    fn from(value: std::io::Error) -> Self {
        DenseFamilyProfileError::Io(value)
    }
}

impl From<serde_json::Error> for DenseFamilyProfileError {
    fn from(value: serde_json::Error) -> Self {
        DenseFamilyProfileError::Json(value)
    }
}

impl DenseFamilyProfile {
    pub fn from_json_str(text: &str) -> Result<Self, DenseFamilyProfileError> {
        let profile: Self = serde_json::from_str(text)?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, DenseFamilyProfileError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    pub fn validate(&self) -> Result<(), DenseFamilyProfileError> {
        if self.schema_version != DENSE_FAMILY_PROFILE_SCHEMA_V1 {
            return Err(DenseFamilyProfileError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.profile_id.trim().is_empty() {
            return Err(DenseFamilyProfileError::InvalidField {
                field: "profile_id",
                reason: "must be non-empty".to_string(),
            });
        }
        if self.anchor_model.trim().is_empty() {
            return Err(DenseFamilyProfileError::InvalidField {
                field: "anchor_model",
                reason: "must be non-empty".to_string(),
            });
        }
        if matches!(self.instruction_patch_spec.as_deref(), Some("")) {
            return Err(DenseFamilyProfileError::InvalidField {
                field: "instruction_patch_spec",
                reason: "if present, must be non-empty".to_string(),
            });
        }

        let rows = self.stored_weight_rows();
        let cols = self.stored_weight_cols();
        let computed_len = dense_param_stream_len(rows, cols).map_err(|err| {
            DenseFamilyProfileError::InvalidField {
                field: "stored_weight_shape",
                reason: err.to_string(),
            }
        })?;

        if let Some(expected) = self.expected_param_stream_len {
            if expected != computed_len {
                return Err(DenseFamilyProfileError::InvalidField {
                    field: "expected_param_stream_len",
                    reason: format!(
                        "expected {} does not match computed rows*cols {} (rows={} cols={})",
                        expected, computed_len, rows, cols
                    ),
                });
            }
        }

        if matches!(self.replay_defaults.input_bytes, Some(0)) {
            return Err(DenseFamilyProfileError::InvalidField {
                field: "replay_defaults.input_bytes",
                reason: "must be >= 1 when present".to_string(),
            });
        }
        if matches!(self.replay_defaults.output_bytes, Some(0)) {
            return Err(DenseFamilyProfileError::InvalidField {
                field: "replay_defaults.output_bytes",
                reason: "must be >= 1 when present".to_string(),
            });
        }

        Ok(())
    }

    pub fn stored_weight_rows(&self) -> usize {
        self.stored_weight_shape[0]
    }

    pub fn stored_weight_cols(&self) -> usize {
        self.stored_weight_shape[1]
    }

    pub fn computed_param_stream_len(&self) -> Result<usize, DenseFamilyProfileError> {
        dense_param_stream_len(self.stored_weight_rows(), self.stored_weight_cols()).map_err(
            |err| DenseFamilyProfileError::InvalidField {
                field: "stored_weight_shape",
                reason: err.to_string(),
            },
        )
    }

    pub fn resolve_anchor_model_path(&self, profile_path: &Path) -> PathBuf {
        resolve_path_relative_to_profile(profile_path, &self.anchor_model)
    }

    pub fn resolve_instruction_patch_spec_path(&self, profile_path: &Path) -> Option<PathBuf> {
        self.instruction_patch_spec
            .as_ref()
            .map(|v| resolve_path_relative_to_profile(profile_path, v))
    }
}

fn resolve_path_relative_to_profile(profile_path: &Path, value: &str) -> PathBuf {
    let value_path = Path::new(value);
    if value_path.is_absolute() {
        return value_path.to_path_buf();
    }
    if let Some(parent) = profile_path.parent() {
        return parent.join(value_path);
    }
    value_path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{DenseFamilyProfile, DenseFamilyProfileError};
    use std::path::Path;

    fn sample_profile_json() -> &'static str {
        r#"{
  "schema_version": 1,
  "profile_id": "holdout-family-8976-2352",
  "anchor_model": "anchors/dense_anchor_1792.tflite",
  "instruction_patch_spec": "patches/family_8976_2352.patchspec",
  "stored_weight_shape": [1792, 1792],
  "expected_param_stream_len": 3211264,
  "replay_defaults": {
    "input_bytes": 1792,
    "output_bytes": 1792,
    "bootstrap_known_good_order": true
  }
}"#
    }

    #[test]
    fn parse_valid_profile() {
        let profile = DenseFamilyProfile::from_json_str(sample_profile_json()).unwrap();
        assert_eq!(profile.profile_id, "holdout-family-8976-2352");
        assert_eq!(profile.stored_weight_rows(), 1792);
        assert_eq!(profile.stored_weight_cols(), 1792);
        assert_eq!(profile.computed_param_stream_len().unwrap(), 3_211_264);
    }

    #[test]
    fn reject_bad_schema() {
        let text = sample_profile_json().replace("\"schema_version\": 1", "\"schema_version\": 2");
        let err = DenseFamilyProfile::from_json_str(&text).unwrap_err();
        assert!(matches!(
            err,
            DenseFamilyProfileError::UnsupportedSchemaVersion(2)
        ));
    }

    #[test]
    fn reject_bad_dims() {
        let text = sample_profile_json().replace("[1792, 1792]", "[1792, 897]");
        let err = DenseFamilyProfile::from_json_str(&text).unwrap_err();
        match err {
            DenseFamilyProfileError::InvalidField { field, reason } => {
                assert_eq!(field, "stored_weight_shape");
                assert!(reason.contains("multiples of 64"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn resolve_relative_paths_from_profile() {
        let profile = DenseFamilyProfile::from_json_str(sample_profile_json()).unwrap();
        let profile_path = Path::new("docs/artifacts/family_profiles/sample.json");
        assert_eq!(
            profile.resolve_anchor_model_path(profile_path),
            Path::new("docs/artifacts/family_profiles/anchors/dense_anchor_1792.tflite")
        );
        assert_eq!(
            profile
                .resolve_instruction_patch_spec_path(profile_path)
                .unwrap(),
            Path::new("docs/artifacts/family_profiles/patches/family_8976_2352.patchspec")
        );
    }
}
