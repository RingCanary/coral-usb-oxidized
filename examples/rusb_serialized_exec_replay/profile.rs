use super::*;

pub(crate) struct FamilyProfileResolution {
    pub(crate) loaded: Option<(PathBuf, DenseFamilyProfile)>,
    pub(crate) instruction_patch_paths: Vec<String>,
}

pub(crate) fn resolve_family_profile(
    config: &mut Config,
) -> Result<FamilyProfileResolution, Box<dyn Error>> {
    let mut loaded = None;
    let mut instruction_patch_paths = Vec::new();
    if let Some(profile_path_raw) = config.family_profile.as_ref() {
        let profile_path = PathBuf::from(profile_path_raw);
        let profile = DenseFamilyProfile::from_path(&profile_path)?;
        let anchor_model = profile.resolve_anchor_model_path(&profile_path);
        if config.model_path.is_empty() {
            config.model_path = anchor_model.to_string_lossy().into_owned();
        }
        if !config.bootstrap_known_good_order
            && profile.replay_defaults.bootstrap_known_good_order == Some(true)
        {
            config.bootstrap_known_good_order = true;
        }
        if config.input_bytes == 150_528 {
            if let Some(v) = profile.replay_defaults.input_bytes {
                config.input_bytes = v;
            }
        }
        if config.output_bytes == 1001 {
            if let Some(v) = profile.replay_defaults.output_bytes {
                config.output_bytes = v;
            }
        }
        if config.instruction_patch_spec.is_none() {
            if let Some(spec_path) = profile.resolve_instruction_patch_spec_path(&profile_path) {
                instruction_patch_paths.push(spec_path.to_string_lossy().into_owned());
            }
        }
        loaded = Some((profile_path, profile));
    }
    Ok(FamilyProfileResolution {
        loaded,
        instruction_patch_paths,
    })
}
