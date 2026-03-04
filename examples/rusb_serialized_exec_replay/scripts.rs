use super::*;

pub(crate) fn apply_script_defaults(config: &mut Config) {
    if config.script1_interleave && config.param_interleave_window_bytes.is_none() {
        config.param_interleave_window_bytes = Some(32_768);
    }
    if config.script2_queue_probe {
        if config.param_csr_probe_offsets.is_empty() {
            config.param_csr_probe_offsets = vec![32_000];
        }
        if config.param_stream_chunk_size.is_none() {
            config.param_stream_chunk_size = Some(256);
        }
    }
    if config.script3_poison_diff {
        if config.param_poison_probe_offset.is_none() {
            config.param_poison_probe_offset = Some(33_024);
        }
        if config.param_stream_chunk_size.is_none() {
            config.param_stream_chunk_size = Some(256);
        }
    }
}
