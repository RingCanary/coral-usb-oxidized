pub(crate) fn header_and_stream_len(
    payload_len: usize,
    cap: Option<usize>,
    force_full_header_len: bool,
) -> (usize, usize) {
    let stream_len = cap.map(|m| m.min(payload_len)).unwrap_or(payload_len);
    let header_len = if force_full_header_len {
        payload_len
    } else {
        stream_len
    };
    (header_len, stream_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_chunk_boundary_header_vs_capped_len() {
        assert_eq!(header_and_stream_len(8192, Some(4096), false), (4096, 4096));
        assert_eq!(header_and_stream_len(8192, Some(4096), true), (8192, 4096));
    }
}
