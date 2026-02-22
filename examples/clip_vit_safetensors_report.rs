use coral_usb_oxidized::{
    quantize_linear_out_in_to_row_major_qi8, ClipSafeTensorFile, ClipVitB32Dims,
    ClipVitLayerLinearNames,
};
use std::env;
use std::error::Error;

fn usage(program: &str) {
    println!("Usage: {program} <clip_model.safetensors> [layer_idx] [qmax]");
    println!("Defaults: layer_idx=0 qmax=127");
    println!("Example: cargo run --example clip_vit_safetensors_report -- model.safetensors 0 127");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "clip_vit_safetensors_report".to_string());

    if args.len() < 2 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage(&program);
        return Ok(());
    }

    let model_path = &args[1];
    let layer_idx = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let qmax = args
        .get(3)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(127);

    let dims = ClipVitB32Dims::default();
    let model = ClipSafeTensorFile::load(model_path)?;
    let tensor_count = model.tensor_count()?;
    let layer_indices = model.discover_clip_vit_encoder_layers()?;

    println!("Model: {}", model_path);
    println!("Tensor count: {}", tensor_count);
    println!(
        "Discovered CLIP vision encoder layers: count={} first={:?} last={:?}",
        layer_indices.len(),
        layer_indices.first(),
        layer_indices.last()
    );

    let names = model.validate_clip_vit_layer_linears(layer_idx, dims)?;
    println!(
        "Validated layer {} linear tensors (expected d_model={} mlp_hidden={})",
        layer_idx, dims.d_model, dims.mlp_hidden
    );

    print_tensor_info(&model, &names)?;

    let q_proj = model.tensor_f32(&names.q_proj)?;
    let (q_proj_q, q_proj_info) =
        quantize_linear_out_in_to_row_major_qi8(&q_proj, dims.d_model, dims.d_model, qmax)?;
    println!(
        "Quantized q_proj: q_bytes={} scale={:.9} max_abs={:.9} qmax={}",
        q_proj_q.len(),
        q_proj_info.scale,
        q_proj_info.max_abs,
        q_proj_info.qmax
    );

    let fc1 = model.tensor_f32(&names.mlp_fc1)?;
    let (fc1_q, fc1_info) =
        quantize_linear_out_in_to_row_major_qi8(&fc1, dims.d_model, dims.mlp_hidden, qmax)?;
    println!(
        "Quantized mlp_fc1: q_bytes={} scale={:.9} max_abs={:.9} qmax={}",
        fc1_q.len(),
        fc1_info.scale,
        fc1_info.max_abs,
        fc1_info.qmax
    );

    let fc2 = model.tensor_f32(&names.mlp_fc2)?;
    let (fc2_q, fc2_info) =
        quantize_linear_out_in_to_row_major_qi8(&fc2, dims.mlp_hidden, dims.d_model, qmax)?;
    println!(
        "Quantized mlp_fc2: q_bytes={} scale={:.9} max_abs={:.9} qmax={}",
        fc2_q.len(),
        fc2_info.scale,
        fc2_info.max_abs,
        fc2_info.qmax
    );

    Ok(())
}

fn print_tensor_info(
    model: &ClipSafeTensorFile,
    names: &ClipVitLayerLinearNames,
) -> Result<(), Box<dyn Error>> {
    for name in names.all() {
        let info = model.tensor_info(name)?;
        println!(
            "  {}: dtype={:?} shape={:?} bytes={}",
            info.name, info.dtype, info.shape, info.bytes
        );
    }
    Ok(())
}
