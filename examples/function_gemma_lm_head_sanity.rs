use coral_usb_oxidized::{FunctionGemmaError, FunctionGemmaSafeTensorFile};
use std::env;
use std::error::Error;
use std::time::Instant;

fn usage(program: &str) {
    println!("Usage: {program} <model.safetensors> <token_id> [topk]");
    println!("Defaults: topk=10");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "function_gemma_lm_head_sanity".to_string());
    if args.len() < 3 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage(&program);
        if args.len() < 3 {
            std::process::exit(2);
        }
        return Ok(());
    }

    let model_path = &args[1];
    let token_id = args[2].parse::<usize>()?;
    let topk = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    if topk == 0 {
        return Err(Box::new(FunctionGemmaError::InvalidArgument(
            "topk must be >= 1".to_string(),
        )));
    }

    let model = FunctionGemmaSafeTensorFile::load(model_path)?;
    let (vocab_size, hidden_size) = model.embedding_dims()?;
    if token_id >= vocab_size {
        return Err(Box::new(FunctionGemmaError::InvalidArgument(format!(
            "token id {} out of range [0, {})",
            token_id, vocab_size
        ))));
    }

    let started = Instant::now();
    let embedding = model.token_embedding_row_f32(token_id)?;
    let embedding_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let logits_topk = model.lm_head_topk_from_hidden(&embedding, topk)?;
    let lm_head_ms = started.elapsed().as_secs_f64() * 1000.0;

    println!("Model: {}", model_path);
    println!("Vocab: {} Hidden: {}", vocab_size, hidden_size);
    println!("Input token id: {}", token_id);
    println!("Embedding lookup: {:.3} ms", embedding_ms);
    println!("LM-head top{}: {:.3} ms", topk, lm_head_ms);
    println!("Top logits:");
    for (rank, (candidate_id, score)) in logits_topk.iter().enumerate() {
        println!(
            "  {:>2}: token_id={} score={:.6}",
            rank + 1,
            candidate_id,
            score
        );
    }

    Ok(())
}
