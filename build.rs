use std::env;
use std::path::Path;

fn add_link_search_path(path: &str) {
    if Path::new(path).exists() {
        println!("cargo:rustc-link-search=native={path}");
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=CORAL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=EDGETPU_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TFLITE_LIB_DIR");

    for env_var in ["CORAL_LIB_DIR", "EDGETPU_LIB_DIR", "TFLITE_LIB_DIR"] {
        if let Ok(path) = env::var(env_var) {
            add_link_search_path(&path);
        }
    }

    // Common Linux library locations (x86_64 + Raspberry Pi/aarch64).
    for path in [
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/arm-linux-gnueabihf",
    ] {
        add_link_search_path(path);
    }

    println!("cargo:rustc-link-lib=dylib=edgetpu");
    println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
}
