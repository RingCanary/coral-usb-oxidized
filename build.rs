use std::env;
use std::path::Path;

fn add_link_search_path(path: &str, link_paths: &mut Vec<String>) {
    if Path::new(path).exists() {
        println!("cargo:rustc-link-search=native={path}");
        if !link_paths.iter().any(|p| p == path) {
            link_paths.push(path.to_string());
        }
    }
}

fn has_library(link_paths: &[String], filename: &str) -> bool {
    link_paths
        .iter()
        .any(|path| Path::new(path).join(filename).exists())
}

fn main() {
    println!("cargo:rerun-if-env-changed=CORAL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=EDGETPU_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TFLITE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TFLITE_LINK_LIB");

    let legacy_runtime_enabled = env::var_os("CARGO_FEATURE_LEGACY_RUNTIME").is_some();
    if !legacy_runtime_enabled {
        println!(
            "cargo:warning=legacy-runtime feature disabled; skipping libedgetpu/tflite linkage (pure-rusb mode)"
        );
        return;
    }

    let mut link_paths = Vec::new();

    for env_var in ["CORAL_LIB_DIR", "EDGETPU_LIB_DIR", "TFLITE_LIB_DIR"] {
        if let Ok(path) = env::var(env_var) {
            add_link_search_path(&path, &mut link_paths);
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
        add_link_search_path(path, &mut link_paths);
    }

    if has_library(&link_paths, "libedgetpu.so") {
        println!("cargo:rustc-link-lib=dylib=edgetpu");
    } else if has_library(&link_paths, "libedgetpu.so.1") {
        // Some distros ship only the SONAME without an unversioned linker symlink.
        println!("cargo:rustc-link-arg=-Wl,-l:libedgetpu.so.1");
    } else {
        println!("cargo:warning=Could not find libedgetpu in configured library paths");
        println!("cargo:rustc-link-lib=dylib=edgetpu");
    }

    if let Ok(link_lib) = env::var("TFLITE_LINK_LIB") {
        println!("cargo:rustc-link-lib=dylib={link_lib}");
    } else if has_library(&link_paths, "libtensorflowlite_c.so") {
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
    } else if has_library(&link_paths, "libtensorflow-lite.so") {
        println!("cargo:rustc-link-lib=dylib=tensorflow-lite");
    } else {
        println!(
            "cargo:warning=Could not find libtensorflowlite_c.so or libtensorflow-lite.so in configured library paths"
        );
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
    }
}
