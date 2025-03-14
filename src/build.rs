use std::env;
use std::path::PathBuf;

fn main() {
    let project_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    
    // Only link with libedgetpu when not using mock feature
    #[cfg(not(feature = "mock"))]
    {
        // Link dynamically with the system library
        println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=dylib=edgetpu");
        
        // Fallback to project-specific library if needed
        println!("cargo:rustc-link-search={}", project_dir.join("edgetpu_runtime/libedgetpu/direct/k8").display());
    }
    
    // Rerun build if these files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/build.rs");
}