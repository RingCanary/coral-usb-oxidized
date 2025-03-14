use std::env;
use std::path::PathBuf;
use bindgen;

fn main() {
    // Tell cargo to look for libraries in the system paths
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    
    // Add the TensorFlow Lite C API library path
    println!("cargo:rustc-link-search=native=/home/bhavesh/Devmnt/ai-dev/CORAL/tensorflow-source/bazel-bin/tensorflow/lite/c");
    
    // Link with the real libraries
    println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
    println!("cargo:rustc-link-lib=dylib=edgetpu");
    
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    
    // Add the local edgetpu_runtime directory to the include path
    let edgetpu_include_path = std::path::Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("edgetpu_runtime")
        .join("libedgetpu");
    
    println!("cargo:rustc-link-search={}", edgetpu_include_path.display());
    
    // Add TensorFlow include paths - use the source directory directly
    let tensorflow_include_path = "/home/bhavesh/Devmnt/ai-dev/CORAL/tensorflow-source";
    
    // Create a bindgen builder
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header("src/wrapper.h");
    
    // Add the edgetpu_runtime/libedgetpu directory to the include path
    builder = builder.clang_arg(format!("-I{}", edgetpu_include_path.display()));
    
    // Add the TensorFlow include directory to the include path
    builder = builder.clang_arg(format!("-I{}", tensorflow_include_path));
    
    // Define that we have TensorFlow Lite available
    builder = builder.clang_arg("-DHAVE_TENSORFLOW_LITE");
    
    // Generate the bindings
    let bindings = builder
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings
        .generate()
        // Unwrap the Result and panic on failure
        .expect("Unable to generate bindings");
    
    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
