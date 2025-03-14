#!/bin/bash

# Set the library path to include the TensorFlow Lite C API library
export LD_LIBRARY_PATH=/home/bhavesh/Devmnt/ai-dev/CORAL/tensorflow-source/bazel-bin/tensorflow/lite/c:$LD_LIBRARY_PATH

# Run the test program
cargo run --example tflite_test
