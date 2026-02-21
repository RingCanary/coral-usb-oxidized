# External Research Snapshot (2026-02-21)

Scope: constraints and opportunities for repurposing Coral USB EdgeTPU toward
general math/GEMM-style acceleration.

Date note: this snapshot reflects sources checked on **February 21, 2026**.

## Official constraints (EdgeTPU USB)

1. EdgeTPU expects TensorFlow Lite models that are fully 8-bit quantized and
   compiled with `edgetpu_compiler`.
   - Source: https://coral.ai/docs/edgetpu/models-intro/
2. Model requirements include static tensor sizes, static parameters, supported
   op set only, and compile-time constraints.
   - Source: https://coral.ai/docs/edgetpu/models-intro/
3. Compiler partitioning is limited: at first unsupported op, later graph
   sections run on CPU (single partition behavior).
   - Source: https://coral.ai/docs/edgetpu/models-intro/
4. Runtime path in C++ is through TFLite + EdgeTPU custom op registration
   (`kCustomOp` / `RegisterCustomOp()`).
   - Source: https://coral.ai/docs/edgetpu/tflite-cpp/

## Compiler/runtime coupling

1. The compiler emits models tied to runtime compatibility (`min_runtime_version`
   options and version guidance).
   - Source: https://coral.ai/docs/edgetpu/compiler/
2. The compiler is distributed for Debian/x86-64 host compilation flow.
   - Source: https://coral.ai/docs/edgetpu/compiler/

## USB protocol openness

1. Public docs describe usage and model workflow, but do not publish a detailed
   USB wire-protocol spec for arbitrary host programming.
   - Inference from documentation scope:
     - https://coral.ai/docs/edgetpu/models-intro/
     - https://coral.ai/docs/edgetpu/tflite-cpp/
2. The userspace runtime implementation source is public in `libedgetpu`, so
   behavior can be studied at code level.
   - Source: https://github.com/google-coral/libedgetpu
3. Firmware/init USB identity behavior (`1a6e:089a` -> `18d1:9302`) is
   documented in WebCoral setup notes.
   - Source: https://coral.googlesource.com/webcoral/+/refs/heads/master/README.md

## Ecosystem maintenance status (risk)

1. `libedgetpu` repository is archived/read-only (archived October 14, 2025).
   - Source: https://github.com/google-coral/libedgetpu
2. `libcoral` repository is archived/read-only (archived October 14, 2025).
   - Source: https://github.com/google-coral/libcoral
3. `pycoral` repository is archived/read-only (archived July 3, 2025).
   - Source: https://github.com/google-coral/pycoral

## Practical implication for GEMM/custom math use

1. **Direct arbitrary-kernel execution on USB EdgeTPU is not exposed** in the
   public host API surface.
2. Realistic route is to encode workloads into supported quantized TFLite
   graphs (for example, `FullyConnected`/`Conv2D` forms), compile with
   `edgetpu_compiler`, and accept model/op constraints.
3. This means EdgeTPU can serve as a specialized accelerator for graph-shaped
   int8 compute, not a drop-in general GEMM coprocessor.

## Actionable direction for this repo

1. Keep reverse engineering focused on:
   - model package/executable structure (`DWN1`, serialized executables)
   - stable USB transport loop signatures and register patterns
   - mapping runtime behavior to compiler artifacts
2. Treat protocol understanding as compatibility/observability work, not a path
   to unrestricted custom kernel dispatch unless new evidence appears.
