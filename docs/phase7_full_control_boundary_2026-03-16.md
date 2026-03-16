# Phase 7 Full-Control Boundary

Date: 2026-03-16

## Purpose

State precisely what the repo now controls natively, what still depends on vendor
artifacts or legacy surfaces, and what remains before the Coral USB stick can be
described as a fully custom-controlled EdgeTPU or general GEMM device.

## Short answer

No, the repo does not yet prove full end-to-end control of the Coral USB /
EdgeTPU stack.

Yes, it does prove a strong bounded native-control milestone:

- pure-`rusb` runtime execution and replay on real hardware
- native parameter materialization for tested Dense and Conv2D families
- native EO emission for bounded Conv2D families
- exact DUT hash reproduction without `libedgetpu` or `edgetpu_compiler` in the
  active Phase 4-6 loop

The correct claim today is:

> the Coral USB stick is a bounded native offload engine under our control, not
> yet a fully general native compiler target or fully owned firmware/runtime
> stack.

## What is already under native control

### 1. USB runtime control

The repo has a native Rust USB path for:

- device discovery in boot and runtime modes
- DFU-style firmware upload of a known runtime image
- runtime setup/control-plane probing
- replay of compiled or natively materialized executable state
- on-device validation via Pi replay helpers

Relevant surfaces:

- `examples/rusb_control_plane_probe.rs`
- `examples/rusb_serialized_exec_replay.rs`
- `src/usb/driver.rs`
- `docs/rusb_control_plane_probe.md`

This means execution no longer depends on `libedgetpu` in the active path.

### 2. Native bounded artifact generation

Dense:

- parameter stream packing is compilerless and exact for the tested Dense regimes
- the unresolved part is Dense EO target-state generation for unseen target
  dimensions

Reference:

- `docs/phase2_dense_completion_2026-03-06.md`

Conv2D:

- Phase 4 completed a bounded same-product `EO=6512` family
- Phase 5 completed a bounded mixed-product `fixed_height=8` `EO=6496` family
- Phase 6 widened that `H=8` family to the full checked-in corridor
  `8x72..8x192` across `p32/p64/p128`

References:

- `docs/phase4_conv2d_k3_completion_2026-03-16.md`
- `docs/phase5_conv2d_k3_h8_band_completion_2026-03-16.md`
- `docs/phase6_conv2d_k3_h8_corridor_completion_2026-03-16.md`

This is stronger than compiler-assisted replay control: the active bounded path
materializes params and EO natively and reproduces target hashes on DUT.

### 3. Practical GEMV/GEMM-style offload

The Dense replay path is practically usable as a bounded GEMV/GEMM-style offload
engine:

- steady-state service path about `67-72 MB/s`
- replay-side GEMV about `14.3-24.0 GMAC/s`
- tiled logical GEMM about `22.2 GMAC/s`
- Pi CPU baseline about `14 GMAC/s` on the matched GEMM comparison

Reference:

- `docs/dense_replay_vs_pi_cpu_2026-03-07.md`

This is enough to call the device a bounded custom linear offload engine.
It is not yet enough to call it a fully general native GEMM device.

## What is not yet under full control

### 1. Firmware ownership

The active path still uploads a known external firmware image such as
`apex_latest_single_ep.bin`.

The repo proves:

- native host-side DFU upload of that image
- native use of the runtime it brings up

The repo does not prove:

- a custom firmware toolchain
- a custom firmware image
- full reverse-engineered ownership of runtime firmware behavior

So the current stack is:

> native host control over vendor runtime firmware

not:

> native ownership of the full device software stack

### 2. General instruction generation

The repo still does not have a general native compiler/assembler for EdgeTPU
executable state.

Specifically missing:

- general fresh EO generation for new Dense target dimensions
- general fresh EO/PC generation outside the bounded Conv2D families
- a native lowering path for arbitrary new kernels, strides, paddings, bias
  choices, or multi-op graphs

Current bounded emitters are family-specific and intentionally scoped.

### 3. General allocator / scheduler / memory-planning control

The current proofs are sufficient for bounded replay/materialization, but they
do not yet establish a fully recovered allocator/scheduler model for arbitrary
program generation.

That means the repo still lacks:

- a general SRAM/tensor placement model
- a general executable layout planner
- a general graph scheduler for arbitrary workloads

### 4. General GEMM-device semantics

The current Dense path is still replay/template-driven:

- params are native
- runtime is native
- steady-state offload is real
- but fresh compute-program generation is not general

So the repo does not yet support:

- arbitrary `M/N/K` native lowering
- a fresh GEMM code generator independent of frozen template families
- fused epilogues or broader operator coverage from native compilation

## Legacy dependency boundary

### Active path

The active path is already free of:

- `libedgetpu`
- TensorFlow Lite delegate execution
- `edgetpu_compiler`
- old Python/Bazel/TensorFlow flows

for the checked-in bounded completion families.

References:

- `README.md`
- `docs/active_path.md`
- `docs/legacy_compatibility.md`

### Remaining compatibility/archive surface

The repo still contains compatibility-only surfaces for:

- `legacy-runtime`
- delegate/TFLite examples
- `edgetpu_compiler` bootstrap/pipelines
- Python/TensorFlow-based model-generation and artifact-inspection helpers

These are no longer the active path, but they still exist for:

- interoperability
- historical reproduction
- research corpus generation

So the honest dependency statement is:

> active-path escape from the legacy stack is real, but repo-wide retirement of
> the legacy stack is not complete yet.

## Exact work left before claiming full control

The remaining work is now well-bounded.

### A. Close the remaining compute-program gap

Highest-value technical target:

- Dense EO target-state generation for unseen target dimensions

This is the strongest remaining negative proof inside the current native path.
Until that closes, the repo does not have a general native Dense codegen story.

### B. Extend beyond the Phase 6 corridor

For Conv2D, the next frontier is:

- test whether the `schema_version=2` corridor model generalizes beyond
  `fixed_height=8`
- reduce lookup residue toward a smaller reusable law
- determine whether adjacent geometry bands can also be frozen and emitted
  natively

### C. Build a native executable-generation layer

The eventual milestone is not another frozen family. It is a real native
artifact-generation layer that can emit runnable instruction state from operator
descriptions instead of only from family specs.

### D. Decide the firmware ambition explicitly

If the project wants the strongest possible claim, it must choose one of:

- fully characterize and treat the vendor firmware contract as fixed but owned
  enough for native control claims
- or pursue custom firmware ownership/tooling

Without that decision, "full control" remains host-side rather than full-stack.

### E. Promote GEMM from bounded replay to native lowering

A fully custom GEMM-device claim requires:

- arbitrary shape support
- native lowering from GEMM dimensions to runnable instruction state
- no reliance on frozen template executables for the compute program

That is still ahead of the current repo state.

## Conclusion

The repo has crossed the important boundary from:

- `libedgetpu`-driven execution
- compiler-assisted artifact patching

to:

- pure-`rusb` execution
- bounded native artifact materialization
- exact native replay proofs on hardware

But it has not yet crossed the final boundary to:

- full firmware ownership
- general executable/codegen control
- a truly general custom GEMM compiler target

So the correct 2026-03-16 status line is:

> We have strong bounded native control of Coral USB, but not full general
> control of the EdgeTPU stack yet.
