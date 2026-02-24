# Coral EdgeTPU RE Analysis & Plan

## 1. The RE Path: Converting EdgeTPU to a GEMM Device

The Coral EdgeTPU is historically an inflexible, black-box AI accelerator. It accepts only `edgetpu_compiler` generated `.tflite` models and runs them through a closed-source compilation heuristic. The RE path taken in this repository ingeniously sidesteps this through **"Template Parameter Patching" (Tensorizer MVP)**:

1.  **Template Generation**: Create a synthetic, single-operation TFLite model containing just one `FULLY_CONNECTED` (Dense) or `CONV_2D` layer with the exact matrix dimensions needed (e.g., 2304x2304).
2.  **Black-box Compilation**: Pass this TFLite model through the official `edgetpu_compiler` to produce a `*_edgetpu.tflite` file. This yields a valid instruction stream (EXECUTION_ONLY) and a static weight payload (PARAMETER_CACHING).
3.  **FlatBuffer Extraction**: Use a custom parser based on `libedgetpu_executable.fbs` schema to extract the compiled `DWN1` package, carving out the exact memory region of the `PARAMETER_CACHING` blob within the file.
4.  **Layout Probing (Single-Hot Recovery)**: Because the `edgetpu_compiler` swizzles/re-strides the weight matrix for memory bandwidth optimization, the layout is non-intuitive. The RE approach compiled hundreds of TFLite models, each with exactly *one* weight set to a non-zero value, and diffed the resulting compiled payloads. This recovered the exact non-linear mapping formula (e.g., `offset = (col//64)*16384 + (row//64)*4096...`).
5.  **Dynamic Inference Injection**: At runtime, written in Rust, the app takes a new arbitrary matrix, quantizes it to INT8, applies the recovered spatial offset mapping, and overwrites the bytes in the `PARAMETER_CACHING` memory directly.
6.  **Tiling and Pipelines**: By reusing these single-op templates, they chain together Transformer blocks (Function-Gemma layers) and manually orchestrate multi-tile matrix multiplication from the host CPU.

> [!NOTE]
> The brilliance of this approach is that they bypassed the compiler's instruction generation entirely (`EXECUTION_ONLY` remains untouched) and hijacked the data transport to build a dynamic GEMM engine.

---

## 2. Critical Flaws in the Current Approach

While highly creative, this approach relies heavily on reverse-engineered artifacts rather than foundational system-level control.

1.  **Template Dimensions Fragility (The Shape Prison)**: Since you don't control the instruction stream, you are strictly limited to the exact matrix dimensions you pre-compiled. If an NLP model needs a projection of `2048x768`, and you only have templates for `2304x2304` or `256x256`, you either waste cycles padding or fail outright. The compiler might outright reject shapes it deems "suboptimal" for the hardware.
2.  **Extreme Host-Loop Bottlenecks**: Tiling large matrices (like an 8192x8192 GEMM) by re-calling the TFLite interpreter for smaller `2688x2688` templates requires the host CPU to intervene continuously. This breaks the pipeline logic. A native compiler would DMA loop the tiles directly on the accelerator via hardware control, doing it all in one USB submit. Currently, this causes intense USB IO latency (linear scaling of `SUBMITURB` per tile).
3.  **Lack of Opcode/Activation Control**: The `EXECUTION_ONLY` chunk is completely opaque. You cannot fuse an activation function (like SwiGLU or SiLU) into the GEMM because you don't know the bytecode to trigger the activation lookup table on the TPU. You must bounce back to the CPU for activation.
4.  **Toolchain Bitrot**: The official ecosystem (`libedgetpu`, `edgetpu_compiler`) has been archived by Google for years. Relying on ancient dependencies (TF 2.10.1, Python 3.9) to generate templates is mathematically perilous moving forward into 2026.

---

## 3. Methodology for Completeness

To achieve true control—running a fully transparent, bare-metal acceleration flow—we must transition from **parameter hijacking** to **instruction and protocol synthesis**.

### Phase A: Pure protocol control (Host/Driver level)
Currently, `libedgetpu` is a black box that executes the USB traffic.
1.  **Map Control Plane (MMIO)**: Finalize the `usb_register_map_candidates.md` by identifying which `wValue`/`wIndex` control transfers are writing to the TPU's Command Queue pointer, Status registers, and Reset controls.
2.  **Rust-Native libusb Driver**: Build a pure-Rust driver (`rusb`) that implements the transition `1a6e:089a -> 18d1:9302`, submits raw bulk endpoint arrays (the `EXECUTION_ONLY` buffers), and polls the interrupt endpoint for completion.

### Phase B: Disassembly and Opcodes (ISA level)
1.  **Structured Opcode Fuzzing**: Instead of single-hot weights, build a fuzzer that varies the **operations** in TFLite (e.g., Conv2D vs Depthwise, stride 1 vs stride 2, no activation vs ReLU).
2.  **Bytecode Diffing**: Extract the `EXECUTION_ONLY` chunk, group it into instruction sizes (likely 64-bit or 128-bit VLIW instructions), and diff them.
3.  **Map the Hardware Data Path**: We must identify instructions for:
    *   **DMA Reads**: Loading activations/weights from USB Host Memory to On-chip SRAM.
    *   **Matrix Multiply Unit (MMU) Dispatch**: Kicking off the systolic array.
    *   **Vector Processing Unit (VPU)**: Quantization shifts, zero-point additions, activations.
    *   **DMA Writes**: Pushing the final tensor back to Host Memory via Bulk-IN endpoint.

### Phase C: Graph Lowering
Once the ISA is known, we implement a lightweight JIT compiler that lowers dynamic matrix shapes from PyTorch/SafeTensors into bare TPU instructions, and blasts them directly over USB without ever touching `edgetpu_compiler`.

---

## 4. The Key Missing Piece: TFLite to TPU Compilation

> [!CAUTION]  
> The critical missing piece in building a custom compiler is **The TPU Memory Allocator and Instruction Set Architecture (ISA) definition for the underlying "Darwinn" (Tensor Processing Core) architecture.**

We understand the Flatbuffer metadata (`libedgetpu_executable.fbs` tells us where chunks are), and we know the Data (the `PARAMETER_CACHING` blobs). **We do not know the Compute.**

To build a custom compiler, we are missing the semantic understanding of:
1.  **The Instruction Encoding**: the bit-fields that make up a TPU hardware instruction.
2.  **SRAM Address Allocation**: When the `edgetpu_compiler` generates `BASE_ADDRESS_INPUT_ACTIVATION` and `BASE_ADDRESS_OUTPUT_ACTIVATION`, it determines where in the 8MB SRAM chip the tensors will securely reside during compute. We do not know the bounds of these memory blocks or the rules the TPU uses to avoid overwriting its own instruction cache.
3.  **Descriptor Rings/Command Queues**: How are the `EXECUTION_ONLY` blobs unpacked by the hardware? Are they DMA descriptors that point to memory locations, or are they actual ALU instructions? The high latency and bulk USB loops suggest a Descriptor-driven DMA engine orchestrating a systolic array.

**Solution Plan**: We must stop looking at `.tflite` payload layouts and start diffing the `EXECUTION_ONLY` chunks. By capturing hundreds of models with slight parameter variations, we can build a heuristic disassembler. Only when we map a bitfield to "Multiply row by column" can we build a custom compiler.
