# WORKLOG

## 2026-02-24

### Objective

Push Function-Gemma decode runtime forward on Pi5 by adding LM-head runtime
policies, reducing memory pressure, capturing LM-head USB deltas, and probing a
pure-`rusb` control-plane baseline.

### Changes

1. Decode runtime modes and quality controls in
   `examples/function_gemma_decode_loop.rs`:
   - LM-head mode split:
     - `cpu`
     - `coral-preload` (`coral` alias)
     - `coral-lazy` (LRU tile cache)
   - `--lm-cache-capacity` for lazy mode.
   - `--rounds` for repeated decode rounds in one process.
   - `--prefill-logits` made opt-in (default off).
   - `--weight-quant per-tensor|per-channel` (default `per-channel`).
   - `--verify-rows` for stage verification rows.
2. Added lazy-cache thrash warning for exact full-vocab decode when
   `cache_capacity < tile_count`.
3. Added pure-`rusb` control-plane probe:
   - `examples/rusb_control_plane_probe.rs`
   - docs: `docs/rusb_control_plane_probe.md`
4. Added LM-head capture/diff helper:
   - `tools/re_capture_decode_lm_compare.sh`
   - compares CPU LM-head vs Coral LM-head decode usbmon traces and emits
     phase/bulk diffs.
5. Updated docs/README:
   - `docs/function_gemma_decode_loop.md`
   - `README.md`
6. Added approximate lazy shortlist mode in
   `examples/function_gemma_decode_loop.rs`:
   - new flag: `--lm-shortlist-tiles N` (`0` keeps exact full-vocab lazy mode)
   - lazy tile candidate policy per decode step:
     - current token tile
     - previous-step winning tiles
     - recent LRU tiles
     - round-robin fill
   - added eval counters in lazy cache stats (`eval_calls`, `avg_eval_tiles`).

### Pi5 validation

Host: `rpilm3.local` (`Linux 6.12.62+rpt-rpi-2712`, aarch64)

Runtime matrix artifact root:

- `/home/rpc/clip-traces/functiongemma-runtime-matrix-20260224T155555Z`

Selected results:

1. `cpu_l1` (`--lm-head cpu`, `--max-layers 1`, `--steps 1`)
   - `setup_ms ~= 5953`
   - `ms_per_token ~= 16670`
2. `coral_preload_l1` (`--lm-head coral-preload`, `--max-layers 1`, `--steps 1`)
   - `setup_ms ~= 56737`
   - `ms_per_token ~= 642`
3. `coral_lazy32_l1` (`--lm-head coral-lazy --lm-cache-capacity 32`)
   - `setup_ms ~= 4571`
   - `ms_per_token ~= 51883`
   - cache stats indicated eviction-heavy churn (`misses=200`, `evictions=168`)
4. Prefill logits toggle:
   - `prefill off`: `~47.6 ms`
   - `prefill on`: `~103810 ms`
5. Quant mode comparison (`--max-layers 18`, lazy cache32 constrained run):
   - `per-tensor`: broader stage correlation spread (~0.92-0.99)
   - `per-channel`: mostly `~0.998-0.999+` stage correlation
6. New shortlist sweep (`--max-layers 1 --steps 1`, `coral-lazy`, cache `32`):
   - `shortlist=8`: `~4015 ms/token`
   - `shortlist=16`: `~7954 ms/token`
   - `shortlist=24`: `~11871 ms/token`
   - `shortlist=32`: `~15844 ms/token`
   - exact lazy (`shortlist=0`): `~48047 ms/token`
7. Two-step shortlist validation (`shortlist=16`, `steps=2`):
   - step0: `~7768 ms` (initial tile preparation)
   - step1: `~123 ms` (cache hits; no evictions)
   - final lazy stats: `hits=16`, `misses=16`, `evictions=0`

### USBMON LM-head compare

Capture root:

- `/home/rpc/clip-traces/re-decode-lm-compare-20260224T1610Z`

Produced:

- `cpu_phase.json`
- `coral_phase.json`
- `cpu_vs_coral_diff.json`
- `cpu_bulk.txt`
- `coral_bulk.txt`

Key diff (CPU LM-head vs Coral LM-head run):

1. Transfer counts:
   - CPU: `Bi=302`, `Bo=584`, `Ii=8`
   - Coral: `Bi=3066`, `Bo=4984`, `Ii=20`
2. Duration:
   - CPU run: `~21.6 s`
   - Coral run: `~57.4 s` (includes Coral LM tile setup/preload path)

### Notes

1. `coral-preload` remains the best exact full-vocab decode mode for throughput.
2. `coral-lazy` now supports shortlist/approx decode and can avoid full-vocab
   thrash on low cache capacities; it trades token quality for speed.

### End-to-end completion matrix (Pi5, 2026-02-24)

Artifact root:

- `/home/rpc/clip-traces/functiongemma-e2e-20260224T163033Z`

Cases executed end-to-end (same prompt `2,2516,29901`):

1. `C1_cpu_l1_s1`
2. `C2_coral_preload_l1_s1`
3. `C3_coral_lazy_exact_l1_s1`
4. `C4_coral_lazy_short16_l1_s1`
5. `C5_coral_lazy_short16_l1_s2`
6. `C6_coral_preload_l18_s1`
7. `C7_coral_lazy_short16_l18_s1`

Key outcomes:

1. Exact parity check (`l1`, `steps=1`):
   - CPU next token: `155904`
   - Coral preload next token: `155904` (match)
2. Per-token latency:
   - CPU `l1`: `~14628.6 ms`
   - Coral preload `l1`: `~640.1 ms`
   - Coral lazy exact `l1` (cache32): `~48030.2 ms`
   - Coral lazy shortlist16 `l1` (cache32): `~7932.6 ms`
3. Multi-step shortlist reuse (`l1`, `steps=2`, shortlist16):
   - step0 `~7918.9 ms`
   - step1 `~122.4 ms`
   - final cache stats: `hits=16`, `misses=16`, `evictions=0`
4. Full-depth (`18` layers) decode:
   - Coral preload:
     - `setup ~83816.6 ms`
     - `decode ~995.3 ms/token`
   - Coral lazy shortlist16:
     - `setup ~34558.8 ms`
     - `decode ~8085.4 ms/token`

### Phase-B RE tooling upgrade (EXECUTION_ONLY diff prep)

1. Added dedicated chunk-diff tool:
   - `tools/exec_chunk_diff.py`
   - compares instruction chunks from `serialized_executable_*.bin`
   - reports:
     - changed bytes per chunk
     - changed instruction-word count (configurable width)
     - relocation-byte overlap vs non-relocation changes
   - supports chunk dump export (`--dump-chunks`) and exec-index filtering for
     directory inputs (`--only-exec-index`).
   - fixed absolute-path chunk-label handling after Pi validation (prevents
     dump path escaping to `/...`).
2. Added workflow documentation:
   - `docs/executable_opcode_diff.md`
3. Added persistent four-point focus tracker:
   - `docs/focus_points.md`
   - used as the long-standing list to keep work centered on the primary
     outcome.
4. Local validation sample:
   - command:
     - `python3 tools/exec_chunk_diff.py traces/dense-template-256x256-20260222T062154Z/extract/package_000 traces/dense-template-512x512-20260222T062006Z/extract/package_000 --only-exec-index 0 --out-dir traces/phase-b-diff-256-vs-512-exec0 --dump-chunks`
   - output:
     - `changed_bytes=3433`
     - `instr_changed=360/514`
     - `reloc_changed=3`
   - artifacts:
     - `traces/phase-b-diff-256-vs-512-exec0/exec_chunk_diff_report.json`
     - `traces/phase-b-diff-256-vs-512-exec0/chunk_dump_manifest.json`

### External findings ingestion (`coral_research_findings-1/2`)

1. Verified key claims against upstream/public sources (`libedgetpu` and
   `libredgetpu`):
   - register mappings:
     - `0x44018 -> scalarCoreRunControl`
     - `0x48788 -> tileconfig0`
     - `0x1a30c -> scu_ctrl_0`
   - endpoint roles:
     - interrupt endpoint `0x83` (`kInterruptInEndpoint = 3`)
     - event endpoint `0x82` (`kEventInEndpoint = 2`)
   - 8-byte bulk-out header format:
     - `[len_le32][tag_u8][pad_u24]`
2. Upgraded `tools/usbmon_register_map.py`:
   - now computes full CSR offsets with:
     - `full_offset = (wIndex << 16) | wValue`
   - includes known register name annotations for key DarwiNN offsets.
3. Upgraded `tools/usbmon_bulk_signature.py`:
   - decodes `Bo size=8` headers into payload length + stream tag names.
4. Updated protocol notes in `docs/usb_register_map_candidates.md` to include
   verified mappings and header semantics.

### Phase-A probe expansion (`rusb_control_plane_probe`)

1. Upgraded `examples/rusb_control_plane_probe.rs` from passive topology probe
   to actionable control-plane harness with:
   - vendor register reads:
     - `--vendor-read32 OFF`
     - `--vendor-read64 OFF`
   - vendor register writes:
     - `--vendor-write32 OFF=VAL`
     - `--vendor-write64 OFF=VAL`
   - endpoint reads:
     - `--read-event N` (endpoint `0x82`, event descriptor decode)
     - `--read-interrupt N` (endpoint `0x83`, `raw_data` decode)
   - configurable timeout:
     - `--timeout-ms N`
2. Added known CSR name annotation in probe output for key mapped registers:
   - `scalarCoreRunControl`, `tileconfig0`, `scu_ctrl_0`,
     queue base/tail/completed head offsets.
3. Updated docs:
   - `docs/rusb_control_plane_probe.md`
   - `docs/focus_points.md` status notes for points 1 and 2.

### Control-sequence extraction tooling

1. Extended `tools/usbmon_register_map.py` with `sequence` mode:
   - emits ordered vendor control operations (timestamped) for selected phases
   - includes:
     - full CSR offset (`(wIndex << 16) | wValue`)
     - operation class (`read32/read64/write32/write64`)
     - known register name (when mapped)
     - decoded write payload values for CSR writes
2. Validation sample (local trace):
   - command:
     - `python3 tools/usbmon_register_map.py sequence traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log --bus 4 --device 005 --phase pre_loop --json`
   - result:
     - `sequence_count=52` pre-loop vendor ops
     - includes decoded writes such as:
       - `scu_ctrl_0` write value `0x000f0059`
       - `scalarCoreRunControl` write value `0x1`
       - `tileconfig0` write value `0x7f`
3. Updated `docs/usb_register_map_candidates.md` reproduction commands with
   `sequence` extraction flow.

### Fresh Pi capture + replay-candidate extraction

1. Recovered device from DFU state and captured fresh delegate bring-up trace:
   - `traces/usbmon-20260224T182649Z-bus4/usbmon-bus4-20260224T182649Z.log`
   - observed expected transition:
     - `1a6e:089a -> 18d1:9302`
2. Extracted ordered control sequence from the runtime-side device ID stream
   (`device=016` in this capture):
   - artifacts:
     - `/home/rpc/clip-traces/replay-seq-usbmon-20260224T182649Z-bus4/device016-seq.json`
     - `/home/rpc/clip-traces/replay-seq-usbmon-20260224T182649Z-bus4/device016-seq.txt`
   - `sequence_count=95` vendor ops
3. Added replay-oriented sequence filters:
   - `--writes-only`
   - `--known-only`
4. Replay candidate artifacts from same capture:
   - all writes:
     - `/home/rpc/clip-traces/replay-seq-usbmon-20260224T182649Z-bus4/device016-writes.json`
     - `count=73`
   - known-register writes only:
     - `/home/rpc/clip-traces/replay-seq-usbmon-20260224T182649Z-bus4/device016-known-writes.json`
     - `count=6`
     - sequence:
       - `scu_ctrl_0 <= 0x000f0059`
       - `tileconfig0 <= 0x7f`
       - `scalarCoreRunControl <= 0x1`
       - `tileconfig0 <= 0x7f`
     - `scalarCoreRunControl <= 0x2`
     - `tileconfig0 <= 0x7f`

### Phase-A replay smoke (known-write subset)

1. Applied known-write subset from fresh capture directly via
   `examples/rusb_control_plane_probe.rs` on Pi5 runtime state (`18d1:9302`):
   - `scu_ctrl_0 <= 0x000f0059`
   - `tileconfig0 <= 0x7f`
   - `scalarCoreRunControl <= 0x1`
   - `tileconfig0 <= 0x7f`
   - `scalarCoreRunControl <= 0x2`
   - `tileconfig0 <= 0x7f`
2. Result:
   - all writes returned success (no timeout)
   - device remained enumerated as `18d1:9302` after replay
3. Post-replay health check:
   - `cargo run --example delegate_usage` succeeded with delegate creation and
     expected initialized-state behavior.

## 2026-02-22

### Objective

Wire a CLIP ViT-B/32 per-layer linear-stage pipeline (`q/k/v/o/fc1/fc2`) onto
existing Dense EdgeTPU templates with explicit stage metadata and validation
metrics.

### Changes

1. Added CLIP stage metadata helpers in `src/clip.rs`:
   - `ClipVitLinearStage`
   - `ClipVitLinearStageMeta`
   - `ClipVitLayerLinearNames::tensor_name_for_stage`
   - `ClipVitLayerLinearNames::stage_metas`
   - `ClipSafeTensorFile::clip_vit_layer_stage_metas`
2. Exported new clip helpers from `src/lib.rs` for example use.
3. Added `examples/clip_vit_block_tpu_pipeline.rs`:
   - loads layer tensors from SafeTensors
   - quantizes all six stages
   - patches three template classes (`768x768`, `768x3072`, `3072x768`)
   - prepares six TPU stages and executes a full linear chain over row batches
   - reports stage timing and CPU-accumulator vs TPU affine fit metrics.
4. Added docs entry:
   - `docs/clip_vit_block_tpu_pipeline.md`
5. Updated `README.md` example lists with the new pipeline command.

### Validation

1. `cargo check --example clip_vit_block_tpu_pipeline`
   - passed
2. `cargo test -q clip::tests::`
   - compile phase reached linking and failed in this environment due to
     missing `libedgetpu`/`libtensorflowlite_c` shared libs.

### Quantization improvements

1. Added configurable CLIP quantization API in `src/clip.rs`:
   - `LinearQuantConfig { qmax, clip_percentile }`
   - `quantize_linear_out_in_to_row_major_qi8_with_config(...)`
2. Extended `QuantizationInfo` with:
   - `clipped_max_abs`
   - `clip_percentile`
   - `clipped_values`
3. Kept compatibility wrapper:
   - `quantize_linear_out_in_to_row_major_qi8(...)` now forwards to the
     configurable API with `clip_percentile=100`.
4. Upgraded `examples/clip_vit_block_tpu_pipeline.rs`:
   - default `qmax` changed from `127` to `32`
   - added `--clip-percentile`
   - added `--auto-qmax A,B,C` per-stage auto-tune based on calibration
     correlation/RMSE
   - stage setup now prints clipping stats and selected `qmax` values.
5. Updated docs and README command examples to show recommended CLIP settings.
6. Pi5 checkpoint sweeps for CLIP ViT-B/32 (`12` layers, `8` rows, `3` runs):
   - legacy `qmax=127`: stage-mean correlation around `0.88` (from prior run)
   - tuned `qmax=32`: `72` stage checks, mean corr `0.997553`, min corr
     `0.912842` (`fc2`), with unchanged pipeline latency (~`38 ms/layer`)

### Full CLIP forward wiring

1. Added `examples/clip_vit_full_forward.rs`:
   - full vision path: patch embedding conv + pre/post layernorm + 12
     transformer blocks + projection head
   - Coral-backed linears (`q/k/v/o/fc1/fc2`) with calibrated affine
     dequantization per stage
   - optional embedding dump (`--out-f32le`, `--out-norm-f32le`) and optional
     reference compare (`--reference-f32le`)
2. Added `docs/clip_vit_full_forward.md`.
3. Added helper script `tools/clip_hf_reference.py` (HF Transformers reference
   embedding generator from `f32le` image tensor).
4. Pi5 validation with real checkpoint:
   - command used `--max-layers 12 --weight-qmax 32 --act-qmax 32`
   - stage calibration stayed strong across all layers (`corr` mostly
     `0.9998+`)
   - end-to-end timing:
     - `prepare_ms=39552.239`
     - `forward_ms=6358.528`
     - `total_ms=45910.767`
   - artifacts:
     - `/home/rpc/clip-traces/clip-full-forward-20260222T130016Z/`
5. HF compare on same deterministic input tensor:
   - artifact:
     - `/home/rpc/clip-traces/clip-full-forward-hf-compare-20260222T130159Z/`
   - raw embedding: cosine `0.73293731`, MAE `0.26041435`, RMSE `0.34962000`
   - normalized embedding: cosine `0.73293731`, MAE `0.02508840`,
     RMSE `0.03229882`

### Batched template speed/fidelity sweep (Pi5)

1. Added batch-size plumbing for dense template generation/execution:
   - `PreparedDenseGemm` now discovers and uses interpreter batch capacity.
   - `tools/generate_dense_quant_tflite.py` supports `--batch-size`.
   - `tools/dense_template_pipeline.sh` supports `--batch-size` end-to-end.
2. Pi5 compiler note:
   - local Pi compiler binary under `~/.local/bin/edgetpu_compiler` was x86-only.
   - workaround used: compile templates on x86 workstation, copy to Pi5.
3. Controlled CLIP full-forward sweep on Pi5 (same fixed input/reference as above):
   - base artifacts:
     - `/home/rpc/clip-traces/clip-batch-sweep/runs-20260222T133711Z/`
     - summary: `/home/rpc/clip-traces/clip-batch-sweep/runs-20260222T133711Z/SUMMARY.txt`
   - extra batch-5 confirmation:
     - `/home/rpc/clip-traces/clip-batch-sweep/runs-extra-20260222T134351Z-b5/`
4. Results (`forward_ms`, normalized cosine vs HF):
   - `batch=1`: `6486.240 ms`, `cos=0.73293728`
   - `batch=4`: `5902.632 ms`, `cos=0.73293728`
   - `batch=5`: `5876.642 ms`, `cos=0.36591250`
   - `batch=8`: `5872.174 ms`, `cos=0.36591250`
   - `batch=16`: `5852.587 ms`, `cos=0.34255037`
   - `batch=28`: `5708.579 ms`, `cos=0.43851820`
5. Conclusion:
   - `batch=4` is the highest tested setting that preserved fidelity on Pi5.
   - `batch>=5` enters a quality cliff with only marginal additional speedup.
   - recommended operational point for CLIP full-forward on Pi5: `batch=4`.

### Function-Gemma layer bring-up (real checkpoint on Pi5)

1. Added Function-Gemma safetensors support in Rust:
   - `src/function_gemma.rs`
   - BF16/F16/F32 tensor decode to `f32`
   - Gemma-style stage mapping:
     - `q/k/v/o/gate/up/down`
     - `model.layers.<N>.{self_attn,mlp}.*.weight`
   - inferred dims from layer tensors (`hidden`, `q_out`, `kv_out`, `mlp_hidden`)
2. Exported new API from crate root in `src/lib.rs`:
   - `FunctionGemmaSafeTensorFile`
   - `FunctionGemmaLinearStage`
   - `FunctionGemmaLayerLinearNames`
   - `FunctionGemmaLinearStageMeta`
   - `FunctionGemmaDims`
   - `FunctionGemmaError`
3. Added new example:
   - `examples/function_gemma_layer_tpu_probe.rs`
   - flow: load safetensors stage -> quantize -> patch template -> execute ->
     fit CPU accumulator affine (`alpha/beta/corr/mae/rmse`)
4. Added docs:
   - `docs/function_gemma_layer_tpu_probe.md`
5. Pi5 checkpoint + stage sweep:
   - model: `distil-labs/distil-home-assistant-functiongemma`
   - local model path: `/home/rpc/functiongemma-models/model.safetensors`
   - trace output: `/home/rpc/clip-traces/functiongemma-layer-probe-20260222T140422Z`
   - inferred dims:
     - `hidden=640`
     - `q_out=1024`
     - `kv_out=256`
     - `mlp_hidden=2048`
6. Stage results (`runs=20`, `qmax=32`, `clip_percentile=100`):
   - `q` (`640x1024`): `avg_ms=0.459`, `corr=0.999537`
   - `k` (`640x256`): `avg_ms=0.331`, `corr=0.994594`
   - `v` (`640x256`): `avg_ms=0.329`, `corr=0.999965`
   - `o` (`1024x640`): `avg_ms=0.627`, `corr=0.999872`
   - `gate` (`640x2048`): `avg_ms=0.634`, `corr=0.999723`
   - `up` (`640x2048`): `avg_ms=0.821`, `corr=0.999793`
   - `down` (`2048x640`): `avg_ms=0.675`, `corr=0.999740`

### Function-Gemma embedding + LM-head sanity (CPU first)

1. Extended `FunctionGemmaSafeTensorFile` with embedding helpers:
   - `embedding_dims()`
   - `token_embedding_row_f32(token_id)`
   - `lm_head_topk_from_hidden(hidden_state, topk)`
2. Added example:
   - `examples/function_gemma_lm_head_sanity.rs`
   - docs: `docs/function_gemma_lm_head_sanity.md`
3. Pi5 run:
   - output: `/home/rpc/clip-traces/functiongemma-lmhead-sanity-20260222T142358Z/run.log`
   - command:
     - `cargo run --example function_gemma_lm_head_sanity -- /home/rpc/functiongemma-models/model.safetensors 42 10`
   - results:
     - `embedding_lookup_ms=2.887`
     - `lm_head_top10_ms=14213.630`
     - top-1 token id was self token (`42`) with score `1.062339`

## 2026-02-21

### Objective

Bring up a modern local stack for Coral USB (`libedgetpu` + TFLite C), recover
delegate/inference functionality, and start protocol reverse-engineering from
`*_edgetpu.tflite` and USB traffic.

### Bring-up milestones

1. Implemented package extraction tooling:
   - `tools/extract_edgetpu_package.py`
   - Verified `DWN1` package extraction on
     `mobilenet_v1_1.0_224_quant_edgetpu.tflite`.
2. Added capture tooling:
   - `tools/usbmon_capture.sh` (kernel usbmon)
   - `tools/usb_syscall_trace.sh` (strace fallback)
   - `docs/usb_tracing.md`
3. Added local bootstrap tooling:
   - `tools/bootstrap_arch_stack.sh`
   - Built local libs into `~/.local/lib`.

### Key failures and fixes

1. **Missing runtime libs at link time**
   - Symptom: `unable to find library -ledgetpu` / `-ltensorflowlite_c`.
   - Fix: local-prefix build via `tools/bootstrap_arch_stack.sh`.
2. **Arch/AUR dependency drift**
   - Symptom: `flatbuffers=24.3.25` unavailable; TF/libedgetpu mismatch.
   - Fix: patched TF-generated FlatBuffers version asserts during local build.
3. **TensorFlow hermetic python mismatch**
   - Symptom: TF 2.18 requested lockfiles for Python 3.14 (unsupported).
   - Fix: forced `HERMETIC_PYTHON_VERSION` and `TF_PYTHON_VERSION` (default 3.12).
4. **Corrupted/empty embedded DFU firmware in libedgetpu build**
   - Symptom: delegate creation failed with `Invalid DFU image file`.
   - Root cause: missing `xxd`; firmware arrays generated empty.
   - Fixes:
     - installed `xxd`
     - added hard checks in `tools/bootstrap_arch_stack.sh` for `xxd`
       presence and non-empty `usb_latest_firmware.h`.
5. **`sudo` capture command lost user-local libs**
   - Symptom: `usbmon_capture.sh` run under sudo linked against `/root/.local/lib`.
   - Fix: `tools/usbmon_capture.sh` now executes traced command as `$SUDO_USER`
     by default (override via `USBMON_RUN_COMMAND_AS_ROOT=1`).

### API behavior fix

1. **Stale `CoralDevice` VID/PID reporting**
   - Symptom: `CoralDevice::new()` reported static DFU IDs even when device was
     already in initialized mode (`18d1:9302`).
   - Fix: `src/lib.rs` constructors now derive IDs from live `find_coral_devices()`.

### Verified functional outcomes

1. `cargo run --example delegate_usage`
   - delegate creation successful
   - live VID/PID reporting correct
2. `cargo run --example tflite_test`
   - TFLite + EdgeTPU integration successful
3. `cargo run --example tflite_standard_example`
   - standard TFLite interpreter path successful
4. `cargo run --example inference_benchmark -- models/mobilenet_v1_1.0_224_quant.tflite 100 10`
   - avg latency around 18 ms
5. `cargo run --example inference_benchmark -- models/mobilenet_v1_1.0_224_quant_edgetpu.tflite 30 5`
   - avg latency around 2.6-2.8 ms

### Reverse-engineering artifacts

1. Successful syscall traces:
   - `traces/usb-syscall-20260221T085116Z/strace-usb-20260221T085116Z.log`
   - `traces/usb-syscall-20260221T085310Z/strace-usb-20260221T085310Z.log`
2. Successful usbmon capture:
   - `traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log`
   - `traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.summary.txt`

### Reverse-engineering findings (current hypotheses)

1. Capture is healthy:
   - `total_lines=690`, balanced `S/C` (`345/345`), `command_exit=0`.
2. Dominant device traffic is on USB device `005`:
   - `Bo=296`, `Bi=160`, `Co=144`, `Ci=60`, `Ii=2`.
3. Pre-inference load burst likely corresponds to program/model upload:
   - bulk-out completed bytes before first input-sized transfer:
     `4,697,104` bytes.
4. Inference loop signature appears stable and exact:
   - repeated `35` times (5 warmup + 30 measured):
     - `Bo 225824` -> `Bo 150528` -> `Bi 1008`
   - timing:
     - `Bo225824 -> Bo150528`: ~`0.345 ms` avg
     - `Bo150528 -> Bi1008`: ~`1.477 ms` avg
     - interval between `Bo150528`: ~`2.637 ms` avg
5. `Bo 150528` matches input tensor size (`224*224*3` bytes).
6. `Bi 1008` is consistent with output logits buffer (1000-class model + alignment).
7. Negative usbmon statuses are expected async URB behavior:
   - mostly `-115` (`EINPROGRESS`) and teardown/cancel `-2` (`ENOENT`).

### New analysis tool

1. Added `tools/usbmon_phase_report.py`:
   - `report` subcommand for phase timeline and cycle metrics
   - `diff` subcommand for run-to-run comparison
2. Example:
   - `python3 tools/usbmon_phase_report.py report <usbmon.log> --bus 4 --device 005`
   - `python3 tools/usbmon_phase_report.py diff <run_a.log> <run_b.log> --bus 4 --device 005`

### Open questions

1. Exact semantic meaning of the `Bo 225824` payload per inference.
2. Mapping of vendor control register addresses (`wValue`) to hardware functions.
3. Relationship between `Bi ep2 size=16` status packets and inference completion.

## 2026-02-21 (RE matrix batch)

### Matrix output directory

- `traces/re-matrix-20260221T092342Z`

### Runs executed

1. `R1_basic_usage` (`usb_syscall_trace`)
2. `R2_edgetpu_delegate_smoke` (`usb_syscall_trace`)
3. `R3_simple_delegate` (`usb_syscall_trace`)
4. `R4_tflite_standard_cpu_only` (`usb_syscall_trace`)
5. `R5_infer_plain_short` (`usb_syscall_trace`)
6. `R6_infer_plain_short_repeat` (`usb_syscall_trace`)
7. `R7_infer_plain_long` (`usb_syscall_trace`)
8. `R8_infer_edgetpu_short` (`usb_syscall_trace`)
9. `R9_infer_edgetpu_long` (`usb_syscall_trace`)
10. `R10_tflite_test` (`usb_syscall_trace`)
11. `R11b_cpu_vs_edgetpu_mvp` (`usb_syscall_trace`)

All runs exited with `command_exit=0`.

### Batch artifacts

1. `traces/re-matrix-20260221T092342Z/RE_MATRIX_SUMMARY.md`
2. `traces/re-matrix-20260221T092342Z/USBMON_BASELINE_PHASE.txt`
3. `traces/re-matrix-20260221T092342Z/USBMON_BASELINE_PHASE.json`
4. `traces/re-matrix-20260221T092342Z/USBMON_RUNS_TO_EXECUTE.txt`
5. `traces/re-matrix-20260221T092342Z/USBMON_INTERACTIVE_SUMMARY.md`
6. Per-run trace logs and summaries under each `R*/` directory.

### Batch findings

1. CPU-only baseline (`R4`) generated zero USB syscalls, confirming trace
   filtering and baseline behavior.
2. Plain quantized model runs (`R5`/`R6`/`R7`) produced near-identical USB
   ioctl counts (`SUBMITURB=105`, `REAPURBNDELAY=201`) despite run-count
   changes.
3. EdgeTPU-compiled model run (`R9`) showed significantly higher USB activity
   (`SUBMITURB=324`, `REAPURBNDELAY=565`) and much lower latency (`avg ~2.7 ms`),
   consistent with strong accelerator engagement.
4. Delegate-only path parity was observed between C and Rust:
   `R2` and `R3` both showed the same control-heavy bring-up class with
   successful delegate create/free.
5. CPU-vs-EdgeTPU harness with the plain quantized model (`R11b`) showed no
   acceleration benefit in this environment (`edgetpu_int8` p50 slower than
   `cpu_int8`), consistent with partial/off-target offload behavior for
   non-EdgeTPU-compiled graphs.
6. Interactive usbmon captures (`U1`..`U4`) confirmed packet-level behavior:
   - `U1`/`U2` delegate traces are control-heavy and bulk-free
   - `U3` plain-model inference remains bulk-free and indistinguishable from setup
   - `U4` EdgeTPU-model inference reproduces full bulk cycle signature
     (`Bo225824 -> Bo150528 -> Bi1008`, count `35`) and `4,697,104` bytes
     pre-inference upload burst.
7. Added register-map extraction pass:
   - tool: `tools/usbmon_register_map.py`
   - artifacts: `REGISTER_MAP_MATRIX.*` and per-run `*_REGISTER_REPORT.*`
   - finding: control/register address-op counts are invariant across
     `U1/U2/U3/U4/baseline`; differentiator is bulk path activation.

### Notes

1. This matrix uses `usb_syscall_trace` for automation (unprivileged).
2. Kernel usbmon remains the primary packet-level source; latest successful
   baseline capture remains:
   - `traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log`
3. An initial `R11` attempt failed only due a CSV path typo (permission on
   `/R11_mvp_results.csv`); corrected in `R11b` with successful completion.
4. Candidate register-map write-up: `docs/usb_register_map_candidates.md`.

## 2026-02-21 (RE continuation: invoke scaling + payload signatures)

### Additional runs executed

1. `R12_infer_edgetpu_5_1` (`usb_syscall_trace`)
2. `R13_infer_edgetpu_10_0` (`usb_syscall_trace`)
3. `R14_infer_edgetpu_20_5` (`usb_syscall_trace`)
4. `R15_infer_plain_1_0` (`usb_syscall_trace`)

All runs exited with `command_exit=0`.

### New artifacts

1. `traces/re-matrix-20260221T092342Z/USB_IOCTL_SCALING.md`
2. `traces/re-matrix-20260221T092342Z/U1_BULK_SIG.{txt,json}`
3. `traces/re-matrix-20260221T092342Z/U2_BULK_SIG.{txt,json}`
4. `traces/re-matrix-20260221T092342Z/U3_BULK_SIG.{txt,json}`
5. `traces/re-matrix-20260221T092342Z/U4_BULK_SIG.{txt,json}`
6. `traces/re-matrix-20260221T092342Z/BASE_BULK_SIG.{txt,json}`
7. `traces/re-matrix-20260221T092342Z/USBMON_BULK_SIGNATURE_SUMMARY.md`
8. `traces/re-matrix-20260221T092342Z/DIFF_BASE_vs_U4_BULK_SIG.txt`

### New tooling

1. Added `tools/usbmon_bulk_signature.py`:
   - extracts top bulk payload prefix signatures from usbmon logs
   - classifies signatures by phase (`pre_loop`, `loop`, `post_loop`)
   - surfaces recurring per-invoke command/header candidates

### New findings

1. EdgeTPU model USB ioctl counts scale linearly with total invokes:
   - `SUBMITURB = 114 + 6 * total_invokes`
   - `REAPURBNDELAY = 215 + 10 * total_invokes`
2. Plain model USB ioctl counts remain flat at:
   - `SUBMITURB=105`, `REAPURBNDELAY=201`
   across `1`, `6`, and `55` total invokes.
3. `U4` and baseline usbmon captures are signature-identical for normalized
   bulk payload headers (`DIFF_BASE_vs_U4_BULK_SIG.txt` empty).
4. Strong per-invoke loop markers in `U4/baseline`:
   - `Bo 8: 20720300 00000000`
   - `Bo 8: 004c0200 01000000`
   - `Bo 225824: 800f0080 dc000000`
   - `Bo 150528: 00010203 04050607`
5. `U1/U2/U3` remain bulk-submit-free (interrupt polling only), reinforcing
   that loop markers are EdgeTPU-compiled-model specific in this matrix.

### External research snapshot

1. Added source-backed constraints summary:
   - `docs/external_research_2026-02-21.md`
2. Key conclusion:
   - practical path is constrained TFLite graph compilation (int8/static/op-set),
     not arbitrary custom-kernel USB dispatch via public APIs.

### Executable-vs-transport correlation pass

1. Added note:
   - `docs/usb_executable_transport_correlation.md`
2. New finding:
   - `U4` bulk signature markers (`20720300`, `800f0080dc...`, `501c0000`,
     `800f000c07...`) are present at specific offsets inside extracted
     serialized executables (`exec0` vs `exec1` split), strengthening the
     hypothesis that loop/preload transport headers are compiled artifact data.

### Cross-model invoke scaling extension

1. Added additional strace runs:
   - `R16_infer_edgetpu_bird_10_2`
   - `R17_infer_plain_bird_10_2`
   - `R18_infer_edgetpu_bird_1_0`
   - `R19_infer_edgetpu_bird_20_5`
   - `R20_infer_edgetpu_bird_5_1`
   - `R21_infer_edgetpu_bird_30_5`
   - `R22_infer_plain_bird_1_0`
   - `R23_infer_edgetpu_bird_10_0`
   - `R24_infer_edgetpu_bird_1_2`
   - `R25_infer_edgetpu_bird_10_2_repeat`
   - `R26_infer_inception_edgetpu_1_0`
   - `R27_infer_inception_edgetpu_10_0`
   - `R28_infer_inception_plain_10_0`
   - `R29_infer_inception_plain_1_0`
   - `R30_infer_inception_edgetpu_20_0`
   - `R31_infer_inception_plain_20_0`
2. Added summary note:
   - `docs/usb_invoke_scaling_by_model.md`
3. Added automation tool:
   - `tools/strace_usb_scaling.py`
   - emits per-model linear fits and residuals from `R*/` strace summaries
4. New finding:
   - EdgeTPU ioctl scaling differs by model:
     - model A (`mobilenet_v1..._edgetpu`): exact `+6 submit / +10 reap` per invoke
     - model B (`mobilenet_v2...inat_bird..._edgetpu`): exact
       `+8 submit / +12 reap` on primary fit, with one outlier run (`R16`)
       that did not reproduce in `R25` repeat
     - model C (`inception_v1..._edgetpu`): approximately
       `+10 submit / +15 reap` per invoke (small jitter)
   - plain-model path remains flat at setup-level USB counts in this environment.
5. Added privileged packet-capture next-step matrix:
   - `docs/next_usbmon_capture_matrix.md`

## 2026-02-21 (packet-level validation from new usbmon captures)

### New privileged captures

1. `traces/usbmon-20260221T103521Z-bus4/usbmon-bus4-20260221T103521Z.log`
   - `mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite` (`warmup=5`, `runs=20`)
2. `traces/usbmon-20260221T103552Z-bus4/usbmon-bus4-20260221T103552Z.log`
   - `inception_v1_224_quant_edgetpu.tflite` (`warmup=0`, `runs=20`)
3. `traces/usbmon-20260221T103631Z-bus4/usbmon-bus4-20260221T103631Z.log`
   - `inception_v1_224_quant.tflite` (`warmup=0`, `runs=20`)

### New analysis artifacts

1. `traces/re-matrix-20260221T092342Z/U5_*`
2. `traces/re-matrix-20260221T092342Z/U6_*`
3. `traces/re-matrix-20260221T092342Z/U7_*`
4. `traces/re-matrix-20260221T092342Z/REGISTER_MAP_MATRIX_U1_U7.{md,json}`
5. `traces/re-matrix-20260221T092342Z/USBMON_PACKET_VALIDATION_20260221T1035Z.md`

### New findings

1. Bird EdgeTPU run (`U5`) shows repeating 3-stage loop:
   - `Bo 261920 -> Bo 150528 -> Bo 10224 -> Bi 968` (`count=25`)
2. Inception EdgeTPU run (`U6`) shows a distinct 3-stage loop:
   - `Bo 254656 -> Bo 150528 -> Bo 393664 -> Bi 1008` (`count=20`)
3. Inception plain run (`U7`) remains bulk-submit-free (control/interrupt only).
4. Control/register address-op counts remain invariant across `U1..U7`; model
   differences are isolated to bulk transport choreography.
5. Packet-level data now confirms the syscall-level model classes:
   - model A: `+6/+10`
   - model B: `+8/+12`
   - model C: about `+10/+15`

## 2026-02-21 (dedicated 3-stage signature parser)

### New tooling

1. Added `tools/usbmon_three_stage_signature.py`:
   - auto-discovers repeated `Bo -> Bo -> Bo -> Bi` completion candidates
   - extracts non-overlapping cycles with per-leg timing stats
   - reports inter-stage gap-event counts to surface hidden intermediate steps
   - supports explicit pattern forcing (`--bo-1/--bo-2/--bo-3/--bi-out`)

### New analysis artifacts

1. `traces/re-matrix-20260221T092342Z/U4_STAGE3_SIG.{txt,json}`
2. `traces/re-matrix-20260221T092342Z/U5_STAGE3_SIG.{txt,json}`
3. `traces/re-matrix-20260221T092342Z/U6_STAGE3_SIG.{txt,json}`
4. `traces/re-matrix-20260221T092342Z/U6_STAGE3_SIG_254K_150K_393K.{txt,json}`
5. `traces/re-matrix-20260221T092342Z/U7_STAGE3_SIG.{txt,json}`

### New findings

1. `U5` auto-detected 3-stage signature:
   - `Bo 261920 -> Bo 150528 -> Bo 10224 -> Bi 968` (`count=25`)
2. `U6` auto-detected 3-stage tail signature:
   - `Bo 150528 -> Bo 393664 -> Bo 103200 -> Bi 1008` (`count=20`)
3. Explicit `U6` match for prior hypothesis
   (`Bo 254656 -> Bo 150528 -> Bo 393664 -> Bi 1008`) still yields `count=20`,
   but with `gap_events_3_4=2`, indicating additional completions between
   `Bo 393664` and `Bi 1008`.
4. `U7` and `U4` do not auto-select a 3-stage signature under
   `min_count=3` (expected for plain path and 2-stage baseline class).

## 2026-02-21 (schema-aware executable parser)

### New tooling

1. Added `tools/parse_edgetpu_executable.py`:
   - parses `serialized_executable_*.bin` directly as FlatBuffers
   - extracts instruction chunks, relocation metadata (`field_offsets`), layer
     metadata, parameter sizes, and DMA-hint summaries
   - scans known transport-marker byte patterns with offsets/counts
2. Added vendored schema reference:
   - `docs/schema/libedgetpu_executable.fbs`
   - provenance: `docs/schema/README.md`

### New analysis artifacts

1. `traces/re-matrix-20260221T092342Z/EXEC_PARSE_MOBILENET_V1.{txt,json}`
2. `traces/re-matrix-20260221T092342Z/EXEC_PARSE_BIRD_V2.{txt,json}`
3. `traces/re-matrix-20260221T092342Z/EXEC_PARSE_INCEPTION_V1.{txt,json}`

### New findings

1. Model A (`mobilenet_v1..._edgetpu`):
   - `EXECUTION_ONLY` chunk: `225824`
   - `PARAMETER_CACHING` chunk: `7248`
   - `PARAMETER_CACHING` params: `4464000`
2. Model B (`mobilenet_v2...bird..._edgetpu`):
   - `EXECUTION_ONLY` chunks: `261920`, `10224`
   - `PARAMETER_CACHING` chunk: `10064`
   - `PARAMETER_CACHING` params: `3947392`
3. Model C (`inception_v1..._edgetpu`):
   - `EXECUTION_ONLY` chunks: `254656`, `103200`
   - `EXECUTION_ONLY` params: `393664`
   - `PARAMETER_CACHING` chunk: `9680`
   - `PARAMETER_CACHING` params: `6581440`
4. Cross-correlation to usbmon loops:
   - model A loop stage `Bo225824` maps to exec0 chunk
   - model B loop stages `Bo261920` + `Bo10224` map to exec0 chunks
   - model C loop stages `Bo254656` + `Bo393664` + `Bo103200` map to
     exec0 chunk + exec0 params + exec0 chunk respectively, with `Bo150528`
     as input activation transfer
5. Relocation metadata confirms address patch points are explicit:
   - `BASE_ADDRESS_PARAMETER`, `BASE_ADDRESS_SCRATCH`,
     `BASE_ADDRESS_INPUT_ACTIVATION`, `BASE_ADDRESS_OUTPUT_ACTIVATION`
   appear in `EXECUTION_ONLY` executable field offsets, with layer names carried
   on input/output relocations.

## 2026-02-21 (tensorizer MVP: in-place parameter patching)

### New tooling

1. Added `tools/tensorizer_patch_edgetpu.py`:
   - `inspect` subcommand finds package/executable layout and absolute
     `Executable.parameters` byte ranges inside compiled `*_edgetpu.tflite`
   - `patch` subcommand rewrites selected parameter regions in-place while
     preserving file size and FlatBuffer offsets
   - patch modes: `zero`, `byte`, `ramp`, `xor`, `random`

### New docs

1. Added `docs/tensorizer_mvp.md`:
   - end-to-end commands and observed results for model A tensorizer sanity test

### New findings

1. Model A patch target located:
   - `exe[1]` (`PARAMETER_CACHING`) parameter region
   - absolute range `[13276, 4477276)` (`4464000` bytes)
2. Baseline inference (`runs=5`, `warmup=1`):
   - top output `index=905 score=38`
3. Patched inference (three patterns: `zero`, `ramp`, `xorff`):
   - model still runs on EdgeTPU
   - top output collapses to `index=1000 score=0`
4. Interpretation:
   - parameter payload mutation is sufficient to alter behavior while keeping
     compiled execution stream runnable, validating the tensorizer direction.

## 2026-02-21 (single-op Dense template tensorizer path)

### New tooling

1. Added `tools/bootstrap_edgetpu_compiler.sh`:
   - installs `edgetpu_compiler` from Coral apt package into local prefix
   - installs both wrapper and runtime bundle (`edgetpu_compiler_bin`)
2. Added `tools/generate_dense_quant_tflite.py`:
   - generates a single-layer Dense quantized `.tflite` with deterministic
     kernel initialization
   - supports `identity`, `permutation`, `ones`, `random_uniform`
3. Added `tools/dense_template_pipeline.sh`:
   - end-to-end `uv` flow: generate -> compile -> extract -> parse -> inspect
   - optional parameter patch generation
4. Added `examples/inference_dump.rs`:
   - single-invoke deterministic input/output dump (`zeros|ones|ramp|alt`)
   - prints full int8 output preview for tensorizer verification

### New docs

1. Added `docs/tensorizer_dense_template.md`.
2. Updated `README.md` with compiler bootstrap + dense template workflow.
3. Updated `docs/tensorizer_mvp.md` to link the single-op template track.

### Toolchain compatibility finding

1. Dense templates generated with `tensorflow-cpu==2.18.0` failed
   `edgetpu_compiler` acceptance in this environment.
2. Dense templates generated with:
   - `python 3.9`
   - `tensorflow-cpu 2.10.1`
   - `numpy 1.23.5`
   compile successfully and map `FULLY_CONNECTED` to EdgeTPU.

### New analysis artifacts

1. `traces/dense-template-20260221T120206Z/PIPELINE_SUMMARY.txt`
2. `traces/dense-template-20260221T120206Z/exec_parse.{txt,json}`
3. `traces/dense-template-20260221T120206Z/tensorizer_inspect.{txt,json}`
4. `traces/dense-template-20260221T120206Z/inference_dump_*.log`
5. `traces/dense-template-20260221T120206Z/inference_benchmark_*.log`
6. `traces/usb-syscall-20260221T120304Z/*.summary.txt`
7. `traces/usb-syscall-20260221T120317Z/*.summary.txt`

### New findings

1. Single-op compiled model (`dense_256x256_quant_edgetpu.tflite`) contains:
   - `EXECUTION_ONLY` executable with instruction chunk `4112`
   - `PARAMETER_CACHING` executable with params `65536` bytes
2. Baseline deterministic runtime output (`inference_dump`, `ramp` input):
   - output vector preserves ramp `[-128..127]` for 256 lanes
3. Parameter patch impact (same instruction stream):
   - `zero`: near-constant high saturated output (`127`-dominant)
   - `ramp`: periodic transformed pattern
   - `xorff`: descending transformed pattern
4. Latency remains stable despite payload mutation:
   - baseline avg `0.210 ms` (`runs=30`, `warmup=5`)
   - patched_zero avg `0.210 ms`
5. USB syscall profile is transport-invariant across baseline vs patched:
   - `SUBMITURB` stayed `320`
   - `REAPURBNDELAY` differed only slightly (`456` vs `466`)

## 2026-02-21 (single-hot layout recovery + structured matrix patch)

### New tooling

1. Added `tools/dense_layout_probe.py`:
   - generates/compiles repeated single-hot Dense probes
   - extracts `PARAMETER_CACHING` payload bytes
   - diffs against reference probe and emits `(row,col)->offset` candidates
2. Added `tools/dense_template_matrix_patch.py`:
   - patches Dense template payload with structured matrix modes
   - supports `identity`, `zero`, `single_hot`, `shift_plus1`, `shift_minus1`
   - uses recovered offset map for deterministic row/col addressing
3. Updated `tools/generate_dense_quant_tflite.py`:
   - added init modes `zero` and `single_hot`
   - added `--hot-row/--hot-col/--hot-value`

### New docs

1. Added `docs/dense_layout_probe.md`.
2. Updated `docs/tensorizer_dense_template.md` with follow-on path after layout
   recovery.
3. Updated `README.md` tool/docs indexes for layout-probe and matrix patch tools.

### New analysis artifacts

1. `traces/dense-layout-probe-20260221T121033Z/*`
2. `traces/dense-layout-probe-20260221T121109Z/*`
3. `traces/dense-layout-probe-20260221T121249Z/*`
4. `traces/dense-layout-probe-20260221T121345Z/*`
5. `traces/dense-layout-probe-20260221T121612Z/*`
6. `traces/dense-template-20260221T120206Z/patch_shift_plus1.json`
7. `traces/dense-template-20260221T120206Z/patch_shift_minus1.json`
8. `traces/dense-template-20260221T120206Z/inference_dump_shift_plus1.log`
9. `traces/dense-template-20260221T120206Z/inference_dump_shift_minus1.log`

### New findings

1. Single-hot payload behavior:
   - background byte: `128`
   - active byte: `255`
   - each probe changes exactly one payload byte (`65536`-byte region)
2. Recovered offset map for Dense(256x256):
   - `offset = (col//64)*16384 + (row//64)*4096 + ((row%64)//4)*256 + (col%64)*4 + (row%4)`
3. Formula validation:
   - matched all sampled points across multi-run probe sets (small grid,
     boundary points, random points)
4. Structured runtime verification:
   - `shift_plus1` patch rotates ramp output by +1
   - `shift_minus1` patch rotates ramp output by -1
   - confirms recovered mapping drives expected `W @ x`-style behavior.

## 2026-02-21 (quantization byte mapping verification)

### New tooling

1. Added `tools/dense_quant_value_probe.py`:
   - compiles repeated single-hot float-value probes
   - extracts weight tensor quant params (`scale`, `zero_point`)
   - compares expected signed-int8 quantization vs raw quant bytes vs compiled
     payload bytes at recovered offset

### New analysis artifacts

1. `traces/dense-quant-value-probe-20260221T122533Z/value_probe.{txt,json}`
2. `traces/dense-template-20260221T120206Z/value_byte_sweep/*`

### New findings

1. Weight tensor quantization is signed `INT8` with `zero_point=0`.
2. Compiled payload byte mapping is:
   - `payload_u8 = (q_i8 + 128) & 0xff`
3. Sweep verification matched exactly for all tested values:
   - quant `127` -> compiled `255`
   - quant `0` -> compiled `128`
   - quant `-127` (raw byte `129`) -> compiled `1`
4. Fixed-template byte sweeps (same compiled template, varying patched bytes)
   showed linear lane-0 output progression around center `128`, validating
   direct byte-level control for tensorizer weight encoding.

## 2026-02-22 (Rust-native GEMM template API)

### New library capabilities

1. Added in-memory interpreter constructor:
   - `CoralInterpreter::new_from_memory(&[u8], &EdgeTPUDelegate)`
2. Added Rust-native DWN1/FlatBuffer inspection for compiled templates:
   - package scan by `DWN1` marker
   - executable parsing and `parameters` vector region discovery
   - automatic preference for `PARAMETER_CACHING` executable payload
3. Added Dense tensorizer types:
   - `DenseGemm256Template`
   - `GemmTemplate256`
4. Added public constants and mapping helpers:
   - `DENSE_GEMM256_DIM`
   - `DENSE_GEMM256_WEIGHT_COUNT`
   - `DENSE_GEMM256_WEIGHT_BYTES`
   - `DENSE_GEMM256_ZERO_POINT`
   - `dense_256_param_offset(row, col)`

### New API behavior

1. `DenseGemm256Template` can now:
   - locate and validate the `65536`-byte Dense parameter payload
   - encode/decode quantized weights (`q_i8 <-> payload_u8`)
   - patch weights by full matrix or structured modes:
     - identity
     - shift plus/minus one
     - diagonal vector
2. `GemmTemplate256` can execute patched templates directly on EdgeTPU:
   - creates interpreter from patched bytes in memory
   - copies int8 input vector (`256`)
   - invokes delegate path and returns int8 output vector (`256`)

### New Rust example

1. Added `examples/gemm_int8.rs`:
   - loads compiled Dense template
   - patches matrix entirely in Rust
   - runs inference via `GemmTemplate256::execute`
   - compares observed output against expected output for:
     - identity
     - shift plus one
     - shift minus one

### Validation

1. `cargo fmt`
2. `cargo check --lib`
3. `cargo check --tests`
4. `cargo check --example gemm_int8`

### Hardware verification (Rust-only path)

Environment:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
```

Commands and outcomes:

1. `cargo run --example gemm_int8 -- traces/dense-template-20260221T120206Z/dense_256x256_quant_edgetpu.tflite shift_plus1 ramp`
   - output matched expected shift-by-1 exactly
   - `mismatches(|delta|>1)=0`, `max_abs_delta=0`
2. `cargo run --example gemm_int8 -- traces/dense-template-20260221T120206Z/dense_256x256_quant_edgetpu.tflite identity ramp`
   - output matched expected identity exactly
   - `mismatches(|delta|>1)=0`, `max_abs_delta=0`
3. `cargo run --example gemm_int8 -- traces/dense-template-20260221T120206Z/dense_256x256_quant_edgetpu.tflite shift_minus1 ramp`
   - output matched expected shift-by-minus-1 exactly
   - `mismatches(|delta|>1)=0`, `max_abs_delta=0`
4. `cargo run --example gemm_int8 -- traces/dense-template-20260221T120206Z/dense_256x256_quant_edgetpu.tflite diag_ramp ramp`
   - completed successfully with nontrivial transformed output profile

## 2026-02-22 (dimension scaling + prepared execution path)

### New API additions

1. Added dimension-aware mapping helper:
   - `dense_param_offset(input_dim, output_dim, row, col)`
2. Added dimension-aware template/executor types:
   - `DenseGemmTemplate`
   - `PreparedDenseGemm`
3. Added interpreter-reuse path for fixed 256 templates:
   - `PreparedGemm256`
   - `GemmTemplate256::prepare(...)`
4. Added dynamic Rust example:
   - `examples/gemm_int8_dynamic.rs`
5. Updated `examples/gemm_int8.rs`:
   - optional `runs` arg
   - executes via prepared interpreter (reuse path)

### Dimension scaling sweep runs

Executed pipeline runs (`warmup=1`, `runs=5`) for:

1. `256x256` -> `traces/dense-template-256x256-20260222T062154Z`
2. `512x512` -> `traces/dense-template-512x512-20260222T062006Z`
3. `1024x1024` -> `traces/dense-template-1024x1024-20260222T062017Z`
4. `2048x2048` -> `traces/dense-template-2048x2048-20260222T062027Z`
5. `2304x2304` -> `traces/dense-template-2304x2304-20260222T062229Z`
6. `2688x2688` -> `traces/dense-template-2688x2688-20260222T062240Z`
7. `2752x2752` -> `traces/dense-template-2752x2752-20260222T062306Z`
8. `2816x2816` -> `traces/dense-template-2816x2816-20260222T062218Z`
9. `3072x3072` -> `traces/dense-template-3072x3072-20260222T062252Z`
10. `4096x4096` -> `traces/dense-template-4096x4096-20260222T062039Z`

### Scaling findings

1. Compiler acceptance is solid through `4096x4096` for single-op Dense.
2. Two distinct runtime regimes were observed:
   - Cached regime (`256`..`2688`):
     - compiler emits `2` executables
     - parameter bytes kept on-chip
     - throughput climbs up to about `16 GMAC/s`
   - Streaming regime (`2752` and above):
     - compiler emits `1` executable
     - on-chip parameter caching drops to `0`
     - off-chip parameter streaming dominates
     - throughput collapses to about `0.44 GMAC/s`
3. Crossover is narrow between `2688` and `2752` in this environment.

### Layout mapping generalization

1. Ran single-hot probe suites:
   - `traces/dense-layout-probe-512x512-20260222T062104Z`
   - `traces/dense-layout-probe-1024x1024-20260222T062119Z`
2. All sampled offsets matched the generalized formula implemented in
   `dense_param_offset(...)`.

### Prepared execution validation

1. `cargo run --example gemm_int8 -- traces/dense-template-256x256-20260222T062154Z/dense_256x256_quant_edgetpu.tflite shift_plus1 ramp 30`
   - verified exact shift behavior with interpreter reuse
   - measured `avg_ms=0.216` over 30 runs
2. `cargo run --example gemm_int8_dynamic -- traces/dense-template-1024x1024-20260222T062017Z/dense_1024x1024_quant_edgetpu.tflite 1024 1024 identity ramp 30`
   - validated dimension-aware Rust path on hardware
   - output preview matched expected identity behavior

## 2026-02-22 (MVP consolidation: module split + API simplification + bundled templates)

### Codebase structure changes

1. Split previous monolithic `src/lib.rs` into focused modules:
   - `src/error.rs`
   - `src/device.rs`
   - `src/delegate.rs`
   - `src/interpreter.rs`
   - `src/flatbuffer.rs`
   - `src/gemm.rs`
2. Reduced `src/lib.rs` to module declarations + re-exports.

### GEMM API consolidation

1. Removed duplicate fixed-256 wrapper types:
   - `DenseGemm256Template`
   - `GemmTemplate256`
   - `PreparedGemm256`
2. Kept one canonical path:
   - `DenseGemmTemplate`
   - `PreparedDenseGemm`
3. Kept compatibility constants/helpers for 256 workflows:
   - `DENSE_GEMM256_DIM`
   - `DENSE_GEMM256_WEIGHT_COUNT`
   - `DENSE_GEMM256_WEIGHT_BYTES`
   - `DENSE_GEMM256_ZERO_POINT`
   - `dense_256_param_offset(...)`
4. Added `DenseGemmTemplate::set_diagonal(...)` so structured 256 examples stay supported via generic API.

### Bundled template support

1. Added precompiled templates under `templates/`:
   - `dense_2048x2048_quant_edgetpu.tflite`
   - `dense_2304x2304_quant_edgetpu.tflite`
   - `dense_2688x2688_quant_edgetpu.tflite`
2. Exposed constants:
   - `TEMPLATE_2048`
   - `TEMPLATE_2304`
   - `TEMPLATE_2688`
3. Added constructors:
   - `DenseGemmTemplate::from_bundled_2048()`
   - `DenseGemmTemplate::from_bundled_2304()`
   - `DenseGemmTemplate::from_bundled_2688()`
4. Updated `.gitignore` to track bundled template binaries while keeping other `*.tflite` files ignored.

### Example updates

1. Updated `examples/gemm_int8.rs` to use `DenseGemmTemplate` directly.
2. Kept `examples/gemm_int8_dynamic.rs` on generic path.
3. Added `examples/gemm_int8_bundled.rs` to run bundled templates without external model paths.

### Validation

1. `cargo fmt`
2. `cargo check --lib`
3. `cargo check --tests`
4. `cargo check --example gemm_int8`
5. `cargo check --example gemm_int8_dynamic`
6. `cargo check --example gemm_int8_bundled`

## 2026-02-22 (public API surface cleanup)

### API changes

1. Trimmed crate-root re-exports in `src/lib.rs` to user-facing types only.
2. Removed crate-root exports of internal FFI/raw types:
   - `TfLiteModel`, `TfLiteInterpreter`, `TfLiteInterpreterOptions`, `TfLiteTensor`, `TfLiteDelegate`
   - `TfLiteModelWrapper`
   - `EdgeTPUDelegateRaw`, `EdgeTPUDelegatePtr`, `EdgeTPUOption`, `EdgeTPUDeviceType`
3. Removed crate-root exports for device listing helpers/constants:
   - `list_devices`, `get_device_info`
   - VID/PID constants

### Follow-up cleanup

1. Removed now-unused `list_devices` and `get_device_info` functions from `src/device.rs`.
2. Updated examples to rely on `CoralDevice` accessors instead of removed listing helpers:
   - `examples/basic_usage.rs`
   - `examples/delegate_usage.rs`
   - `examples/verify_device.rs`
   - `examples/inference_benchmark.rs`
   - `examples/tflite_test.rs`

### Validation

1. `cargo fmt`
2. `cargo check --all-targets`

## 2026-02-22 (RE frontier platform expansion)

### Dense GEMM runtime improvements

1. Optimized `DenseGemmTemplate::set_weights_from_slice(...)`:
   - replaced per-element `dense_param_offset(...)` calls with direct tile-native restride loops.
2. Added correctness unit test:
   - `fast_restride_matches_formula_mapping`
   - verifies fast restride output against formula-derived offsets.
3. Added host-loop batch helper:
   - `PreparedDenseGemm::execute_batch_rows(&[i8]) -> Result<Vec<i8>, DenseGemmError>`
4. Added new error variant:
   - `DenseGemmError::BatchInputSizeMismatch`

### Large matrix tiling example

1. Added `examples/gemm_tiled_rows.rs`.
2. Implements row-tiling over bundled `2688x2688` template:
   - composes outputs for `rows_total > 2688`
   - demonstrates matrix-vector execution beyond one on-chip parameter block.

### Conv2D exploration toolchain

1. Added `tools/generate_conv2d_quant_tflite.py`:
   - deterministic single-layer Conv2D model generator
   - full INT8 TFLite conversion + metadata output
2. Added `tools/conv_template_pipeline.sh`:
   - `uv` environment + model generation + `edgetpu_compiler`
   - DWN1 extraction + executable parser + tensorizer inspect
   - optional benchmark via `--run-benchmark`

### Documentation

1. Added `docs/research_frontier_platform.md` summarizing:
   - fast restride path
   - batch helper semantics
   - tiled row execution
   - Conv2D pipeline usage
2. Updated `README.md` to include:
   - Conv2D pipeline/tool entries
   - `gemm_tiled_rows` example
   - new frontier platform doc link
3. Added initial pure-Rust USB driver milestones to frontier docs:
   - baseline contract capture
   - `rusb` init replay prototype
   - command-path emulation
   - feature-gated interpreter integration

## 2026-02-22 (frontier runs: Conv2D probe + multi-op chain + tiled throughput)

### Tiled GEMM validation and bug fix

1. Ran `examples/gemm_tiled_rows` on hardware at `rows_total=8192`.
2. Detected orientation bug in shift mode:
   - fixed per-tile mapping from `set_weight_qi8(local_row, col, ...)` to
     `set_weight_qi8(input_row, local_row, ...)`
   - root cause: row/col orientation at write site (`row=input`, `col=output`).
3. Post-fix validation:
   - `identity_cycle`: `mismatches=0`, `max_abs_delta=0`
   - `shift_plus1_cycle`: `mismatches=0`, `max_abs_delta=0`
4. Added throughput print to example:
   - `effective_gmac_per_s` based on `rows_total * tile_dim` MACs per run
5. Measured throughput at `8192` rows (`runs=5`):
   - `identity_cycle`: `avg_ms=119.554`, `effective_gmac_per_s=0.184`
   - `shift_plus1_cycle`: `avg_ms=117.630`, `effective_gmac_per_s=0.187`

### Conv2D pipeline runs

1. Ran:
   - `./tools/conv_template_pipeline.sh --height 224 --width 224 --in-channels 3 --out-channels 16 --kernel-size 3 --stride 1 --padding same --run-benchmark`
   - compile mapped `CONV_2D=1`, `executables=2`, on-chip params `2.75KiB`
   - benchmark avg `31.054 ms` (`1x224x224x3 -> 1x224x224x16`)
2. Ran:
   - `./tools/conv_template_pipeline.sh --height 224 --width 224 --in-channels 16 --out-channels 32 --kernel-size 1 --stride 1 --padding same --init-mode ones --run-benchmark`
   - compile mapped `CONV_2D=1`, `executables=2`, on-chip params `1.25KiB`
   - benchmark avg `69.552 ms` (`1x224x224x16 -> 1x224x224x32`)

### Conv2D tensorizer probing additions

1. Extended `tools/generate_conv2d_quant_tflite.py` with `single_hot` kernel mode.
2. Added `tools/conv_layout_probe.py`:
   - compiles single-hot Conv probes
   - extracts parameter region payloads
   - reports `mapping_candidate_offset` candidates per coordinate
3. Hardware-backed run:
   - `./tools/conv_layout_probe.py --height 32 --width 32 --in-channels 64 --out-channels 64 --kernel-size 1 --rep-samples 32`
   - output: `traces/conv-layout-probe-20260222T071933Z/layout_probe.{json,txt}`
4. Recovered 1x1 Conv2D channel-layout candidate (64x64 channels):
   - `offset = 512 + ((ic // 4) * 256) + (oc * 4) + (ic % 4)`
   - indicates a fixed `512`-byte prefix followed by Dense-like 4-lane packing.

### Multi-op Conv2D->Dense chain

1. Added `tools/generate_dense_conv_quant_tflite.py` (Conv2D -> GAP -> Dense).
2. Added `tools/multiop_template_pipeline.sh` for one-command generation+compile+parse+inspect+benchmark.
3. Ran:
   - `./tools/multiop_template_pipeline.sh --run-benchmark`
   - compiled model: `denseconv_16x16x16_conv64_k1_dense256_quant_edgetpu.tflite`
   - mapped ops:
     - `CONV_2D=1`
     - `MEAN=1`
     - `FULLY_CONNECTED=1`
   - benchmark avg `0.286 ms` (`1x16x16x16 -> 1x256`)
   - artifacts: `traces/multiop-template-20260222T071951Z/*`

## 2026-02-22 (transformer-linear milestone harness, 2304)

### Objective

Bridge the Dense tensorizer path to a transformer-like block benchmark focused
on Coral integration questions:

- six same-size linear stages (`Q/K/V/O/MLP_up/MLP_down`)
- `d_model=2304` to stay in the on-chip regime
- prefill-style row batches (`seq_len` rows)
- explicit model-switch overhead baseline

### Added benchmark example

1. Added `examples/transformer_linear_block.rs`.
2. Uses six separately patched `DenseGemmTemplate::from_bundled_2304()` models
   with stage-specific matrices:
   - `q_proj`: identity
   - `k_proj`: shift+1
   - `v_proj`: shift-1
   - `o_proj`: identity
   - `mlp_up`: shift+1
   - `mlp_down`: shift-1
3. Reports:
   - per-stage setup timing (`prepare_ms`, `first_invoke_ms`)
   - per-stage average latency (`q/k/v/attn/o/up/down`)
   - linear-only and end-to-end totals
   - derived throughput (`linear_gmac_per_s`, `end_to_end_gmac_per_s`)
4. Includes a model-switch baseline:
   - `same_stage6_ms` (runs one prepared stage six times) versus full
     stage-switched linear path.

### Attention path in milestone

1. Includes optional CPU single-head attention (`softmax(QK^T/sqrt(d))V`) between
   `V` and `O` to mimic transformer block flow.
2. Can be disabled with `--no-attention` to isolate pure linear-stage timing.

### Documentation updates

1. Added `docs/transformer_linear_block.md`.
2. Updated `README.md`:
   - example command list includes `transformer_linear_block`
   - tooling/example index includes the new benchmark
   - reverse-engineering notes list includes the new doc
   - added a dedicated milestone usage section

### Hardware runs

1. Ran:
   - `cargo run --example transformer_linear_block -- 4 1 0`
   - results:
     - `linear_only_ms=10.621`, `total_ms=11.370`
     - `linear_gmac_per_s=11.995`
     - `end_to_end_gmac_per_s=11.205`
2. Ran:
   - `cargo run --example transformer_linear_block -- 16 3 1 --no-attention`
   - results:
     - `linear_only_ms=30.757`, `total_ms=30.760`
     - `same_stage6_ms=30.936`
     - `linear_gmac_per_s=16.569`
3. Ran:
   - `cargo run --example transformer_linear_block -- 16 3 1`
   - results:
     - `linear_only_ms=33.195`, `attn_cpu=10.903`, `total_ms=44.099`
     - `same_stage6_ms=32.092`
     - `linear_gmac_per_s=15.352`
     - `end_to_end_gmac_per_s=11.556`

## 2026-02-22 (weight-loading bridge: f32 -> int8 -> patched template)

### Objective

Add a concrete integration bridge from model-style floating-point weights to
patched Coral templates with CPU reference verification.

### Added example + docs

1. Added `examples/gemm_weight_load_verify.rs`.
2. Added `docs/gemm_weight_load_verify.md`.
3. Updated `README.md`:
   - new example command in list
   - new dedicated section for the weight-loading bridge
   - linked new doc in RE notes.

### Example capabilities

1. Input:
   - synthetic deterministic f32 weights/inputs, or
   - raw little-endian f32 files (`--weights-f32-le`, `--inputs-f32-le`).
2. Quantization:
   - symmetric int8 quantization with configurable target dynamic range:
     - `--input-qmax` (default `32`)
     - `--weight-qmax` (default `16`)
3. Execution:
   - patches bundled `2304x2304` template via `set_weights_from_slice`
   - runs batched rows through `PreparedDenseGemm::execute_batch_rows`
4. Verification:
   - CPU int32 accumulator reference matmul
   - affine calibration against TPU output:
     - global affine
     - per-output affine
   - reports holdout/all-point error metrics and correlation.

### Hardware runs

1. Ran:
   - `cargo run --example gemm_weight_load_verify -- 8 3 1 2`
   - results:
     - `avg_ms=3.191`, `gmac_per_s=13.311`
     - holdout global affine: `mae=0.0019`, `rmse=0.0434`, `max_abs_delta=1`
2. Ran:
   - `cargo run --example gemm_weight_load_verify -- 16 2 1 8`
   - results:
     - `avg_ms=5.494`, `gmac_per_s=15.458`
     - holdout global affine: `mae=0.0002`, `rmse=0.0128`, `max_abs_delta=1`
     - holdout per-output affine: `mae=0.1227`, `rmse=0.3502`, `max_abs_delta=1`

## 2026-02-22 (wired f32 stage loading into transformer block + TPU validation)

### Objective

Wire the f32 weight-loading bridge into the six-stage transformer-linear block
benchmark so the block can run with quantized patched weights end-to-end on
TPU, not only pattern matrices.

### Implementation

1. Updated `examples/transformer_linear_block.rs` to support:
   - `--weight-source pattern|f32` (default `pattern`)
   - `--weights-dir <dir>` (`q_proj.f32le`, `k_proj.f32le`, `v_proj.f32le`,
     `o_proj.f32le`, `mlp_up.f32le`, `mlp_down.f32le`)
   - `--input-f32-le <path>`
   - `--input-qmax`, `--weight-qmax`, `--verify-calibration-rows`, `--seed`
2. Added full in-example path:
   - load/generate f32 stage weights
   - symmetric int8 quantization
   - per-stage `set_weights_from_slice` patch
   - stage prepare + execute on Coral
3. Added optional `q_proj` CPU-vs-TPU affine verification printout for the
   wired mode.

### Docs updates

1. Updated `README.md` with f32 wired transformer-block command and weight-file
   naming conventions.
2. Updated `docs/transformer_linear_block.md` with new CLI flags and run
   results for f32-weight mode.

### Hardware runs

1. Ran:
   - `cargo run --example transformer_linear_block -- 16 3 1 --no-attention --weight-source f32 --verify-calibration-rows 8`
   - results:
     - `linear_only_ms=31.926`, `total_ms=31.930`
     - `same_stage6_ms=31.821`
     - `linear_gmac_per_s=15.962`
2. Ran:
   - `cargo run --example transformer_linear_block -- 16 2 1 --weight-source f32 --verify-calibration-rows 8`
   - results:
     - `linear_only_ms=32.514`, `attn_cpu=10.501`, `total_ms=43.016`
     - `same_stage6_ms=32.532`
     - `linear_gmac_per_s=15.674`
     - `end_to_end_gmac_per_s=11.847`

## 2026-02-22 (Raspberry Pi 5 stack alignment + successful TPU runs)

### Objective

Recover stable EdgeTPU interpreter creation on Pi5 by aligning local
`libedgetpu` + `libtensorflowlite_c` stack, then validate GEMM/transformer
flows and usbmon capture.

### Root causes found on Pi

1. Local `libedgetpu` build failed with `collect2: fatal error: cannot find 'ld'`
   because upstream Makefile forced `-fuse-ld=gold` and Pi image lacked `ld.gold`.
2. After link-fallback fix, built `libedgetpu.so` had unresolved Abseil symbols
   at Rust link time because transitive dependencies were dropped.
3. Additional unresolved symbol (`absl_bad_optional_access`) required explicit
   link inclusion in the libedgetpu Makefile link flags.

### Bootstrap fixes landed

1. `7dc25ad`: fallback to `-fuse-ld=bfd` when `ld.gold` is unavailable.
2. `f062aaa`: force `-Wl,--no-as-needed` for libedgetpu so DT_NEEDED entries
   are preserved.
3. `af835c4`: add `-labsl_bad_optional_access` to libedgetpu link flags.

### Pi package/tooling setup used

1. Installed build deps used by bootstrap path:
   - `xxd`, `libabsl-dev`, `flatbuffers-compiler`, `libflatbuffers-dev`,
     `libusb-1.0-0-dev`, `bazelisk`
2. Built local libs:
   - `./tools/bootstrap_arch_stack.sh build-tflite-c --py-version 3.12`
   - `./tools/bootstrap_arch_stack.sh build-libedgetpu`
3. Runtime env:
   - `eval "$(./tools/bootstrap_arch_stack.sh print-env)"`

### Verified Pi outcomes (local stack, Coral attached)

1. `cargo run --example simple_delegate`
   - pass, EdgeTPU version:
     `BuildLabel(COMPILER=14.2.0,DATE=Feb 22 2026,TIME=09:51:57), RuntimeVersion(14)`
2. `cargo run --example gemm_int8_bundled -- 2304 identity 2`
   - pass, `avg_ms=7.968`, identity output preserved.
3. `cargo run --example transformer_linear_block -- 16 2 1 --no-attention --weight-source f32 --seed 123`
   - pass, `linear_only_ms=48.737`, `linear_gmac_per_s=10.456`.
4. `cargo run --example gemm_weight_load_verify -- 8 3 1 2`
   - pass, `avg_ms=4.104`, `gmac_per_s=10.347`,
     global-affine holdout `max_abs_delta=1`.
5. `sudo ./tools/usbmon_capture.sh -b 4 -- ... gemm_int8_bundled -- 2304 identity 5`
   - pass, capture + summary generated:
     - `traces/usbmon-20260222T095307Z-bus4/usbmon-bus4-20260222T095307Z.log`
     - `traces/usbmon-20260222T095307Z-bus4/usbmon-bus4-20260222T095307Z.summary.txt`

## 2026-02-22 (CLIP ViT SafeTensors parser + mapping preflight)

### Objective

Start real-model ingestion for the CLIP milestone by adding SafeTensors parsing,
layer-name mapping, and quantization preflight in Rust.

### Implementation

1. Added `src/clip.rs`:
   - `ClipSafeTensorFile` loader/introspection (`tensor_count`, `tensor_info`,
     `tensor_f32`, layer discovery).
   - `ClipVitLayerLinearNames` and ViT-B/32 shape validation helper.
   - `quantize_linear_out_in_to_row_major_qi8` to transpose PyTorch
     `[out, in]` linear weights into Coral row-major `[in, out]`.
2. Re-exported CLIP parser APIs at crate root (`src/lib.rs`).
3. Added runnable example:
   - `examples/clip_vit_safetensors_report.rs`
4. Added docs:
   - `docs/clip_vit_safetensors_report.md`
   - updated `README.md` example/doc index entries.

### Validation

1. Compile checks:
   - `cargo check --lib` (pass)
   - `cargo check --example clip_vit_safetensors_report` (pass)
2. Pi run against a synthetic CLIP-like SafeTensors file:
   - generated `/tmp/clip_vit_b32_fake_layer0.safetensors` with required layer-0
     tensors and exact ViT-B/32 shapes.
   - command:
     `cargo run --example clip_vit_safetensors_report -- /tmp/clip_vit_b32_fake_layer0.safetensors 0 127`
   - result: pass (`RC=0`)
   - highlights:
     - `Tensor count: 6`
     - discovered encoder layers: `count=1`, `first=0`, `last=0`
     - all six required tensors validated (`q/k/v/o`, `mlp.fc1`, `mlp.fc2`)
     - quantization preflight completed:
       - `q_proj q_bytes=589824 scale=0.007874016`
       - `mlp_fc1 q_bytes=2359296 scale=0.039370079`
       - `mlp_fc2 q_bytes=2359296 scale=0.047244094`
3. Negative-path check:
   - requesting nonexistent layer index (`1`) reports
     `MissingTensor("vision_model.encoder.layers.1.self_attn.q_proj.weight")`
     as expected.

## 2026-02-22 (CLIP rectangular templates + real checkpoint TPU probe)

### Objective

Validate CLIP-relevant rectangular Dense templates (`768x768`, `768x3072`,
`3072x768`) and run real CLIP ViT layer weights on Coral TPU.

### Template compilation results

1. Compiled on workstation (x86_64) with `tools/dense_template_pipeline.sh`:
   - `/tmp/clip-rect-templates/dense-768x768-20260222T105439Z`
   - `/tmp/clip-rect-templates/dense-768x3072-20260222T105442Z`
   - `/tmp/clip-rect-templates/dense-3072x768-20260222T105445Z`
2. All three models compiled successfully with:
   - `min_runtime_version=14`
   - `executables=2` (`PARAMETER_CACHING`)
   - `FULLY_CONNECTED` mapped to Edge TPU
3. On-chip parameter cache usage from compiler logs:
   - `768x768`: `576 KiB`
   - `768x3072`: `2.25 MiB`
   - `3072x768`: `2.25 MiB`
   (all well below the observed ~7 MiB cliff).

### Pi execution sanity (templates copied to Pi)

Template files on Pi:

- `/home/rpc/clip-traces/clip-rect-templates/dense_768x768_quant_edgetpu.tflite`
- `/home/rpc/clip-traces/clip-rect-templates/dense_768x3072_quant_edgetpu.tflite`
- `/home/rpc/clip-traces/clip-rect-templates/dense_3072x768_quant_edgetpu.tflite`

Using `examples/gemm_int8_dynamic.rs`:

1. `768x768` (`identity`, `runs=100`):
   - `avg_ms=0.403` (identity output preserved)
2. `768x3072` (`zero`, `runs=100`):
   - `avg_ms=0.544` (zero output as expected)
3. `3072x768` (`zero`, `runs=100`):
   - `avg_ms=0.519` (zero output as expected)

Approximate throughput (single-vector matvec):

- `768x768`: ~`1.46 GMAC/s`
- `768x3072`: ~`4.34 GMAC/s`
- `3072x768`: ~`4.55 GMAC/s`

### Real CLIP checkpoint ingestion and validation

1. Downloaded real checkpoint on Pi:
   - `https://huggingface.co/Bingsu/clip-vit-base-patch32-ko/resolve/main/model.safetensors`
   - local path:
     `/home/rpc/clip-models/clip-vit-base-patch32-ko-model.safetensors`
   - size: `~578 MiB`
2. SafeTensors parser preflight:
   - `Tensor count: 400`
   - discovered vision encoder layers: `0..11` (`12` layers)
   - layer `0` and `11` linear tensor shapes validated (`q/k/v/o/fc1/fc2`).

### Real-weight TPU probe (single stage)

Added and used `examples/clip_vit_layer_tpu_probe.rs`:

1. Layer `0`, stage `q` (`768x768`, `runs=20`):
   - latency `avg_ms=0.540`
   - CPU-accumulator vs TPU affine fit:
     - `corr=0.936058`, `mae=18.94`, `rmse=27.22`
2. Layer `0`, stage `fc1` (`768x3072`, `runs=20`):
   - latency `avg_ms=0.824`
   - fit:
     - `corr=0.996099`, `mae=1.29`, `rmse=3.97`
3. Layer `0`, stage `fc2` (`3072x768`, `runs=20`):
   - latency `avg_ms=0.794`
   - fit:
     - `corr=0.986527`, `mae=6.08`, `rmse=10.53`
4. Layer `11`, stage `fc1` (`768x3072`, `runs=20`):
   - latency `avg_ms=0.832`
   - fit:
     - `corr=0.946272`, `mae=21.68`, `rmse=28.38`

### Notes

1. `tools/dense_template_pipeline.sh` was updated to be architecture-aware for
   TensorFlow package/version and NumPy defaults:
   - x86 defaults remain `tensorflow-cpu==2.10.1`, `numpy==1.23.5`
   - ARM defaults use `tensorflow==2.19.0`, `numpy==1.26.4`
2. Pi cannot run the current `edgetpu_compiler` bootstrap binary (x86_64
   artifact), so rectangular template compilation currently happens on
   workstation and templates are copied to Pi for execution.

### Function Gemma Pi5 process-use reduction

#### Objective

Reduce Pi5 process RAM and host overhead when loading large Function-Gemma
SafeTensors checkpoints while keeping TPU stage output unchanged.

#### Changes

1. Added mmap-backed checkpoint loading in `src/function_gemma.rs`:
   - `FunctionGemmaStorage::{Mapped, Owned}`
   - `FunctionGemmaSafeTensorFile::load` now prefers `memmap2::Mmap` and
     falls back to owned bytes on mmap failure.
2. Added `storage_kind()` and shared `bytes()` view helpers.
3. Reduced repeated parsing overhead for dim inference by parsing once and
   resolving all stage tensors from one `SafeTensors` handle.
4. Fixed tensor-view lifetime handling after mmap introduction by ensuring each
   tensor view is used while its parsed metadata handle is still alive.

#### Validation (Pi5)

Using `/usr/bin/time -v` on the same commands and same model/template:

1. `function_gemma_layer_tpu_probe` (`layer=0 stage=q runs=20`):
   - baseline: `elapsed=5.67s`, `cpu=10%`, `max_rss=539808 KB`
   - mmap: `elapsed=2.92s`, `cpu=9%`, `max_rss=18192 KB`
2. `function_gemma_lm_head_sanity` (`token=42 topk=10`):
   - baseline: `elapsed=18.46s`, `cpu=80%`, `max_rss=531584 KB`
   - mmap: `elapsed=18.11s`, `cpu=92%`, `max_rss=336032 KB`

Observed effect:

1. TPU stage path keeps low host CPU and drops process RSS by ~`30x`.
2. CPU-only LM-head path drops RSS by ~`195 MB` with similar wall time.

#### Runtime note

On Pi5, commands must evaluate the repo env exports before execution:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
```

Without this, the process may load an older system `libedgetpu` instead of the
repo-managed runtime in `/home/rpc/.local/lib`.

### Function Gemma full decode loop + Coral tiled LM-head

#### Objective

Wire a full autoregressive decode loop in Rust for Function-Gemma, keeping
linear compute on Coral and removing the CPU LM-head bottleneck via tiled Coral
projection.

#### Changes

1. Added new example:
   - `examples/function_gemma_decode_loop.rs`
2. Added docs:
   - `docs/function_gemma_decode_loop.md`
3. Added model helper APIs in `src/function_gemma.rs`:
   - `tensor_names()`
   - `embedding_rows_f32(token_start, token_count)` for block reads used by
     tiled LM-head without loading the full embedding matrix into heap.
4. Updated `README.md` examples/docs list with decode-loop references.

#### Decode implementation details

1. Full per-token transformer path:
   - embedding lookup
   - per-layer RMSNorm
   - Coral stages `q/k/v/o/gate/up/down`
   - CPU single-token GQA attention with KV cache + RoPE
   - SwiGLU MLP (`silu(gate) * up` then `down`)
   - final RMSNorm
2. LM-head backends:
   - `cpu`: tied embedding projection on CPU
   - `coral`: tiled Coral projection using `640x2624` template across vocab
     tiles (`100` tiles for vocab `262146`).
3. Per-stage affine calibration retained (CPU accumulator vs TPU output fit)
   for dequantized f32 stage outputs.

#### Artifacts and runs (Pi5)

Commands were run with:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
```

LM template generated on workstation and copied to Pi:

- `/tmp/functiongemma-lm-template-20260222T145912Z/dense_640x2624_quant_edgetpu.tflite`
- `/home/rpc/functiongemma-templates-b1/dense_640x2624_quant_edgetpu.tflite`

1. Baseline (CPU LM-head, 1 layer, 2 decode tokens):
   - artifact: `~/clip-traces/functiongemma-decode-cpu-20260222T145726Z`
   - setup: `4619.589 ms`
   - decode: `ms_per_token=16870.889`
2. Coral LM-head (1 layer, 2 decode tokens):
   - artifact: `~/clip-traces/functiongemma-decode-coral-20260222T145932Z`
   - setup: `58478.197 ms` (includes preparing 100 LM tiles)
   - decode: `ms_per_token=633.598`
3. Full depth (18 layers, CPU LM-head, 1 decode token):
   - artifact: `~/clip-traces/functiongemma-decode-cpu-l18-20260222T150236Z`
   - setup: `35367.365 ms`
   - decode: `ms_per_token=17226.403`
4. Full depth (18 layers, Coral LM-head, 2 decode tokens):
   - artifact: `~/clip-traces/functiongemma-decode-coral-l18-s2-20260222T150421Z`
   - setup: `88877.781 ms` (layer prep + 100 LM tiles)
   - decode: `ms_per_token=978.161`

#### Outcome

1. Coral-tiled LM-head cuts full-depth token latency from ~`17.2s` to
   ~`0.98s` on Pi5 (same prompt/config), removing the dominant CPU vocab
   projection bottleneck.
2. Startup/setup cost rises due to one-time LM tile preparation; steady-state
   decode throughput improves by ~`17.6x`.

### 2026-02-24 - Side Milestone: LED Pattern Probe (Pi5 + Coral USB)

#### Objective

Add a small, reproducible side-channel experiment to toggle a likely board LED
control-related register path with strict restore safety.

#### Code and docs changes

1. Extended `examples/rusb_control_plane_probe.rs`:
   - New flags:
     - `--led-blink N`
     - `--led-reg OFF`
     - `--led-mask MASK`
     - `--led-on-ms N`
     - `--led-off-ms N`
   - Behavior:
     - read baseline register value
     - write `baseline ^ mask`
     - sleep for on-duration
     - restore baseline
     - repeat and verify final readback
   - Added known-register labels for SCU/OMC/ABM candidates used in prior RE.
2. Updated docs:
   - `docs/rusb_control_plane_probe.md`
   - Added LED mode usage and safety warning.

#### Pi5 execution

Host: `rpc@rpilm3.local`

1. Runtime init:

```bash
cd ~/coral-usb-oxidized
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
cargo run --example delegate_usage
```

2. LED pattern run:

```bash
cargo run --example rusb_control_plane_probe -- --led-blink 8 --led-on-ms 120 --led-off-ms 80
```

Observed register toggles:
- baseline `0x0001a704 (rambist_ctrl_1) = 0x0070007f`
- toggled `0x0000007f`
- restored to baseline each pulse
- final readback restored to `0x0070007f`

3. usbmon artifact:

```bash
sudo ./tools/usbmon_capture.sh -b 4 -- \
  cargo run --example rusb_control_plane_probe -- --led-blink 6 --led-on-ms 100 --led-off-ms 100
```

Artifact:
- `traces/usbmon-20260224T184251Z-bus4/usbmon-bus4-20260224T184251Z.log`

Trace contains alternating write32 submissions on `0x1a704`:
- `= 7f000000` (LE `0x0000007f`)
- `= 7f007000` (LE `0x0070007f`)
repeated for 6 pulses.

#### Safety finding

- Direct low-bit toggle attempt on `0x0001a30c (scu_ctrl_0)` failed with
  `Input/Output Error` and was followed by temporary USB instability on Pi5.
- Default LED probe target was moved to `0x0001a704` mask `0x00700000`, which
  is observed in normal runtime transitions and showed stable restore behavior.

### 2026-02-24 - Descriptor Contract Perturbation Campaign (Pi5)

#### Objective

Stress-test candidate control CSRs during a real GEMM invoke to move from
address-name mapping to behavioral semantics.

#### New tooling

1. Added invoke-coupled perturbation example:
   - `examples/gemm_csr_perturb_probe.rs`
   - Applies one CSR mutation in-process, runs bundled GEMM invoke, restores
     register, emits structured `RESULT` lines.
2. Added matrix runner:
   - `tools/csr_perturbation_matrix.sh`

#### Benign matrix result (clean state)

Artifact:
- `traces/csr-perturb-benign-20260224T191025Z/summary.tsv`

Findings:
1. Benign cases (`tileconfig` variants, `scu_ctr_7`, `rambist_ctrl_1`, ABM
   toggles, `runcontrol=0`) all returned `status=ok`.
2. Output head stayed stable (`0..15`) and restore checks passed.

#### Poison case reproduction

Poison mutation:
- `scalarCoreRunControl (0x00044018) <- 0x2` before invoke.

Artifact:
- `traces/usbmon-20260224T191133Z-bus4/usbmon-bus4-20260224T191133Z.log`
- plus derived reports (`register-report.json`, sequence and bulk signature
  JSONs in the same folder).

Observed:
1. Runtime abort during invoke:
   - `transfer on tag 2 failed ... USB transfer error 2`.
2. Subsequent delegate creation failed repeatedly (`DelegateCreationFailed`)
   while still enumerated as `18d1:9302`.
3. Kernel showed repeated USB resets.
4. User observed brighter/blinking white LED during this poison case.

Interpretation:
- `runcontrol=2` appears to trigger a sensitive control-state transition with
  strict preconditions tied to descriptor/queue runtime state.

### 2026-02-24 - Clean A/B poison boundary + LED correlation (Pi5)

#### Objective

Re-run a clean controlled A/B experiment to isolate `scalarCoreRunControl`
semantics with user-visible LED state included in observations.

#### Run set

Artifact root:
- `traces/led-poison-ab-20260224T194024Z`

Cases:
1. `baseline` (`none`)
2. `runcontrol=1` (`0x00044018 <- 0x1`)
3. `runcontrol=2` (`0x00044018 <- 0x2`)
4. post-poison `delegate_usage`

#### Findings

1. Baseline and `runcontrol=1` were both healthy:
   - `RESULT status=ok`
   - identical output head `0,1,2,...,15`
2. `runcontrol=2` triggered immediate runtime abort:
   - `transfer on tag 2 failed. Abort. Deadline exceeded: USB transfer error 2`
3. Post-poison runtime state:
   - device still enumerated as `18d1:9302`
   - delegate creation failed (`Failed to create EdgeTPU delegate`)
4. User observed LED starts blinking/brighter at poison trigger.

#### Progression observed after poison

1. Intermediate state: `Enumerated-But-Unusable` (`18d1:9302`, delegate dead).
2. Later state: device disappeared from bus entirely.
3. Manual reattach restored healthy bring-up:
   - `1a6e:089a -> 18d1:9302`
   - delegate creation successful again.

#### Additional synced artifacts

1. `traces/led-poison-cycle-20260224T193536Z`
2. `traces/led-blink-20260224T193434Z`

These captures reinforce that enumeration status alone is not a valid readiness
signal after poison conditions.

### 2026-02-24 - No-replug recovery experiment after poison

#### Objective

Test whether a poisoned runtime (`runcontrol=2`) can be recovered in software,
without physical reattach.

#### Artifact

- `traces/runcontrol-recovery-20260224T194932Z`

#### Procedure

1. Confirm healthy delegate path from clean attach.
2. Induce poison once:
   - `gemm_csr_perturb_probe 0x00044018 64 0x2 ...`
3. Recovery attempts in same poisoned session:
   - write `0x44018 <- 1`
   - write `0x44018 <- 0`, then `0x44018 <- 1`
   - `rusb_control_plane_probe --reset-device`
4. Re-check delegate creation after each step.

#### Results

1. Pre-state was healthy (delegate creation succeeded).
2. Poison reproduced immediately with known abort:
   - `transfer on tag 2 failed ... USB transfer error 2`
3. All direct recovery control writes timed out:
   - `VENDOR_WRITE64 0x00044018 ... timeout after 500 ms`
4. Delegate remained failed after all software recovery attempts.
5. After reset attempt, delegate path reported no Coral device.

#### Conclusion

In this run, no-replug recovery was not possible once poisoned:
- the control plane itself became non-responsive to runcontrol writes.
- practical recovery remains physical reattach (or lower-level host/port power
  recovery) after deep poison.
