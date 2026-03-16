# Phase 4 completion/control plan (2026-03-07)

## Terminology correction

If `edgetpu_compiler` is still in the artifact-creation path, then the path is **not** custom/native yet.

Use these terms precisely:

- **pure-rusb runtime control**
  - no `libedgetpu` in the execution path
  - native USB control/data-plane replay
- **compiler-assisted artifact control**
  - we can patch, replay, and manipulate compiled artifacts
  - but artifact creation still depends on Google tooling
- **native/custom artifact generation**
  - no `edgetpu_compiler`
  - no legacy TensorFlow/Bazel/Python toolchain in the active path
  - native code emits the runnable artifact state

Phase 4 should aim for the third state, not just the second.

## Current Phase 4 state

What is actually proven today:

- `k=3` is already different from the completed `1x1` story
- `PC` is still not the blocker in the tested same-product `k=3` regime
- parameters are **not** invariant across same-product spatial moves
- EO alone is **not** sufficient
- target params + target EO are sufficient

References:

- `docs/phase4_conv2d_k3_scout_2026-03-06.md`
- `docs/phase4_conv2d_k3_crossdim_oracle_matrix_2026-03-06.md`

So Phase 4 is currently at:

> boundary found, mechanism unsolved

not at:

> completion and control

## Phase 4 success contract

Phase 4 is complete only when all of the following are true for one bounded family:

1. pure-rusb replay is the execution path
2. no `libedgetpu` is needed in the active runtime path
3. no `edgetpu_compiler` is needed in the active artifact-creation path
4. no old TensorFlow/Python/Bazel stack is needed in the active artifact-creation path
5. native code emits the runnable param + EO state
6. the Pi replay command reproduces the target hash on DUT

Bounded target family:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- same-product spatial moves

Anything short of that remains transitional.

## Active-path rule

For Phase 4, the active path should be defined as:

- pure-rusb runtime execution
- native Rust RE/materialization code
- bounded checked-in seed artifacts only when unavoidable
- Pi execution helpers under `scripts/`

Everything else is either:

- compatibility-only
- legacy
- archived research scaffolding

and should be quarantined accordingly.

## Milestones

### P4-M0: Entropy reduction and contract freeze

Goal:

- narrow the supported surface
- make the active path explicit
- stop mixing legacy and native control language

Deliverables:

- split active docs from legacy docs
- define one official Phase 4 target family
- mark `legacy-runtime` as compatibility-only, not strategic
- quarantine archive-forwarding shims from the active path

Exit test:

- a new engineer can identify the active path in under 5 minutes from repo root

### P4-M1: Explain `k=3` parameter change

Goal:

- determine exactly why params differ across same-product spatial moves for `k=3`

Questions:

- which param subregions change?
- are changes concentrated in prefix/meta, layout, or both?
- does a simple recurrent law exist per channel/block regime?

Deliverables:

- region-diff report for `k=3` same-product pairs
- small local probe helpers if needed
- explicit conclusion on whether `k=3` param change is structurally simple enough for native synthesis

Exit test:

- written explanation of param change by region, not just `param_equal=False`

### P4-M2: Native `k=3` param materializer in Rust

Goal:

- produce `k=3` target params without compiler involvement for the bounded family

Deliverables:

- Rust param-region materializer
- local byte-equivalence checks where possible
- DUT proof that target params from native code match target replay needs

Exit test:

- target params can be generated in native code for the bounded family and used in replay successfully

### P4-M3: EO target-state localization and synthesis

Goal:

- solve the remaining EO target-state problem for the same bounded `k=3` family

Constraints:

- do not restart broad whole-stream minimization
- use the Phase 2/3 discipline: isolate only the regions that matter

Deliverables:

- EO localization report
- native synthesis or native patch emission for the minimal required EO state

Exit test:

- anchor + native target params + native EO state reproduces target-equivalent replay

### P4-M4: Native family materialization

Goal:

- stop describing the flow as “patch compiled TFLite”
- promote it to a native materialization path

Deliverables:

- one bounded family package/materialization format in Rust
- seed executable blobs or family descriptors only if strictly needed
- no compiler call in the active loop

Exit test:

- the runnable Phase 4 artifact can be produced entirely from repo-native inputs and native code

### P4-M5: Completion demo

Goal:

- prove completion/control, not just understanding

Deliverable:

- one command that:
  - does not call `edgetpu_compiler`
  - does not depend on old Python/TensorFlow/Bazel in the active path
  - runs pure-rusb replay on Pi
  - reproduces target hash

That is the Phase 4 completion milestone.

## Highest-value next experiments

Ordered by value:

1. parameter diff by region for `k=3` same-product pairs
2. native `k=3` param-region probe/materializer
3. EO localization only after param synthesis is working
4. native bounded-family materialization prototype
5. no-compiler Pi end-to-end replay

The immediate next experiment should be:

> explain the `k=3` parameter delta by region

because Phase 4 already knows params matter, but not why.

## Entropy-reduction plan

### 1. Split active vs legacy documentation

Current problem:

- `README.md` still mixes pure-rusb/native work with legacy delegate/compiler flows

Plan:

- keep `README.md` for the active pure-rusb/native-control path
- move legacy delegate/TFLite/compiler instructions into `README.legacy.md` or `docs/legacy_runtime.md`

### 2. Demote `legacy-runtime`

Current problem:

- `Cargo.toml` still exposes a large first-class `legacy-runtime` example surface

Plan:

- keep it for compatibility only
- mark it as non-strategic in docs
- stop using it as repo-front-door material

### 3. Quarantine archive-forwarding shims

Current problem:

- `tools/README.md` lists many top-level tools that only forward to `tools/archive/`

Plan:

- either keep a tool as a real active top-level tool
- or remove the shim and keep it only under archive

Active path should not depend on archive shims.

### 4. Prune or isolate stale examples

Candidates for quarantine review:

- `examples/archive/`
- older `legacy-runtime` benchmark/demo examples superseded by pure-rusb scripts
- overlapping benchmark examples once the new script-based benchmark path is adopted

Rule:

- if it does not help the active Phase 4 path, move it out of the front door

### 5. Remove generated clutter from supported surfaces

Examples:

- `tools/__pycache__/`

Rule:

- do not keep generated/runtime clutter in active repo surfaces

### 6. Tighten language

Stop calling compiler-assisted paths “compilerless.”

Use:

- native runtime control
- compiler-assisted artifact control
- native artifact generation

This reduces conceptual entropy and forces milestones to stay honest.

## Concrete pruning candidates

### Front-door docs

- split `README.md`
- add a short active-path entrypoint doc

### Legacy surfaces

- `legacy-runtime` examples in `Cargo.toml`
- legacy delegate/TFLite example documentation

### Archive shims to review

Per `tools/README.md`, these should be reviewed for removal from the top-level active namespace:

- `bootstrap_arch_stack.sh`
- `bootstrap_edgetpu_compiler.sh`
- `build_instruction_patch_spec.py`
- `conv_layout_probe.py`
- `conv_template_pipeline.sh`
- `dense_layout_probe.py`
- `dense_quant_value_probe.py`
- `dense_template_matrix_patch.py`
- `exec_chunk_diff.py`
- `instruction_dim_field_analysis.py`
- `multiop_template_pipeline.sh`
- `strace_usb_scaling.py`
- `synthesize_instruction_patch_spec.py`
- `usbmon_param_handshake_probe.py`
- `usbmon_register_map.py`
- `word_field_holdout_validate.py`

These are not necessarily to be deleted immediately, but they should stop pretending to be part of the Phase 4 active path unless proven otherwise.

## What not to do

- do not widen operator scope before solving `k=3` params
- do not restart broad EO minimization before param mechanics are native
- do not keep both legacy and active paths as equal citizens in docs
- do not claim “custom/native control” while `edgetpu_compiler` remains mandatory

## Recommended immediate sequence

1. execute P4-M0 first
2. run the `k=3` param-region explanation experiment
3. implement native `k=3` param materialization
4. only then attack EO synthesis
5. end with the no-compiler Pi replay demo

That sequence minimizes entropy while maximizing the chance that Phase 4 ends in actual control rather than another partial scout.
