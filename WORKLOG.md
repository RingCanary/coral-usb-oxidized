# WORKLOG

Concise event wise / epoch wise logs of activities, marked by epoch timestamp ( `date +%s` )


## 2026-03-01
- [epoch:1772342379] Revist by user, and some cleanup, verbose logs marked archive and moved to `docs/`
- [epoch:1772344130] Pi5 tmux run: added interposer payload dump mode (`LIBUSB_TRACE_DUMP_DIR/LENS`), captured full good/replay `2608`+`9872` submit payloads; byte deltas are sparse and structured (2608: 6 bytes at `74-76,88-90`; 9872: 13 bytes at `232-234,249-252,7672-7674,7690-7692`), with replay side zeros at these fields.
- [epoch:1772345128] Added instruction patch-injection path to replay (`--instruction-patch-spec`) with per-chunk hash telemetry (FNV), wired across all instruction send paths (bootstrap/preload/run/interleave); added `tools/build_instruction_patch_spec.py` to derive patch specs from good-vs-replay payload dumps.
- [epoch:1772346214] Derived concrete patch bytes (19 total) for len `2608` and `9872` from Pi5 dumps; wrote reusable spec at `docs/replay_instruction_patch_spec_2026-03-01.txt`.
- [epoch:1772347007] Pi5 reboot-first A/B with firmware upload (using rsync-synced patch spec): both unpatched and patched reached the same class-2 wall at offset 49152 (chunk 48 timeout); patched path confirmed active (2 instruction patch telemetry lines), so byte patching alone does not bypass admission wall.
- [epoch:1772355000] Executed #2-#6 campaign on Pi5 and wrote consolidated report `docs/usb_re_strategy_2_to_6_2026-03-01.md`: (a) URB metadata diff shows large submit/reap/discard churn delta, (b) reboot-first async parity (`local` vs `global lanes`) still fails at 49152, (c) deterministic fault signatures captured at ~33024 and ~49KiB with poison probes timing out bridge+SCU reads, (d) header grammar mined across 15 logs confirms stable `(len,tag)` structure, (e) descriptor parity verified DFU boot mode vs 6-endpoint runtime mode.
- [epoch:1772357397] Breakthrough: added `--bootstrap-known-good-order` to `examples/rusb_serialized_exec_replay.rs` and validated on Pi5 reboot-first A/B. Baseline order (`9872 -> 2608 -> 4194304`) fails at first 1MiB chunk (`actual=49152`), while reordered path (`2608 -> 4194304 -> 9872 -> 2048`) succeeds end-to-end with repeatable output hash (`0x3ce2a859ce7ed025`). usbmon capture confirms reordered header sequence on wire.
