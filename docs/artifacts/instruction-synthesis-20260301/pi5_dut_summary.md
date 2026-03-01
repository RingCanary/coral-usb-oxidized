# Pi5 DUT Validation Summary (2026-03-01)

Model under test:
- `/tmp/dense-1536/dense_1536x1536_quant_edgetpu.tflite`

Replay mode:
- `--bootstrap-known-good-order`
- firmware when needed: `/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin`

## Cases

1. Baseline (no instruction patch)
- Result: PASS
- Output hash: `0xdc8c52f84cb2e9c0`
- Log: `pi5-logs/coral_dut_baseline_1536_fw.log`

2. Full synth patch (`EO+PC`, 191 rules)
- Result: FAIL at class-2 admission
- Error: `descriptor tag=2 payload write failed at offset 0 ... Operation timed out`
- Clean reboot-first log: `pi5-logs/coral_dut_full_synth_fw_clean.log`

3. PC-only synth patch (97 rules, len=2096)
- Result: FAIL at class-2 admission
- Error: `descriptor tag=2 payload write failed at offset 0 ... Operation timed out`
- Log: `pi5-logs/coral_dut_pc_only_fw.log`

4. EO-only synth patch (94 rules, len=7952)
- Result: class-2 preload succeeds; execution fails
- Observed: completion event received, then `UsbError(Timeout)` on output read
- Log: `pi5-logs/coral_dut_eo_only_fw.log`

## Interpretation
- PARAMETER_CACHING instruction mutations are admission-critical (preload wall regression).
- EXECUTION_ONLY instruction mutations are execution/output-critical after preload.
- Family-scoped instruction synthesis needs stronger constraints for PC stream before DUT-safe use.
