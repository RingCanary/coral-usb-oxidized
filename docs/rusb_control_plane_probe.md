# rusb Control Plane Probe

`examples/rusb_control_plane_probe.rs` is a pure-`rusb` control-plane harness
for Coral USB bring-up and low-level register/event probing.

It supports both safe baseline checks and explicit vendor register access:

1. Enumerate Coral IDs (`1a6e:089a`, `18d1:9302`)
2. Print bus/address/state
3. Optional interface claim
4. Optional standard USB `GET_STATUS`
5. Optional libusb `reset()`
6. Optional vendor CSR reads/writes (32-bit and 64-bit)
7. Optional event endpoint (`0x82`) packet reads
8. Optional interrupt endpoint (`0x83`) packet reads

## Command

```bash
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_control_plane_probe -- --claim-interface --get-status
cargo run --example rusb_control_plane_probe -- --claim-interface --vendor-read64 0x00044018
cargo run --example rusb_control_plane_probe -- --claim-interface --vendor-read64 0x00048788 --vendor-read32 0x0001a30c
cargo run --example rusb_control_plane_probe -- --claim-interface --read-event 1 --read-interrupt 1
```

## Notes

- Use this as the first pure-Rust USB layer check before replaying captured
  vendor control sequences.
- Vendor register offset format is full 32-bit CSR address (`0xHHHHLLLL`);
  internally this is split as:
  - `wValue = low16`
  - `wIndex = high16`
- `--reset-device` can trigger device re-enumeration.
- `--vendor-write32` / `--vendor-write64` are explicit mutation paths; prefer
  reads first while validating mappings.
- For permission issues, add proper udev rules or run with elevated privileges.
