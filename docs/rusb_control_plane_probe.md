# rusb Control Plane Probe

`examples/rusb_control_plane_probe.rs` is a minimal pure-`rusb` baseline for
Coral USB control-plane bring-up.

It does not issue proprietary vendor control writes yet. It focuses on safe
steps:

1. Enumerate Coral IDs (`1a6e:089a`, `18d1:9302`)
2. Print bus/address/state
3. Optional interface claim
4. Optional standard USB `GET_STATUS`
5. Optional libusb `reset()`

## Command

```bash
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_control_plane_probe -- --claim-interface --get-status
```

## Notes

- Use this as the first pure-Rust USB layer check before replaying captured
  vendor control sequences.
- `--reset-device` can trigger device re-enumeration.
- For permission issues, add proper udev rules or run with elevated privileges.
