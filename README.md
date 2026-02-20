# Coral USB Oxidized

Rust SDK/driver layer for Google Coral USB Accelerator discovery, delegate creation, and TensorFlow Lite C API interop.

## Raspberry Pi 5 compatibility

This project is now set up to build on Raspberry Pi 5 (64-bit Raspberry Pi OS / Debian-based Linux) without hardcoded workstation paths.

### Supported runtime layout

The build now searches standard library locations for both x86_64 and ARM:

- `/usr/lib`
- `/usr/local/lib`
- `/usr/lib/x86_64-linux-gnu`
- `/usr/lib/aarch64-linux-gnu` (Pi 5)
- `/usr/lib/arm-linux-gnueabihf`

You can also override library lookup with environment variables:

- `CORAL_LIB_DIR`
- `EDGETPU_LIB_DIR`
- `TFLITE_LIB_DIR`

## Device behavior

Coral USB commonly appears as:

- Initial: `1a6e:089a`
- After delegate/init/inference: `18d1:9302`

Both IDs are handled by this library, and should both be in udev rules.

```bash
echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", MODE="0664", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/71-edgetpu.rules
echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", MODE="0664", TAG+="uaccess"' | sudo tee -a /etc/udev/rules.d/71-edgetpu.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Raspberry Pi 5 setup

### 1) System packages

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl build-essential pkg-config libusb-1.0-0-dev clang llvm-dev libclang-dev
```

### 2) Install EdgeTPU runtime

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install -y libedgetpu1-std
```

### 3) Install TensorFlow Lite C library

You need `libtensorflowlite_c.so` installed in a standard linker path (recommended), or provide `TFLITE_LIB_DIR`.

If installed in a non-standard location:

```bash
export TFLITE_LIB_DIR=/path/to/tflite/lib
export LD_LIBRARY_PATH="$TFLITE_LIB_DIR:$LD_LIBRARY_PATH"
```

### 4) Build

```bash
cargo build
```

## Usage

### Examples

```bash
cargo run --example basic_usage
cargo run --example verify_device
cargo run --example delegate_usage
cargo run --example simple_delegate
cargo run --example tflite_test
cargo run --example tflite_standard_example
```

### Optional helper script

`run_test.sh` now supports an optional `TFLITE_LIB_DIR` override:

```bash
TFLITE_LIB_DIR=/usr/lib/aarch64-linux-gnu ./run_test.sh
```

## Notes

- Real hardware validation is expected for meaningful verification.
- API behavior for `CoralDevice`, `EdgeTPUDelegate`, and `CoralInterpreter` remains unchanged.
