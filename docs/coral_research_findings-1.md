# Research Findings: Coral TPU over USB / DarwiNN Protocol

I've researched the control-plane and execution structures for `libedgetpu` over the DarwiNN driver interfaces. Here are the exact mappings and semantics you requested to wire into your pure `rusb` replay tooling.

## 1. Mappings for 0x4018, 0x8788, 0xa30c
The addresses you observed over `usbmon` correspond to DarwiNN's Beagle core CSR offsets, stripped of their higher-order base digits. Let's map them to their specific literal DarwiNN CSR names:

*   **`0x4018`** maps to **`scalarCoreRunControl`** (Base offset in Beagle: `0x44018`)
*   **`0x8788`** maps to **`tileconfig0`** (Base offset in Beagle: `0x48788`)
*   **`0xa30c`** maps to **`scu_ctrl_0`** (Base offset in Beagle: `0x1a30c`)

## 2. Host Queue & Doorbell Semantics
The host queues in DarwiNN are managed through a unified descriptor structure mapped via [QueueCsrOffsets](file:///home/bhav/Documents/experiments/rngcnr-gh/coral-usb-oxidized/.research_tmp/libedgetpu/driver/config/queue_csr_offsets.h#29-45). Here is how the roles map to the CSRs (using Beagle's `InstructionQueue` as a concrete example):

*   **Queue / Descriptor Base Pointer**: `queue_base` (e.g. `0x48590` for instruction stream)
*   **Doorbell / Kick**: `queue_tail` (e.g. `0x485a8` for instruction stream)
*   **Completion / Ack Gate**: `queue_fetched_head` and `queue_completed_head` (e.g. `0x485b0` and `0x485b8`)

*(Other standard queue CSRs include `queue_control`, `queue_status`, `queue_descriptor_size`, and `queue_size`)*

## 3. Interrupt Endpoint 0x83 Completion Meaning
Endpoint `0x83` is the main asynchronous interrupt IN endpoint (`kInterruptInEndpoint = 3`, `0x80 | 3`).
*   **Expected Sequence**: `libedgetpu` continuously polls this endpoint asynchronously for a fixed **4-byte payload**.
*   **Payload Semantics**: The 4-byte payload is an unsigned 32-bit `raw_data` mask representing top-level hardware IRQs:
    *   **Bit 0** (`raw_data & 1`): Represents a **Fatal Error**. Standard procedure is to check the HIB Error registers (`hib_error_status` and `hib_first_error_status`).
    *   **Bits 1-31** (`raw_data >> 1`): Represents an array of boolean flags for top-level DarwiNN interrupts mapping to the execution pipeline (e.g., triggering the host-side completion handlers).

## 4. Decoding the 8-byte Control Words (`20720300...` and `004c0200...`)
These invariant 8-byte control words are the [UsbMlCommands](file:///home/bhav/Documents/experiments/rngcnr-gh/coral-usb-oxidized/.research_tmp/libedgetpu/driver/usb/usb_ml_commands.h#82-85) **Packet Headers** sent over the bulk-out endpoint right before a stream payload during `EXECUTION_ONLY` bulk transfers.

**Structure**: `<32-bit little-endian length> <8-bit tag> <3 unused zero bytes>`
*   **Bytes 0-3**: Payload length (Little-endian uint32).
*   **Byte 4**: Descriptor Tag (`tag & 0xF`). The bit indicates the destination stream:
    *   `0`: Instructions (`kInstructions`)
    *   `1`: Input Activations (`kInputActivations`)
    *   `2`: Parameters (`kParameters`)
    *   `3`: Output Activations (`kOutputActivations`)
*   **Bytes 5-7**: Unused zero-padding.

**Decoding your traces:**
*   `20720300...`: Length bytes are `20 72 03 00`, which translates to little-endian `0x00037220` (**225,824 bytes** payload length). If the next byte is `0x00`, this corresponds to an `Instruction` stream.
*   `004c0200...`: Length bytes are `00 4c 02 00`, which translates to little-endian `0x00024c00` (**150,528 bytes** payload length). If the next byte is `0x02`, it's the `Parameters` layer stream.

This should give you exactly what you need to interpret the DarwiNN layout in `rusb`.
