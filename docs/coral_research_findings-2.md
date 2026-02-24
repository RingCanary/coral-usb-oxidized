
Here’s what I found on the public web that directly answers your 5 “parallel input” asks, and it lines up *very* cleanly with the exact offsets/markers you’re seeing.

## 1) CSR name mapping for your exact offsets (0x4018, 0x8788, 0xa30c)

A public, machine‑readable CSR name→offset map exists in `ricardodeazambuja/libredgetpu` as `REGISTER_MAP` (auto‑generated from Google’s `beagle_csr_offsets.h`). It explicitly maps:

* **0x44018 → `scalarCoreRunControl`**
* **0x48788 → `tileconfig0`**
* **0x1A30C → `scu_ctrl_0`**

You’re seeing **0x4018 / 0x8788 / 0xA30C** because USB CSR accesses split the 32‑bit CSR offset across **wValue=low16** and **wIndex=high16**. So the *full* CSR offsets are:

* `0x4018` + `wIndex=0x0044` ⇒ **0x44018 = scalarCoreRunControl**
* `0x8788` + `wIndex=0x0048` ⇒ **0x48788 = tileconfig0**
* `0xA30C` + `wIndex=0x0001` ⇒ **0x1A30C = scu_ctrl_0** ([GitHub][1])

That’s exactly why those “same addresses” appear everywhere in usbmon — you’re looking at the low 16 bits.

## 2) The “true doorbell / kick” (control-plane)

Both the reverse‑engineered driver and libedgetpu itself treat **`scalarCoreRunControl` (0x44018)** as the key run-control knob:

* `libredgetpu` init sequence polls `scalarCoreRunControl` to reach 0 (reset complete), then writes 1 to move to run; and later uses it as the run-control primitive for execution enablement. ([GitHub][2])
* A separate independent reverse‑engineering doc calls out run control at **`scalarCoreRunControl` (0x44018)** and tile enable at `tileconfig0` (0x48788). ([GitHub][3])

So if you’re trying to label:

* **doorbell/kick:** *most likely* `scalarCoreRunControl` write(s), especially the transition writing **`… := 1`** (and possibly `:= 0/2` for stop/reset-like behavior). ([GitHub][2])

## 3) What the 0x83 “interrupt endpoint” is and what it contains

This is spelled out in **official** open‑source libedgetpu USB code:

* Endpoint IDs (numbers) are:

  * **Bulk‑in output stream:** endpoint **1** → address **0x81**
  * **Bulk‑in event stream:** endpoint **2** → address **0x82**
  * **Interrupt‑in stream:** endpoint **3** → address **0x83** ([GitHub][4])

And the payload formats are:

* **0x82 (event stream): 16 bytes** per event. It is parsed as:

  * first 8 bytes: **offset (uint64)**
  * next 4 bytes: **length (uint32)**
  * next byte: low nibble = **tag** (DescriptorTag) ([GitHub][5])
* **0x83 (interrupt stream): 4 bytes** (“raw_data uint32”), with a TODO in libedgetpu saying it’s not yet further decoded. ([GitHub][4])

So: **0x83 is *not* the 16‑byte completion/event descriptor; it’s a separate 4‑byte interrupt channel.** The “completion-like” structured thing is on **0x82**.

This directly answers your point (3): endpoint **0x83** carries a 32‑bit raw interrupt word; the more informative completion descriptors (tag/offset/length) arrive on **0x82**. ([GitHub][4])

## 4) The two 8‑byte “control words” in the invoke loop (20720300…, 004c0200…)

Those are the **bulk‑out header** used in “single bulk out endpoint” mode.

In libedgetpu, the header construction is:

* 8 bytes total
* bytes `[0..3]` = **length** (uint32 little‑endian)
* byte `[4]` low nibble = **descriptor tag**
* bytes `[5..7]` = 0 ([GitHub][5])

So the “words” you’re seeing in usbmon:

* `20720300` (bytes `20 72 03 00`) → length **0x00037220**
* `004c0200` (bytes `00 4c 02 00`) → length **0x00024c00** (which is *exactly* 224×224×3 RGB input size = 150,528 bytes)

To identify what each header is *for*, you need the 5th byte’s low nibble (tag):

* 0 = Instructions
* 1 = InputActivations
* 2 = Parameters
* 3 = OutputActivations
* 4–7 = Interrupt0–3 ([GitHub][4])

So your “two 8‑byte control words” are very plausibly:

* header for some **instructions or parameter chunk** (len 0x37220)
* header for **input activations** (len 0x24c00)

…and the tag nibble tells you which.

## 5) Candidate queue / descriptor base pointer registers (what to grep for in usbmon)

This is the big unlock from the CSR map: it contains explicit “queue base/control/head/tail” registers by name. In `REGISTER_MAP` you have (examples):

* `instruction_queue_base` **0x48590**
* `instruction_queue_control` **0x48568**
* `instruction_queue_tail` **0x485A8**
* `instruction_queue_completed_head` **0x485B8**
* `instruction_queue_int_status` **0x485C8**

And similarly for:

* `input_actv_queue_base` **0x485F8**
* `output_actv_queue_base` **0x48528**
* `param_queue_base` **0x48660**
  …plus their *_control / *_tail / *_completed_head / *_int_status regs. ([GitHub][1])

### Why this matters for your “which write is the true X?”

If you’re currently only tracking low16 values like `0x4018/0x8788/0xa30c`, the queue bases will show up as low16 values like:

* `0x8590` (instruction_queue_base low16)
* `0x85F8` (input_actv_queue_base low16)
* `0x8528` (output_actv_queue_base low16)
* `0x8660` (param_queue_base low16)

with high16 typically `0x0048` for those (since they’re 0x48xxx). So: **grep your usbmon for 0x8590 / 0x85F8 / 0x8528 / 0x8660** and you’ll likely see the “real queue base pointer” writes if the host programs them at all.

## Extra: the “USB-side regs” you’re likely already seeing (descr_ep, multi_bo_ep, outfeed chunk)

Your question mentioned “queue/descriptor base pointer” and you also cited stable setup/teardown writes.

There’s a second cluster of very protocol‑relevant USB bridge CSRs, and libedgetpu explains them:

* `descr_ep` controls which descriptors are enabled; libedgetpu writes **0xF0** (only SC host interrupt descriptors) or **0xFF** (all descriptors), depending on options. ([GitHub][6])
* `multi_bo_ep` selects single‑endpoint vs multi‑bulk‑out endpoint mode (writes 0 or 1). ([GitHub][6])
* `outfeed_chunk_length` is set (0x20 for capped 256B in some USB2 HS conditions, else 0x80 for “largest chunk” mode). ([GitHub][6])

These three are very likely the “invariant control writes” you see bracketing invoke loops.

## What this implies for Phase A + Phase B

### Phase A (pure rusb replay)

You can implement the protocol in three layers, all now source‑confirmed:

1. **CSR access control transfers** (read/write 32 vs 64):

   * 64‑bit uses bRequest=0; 32‑bit uses bRequest=1; offset split into low16/high16. ([GitHub][5])

2. **Bulk‑out framing** (single BO endpoint mode):

   * write 8‑byte header (len + tag nibble) then data. ([GitHub][5])

3. **Completion**:

   * drain **event packets on 0x82** (16 bytes; tag/offset/length),
   * optionally also read 0x83 raw interrupts if you’re running async/overlap. ([GitHub][4])

### Phase B (mapping EXECUTION_ONLY fields to hardware actions)

The clean “cached weights” semantic is implemented plainly in `libredgetpu`:

* PARAMETER_CACHING phase: send instructions(tag 0) + params(tag 2), then read status (0x82)
* EXECUTION_ONLY phase: send instructions(tag 0) + input(tag 1), then read output(0x81), then read status(0x82) ([GitHub][2])

That should let you wire “EXECUTION_ONLY fields” to concrete transfers and expected barriers without guessing.

## Key sources (direct links)

(These are all public, and contain the exact structs/offsets you asked for.)

```text
https://github.com/google-coral/libedgetpu/blob/master/driver/usb/usb_ml_commands.h
https://github.com/google-coral/libedgetpu/blob/master/driver/usb/usb_ml_commands.cc
https://github.com/google-coral/libedgetpu/blob/master/driver/usb/usb_driver.cc

https://github.com/ricardodeazambuja/libredgetpu
  - libredgetpu/registers.py (CSR name->offset map)
  - libredgetpu/driver.py (init + execute protocol)

https://github.com/geohot/edgetpuxray
  - connect.py / simple.py (raw libusb traces + minimal driver)
```

If you want the fastest “actionable next grep” from your usbmon logs: **search for writes to low16 = `8590`, `85F8`, `8528`, `8660`** (queue bases) and correlate them with any immediately-following bursts of event packets on **0x82** (tag/offset/length). That’s the most direct route to labeling (1) queue base pointer and (3) completion gate using *named* regs.

[1]: https://raw.githubusercontent.com/ricardodeazambuja/libredgetpu/main/libredgetpu/registers.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/ricardodeazambuja/libredgetpu/main/libredgetpu/driver.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/ricardodeazambuja/libredgetpu/main/docs/HARDWARE_ANALYSIS.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/google-coral/libedgetpu/master/driver/usb/usb_ml_commands.h "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/google-coral/libedgetpu/master/driver/usb/usb_ml_commands.cc "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/google-coral/libedgetpu/master/driver/usb/usb_driver.cc "raw.githubusercontent.com"
