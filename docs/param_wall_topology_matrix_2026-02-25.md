# Param Wall Topology Matrix (2026-02-25)

## Goal
Determine whether the 49,152-byte class-2 ingress wall is caused by host transfer topology or by device-side admission/state progression.

## Confirmed runs

| ID | Mode | Chunk | Partial/Retry | Result |
|---|---|---:|---|---|
| R1 | submit-OUT + event/interrupt lanes (2/1) | 16384 | off | fails at 49152 |
| R2 | submit-OUT + event/interrupt lanes (2/1) | 16384 | on + retries=8 | fails at 49152 (zero progress retries) |
| R3 | submit-OUT + event/interrupt lanes (2/1) | 8192 | on + retries=8 | fails at 49152 (zero progress retries) |
| R4 | submit-OUT + event/interrupt lanes (2/1), true inflight bulk-out depth=2 | 16384 | on + retries=8 | fails at 49152 (`inflight chunk 3` timeout, queued chunks cannot retire) |
| R5 | submit-OUT + event/interrupt lanes (2/1), true inflight bulk-out depth=4 | 16384 | on + retries=8 | fails at 49152 (`inflight chunk 3` timeout, queued chunks cannot retire) |
| R6 | sync bulk write, libedgetpu-like `1 MiB` payload chunk + full 4 MiB header | 1048576 | n/a | fails at offset 0 (single write timeout) |
| R7 | submit-OUT depth=1, `1 MiB` payload chunk + full 4 MiB header | 1048576 | on + retries=2 | first transfer progresses exactly 49152 then stalls (`actual=49152`, then zero progress) |
| R8 | R7 + preposted submit bulk-in lanes (`0x81`, lanes=2) | 1048576 | on + retries=2 | unchanged: first transfer progresses 49152 then stalls; bulk-in lanes report only timeout callbacks |
| R9 | R7 with long submit timeout (`12000 ms`) | 1048576 | on + retries=0 | unchanged: first transfer progresses exactly 49152, then timeout with zero additional progress |
| R10 | tmux batch (`nopower`) repeat of R9 topology | 1048576 | on + retries=2 | unchanged under watcher load: first transfer progresses exactly 49152, then stalls |
| R11 | dual-writer DoS race (two replay processes) | 4096 | retries=0 | no tunnel: contention only (`Busy` + I/O/setup-timeout), no progress bypass evidence |
| R12 | reboot-first split framing (`split=524288`, `chunk=262144`, header=1048576) | 262144 | on + retries=2 | unchanged: first chunk progresses exactly 49152 then stalls |
| R13 | reboot-first hybrid gate/handshake (`chunk=1024`, `split=8192`, gates at 32768/33792) | 1024 | depth=2 + retries=4 | gate#1 at 32768 succeeds, gate#2 at 33792 times out on `a0d4` read (control cliff) |
| R14 | reboot-first dual-writer race (`4KiB` chunks) | 4096 | retries=0 | writer A reaches 49152 wall; writer B fails early in boot-mode race path; no bypass |
| R15 | reboot-first `pyusb` parity harness (`1MiB` class-2 chunk, full 4MiB header, xray setup) | 1048576 | n/a | identical wall: first `dev.write` returns `49152`, next write times out |
| R16 | reboot-first `pyusb` parity harness, request-shape parity (`1048576 -> 49152`, then `999424`) | 1048576 | n/a | identical wall with same second-request size as rusb (`999424`) |
| R17 | reboot-first full preload (`dense_2048`, libedgetpu 60-step setup, submit lanes depth=1) | 1048576 | accept_partial + retries=2 | unchanged: first class-2 transfer reports exactly `49152` progress, then zero-progress retries |
| R18 | reboot-first capped class-2 (`max=32768`, full header len kept at `4194304`) + continue to run phases | 32768 | accept_partial + retries=2 | preload write succeeds, replay reaches run instruction+input phase, but no completion event and output read times out |
| R19 | reboot-first zero-byte class-2 stream (`max=0`, full header mode) + continue to run phases | 0 | accept_partial + retries=2 | no class-2 ingress attempted; replay still times out waiting for completion event/output |
| R20 | reboot-first libedgetpu-topology attempt (`1MiB`, submit depth=3, submit lanes: bulk-in=8/event=1/intr=1) | 1048576 | accept_partial=off retries=0 | unchanged: first class-2 completion `C Bo -2 49152`; queued class-2 URBs retire `C Bo -2 0` |
| R21 | reboot-first global submit lanes (started before first Bo): `Bi(1024)x8 + Bi(16)x1 + Ii(4)x1`, plus `1MiB` class-2 depth=3 | 1048576 | accept_partial=off retries=0 | unchanged: first class-2 completion `C Bo -2 49152`; queued class-2 URBs retire `C Bo -2 0` |
| R22 | reboot-first `LD_PRELOAD` interposer hash A/B (`good=gemm_int8_dynamic`, `bad=rusb_serialized_exec_replay R21`) | 1048576 | n/a | decisive: `2608` and `9872` tag-0 payload SHA-256 differ even with same first/last bytes; runtime mutates instruction payload before submit |

Traces:
- `traces/usbmon-20260225T194139Z-bus4/usbmon-bus4-20260225T194139Z.log`
- `traces/usbmon-20260225T194331Z-bus4/usbmon-bus4-20260225T194331Z.log`
- `traces/usbmon-20260225T194426Z-bus4/usbmon-bus4-20260225T194426Z.log`
- `traces/usbmon-20260225T200121Z-bus4/usbmon-bus4-20260225T200121Z.log`
- `traces/usbmon-20260225T200228Z-bus4/usbmon-bus4-20260225T200228Z.log`
- `traces/usbmon-20260225T200806Z-bus4/usbmon-bus4-20260225T200806Z.log`
- `traces/usbmon-20260225T200905Z-bus4/usbmon-bus4-20260225T200905Z.log`
- `traces/usbmon-20260225T201112Z-bus4/usbmon-bus4-20260225T201112Z.log`
- `traces/usbmon-20260225T201223Z-bus4/usbmon-bus4-20260225T201223Z.log`
- `traces/usbmon-20260225T202242Z-bus4/usbmon-bus4-20260225T202242Z.log`
- `traces/usbmon-20260225T202541Z-bus4/usbmon-bus4-20260225T202541Z.log`
- `traces/usbmon-20260225T202800Z-bus4/usbmon-bus4-20260225T202800Z.log`
- `traces/usbmon-20260225T202926Z-bus4/usbmon-bus4-20260225T202926Z.log`
- `traces/usbmon-20260225T203133Z-bus4/usbmon-bus4-20260225T203133Z.log`
- `traces/usbmon-20260225T210652Z-bus4/usbmon-bus4-20260225T210652Z.log`
- `traces/usbmon-20260225T210852Z-bus4/usbmon-bus4-20260225T210852Z.log`
- `traces/usbmon-20260225T213320Z-bus4/usbmon-bus4-20260225T213320Z.log`
- `traces/usbmon-20260225T214612Z-bus4/usbmon-bus4-20260225T214612Z.log` (known-good dense inference)
- `traces/usbmon-20260225T214836Z-bus4/usbmon-bus4-20260225T214836Z.log`
- `traces/usbmon-20260225T220950Z-bus4/usbmon-bus4-20260225T220950Z.log`
- `traces/libusb-preload-20260225T221846Z/good.tsv`
- `traces/libusb-preload-20260225T221846Z/replay.tsv`
- `traces/libusb-preload-20260225T221846Z/diff.txt`

## Interpretation

1. Cliff is byte-total anchored (`49152`), not chunk-count anchored.
2. Host-side sliding offset semantics do not recover once at boundary (`actual_length=0` repeatedly).
3. Preposted status lanes (`0x82/0x83`) are not sufficient to admit additional parameter bytes.
4. True concurrent EP `0x01` in-flight depth (`1 -> 2 -> 4`) does not move the wall; this weakens pure host queue-depth hypotheses.
5. Even when matching known-good payload topology (`1 MiB` Bo chunks under a full `4 MiB` class-2 header), our submit path still stalls after exactly `49152` accepted bytes.
6. Near-anchor good-vs-bad diff still shows no distinguishing control tuples; discriminant remains bulk completion semantics (`C Bo 0 1048576` in good vs `C Bo -2 49152` in bad).
7. Mirroring known-good pre-anchor read activity with preposted bulk-in submit lanes (`0x81`) does not unlock progress past `49152`.
8. Increasing submit timeout by 40x (`300 ms -> 12000 ms`) does not move the wall; this is not a short-timeout artifact.
9. tmux/watcher load does not alter the wall behavior (`R10` reproduces `49152`).
10. Parallel writer DoS races produce interface contention (`Busy`) and early transport errors, not admission bypass.
11. Reboot-first split-framing run (`R12`) confirms descriptor-split topology changes still do not move the wall.
12. Reboot-first hybrid run (`R13`) directly reproduces the control poison cliff at `33792` while confirming `32768` gate viability.
13. `pyusb` parity runs (`R15`, `R16`) reproduce the same `49152` boundary and timeout pattern, ruling out a rusb-wrapper-only explanation.
14. Forcing replay to continue into later descriptor phases (`R18`, `R19`) is not sufficient to trigger completion; without successful class-2 admission/drain, event/output completion still does not occur.
15. Even with `1 MiB` class-2 payloads, in-flight depth `3`, and submit lanes (`8/1/1`) (`R20`), replay still fails at `49152` then `0` on queued chunks.
16. Known-good dense inference (`214612`) completes all four `1 MiB` class-2 chunks (`C Bo 0 1048576` x4); replay variants do not complete a single class-2 `1 MiB` chunk.
17. Matching known-good preposted read geometry before first Bo (`Bi1024x8 + Bi16 + Ii4`) (`R21`) is still insufficient; class-2 completion remains `49152 -> 0`.
18. `LD_PRELOAD` hash diff (`R22`) shows known-good and replay do not submit identical tag-0 instruction payloads (`len=2608`, `len=9872`) even though headers and edge bytes match; this upgrades the “dynamic patch layer” from hypothesis to primary target.

## Hygiene note

After aggressive/repeated failure runs, runtime setup can fail immediately at
step 0 (`0x0001a30c` timeout). For valid A/B comparisons, prefer reboot-first
state restoration per high-value run instead of iterative hub toggles.

## Next targeted runs

| ID | Variable under test | Expected discriminator |
|---|---|---|
| N1 | Port/topology move (alternate host controller path) | If host-topology issue, boundary shifts/disappears |
| N2 | True multi-URB in-flight bulk-out queue depth >1 (single phase) | **Completed**: wall unchanged at `49152` for depth `2` and `4` |
| N3 | Dynamic control transfers extracted from known-good trace at near-wall offsets | If missing runtime semantic, wall may clear without topology change |
| N4 | Required event dependency before/within parameter stream (strict) | If hidden progress token exists, event-correlated unlock appears |

## Stop condition

If `N1` and `N2` still fail at exactly 49,152 with zero-progress retries, prioritize firmware/runtime state semantics over host transfer mechanics.
