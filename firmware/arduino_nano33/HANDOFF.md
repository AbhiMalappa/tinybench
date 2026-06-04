# Arduino Nano 33 BLE — TinyBench-KWS Deployment Handoff

State snapshot for resuming work in a fresh session (Claude Code or otherwise).

## Status: end-to-end pipeline is GREEN ✅

20-sample smoke test result (latest):
```
MCU accuracy:                 95.00%
TFLite ref accuracy (subset): 95.00%
MCU↔TFLite agreement:         100.00%  ← firmware is bit-perfect
Samples:                      20/20 (0 failed)
Latency median:               1145.7 ms (p95: 1146.1, p99: 1146.1)
Wall clock:                   243 s (12.2 s/sample)
Peak arena RAM:               78 KB
Flash:                        397 KB (40%)
```

Latency is **10× too slow** because `AllOpsResolver` uses portable kernels instead of CMSIS-NN. Open optimization task.

## Goal
DS-CNN INT8 keyword spotting on Arduino Nano 33 BLE Sense Rev2, driven by a host Python benchmark over USB serial. Output: latency, peak RAM, flash, accuracy vs Python TFLite reference. Part of TinyBench Paper 1.

## Environment
- Host: macOS (Intel MacBook Pro 2015, 16 GB), running as root
- Project user: `rohini` — wrap commands with `sudo -u rohini env HOME=/Users/rohini ...`
- Project root: `/Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/`
- Board: Arduino Nano 33 BLE Sense Rev2 at `/dev/cu.usbmodem14101`, FQBN `arduino:mbed_nano:nano33ble`
- Tools: `arduino-cli 1.4.1`, `arduino:mbed_nano@4.5.0`, `Chirale_TensorFLowLite@2.0.0`, Python 3.9 + `pyserial` + `tensorflow 2.16.2`

## Critical learnings (do NOT relearn)

1. **TFLite Micro reuses the input tensor's arena slot during `Invoke()`.** After invoke #1 the input is garbage. For multi-run timing, keep an unmodified copy in a static buffer and `memcpy` into `input->data.int8` before every `Invoke()`. **This was the bug that caused all-class-29 predictions.**
2. **C++ namespace-scope `const` defaults to internal linkage.** Model array MUST be declared `extern const unsigned char g_model_data[]` for the linker to find it across translation units.
3. **Mbed Nano 33 BLE USB CDC drops bytes on bulk writes** larger than ~256 B. Host must trickle: 32-byte chunks with 50 ms delay (~0.83 s per 490-byte tensor). Other boards may not need this.
4. **Nano 33 BLE `setup()` only runs on hardware reset (flash).** DTR toggles do NOT reset it. Boot message is one-shot. **DO NOT toggle DTR in host code** — it confuses USBSerial state and bytes get lost.
5. **macOS App Nap throttles Python `time.sleep()` when the laptop is locked.** This can stretch the 50 ms inter-chunk delay to seconds, causing the device's read_exact to time out mid-tensor. Always wrap long runs in `caffeinate -dimsu`.
6. **Always re-flash before starting a benchmark.** Avoids stale device state from prior failed runs. The `--reflash-build-dir` flag automates this.

## File inventory

### Firmware
- `firmware/arduino_nano33/kws_dscnn/kws_dscnn.ino` — main sketch. Uses `input_buffer[490]` + memcpy pattern. `kBenchRuns=10`, `kReadTimeoutMs=30000`. No diagnostic prints in current build.
- `firmware/arduino_nano33/kws_dscnn/dscnn_model_data.cpp` — model bytes. `extern const unsigned char g_model_data[] alignas(8)`.
- `firmware/arduino_nano33/kws_dscnn/dscnn_model_data.h`
- `firmware/arduino_nano33/kws_dscnn/dscnn_int8.tflite` — source model (~48 KB)

### Host (cross-board)
- `kws/host/protocol.md` — wire contract. **This is what the partner needs** for STM32 + ESP32 firmware.
- `kws/host/prepare_test_vectors.py` — one-time, builds the test vector pack.
- `kws/host/benchmark_serial.py` — drives any conforming board. Supports:
  - `--reflash-build-dir <dir>` — auto-uploads via arduino-cli before opening serial
  - `--resume-from <jsonl>` — appends to existing log, skips clips already completed
  - `--n-samples` (default 20, use 0 for full)
  - Per-clip `log_f.flush()` — JSONL safe to read mid-run
  - Cumulative summary on completion

### Test vector pack
- `kws/host/test_vectors/test_vectors_int8.npy` — (11005, 490) int8
- `kws/host/test_vectors/test_labels.npy` — (11005,) int64
- `kws/host/test_vectors/tflite_reference_preds.npy` — (11005,) int64
- `kws/host/test_vectors/test_metadata.json` — SHA256 hashes, quant params
- Python TFLite reference accuracy: **92.70%** on 11,005 samples

### Results so far
- Latest smoke: `kws/host/results/unknown_dscnn_20260530_201808.jsonl` (+ summary.json)

## Known issues / TODO

| Item | Priority | Effort | Notes |
|---|---|---|---|
| Switch AllOpsResolver → MicroMutableOpResolver with CMSIS-NN | HIGH (Paper) | ~30 min | Brings latency from 1146 ms → ~100 ms, flash 397 KB → ~150 KB. **Required for credible paper numbers.** |
| Bump post-reflash sleep from 2 s → 3 s | LOW | trivial | Boot message sometimes missed → `arena_used_kb` shows 0 in summary. Cosmetic. |
| Train TC-ResNet8 + GRU-48 + repeat pipeline | MED | days | Two more rows of the headline table. |
| Hand `protocol.md` + test vectors + reference sketch to partner | MED | minutes | Unblocks partner's STM32 + ESP32 implementation. |

## How to resume

### Quick health check (1 minute)
```bash
sudo -u rohini -i arduino-cli board list
# Expect: /dev/cu.usbmodem14101  Arduino Nano 33 BLE  arduino:mbed_nano:nano33ble
```

### 20-sample smoke test (4 minutes, auto-reflash, validates green state)
```bash
sudo caffeinate -dimsu sudo -u rohini env HOME=/Users/rohini python3 -u \
    /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/benchmark_serial.py \
    --port /dev/cu.usbmodem14101 \
    --vectors-dir /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/test_vectors \
    --results-dir /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/results \
    --n-samples 20 \
    --reflash-build-dir /tmp/arduino_build
```

### Resume an interrupted run
```bash
# Same command as above, plus:
    --resume-from /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/results/<prior_log>.jsonl
```

### For long runs (500+ samples), prevent laptop sleep
```bash
# Snapshot current values first
sudo pmset -g | grep -E "sleep|hibernatemode|standby|powernap|lidwake"

# Disable
sudo pmset -a sleep 0 displaysleep 0 disksleep 0

# Restore afterwards (defaults seen on this machine)
sudo pmset -a sleep 1 displaysleep 10 disksleep 10
```

Default `pmset` values on this machine (captured 2026-05-31):
- `sleep 1, displaysleep 10, disksleep 10, hibernatemode 3, powernap 1, standby 1, lidwake 1`

### Wrap long runs in nohup (survives terminal close)
```bash
sudo nohup caffeinate -dimsu sudo -u rohini env HOME=/Users/rohini python3 -u \
    /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/benchmark_serial.py \
    ... > /tmp/run.log 2>&1 &
```

## Bugs hit and how each was fixed

1. **Homebrew refused to run as root** → use `sudo -u rohini -i brew install ...`
2. **arduino-cli `ctags: cannot open temporary file`** → pass `TMPDIR=/tmp HOME=/Users/rohini` explicitly
3. **Linker: undefined reference to `g_model_data`** → add `extern` to the array definition (C++ internal linkage default)
4. **`allocate_tensors_failed`** → bumped arena from 50 → 96 KB (actual usage: 78 KB)
5. **`PermissionError` writing to `kws/host/test_vectors/`** → `chown -R rohini:staff`
6. **Bytes lost on 490-byte bulk write** → chunked 32 B / 50 ms (Mbed USB CDC quirk)
7. **All-class-29 predictions** → TFLite Micro arena reuses input; added static `input_buffer[490]` + memcpy before each Invoke
8. **Mid-clip silent corruption on resume** → added `--reflash-build-dir` for guaranteed clean device state
9. **App Nap during locked laptop** → wrap commands in `caffeinate -dimsu`, run on AC power
10. **GRU-96 deployment blocked by TFLite Micro buffer cap** → 840-op unrolled GRU exceeds the per-subgraph buffer-tracking limit (`max is 101`). `AllocateTensors()` fails regardless of arena size. CNNs (DS-CNN ~12 ops, TC-ResNet8 ~28 ops) are unaffected. Documented in `tinybench_kws_results.md` as a paper finding; not pursued via library patch.

## Decisions worth remembering
- No Docker for embedded dev (USB passthrough on Mac is painful; arduino-cli is self-contained)
- arduino-cli over Arduino IDE / PlatformIO (scriptable, reviewer-friendly)
- Chirale_TensorFLowLite over the ArduTFLite wrapper (need direct arena control for RAM metrics)
- AllOpsResolver for now (works, but slow) → MicroMutableOpResolver is the optimization
- Skip `board_hal.h` until 2nd board lands (rule of three — premature abstraction with one impl)
- `protocol.md` IS the cross-board contract; `board_hal.h` would be internal-only

## What the partner needs (handoff bundle)
1. `kws/host/protocol.md` — the spec
2. `kws/host/test_vectors/*.npy` + `test_metadata.json` — identical byte stream for all boards
3. `kws/host/benchmark_serial.py` — same script drives all 3 boards (just change `--port`)
4. `to_rohini/dscnn_int8.tflite` (for ESP32 TFLite Micro) + `to_rohini/dscnn_int8.onnx` (for STM32 Cube.AI)
5. `firmware/arduino_nano33/kws_dscnn/` as reference implementation
6. This HANDOFF.md so they don't re-discover the 9 bugs listed above

## In-session task tracker
1. ✅ Install arduino-cli + Nano 33 BLE core
2. ✅ Install TFLite Micro library
3. ✅ Convert .tflite → C array
4. ✅ Write Arduino sketch
5. ✅ Compile and flash
6. ✅ Write Python host runner (with reflash + resume)
7. ✅ Run 20-sample smoke test (95% MCU accuracy = TFLite ref, 100% agreement)
8. ✅ Optimize AllOpsResolver → MicroMutableOpResolver (flash 397→210 KB; latency unchanged since CMSIS-NN was already engaged)
9. ✅ Full 11K paper run for DS-CNN (92.15% MCU acc, 97.70% agreement, 1144 ms p50, 0 failures over 6h)
10. ✅ Train + deploy + 11K paper run for TC-ResNet8 (93.04% MCU acc, 99.06% agreement, 76 ms p50 — 15× faster than DS-CNN)
11. ✅ Train GRU-96 (float32 92.62%, INT8 92.52%)
12. ❌ Deploy GRU-96 to Arduino — infeasible on stock TFLite Micro (buffer-tracking cap; paper finding documented)

Next:
- Hand off bundle to partner (protocol.md + test vector packs + sketches as reference)
- Partner: STM32N6 + ESP32 deployment for all three models (GRU-96 on STM32N6 is the most interesting cell — Arduino infeasibility makes the cross-board contrast sharper)
- Start drafting the paper (2 of 12 headline cells filled, GRU-96 infeasibility = paper-worthy finding)
