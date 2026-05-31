# Arduino Nano 33 BLE — TinyBench-KWS Deployment Handoff

Status snapshot for resuming work in a fresh session (Claude Code or otherwise).

## Goal
Get DS-CNN INT8 keyword spotting running on Arduino Nano 33 BLE Sense Rev2, driven by a host Python benchmark over USB serial. Output: latency, peak RAM, flash, accuracy vs Python TFLite reference. Part of TinyBench Paper 1.

## Working environment
- Host: macOS (Intel MacBook Pro 2015, 16 GB), running as root
- Project user: `rohini` — most commands wrapped with `sudo -u rohini env HOME=/Users/rohini ...`
- Project root: `/Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/`
- Board: Arduino Nano 33 BLE Sense Rev2 — port `/dev/cu.usbmodem14101`, FQBN `arduino:mbed_nano:nano33ble`
- Tools installed: `arduino-cli 1.4.1` (via Homebrew), `arduino:mbed_nano@4.5.0` core, `Chirale_TensorFLowLite@2.0.0`, Python 3.9 + `pyserial`, `tensorflow 2.16.2`

## File inventory

### Firmware
- `firmware/arduino_nano33/kws_dscnn/kws_dscnn.ino` — Arduino sketch (TFLite interpreter + DWT cycle counter + serial JSON protocol). Currently contains DIAGNOSTIC prints (`{"dbg":"rx",...}` every 500ms while reading input). Strip these once bytes flow.
- `firmware/arduino_nano33/kws_dscnn/dscnn_model_data.cpp` — model bytes as C array. Symbols use `extern const` so they have external linkage (C++ default is internal — this was a bug fixed earlier).
- `firmware/arduino_nano33/kws_dscnn/dscnn_model_data.h` — header for the above
- `firmware/arduino_nano33/kws_dscnn/dscnn_int8.tflite` — local copy of source model (~48 KB)

### Host (cross-board, Python)
- `kws/host/protocol.md` — wire contract between firmware and host runner. **This is what the partner needs** for ESP32 and STM32 firmware.
- `kws/host/prepare_test_vectors.py` — builds the test vector pack from MFCC cache + TFLite quantization params. One-time.
- `kws/host/benchmark_serial.py` — drives any conforming board, collects metrics.

### Test vector pack (already built)
- `kws/host/test_vectors/test_vectors_int8.npy` — (11005, 490) int8
- `kws/host/test_vectors/test_labels.npy` — (11005,) int64
- `kws/host/test_vectors/tflite_reference_preds.npy` — (11005,) int64
- `kws/host/test_vectors/test_metadata.json` — SHA256 hashes, quant params, source files
- Python TFLite reference accuracy = **92.70%** on 11,005 samples

## Verified working
- Compile + flash succeeds
- Boot message comes through over serial:
  ```json
  {"event":"boot","board":"arduino_nano33","model":"dscnn","input_bytes":490,"input_dtype":9,"output_bytes":35,"output_dtype":9,"arena_used":78436,"arena_size":98304,"clock_hz":64000000}
  ```
- Followed by `READY`
- Flash usage: **397 KB (40%)** — AllOpsResolver is most of it
- RAM usage: 150 KB (57%) — 96 KB tensor arena + Mbed-OS overhead
- Peak arena used: **78 KB** ← this is the Paper 1 RAM metric

## Open issue at handoff

Host→device direction: Python writes 490 INT8 bytes after seeing `READY`, but the firmware never reports receiving them (no `{"dbg":"rx",...}` progress prints). User noted the **Arduino was physically disconnected** at some point during testing — this likely explains the missing bytes. **Reconnect and retest from step 1 below.**

If retest still fails:
- Read `firmware/arduino_nano33/kws_dscnn/kws_dscnn.ino` — current diagnostic version logs rx-progress every 500 ms
- The diagnostic firmware needs to be re-built and flashed (the version on the board may be stale)

## How to resume

```bash
# 1. Confirm board is connected
sudo -u rohini -i arduino-cli board list
# Expect: /dev/cu.usbmodem14101  Arduino Nano 33 BLE  arduino:mbed_nano:nano33ble

# 2. Re-flash the current (diagnostic) firmware
rm -rf /tmp/arduino_build
sudo -u rohini env TMPDIR=/tmp HOME=/Users/rohini arduino-cli compile \
    --fqbn arduino:mbed_nano:nano33ble \
    --build-path /tmp/arduino_build \
    /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/firmware/arduino_nano33/kws_dscnn

sudo -u rohini env TMPDIR=/tmp HOME=/Users/rohini arduino-cli upload \
    --fqbn arduino:mbed_nano:nano33ble \
    --port /dev/cu.usbmodem14101 \
    --input-dir /tmp/arduino_build \
    /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/firmware/arduino_nano33/kws_dscnn

# 3. Send one tensor and watch progress
sudo -u rohini env HOME=/Users/rohini python3 -u -c "
import serial, time, numpy as np
s = serial.Serial('/dev/cu.usbmodem14101', 115200, timeout=15)
time.sleep(1.5)
for _ in range(3):
    line = s.readline()
    if not line: break
    print('rx:', line.decode().rstrip())
vec = np.load('/Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/test_vectors/test_vectors_int8.npy')[0]
s.write(vec.tobytes()); s.flush()
deadline = time.time() + 30
while time.time() < deadline:
    line = s.readline()
    if not line: continue
    print('rx:', line.decode().rstrip())
    if line.decode().strip() == 'READY': break
"

# 4. Once roundtrip works, restore kBenchRuns=100 in kws_dscnn.ino, strip the {"dbg":...} prints, re-flash, then:
sudo -u rohini env HOME=/Users/rohini python3 -u \
    /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/benchmark_serial.py \
    --port /dev/cu.usbmodem14101 \
    --vectors-dir /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/test_vectors \
    --results-dir /Users/rohini/Desktop/Abhiraj/tiny_bench_research/tinybench/kws/host/results \
    --n-samples 20
```

## Bugs hit and how they were fixed
1. **Homebrew refused to run as root** → use `sudo -u rohini -i brew install ...`
2. **arduino-cli `ctags: cannot open temporary file`** → pass `TMPDIR=/tmp HOME=/Users/rohini` explicitly
3. **Linker: `undefined reference to g_model_data`** → C++ defaults `const` at namespace scope to internal linkage. Add `extern` keyword to the array definition in `dscnn_model_data.cpp`.
4. **`allocate_tensors_failed`** → 50 KB arena too small. Bumped to 96 KB. Actual usage: 78 KB.
5. **Missing boot message on reconnect** → Nano 33 BLE has no USB-to-serial chip; `setup()` only runs on hardware reset (flash). Host runner now tolerates missed boot.
6. **`PermissionError` writing to `kws/host/test_vectors/`** → directory had been created by root earlier; `chown -R rohini:staff` fixed it.

## Task tracker (in-session)
1. ✅ Install arduino-cli + Nano 33 BLE core
2. ✅ Install TFLite Micro library
3. ✅ Convert .tflite → C array
4. ✅ Write Arduino sketch
5. ✅ Compile and flash
6. 🟡 Write Python host runner — scripts written, but end-to-end roundtrip not yet verified
7. ⬜ Run benchmark and validate accuracy

## Decisions worth remembering
- **No Docker** for embedded dev (USB passthrough painful on Mac; arduino-cli already self-contained)
- **arduino-cli over Arduino IDE/PlatformIO** (scriptable, reviewer-friendly, same toolchain as IDE)
- **Chirale_TensorFLowLite** (maintained TFLite Micro fork) over `ArduTFLite` wrapper (need direct arena control for RAM metrics)
- **AllOpsResolver** for now; switch to `MicroMutableOpResolver` later to reclaim ~200 KB flash
- **Skip `board_hal.h` until 2nd board lands** — premature abstraction with only one implementation
- **`protocol.md` is the cross-board contract**, not the HAL — protocol is what the partner needs to mirror in STM32/ESP32 firmware

## What the partner needs from this work
- `kws/host/protocol.md` (cross-board firmware contract)
- `kws/host/test_vectors/*.npy` + `test_metadata.json` (identical INT8 byte stream for all boards)
- A copy of the Arduino sketch as reference implementation
