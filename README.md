# TinyBench-KWS

Cross-platform keyword spotting benchmark on 3 MCU boards.
Part of the TinyBench series — Paper 1.

**ML pipeline (Python):** Training, quantization, benchmark orchestration
**Firmware (C):** STM32CubeIDE / TFLite Micro inference, UART protocol, cycle-counter timing

## Status (2026-06-17) — complete

All boards done **in-house** (the "partner" split mentioned below is historical — built by Abhiraj).

| Board | DS-CNN (v2) | TC-ResNet8 | GRU-96 |
|---|---|---|---|
| Arduino Nano 33 BLE | ✅ full | ✅ full | ❌ infeasible (TFLite-Micro buffer cap) |
| ESP32-S3 (SIMD) | ✅ full | ✅ full | ❌ infeasible (TFLite-Micro buffer cap) |
| STM32N6 CPU (NPU off) | ✅ full | ✅ full | ✅ full |
| STM32N6 NPU (on) | ✅ full | ✅ full | ❌ infeasible (Neural-ART 4-D layout) |

### DS-CNN v2 — HW-friendly re-architecture, deployed identically across all 4 platforms (N = 11,005 each)

| Platform | Engine | MCU acc | MCU↔ref agreement | Latency median | Speedup vs original DS-CNN |
|---|---|---|---|---|---|
| Arduino Nano 33 BLE | CMSIS-NN TFLite Micro | 92.45% | 99.47% | 312.26 ms | 3.7× (1144 ms) |
| ESP32-S3 (esp-nn SIMD) | esp-tflite-micro | 92.46% | 99.45% | 27.24 ms | 3.5× (94.6 ms) |
| STM32N6 CPU | X-CUBE-AI | 92.36% | 98.35% | 64.27 ms | 3.9× (249 ms) |
| STM32N6 **NPU** | Neural-ART | 92.40% | 98.26% | **0.463 ms** | **138.8× vs same-model CPU** |

The **original DS-CNN (50×11 feature map) is incompatible with the Neural-ART NPU** — its 2-D global average
pool exceeds the accelerator's ≤3×3 HW pooling window, forcing a software-fallback float epoch that
produces wrong output (documented in `firmware/stm32n6/FINDINGS.md`). A **one-line fix** — a stride-(2,2)
stem shrinking the pre-pool map to **25×6** — makes the model **fully HW (0 software epochs)** at **no
accuracy cost** (92.4% vs the original 92.7%) and **3.5–3.9× faster on every CPU/MCU** with less RAM
(STM32 CPU 20.4 vs 137.8 KiB). v2 is therefore the single consistent DS-CNN across the whole matrix; the
original is retained as a documented model–accelerator incompatibility (alongside GRU-96). Accuracy is
remarkably consistent across platforms (92.36–92.46%). Latency = one timed inference/sample (N=1),
distribution over all 11,005 samples.

Numbers + analysis: `../tinybench_kws_results.md` (pending v2 update). STM32N6 detail:
`firmware/stm32n6/RESULTS_CPU.md` (NPU section + same-model speedup) and `firmware/stm32n6/FINDINGS.md`
(DS-CNN FINDING + RESOLVED).

---

## Boards

| Board | CPU | Clock | RAM | Flash | Inference engine |
|---|---|---|---|---|---|
| STM32N6570-DK | Cortex-M55 | 800 MHz | 4.2 MB | external | STM32 Cube.AI (NPU on + off) |
| ESP32-S3-DevKitC-1 N8R8 | Xtensa LX7 | 240 MHz | 512 KB | 8 MB | TFLite Micro |
| Arduino Nano 33 BLE Sense Rev2 | Cortex-M4 | 64 MHz | 256 KB | 1 MB | TFLite Micro |

---

## Models

| Model | Params | Float32 acc | INT8 TFLite acc | Arduino MCU acc (n=11,005) | INT8 ONNX | TFLite INT8 |
|---|---|---|---|---|---|---|
| DS-CNN v2 (25×6) | 25,251 | 92.35% | 92.55% | **92.45%** ✅ | `dscnn_v2_int8.onnx` | `dscnn_v2_int8.tflite` |
| DS-CNN orig (50×11) | 25,251 | 92.71% | 92.70% | 92.15% | `dscnn_int8.onnx` | `dscnn_int8.tflite` |
| TC-ResNet8 | 65,827 | 93.18% | 92.99% | **93.04%** ✅ | `tcresnet_int8.onnx` | `tcresnet_int8.tflite` |
| GRU-96 | ~34K | 92.62% | 92.52% | ❌ infeasible — see results | `gru_int8.onnx` | `gru_int8.tflite` |

Full benchmark data (latency, RAM, flash, MCU↔TFLite agreement, disagreement analysis): see `../tinybench_kws_results.md`. GRU-96 deployment was blocked by TFLite Micro's per-subgraph buffer-tracking cap on Arduino Nano 33 BLE (840-op unrolled GRU exceeds `max is 101`).

Model files are in `kws/checkpoints/` after training + quantization. The current INT8 artifacts for firmware deployment (`dscnn_int8.tflite`, `dscnn_int8.onnx`, `mfcc_config.json`) are also mirrored in `to_rohini/` as a partner-handoff folder. **Arduino Nano 33 BLE deployment status and gotchas: see `firmware/arduino_nano33/HANDOFF.md`. Cross-board firmware contract: see `kws/host/protocol.md`. Current research plan: see `../tinybench_research_plan_updated.md`.**

---

## Repository layout

```
tinybench/
├── kws/
│   ├── models/          # PyTorch model definitions
│   │   ├── dscnn.py
│   │   ├── tcresnet.py
│   │   └── gru.py
│   ├── data/
│   │   └── dataset.py   # Speech Commands loader + MFCC cache
│   ├── train.py         # Training script
│   ├── quantize.py      # INT8 ONNX + TFLite conversion
│   └── mfcc_config.json # Single source of truth for MFCC parameters
├── anomaly/             # Paper 2 — TinyBench-AD (in progress)
├── requirements.txt
└── README.md
```

---

## Quick start — Python side

```bash
pip install -r requirements.txt

# Train (seed=42 default; checkpoint selected on val accuracy)
python kws/train.py --model dscnn --cache-dir ./mfcc_cache

# Quantize (calibration uses training split; produces INT8 ONNX + TFLite INT8)
python kws/quantize.py --model dscnn --cache-dir ./mfcc_cache

# Build test vectors (run once per model × backend before benchmarking)
python kws/host/prepare_test_vectors.py --model dscnn --backend tflite  # Arduino / ESP32-S3
python kws/host/prepare_test_vectors.py --model dscnn --backend onnx    # STM32 / Cube.AI
```

---

## MFCC parameters — `kws/mfcc_config.json`

This file is the **single source of truth** shared between Python and C firmware.
The C preprocessing must match these values exactly.

| Parameter | Value | Notes |
|---|---|---|
| `sample_rate` | 16000 Hz | 16 kHz mono |
| `n_mfcc` | 10 | MFCC coefficients per frame |
| `n_fft` | 512 | FFT size |
| `hop_length` | 320 | 20 ms stride → exactly 49 frames per 1 s clip |
| `win_length` | 400 | 25 ms window |
| `n_mels` | 40 | Mel filter banks |
| `n_frames` | 49 | Frames per clip |
| `n_classes` | 35 | Output classes |

Input tensor shape: **(1, 49, 10)** — batch × frames × MFCC coefficients, INT8.

---

## Deployment guide

### STM32N6570-DK — X-CUBE-AI / `stedgeai` path (done in-house)

**Model file:** `checkpoints/<model>_int8.onnx` → preprocessed to `<model>_int8_v17.onnx` (IR/opset
downgrade + static-batch fixes) for `stedgeai`. **Full recipe in `firmware/stm32n6/FINDINGS.md`.**
The original CubeIDE-GUI steps below are superseded by the CLI flow actually used:

1. CPU build: `stedgeai generate --target stm32 --name network -m <model>_int8_v17.onnx -o kws_<model>_cpu/`
2. NPU build: `stedgeai generate --target stm32n6 --st-neural-art profile-default ...` (weights → ext-flash blob)
3. Compile: `make MODEL=<model>` (firmware in `firmware/stm32n6/`), then load to SRAM via
   `STM32_Programmer_CLI -c port=SWD mode=UR -halt -d ...elf -coreReg MSP=0x34200000 PC=<Reset_Handler> -run`
   (the board has **no internal flash**; `--start` would reset into the bootrom — use `mode=UR` + `-coreReg`/`-run`).
4. Drive with `kws/host/benchmark_serial.py --backend onnx`. **Complete**: CPU (NPU-off) for all 3 models; NPU-on for both CNNs (DS-CNN v2 + TC-ResNet8); GRU-96 NPU documented infeasible.

### ESP32-S3-DevKitC-1 and Arduino Nano 33 — TFLite Micro path

**Model file:** `kws/checkpoints/dscnn_int8.tflite`

Step 1 — Convert TFLite flatbuffer to C array:
```bash
xxd -i dscnn_int8.tflite > dscnn_model_data.cc
```
Rename the array in `dscnn_model_data.cc` to `g_model_data` and the length to `g_model_data_len`.

Step 2 — Add to firmware project:
- Include `dscnn_model_data.cc` and `dscnn_model_data.h` in your build
- Link TFLite Micro runtime (ESP-IDF component or Arduino library)

Step 3 — Inference call:
```c
#include "tensorflow/lite/micro/micro_interpreter.h"

// Input: int8_t tensor of shape [1, 49, 10]
// Output: int8_t tensor of shape [1, 35] — argmax gives class index
```

---

## board_hal.h — firmware abstraction

Each board implements these 5 functions. Everything above this layer is identical.

```c
void  hal_init();
void  hal_run_inference(int8_t* input, int8_t* output);
float hal_get_cycles_ms();
void  hal_uart_send(const char* result_json);
int   hal_get_ram_used_kb();
```

### UART result format (one JSON per inference)
```json
{"board":"stm32n6","model":"dscnn","npu":1,"latency_ms":2.3,"class":7}
```

---

## Timing standard

| Board | Timer | How to read |
|---|---|---|
| STM32N6 | DWT cycle counter | `DWT->CYCCNT / SystemCoreClock * 1000` |
| ESP32-S3 | CCOUNT register | `xthal_get_ccount() / 240e6 * 1000` |
| Arduino Nano 33 | DWT cycle counter | Same as STM32, `SystemCoreClock = 64e6` |

**Latency definition:** one timed `Invoke()` per sample (`N=1` on **all** boards — Arduino, ESP32-S3, STM32N6); the latency distribution (median/p95/p99) is taken host-side across all 11,005 samples. Single-input variance is <0.1 ms on every board (and ~0.02 ms on STM32N6), so N=1 is statistically tighter than the MLPerf same-input×100 convention at this sample count.

---

## Measurement checklist

- [ ] Correct model format per board: `.tflite` for Arduino/ESP32-S3; `.onnx` for STM32/Cube.AI
- [ ] Test vectors generated per model+backend (`prepare_test_vectors.py --model X --backend Y`)
- [ ] MFCC computed on host (Python), tensors sent over UART — no mic needed
- [ ] Latency = one timed Invoke per sample (N=1 on all boards); distribution taken host-side over all samples
- [ ] RAM = peak activation memory (Cube.AI reports at compile time; TFLite Micro: arena_used from boot JSON)
- [ ] Flash = model binary size in bytes
- [ ] STM32N6: run each model with NPU ON and NPU OFF, report both
- [ ] All 3 models on all 3 boards before writing results table
