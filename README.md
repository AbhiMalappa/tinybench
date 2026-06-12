# TinyBench-KWS

Cross-platform keyword spotting benchmark on 3 MCU boards.
Part of the TinyBench series — Paper 1.

**ML pipeline (Python):** Training, quantization, benchmark orchestration
**Firmware (C):** STM32CubeIDE / TFLite Micro inference, UART protocol, cycle-counter timing

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
| DS-CNN-M | 25,251 | 92.71% | 92.70% | **92.15%** ✅ | `dscnn_int8.onnx` | `dscnn_int8.tflite` |
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

## Partner deployment guide

### STM32N6570-DK — Cube.AI path

**Model file:** `kws/checkpoints/dscnn_int8.onnx`

1. Open STM32CubeIDE → New STM32 Project → STM32N6570-DK board
2. Add X-CUBE-AI middleware pack
3. In Cube.AI: import `dscnn_int8.onnx` → validate → generate C code
4. Run each model **twice**: NPU enabled and CPU-only (toggle in Cube.AI config)
5. Wire up `board_hal.h` (see below)

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

**Latency definition:** median of N inference runs on the same input tensor (N is firmware-defined: Arduino=10, ESP32-S3=1, STM32=TBD). Single-run variance on ESP32-S3 is <0.1 ms so N=1 is sufficient there.

---

## Measurement checklist

- [ ] Correct model format per board: `.tflite` for Arduino/ESP32-S3; `.onnx` for STM32/Cube.AI
- [ ] Test vectors generated per model+backend (`prepare_test_vectors.py --model X --backend Y`)
- [ ] MFCC computed on host (Python), tensors sent over UART — no mic needed
- [ ] Latency = median of N runs per sample (N per firmware: Arduino=10, ESP32-S3=1, STM32=TBD)
- [ ] RAM = peak activation memory (Cube.AI reports at compile time; TFLite Micro: arena_used from boot JSON)
- [ ] Flash = model binary size in bytes
- [ ] STM32N6: run each model with NPU ON and NPU OFF, report both
- [ ] All 3 models on all 3 boards before writing results table
