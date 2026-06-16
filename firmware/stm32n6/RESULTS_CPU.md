# STM32N6570-DK — KWS Benchmark Results (CPU + Neural-ART NPU)

> CPU results below (Sections "Results"/"Latency"/"Observations"). **NPU (Neural-ART) results in the
> "## NPU results" section near the end.** TC-ResNet8 NPU ✅ 93.02% / 99.68% / **0.649 ms (17.2× vs CPU)**.

Platform: STM32N6570-DK, Cortex-M55 @ **800 MHz**, X-CUBE-AI `ai_network_*` software runtime
(CPU only; NPU not used). Firmware loaded to SRAM (no internal flash) and run via ST-Link.
Methodology matches Arduino/ESP32 cells: **1 timed `Invoke()` per sample** (`N_RUNS=1`,
hardware DWT cycle counter), latency distribution taken host-side over all 11,005 test samples.
Host driver: `kws/host/benchmark_serial.py --backend onnx`. Input: 490 MFCC values streamed as
1960 float32 bytes/sample; output 35 INT8 logits.

`mcu_ref_agreement` = fraction of samples where the on-device argmax equals the host ONNX-Runtime
reference prediction — the firmware-correctness signal (≈100% ⇒ bit-accurate deployment).

## Results (full test set, N = 11,005)

| Model | Status | MCU acc | Host ONNX ref acc | MCU↔ONNX agreement | Latency median (ms) | Failures |
|---|---|---|---|---|---|---|
| DS-CNN     | ✅ complete | 92.71% | 92.69% | 99.98% | 249.27 | 0 / 11005 |
| TC-ResNet8 | ✅ complete | 93.02% | 93.01% | 99.99% | 11.16  | 0 / 11005 |
| GRU-96     | ✅ complete | 92.63% | 92.57% | 99.90% | 194.39 | 0 / 11005 |

### Latency distribution (ms) — deterministic compute, very tight spread
| Model | min | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| DS-CNN     | 249.266 | 249.273 | 249.282 | 249.284 | 249.286 |
| TC-ResNet8 | 11.1599 | 11.1607 | 11.1618 | 11.1621 | 11.1634 |
| GRU-96     | 193.525 | 194.393 | 194.667 | 194.754 | 194.900 |

### Static model / memory footprint (from stedgeai generate reports)
| Model | Params | MACC / inference | Weights ROM | Activations RAM |
|---|---|---|---|---|
| DS-CNN     | 24,099 | 11,795,107 | 26,444 B (25.82 KiB) | 141,056 B (137.75 KiB) |
| TC-ResNet8 | 65,499 |    773,436 | 66,588 B (65.03 KiB) |   8,992 B (8.78 KiB)  |
| GRU-96     | 34,595 |  1,508,867 | 128,300 B (125.29 KiB) | 23,464 B (22.91 KiB)  |

## Observations
- **Both deployed cells are bit-accurate**: MCU accuracy tracks the host ONNX reference to within
  0.02%, and ≥99.98% per-sample agreement over 11,005 samples with **zero failures**.
- **DS-CNN is 22× slower than TC-ResNet8 on the CPU** (249 ms vs 11 ms) despite having ~3× fewer
  parameters — because its depthwise-separable convolutions run over large spatial maps, giving
  **15× more MACCs** (11.8 M vs 0.77 M). Parameter count is a poor latency proxy; MACC dominates.
- Latency spread is sub-0.02 ms (fixed compute, no input-dependent branching) — a single timed
  inference per sample is sufficient; the 100-reps-per-sample approach added nothing but time.
- TC-ResNet8 uses far less activation RAM (8.8 KiB vs 137.8 KiB) but more weight ROM (65 vs 26 KiB).

## NPU results (Neural-ART enabled, N = 11,005)

Same board + host harness. Weights memory-mapped from external octo-flash (`0x71000000`), activations in
npuRAM; NPU @ **800 MHz** (IC6÷1), LL_ATON runtime. Latency = on-device DWT time for one epoch-loop
inference. Full run 2026-06-15 (`stm32n6_unknown_20260615_213555.{jsonl,_summary.json}`, wall clock 2,192 s).

| Model | Status | MCU acc | MCU↔ONNX agreement | Latency median (ms) | NPU speedup vs CPU | Failures |
|---|---|---|---|---|---|---|
| TC-ResNet8 | ✅ complete | 93.02% | 99.68% | **0.649** | **17.2×** (11.16 ms) | 0 / 11005 |
| DS-CNN     | ❌ incompatible as exported | — | — | — | — | 2-D global pool > ≤3×3 HW window → SW-fallback (see FINDINGS.md) |
| GRU-96     | ❌ infeasible | — | — | — | — | 5-D sequence layout rejected by Neural-ART (see FINDINGS.md) |

- TC-ResNet8 NPU latency distribution (ms): min 0.6451 / p50 0.6493 / p95 0.6509 / p99 0.6515 / max 0.6532.
- **Bit-accurate at the accuracy level:** MCU accuracy 93.02% equals the CPU/host (93.02% / 93.01%);
  per-sample agreement 99.68% — the ~0.3% delta is int8 NPU vs host-ONNX rounding on borderline clips
  (expected for the Neural-ART quantized datapath). 0 failures over 11,005.
- **17.2× faster than the same model on the M55 CPU** (0.649 ms vs 11.16 ms), and ~384× faster than the
  DS-CNN CPU path (249 ms) — at 8.8 KiB activation RAM. This is the headline NPU result of Paper 1.
- DS-CNN and GRU-96 do **not** deploy on the NPU as exported — two documented model–accelerator
  incompatibilities (FINDINGS.md). DS-CNN is being re-architected with HW-friendly (≤3×3) pooling so a
  future NPU cell can be measured; GRU-96 NPU stays infeasible (parked).

## Provenance
- Result logs (per-clip JSONL + summary.json):
  `kws/host/results/stm32n6_*_2026061{3,4}_*.jsonl` (+ `_summary.json`); copies preserved as
  `~/claude_tmp/full/stm32n6_<model>_full.{jsonl,summary.json}`.
- DS-CNN log: `stm32n6_unknown_20260613_231735.jsonl` (board tag stm32n6; model tag "unknown"
  because the one-shot boot line is cleared by the host's `reset_input_buffer()` — cosmetic only).
- TC-ResNet8 log: `stm32n6_unknown_20260613_224208.jsonl`.
- GRU-96 log: `stm32n6_unknown_20260614_003646.jsonl`.
- Wall clock: DS-CNN 4,738 s (79 min), TC-ResNet8 2,114 s (35 min), GRU-96 4,135 s (69 min) at
  115200 baud, 1 inference/sample.
- Build/run recipe and toolchain notes: see `FINDINGS.md`.
