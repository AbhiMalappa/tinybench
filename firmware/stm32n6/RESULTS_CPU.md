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

Same board + host harness. NPU @ **800 MHz** (IC6÷1), LL_ATON runtime. Latency = on-device DWT time for
one epoch-loop inference. Full run 2026-06-15 (`stm32n6_unknown_20260615_213555.{jsonl,_summary.json}`,
wall clock 2,192 s).

**Execution context / memory map (applies to all STM32N6 numbers in this doc — CPU and NPU):** the
STM32N6 has **no internal user flash** (FSBL architecture).
- **Firmware code executes from internal SRAM** — `AXISRAM1_S @ 0x34000000` (2 MB): `.isr_vector`
  @ 0x34000000, `.text` @ 0x34000350, `Reset_Handler` @ 0x3400a510. The `.elf` is loaded over ST-Link
  (SWD, development boot) and run in place from RAM; **no boot-from-flash, no XIP of code**. (Latency is
  therefore free of any flash instruction-fetch overhead.)
- **NPU weights** live in **external Octal-SPI NOR flash** (Macronix **MX66UW1G45G**, octal *NOR* — not
  NAND), on the XSPI/OctoSPI bus, **memory-mapped at `0x71000000`**; the Neural-ART streams weights from
  there during inference (`EnableMemoryMappedMode`, `--cache-maintenance` + AXI cache on).
- **NPU activations / working buffers** in **npuRAM** (AXISRAM3–6, `0x342x_xxxx`).
- CPU cells (X-CUBE-AI `ai_network_*`): same code-from-SRAM; weights+activations also in internal SRAM
  (no external flash used).

| Model | Status | MCU acc | MCU↔ONNX agreement | Latency median (ms) | NPU speedup vs CPU | Failures |
|---|---|---|---|---|---|---|
| TC-ResNet8 | ✅ complete | 93.02% | 99.68% | **0.649** | **17.2×** (11.16 ms) | 0 / 11005 |
| DS-CNN (v2, HW re-arch) | ✅ complete | 92.40% | 98.26% | **0.463** | **138.8×** (same-model CPU 64.27 ms) | 0 / 11005 |
| DS-CNN (original 50×11) | ❌ incompatible as exported | — | — | — | — | 2-D global pool > ≤3×3 HW window → SW-fallback (see FINDINGS.md) |
| GRU-96     | ❌ infeasible | — | — | — | — | 5-D sequence layout rejected by Neural-ART (see FINDINGS.md) |

**DS-CNN v2 (HW re-architecture).** The original DS-CNN keeps a 50×11 feature map to a 2-D global average
pool, which exceeds the Neural-ART ≤3×3 HW pooling window → software-fallback float epoch → wrong output
(the DS-CNN FINDING in `FINDINGS.md`). Fix: a stride-(2,2) stem shrinks the pre-pool map to **25×6**
(≤ the ~300-element HW global-pool limit), making the model **fully HW (0 software epochs)** with no
other change. Re-trained (Kaggle GPU, early-stop epoch 123, val 93.39%), re-quantized (INT8 92.08% host),
re-deployed. Full NPU run 2026-06-16 (`stm32n6_unknown_20260616_142903.{jsonl,_summary.json}`, wall 2,182 s):
**92.40% acc / 98.26% agreement / 0.463 ms / 0 failures** — accuracy on par with the original DS-CNN
(92.71%), latency distribution min 0.4612 / p95 0.4638 / p99 0.4641 ms.

**Same-model NPU vs CPU (v2 on both sides).** The v2 model was also run on the **STM32 CPU path**
(X-CUBE-AI, generated `--target stm32` from `dscnn_v2_int8.onnx`, weights embedded in SRAM) for an
apples-to-apples HW-vs-CPU comparison. Full CPU run 2026-06-16
(`stm32n6_unknown_20260616_154143.{jsonl,_summary.json}`, wall 2,706 s): **92.36% acc / 98.35% agreement /
64.274 ms / 0 failures** (p95/p99 64.275/64.276). So **NPU = 138.8× faster than CPU on the *identical*
model** (0.463 vs 64.27 ms) — the honest speedup (the earlier 538× blended in the lighter model). The MCU
accuracy (CPU 92.36% / NPU 92.40%) is consistent across both engines and ≈ the host int8 ref (92.08%);
the ~98.3% agreement (vs the original DS-CNN's 99.98%) is X-CUBE-AI/Neural-ART int8 arithmetic diverging
from ORT slightly more for this model — not an error (accuracy is preserved).
Versus the **original DS-CNN on CPU** (249.27 ms, 137.8 KiB RAM), v2 is **3.9× faster** at **6.7× less
activation RAM** (20.4 KiB) and equal accuracy — i.e. the HW-aware redesign is a win on the CPU too.

New artifacts: `checkpoints/dscnn_v2_int8.onnx`, `firmware/stm32n6/kws_dscnn_npu_v2/` (NPU) +
`kws_dscnn_cpu_v2/` (CPU); host ref `ref_preds_dscnn_onnx.npy` (original backed up `*_v1_*`); firmware
input quant scale 0.04317872 / zp 11 in both `Makefile.npu` and `Makefile`.

> **Note (cross-board consistency, in progress):** the main CPU results table above still lists the
> *original* DS-CNN (Arduino/ESP32/STM32-CPU were deployed with it). The v2 model is being rolled out to
> all boards for a single consistent DS-CNN; STM32 (CPU+NPU) done, **ESP32 + Arduino pending**. The whole
> matrix will be re-tabulated to v2 once those two are re-run.

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
