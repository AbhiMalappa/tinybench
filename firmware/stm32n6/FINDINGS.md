# STM32N6570-DK KWS Benchmark — Findings & Status

Board: STM32N6570-DK (Cortex-M55 @ 800 MHz, Neural-ART NPU), no internal flash (FSBL arch).
Toolchain: ST Edge AI Core v2.2.0 / X-CUBE-AI 10.2.0-RC1 (`stedgeai`), arm-none-eabi-gcc 14.3.
Protocol: `kws/host/protocol.md` (host sends 1960 float32 bytes/sample; output 35 INT8 logits).

## Code-generation matrix (stedgeai → C)

| Model | CPU (`--target stm32`) | NPU (`--target stm32n6 --st-neural-art profile-default`) |
|---|---|---|
| DS-CNN     | ✅ | ✅ |
| TC-ResNet8 | ✅ | ✅ |
| GRU-96     | ✅ | ❌ infeasible (see below) |

### CPU build characteristics (X-CUBE-AI `ai_network_*` runtime)
| Model | Model input fmt | Activations (RW) | Output | Input quant (scale, zp) |
|---|---|---|---|---|
| DS-CNN     | INT8, 490 B  | 141,056 B | 35×INT8 | 0.043562427, 3 |
| TC-ResNet8 | float32, 1960 B | 8,992 B | 35×INT8 | (model ingests float; QuantizeLinear internal) |
| GRU-96     | float32, 1960 B | 23,464 B | 35×INT8 | (model ingests float) |

Output quant: DS-CNN QLinear(0.163713858, 6); TC-ResNet QLinear(0.040224735, -22); GRU QLinear(0.040211614, -46).

Implication for firmware: host always streams 1960 float32 bytes. DS-CNN firmware quantizes
those 490 floats → 490 INT8 before `ai_network_run`; TC-ResNet & GRU feed the 1960 float bytes
directly. The benchmark loop branches on `AI_NETWORK_IN_1_SIZE_BYTES` (490 vs 1960).

### NPU build characteristics (LL_ATON runtime) — fundamentally different from CPU
- API: `LL_ATON_*` (not `ai_network_*`). Generated `network.h` exposes `LL_ATON_DEFAULT_*`.
- Weights live in an **external-flash blob** `network_atonbuf.xSPI2.raw` (~43 KB for DS-CNN),
  mapped at **0x71000000** (octo-flash). Activations in `npuRAM` banks.
- Requires a separate firmware (LL_ATON runtime sources from `Middlewares/ST/AI/Npu/ll_aton/`)
  AND an external-flash programming step (STM32CubeProgrammer + MX66 octo-flash loader, signed
  FSBL boot-from-flash) — i.e. cannot run from the simple SRAM-debug model used for CPU builds.
- DS-CNN NPU: 18 epochs, 15 HW / **3 SW (DequantizeLinear, GlobalAveragePool, QuantizeLinear)** — the
  SW global-pool epoch is the incompatibility (≤3×3 HW pooling window); see the DS-CNN FINDING below.

## FINDING: GRU-96 is infeasible on the Neural-ART NPU (as exported)

`stedgeai generate --target stm32n6 --st-neural-art` aborts with:

```
TOOL ERROR: Shape and shape map lengths must be the same:
            [49, 1, 1, 1, 96] vs. (H, BATCH, W, CH)
```

Cause: the Neural-ART NPU's tensor model is 4-D — `(H, BATCH, W, CH)`. The standard PyTorch
GRU ONNX export is sequence-major with an explicit `num_directions` axis and produces a 4-D
sequence output `Y = [seq=49, dir=1, batch=1, hidden=96]`, which the Neural-ART mapper expands
to a 5-D internal tensor `[49,1,1,1,96]` that does not fit the 4-D layout.

Verification performed:
- `supported-ops --target stm32n6` lists GRU as a *supported* op — so this is not a blanket
  op-unsupported case; it is a **layout/shape** incompatibility for this export.
- Replacing the GRU's dynamic initial-hidden-state construction
  (`Shape→Gather→Concat→ConstantOfShape`) with a static zero `h0` did **not** resolve it —
  confirming the 5-D tensor originates from the GRU sequence output itself, not the h0 path.

Significance (paper): this mirrors the documented **Arduino/TFLite-Micro GRU-96 infeasibility**
(840-op unrolled GRU exceeds the per-subgraph buffer-tracking cap). Two independent embedded
ML stacks — TFLite Micro on Cortex-M4 and Neural-ART on Cortex-M55 — both reject the same
recurrent model for structural reasons, while both CNNs (DS-CNN, TC-ResNet8) deploy cleanly.
This strengthens the cross-platform "recurrent models are second-class citizens on current
TinyML accelerators/runtimes" narrative.

Parked, not closed. Possible future routes (not yet attempted), in rough order of effort:
1. Re-export the GRU from PyTorch with NPU-friendly settings (batch-major `layout=1`,
   drop the unused `Y` sequence output, keep only `Y_h`).
2. ONNX surgery: remove the unused GRU `Y` output and force a 3-D/4-D-compatible layout.
3. Manual time-step unrolling into Gemm/elementwise ops the NPU maps natively.

GRU-96 NPU build artifacts: `kws_gru_npu/` is intentionally empty (build produced no output).

## FINDING: DS-CNN's 2-D global average pooling cannot map to the Neural-ART NPU HW

DS-CNN runs on the NPU but returns **numerically wrong output** (~21% acc / ~21% agreement vs the
~92% expected; the NPU logits show near-zero/negative correlation with the reference — structurally
scrambled, not noisy). After the RISAF firewall fix that made the NPU run real inference, this was
the one remaining gap. **Definitively root-caused (2026-06-15) to DS-CNN's head, NOT the firmware.**

### Root cause — a hardware pooling-window limit
DS-CNN's head is a **2-D global average pool over a [50×11] feature map** (`AdaptiveAvgPool2d(1)`,
exported as `ReduceMean axes=[-1,-2]`). Per ST's Neural-ART operator doc
(`stedgeai .../Documentation/stneuralart_operator_support.html`):
- **AveragePool / GlobalAveragePool in HW is limited to ≤3×3 windows** ("Kernel Height and Width
  between 1 and 3 included"; larger windows "will be decomposed by the compiler"), plus a pooling
  line-buffer limit of `width × input_channels ≤ 2048`.
- `ReduceMean` maps to HW only "if it can be converted to GlobalAveragePool" — which still hits the
  same window limit.

A [50×11] (= 550-element) global pool vastly exceeds the ≤3×3 window, and **50 (=2·5²) and 11 (prime)
cannot be tiled exactly by ≤3 kernels**, so the compiler emits a **software-fallback float epoch**:
`DequantizeLinear → (Global)AveragePool → QuantizeLinear` (epochs 14–16 of 18). On this toolchain
(ST Edge AI Core 2.2.0 / X-CUBE-AI 10.2.0-RC1) the resulting **HW→SW→HW handoff produces the scrambled
output**. The SW epoch also forms a float "island" that pulls neighbouring ops into software.

### Proven NOT firmware (the decisive test)
The SAME NPU firmware runs **TC-ResNet8 at 95% acc / 100% agreement / 0.649 ms**. TC-ResNet8 is a
**1-D temporal CNN**: its pre-pool feature map is **`[48, 7]` (width 7)**, so its `GlobalAveragePool`
decomposes into ≤3×3 HW pools and stays in hardware (only 1 SW epoch = the input quantize). The NPU
favours temporal (1-D) CNNs; a standard 2-D CNN that keeps a large spatial map until a global pool
does not map.

### Ruled out by experiment (6 in-place ONNX-surgery variants — all fall back to SW)
Each generated with `stedgeai … --st-neural-art` and checked via the epoch report; every one lands in
the software float island (artifacts in `kws_dscnn_npu_{hwpool,gap,gap3d,hier,convpool}/` and
`checkpoints/dscnn_int8_v17_*.onnx`):
1. reshape to 1-D `[b,64,1,550]` + `ReduceMean`
2. `ReduceMean` → `GlobalAveragePool`
3. 3-D reshape `[b,64,550]` + `GlobalAveragePool` (output `[b:1,h:1,c:64]`, identical form to TC-ResNet's)
4. hierarchical small pools (`AveragePool` kernels 11 → 10 → 5)
5. depthwise averaging `Conv2d` (kernel 50×11, weight 1/550) — large conv kernel also falls back
6. `AveragePool` 3×3 — also SW (the float island swallows it)

### Significance (paper) & resolution
This is a **second model–accelerator incompatibility alongside GRU-96**: as exported, DS-CNN is not
deployable on the Neural-ART NPU without architectural change — and the reason (the accelerator's
≤3×3 pooling window favouring downsampled / 1-D feature maps) is itself an interesting cross-model
result.

**RESOLVED (2026-06-16) — DS-CNN v2.** A single change fixes it: a **stride-(2,2) stem** shrinks the
pre-pool map from 50×11 (550, SW) to **25×6** (150) — under the empirically-measured ~300-element HW
global-pool ceiling — so the model becomes **fully HW (0 software epochs)** with no other change (whole
DS-CNN body, BN, ReLU, Gemm already map to HW). Re-trained on Kaggle GPU (early-stop epoch 123,
val 93.39%), re-quantized (INT8 92.08% host), regenerated (`stedgeai` report: 14 epochs, **0 SW**),
and deployed. Full NPU run: **92.40% acc / 98.26% agreement / 0.463 ms / 0 failures over 11,005** —
accuracy on par with the original (92.71%), and the scramble is gone (98.26% vs the old ~21%). The
HW ceiling probe found 7×2/13×3/25×6/25×11/50×6 all fully HW; 25×6 chosen (fastest, 3.5 M MACC, −70%).
Model change is one line in `kws/models/dscnn.py` (`stem_stride=(2,2)`); see RESULTS_CPU.md NPU section
and `NEURAL_ART_MODELING_CONSTRAINTS.md`. (The original 50×11 DS-CNN remains the documented
incompatibility above — both are kept for the paper's narrative.)

## Firmware status

Hand-written, CPU path (`firmware/stm32n6/src/`): `main.c` (800 MHz bringup, USART1 PE5/PE6
@115200, cache/MPU), `kws_bench.c` (protocol loop, DWT cycle timing, 100 runs/sample, median),
`stm32n6xx_it.c`. Targets the X-CUBE-AI `ai_network_*` API (CPU builds only).

NPU firmware (LL_ATON), `src_npu/`: DONE and running. `main_npu.c` (RISAF NPU-firewall open, NPU+cache
clocks, XSPI mem-map for ext-flash, NPU @800 MHz), `kws_bench_npu.c` (LL_ATON epoch-loop driver matched
to ST `ai_wrapper_ATON`). **TC-ResNet8 NPU complete: full 11,005-sample run = 93.02% acc / 99.68%
agreement / 0.649 ms median / 0 failures (17.2× vs CPU).** DS-CNN NPU
runs but is numerically wrong (~21%) — definitively root-caused to its 2-D global-average-pool head
exceeding the Neural-ART ≤3×3 HW pooling window (forces a SW-fallback float epoch), NOT the firmware
(proven: same firmware runs TC-ResNet8 at 100% agreement). See the dedicated **FINDING** section above
and the full bring-up story in `NPU_BRINGUP_JOURNEY.md`.

## Build & run (CPU path) — working recipe

Build: `make MODEL=dscnn|tcresnet|gru`. Required toolchain facts (all resolved):
- `-mcmse` → CMSIS defines `CPU_IN_SECURE_STATE` → secure PWR clock macros become available.
- Provide `__errno` in `src/syscalls.c` (libm needs it; absence shows as a misleading
  "dangerous relocation / Unknown destination type" link error, not a plain undefined-symbol).
- `-DUSE_FULL_LL_DRIVER -DSTM32N657xx`; own minimal `system_stm32n6xx.c` (FPU enable only).
- `power_init()` (RCC->BUSENSR + HAL_PWREx_EnableVddA/VddIO2..5 + ConfigSupply) MUST run before
  UART, or PE5/PE6 stay dark. USART1 PE5/PE6 AF7 = ST-Link VCP (/dev/cu.usbmodem14102 @115200).

Run (no internal flash → load to SRAM and run). Use **mode=UR** (under hardware reset):
```
STM32_Programmer_CLI -c port=SWD mode=UR -halt \
  -d build/<MODEL>_cpu/Project.elf -coreReg MSP=0x34200000 PC=<Reset_Handler> -run
```
- `--start`/`-g` RESETS → N6 bootrom takes over (flash-boot) and the SRAM image never runs
  (PC ~0x18003xxx). Use `-coreReg PC=<ResetHandler> -run` instead.
- **mode=HOTPLUG is flaky** — if a prior firmware faulted the core into LOCKUP (PC=0xEFFFFFFE),
  HOTPLUG download fails ("failed to download Sector[0]") and software reset can't recover.
  **mode=UR asserts NRST → clean halted core, recovers lockup, SRAM writable.** Always use mode=UR.
- Reset_Handler from `arm-none-eabi-nm Project.elf | grep 'W Reset_Handler'` (changes each build);
  MSP = _estack = 0x34200000. Boot/init must avoid SysTick (HAL_Delay) and DWT spin-waits — both
  can be non-functional right after the bootrom hand-off; SMPS ramp uses a plain busy-loop.
  (ST-LINK_gdbserver handshake is version-broken with the bundled gdb — use CubeProgrammer.)

## Results — CPU full runs (N=11,005 @ 800 MHz). Detail: `RESULTS_CPU.md`

| Config | Status | MCU acc | MCU↔ONNX agreement | Median latency | Failures |
|---|---|---|---|---|---|
| TC-ResNet8 CPU | ✅ full complete | 93.02% | 99.99% | 11.16 ms | 0/11005 |
| DS-CNN CPU | ✅ full complete | 92.71% | 99.98% | 249.27 ms | 0/11005 |
| GRU-96 CPU | ✅ full complete | 92.63% | 99.90% | 194.39 ms | 0/11005 |
| TC-ResNet8 NPU | ✅ complete (N=11,005) | 93.02% | 99.68% | 0.649 ms (17.2×) | 0 / 11005 |
| DS-CNN NPU (v2 HW re-arch) | ✅ complete | 92.40% | 98.26% | 0.463 ms (538×†) | 0 / 11005 — stride-(2,2) stem → 25×6, fully HW |
| DS-CNN NPU (original 50×11) | ❌ incompatible | ~21% | ~21% | 1.36 ms | 2-D global pool > ≤3×3 HW window → SW-fallback (see FINDING) |

All three also passed 5-sample smoke tests (100% agreement) before the full runs.

## Plan
1. ✅ Generate model C code (5/6).
2. ✅ CPU path COMPLETE: build system, smoke tests, and full 11,005-sample runs for all 3 models
   (0 failures, ≥99.9% MCU↔ONNX agreement). Detail in `RESULTS_CPU.md`.
3. ⏳ NPU path: LL_ATON firmware + external-flash programming for DS-CNN, TC-ResNet8.
4. ⏸  GRU-96 NPU: revisit per routes above (parked).
