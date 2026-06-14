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
- DS-CNN NPU: 18 epochs, 15 HW / 3 SW (DequantizeLinear, GlobalAveragePool, QuantizeLinear).

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

## Firmware status

Hand-written, CPU path (`firmware/stm32n6/src/`): `main.c` (800 MHz bringup, USART1 PE5/PE6
@115200, cache/MPU), `kws_bench.c` (protocol loop, DWT cycle timing, 100 runs/sample, median),
`stm32n6xx_it.c`. Targets the X-CUBE-AI `ai_network_*` API (CPU builds only).

NPU firmware (LL_ATON) + external-flash programming: not yet started.

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
| DS-CNN / TC-ResNet8 NPU | generated, firmware TODO | — | — | — | — |

All three also passed 5-sample smoke tests (100% agreement) before the full runs.

## Plan
1. ✅ Generate model C code (5/6).
2. ✅ CPU path COMPLETE: build system, smoke tests, and full 11,005-sample runs for all 3 models
   (0 failures, ≥99.9% MCU↔ONNX agreement). Detail in `RESULTS_CPU.md`.
3. ⏳ NPU path: LL_ATON firmware + external-flash programming for DS-CNN, TC-ResNet8.
4. ⏸  GRU-96 NPU: revisit per routes above (parked).
