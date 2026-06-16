# STM32N6 Neural-ART NPU Bring-up — Journey, Blocks & Resolutions

High-level log of getting the KWS models running on the STM32N6570-DK **Neural-ART NPU**
(via the LL_ATON runtime). Companion to `FINDINGS.md` (overview) and `RESULTS_CPU.md` (CPU results).
Last updated: 2026-06-15.

> TL;DR — The NPU now runs. **TC-ResNet8 NPU works perfectly (95% acc, 100% agreement, 0.649 ms,
> ~17× vs CPU).** DS-CNN NPU runs at full speed (1.36 ms, ~183× vs CPU) but is numerically wrong
> (~21%) — **definitively root-caused** to its **2-D global average pool over a [50×11] map
> exceeding the Neural-ART ≤3×3 HW pooling window**, which forces a software-fallback float epoch
> (`Dequant→Pool→Quant`) whose HW→SW→HW handoff scrambles the output. NOT firmware (same firmware
> runs TC-ResNet8 — a 1-D temporal CNN, width-7 pool — at 100%). Fix = re-architect DS-CNN with
> spatial downsampling so the pre-pool map is small (retrain on GPU); see `FINDINGS.md` DS-CNN FINDING.

---

## Current status

| Model | NPU result | Latency | vs CPU | State |
|---|---|---|---|---|
| **TC-ResNet8** | **95.0% acc / 100% agreement** | **0.649 ms** | ~17× (11.16 ms) | ✅ DONE, paper-ready |
| **DS-CNN** | ~21% acc / ~21% agreement (wrong) | 1.36 ms | ~183× (249 ms) | ❌ incompatible as exported (HW pool ≤3×3) |
| GRU-96 | — | — | — | ❌ infeasible on NPU (documented) |

Open task: **re-architect** DS-CNN with spatial downsampling so the pre-pool map is small enough that
the global pool maps to HW (≤3×3 windows), then **retrain on GPU** → requantize → regenerate → deploy.
Six in-place ONNX-surgery variants all fail (pooling/large-conv windows > the HW limit force a SW
float epoch) — see the DS-CNN FINDING in `FINDINGS.md`. Pure surgery cannot fix it; the model's
spatial resolution must shrink before the pool.

---

## The platform gotchas (these cost the most time — read first)

1. **No internal flash.** Everything runs by loading firmware into **SRAM** over ST-Link and
   jumping to it: `STM32_Programmer_CLI -c port=SWD mode=UR -halt -d Project.elf
   -coreReg MSP=0x34200000 PC=<Reset_Handler> -run`. **Never `--start`/`-g`** (resets → bootrom
   takes over → SRAM image never runs). Reset_Handler addr changes every build (`arm-none-eabi-nm`).
2. **`mode=UR` always** (connect under hardware reset). `mode=HOTPLUG` is flaky and can't recover a
   locked-up core.
3. **Boot switch must be in "development".** If it (or a wedged state) flips to boot-from-flash, the
   device reports **generically** ("Rev Z" / "STM32N6xx") and **denies all memory access** — looks
   like a security lock but is just the boot mode. Healthy = "Rev B" / "ST32N657".
4. **Recoverable lockups need a physical USB power-cycle.** A wedged core (e.g. after a fault, or
   after flash-program + failed download) can enter the Rev-Z/denied state that **survives software
   reset**. Only unplug/replug USB (or a boot-switch toggle) clears it.
5. **STALE-BUILD TRAP.** `make -C` / incremental builds silently keep old objects. **Always
   `rm -rf build/<target> && make ...`** and verify with `strings Project.elf | grep <new marker>`.
   Hours were lost testing stale binaries.

---

## Block → Root cause → Resolution (chronological)

### 1. Firmware won't even boot to UART
- **Cause:** N6-specific power/clock bring-up. Needs `-mcmse` (→ `CPU_IN_SECURE_STATE` for secure PWR
  macros), a self-provided `__errno` (libm), `-DUSE_FULL_LL_DRIVER`, and `power_init()`
  (`RCC->BUSENSR=0xFFFFFFFF` + `HAL_PWREx_EnableVddA/VddIO2..5`) before UART.
- **Resolution:** documented in `FINDINGS.md` / `RESULTS_CPU.md`; CPU path went green first.

### 2. (NPU) RISAF/RIF config locked out the debugger
- **Symptom:** after the first NPU firmware ran, the ST-Link could no longer read/write SRAM;
  survived software reset; needed a physical power-cycle.
- **Cause:** the firmware reconfigured the **SRAM firewalls (RISAF2/3 — the code-load region)**,
  copied from ST's signed-FSBL example, which cut off the debugger's own access path.
- **Resolution:** never reconfigure RISAF2/3. (See block #6 for the RISAF we *do* need.)

### 3. (NPU) Weights couldn't be staged to npuRAM at load time
- **Cause:** npuRAM banks (AXISRAM3-6 @ 0x342E0000) are **not enabled by the bootrom**, so a direct
  ST-Link write there at load time faults.
- **Resolution (SRAM profile):** embed weights as a `.rodata` blob in the firmware (loaded into
  bootrom-enabled AXISRAM1) and have the firmware copy them to npuRAM after `memory_init()`.
  **`__HAL_RCC_RAMCFG_CLK_ENABLE()` must precede the RAMCFG_SRAMx CR writes** or the banks never wake.
- **Resolution (EXTMEM profile):** put weights in external octo-flash @ 0x71000000 and memory-map
  via `BSP_XSPI_NOR_Init` + `EnableMemoryMappedMode`; no copy needed.

### 4. (NPU) Output saturated at constant 127, ignored the input  ← the big one
- **Symptom:** NPU ran (~11 ms) but every logit = 127 → always class 0, identical for every input.
  Both weight profiles (SRAM + EXTMEM) failed identically.
- **Misleads ruled out (a lot of time):** input bytes (correct), weights (byte-perfect in flash),
  output scale (correct), cache, BUSIF keys, EC-blob loader (no error), ST driver order, MPU,
  reset-vs-input ordering, NPU clock speed. None of these were it.
- **ROOT CAUSE:** the **NPU master memory firewalls (RISAF4/5) and external-flash firewall (RISAF12)**
  are left restrictive (secure CID=1 only) by the dev-boot bootrom → the Neural-ART's reads are
  **denied → it computes on zeros → constant saturated output**.
- **RESOLUTION:** open the **NPU-specific** firewalls only (RISAF4 = NPU MST0, RISAF5 = NPU MST1,
  RISAF6 = npuRAM, RISAF12 = octo-flash for EXTMEM), via ST's fully-permissive `set_risaf_default`
  pattern (`HAL_RIF_RISAF_ConfigBaseRegion`). **Crucial:** leave RISAF2/3 (SRAM code region) alone so
  the debugger reload path stays intact (no power-cycle on a hung firmware). Two sub-gotchas:
  - RISAF regs fault unless `__HAL_RCC_RIFSC_CLK_ENABLE()` + `__HAL_RCC_RISAF_CLK_ENABLE()` first.
  - **RISAF4/5 fault unless the NPU is clocked first** (`__HAL_RCC_NPU_CLK_ENABLE()`) — they are the
    NPU's own master firewalls.
- **Effect:** output became **data-dependent and partially correct**. NPU went from 0% to running.

### 5. (NPU) Output now data-dependent but ~24% (scrambled, not noisy)
- **Investigation:** NPU logits had near-zero/negative correlation with the reference → structurally
  wrong, not rounding. Verified the firmware is faithful to ST's `ai_wrapper_ATON` and all 18 epochs
  run. Cacheable-vs-non-cacheable npuRAM and NPU-cache on/off changed only latency, not correctness.
  Matched ST: **no MPU region (npuRAM left Normal write-back cacheable), D-cache + NPU cache on, the
  generated `--cache-maintenance` schedule handles coherency.** Latency dropped to 3.4 ms.
- **Measurement gotcha (important):** the test vectors are **not shuffled** — the first ~165 are all
  class 22. Early "agreement %" numbers were on a single-class subset. **Always evaluate on a random
  diverse subset.** Data/reference verified sound: reference ONNX = 91.3% acc / 100% vs ref_preds.

### 6. (NPU) Why DS-CNN is wrong but TC-ResNet8 is perfect  ← the decisive test
- **Test:** ran the *other* model (TC-ResNet8) on the *same* firmware → **95% / 100% / 0.649 ms.**
- **Conclusion:** **the firmware is correct.** The DS-CNN problem is **model/generation-specific.**
- **Pinpointed cause:** epoch structure differs. TC-ResNet8 = 38 epochs, **1 SW epoch** (input
  QuantizeLinear, at the start). DS-CNN = 18 epochs, **3 SW epochs in the middle**
  (Dequantize → **GlobalAveragePool** → Quantize). DS-CNN's 2-D AdaptiveAvgPool2d falls back to
  **software in the middle of the graph**; that HW→SW→HW handoff is where it breaks. TC-ResNet8's
  1-D pool maps to hardware, so it never hits this path.
- **Things that did NOT fix DS-CNN (confirming it's the SW-pool, not config):** `profile-minimal`
  generation (dropped --Os/--Oauto-sched), NPU clock 200→800 MHz, both weight profiles.

---

## Fix (DS-CNN) — re-architecture required (pure surgery ruled out)

ST's Neural-ART operator doc (`Documentation/stneuralart_operator_support.html`) settles it: HW
pooling supports **≤3×3 windows only** (larger windows are decomposed; pooling line-buffer
`width×channels ≤ 2048`). DS-CNN's global pool is over a **[50×11]=550** map, and 50 (=2·5²) / 11
(prime) **cannot be tiled exactly by ≤3 kernels** → a SW float epoch is unavoidable in-place. Six
ONNX-surgery variants confirmed this empirically (reshape-to-1D, GlobalAveragePool, 3-D reshape,
hierarchical small pools, depthwise averaging conv 50×11, 3×3 pool — **all land in the SW island**;
once any op dequantizes to float, the island swallows neighbours).

**Therefore the model itself must change:** add spatial downsampling (strided convs/pools ≤3) so the
**pre-pool feature map is small** (≤3×3-tileable, like TC-ResNet8's width-7 map), then the existing
global pool maps to HW. Pipeline: edit `kws/models/dscnn.py` → **retrain on GPU** (CPU here is
~12.7 min/epoch, too slow — cached MFCCs in `data/mfcc_cache/`) → `kws/quantize.py` → re-export v17 →
`stedgeai` generate (confirm all-HW via the epoch report) → flash → full 11,005 run. Architecture is
probed **host-only first** (HW/SW mapping depends only on tensor shapes, no training needed) so we
retrain the correct design exactly once.

---

## Current firmware config (as left, 2026-06-15)

`src_npu/main_npu.c`: `risaf_npu_config()` opens RISAF4/5/6 (+RISAF12 for EXTMEM) after clocks;
`mpu_config()` = `HAL_MPU_Disable()` (npuRAM cacheable); `npu_config()` enables NPU + CACHEAXI cache;
IC6÷1 → NPU @ 800 MHz; EXTMEM path memory-maps octo-flash. `src_npu/kws_bench_npu.c`: run order
load_input → MCU clean/inval + npu_cache_inval → `npu_reset()` (Reset_Network + LL_Busif_SetKeys ×4)
→ `npu_run_epochs()`. Debug markers (A–F, R0/R4/R5/R6/R12, DBGpre, DBGEPOCHS, DBGLOGITS) still in —
strip once DS-CNN is fixed. Build: `cd firmware/stm32n6 && rm -rf build/<t> && make -f Makefile.npu
MODEL=<dscnn|tcresnet> [EXTMEM=1] [MODELDIR=...]`.

Model dirs: `kws_dscnn_npu` (profile-default, ext-flash), `kws_dscnn_npu_sram` (internal-memories,
embedded blob), `kws_dscnn_npu_min` (profile-minimal), `kws_tcresnet_npu` (ext-flash, ✅ works).
