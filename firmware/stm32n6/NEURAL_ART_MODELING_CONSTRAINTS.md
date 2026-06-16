# Neural-ART (STM32N6) — Modeling Constraints Cheat-Sheet

Distilled from ST Edge AI Core 2.2.0 docs (`stedgeai/Documentation/stneuralart_operator_support.html`,
`…_getting_started.html`, `…_neural_art_compiler.html`, `…_programming_model.html`). Purpose: design
HW-friendly models **once** so we don't iterate on the board. Anything not HW-mapped runs on the M55
CPU (a "software-fallback epoch") — slow, and a mid-graph SW float island can scramble output (the
DS-CNN failure). **Goal: every layer maps to HW.**

## Quantization / model format (must-haves)
- **8-bit / 8-bit, ss/sa per-channel.** ONNX QDQ (DequantizeLinear/QuantizeLinear around float ops) or
  TFLite int8. Our `quantize.py` (ORT `quantize_static`, `per_channel=True`, `QInt8`, QDQ) ✅ matches.
- **No hybrid ops** — both activations AND weights must be quantized.
- **Static shapes**; batch dim = 1 (variable batch treated as 1).
- Internal tensor model is **4-D `(H, BATCH, W, CH)`** — avoid anything that expands to 5-D
  (this is exactly why GRU-96 is infeasible: its seq output → `[49,1,1,1,96]`).
- Layout: **channel-first (PyTorch NCHW) is accepted** (compiler inserts transpose if needed).

## Operator HW/SW map (the ones relevant to KWS CNNs)
**HW (use freely):** Conv (Conv1D/Conv2D), DepthwiseConv2D, **Gemm / FULLY_CONNECTED / MatMul**,
Relu / Relu6 / Clip, LeakyRelu / PRelu, HardSwish, Sigmoid / Tanh / Logistic, Add / Sub / Mul / Max /
Min, Concat, Pad (ConstantPad), Reshape / Flatten / Transpose / Squeeze / Unsqueeze / Identity,
AveragePool / MaxPool / GlobalAveragePool / GlobalMaxPool *(≤3×3 window — see below)*,
(Re)Quantize int8↔int8.
**SW_FLOAT (AVOID inside the graph):** **Mean / ReduceMean*** , ReduceSum/Min/Prod, BatchNorm *(HW only
if folded after a Conv)*, Softmax (SW_INT), Exp/Log/Pow/Sqrt-as-float, Gather, Resize-bilinear,
DequantizeLinear/QuantizeLinear *(the float↔int8 boundary — fine at model I/O, costly mid-graph)*.
  - *ReduceMean/ReduceMax **can** be HW if convertible to Global(Avg/Max)Pool: reduced axes are the
    right-most ones AND `input.rank == n_reduced_axes + 2`. Still bound by the pool window limit.*
- **BatchNorm after Conv → HW** (folds). So conv→BN→ReLU is fine; a standalone BN is SW.

## Convolution limits (Conv2D / DepthwiseConv2D — both HW)
- **Kernel width ≤ 6 for stride 1, ≤ 12 for stride ≥ 2. Kernel height ≤ 3.** Larger dims are
  *decomposed* by the compiler (still HW, but iterative = slower).
- **Kernel HEIGHT = 1 ⇒ no restriction on feature width** (theoretical max 2¹⁶−1). 1×W convs are cheap.
- Best kernel height = 3; best width ∈ {3, 6, 12}. 1×1 / 2×1 / 1×3 / 1×6 / 1×12 etc. are special-cased fast.
- **Strides: prefer {1, 2, 4}.** Vertical (height) strides are generally inefficient. Don't use a stride
  larger than the kernel. Pad ≤ 2 is free.
- **Avoid prime channel counts** (kernels/channels) — limits the split heuristic. 1×1 conv most efficient
  when ICH ≈ N·(72…128) and OCH ≈ M·(16…24). Layers with **ICH > OCH** map more efficiently.
- Avoid large dilation.
- (Note: DS-CNN's current `(10,4)` stem has height 10 > 3 → it runs HW but *decomposed*; acceptable.)

## Pooling limits  ← THE constraint that breaks DS-CNN
- **AveragePool / MaxPool / Global*Pool kernel Height and Width must be in [1, 3].** Windows > 3 are
  decomposed — works up to a point (TC-ResNet's width-7 global pool decomposes & stays HW), but a large
  2-D global pool (DS-CNN's **50×11**) falls back to a **SW float epoch** → wrong output.
- Strides 1–15; pad L/R/top 0–7, bottom ≤ window_height−1.
- **Line buffer: `feature_width × input_channels ≤ 2048`** (else the op is split into columns).
- Only 1-D and 2-D pooling.
- **Design rule:** the **pre-pool feature map must be small** — target **≤ 3×3** (single HW pool, zero
  risk), or at most a TC-ResNet-like single small axis (e.g. width ≤ ~7, height 1). Reach it with
  strided convs/pools (strides 2).

## Feature-width / memory
- 8-bit: `feature_width × input_channels ≤ 2048` per column, else split (perf hit). KWS maps are tiny
  (width ≤ 10 × 64 ch = 640) — never an issue for us.
- Weights live in external octo-flash (`0x71000000`); activations in npuRAM (4× ~448 KB banks). Model
  must fit; KWS models are ≤ ~130 KB weights — fine.
- HW has 4 CONV_ACC units and ≤ 5 stream chains (parallelism limits, not correctness).

## Broadcasting
- Unidirectional only; best with ≤ 512 elements; scalar/channel/H/W/H·W broadcasting HW-mapped.

---

## What this means for the DS-CNN re-architecture
**Already HW-fine (keep):** depthwise-separable Conv blocks, BN-after-conv, ReLU, the final Gemm
classifier, 64 channels (non-prime), int8 QDQ via `quantize.py`. The `(10,4)` stem works (decomposed).

**Must change — the head only:** the global average pool currently sees a **50×11** map → SW fallback.
Add **spatial downsampling** (strided convs/pools, strides ∈ {1,2,4}, kernels ≤3) to shrink the pre-pool
map to **≤3×3** (e.g. → `[3,2]` or `[3,3]`), then `AdaptiveAvgPool2d(1)` maps to a HW ≤3×3 pool.
This is *more* faithful to the canonical Hello-Edge DS-CNN (which downsamples), not less — and it cuts
MACCs (currently 11.8 M over large maps → faster CPU too).

**Bake-in design rules for the new head/blocks:**
1. Downsample with **stride-2** (not 5/other) in the stem and/or 1–2 DS blocks until spatial ≤ 3×3.
2. Keep all conv kernel **heights ≤ 3** in new layers; widths ∈ {1,3} (our W is ≤10, so small anyway).
3. Final pool window **≤ 3×3** (after downsampling) — verify via the stedgeai epoch report (all-HW).
4. Channels non-prime (stick with 64); ICH ≥ OCH where possible.
5. No new SW_FLOAT ops (no standalone BN, no Mean/Reduce except the final global pool, no Softmax in-graph).

**Verification (host-only, no training):** build the candidate architecture with random weights → export
→ quantize → `stedgeai generate --st-neural-art` → confirm the report shows **0 SW epochs** (or just the
input-quantize). Only then commit to GPU training.
