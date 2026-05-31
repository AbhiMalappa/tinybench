# TinyBench-KWS Host ↔ Firmware Protocol

Cross-board contract between `benchmark_serial.py` (host) and the device firmware on STM32N6570-DK, ESP32-S3-DevKitC-1, and Arduino Nano 33 BLE Sense Rev2.

Every implementation MUST conform to this protocol. The host code is identical across boards; only the firmware-side implementation differs.

## Serial parameters
- Baud rate: **115200**
- 8 data bits, 1 stop bit, no parity, no flow control
- Encoding: **raw bytes** for the input tensor, **UTF-8** for everything else
- Line terminator for text lines: `\n`

## Lifecycle

```
┌── device powered/reset ──┐
│                          │
│   setup():               │
│     init interpreter     │
│     emit BOOT line       │ → host validates
│     emit "READY"         │ → host begins
│                          │
│   loop():                │
│     read 490 bytes       │ ← host writes one tensor
│     run 100 inferences   │
│     emit RESULT JSON     │ → host parses
│     emit "READY"         │ → host sends next
│                          │
└──────────────────────────┘
```

## Boot message (one-shot, on reset)

Single line, valid JSON, key `event` is `"boot"`:

```json
{"event":"boot","board":"<id>","model":"dscnn","input_bytes":490,"input_dtype":9,"output_bytes":35,"output_dtype":9,"arena_used":78436,"arena_size":98304,"clock_hz":64000000}
```

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `event` | `"boot"` | Discriminator |
| `board` | string | Stable board ID (e.g. `arduino_nano33`, `esp32s3`, `stm32n6`) |
| `model` | string | Model identifier (e.g. `dscnn`) |
| `input_bytes` | int | Must equal 490 |
| `input_dtype` | int | TFLite enum; INT8 = 9 |
| `output_bytes` | int | Must equal 35 |
| `output_dtype` | int | TFLite enum; INT8 = 9 |
| `arena_used` | int | Peak tensor arena bytes (peak RAM metric) |
| `arena_size` | int | Allocated arena bytes |
| `clock_hz` | int | CPU clock; latency = cycles / clock_hz |

After the boot line, the firmware MUST emit a single line containing exactly `READY\n` and enter the main loop.

## Per-inference exchange

1. Host writes exactly **490 raw INT8 bytes** to the serial port. Order: row-major `(n_frames=49, n_mfcc=10)`. **The host MUST trickle the bytes — Arduino Nano 33 BLE Mbed USB CDC drops bytes on bulk writes >256 B. Reference host implementation uses 32-byte chunks with 50 ms inter-chunk delay (~0.8 s per 490-byte tensor). Boards with deeper RX buffers may accept fewer/faster chunks.**
2. Firmware reads 490 bytes into the input tensor.
3. Firmware runs **100 inferences** on the same input, measuring each with a hardware cycle counter (DWT->CYCCNT on Cortex-M; CCOUNT on Xtensa LX7).
4. Firmware computes the **median** of the 100 cycle counts.
5. Firmware emits one RESULT line (valid JSON, key `class` present):

```json
{"board":"arduino_nano33","model":"dscnn","npu":0,"latency_ms":12.3456,"class":7,"score":-32,"cycles":790118}
```

| Field | Type | Meaning |
|---|---|---|
| `board` | string | Same as boot |
| `model` | string | Same as boot |
| `npu` | int | `1` if NPU enabled (STM32N6 with Ethos-U55), else `0` |
| `latency_ms` | float | Median latency in ms (cycles / clock_hz × 1000) |
| `class` | int | Argmax over the 35 INT8 output logits |
| `score` | int | The winning INT8 score (signed) — sanity field |
| `cycles` | int | Median raw cycle count |

6. Firmware emits a single line `READY\n` and waits for the next 490 bytes.

## Errors

If anything fails (allocation, invocation, schema mismatch, input read timeout), firmware emits a single JSON line with key `error` and either halts or emits `READY` to recover:

```json
{"error":"invoke_failed"}
{"error":"allocate_tensors_failed"}
{"error":"schema_mismatch","got":3,"expected":3}
```

The host treats any line containing `"error"` at the top level as a clip failure and continues.

## Timing definition (non-negotiable)
- Cycle counter must be a **hardware** counter (DWT on Cortex-M, CCOUNT on Xtensa).
- Measured interval: **strictly the `Invoke()` call**, no Serial I/O, no host bytes counted.
- Reported `latency_ms` is the **median over 100 runs of the same input tensor**.

## Test vectors
- Identical byte stream consumed by every board: `kws/host/test_vectors/test_vectors_int8.npy`.
- Identical ground-truth labels: `kws/host/test_vectors/test_labels.npy`.
- Manifest with byte stream and source-model hashes: `kws/host/test_vectors/test_metadata.json`.
- Reviewers verify reproducibility by SHA256 of the `.npy` files.
