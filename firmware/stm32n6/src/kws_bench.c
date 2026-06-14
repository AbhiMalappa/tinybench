/*
 * TinyBench-KWS STM32N6570-DK benchmark loop (CPU / X-CUBE-AI path).
 *
 * Protocol: kws/host/protocol.md
 *   Boot:   {"event":"boot","board":"stm32n6","model":"<MODEL>",
 *             "input_bytes":1960,"input_dtype":1,"output_bytes":35,
 *             "output_dtype":9,"arena_used":<N>,"arena_size":<N>,"clock_hz":800000000}
 *   READY
 *   Loop:   recv 1960 float32 bytes →
 *           (DS-CNN) quantize f32→int8  OR  (TC-ResNet/GRU) feed f32 directly →
 *           run 100 inferences → median cycles →
 *           {"board":"stm32n6","model":"<MODEL>","npu":<0|1>,"latency_ms":<f>,
 *            "class":<i>,"score":<i>,"cycles":<u>}
 *           READY
 *
 * The host ALWAYS streams 1960 float32 bytes (490 MFCC values). Whether the
 * generated model ingests INT8 (490 B) or float32 (1960 B) is detected at
 * compile time from AI_NETWORK_IN_1_SIZE_BYTES in the generated network.h.
 *
 * Compile-time defines (set per-build via -D):
 *   BENCH_MODEL_STR     string literal, e.g. "dscnn"
 *   BENCH_NPU           0 (this file is the CPU path)
 *   AI_NETWORK_IN_SCALE float  (only needed for INT8-input models, e.g. DS-CNN)
 *   AI_NETWORK_IN_ZP    int    (only needed for INT8-input models)
 */
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "stm32n6xx_hal.h"

/* X-CUBE-AI runtime */
#include "network.h"
#include "network_data.h"
#include "ai_platform.h"
#include "ai_datatypes_defines.h"

/* ── Config defaults ─────────────────────────────────────────────────────── */
#ifndef BENCH_MODEL_STR
#  define BENCH_MODEL_STR "network"
#endif
#ifndef BENCH_NPU
#  define BENCH_NPU 0
#endif
#ifndef BENCH_CLOCK_HZ
#  define BENCH_CLOCK_HZ 800000000UL
#endif

/* Detect input path from the generated header:
 *   490  bytes → model ingests INT8   → firmware quantizes f32→int8
 *   1960 bytes → model ingests float32 → firmware feeds floats directly
 */
#if (AI_NETWORK_IN_1_SIZE_BYTES == 490)
#  define BENCH_INPUT_FLOAT 0
#elif (AI_NETWORK_IN_1_SIZE_BYTES == 1960)
#  define BENCH_INPUT_FLOAT 1
#else
#  error "Unexpected AI_NETWORK_IN_1_SIZE_BYTES"
#endif

/* Inferences per sample. =1 to match the Arduino/ESP32 firmware (kBenchRuns=1):
   one timed Invoke per sample; the latency distribution is taken host-side over
   all samples. (Earlier value 100 was non-conformant and ~100x slower.) */
#define N_RUNS      1
#define N_FEATURES  490
#define INPUT_UART_BYTES (N_FEATURES * 4)   /* 1960 — what the host streams */
#define N_CLASSES   35

/* ── Globals ─────────────────────────────────────────────────────────────── */
extern UART_HandleTypeDef huart1;

/* Activations buffer (input/output allocated inside it by default) */
AI_ALIGNED(32) static uint8_t g_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

/* Receive buffer for the 490 float32 MFCC values streamed by the host */
AI_ALIGNED(4) static float g_input_f32[N_FEATURES];

static uint32_t g_cycles[N_RUNS];

/* ── DWT cycle counter ───────────────────────────────────────────────────── */
static inline void dwt_enable(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_now(void) { return DWT->CYCCNT; }

/* ── UART ────────────────────────────────────────────────────────────────── */
static void uart_puts(const char *s)
{
    HAL_UART_Transmit(&huart1, (uint8_t *)s, strlen(s), HAL_MAX_DELAY);
}
static HAL_StatusTypeDef uart_recv(uint8_t *buf, uint32_t len, uint32_t to_ms)
{
    return HAL_UART_Receive(&huart1, buf, len, to_ms);
}

/* ── Median (insertion sort, n=100) ──────────────────────────────────────── */
static uint32_t median_u32(uint32_t *a, int n)
{
    for (int i = 1; i < n; i++) {
        uint32_t k = a[i]; int j = i - 1;
        while (j >= 0 && a[j] > k) { a[j+1] = a[j]; j--; }
        a[j+1] = k;
    }
    return a[n / 2];
}

/* ── Load the host's float features into the model input buffer ──────────── */
static void load_input(const float *src, void *dst)
{
#if BENCH_INPUT_FLOAT
    memcpy(dst, src, N_FEATURES * sizeof(float));   /* feed float32 directly */
#else
    /* ONNX QuantizeLinear: q = round(x/scale) + zp, clamped to int8 */
    int8_t *q = (int8_t *)dst;
    const float scale = (float)AI_NETWORK_IN_SCALE;
    const int   zp    = (int)AI_NETWORK_IN_ZP;
    for (int i = 0; i < N_FEATURES; i++) {
        float v = roundf(src[i] / scale) + (float)zp;
        if (v < -128.0f) v = -128.0f;
        if (v >  127.0f) v =  127.0f;
        q[i] = (int8_t)(int)v;
    }
#endif
}

/* ── Benchmark loop ──────────────────────────────────────────────────────── */
void kws_benchmark_run(void)
{
    dwt_enable();

    ai_handle network = AI_HANDLE_NULL;
    const ai_handle acts[] = { (ai_handle)g_activations };
    ai_error err = ai_network_create_and_init(&network, acts, NULL);
    if (err.type != AI_ERROR_NONE) {
        uart_puts("{\"error\":\"allocate_tensors_failed\"}\n");
        while (1);
    }

    ai_buffer *ai_inp = ai_network_inputs_get(network, NULL);
    ai_buffer *ai_out = ai_network_outputs_get(network, NULL);

    char line[256];
    snprintf(line, sizeof(line),
        "{\"event\":\"boot\",\"board\":\"stm32n6\",\"model\":\"" BENCH_MODEL_STR "\","
        "\"input_bytes\":%u,\"input_dtype\":%d,\"output_bytes\":%u,\"output_dtype\":9,"
        "\"arena_used\":%u,\"arena_size\":%u,\"clock_hz\":%lu}\n",
        (unsigned)INPUT_UART_BYTES,
        BENCH_INPUT_FLOAT ? 1 : 9,
        (unsigned)AI_NETWORK_OUT_1_SIZE_BYTES,
        (unsigned)AI_NETWORK_DATA_ACTIVATIONS_SIZE,
        (unsigned)AI_NETWORK_DATA_ACTIVATIONS_SIZE,
        (unsigned long)BENCH_CLOCK_HZ);
    uart_puts(line);
    uart_puts("READY\n");

    while (1) {
        if (uart_recv((uint8_t *)g_input_f32, INPUT_UART_BYTES, 30000) != HAL_OK) {
            uart_puts("{\"error\":\"recv_timeout\"}\n");
            uart_puts("READY\n");
            continue;
        }

        int ok = 1;
        for (int r = 0; r < N_RUNS; r++) {
            /* input lives in the activations arena and is clobbered by run();
               reload it each iteration (outside the timed window). */
            load_input(g_input_f32, ai_inp->data);
            uint32_t t0 = dwt_now();
            ai_i32 n = ai_network_run(network, ai_inp, ai_out);
            uint32_t t1 = dwt_now();
            if (n <= 0) { ok = 0; break; }
            g_cycles[r] = t1 - t0;
        }
        if (!ok) {
            uart_puts("{\"error\":\"invoke_failed\"}\n");
            uart_puts("READY\n");
            continue;
        }

        uint32_t med = median_u32(g_cycles, N_RUNS);
        float lat_ms = (float)med / ((float)BENCH_CLOCK_HZ / 1000.0f);

        const int8_t *out = (const int8_t *)ai_out->data;
        int best = 0; int8_t bs = out[0];
        for (int i = 1; i < N_CLASSES; i++)
            if (out[i] > bs) { bs = out[i]; best = i; }

        snprintf(line, sizeof(line),
            "{\"board\":\"stm32n6\",\"model\":\"" BENCH_MODEL_STR "\",\"npu\":%d,"
            "\"latency_ms\":%.4f,\"class\":%d,\"score\":%d,\"cycles\":%lu}\n",
            BENCH_NPU, lat_ms, best, (int)bs, (unsigned long)med);
        uart_puts(line);
        uart_puts("READY\n");
    }
}
