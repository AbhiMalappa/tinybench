/*
 * TinyBench-KWS STM32N6570-DK benchmark loop — NPU path (Neural-ART / LL_ATON).
 *
 * Same wire protocol as the CPU path (kws/host/protocol.md): host streams 1960
 * float32 bytes/sample; firmware emits boot JSON + READY, then per sample runs one
 * timed inference and replies with the result JSON. Latency = single DWT-timed
 * NPU inference (N_RUNS=1, matching the other boards/CPU path); distribution taken
 * host-side over all samples.
 *
 * Weights + activations live in internal npuRAM (loaded to SRAM alongside the
 * firmware), so this is a pure SRAM-resident run — no external flash.
 *
 * Compile-time -D:
 *   BENCH_MODEL_STR  e.g. "dscnn"
 *   AI_NETWORK_IN_SCALE / AI_NETWORK_IN_ZP  (only for INT8-input models, e.g. DS-CNN)
 */
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "stm32n6xx_hal.h"
#include "ll_aton_runtime.h"
#include "ll_aton_caches_interface.h"
#include "ll_aton.h"
#include "npu_cache.h"
#include "mcu_cache.h"
#include "ATON.h"
#include "network.h"

#define BUSIF_KEY  0xAABBCCDDAABBCCDDULL

#ifndef BENCH_MODEL_STR
#  define BENCH_MODEL_STR "network"
#endif
#ifndef BENCH_CLOCK_HZ
#  define BENCH_CLOCK_HZ 800000000UL
#endif

/* Host always streams 1960 float32 bytes (490 MFCC values). */
#define N_FEATURES       490
#define INPUT_UART_BYTES (N_FEATURES * 4)   /* 1960 */
#define N_CLASSES        35

/* Model input path: 490 B → INT8 (quantize), 1960 B → float32 (copy direct). */
#if (LL_ATON_DEFAULT_IN_1_SIZE_BYTES == 490)
#  define BENCH_INPUT_FLOAT 0
#elif (LL_ATON_DEFAULT_IN_1_SIZE_BYTES == 1960)
#  define BENCH_INPUT_FLOAT 1
#else
#  error "Unexpected LL_ATON_DEFAULT_IN_1_SIZE_BYTES"
#endif

extern UART_HandleTypeDef huart1;

/* Bind an execution instance to the generated "Default" network interface. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(Default);

static float g_input_f32[N_FEATURES] __attribute__((aligned(4)));

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

/* ── Load host floats into the NPU input buffer ──────────────────────────── */
static void load_input(const float *src, void *dst)
{
#if BENCH_INPUT_FLOAT
    memcpy(dst, src, N_FEATURES * sizeof(float));
#else
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

/* ── Reset the network state for a fresh inference ───────────────────────── */
/* MUST be called BEFORE writing the input: ST's wrapper resets first, then sets
   the input buffer. Resetting after the input write clobbers it (zeroed buffers
   → constant saturated output regardless of input). */
static void npu_reset(void)
{
    LL_ATON_RT_Reset_Network(&NN_Instance_Default);
    /* Bus-interface keys — the Neural-ART scrambles memory accesses unless these
       are set (ST sets them every inference, even for non-encrypted models). */
    LL_Busif_SetKeys(0, 0, BUSIF_KEY, BUSIF_KEY);
    LL_Busif_SetKeys(0, 1, BUSIF_KEY, BUSIF_KEY);
    LL_Busif_SetKeys(1, 0, BUSIF_KEY, BUSIF_KEY);
    LL_Busif_SetKeys(1, 1, BUSIF_KEY, BUSIF_KEY);
}

/* ── Run the epoch-block loop to completion (input must already be in place) ─ */
static volatile uint32_t g_epoch_blocks;  /* count of RunEpochBlock calls (excl. WFE) */
static int npu_run_epochs(void)
{
    LL_ATON_RT_RetValues_t r;
    g_epoch_blocks = 0;
    do {
        r = LL_ATON_RT_RunEpochBlock(&NN_Instance_Default);
        if (r == LL_ATON_RT_WFE) LL_ATON_OSAL_WFE();
        else g_epoch_blocks++;
    } while (r != LL_ATON_RT_DONE);
    return 0;
}

void kws_benchmark_run(void)
{
    dwt_enable();

    /* Match ST's ai_wrapper npu_init(mode==1) sequence: clean caches, clear the
       ATON internal pipeline, THEN init runtime + network. The pipeline clear was
       the missing step that left the NPU computing from a stale state. */
    npu_cache_invalidate();
    mcu_cache_clean_invalidate();
    {
        uint32_t t = ATON_CLKCTRL_CTRL_GET(0);
        t = ATON_CLKCTRL_CTRL_SET_CLR(t, 1);
        ATON_CLKCTRL_CTRL_SET(0, t);
    }

    uart_puts("D:pre-rtinit\n");
    LL_ATON_RT_RuntimeInit();
    uart_puts("E:rtinit\n");
    LL_ATON_RT_Init_Network(&NN_Instance_Default);
    uart_puts("F:netinit\n");

    const LL_Buffer_InfoTypeDef *in_info  = LL_ATON_Input_Buffers_Info_Default();
    const LL_Buffer_InfoTypeDef *out_info = LL_ATON_Output_Buffers_Info_Default();
    void   *in_addr  = (void *)LL_Buffer_addr_start(&in_info[0]);
    int8_t *out_addr = (int8_t *)LL_Buffer_addr_start(&out_info[0]);
    uint32_t in_len  = LL_Buffer_len(&in_info[0]);
    uint32_t out_len = LL_Buffer_len(&out_info[0]);

    char line[256];
    snprintf(line, sizeof(line),
        "{\"event\":\"boot\",\"board\":\"stm32n6\",\"model\":\"" BENCH_MODEL_STR "\","
        "\"input_bytes\":%u,\"input_dtype\":%d,\"output_bytes\":%u,\"output_dtype\":9,"
        "\"arena_used\":%u,\"arena_size\":%u,\"clock_hz\":%lu}\n",
        (unsigned)INPUT_UART_BYTES,
        BENCH_INPUT_FLOAT ? 1 : 9,
        (unsigned)out_len,
        (unsigned)in_len, (unsigned)in_len,
        (unsigned long)BENCH_CLOCK_HZ);
    uart_puts(line);
    uart_puts("READY\n");

    while (1) {
        /* Block until the full tensor arrives (HAL_MAX_DELAY: no SysTick dependence). */
        if (uart_recv((uint8_t *)g_input_f32, INPUT_UART_BYTES, HAL_MAX_DELAY) != HAL_OK) {
            uart_puts("{\"error\":\"recv_timeout\"}\n");
            uart_puts("READY\n");
            continue;
        }

        /* Write input, push to RAM + invalidate NPU cache so the NPU sees it.
           Order matches ST's npu_run(): fill input buffer → cache-prep → reset
           → set keys → run epochs. */
        load_input(g_input_f32, in_addr);
        LL_ATON_Cache_MCU_Clean_Invalidate_Range((uintptr_t)in_addr, in_len);
        npu_cache_invalidate();

        /* DEBUG (pre-run): quantized input bytes at the runtime input buffer. */
        {
            char db[200];
            const int8_t *q = (const int8_t *)in_addr;
            snprintf(db, sizeof(db),
                "DBGpre in[0..7]=%d,%d,%d,%d,%d,%d,%d,%d\n",
                q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]);
            uart_puts(db);
        }

        uint32_t t0 = dwt_now();
        npu_reset();
        int ok = (npu_run_epochs() == 0);
        uint32_t t1 = dwt_now();

        if (!ok) {
            uart_puts("{\"error\":\"invoke_failed\"}\n");
            uart_puts("READY\n");
            continue;
        }

        /* Make NPU-written output visible to the CPU. */
        LL_ATON_Cache_MCU_Invalidate_Range((uintptr_t)out_addr, out_len);

        /* DEBUG: dump all 35 output logits (host ignores non-result lines) */
        {
            char db[400];
            int n = snprintf(db, sizeof(db), "DBGEPOCHS=%lu DBGLOGITS", (unsigned long)g_epoch_blocks);
            for (int i = 0; i < N_CLASSES && n < (int)sizeof(db) - 8; i++)
                n += snprintf(db + n, sizeof(db) - n, " %d", out_addr[i]);
            n += snprintf(db + n, sizeof(db) - n, "\n");
            uart_puts(db);
        }

        uint32_t cyc = t1 - t0;
        float lat_ms = (float)cyc / ((float)BENCH_CLOCK_HZ / 1000.0f);

        int best = 0; int8_t bs = out_addr[0];
        for (int i = 1; i < N_CLASSES; i++)
            if (out_addr[i] > bs) { bs = out_addr[i]; best = i; }

        snprintf(line, sizeof(line),
            "{\"board\":\"stm32n6\",\"model\":\"" BENCH_MODEL_STR "\",\"npu\":1,"
            "\"latency_ms\":%.4f,\"class\":%d,\"score\":%d,\"cycles\":%lu}\n",
            lat_ms, best, (int)bs, (unsigned long)cyc);
        uart_puts(line);
        uart_puts("READY\n");
    }
}
