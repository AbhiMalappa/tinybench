#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_cpu.h"
#include "esp_log.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gru_model_data.h"

#define BENCH_UART       UART_NUM_0
#define UART_RX_BUF      2048
#define UART_TX_BUF      0

static constexpr int kTensorArenaSize = 256 * 1024;
static constexpr int kExpectedInputBytes = 49 * 10;
static constexpr int kBenchRuns = 1;
static constexpr uint32_t kReadTimeoutMs = 30000;

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
static int8_t input_buffer[kExpectedInputBytes];

static void uart_print(const char *s) {
    uart_write_bytes(BENCH_UART, s, strlen(s));
}
static void uart_println(const char *s) {
    uart_write_bytes(BENCH_UART, s, strlen(s));
    uart_write_bytes(BENCH_UART, "\r\n", 2);
}
static bool uart_read_exact(uint8_t *dst, int n, uint32_t timeout_ms) {
    int got = 0;
    uint32_t start = esp_log_timestamp();
    while (got < n) {
        int r = uart_read_bytes(BENCH_UART, dst + got, n - got, pdMS_TO_TICKS(100));
        if (r > 0) got += r;
        if ((esp_log_timestamp() - start) > timeout_ms) return false;
    }
    return true;
}
static int cmp_u32(const void *a, const void *b) {
    uint32_t va = *(const uint32_t *)a;
    uint32_t vb = *(const uint32_t *)b;
    return (va > vb) - (va < vb);
}

extern "C" void app_main(void) {
    uart_config_t cfg = {};
    cfg.baud_rate  = 115200;
    cfg.data_bits  = UART_DATA_8_BITS;
    cfg.parity     = UART_PARITY_DISABLE;
    cfg.stop_bits  = UART_STOP_BITS_1;
    cfg.flow_ctrl  = UART_HW_FLOWCTRL_DISABLE;
    uart_param_config(BENCH_UART, &cfg);
    uart_driver_install(BENCH_UART, UART_RX_BUF, UART_TX_BUF, 0, NULL, 0);

    vTaskDelay(pdMS_TO_TICKS(6000));

    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        char buf[128];
        snprintf(buf, sizeof(buf), "{\"error\":\"schema_mismatch\",\"got\":%lu,\"expected\":%d}\r\n",
            (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        uart_print(buf);
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    static tflite::MicroMutableOpResolver<13> resolver;
    if (resolver.AddShape()           != kTfLiteOk ||
        resolver.AddStridedSlice()    != kTfLiteOk ||
        resolver.AddPack()            != kTfLiteOk ||
        resolver.AddFill()            != kTfLiteOk ||
        resolver.AddFullyConnected()  != kTfLiteOk ||
        resolver.AddTranspose()       != kTfLiteOk ||
        resolver.AddUnpack()          != kTfLiteOk ||
        resolver.AddSplit()           != kTfLiteOk ||
        resolver.AddAdd()             != kTfLiteOk ||
        resolver.AddLogistic()        != kTfLiteOk ||
        resolver.AddMul()             != kTfLiteOk ||
        resolver.AddSub()             != kTfLiteOk ||
        resolver.AddTanh()            != kTfLiteOk) {
        uart_println("{\"error\":\"resolver_add_failed\"}");
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        char buf[192];
        snprintf(buf, sizeof(buf),
            "{\"error\":\"allocate_tensors_failed\",\"arena_size\":%d"
            ",\"arena_used_attempt\":%lu}\r\n",
            kTensorArenaSize,
            (unsigned long)interpreter.arena_used_bytes());
        uart_print(buf);
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    TfLiteTensor *input  = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    static const uint32_t clock_hz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL;
    char boot[256];
    snprintf(boot, sizeof(boot),
        "{\"event\":\"boot\",\"board\":\"esp32s3\",\"model\":\"gru\""
        ",\"engine\":\"esp-tflite-micro\""
        ",\"input_bytes\":%lu,\"input_dtype\":%d"
        ",\"output_bytes\":%lu,\"output_dtype\":%d"
        ",\"arena_used\":%lu,\"arena_size\":%d"
        ",\"clock_hz\":%lu}",
        (unsigned long)input->bytes, (int)input->type,
        (unsigned long)output->bytes, (int)output->type,
        (unsigned long)interpreter.arena_used_bytes(), kTensorArenaSize,
        (unsigned long)clock_hz);
    uart_println(boot);
    uart_println("READY");

    uint32_t cycles[kBenchRuns];
    while (true) {
        if ((int)input->bytes != kExpectedInputBytes) {
            char buf[128];
            snprintf(buf, sizeof(buf), "{\"error\":\"input_size_mismatch\",\"got\":%lu,\"expected\":%d}\r\n",
                (unsigned long)input->bytes, kExpectedInputBytes);
            uart_print(buf);
            vTaskDelay(pdMS_TO_TICKS(2000));
            continue;
        }
        if (!uart_read_exact((uint8_t *)input_buffer, kExpectedInputBytes, kReadTimeoutMs)) {
            continue;
        }
        for (int i = 0; i < kBenchRuns; ++i) {
            memcpy(input->data.int8, input_buffer, kExpectedInputBytes);
            uint32_t t0 = esp_cpu_get_cycle_count();
            TfLiteStatus status = interpreter.Invoke();
            cycles[i] = esp_cpu_get_cycle_count() - t0;
            if (status != kTfLiteOk) {
                uart_println("{\"error\":\"invoke_failed\"}");
                uart_println("READY");
                goto next_sample;
            }
        }
        {
            qsort(cycles, kBenchRuns, sizeof(uint32_t), cmp_u32);
            uint32_t median_cycles = cycles[kBenchRuns / 2];
            float latency_ms = (float)median_cycles / (float)clock_hz * 1000.0f;
            int best_idx = 0;
            int8_t best_score = output->data.int8[0];
            for (size_t i = 1; i < output->bytes; ++i) {
                if (output->data.int8[i] > best_score) {
                    best_score = output->data.int8[i];
                    best_idx = (int)i;
                }
            }
            char result[256];
            snprintf(result, sizeof(result),
                "{\"board\":\"esp32s3\",\"model\":\"gru\",\"npu\":0"
                ",\"latency_ms\":%.4f,\"class\":%d,\"score\":%d,\"cycles\":%lu}",
                latency_ms, best_idx, (int)best_score, (unsigned long)median_cycles);
            uart_println(result);
            uart_println("READY");
        }
        next_sample:;
    }
}
