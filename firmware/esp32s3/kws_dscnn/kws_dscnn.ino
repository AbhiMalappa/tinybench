#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_cpu.h"

#include "dscnn_model_data.h"

// arduino-esp32 3.x routes Serial to USB JTAG/CDC when USBMode=hwcdc.
// CP2102N is on UART0 = Serial0.
#define MCU_SERIAL Serial0

constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr int kExpectedInputBytes = 49 * 10;
constexpr int kBenchRuns = 1;
constexpr uint32_t kReadTimeoutMs = 30000;

static int8_t input_buffer[kExpectedInputBytes];

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
}

static int cmp_u32(const void* a, const void* b) {
  uint32_t va = *(const uint32_t*)a;
  uint32_t vb = *(const uint32_t*)b;
  return (va > vb) - (va < vb);
}

static bool read_exact(uint8_t* dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t start = millis();
  while (got < n) {
    if (MCU_SERIAL.available() > 0) {
      int b = MCU_SERIAL.read();
      if (b >= 0) dst[got++] = (uint8_t)b;
    }
    if (millis() - start > timeout_ms) return false;
  }
  return true;
}

void setup() {
  MCU_SERIAL.begin(115200);
  delay(6000);  // wait for host to open port: 4s post-flash sleep + margin

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MCU_SERIAL.print("{\"error\":\"schema_mismatch\",\"got\":");
    MCU_SERIAL.print(model->version());
    MCU_SERIAL.print(",\"expected\":");
    MCU_SERIAL.print(TFLITE_SCHEMA_VERSION);
    MCU_SERIAL.println("}");
    while (true) {}
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  if (resolver.AddPad()             != kTfLiteOk ||
      resolver.AddConv2D()          != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      resolver.AddMean()            != kTfLiteOk ||
      resolver.AddFullyConnected()  != kTfLiteOk) {
    MCU_SERIAL.println("{\"error\":\"resolver_add_failed\"}");
    while (true) {}
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MCU_SERIAL.println("{\"error\":\"allocate_tensors_failed\"}");
    while (true) {}
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  MCU_SERIAL.print("{\"event\":\"boot\",\"board\":\"esp32s3\",\"model\":\"dscnn\"");
  MCU_SERIAL.print(",\"input_bytes\":"); MCU_SERIAL.print(input->bytes);
  MCU_SERIAL.print(",\"input_dtype\":"); MCU_SERIAL.print(input->type);
  MCU_SERIAL.print(",\"output_bytes\":"); MCU_SERIAL.print(output->bytes);
  MCU_SERIAL.print(",\"output_dtype\":"); MCU_SERIAL.print(output->type);
  MCU_SERIAL.print(",\"arena_used\":"); MCU_SERIAL.print(interpreter->arena_used_bytes());
  MCU_SERIAL.print(",\"arena_size\":"); MCU_SERIAL.print(kTensorArenaSize);
  MCU_SERIAL.print(",\"clock_hz\":"); MCU_SERIAL.print(ESP.getCpuFreqMHz() * 1000000UL);
  MCU_SERIAL.println("}");
  MCU_SERIAL.println("READY");
}

void loop() {
  if ((int)input->bytes != kExpectedInputBytes) {
    MCU_SERIAL.print("{\"error\":\"input_size_mismatch\",\"got\":");
    MCU_SERIAL.print(input->bytes);
    MCU_SERIAL.print(",\"expected\":");
    MCU_SERIAL.print(kExpectedInputBytes);
    MCU_SERIAL.println("}");
    delay(2000);
    return;
  }

  if (!read_exact((uint8_t*)input_buffer, kExpectedInputBytes, kReadTimeoutMs)) {
    return;
  }

  uint32_t cycles[kBenchRuns];
  for (int i = 0; i < kBenchRuns; ++i) {
    memcpy(input->data.int8, input_buffer, kExpectedInputBytes);
    uint32_t t0 = esp_cpu_get_cycle_count();
    TfLiteStatus status = interpreter->Invoke();
    cycles[i] = esp_cpu_get_cycle_count() - t0;
    if (status != kTfLiteOk) {
      MCU_SERIAL.println("{\"error\":\"invoke_failed\"}");
      MCU_SERIAL.println("READY");
      return;
    }
  }

  qsort(cycles, kBenchRuns, sizeof(uint32_t), cmp_u32);
  uint32_t median_cycles = cycles[kBenchRuns / 2];
  float latency_ms = (float)median_cycles / (float)(ESP.getCpuFreqMHz() * 1000000UL) * 1000.0f;

  int best_idx = 0;
  int8_t best_score = output->data.int8[0];
  for (size_t i = 1; i < output->bytes; ++i) {
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      best_idx = (int)i;
    }
  }

  MCU_SERIAL.print("{\"board\":\"esp32s3\",\"model\":\"dscnn\",\"npu\":0");
  MCU_SERIAL.print(",\"latency_ms\":"); MCU_SERIAL.print(latency_ms, 4);
  MCU_SERIAL.print(",\"class\":"); MCU_SERIAL.print(best_idx);
  MCU_SERIAL.print(",\"score\":"); MCU_SERIAL.print((int)best_score);
  MCU_SERIAL.print(",\"cycles\":"); MCU_SERIAL.print(median_cycles);
  MCU_SERIAL.println("}");
  MCU_SERIAL.println("READY");
}
