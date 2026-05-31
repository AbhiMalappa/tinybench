#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dscnn_model_data.h"

constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr int kExpectedInputBytes = 49 * 10;
constexpr int kBenchRuns = 10;
constexpr uint32_t kReadTimeoutMs = 30000;

static int8_t input_buffer[kExpectedInputBytes];

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
}

static void enable_cycle_counter() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
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
    if (Serial.available() > 0) {
      int b = Serial.read();
      if (b >= 0) dst[got++] = (uint8_t)b;
    }
    if (millis() - start > timeout_ms) return false;
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  enable_cycle_counter();

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("{\"error\":\"schema_mismatch\",\"got\":");
    Serial.print(model->version());
    Serial.print(",\"expected\":");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("}");
    while (true) {}
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("{\"error\":\"allocate_tensors_failed\"}");
    while (true) {}
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("{\"event\":\"boot\",\"board\":\"arduino_nano33\",\"model\":\"dscnn\"");
  Serial.print(",\"input_bytes\":"); Serial.print(input->bytes);
  Serial.print(",\"input_dtype\":"); Serial.print(input->type);
  Serial.print(",\"output_bytes\":"); Serial.print(output->bytes);
  Serial.print(",\"output_dtype\":"); Serial.print(output->type);
  Serial.print(",\"arena_used\":"); Serial.print(interpreter->arena_used_bytes());
  Serial.print(",\"arena_size\":"); Serial.print(kTensorArenaSize);
  Serial.print(",\"clock_hz\":"); Serial.print(SystemCoreClock);
  Serial.println("}");
  Serial.println("READY");
}

void loop() {
  if ((int)input->bytes != kExpectedInputBytes) {
    Serial.print("{\"error\":\"input_size_mismatch\",\"got\":");
    Serial.print(input->bytes);
    Serial.print(",\"expected\":");
    Serial.print(kExpectedInputBytes);
    Serial.println("}");
    delay(2000);
    return;
  }

  if (!read_exact((uint8_t*)input_buffer, kExpectedInputBytes, kReadTimeoutMs)) {
    return;
  }

  uint32_t cycles[kBenchRuns];
  for (int i = 0; i < kBenchRuns; ++i) {
    memcpy(input->data.int8, input_buffer, kExpectedInputBytes);
    DWT->CYCCNT = 0;
    TfLiteStatus status = interpreter->Invoke();
    uint32_t c = DWT->CYCCNT;
    if (status != kTfLiteOk) {
      Serial.println("{\"error\":\"invoke_failed\"}");
      Serial.println("READY");
      return;
    }
    cycles[i] = c;
  }

  qsort(cycles, kBenchRuns, sizeof(uint32_t), cmp_u32);
  uint32_t median_cycles = cycles[kBenchRuns / 2];
  float latency_ms = (float)median_cycles / (float)SystemCoreClock * 1000.0f;

  int best_idx = 0;
  int8_t best_score = output->data.int8[0];
  for (size_t i = 1; i < output->bytes; ++i) {
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      best_idx = (int)i;
    }
  }

  Serial.print("{\"board\":\"arduino_nano33\",\"model\":\"dscnn\",\"npu\":0");
  Serial.print(",\"latency_ms\":"); Serial.print(latency_ms, 4);
  Serial.print(",\"class\":"); Serial.print(best_idx);
  Serial.print(",\"score\":"); Serial.print((int)best_score);
  Serial.print(",\"cycles\":"); Serial.print(median_cycles);
  Serial.println("}");
  Serial.println("READY");
}
