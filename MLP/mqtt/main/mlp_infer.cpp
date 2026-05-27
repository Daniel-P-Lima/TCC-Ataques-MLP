#include "mlp_infer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_log.h"

static const char *TAG = "mlp_infer";

extern const uint8_t _binary_mlp_iot23_tflite_start[];
extern const uint8_t _binary_mlp_iot23_tflite_end[];

// Model uses only FullyConnected (with BatchNorm folded in) and Softmax
static tflite::MicroMutableOpResolver<2> s_resolver;

static constexpr int kArenaSize = 40 * 1024;
static uint8_t tensor_arena[kArenaSize];

static tflite::MicroInterpreter *s_interp = nullptr;

extern "C" void mlp_init(void)
{
    tflite::InitializeTarget();

    const tflite::Model *model = tflite::GetModel(_binary_mlp_iot23_tflite_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema version mismatch: got %d, expected %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    s_resolver.AddFullyConnected();
    s_resolver.AddSoftmax();

    static tflite::MicroInterpreter interp(
        model, s_resolver, tensor_arena, kArenaSize);
    s_interp = &interp;

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors falhou — aumente kArenaSize");
        s_interp = nullptr;
        return;
    }

    ESP_LOGI(TAG, "Arena usada: %d / %d bytes",
             (int)s_interp->arena_used_bytes(), kArenaSize);
    ESP_LOGI(TAG, "Modelo pronto. Input shape: [1, %d]",
             s_interp->input(0)->dims->data[1]);
}

extern "C" uint8_t mlp_infer(const float input[24])
{
    if (!s_interp) return 0;

    float *in = s_interp->input(0)->data.f;
    for (int i = 0; i < 24; i++) in[i] = input[i];

    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke falhou");
        return 0;
    }

    float *out = s_interp->output(0)->data.f;
    uint8_t best = 0;
    for (int i = 1; i < 11; i++) {
        if (out[i] > out[best]) best = i;
    }
    return best;
}
