#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "protocol_examples_common.h"

#include "esp_crt_bundle.h"
#include "mqtt_client.h"
#include "esp_cpu.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "mlp_infer.h"

static const char *TAG = "mqtt_mlp";

// 24 float32 features + 1 uint8 label, packed (no padding)
// Python: struct.pack('<' + 'f'*24 + 'B', *features, label)
typedef struct __attribute__((packed)) {
    float   features[24];
    uint8_t label;
} MlpMqttMessage;

_Static_assert(sizeof(MlpMqttMessage) == 97, "MlpMqttMessage size mismatch");

static QueueHandle_t mqtt_queue;
static esp_mqtt_client_handle_t client = NULL;

static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t event = event_data;
    esp_mqtt_client_handle_t c = event->client;
    int msg_id;

    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
            msg_id = esp_mqtt_client_subscribe(c, "/esp32/sub", 0);
            ESP_LOGI(TAG, "subscribed, msg_id=%d", msg_id);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
            break;
        case MQTT_EVENT_SUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_DATA:
            xQueueSend(mqtt_queue, (MlpMqttMessage *)event->data, portMAX_DELAY);
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGI(TAG, "esp-tls error: 0x%x", event->error_handle->esp_tls_last_esp_err);
                ESP_LOGI(TAG, "tls stack error: 0x%x", event->error_handle->esp_tls_stack_err);
                ESP_LOGI(TAG, "transport errno: %d (%s)",
                         event->error_handle->esp_transport_sock_errno,
                         strerror(event->error_handle->esp_transport_sock_errno));
            } else if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
                ESP_LOGI(TAG, "Connection refused: 0x%x",
                         event->error_handle->connect_return_code);
            } else {
                ESP_LOGW(TAG, "Unknown error type: 0x%x",
                         event->error_handle->error_type);
            }
            break;
        default:
            ESP_LOGI(TAG, "Other event id: %d", event->event_id);
            break;
    }
}

void mqtt_publish_task(void *pv)
{
    MlpMqttMessage msg;
    char pub_msg[24];

    while (1) {
        if (xQueueReceive(mqtt_queue, &msg, portMAX_DELAY)) {
            uint32_t t0 = esp_cpu_get_cycle_count();
            uint8_t pred = mlp_infer(msg.features);
            uint32_t t1 = esp_cpu_get_cycle_count();

            snprintf(pub_msg, sizeof(pub_msg),
                     "%" PRIu8 ":%" PRIu8 ":%" PRIu32,
                     pred, msg.label, (t1 - t0));

            esp_mqtt_client_publish(client, "/esp32/pub", pub_msg, 0, 0, 0);
        }
    }
}

static void mqtt_app_start(void)
{
    const esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address.uri = "mqtts://8fecfa22d79b48eb9d9e2009dd10c430.s1.eu.hivemq.cloud:8883",
        },
        .credentials = {
            .username = "hivemq.webclient.1779066043059",
            .authentication.password = "LI;@Nwe*M#P07dW4r2cp"
        },
    };

    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(client);
}

void app_main(void)
{
    ESP_LOGI(TAG, "[APP] Startup..");
    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "[APP] IDF version: %s", esp_get_idf_version());

    esp_log_level_set("*", ESP_LOG_ERROR);

    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // Initialize TFLM before network — see arena errors in logs early
    mlp_init();

    ESP_ERROR_CHECK(example_connect());

    mqtt_queue = xQueueCreate(10, sizeof(MlpMqttMessage));

    // 8 KB stack: TFLM Invoke needs ~3-5 KB of stack depth
    xTaskCreate(mqtt_publish_task, "mqtt_mlp", 8192, NULL, 5, NULL);

    mqtt_app_start();
}
