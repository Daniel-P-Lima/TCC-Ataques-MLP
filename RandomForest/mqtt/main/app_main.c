/*
 * SPDX-FileCopyrightText: 2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Unlicense OR CC0-1.0
 */
/* MQTT over TLS Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

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

#include "RF.h"
#include "esp_cpu.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

static const char *TAG = "mqtts_example";

static const uint8_t FEATURE_COUNT = 24;
Instance instance;
static QueueHandle_t mqtt_queue;
static esp_mqtt_client_handle_t client = NULL;

/*
 * @brief Event handler registered to receive MQTT events
 *
 *  This function is called by the MQTT client event loop.
 *
 * @param handler_args user data registered to the event.
 * @param base Event base for the handler(always MQTT Base in this example).
 * @param event_id The id for the received event.
 * @param event_data The data for the event, esp_mqtt_event_handle_t.
 */
static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    ESP_LOGD(TAG, "Event dispatched from event loop base=%s, event_id=%" PRIi32, base, event_id);
    esp_mqtt_event_handle_t event = event_data;
    esp_mqtt_client_handle_t client = event->client;
    int msg_id;
    switch((esp_mqtt_event_id_t)event_id)
    {
        case MQTT_EVENT_CONNECTED:
        {
            ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
            msg_id = esp_mqtt_client_subscribe(client, "/esp32/sub", 0);
            ESP_LOGI(TAG, "sent subscribe successful, msg_id=%d", msg_id);
            break;
        }
        case MQTT_EVENT_DISCONNECTED:
        {
            ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
            break;
        }
        case MQTT_EVENT_SUBSCRIBED:
        {
            ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED, msg_id=%d, return code=0x%02x ", event->msg_id, (uint8_t)*event->data);
            // msg_id = esp_mqtt_client_publish(client, "/esp32/pub", "data", 0, 0, 0);
            // ESP_LOGI(TAG, "sent publish successful, msg_id=%d", msg_id);
            break;
        }
        case MQTT_EVENT_UNSUBSCRIBED:
        {
            ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED, msg_id=%d", event->msg_id);
            break;
        }
        case MQTT_EVENT_PUBLISHED:
        {
            ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
            break;
        }
        case MQTT_EVENT_DATA:
        {
            ESP_LOGI(TAG, "MQTT_EVENT_DATA");

            xQueueSend(mqtt_queue, (MqttMessage*)event->data, portMAX_DELAY);
            break;
        }
        case MQTT_EVENT_ERROR:
        {
            ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGI(TAG, "Last error code reported from esp-tls: 0x%x", event->error_handle->esp_tls_last_esp_err);
                ESP_LOGI(TAG, "Last tls stack error number: 0x%x", event->error_handle->esp_tls_stack_err);
                ESP_LOGI(TAG, "Last captured errno : %d (%s)", event->error_handle->esp_transport_sock_errno,
                        strerror(event->error_handle->esp_transport_sock_errno));
            }
            else if(event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED)
            {
                ESP_LOGI(TAG, "Connection refused error: 0x%x", event->error_handle->connect_return_code);
            }
            else
            {
                ESP_LOGW(TAG, "Unknown error type: 0x%x", event->error_handle->error_type);
            }
            break;
        }
        default:
        {
            ESP_LOGI(TAG, "Other event id:%d", event->event_id);
            break;
        }
    }
}

void mqtt_publish_task(void *pv)
{
    MqttMessage rcv_msg;
    char pub_msg[19];

    while(1)
    {
        if(xQueueReceive(mqtt_queue, &rcv_msg, portMAX_DELAY))
        {
            // to do: tornar instance local
            instance = rcv_msg.inst;

            uint32_t start = esp_cpu_get_cycle_count();
            uint8_t pred = predict();
            uint32_t stop = esp_cpu_get_cycle_count();
            
            sprintf(pub_msg, "%" PRIu8 ":%" PRIu8 ":%" PRIu32, pred, rcv_msg.label, (stop - start));

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
        }
        //.buffer.size = 4096,
    };

    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    client = esp_mqtt_client_init(&mqtt_cfg);
    /* The last argument may be used to pass data to the event handler, in this example mqtt_event_handler */
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(client);
}

float getAttribute(int idx)
{
    switch(idx)
    {
        case 0: return instance.duration;

        case 1: return (float)(instance.orig_bytes);
        case 2: return (float)(instance.resp_bytes);
        case 3: return (float)(instance.missed_bytes);
        case 4: return (float)(instance.orig_pkts);
        case 5: return (float)(instance.orig_ip_bytes);
        case 6: return (float)(instance.resp_pkts);
        case 7: return (float)(instance.resp_ip_bytes);

        case 8:  return instance.proto_icmp ? 1.0f : 0.0f;
        case 9:  return instance.proto_tcp  ? 1.0f : 0.0f;
        case 10: return instance.proto_udp  ? 1.0f : 0.0f;

        case 11: return instance.conn_state_OTH ? 1.0f : 0.0f;
        case 12: return instance.conn_state_REJ ? 1.0f : 0.0f;
        case 13: return instance.conn_state_RSTO ? 1.0f : 0.0f;
        case 14: return instance.conn_state_RSTOS0 ? 1.0f : 0.0f;
        case 15: return instance.conn_state_RSTR ? 1.0f : 0.0f;
        case 16: return instance.conn_state_RSTRH ? 1.0f : 0.0f;
        case 17: return instance.conn_state_S0 ? 1.0f : 0.0f;
        case 18: return instance.conn_state_S1 ? 1.0f : 0.0f;
        case 19: return instance.conn_state_S2 ? 1.0f : 0.0f;
        case 20: return instance.conn_state_S3 ? 1.0f : 0.0f;
        case 21: return instance.conn_state_SF ? 1.0f : 0.0f;
        case 22: return instance.conn_state_SH ? 1.0f : 0.0f;
        case 23: return instance.conn_state_SHR ? 1.0f : 0.0f;

        default: return 0.0f;
    }
}

uint8_t predict()
{
    uint8_t results[FEATURE_COUNT] = {};
    memset(results, 0, FEATURE_COUNT);

    uint8_t highest = 0;
    uint8_t tie_amount = 1;
    uint8_t ret = 0;

    for(uint8_t i = 0; i < TREE_COUNT; i++)
    {
        Node *arena = (Node*)(TREES[i]);
        Node *it = arena;
        bool is_leaf = false;

        while(!is_leaf)
        {
            if(getAttribute(it->feature) <= it->threshold)
            {
                // Go left
                it = arena + it->left;
            }
            else
            {
                // Go right
                it = arena + it->right;
            }

            is_leaf = (it->left == it->right);
        }

        results[it->feature]++;

        if(results[it->feature] > highest)
        {
            highest = results[it->feature];
            tie_amount = 1;
            ret = it->feature;
        }
        else if(results[it->feature] == highest)
        {
            tie_amount++;
        }
    }

    if(tie_amount > 1)
    {
        uint8_t cont = (rand() % tie_amount);

        for(uint8_t i = 0; i < FEATURE_COUNT; i++)
        {
            if(results[i] == highest)
            {
                if(cont == 0)
                {
                    ret = i;
                }
                else
                {
                    cont--;
                }
            }
        }
    }

    return ret;
}

void app_main(void)
{
    ESP_LOGI(TAG, "[APP] Startup..");
    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "[APP] IDF version: %s", esp_get_idf_version());

    esp_log_level_set("*", ESP_LOG_ERROR);
    // esp_log_level_set("*", ESP_LOG_INFO);
    // esp_log_level_set("esp-tls", ESP_LOG_VERBOSE);
    // esp_log_level_set("mqtt_client", ESP_LOG_VERBOSE);
    // esp_log_level_set("mqtt_example", ESP_LOG_VERBOSE);
    // esp_log_level_set("transport_base", ESP_LOG_VERBOSE);
    // esp_log_level_set("transport", ESP_LOG_VERBOSE);
    // esp_log_level_set("outbox", ESP_LOG_VERBOSE);

    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    srand(42);

    /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
     * Read "Establishing Wi-Fi or Ethernet Connection" section in
     * examples/protocols/README.md for more information about this function.
     */
    ESP_ERROR_CHECK(example_connect());

    mqtt_queue = xQueueCreate(10, sizeof(MqttMessage));

    xTaskCreate(mqtt_publish_task, "mqtt_pub", 4096, NULL, 5, NULL);

    mqtt_app_start();
}
