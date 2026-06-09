#ifndef RF_H
#define RF_H

#include <stdint.h>
#include <stdbool.h>

static const uint8_t TREE_COUNT = 26;

extern const uint8_t tree0[] asm("_binary_iot23_0_bin_start");
extern const uint8_t tree1[] asm("_binary_iot23_1_bin_start");
extern const uint8_t tree2[] asm("_binary_iot23_2_bin_start");
extern const uint8_t tree3[] asm("_binary_iot23_3_bin_start");
extern const uint8_t tree4[] asm("_binary_iot23_4_bin_start");
extern const uint8_t tree5[] asm("_binary_iot23_5_bin_start");
extern const uint8_t tree6[] asm("_binary_iot23_6_bin_start");
extern const uint8_t tree7[] asm("_binary_iot23_7_bin_start");
extern const uint8_t tree8[] asm("_binary_iot23_8_bin_start");
extern const uint8_t tree9[] asm("_binary_iot23_9_bin_start");
extern const uint8_t tree10[] asm("_binary_iot23_10_bin_start");
extern const uint8_t tree11[] asm("_binary_iot23_11_bin_start");
extern const uint8_t tree12[] asm("_binary_iot23_12_bin_start");
extern const uint8_t tree13[] asm("_binary_iot23_13_bin_start");
extern const uint8_t tree14[] asm("_binary_iot23_14_bin_start");
extern const uint8_t tree15[] asm("_binary_iot23_15_bin_start");
extern const uint8_t tree16[] asm("_binary_iot23_16_bin_start");
extern const uint8_t tree17[] asm("_binary_iot23_17_bin_start");
extern const uint8_t tree18[] asm("_binary_iot23_18_bin_start");
extern const uint8_t tree19[] asm("_binary_iot23_19_bin_start");
extern const uint8_t tree20[] asm("_binary_iot23_20_bin_start");
extern const uint8_t tree21[] asm("_binary_iot23_21_bin_start");
extern const uint8_t tree22[] asm("_binary_iot23_22_bin_start");
extern const uint8_t tree23[] asm("_binary_iot23_23_bin_start");
extern const uint8_t tree24[] asm("_binary_iot23_24_bin_start");
extern const uint8_t tree25[] asm("_binary_iot23_25_bin_start");

const uint8_t* TREES[26] = {
    tree0,
    tree1,
    tree2,
    tree3,
    tree4,
    tree5,
    tree6,
    tree7,
    tree8,
    tree9,
    tree10,
    tree11,
    tree12,
    tree13,
    tree14,
    tree15,
    tree16,
    tree17,
    tree18,
    tree19,
    tree20,
    tree21,
    tree22,
    tree23,
    tree24,
    tree25
};

typedef struct __attribute__((packed)) Node
{
    uint8_t feature;
    float threshold;
    uint16_t left;
    uint16_t right;
}Node;

typedef struct __attribute__((packed)) Instance
{
    float duration;
    int32_t orig_bytes;
    int32_t resp_bytes;
    int32_t missed_bytes;
    int32_t orig_pkts;
    int32_t orig_ip_bytes;
    int32_t resp_pkts;
    int32_t resp_ip_bytes;

    bool proto_icmp;
    bool proto_tcp;
    bool proto_udp;

    bool conn_state_OTH;
    bool conn_state_REJ;
    bool conn_state_RSTO;
    bool conn_state_RSTOS0;
    bool conn_state_RSTR;
    bool conn_state_RSTRH;
    bool conn_state_S0;
    bool conn_state_S1;
    bool conn_state_S2;
    bool conn_state_S3;
    bool conn_state_SF;
    bool conn_state_SH;
    bool conn_state_SHR;
}Instance;

typedef struct __attribute__((packed)) MqttMessage
{
    Instance inst;
    uint8_t label;
    uint32_t id;
}MqttMessage;

float getAttribute(int idx);
uint8_t predict();

#endif