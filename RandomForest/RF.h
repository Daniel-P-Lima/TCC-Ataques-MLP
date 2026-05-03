#ifndef RF_H
#define RF_H

#include <cstdint>

struct __attribute__((packed)) Node
{
    uint8_t feature;
    float threshold;
    uint16_t left;
    uint16_t right;
};

struct __attribute__((packed)) Instance
{
    float duration;
    uint32_t orig_bytes;
    uint32_t resp_bytes;
    uint32_t missed_bytes;
    uint32_t orig_pkts;
    uint32_t orig_ip_bytes;
    uint32_t resp_pkts;
    uint32_t resp_ip_bytes;

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
};

#endif