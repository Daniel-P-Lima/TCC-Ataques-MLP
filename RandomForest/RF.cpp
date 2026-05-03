#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include "RF.h"
#include <cstdio>
#include <algorithm>

const uint8_t TREE_COUNT = 50;
const std::string BIN_TREES_PATH{"./binTrees/iot23-"};
const std::string FILE_EXTENSION{".bin"};

const uint8_t FEATURE_COUNT = 24;

std::array<std::pair<Node*, uint16_t>, TREE_COUNT> trees;

void readInputFiles()
{
    for(uint8_t i = 0; i < TREE_COUNT; i++)
    {
        const std::string FILE_PATH{BIN_TREES_PATH + std::to_string(i) + FILE_EXTENSION};

        std::cout << "Opening file " << FILE_PATH << std::endl;
        std::ifstream file{FILE_PATH, std::ios::binary};

        uint16_t node_count;
        file.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
        //std::cout << node_count << std::endl;

        trees[i] = {static_cast<Node*>(std::malloc(sizeof(Node) * node_count)), node_count};

        file.read(reinterpret_cast<char*>(trees[i].first), sizeof(Node) * node_count);

        file.close();
    }

    std::cout << std::endl;
}

void visualizeNodes()
{
    for(std::pair p : trees)
    {
        Node* tmp = p.first;
        uint16_t node_count = p.second;

        for(uint16_t i = 0; i < node_count; i++, tmp++)
        {
            printf("Feature: %d\n", tmp->feature);
            printf("Threshold: %.2f\n", tmp->threshold);
            printf("Left: %hu\n", tmp->left);
            printf("Right: %hu\n\n", tmp->right);
        }
    }
}

Instance fileToinstance(const char* filename)
{
    Instance instance;

    std::ifstream file{filename, std::ios::binary};

    file.read(reinterpret_cast<char*>(&instance), sizeof(Instance));

    file.close();

    return instance;
}

float getAttribute(const Instance& inst, int idx) {
    switch(idx)
    {
        case 0: return inst.duration;

        case 1: return static_cast<float>(inst.orig_bytes);
        case 2: return static_cast<float>(inst.resp_bytes);
        case 3: return static_cast<float>(inst.missed_bytes);
        case 4: return static_cast<float>(inst.orig_pkts);
        case 5: return static_cast<float>(inst.orig_ip_bytes);
        case 6: return static_cast<float>(inst.resp_pkts);
        case 7: return static_cast<float>(inst.resp_ip_bytes);

        case 8:  return inst.proto_icmp ? 1.0f : 0.0f;
        case 9:  return inst.proto_tcp  ? 1.0f : 0.0f;
        case 10: return inst.proto_udp  ? 1.0f : 0.0f;

        case 11: return inst.conn_state_OTH ? 1.0f : 0.0f;
        case 12: return inst.conn_state_REJ ? 1.0f : 0.0f;
        case 13: return inst.conn_state_RSTO ? 1.0f : 0.0f;
        case 14: return inst.conn_state_RSTOS0 ? 1.0f : 0.0f;
        case 15: return inst.conn_state_RSTR ? 1.0f : 0.0f;
        case 16: return inst.conn_state_RSTRH ? 1.0f : 0.0f;
        case 17: return inst.conn_state_S0 ? 1.0f : 0.0f;
        case 18: return inst.conn_state_S1 ? 1.0f : 0.0f;
        case 19: return inst.conn_state_S2 ? 1.0f : 0.0f;
        case 20: return inst.conn_state_S3 ? 1.0f : 0.0f;
        case 21: return inst.conn_state_SF ? 1.0f : 0.0f;
        case 22: return inst.conn_state_SH ? 1.0f : 0.0f;
        case 23: return inst.conn_state_SHR ? 1.0f : 0.0f;

        default: return 0.0f;
    }
}

uint8_t fit(Instance& instance)
{
    std::array<uint8_t, FEATURE_COUNT> results{};

    for(uint8_t i = 0; i < TREE_COUNT; i++)
    {
        Node* tree_start = trees[i].first;
        Node* it = tree_start;
        bool is_leaf{false};

        while(!is_leaf)
        {
            if(getAttribute(instance, it->feature) <= it->threshold)
            {
                // Go left
                it = tree_start + it->left;
            }
            else
            {
                // Go right
                it = tree_start + it->right;
            }

            is_leaf = (it->left == it->right);
        }

        results[it->feature]++;
    }

    auto predicted_feature = std::max_element(results.begin(), results.end());

    return static_cast<uint8_t>(std::distance(results.begin(), predicted_feature));
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <binary instance file>" << std::endl;
        return 1;
    }

    readInputFiles();

    //visualizeNodes();

    Instance instance = fileToinstance(argv[1]);

    printf("%d\n", fit(instance));

    for(std::pair p : trees)
    {
        std::free(p.first);
    }

    return 0;
}