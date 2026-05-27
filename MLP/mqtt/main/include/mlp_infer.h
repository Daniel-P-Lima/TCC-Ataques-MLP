#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void    mlp_init(void);
uint8_t mlp_infer(const float input[24]);

#ifdef __cplusplus
}
#endif
