#ifndef MATMUL_LYC_CUSTOM_HOST_H
#define MATMUL_LYC_CUSTOM_HOST_H

#include <cstdint>

#include "acl/acl.h"

namespace ascend_kernel {

void LaunchMatmulLycCustom(
    uint32_t m, uint32_t k, uint32_t n, int32_t deviceId,
    aclrtStream stream, uint8_t *a, uint8_t *b, uint8_t *c);

} // namespace ascend_kernel

#endif
