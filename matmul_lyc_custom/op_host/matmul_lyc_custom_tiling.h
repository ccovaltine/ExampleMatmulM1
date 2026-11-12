#ifndef MATMUL_LYC_CUSTOM_TILING_H
#define MATMUL_LYC_CUSTOM_TILING_H

#include <cstdint>

namespace ascend_kernel {

struct MatmulLycCustomTiling {
    uint32_t blockNum;
    uint32_t nTiles;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t l1K;
    uint32_t kL1Tiles;
    uint32_t l1Prefetch;
};

// Host-side tiling policy. The returned scalar fields are passed directly to
// the device kernel, so no per-call GM tiling allocation or copy is required.
MatmulLycCustomTiling ComputeMatmulLycCustomTiling(
    uint32_t m, uint32_t k, uint32_t n, uint32_t availableCoreNum);

} // namespace ascend_kernel

#endif
