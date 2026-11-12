#include "matmul_lyc_custom_tiling.h"

#include <algorithm>
#include <stdexcept>

namespace ascend_kernel {

MatmulLycCustomTiling ComputeMatmulLycCustomTiling(
    uint32_t m, uint32_t k, uint32_t n, uint32_t availableCoreNum)
{
    constexpr uint32_t expectedM = 1;
    constexpr uint32_t expectedK = 7168;
    constexpr uint32_t expectedN = 16160;
    constexpr uint32_t baseN = 160;
    constexpr uint32_t baseK = 64;
    constexpr uint32_t l1K = 256;

    if (m != expectedM || k != expectedK || n != expectedN) {
        throw std::invalid_argument(
            "matmul_lyc_custom only supports [1,7168] x [7168,16160]");
    }
    if (availableCoreNum == 0) {
        throw std::invalid_argument("availableCoreNum must be greater than zero");
    }
    if (n % baseN != 0 || k % l1K != 0 || l1K % baseK != 0) {
        throw std::invalid_argument("the fixed shape is incompatible with the tile sizes");
    }

    MatmulLycCustomTiling tiling{};
    tiling.nTiles = n / baseN;
    tiling.baseN = baseN;
    tiling.baseK = baseK;
    tiling.l1K = l1K;
    tiling.kL1Tiles = k / l1K;
    tiling.l1Prefetch = l1K / baseK;
    tiling.blockNum = std::min(tiling.nTiles, availableCoreNum);
    return tiling;
}

} // namespace ascend_kernel
