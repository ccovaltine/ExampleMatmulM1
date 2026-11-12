#include <cstdint>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "matmul_lyc_custom_host.h"

namespace ascend_kernel {

at::Tensor matmul_lyc_custom_torch(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.is_privateuseone() && b.is_privateuseone(),
                "matmul_lyc_custom inputs must be NPU tensors");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
                "matmul_lyc_custom only supports FP16 inputs");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "matmul_lyc_custom inputs must be two-dimensional");
    TORCH_CHECK(a.size(0) == 1 && a.size(1) == 7168 &&
                b.size(0) == 7168 && b.size(1) == 16160,
                "expected fixed shapes [1,7168] and [7168,16160]");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "matmul_lyc_custom requires contiguous ND inputs");
    TORCH_CHECK(a.device() == b.device(), "inputs must be on the same NPU");

    at::Tensor c = at::empty({1, 16160}, a.options());
    // stream(true) flushes torch_npu's queued work before the raw launch and
    // keeps the custom kernel ordered with preceding PyTorch NPU operations.
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    LaunchMatmulLycCustom(
        static_cast<uint32_t>(a.size(0)),
        static_cast<uint32_t>(a.size(1)),
        static_cast<uint32_t>(b.size(1)),
        a.get_device(), aclStream,
        reinterpret_cast<uint8_t *>(a.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(b.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(c.mutable_data_ptr()));
    return c;
}

} // namespace ascend_kernel
