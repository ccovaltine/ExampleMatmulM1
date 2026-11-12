#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {
TORCH_LIBRARY_FRAGMENT(ascend_custom, m)
{
    m.def("matmul_lyc_custom(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(ascend_custom, PrivateUse1, m)
{
    m.impl("matmul_lyc_custom", TORCH_FN(ascend_kernel::matmul_lyc_custom_torch));
}

at::Tensor matmul_lyc_custom_meta(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be matrices");
    TORCH_CHECK(a.size(0) == 1 && a.size(1) == 7168 &&
                b.size(0) == 7168 && b.size(1) == 16160,
                "expected fixed shapes [1,7168] and [7168,16160]");
    return at::empty({1, 16160}, a.options());
}

TORCH_LIBRARY_IMPL(ascend_custom, Meta, m)
{
    m.impl("matmul_lyc_custom", &matmul_lyc_custom_meta);
}
} // namespace
