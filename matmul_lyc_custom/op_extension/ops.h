#pragma once

#include <ATen/ATen.h>

namespace ascend_kernel {
at::Tensor matmul_lyc_custom_torch(const at::Tensor &a, const at::Tensor &b);
}
