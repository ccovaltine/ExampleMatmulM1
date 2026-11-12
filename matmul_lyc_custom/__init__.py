"""Load the compiled matmul_lyc_custom Torch extension."""

from pathlib import Path

import torch
import torch_npu

_LIBRARY_PATH = Path(__file__).resolve().parent / "build" / "libmatmul_lyc_custom_ops.so"
if not _LIBRARY_PATH.is_file():
    raise ImportError(
        f"Custom operator library not found: {_LIBRARY_PATH}\n"
        "Build it first with: bash matmul_lyc_custom/build.sh"
    )

torch.ops.load_library(str(_LIBRARY_PATH))

# Public API requested for this example:
#   torch_npu.npu_matmul_lyc_custom(a, b)
torch_npu.npu_matmul_lyc_custom = torch.ops.ascend_custom.matmul_lyc_custom

__all__ = ["torch_npu"]
