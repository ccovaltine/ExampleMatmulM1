"""Compare custom AscendC matmul_lyc_custom vs torch matmul for [1,7168]x[7168,16160] fp16."""
import torch
import torch_npu
import matmul_lyc_custom  # noqa: F401

M, K, N = 1, 7168, 16160
BYTES = (M * K + K * N + M * N) * 2  # fp16

dev = torch.device("npu:0")
torch.npu.set_device(dev)
torch.manual_seed(42)
a = torch.randn(M, K, dtype=torch.float16).to(dev)
b = torch.randn(K, N, dtype=torch.float16).to(dev)


def bench(fn, name, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"{name}: {ms:.3f} ms/iter   effective_bw={BYTES/ms/1e6:.1f} GB/s")
    return ms


bench(lambda: torch_npu.npu_matmul_lyc_custom(a, b), "custom matmul_lyc_custom")
bench(lambda: torch.mm(a, b), "torch.mm (aclnn)")
bench(lambda: torch.matmul(a, b), "torch.matmul")
