"""商品推荐：预热后统计自定义 matmul_lyc_custom 算子的计算时间。"""

import time

import torch
import torch_npu
import matmul_lyc_custom  # noqa: F401  # 加载自定义算子并注册 torch_npu 调用接口


# 一个用户的兴趣向量为 1 x 7168，商品特征矩阵为 7168 x 16160。
M, K, N = 1, 7168, 16160


def main():
    torch.manual_seed(42)
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    # 示例使用随机数据：user_interests 是当前用户的兴趣特征，
    # product_features 的每一列对应一个商品的特征。
    user_interests = torch.randn(M, K, dtype=torch.float16).to(device)
    product_features = torch.randn(K, N, dtype=torch.float16).to(device)
    torch.npu.synchronize()

    # 预热不计时：触发首次自定义算子加载，并等待其全部完成。
    torch_npu.npu_matmul_lyc_custom(user_interests, product_features)
    torch.npu.synchronize()

    # 预热后正式执行 3 次，每次同步后再记录耗时。
    elapsed_times_ms = []
    for _ in range(3):
        start = time.perf_counter()
        # 正式计算：[1, 7168] x [7168, 16160]，由自研 AscendC kernel 执行。
        recommendation_scores = torch_npu.npu_matmul_lyc_custom(
            user_interests, product_features
        )
        torch.npu.synchronize()
        elapsed_times_ms.append((time.perf_counter() - start) * 1000)

    average_ms = sum(elapsed_times_ms) / len(elapsed_times_ms)

    print(f"设备: {torch.npu.get_device_name(0)}")
    print("算子: torch_npu.npu_matmul_lyc_custom（自研 AscendC MatMul）")
    print(f"输入: ({M}, {K}) x ({K}, {N}), dtype=float16")
    print(f"输出: {tuple(recommendation_scores.shape)}")
    for index, elapsed_ms in enumerate(elapsed_times_ms, start=1):
        print(f"第 {index} 次 MatMul 耗时: {elapsed_ms:.3f} ms")
    print(f"3 次平均耗时（已预热，不含首次加载）: {average_ms:.3f} ms")

    # 取得分最高的 5 个商品，展示这个生活场景的计算结果。
    # top_scores, top_indices = torch.topk(recommendation_scores[0], k=5)
    # print("推荐商品 Top 5:")
    # for rank, (index, score) in enumerate(
    #     zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()), start=1
    # ):
    #     print(f"  {rank}. 商品 {index}: {score:.4f}")


if __name__ == "__main__":
    main()
