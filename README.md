# ExampleMatmulM1：AscendC 自定义 FP16 GEMV

这是一个面向 Ascend 310P3 的 AscendC 自定义算子学习项目，以商品推荐评分为场景，实现：

```text
[1, 7168] × [7168, 16160] → [1, 16160]
```

输入和输出为 FP16，L0C 使用 FP32 累加。Python 通过以下接口调用自研算子：

```python
torch_npu.npu_matmul_lyc_custom(a, b)
```

项目不调用 CANN 高阶 Matmul API，而是在 AscendC kernel 中显式实现 GM→L1→L0 数据搬运、Mmad 和 L0C→UB→GM 输出路径。

## 已验证环境

```text
Ascend 310P3
CANN 9.0.0
SocVersion ASCEND310P
--npu-arch=dav-2002
PyTorch 2.9.0+cpu
torch_npu 2.9.0.post2.dev20260411
```

## 目录结构

```text
ExampleMatmulM1/
├── matmul_m1.py                  # 商品推荐场景、预热和正式三次计时
├── matmul_lyc_custom/
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── __init__.py               # 加载动态库并注册 torch_npu 接口
│   ├── op_extension/             # Torch 扩展和算子注册
│   ├── op_host/                  # Host launcher 和 Tiling 计算
│   └── op_kernel/                # AscendC kernel
└── learning_plan/                # 分阶段学习指导、练习和参考答案
```

## 编译

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/l00949281-benchmark/ExampleMatmulM1/matmul_lyc_custom
bash build.sh
```

构建产物为：

```text
build/libmatmul_lyc_custom_ops.so
```

`build/` 和动态库不纳入 Git 版本管理，克隆仓库后需要在本机 CANN 环境重新编译。

## 运行

```bash
cd /home/l00949281-benchmark/ExampleMatmulM1
python3 matmul_m1.py
```

如需将物理 NPU 3 映射为程序内的 `npu:0`：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 python3 matmul_m1.py
```

程序先预热一次并同步，再正式执行三次，分别报告耗时及平均值，不把首次算子加载时间计入正式结果。`torch.topk` 示例保留在计时循环之后，不计入 MatMul 时间。

当前实现固定支持：

```text
M = 1
K = 7168
N = 16160
dtype = FP16
```

Host Tiling 基线：

```text
blockNum = 8（根据设备可用 Cube 核数确定）
nTiles = 101
baseN = 160
baseK = 64
l1K = 256
kL1Tiles = 28
l1Prefetch = 4
```

## 学习路线

`learning_plan/` 包含一份总计划和六篇阶段指导：

1. [总体学习计划](learning_plan/1_AscendC算子分析与优化学习计划.md)
2. [阶段1：读懂算子与软硬件映射](learning_plan/2_阶段1_读懂算子与软硬件映射.md)
3. [阶段2：建立可信性能与精度基线](learning_plan/3_阶段2_建立可信性能与精度基线.md)
4. [阶段3：用 Profiling 建立可证伪假设](learning_plan/4_阶段3_用Profiling建立可证伪假设.md)
5. [阶段4：Tiling 单变量实验](learning_plan/5_阶段4_Tiling单变量实验指导.md)
6. [阶段5：流水线、Scalar 与数据布局优化](learning_plan/6_阶段5_流水线Scalar与数据布局优化.md)
7. [阶段6：Cube 与 Vector 路线比较](learning_plan/7_阶段6_Cube与Vector路线比较.md)

每篇阶段指导均包含学习目标、实践步骤、课后题和参考答案。

## Profiling 数据说明

普通应用 profiling 的 `task_time`、`op_summary` 与 `msopprof` 单算子采集是不同口径。`msopprof --replay-mode=kernel` 会重放 kernel，应用墙钟时间包含重放等待，不能作为正常性能对比结果。

在当前 Ascend 310P3 + CANN 9.0.0 环境中，`msopprof` 可提供累计 PipeUtilization 等指标，但不支持通过 `TimelineDetail` 获取 MTE1/MTE2/Cube 的真实逐时间内部泳道。现有 `visualize_data.bin` 的累计统计不能还原这种内部 Timeline。

Profiling 输出目录和 Device 二进制文件已被 `.gitignore` 排除。
