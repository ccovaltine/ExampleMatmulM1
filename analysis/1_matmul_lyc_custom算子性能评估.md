# matmul_lyc_custom 算子性能评估

- 算子路径：`/home/l00949281-benchmark/ExampleMatmulM1/matmul_lyc_custom`
- 平台：Ascend 310P3（8 AI Core，DAV_2002）
- 形状：`[1, 7168] x [7168, 16160]`，fp16
- Profiling 数据：`msprof_output/PROF_000001_20261112192136038_00892461DPHHDFPQ/mindstudio_profiler_output/`
- 分析时间：2026-11-18

## 1. 结论

**当前状态：比原生 `torch.mm` 快 1.53×，但只用到设备峰值带宽的 45%，还有约 2.2× 的优化空间。**

同一张 310P3、同机、同 dtype 的实测对照：

| 实现 | 耗时 | 有效带宽 |
|---|---|---|
| `matmul_lyc_custom`（自研） | 2.662 ms | 87.0 GB/s |
| `torch.mm`（aclnn MatMul） | 4.078 ms | 56.8 GB/s |
| `torch.matmul` | 4.071 ms | 56.9 GB/s |
| 纯 `copy_` 256 MB（实测硬件上限） | — | **190.6 GB/s** |
| 理论下限（231.65 MB ÷ 190.6 GB/s） | **1.22 ms** | 100% |

数据量核算：`(1×7168 + 7168×16160 + 1×16160) × 2 B = 231,648,000 B ≈ 231.65 MB`。

## 2. 判断方法（M=1 GEMV 场景专用）

这类形状**不能用 TFLOPS / MFU / Cube 利用率判断好坏**，必须用带宽：

1. **先算算术强度**：`2·M·K·N FLOPs ÷ (M·K+K·N+M·N)·2 B = 2 FLOP/Byte` → 远低于 310P3 的算力/带宽比（约 44 TFLOPS ÷ 200 GB/s ≈ 220 FLOP/B），**天然 memory-bound**，唯一判据是"搬 231.65 MB 需要多久"。
2. **实测硬件上限**：在同设备上跑纯带宽基准（`y.copy_(x)`，256 MB，读写计入），得到 190.6 GB/s，与 310P3 标称 204.8 GB/s 吻合 → 用它当分母，而不是纸面峰值。
3. **读 profiler 定位瓶颈**：`op_summary_*.csv` 中看 `mte2_ratio / mac_ratio / scalar_ratio / aicore_time / Task Duration / Block Num`。
4. **横向对标**：同 shape 与 `torch.mm` / aclnn 比，确认是否比"免费实现"更快。
5. **交叉验证**：wall-clock（2.662 ms）与 profiler Task Duration（2632–2657 µs）一致，说明测量可信、无隐藏开销。

## 3. 现有 Profiling 证据

来源：`PROF_000001_20261112192136038_00892461DPHHDFPQ/mindstudio_profiler_output/op_summary_*.csv`（4 次调用：1 次预热 + 3 次正式）

| 指标 | 值 | 解读 |
|---|---|---|
| Task Duration | 2632–2657 µs | 与 wall-clock 一致 |
| aicore_time | 2435–2458 µs | 差值约 200 µs 是任务等待/启动开销 |
| **mte2_ratio** | **0.99** | **99% 的核内时间在等 GM→L1 数据，MTE2 是唯一瓶颈** |
| mac_ratio | 0.029 | Cube 真正计算只有约 70 µs，算力完全不是瓶颈 |
| scalar_ratio | 0.347 | 约 852 µs 的标量开销，说明搬运被拆成大量细碎指令 |
| vec_ratio | 0.0 | 向量单元几乎不参与 |
| Block Num | 8 | 8 核全部用上；nTiles=101，13/12 分配差约 1%，尾部不均衡影响很小 |
| cube_utilization = 92.5% | — | **口径陷阱**：mac 只占 2.9%，该指标不能作为"性能好"的证据 |

`op_statistic_*.csv` 汇总：`matmul_lyc_custom, AI_CORE, Count=4, Total=10573.58 us, Avg=2643.4 us, Ratio=100%`。

### 根因定位

`op_kernel/matmul_lyc_custom_kernel.cpp:74`：

```cpp
AscendC::DataCopyParams params(C0, 1, (N - C0) * sizeof(half) / 32, 0);  // 16 行 × 32B
```

B 矩阵按 16×16 fractal 逐个搬，**每次只搬 32 B/行**。单核 12~13 个 N tile × 28 个 K 段 × (baseN/16=10) 个 n-fractal × 16 行 ≈ 每核 6 万次 32 B 小块访问，远低于 512 B 有效 burst → 这是只跑到 45% 带宽、且 scalar 占 35% 的直接原因。

其他已确认的结构性开销：

- `ComputeTile` 中 `CopyB` 后立即 `DeQue`，`l1Prefetch_` 只作用于内层循环，跨 L1 段无真正预取重叠（`matmul_lyc_custom_kernel.cpp:110-142`）。
- `CopyA` 每个 K 段都先 `DataCopy(zeroUb)` 8 KB 做零填充，再逐 16 次搬 A，属重复开销（`:54-67`）。
- tiling 为固定常量：`baseN=160, baseK=64, l1K=256, kL1Tiles=28, blockNum=min(nTiles,8)`（`op_host/matmul_lyc_custom_tiling.cpp:14-36`）。

## 4. 优化建议

- **[P0] 把搬运粒度从 32 B 提到 ≥512 B**：不要按 fractal 搬 B，改为一次 2D `DataCopy` 搬 `16 行 × baseN×2B`（baseN=160 → 320 B/行）的连续块进 UB，再做 nd2nz 排布；或改用 `LoadData2DTranspose` 在 L1 内完成转置。这一步直接决定能否从 87 GB/s 上到 150+ GB/s。
- **[P0] 权重预处理（收益最大）**：示例场景 B（商品特征矩阵 231 MB）每次调用不变，可一次性转成 NZ 并常驻，之后逐 K 段做连续大块拷贝，彻底消除 32 B 碎搬和 scalar 开销。
- **[P1] 补 K 段双缓冲预取**：把 `CopyB(outer+1)` 提前到当前 `Mmad` 之前；queue 深度已经是 2，改成实际流水即可。
- **[P1] 去掉 `CopyA` 的零填充重复劳动**：只在 L1 首段初始化一次，或直接用带 padding 的 nd2nz。
- **[P2] 收敛启动开销**：Task Wait Time 119–165 µs，aicore_time 比 Duration 少约 200 µs，检查 host 侧 tiling/下发路径。

## 5. 验证方法

1. 修改后重新采集 `msprof`，检查 `op_summary_*.csv`：
   - 有效带宽 = 231.65 MB ÷ Task Duration，目标 >150 GB/s（即 <1.55 ms）；
   - `mte2_ratio` 仍接近 1 属正常（memory-bound 本该如此），但 `scalar_ratio` 应从 0.35 降到 0.1 以下。
2. 用同脚本与 `torch.mm` 再比一次，确认优势从 1.53× 扩大到 2.6× 以上。
3. 用 `y.copy_(x)` 基准复测峰值带宽，确认分母未变（排除降频/温度影响；当前各卡温度 77–84 ℃，需注意）。

## 6. 复现命令

对照脚本：`analysis/bench_mm_compare.py`（核心逻辑）：

```python
import torch, torch_npu, matmul_lyc_custom
M, K, N = 1, 7168, 16160
a = torch.randn(M, K, dtype=torch.float16).npu()
b = torch.randn(K, N, dtype=torch.float16).npu()
# 预热 3 次后用 torch.npu.Event 计时 10 次
```

执行：在 `ExampleMatmulM1` 目录下运行 `python analysis/bench_mm_compare.py`（确保自定义算子库可加载）。

## 7. 待验证项

- 待验证：`memory_bound=34.138` 字段的官方口径（是否为 memory-bound 时间占比），暂无文档确认，本报告未将其作为判据。
- 待验证：310P3 该卡实际 HBM 标称带宽（190.6 GB/s 为实测 copy 值，标称 204.8 GB/s 需以设备规格书确认）。
- 待验证：改用大块搬运后的实际收益，需按第 5 节重新采集确认。
