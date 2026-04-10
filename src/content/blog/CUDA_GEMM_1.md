---
title: 'CUDA GEMM 优化实践笔记'
description: '从 naive SGEMM 到多级 tiling 的实现思路与计算访存比分析，附 CUDA 核心代码草稿。'
pubDate: 'Apr 11 2026'
heroImage: '../../assets/lolita洛天依.jpg'
---

# CUDA GEMM 优化实践笔记

blog 第一条预定：为什么我他妈调了一个晚上的 ncu！OneDrive 害人不浅，后来者鉴之，以后新电脑绝对换 Linux。

## Naive SGEMM 的实现，以及怎么阅读 ncu 的结果

先解释一下这里的数据流转，尤其是 warp、SM、ALU 与 thread、block。简要来说，SM、warp、ALU、Register 是执行运算的物理硬件。

Block、Thread 是 CUDA 抽象出来的执行与数据绑定单元。粗略理解的话，它们有点像“计算任务的数据容器”；详细一点可以这样看：

Block 是：

- 一段被划定的Shared Memory。
- Register memory中一段被保留的物理晶体管阵列。
- SM 内部硬件同步计数器的一个特定监控目标。

Thread 是：

- 存放其私有变量如 row, col, sum的特定物理寄存器地址的集合。
- 硬件维护的一个状态字，包含其专属的 threadIdx 和 blockIdx 等只读数据。

Block, Thread 上启数据，下启 Warp。

一般人从这个角度开始，就能读懂 CUDA 代码以及尝试写 CUDA 代码了。

## 第一级 tiling 的实现，怎么阅读 ncu 的结果

这里的 blocksize 需要注意：一个 block 会被调度到一个 SM 上，而一个 SM 能容纳的 thread 数有限。比如在 RTX4060 上，一个 SM 最大只能容纳 1536 个 threads。一个 SM 有 4 个分区，每个分区 1 个 Warp Scheduler；每个 Warp Scheduler 有 12 个 instruction buffer 和 PC；每个 instruction buffer + PC 调度 1 个 warp；每个 warp 有 32 个 threads。

$$
1 \times 4 \times 1 \times 12 \times 32 = 1536
$$

所以如果是第一级tiling，一个threads计算一个标量，那么被分给一个SM的block里面就不能超过1536个元素。

然后硬件限制一个block最多有1024个threads，所以此时我们blocksize有两个选择：

- 一是 $16 \times 16 = 256$，此时 $1536/256=6$，一个SM上塞6个block刚好塞满。
- 二是 $32 \times 32 = 1024$，此时 $1536/1024=1.5$，一个SM上只能塞1个block，剩下的线程资源空置。

所以一般而言blocksize为16会是更合适的选择。

## 第二级 tiling 的实现，怎么阅读 ncu 的结果

两个tiling在本质上都是分块矩阵乘法的实现，主要是呢，分块矩阵乘法能显著提高计算访存比，这个东西是可计算能证明的。

为什么分块矩阵 MM 能显著提高计算访存比？

naive：

$$
C=A \times B,\quad A=R^{M \times K},\quad B=R^{K \times N},\quad C=R^{M \times N}
$$

首先，想一下那个立方体模型，总计算量反正是：

$$
2MNK
$$

对于C的单一元素 $C_{ij}$ 需要读取A的i行和B的j列，$2K$。

若最坏情况不复用 $M \cdot N$ 个元素，

$$
M \cdot N \cdot 2K = 2MNK
$$

次访存（读）。$M \times N$ 写回主存，即 $M \cdot N$。总访存量：

$$
2MNK+MN
$$

$$
I_{naive} = \frac{2MNK}{2MNK+MN} \approx 1
$$

Tiling C：多个 block，每个 block 大小为 $BM \cdot BN$。

- A：大小为 $BM \cdot BK$
- B：大小为 $BK \cdot BN$

C被划分为 $\frac{M}{BM} \times \frac{N}{BN}$ 个子块，遍历 $\frac{K}{BK}$ 步。

在每个步数上，从主存加载 $BM \cdot BK + BK \cdot BN$，计算一个 C 的子块。

一共：

$$
(BM \cdot BK + BK \cdot BN)\frac{K}{BK} = KBM+KBN
$$

次读取。

$$
\frac{M}{BM} \times \frac{N}{BN} \times K(BM+BN) = MNK \cdot \frac{BM+BN}{BM \cdot BN}
$$

再写回：$M \cdot N$。

$$
	herefore \text{总访存量为 } MN + MNK \frac{BM+BN}{BM \cdot BN}
$$

$$
I_{tiled} = \frac{2MNK}{MNK \frac{BM+BN}{BM \cdot BN}}
$$

若 $K$ 足够大：

$$
I_{tiled} = \frac{2}{\frac{1}{BM} + \frac{1}{BN}}
$$

若 $BM=BN=b$ 使 $I_{tiled}$ 足够大，此时：

$$
I_{tiled} = b
$$

$$
b>1
$$

这东西构成了多级tiling的理论基石。

register tiling 的思路是：在 shared memory 中的 A、B 子块上再做一次分块矩阵乘法，这次分块落在寄存器上。上文已经论证过分块矩阵乘法能提高计算访存比，这里不重复推导，直接讲实现。

一：先考虑每个 thread 与 A/B/C 三个矩阵的映射关系。首先要明确，一个 thread 计算的不是标量，而是一个 micro-matrix，大小为 TM*TN。也就是说，thread 每次要从 shared memory 读取一个 TM*BK 的 A 子块和一个 BK*TN 的 B 子块。同时，从 GMEM 到 SMEM 的预取还要考虑索引、线程加载与映射关系，这些逻辑耦合在一起，使得坐标计算会比较复杂；再加上 B 的转置，会更复杂。

实现了一下，基本就是这样：

```cpp
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

const int tid = threadIdx.y * blockDim.x + threadIdx.x;

// sharedmemory分配
__shared__ float As[BM][BK];
__shared__ float Bs[BK][BN];

// registermemory分配
float regis_A[TM];
float regis_B[TN];
float accum[TM][TN] = {0.0f};

const int bx = blockIdx.x;
const int by = blockIdx.y;

// 计算 GMEM 到 SMEM 的加载映射关系
// 256 个线程加载 128x8 的 As 矩阵
const int load_A_row = tid / BK;
const int load_A_col = tid % BK;
const int stride_A = (blockDim.x * blockDim.y) / BK;

// 256 个线程加载 8x128 的 Bs 矩阵
const int load_B_row = tid / BN;
const int load_B_col = tid % BN;
const int stride_B = (blockDim.x * blockDim.y) / BN;
const int num_tiles = (K + BK - 1) / BK;
for (int t = 0; t < num_tiles; ++t) {

    // 阶段一：将数据从 GMEM 加载到 SMEM
    for (int i = 0; i < BM; i += stride_A) {
        int global_row = by * BM + load_A_row + i;
        int global_col = t * BK + load_A_col;
        if (global_row < M && global_col < K) {
            As[load_A_row + i][load_A_col] = A[global_row * K + global_col];
        } else {
            As[load_A_row + i][load_A_col] = 0.0f;
        }
    }

    for (int i = 0; i < BK; i += stride_B) {
        int global_row = t * BK + load_B_row + i;
        int global_col = bx * BN + load_B_col;
        if (global_row < K && global_col < N) {
            Bs[load_B_row + i][load_B_col] = B[global_row * N + global_col];
        } else {
            Bs[load_B_row + i][load_B_col] = 0.0f;
        }
    }

    __syncthreads();

    // 阶段二：将数据从 SMEM 加载到寄存器，并进行外积计算
    for (int k = 0; k < BK; ++k) {
        // 将 SMEM 的数据加载至 Thread 本地寄存器
        for (int i = 0; i < TM; ++i) {
            frag_A[i] = As[threadIdx.y * TM + i][k];
        }
        for (int j = 0; j < TN; ++j) {
            frag_B[j] = Bs[k][threadIdx.x * TN + j];
        }

        // 执行核心 FMA 运算
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                accum[i][j] += frag_A[i] * frag_B[j];
            }
        }
    }

    __syncthreads();
}
```

## float4 的实现，怎么阅读 ncu 的结果

float4 的原理很简单：一次搬运 4 个 float 元素。但实现上还是有一些细节问题，需要强制调整 GMEM、SMEM、RMEM 之间的数据流转接口，不过计算逻辑本身不变，这个下期再讲。

## double buffering 的实现，怎么阅读 ncu 的结果