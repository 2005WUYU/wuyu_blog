---
title: 'Qwen2.5-7B LoRA 微调显存分析与 FlashAttention 原理'
description: '从全参数微调到 LoRA 的显存拆解，结合混合精度与 FlashAttention 核心机制的推导说明。'
pubDate: 'Apr 06 2026'
heroImage: '../../assets/八辈子洛天依.jpg'
---

## 第一部分：分析显存占用

## 1. 先分析全参数微调的显存占用

$$
M_{total} = M_{weights} + M_{gradients} + M_{optimizer} + M_{activations}
$$

首先，Qwen 2.5 7B，70亿参数，采用 BF16 精度，$14 \times 10^9 \text{ Bytes} = 14 \text{ GB}$。

其次，在反向传播中，损失函数 $L$ 对参数矩阵求导得到的梯度 $\nabla W$，需要与模型参数的形状完全对齐。梯度同样以 BF16/FP16 格式存储。

$$
M_{gradients} = 2P \text{ (Bytes)}
$$

然后在优化器状态下，AdamW 优化器为了保证更新的精度和稳定性，需要以 FP32 格式进行权重的更新。需要维护以下三个完整的参数副本：主权重副本：用于高精度累加更新，占用 $4P$ Bytes。一阶动量：过去梯度的指数衰减平均值（FP32），占用 $4P$ Bytes。二阶动量：过去梯度平方的指数衰减平均值（FP32），占用 $4P$ Bytes。

$$
M_{static} = 2P (\text{权重}) + 2P (\text{梯度}) + 12P (\text{优化器}) = 16P \text{ (Bytes)}
$$

$$
M_{static} = 16 \times 7 \times 10^9 \text{ Bytes} \approx 112 \text{ GB}
$$

激活值，假设我们用 $b=1, s=4096$ 的数据：

$$
M_{activations} \approx 34 \times 1 \times 4096 \times 3584 \times 28 \approx 13.9 \text{ GB}
$$

在全参数微调的情况下，显存占用约为 $M_{total} \approx 112 + 13.9 = 125.9 \text{ GB}$，所以，呃，几乎不可能做这个事。

## 2.现在考虑 LoRA 情景下的显存占用

这里先讲一下 LoRA 的原理，假设我们认为基模全部参数矩阵与权重矩阵它表示一种预训练语料的全部知识，那么对于特定的下游任务，绝大部分的参数都是没必要的，或者说，不影响的，那就意味着只需要调整与这个特定任务相对应的参数。严谨一点表述的话，这是矩阵的秩，矩阵的秩代表了该矩阵中线性无关的行列向量的最大数量，衡量了矩阵所包含信息的实际维度。对于 $\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}$，它的理论满秩上限是 $\min(d_{out}, d_{in})$，然后有一个内在秩的概念，预训练模型拥有极其庞大的参数空间，但当模型适配某一个特定的下游任务时，所需要更新的参数实际上存在于一个维度极低的子空间中。换言之，就算你用全参数微调暴力计算出了一个满秩的 $\Delta W$，如果你对其进行奇异值分解，你会发现绝大多数奇异值趋近于 0，只有极少数奇异值占据了绝对的主导地位。这说明 $\Delta W$ 的“内在有效信息”是低秩的。

好，具体的实现是冻结预训练权重 $W_0$，并假设权重增量 $\Delta W$ 可以通过两个低维矩阵的乘积来近似：

$$
\Delta W \approx BA
$$

其中 $B \in \mathbb{R}^{d_{out} \times r}$，$A \in \mathbb{R}^{r \times d_{in}}$。将 $BA$ 替换掉上式的 $\Delta W$，即得到 LoRA 的前向传播公式：

$$
h = W_0 x + BAx
$$

数据 $x$ 在前向传播时，会分为两个平行的计算流。一部分进入被冻结的基座模型计算 $W_0 x$，另一部分进入旁路的 LoRA 适配器计算 $BAx$，最后将两个同维度向量逐元素相加。

现在来看 LoRA 的显存占用，假设预训练模型中某一个目标线性层的权重矩阵为 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$。LoRA 将它的更新量拆解为两个低维矩阵：

$$
\Delta W = B \cdot A
$$

其中：

- 降维矩阵 $A$：维度是 $r \times d_{\text{in}}$，参数量为 $r \cdot d_{\text{in}}$
- 升维矩阵 $B$：维度是 $d_{\text{out}} \times r$，参数量为 $d_{\text{out}} \cdot r$

对于任何一个输入为 $d_{\text{in}}$、输出为 $d_{\text{out}}$ 的目标层，配置秩为 $r$ 的 LoRA 后，新增的可训练参数量恒为：

$$
\text{Params} = r \cdot (d_{\text{in}} + d_{\text{out}})
$$

在标准的多头注意力 Transformer 架构中，它们都在模型的隐藏维度 $h$ 上进行线性变换：

- q_proj (Query): 从 $h$ 维映射到 $h$ 维。$d_{\text{in}} = h, d_{\text{out}} = h$，参数量 $= r \cdot (h + h) = 2hr$
- k_proj (Key): 从 $h$ 维映射到 $h$ 维。参数量 $= 2hr$
- v_proj (Value): 从 $h$ 维映射到 $h$ 维。参数量 $= 2hr$
- o_proj (Output): 把各个头的注意力结果拼接后，从 $h$ 维再映射回 $h$ 维。参数量 $= 2hr$

把这 4 个矩阵加起来，单层 Attention 引入的 LoRA 参数就是：

$$
\text{Attention}_{\text{LoRA}} = 4 \times (2hr)
$$

Qwen 使用 SwiGLU 激活函数的门控架构。这种架构包含三个线性层：gate_proj, up_proj, down_proj。由于 MLP 需要将向量投影到一个更宽的中间维度（设为 $h_{\text{ffn}}$，通常是 $h$ 的 3 到 4 倍）再投影回来，维度变化如下：

- gate_proj 门控层: 从隐藏层 $h$ 映射到宽泛层 $h_{\text{ffn}}$。$d_{\text{in}} = h, d_{\text{out}} = h_{\text{ffn}}$，参数量 $= r \cdot (h + h_{\text{ffn}}) = h \cdot r + h_{\text{ffn}} \cdot r$
- up_proj 升维层: 同样从隐藏层 $h$ 映射到宽泛层 $h_{\text{ffn}}$。参数量 $= h \cdot r + h_{\text{ffn}} \cdot r$
- down_proj 降维层: 把激活后的宽向量从 $h_{\text{ffn}}$ 映射回隐藏层 $h$。$d_{\text{in}} = h_{\text{ffn}}, d_{\text{out}} = h$，参数量 $= r \cdot (h_{\text{ffn}} + h) = h_{\text{ffn}} \cdot r + h \cdot r$

把这 3 个矩阵加起来，单层 MLP 引入的 LoRA 参数就是：

$$
\text{MLP}_{\text{LoRA}} = 3 \times (h \cdot r + h_{\text{ffn}} \cdot r)
$$

将单层的 Attention 和 MLP 加起来，再乘以模型的总层数 $L$，就得到了你给出的最终公式：

$$
N_{\text{lora}} = L \times [ 4 \times (2hr) + 3 \times (h \cdot r + h_{\text{ffn}} \cdot r) ]
$$

以 Qwen 2.5 7B 为例进行具体的数值验证。我们查阅 Qwen 2.5 7B 的 config.json，可以得到它的具体物理架构参数：

- 层数 $L = 28$
- 隐藏维度 $h = 3584$
- MLP 中间维度 $h_{\text{ffn}} = 18928$
- 你设置的秩 $r = 64$

代入公式计算单层：

- 单层 Attention: $4 \times (2 \times 3584 \times 64) = 1,835,008$ 个参数
- 单层 MLP: $3 \times (3584 \times 64 + 18928 \times 64) = 3 \times (229,376 + 1,211,392) = 4,322,304$ 个参数
- 单层总计: $1,835,008 + 4,322,304 = 6,157,312$ 个参数

乘以总层数 28 层：

$$
N_{\text{lora}} = 28 \times 6,157,312 \approx 172.4 \text{ M} \text{ (即 1.72 亿个参数)}
$$

1 个 FP32 标量占用 4 Bytes。

$$
M_{\text{lora\_weights}} = 172.4 \times 10^6 \times 4 \text{ Bytes} \approx 689,600,000 \text{ Bytes} \approx 657 \text{ MB}
$$

仅更新 LoRA 参数。AdamW 针对每个可训练参数需维护一阶动量 $m_t$、二阶动量 $v_t$ 以及 FP32 格式的主权重备份。

$$
M_{\text{optimizer}} = N_{\text{lora}} \times (4 + 4 + 4) \text{ bytes} \approx 1.8 \text{ GB}
$$

仅需要为 LoRA 参数分配梯度显存，计算时为 BF16（2 bytes）：

$$
M_{\text{gradients}} = N_{\text{lora}} \times 2 \text{ bytes} \approx 300 \text{ MB}
$$

激活值动态显存的话要考虑 Batch Size $B$、序列长度 $S$、隐藏维度 $h$。在未开启梯度检查点的情况下，假定 $B=1, S=2048$：

$$
M_{\text{activations}} \approx B \times S \times h \times L \times 34 \text{ bytes}
$$

## 第二部分：混合精度

当代码进入 trainer.train() 的前向阶段，Hugging Face 会开启 torch.autocast(device_type='cuda', dtype=torch.bfloat16) 上下文。以其中的一个 LoRA 线性层计算为例。

设输入激活值为 $X \in \mathbb{R}^{B \times S \times h}$，数据类型已为 BF16。基座权重 $W_0 \in \mathbb{R}^{h \times h_{\text{out}}}$ 类型为 BF16。LoRA 降维矩阵 $A \in \mathbb{R}^{h \times r}$ 和升维矩阵 $B \in \mathbb{R}^{r \times h_{\text{out}}}$ 初始类型为 FP32。

在算子执行前，PyTorch 调度器会插入 Cast 算子，执行：

$$
A_{\text{bf16}} = \text{static\_cast<c10::BFloat16>}(A_{\text{fp32}})
$$

$$
B_{\text{bf16}} = \text{static\_cast<c10::BFloat16>}(B_{\text{fp32}})
$$

随后将 $X, W_0, A_{\text{bf16}}, B_{\text{bf16}}$ 送入 Tensor Cores 执行 FMA：

$$
Y = X W_0 + (X A_{\text{bf16}}) B_{\text{bf16}}
$$

损失函数 $L$ 的计算会被强制转换回 FP32，以避免交叉熵计算中的 Softmax 累加下溢。根据链式法则计算梯度：

$$
\frac{\partial L}{\partial A_{\text{bf16}}}, \quad \frac{\partial L}{\partial B_{\text{bf16}}}
$$

这些梯度在 BF16 精度下计算完成。在优化器步骤（optimizer.step()）中，这些 BF16 梯度被读取至寄存器，转换为 FP32，随后更新 AdamW 状态方程：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{\partial L}{\partial \theta_{t}}
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \left( \frac{\partial L}{\partial \theta_{t}} \right)^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
$$

上述公式中的 $\theta$ 代表 LoRA 的 FP32 主权重。更新完成后，在下一次前向传播时，再重复 Cast 为 BF16。

## 第三部分：FlashAttention 机制原理

令查询、键、值矩阵为 $Q, K, V \in \mathbb{R}^{N \times d}$。SRAM 大小限制了只能装载大小为 $B_c \times B_r$ 的块。将 $Q$ 划分为 $T_r$ 个块 $Q_i$，将 $K, V$ 划分为 $T_c$ 个块 $K_j, V_j$。在线 Softmax 必须解决分块计算时的归一化分母缺失问题。

对于外层循环遍历 $K, V$ 块（索引 $j$），内层循环遍历 $Q$ 块（索引 $i$），在 SRAM 内部，核心数学更新逻辑实现如下：

计算局部点积：

$$
S_{ij} = Q_i K_j^T
$$

计算当前的局部最大值：

$$
m_{ij} = \max(m_{i, j-1}, \max(S_{ij}))
$$

计算局部未归一化的权重（此时底层的指数运算会用到我们在性能分析中常见的逐元素算子类似逻辑，但在 FA 中被融合）：

$$
\tilde{P}_{ij} = \exp(S_{ij} - m_{ij})
$$

更新行归一化系数（在线 Softmax 核心）：

$$
l_{ij} = l_{i, j-1} \exp(m_{i, j-1} - m_{ij}) + \sum \tilde{P}_{ij}
$$

最终在寄存器内累加输出块：

$$
O_{ij} = \text{diag}(\exp(m_{i, j-1} - m_{ij})) O_{i, j-1} + \tilde{P}_{ij} V_j
$$

当内层循环结束时，才将最终的 $O_i = \text{diag}(l_{i, T_c})^{-1} O_{i, T_c}$ 写回全局显存（HBM）。
