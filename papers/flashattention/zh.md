## 摘要

Transformer 在长序列上又慢又耗显存，因为自注意力的时间和显存复杂度随序列长度平方增长。近似注意力方法试图以牺牲模型质量为代价降低计算复杂度，但往往没有实际墙钟加速。我们认为缺失的原则是让注意力算法具备 IO 感知：显式考虑 GPU 不同层级显存之间的读写。我们提出 FlashAttention，一种 IO 感知的精确注意力算法，用分块（tiling）减少 GPU 高带宽显存（HBM）与片上 SRAM 之间的读写次数。我们分析了 FlashAttention 的 IO 复杂度，证明它比标准注意力需要更少的 HBM 访问，并在一定 SRAM 容量范围内达到最优。我们还将 FlashAttention 扩展到块稀疏注意力，得到比现有近似方法更快的近似注意力算法。FlashAttention 训练 Transformer 更快：在 BERT-large（序列长度 512）上相比 MLPerf 1.1 训练速度纪录取得 15% 端到端加速，GPT-2（1K）上 3 倍，Long Range Arena（1K–4K）上 2.4 倍。FlashAttention 与块稀疏 FlashAttention 让 Transformer 支持更长上下文，带来更高质量（GPT-2 困惑度改善 0.7，长文档分类提升 6.4 分）以及全新能力：首次在 Path-X（16K，61.4% 准确率）和 Path-256（64K，63.1% 准确率）上取得优于随机的性能。

## 1 精读导读

FlashAttention 是把 IO——而不只是 FLOPs——变成高效注意力组织原则的论文。标准注意力会物化 $N \times N$ 的 score 矩阵：写入 HBM、读回做 softmax、写入概率、再读回与 V 相乘。在现代 GPU 上，这些 HBM 往返比算术更贵。近似注意力方法降低了 FLOPs，却因为忽视这些显存流量而没有转化为墙钟加速。

FlashAttention 在不物化完整矩阵的前提下计算精确注意力。它把 Q、K、V 切成能放进片上 SRAM 的块，在块上做在线 softmax，并把整个注意力操作融合进一个 kernel。反向传播只保存 softmax 归一化统计量，在片上重算注意力块，用少量额外计算换取大幅减少的 HBM 读取。

这是现代 AI 基础设施中最有影响力的 kernel 之一。FlashAttention-2/3 改进了工作划分并利用新硬件特性，但分块、在线 softmax、重算和 IO 感知这些核心思想都来自这里。它也与 PagedAttention 天然互补：一个管理 KV cache 的显存，一个管理注意力的显存流量。

## 2 问题与背景

Transformer 已成为主流架构，但自注意力模块的时间和显存复杂度随序列长度平方增长。稀疏、低秩及其组合等近似注意力方法把计算复杂度降到线性或近线性，但很多方法相对标准注意力并没有墙钟加速，也没有被广泛采用。原因是它们优化 FLOPs，却忽略显存访问开销。

现代 GPU 的算力增速快于显存速度，Transformer 中多数算子受显存带宽限制。IO 感知算法在数据库、图像处理、数值线性代数中已是常识，但 PyTorch、TensorFlow 等深度学习接口并不暴露细粒度显存访问控制。论文主张：注意力需要一种 IO 感知算法，显式考虑 HBM（大而慢）与 SRAM（小而快）之间的读写。

## 3 方法

### 3.1 分块与在线 softmax

标准注意力计算 $S=QK^{\top}$，把 $S$ 写入 HBM，读回做 softmax，写入 $P=\mathrm{softmax}(S)$，再读回计算 $PV$。FlashAttention 通过分块避免这些往返：它把一块 Q 和一块 K 载入 SRAM，计算块内 score，并维护运行中的 softmax 最大值与归一化因子。这就是在线 softmax 技巧——不必一次看到所有 score，也能得到最终 softmax。

分块循环让显存流量随块数而不是完整 $N \times N$ 矩阵增长；输出在 SRAM 中累加，最后只写回一次 HBM。整个注意力计算——矩阵乘、mask、softmax、dropout 和最后的 value 乘积——被融合进一个 CUDA kernel，中间张量不再往返 HBM。

### 3.2 重算与 IO 复杂度

反向传播时，标准注意力会保存 $N \times N$ 注意力矩阵并再次读取。FlashAttention 只保存前向的输出和 softmax 归一化统计量 $(m,\ell)$，在反向时于片上重算注意力块。这会略微增加 FLOPs，却大幅减少 HBM 访问——后者才是真正的瓶颈。

论文证明：标准注意力需要 $\Theta(Nd+N^2)$ 次 HBM 访问，而 FlashAttention 在头维度 $d$、SRAM 大小 $M$ 下需要 $\Theta(N^2d^2M^{-1})$ 次，并在一定 SRAM 范围内达到最优。对典型参数，FlashAttention 的 HBM 访问少很多——论文测量中最多少 9 倍——运行时跟随 HBM 访问次数，而不是 FLOPs。

## 4 实验与结果

FlashAttention 在墙钟时间上更快：BERT-large（序列长度 512）相比 MLPerf 1.1 训练速度纪录取得 15% 端到端加速；GPT-2（1K）相比 HuggingFace 与 Megatron-LM 基线最高 3 倍；Long Range Arena（1K–4K）上 2.4 倍。注意力 kernel 本身最高快 3 倍，显存随序列长度线性增长；论文报告 GPT-2 注意力计算最高 7.6 倍加速。

更长上下文也提升模型质量。FlashAttention 在 GPT-2 上改善 0.7 困惑度，在长文档分类上提升 6.4 分。它让 Transformer 首次在 Path-X（16K，61.4% 准确率）上超过随机；配合块稀疏注意力，在 Path-256（64K，63.1% 准确率）上也取得成功。

## 5 影响与局限

FlashAttention 改变了高效注意力的设计标准：不只看 FLOPs，还要看显存访问的次数与模式。分块、在线 softmax、重算如今已是注意力 kernel 的标配，这项工作也直接推动了长上下文训练与服务。FlashAttention-2、FlashAttention-3 以及分页/块稀疏变体已广泛用于训练和推理栈。

局限同样重要。FlashAttention 是精确注意力，不改变 $O(N^2)$ 的算术复杂度；它降低的是显存流量和显存占用，不是渐近计算量。性能依赖 SRAM 大小、块大小和硬件代际；反向传播用额外计算换更低的显存流量。它也不解决 KV cache 容量与调度问题——那仍是 PagedAttention、vLLM 以及后续服务系统要处理的系统级问题。
