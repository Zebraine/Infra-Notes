## Abstract

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4× speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

## 1 Editor's Note

FlashAttention is the paper that made IO—not just FLOPs—the organizing principle of efficient attention. Standard attention materializes the $N \times N$ score matrix, writes it to GPU HBM, reads it back for softmax, writes the probabilities, and reads them again for the value product. On modern GPUs, those HBM round trips cost more than the arithmetic. Approximate attention methods reduced FLOPs but often did not translate into wall-clock speed because they ignored this memory traffic.

FlashAttention computes exact attention without ever materializing the full matrix. It tiles Q, K, and V into blocks that fit in on-chip SRAM, performs an online softmax over the blocks, and fuses the entire attention operation into one kernel. For the backward pass it saves only the softmax normalization statistics and recomputes the attention blocks on chip, trading a little extra compute for far fewer HBM reads.

This is one of the most influential kernels in modern AI infrastructure. FlashAttention-2 and FlashAttention-3 refine its work partitioning and use newer hardware features, but the core ideas—tiling, online softmax, recomputation, and IO-awareness—originate here. It is also the natural companion to PagedAttention: one manages KV-cache memory, the other manages attention's memory traffic.

## 2 Problem and Background

Transformer models have become the dominant architecture, but their self-attention module has time and memory complexity quadratic in sequence length. Many approximate attention methods—sparse, low-rank, and combinations—reduce the compute complexity to linear or near-linear, yet many do not show wall-clock speedup against standard attention and have not gained wide adoption. The reason is that they optimize FLOPs while ignoring memory access overhead.

On modern GPUs, compute speed has outpaced memory speed, and most Transformer operations are memory-bound. IO-aware algorithms are standard in databases, image processing, and numerical linear algebra, but common deep learning interfaces such as PyTorch and TensorFlow do not expose fine-grained control over memory access. The paper argues that attention needs an IO-aware algorithm that accounts for reads and writes between HBM (large, slow) and SRAM (small, fast).

## 3 Method

### 3.1 Tiling and Online Softmax

Standard attention computes $S=QK^{\top}$, writes $S$ to HBM, reads it for softmax, writes $P=\mathrm{softmax}(S)$, and reads $P$ for $PV$. FlashAttention avoids this by processing Q, K, and V in blocks. It loads a block of Q and a block of K into SRAM, computes the block scores, and updates a running softmax maximum and normalization factor. This is the online softmax trick: the final softmax can be computed without seeing all scores at once.

The tiling loop makes the memory traffic scale with the number of blocks rather than with the full $N \times N$ matrix. The output is accumulated in SRAM and written to HBM once. The entire attention computation—matmul, masking, softmax, dropout, and the final value product—is fused into a single CUDA kernel, so intermediate tensors never round-trip through HBM.

### 3.2 Recomputation and IO Complexity

For the backward pass, standard attention stores the $N \times N$ attention matrix and reads it again. FlashAttention instead stores the output and the softmax normalization statistics $(m,\ell)$ from the forward pass, and recomputes the attention blocks on chip during backward. This increases FLOPs slightly but sharply reduces HBM accesses, which are the real bottleneck.

The paper proves that standard attention requires $\Theta(Nd+N^2)$ HBM accesses, while FlashAttention requires $\Theta(N^2d^2M^{-1})$ for head dimension $d$ and SRAM size $M$, and shows this is optimal over a range of SRAM sizes. For typical values, FlashAttention performs many fewer HBM accesses—up to 9× fewer in the paper's measurements—and the runtime tracks the HBM access count rather than the FLOP count.

## 4 Experiments and Results

FlashAttention trains Transformers faster in wall-clock time: 15% end-to-end speedup on BERT-large (sequence length 512) over the MLPerf 1.1 training speed record, up to 3× speedup on GPT-2 (sequence length 1K) over HuggingFace and Megatron-LM baselines, and 2.4× speedup on Long Range Arena (sequence lengths 1K–4K). The attention kernel itself is up to 3× faster and uses memory linear in sequence length; the paper reports up to 7.6× speedup on the attention computation for GPT-2.

Longer context also improves model quality. FlashAttention gives 0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification. It enables the first Transformer to beat chance on Path-X (sequence length 16K, 61.4% accuracy) and, with block-sparse attention, Path-256 (sequence length 64K, 63.1% accuracy).

## 5 Impact and Limitations

FlashAttention changed how efficient attention is designed: the metric is not only FLOPs but the number and pattern of memory accesses. Tiling, online softmax, and recomputation are now standard in attention kernels, and the work directly enabled longer-context training and serving. Its descendants—FlashAttention-2, FlashAttention-3, and paged/block-sparse variants—are widely used in training and inference stacks.

The limitations are important. FlashAttention is exact attention, so it does not change the $O(N^2)$ arithmetic; it reduces memory traffic and memory footprint, not the asymptotic compute. Performance depends on SRAM size, block size, and hardware generation, and the backward pass trades extra compute for lower memory traffic. It also does not solve KV-cache capacity or scheduling—those remain systems-level problems addressed by PagedAttention, vLLM, and later serving work.
