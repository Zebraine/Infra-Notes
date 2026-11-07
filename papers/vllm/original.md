## Abstract

High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM's source code is publicly available at https://github.com/vllm-project/vllm.

## 1 Editor's Note

vLLM is the paper that made KV-cache memory management a first-class systems problem. Orca showed how to schedule iterations continuously; vLLM shows why the memory those iterations use is so badly fragmented, and how to fix it. The observation is almost embarrassingly simple: existing engines reserve a contiguous KV region for each request's maximum possible length. Since actual lengths are unknown and often much shorter, most of that reservation is wasted. Long-lived requests also block new ones, and sequences that share a prefix cannot share physical memory.

PagedAttention borrows the operating system's answer: fixed-size pages, a per-request page table, and copy-on-write sharing. The KV cache becomes a set of blocks that need not be contiguous. The attention kernel follows the block table to gather K/V, so logical contiguity is preserved for the model while physical memory can be allocated on demand and shared.

This is the design that vLLM turned into a widely used open-source serving engine. It is also the direct predecessor of prefix caching, memory-aware scheduling, and much of the later LLM serving literature, including BidKV's KV-pressure framing.

## 2 Problem and Background

LLM serving is memory-bound. For a 13B model on a 40 GB A100, roughly 65% of memory holds static model weights, close to 30% holds the dynamic KV cache, and only a small fraction is used for activations. The KV cache grows and shrinks as tokens are generated, and its lifetime and final length are not known in advance. Existing systems store each request's KV cache in contiguous memory, because most deep learning frameworks require contiguous tensors.

That design causes three kinds of waste. *Reserved* memory is allocated for the maximum sequence length but never used. *Internal fragmentation* occurs because the actual length is shorter than the reservation. *External fragmentation* occurs when differently sized contiguous chunks leave unusable gaps. In the paper's profiling, only 20.4%–38.2% of KV-cache memory is actually used in existing systems. The waste limits batch size, and batch size limits throughput.

## 3 Method

### 3.1 PagedAttention

PagedAttention divides each request's KV cache into fixed-size blocks, each holding the keys and values of $B$ tokens. A block table maps the request's logical blocks to physical blocks in GPU memory. Blocks need not be contiguous. Given query $q_i$ and block $j$ containing keys $K_j$ and values $V_j$, the block-level attention is

$$
A_{ij}=\frac{\exp(q_i^{\top}K_j/\sqrt{d})}{\sum_{j'}\exp(q_i^{\top}K_{j'}/\sqrt{d})},
\qquad
o_i=\sum_j V_j A_{ij}^{\top}.
$$

The kernel reads the block table, gathers the K/V blocks, and computes attention over them. Because blocks are fixed size, internal fragmentation is bounded by the last block, and external fragmentation disappears. Memory is allocated on demand as generation proceeds, so the system approaches near-zero waste.

### 3.2 KV-Cache Sharing and Scheduling

The block table also enables sharing. For parallel sampling or beam search, multiple output sequences can point to the same physical blocks for their shared prefix; when a sequence needs to write a new token into a shared block, vLLM uses copy-on-write to create a private copy. Prefix sharing across requests works the same way. The paper reports up to 55% memory savings for beam search and substantial savings for shared prompts.

vLLM's scheduler batches requests FCFS and preempts when physical blocks run out. It can swap blocks to CPU or recompute them, and it uses tensor parallelism to serve models that do not fit on one GPU. The block manager, scheduler, and attention kernel are co-designed: the scheduler allocates blocks, the block table is broadcast to workers, and the kernel uses it to access the right physical memory.

## 4 Experiments and Results

The paper evaluates vLLM on OPT-13B, OPT-66B, and OPT-175B with ShareGPT and Alpaca traces. Compared with FasterTransformer and Orca, vLLM improves throughput by 2–4× at the same latency. The gain grows with longer sequences, larger models, and more complex decoding algorithms. vLLM's KV-cache memory waste is under 4% in the main experiments, versus 60%–80% in the baselines.

The paper also shows that PagedAttention preserves model accuracy—it changes memory layout, not the attention math—and that sharing and copy-on-write deliver large savings for parallel sampling and beam search. The system supports models larger than a single GPU through tensor parallelism.

## 5 Impact and Limitations

PagedAttention and vLLM changed the default memory model of LLM serving. Block-based KV management, prefix caching, and copy-on-write sharing are now standard ideas, and vLLM itself became one of the most widely used open-source serving engines. The paper also made the KV cache the central object of serving research: once memory can be paged, one can reason about eviction, sharing, preemption, and scheduling in a much richer design space.

The limitations are mostly about scope. The paper focuses on throughput at a given latency and does not deeply optimize SLOs, fairness, or priority. The block size is a trade-off between fragmentation and kernel efficiency. Prefix caching across requests introduces cache invalidation and security considerations. And PagedAttention still performs full attention over all cached tokens, so it does not reduce the quadratic attention compute for very long contexts—later work such as FlashAttention and MLA attacks that axis.
