## Abstract

Large deep learning models offer significant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation, communication and development efficiency. We develop a novel solution, Zero Redundancy Optimizer (ZeRO), to optimize memory, vastly improving training speed while increasing the model size that can be efficiently trained. ZeRO eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency. Our analysis on memory requirements and communication volume demonstrates: ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware. We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of ZeRO to create the world's largest language model (17B parameters) with record breaking accuracy.

## 1 Editor's Note

ZeRO is the paper that made "data parallelism can train huge models" believable. Standard data parallelism replicates the full model state on every worker: parameters, gradients, and optimizer states. For mixed-precision Adam, that is roughly 16 bytes per parameter, so a 1.4B-parameter model already exhausts a 32 GB GPU even before activations. Model parallelism avoids replication but introduces fine-grained communication and framework complexity.

ZeRO's observation is simple and powerful: data-parallel workers do not need to *keep* the same optimizer state, gradients, and parameters at the same time. Partition them across the data-parallel group and gather them on demand. The three stages—optimizer-state partitioning, gradient partitioning, and parameter partitioning—reduce per-device model-state memory by 4x, 8x, and $N_d$ respectively, while preserving the computation and communication granularity of data parallelism.

This paper is the direct conceptual ancestor of DeepSpeed ZeRO and PyTorch FSDP. It also framed the modern memory hierarchy of large-model training: model states are only part of the problem; activations, temporary buffers, and fragmentation form the "residual" memory that must be managed separately.

## 2 Problem and Background

Deep learning models are becoming larger, and the increase in model size offers significant accuracy gains. In NLP, Transformers enabled BERT-large (0.3B), GPT-2 (1.5B), Megatron-LM (8.3B), and T5 (11B). Continuing the growth from tens of billions to trillions of parameters runs into a hard constraint: the model does not fit in the memory of a single device, and simply adding more devices does not help if the parallelization strategy replicates state.

Basic data parallelism (DP) does not reduce memory per device and runs out of memory for models beyond about 1.4B parameters on 32 GB GPUs. Pipeline parallelism, model parallelism, and CPU offloading make different trade-offs among functionality, usability, memory, and compute/communication efficiency. The paper analyzes memory consumption in two parts: model states (optimizer states, gradients, parameters) and residual states (activations, temporary buffers, fragmented memory).

## 3 Method

### 3.1 ZeRO-DP: Partitioning Model States

For a model with $\Psi$ parameters trained with mixed-precision Adam, the per-device memory for model states under standard data parallelism is approximately $16\Psi$ bytes: 2 bytes for fp16 parameters, 2 bytes for fp16 gradients, and 12 bytes for fp32 optimizer states (momentum, variance, and a master copy). ZeRO-DP partitions these states across $N_d$ data-parallel workers.

The three stages are cumulative. $P_{os}$ partitions optimizer states, reducing model-state memory to about $4\Psi + 12\Psi/N_d$ and keeping communication volume equal to DP. $P_{os+g}$ also partitions gradients, reducing memory to about $2\Psi + 14\Psi/N_d$, again with DP-level communication. $P_{os+g+p}$ partitions parameters as well, so each device stores roughly $16\Psi/N_d$ bytes; memory reduction becomes linear in $N_d$, at the cost of a modest 50% increase in communication volume because parameters must be gathered before each layer's computation.

The key is that ZeRO preserves the computational granularity of data parallelism: each worker still performs large GEMMs on local batches. It changes *what is stored where*, not the shape of the computation. With all three stages, a trillion-parameter model with Adam's 16 TB of model state can fit on 1024 GPUs with 16 GB per device.

### 3.2 ZeRO-R: Managing Residual Memory

After model states are partitioned, activations, temporary buffers, and memory fragmentation can become the next bottleneck. ZeRO-R addresses each. For activations, it partitions activation checkpoints across model-parallel devices and can offload activations to CPU when appropriate. For temporary buffers, it chooses buffer sizes that balance memory and computation efficiency rather than allocating worst-case buffers everywhere.

For fragmentation, ZeRO-R observes that tensors have different lifetimes and that a lack of contiguous memory can cause allocation failure even when enough free memory exists. It proactively manages memory based on tensor lifetime to avoid fragmentation. Together, ZeRO-DP and ZeRO-R form the full ZeRO system.

## 4 Experiments and Results

ZeRO trains models of over 100B parameters on 400 GPUs with super-linear speedup, sustaining about 15 Petaflops. Relative to the state of the art, this is an 8x increase in model size and a 10x increase in achievable performance. With model parallelism, ZeRO can train 170B-parameter models; without any model parallelism, it can train models up to 13B parameters on 128 GPUs, which is larger than Megatron GPT 8.3B and T5 11B.

The paper reports strong scaling and throughput results for 8B to 100B models and shows that ZeRO-100B can outperform the baseline by up to 10x while using the same hardware. It also analyzes the communication volume of each stage and shows that the extra communication is bounded: stage 1 and 2 match DP, and stage 3 costs about 1.5x DP communication.

## 5 Impact and Limitations

ZeRO changed the default assumption in large-model training: one does not need model parallelism to train a model that does not fit on one device. By eliminating redundant model states, it made data parallelism scale to tens and hundreds of billions of parameters and directly enabled DeepSpeed and PyTorch FSDP. Its memory taxonomy—model states versus residual states—is still the right way to reason about training memory.

The limitations are equally instructive. Stage 3 introduces extra parameter gather/scatter communication, which can hurt at small batch sizes or slow interconnects. Activation memory and fragmentation still require careful management, and CPU offloading trades PCIe bandwidth for capacity. ZeRO is also complementary to, not a replacement for, tensor and pipeline parallelism; the largest training runs combine all of them.
