## Abstract

Recent work in language modeling demonstrates that training large transformer models advances the state of the art in Natural Language Processing (NLP) applications. However, very large models can be quite difficult to train due to memory constraints. In this work, we present our techniques for training very large transformer models and implement a simple, efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of 89.4%).

## 1 Editor's Note

Megatron-LM is the paper that made tensor parallelism a standard systems primitive. Before it, training a model that did not fit on one GPU usually meant model parallelism through a custom compiler or framework, or pipeline parallelism with bubbles and scheduling complexity. Megatron showed that a Transformer layer has just enough structure to split its GEMMs across GPUs with only a handful of all-reduce operations—no new compiler, no custom C++ kernels.

The key idea is to split the *inside* of a layer rather than the whole layer. In the MLP, the first weight matrix is column-parallel and the second is row-parallel, so GeLU can be applied locally and only one all-reduce is needed at the end. In self-attention, Q/K/V projections are split by attention heads and the output projection is row-parallel. The result is a small, composable block that can be combined with data parallelism and pipeline parallelism.

This paper is the direct ancestor of the tensor-parallel path in Megatron-LM, DeepSpeed, NeMo, and many later training stacks. It also made an important empirical point for architecture: when scaling BERT-style models, the *placement* of layer normalization matters as much as the parameter count.

## 2 Problem and Background

Natural language processing is advancing in part because of more compute and more data, which enables larger unsupervised language models. Empirical evidence shows that larger models are dramatically more useful for article completion, question answering, and natural language inference. As these models become larger, however, they exceed the memory of modern accelerators, and standard optimizers such as Adam need additional per-parameter state for momentum and variance.

Existing approaches to model parallelism, such as GPipe and Mesh-TensorFlow, partition the model and its optimizer state, but they require rewriting the model and rely on custom compilers and frameworks. Pipeline parallelism is orthogonal and useful, but it does not by itself solve the per-layer memory problem. The paper asks whether a Transformer can be trained at billions of parameters with a simple, native-PyTorch intra-layer split.

## 3 Method

### 3.1 Column- and Row-Parallel MLP

A Transformer layer contains a self-attention block followed by a two-layer MLP. The MLP computes $Y=\mathrm{GeLU}(XA)$. If we split the weight matrix $A$ along its rows and the input $X$ along its columns, then $Y=\mathrm{GeLU}(X_1A_1+X_2A_2)$. Because GeLU is nonlinear, this requires a synchronization point before the nonlinearity, which is undesirable.

Instead, Megatron splits $A$ along its columns: $A=[A_1,A_2]$. Then

$$
[Y_1,Y_2]=[\mathrm{GeLU}(XA_1),\mathrm{GeLU}(XA_2)] .
$$

The GeLU can be applied independently on each GPU. The second GEMM is split along its rows, so it consumes $[Y_1,Y_2]$ directly without communication; only the final output needs an all-reduce before dropout. This "fuse two GEMMs, communicate once" pattern is the core of the method. The forward pass uses one all-reduce (the $g$ operator); the backward pass uses one all-reduce (the $f$ operator); the two are conjugates.

### 3.2 Head-Parallel Attention and Communication Schedule

For self-attention, the Q/K/V GEMMs are split in a column-parallel fashion so that each attention head is computed locally on one GPU. The subsequent output linear layer is row-parallel and consumes the local attention output directly. This removes an immediate communication point inside the attention block and lets the per-head parameters and workload be divided across GPUs.

With this design, all GEMMs in a Transformer layer require only two all-reduces in the forward path and two in the backward path. The layer-norm parameters are duplicated on each GPU, and dropout and residual connections are computed locally, avoiding extra broadcasts. The output embedding is also parallelized along the vocabulary dimension; the paper fuses the parallel GEMM with the cross-entropy loss so that only scalar losses, not the full logits, are communicated. This reduces communication from $b \times s \times v$ to $b \times s$.

## 4 Experiments and Results

The paper scales GPT-2-style models from 1.2B to 8.3B parameters, using up to 8-way model parallelism and 512 V100 GPUs. The 8.3B model sustains 15.1 PetaFLOPs across the application, with 76% scaling efficiency relative to a strong single-GPU baseline that sustains 39 TeraFLOPs (30% of peak). The models are trained on a deduplicated 174 GB corpus with mixed precision, activation checkpointing, and a carefully tuned initialization and learning-rate schedule.

The 8.3B GPT-2 model reaches 10.81 perplexity on WikiText103 and 66.51% zero-shot accuracy on LAMBADA, both state of the art at the time. The BERT-style models scale to 3.9B parameters and reach 90.9% on RACE. A key systems/architecture finding is that simply enlarging BERT with the original layer-norm placement degrades; moving layer normalization to the input of each sub-layer (pre-LN order) stabilizes training and lets BERT-style models improve monotonically with size.

## 5 Impact and Limitations

Megatron-LM established tensor parallelism as a practical, reusable building block. Its column-parallel/row-parallel decomposition, conjugate all-reduce operators, and vocabulary-parallel output layer are now standard in large-model training frameworks. The paper also demonstrated that systems and architecture co-design are inseparable at scale: the layer-norm change was necessary to make the larger BERT models trainable and accurate.

The main limitation is scope: 8-way tensor parallelism is efficient within a node with high-bandwidth NVLink, but it does not by itself scale across many nodes. The paper explicitly positions its method as orthogonal to pipeline parallelism, and later systems combine tensor, pipeline, data, and sequence parallelism. The 8.3B model is also small by today's standards; the enduring contribution is the decomposition and the communication schedule, not the specific model size.
