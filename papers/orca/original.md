## Abstract

Large-scale Transformer-based models trained for generation tasks (e.g., GPT-3) have recently attracted huge interest, emphasizing the need for system support for serving models in this family. Since these models generate a next token in an autoregressive manner, one has to run the model multiple times to process an inference request where each iteration of the model generates a single output token for the request. However, existing systems for inference serving do not perform well on this type of workload that has a multi-iteration characteristic, due to their inflexible scheduling mechanism that cannot change the current batch of requests being processed; requests that have finished earlier than other requests in a batch cannot return to the client, while newly arrived requests have to wait until the current batch completely finishes. In this paper, we propose iteration-level scheduling, a new scheduling mechanism that schedules execution at the granularity of iteration (instead of request) where the scheduler invokes the execution engine to run only a single iteration of the model on the batch. In addition, to apply batching and iteration-level scheduling to a Transformer model at the same time, we suggest selective batching, which applies batching only to a selected set of operations. Based on these two techniques, we have implemented a distributed serving system called Orca, with additional designs for scalability to models with hundreds of billions of parameters. Our evaluation on a GPT-3 175B model shows that Orca can significantly outperform NVIDIA FasterTransformer in terms of both latency and throughput: 36.9× throughput improvement at the same level of latency.

## 1 Editor's Note

Orca is the paper that introduced continuous batching to LLM serving. Before it, a serving system scheduled at request granularity: it formed a batch, ran every request to completion, and only then admitted new work. For image models that is fine—one forward pass finishes a request. For autoregressive LLMs, a request may need hundreds or thousands of iterations, so a batch is a long-lived resource. Short requests are trapped behind long ones, and newly arrived requests wait for the entire batch to drain.

Orca's fix is to schedule at the granularity of a single model iteration. After every iteration the scheduler can retire finished requests, admit new ones, and change the batch composition. The second half of the paper solves the systems problem that makes naive iteration-level batching hard: attention has request-dependent tensor shapes, so it cannot simply be batched across requests at different generation lengths. Selective batching batches the operations that can be batched and leaves attention per-request.

Nearly every modern LLM serving engine—vLLM, TensorRT-LLM, TGI, SGLang—implements some form of continuous batching and iteration-level scheduling. Orca is the conceptual baseline for the serving papers in this library.

## 2 Problem and Background

Transformer-based generative models produce one token per iteration and feed that token back as input. A request therefore requires many iterations. Existing inference serving systems schedule execution at request granularity: when the serving system dispatches a batch to the engine, the engine returns results for the entire batch after processing all requests. Because requests need different numbers of iterations, a request that finishes early cannot return to the client until the slowest request in the batch finishes. Requests that arrive after the batch is dispatched must wait for the whole batch, which adds queueing delay.

The paper argues that this is a fundamental mismatch between the multi-iteration nature of autoregressive generation and request-level scheduling. The fix must let the batch change between iterations, while still preserving batching efficiency for the operations that dominate GPU time.

## 3 Method

### 3.1 Iteration-Level Scheduling

Orca's scheduler invokes the execution engine to run exactly one iteration of the model on the current batch. After each return, the scheduler checks which requests have finished and returns their outputs immediately, then considers newly arrived requests, and launches the next iteration with a possibly different batch. This is iteration-level scheduling, as opposed to request-level scheduling.

The scheduler maintains an iteration-level first-come-first-served policy and uses a configurable maximum batch size; if the batch is not full, it admits requests from the waiting pool in arrival order. It also reserves memory for the KV cache of admitted requests and manages pipelined execution across workers so that a request's next iteration can start as soon as its previous iteration finishes. The result is that a request can join after waiting for only one iteration, and a finished request can leave immediately.

### 3.2 Selective Batching

Iteration-level scheduling creates batches in which requests have processed different numbers of tokens. The attention operation consumes tensors whose shapes depend on each request's current sequence length, so it cannot be batched as a single dense tensor. Orca therefore applies selective batching: it splits the batch and runs attention per request, while batching the other Transformer operations—QKV projections, MLP, layer norm, and so on—across requests.

This is efficient because attention has no model parameters to reuse across requests, so batching it does not reduce parameter reads; the expensive weight-heavy GEMMs remain batched. The paper shows that the per-request attention execution has only a small impact on end-to-end efficiency. Orca also uses intra-layer and inter-layer model parallelism to scale to models with hundreds of billions of parameters.

## 4 Experiments and Results

The evaluation uses GPT-3 models with 13B, 175B, and 341B parameters. On GPT-3 175B, Orca significantly outperforms NVIDIA FasterTransformer in both latency and throughput. At a median normalized latency of 190 ms per token, FasterTransformer provides 0.185 requests/s while Orca provides 6.81 requests/s—a 36.9x throughput improvement at the same latency level. The paper reports similar order-of-magnitude gains across load levels and model sizes.

The results show that iteration-level scheduling and selective batching together are far more important than simply increasing batch size under request-level scheduling. The system also scales to a 341B-parameter model with model and pipeline parallelism.

## 5 Impact and Limitations

Orca established the scheduling model that the rest of the LLM serving stack assumes. Continuous batching, iteration-level scheduling, and selective batching are now baseline features rather than research ideas. The paper also made clear that serving throughput is a scheduling and memory problem, not only a kernel problem.

Its limitations are partly historical. Orca is a research system and was not released as a general open-source engine. It uses a relatively simple FCFS admission policy and does not deeply address KV-cache fragmentation, prefix sharing, or preemption; those are the problems that PagedAttention and vLLM attack next. The paper also focuses on throughput-latency trade-offs rather than fairness or SLO-aware scheduling, leaving room for later work such as BidKV.
