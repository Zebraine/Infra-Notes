## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## 1 Editor's Note

The Transformer is the architectural root of nearly every modern language model. Before 2017, the strongest sequence models were recurrent networks: computation advanced one token at a time, so training could not fully exploit GPU parallelism. The Transformer replaces recurrence with attention, turning a sequence problem into a stack of dense matrix multiplications. That single change made it practical to train on much more data and much larger batches, and it is the reason the later scaling stories of BERT, GPT, T5, LLaMA, and DeepSeek are possible.

For infrastructure, the paper is foundational in three ways. First, self-attention is a GEMM-heavy workload whose arithmetic intensity and memory traffic determine serving cost. Second, the model has no hidden state carried across tokens, so training and inference can be parallelized across the sequence dimension. Third, its $O(N^2)$ attention matrix creates the memory pressure that later work—KV caching, PagedAttention, FlashAttention, MLA, and preemption scheduling—tries to manage.

This reading edition keeps the paper's original abstract and core technical passages, then adds a short guide to the results and legacy. The complete experimental appendix and all ablations remain in the PDF.

## 2 Problem and Background

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$ as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models, allowing modeling of dependencies without regard to their distance in the input or output sequences. In almost all previous work, however, attention was used in conjunction with a recurrent network. The Transformer is the first transduction model that relies entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

## 3 Method

### 3.1 Scaled Dot-Product and Multi-Head Attention

An attention function maps a query and a set of key-value pairs to an output. The output is a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function between the query and the corresponding key. The paper uses scaled dot-product attention:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V .
$$

The dot products grow large in magnitude for large $d_k$, pushing the softmax into regions with extremely small gradients; dividing by $\sqrt{d_k}$ counteracts this. Instead of performing a single attention function, the model projects the queries, keys, and values $h$ times with learned linear projections, applies attention in parallel, concatenates the results, and projects again:

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O,
$$

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V).
$$

Multi-head attention lets the model jointly attend to information from different representation subspaces at different positions. The paper also compares additive attention and dot-product attention, and reports that dot-product attention is much faster and more space-efficient in practice because it can be implemented with highly optimized matrix multiplication.

### 3.2 Encoder-Decoder Stack and Positional Encoding

The encoder is a stack of $N=6$ identical layers. Each layer has two sub-layers: multi-head self-attention and a position-wise fully connected feed-forward network. A residual connection is employed around each sub-layer, followed by layer normalization. The decoder is also a stack of six identical layers; in addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer that performs multi-head attention over the encoder output. Masking in the decoder self-attention prevents positions from attending to subsequent positions, preserving the auto-regressive property.

Because the model contains no recurrence and no convolution, positional information must be injected explicitly. The paper adds sinusoidal positional encodings to the input embeddings:

$$
PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right),\qquad
PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right).
$$

The wavelengths form a geometric progression, which allows the model to attend by relative position and to extrapolate to sequence lengths longer than those seen during training. The base model uses $d_{\mathrm{model}}=512$, $h=8$, $d_k=d_v=64$, and a feed-forward inner dimension of 2048; the big model uses $d_{\mathrm{model}}=1024$, $h=16$, and a feed-forward dimension of 4096.

## 4 Experiments and Results

The model is trained on the WMT 2014 English-German and English-French translation tasks. The big Transformer reaches 28.4 BLEU on English-German, improving over the previous best results—including ensembles—by more than 2 BLEU. On English-French it reaches 41.8 BLEU as a single model after 3.5 days of training on eight P100 GPUs, a small fraction of the training cost of the best models in the literature. The base model can be trained in about 12 hours on eight P100 GPUs and still reaches 27.3 BLEU on English-German and 38.1 BLEU on English-French.

The paper reports that the Transformer generalizes beyond translation: it achieves strong results on English constituency parsing with both the Wall Street Journal and semi-supervised settings. Ablations show that multi-head attention, the sinusoidal positional encoding, and model size all matter; the paper also studies attention head specialization and the effect of the number of layers and attention heads.

## 5 Impact and Limitations

The Transformer replaced the sequential inductive bias of recurrent networks with a parallel, attention-based computation graph. That shift made it possible to train on far more data and to scale model size, and it directly enabled the BERT/GPT/T5 generation of models. In serving systems, the same attention operator is the source of the KV cache and the quadratic prefill cost that dominate LLM inference.

The paper's limitations are equally important for later infrastructure work. Self-attention costs $O(N^2)$ time and memory in sequence length, and the experiments focus on translation and parsing rather than open-ended generation. There is no KV cache in the paper, and the model is trained and evaluated at relatively short sequence lengths. These gaps motivated efficient attention kernels, paged KV memory, latent attention, and long-context training—the subjects of many papers in this library.
