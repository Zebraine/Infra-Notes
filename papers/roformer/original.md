## Abstract

Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding (RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: <https://huggingface.co/docs/transformers/model_doc/roformer>.

## 1 Introduction

The sequential order of words is of great value to natural language understanding. Recurrent neural networks encode token order by recursively computing a hidden state along the time dimension. Convolutional models are typically regarded as position-agnostic, yet recent work shows that the commonly used padding operation can implicitly learn position information. Pre-trained language models (PLMs) built on the Transformer have achieved state-of-the-art results on context representation learning, machine translation, and language modeling. Unlike RNNs and CNNs, PLMs use self-attention to capture contextual representations; this gives them a significant advantage in parallelization and lets them model longer-range relations than a single convolutional layer.

The self-attention architecture of current PLMs is nevertheless position-agnostic: attention is permutation-equivariant, so without an explicit position signal the model cannot distinguish word orders. A large body of work therefore injects position information into learning. On one side, absolute position encoding either adds a pre-defined function of position to the contextual representation (sinusoidal encoding) or learns a position vector per index (BERT, ALBERT, ELECTRA, GPT). On the other side, relative position encoding typically enters the attention mechanism itself (Shaw et al., Transformer-XL, XLNet, T5, DeBERTa). Other proposals model positional dependency through Neural ODEs or in complex space. Despite their differences, these methods commonly add position information to the contextual representation, which makes them unsuitable for linear self-attention. Linear attention reduces the quadratic attention cost to $O(N)$ through a kernel factorization, which matters for long texts; the incompatibility of existing position encodings with it is a direct motivation for this work.

We introduce Rotary Position Embedding (RoPE). RoPE encodes the absolute position with a rotation matrix while incorporating an explicit relative position dependency in the self-attention formulation. Compared with existing methods, it offers sequence-length flexibility, decaying inter-token dependency with increasing relative distance, and compatibility with linear self-attention. Experiments on long-text benchmarks show that the enhanced Transformer, RoFormer, outperforms its alternatives.

Our contributions are three-fold:

1. **A new relative-position formulation.** Existing relative position encodings are mostly built on an additive decomposition of position added to the contextual representation. RoPE instead multiplies the representation by a rotation matrix, encoding relative position with a clear theoretical interpretation.
2. **Property analysis.** We show that RoPE decays with increasing relative distance—a desirable property for natural language—and argue that previous relative position encodings are not compatible with linear self-attention.
3. **Systematic evaluation.** On multiple long-text benchmarks, RoFormer consistently outperforms its alternatives. Some pre-trained models are released on GitHub.

The rest of the paper formalizes position encoding in self-attention and revisits prior work (Section 2), describes RoPE and its properties (Section 3), reports experiments (Section 4), and concludes (Section 5).

## 2 Background and Related Work

### 2.1 Preliminary

Let $S_N=\{w_i\}_{i=1}^{N}$ be a sequence of $N$ input tokens, and let $E_N=\{\mathbf{x}_i\}_{i=1}^{N}$ be the corresponding word embeddings, where $\mathbf{x}_i\in\mathbb{R}^{d}$ is the $d$-dimensional embedding of $w_i$ without position information. Self-attention first incorporates position information into the embeddings and transforms them into queries, keys, and values:

$$
\mathbf{q}_m = f_q(\mathbf{x}_m, m),\qquad
\mathbf{k}_n = f_k(\mathbf{x}_n, n),\qquad
\mathbf{v}_n = f_v(\mathbf{x}_n, n),
\tag{1}
$$

where $\mathbf{q}_m$, $\mathbf{k}_n$, and $\mathbf{v}_n$ incorporate the $m$-th and $n$-th positions through $f_q$, $f_k$, and $f_v$. Queries and keys then produce attention weights, and the output is a weighted sum of values:

$$
a_{m,n}=\frac{\exp\left(\mathbf{q}_m^\top\mathbf{k}_n/\sqrt{d}\right)}
{\sum_{j=1}^{N}\exp\left(\mathbf{q}_m^\top\mathbf{k}_j/\sqrt{d}\right)},
\qquad
\mathbf{o}_m=\sum_{n=1}^{N}a_{m,n}\mathbf{v}_n .
\tag{2}
$$

Existing Transformer position-encoding methods differ mainly in the choice of the functions $f$ in Equation (1).

### 2.2 Absolute Position Embedding

A typical choice is

$$
f_{t}(x_i,i):=W_{t}\left(x_i+\mathbf{p}_i\right),\qquad t\in\{q,k,v\},
\tag{3}
$$

where $\mathbf{p}_i\in\mathbb{R}^{d}$ depends only on the position of token $x_i$. Prior work learns a set of vectors $\mathbf{p}_i\in\{\mathbf{p}_t\}_{t=1}^{L}$, with $L$ the maximum sequence length. Vaswani et al. instead generate $\mathbf{p}_i$ with sinusoids:

$$
p_{i,2t}=\sin\left(i/10000^{2t/d}\right),\qquad
p_{i,2t+1}=\cos\left(i/10000^{2t/d}\right),
\tag{4}
$$

where $p_{i,2t}$ is the $2t$-th element of the $d$-dimensional vector $\mathbf{p}_i$. RoPE is related to this sinusoidal intuition, but instead of adding position to the contextual representation it multiplies by sinusoidal functions to inject relative position information.

Absolute encodings are simple and decoupled from the rest of the model, but the position vectors are tied to the maximum length; with trainable vectors each position is also learned independently, which limits generalization beyond the training length. This limitation motivates relative encodings and the method proposed here.

### 2.3 Relative Position Embedding

Shaw et al. instantiate Equation (1) as

$$
f_q(x_m):=W_q x_m,\quad
f_k(x_n,n):=W_k\left(x_n+\tilde{\mathbf{p}}_k^{r}\right),\quad
f_v(x_n,n):=W_v\left(x_n+\tilde{\mathbf{p}}_v^{r}\right),
\tag{5}
$$

where $\tilde{\mathbf{p}}_k^{r},\tilde{\mathbf{p}}_v^{r}\in\mathbb{R}^{d}$ are trainable relative position embeddings and $r=\operatorname{clip}(m-n,r_{\min},r_{\max})$ is the clipped relative distance; the clipping assumes that precise relative position is no longer useful beyond a certain distance.

Transformer-XL keeps the form of Equation (3) and decomposes $\mathbf{q}_m^\top\mathbf{k}_n$ in Equation (2) as

$$
\mathbf{q}_m^\top\mathbf{k}_n=
\mathbf{x}_m^\top W_q^\top W_k\mathbf{x}_n
+\mathbf{x}_m^\top W_q^\top W_k\mathbf{p}_n
+\mathbf{p}_m^\top W_q^\top W_k\mathbf{x}_n
+\mathbf{p}_m^\top W_q^\top W_k\mathbf{p}_n .
\tag{6}
$$

The idea is to replace the absolute $\mathbf{p}_n$ with its sinusoid-encoded relative counterpart $\tilde{\mathbf{p}}_{m-n}$, replace the absolute $\mathbf{p}_m$ in the third and fourth terms with trainable vectors $\mathbf{u}$ and $\mathbf{v}$ that are independent of the query position, and use separate projections $W_k$ and $W_{f_k}$ for content-based and location-based keys:

$$
\mathbf{q}_m^\top\mathbf{k}_n=
\mathbf{x}_m^\top W_q^\top W_k\mathbf{x}_n
+\mathbf{x}_m^\top W_q^\top W_{f_k}\tilde{\mathbf{p}}_{m-n}
+\mathbf{u}^\top W_q^\top W_k\mathbf{x}_n
+\mathbf{v}^\top W_q^\top W_{f_k}\tilde{\mathbf{p}}_{m-n}.
\tag{7}
$$

Setting $f_v(x_j):=W_v x_j$ removes position information from the value term. Later work (T5, DeBERTa) keeps position only in the attention weights. T5 rewrites Equation (6) as

$$
\mathbf{q}_m^\top\mathbf{k}_n=\mathbf{x}_m^\top W_q^\top W_k\mathbf{x}_n+b_{m,n},
\tag{8}
$$

with a trainable bias $b_{m,n}$; another variant models word pairs and position pairs with different projections,

$$
\mathbf{q}_m^\top\mathbf{k}_n=
\mathbf{x}_m^\top W_q^\top W_k\mathbf{x}_n
+\mathbf{p}_m^\top U_q^\top U_k\mathbf{p}_n+b_{m,n}.
\tag{9}
$$

DeBERTa argues that the relative positions of two tokens are fully modeled only by the middle two terms of Equation (6) and directly replaces $\mathbf{p}_m,\mathbf{p}_n$ with $\tilde{\mathbf{p}}_{m-n}$:

$$
\mathbf{q}_m^\top\mathbf{k}_n=
\mathbf{x}_m^\top W_q^\top W_k\mathbf{x}_n
+\mathbf{x}_m^\top W_q^\top W_k\tilde{\mathbf{p}}_{m-n}
+\tilde{\mathbf{p}}_{m-n}^\top W_q^\top W_k\mathbf{x}_n .
\tag{10}
$$

A comparison of four relative-position variants finds the one closest to Equation (10) most efficient. All of these methods modify Equation (6) under the self-attention setting of Equation (2), and Equation (6) comes from the additive decomposition of Equation (3): position is added to the contextual representation. We instead derive the relative encoding from Equation (1) under constraints and inject it by rotating the representation, which yields a clearer interpretation.

The shared structure deserves emphasis. Position enters $\mathbf{x}_i$ additively and is then projected by $W_q$ and $W_k$, so the attention score is a bilinear function of $(\mathbf{x}_m+\mathbf{p}_m)$ and $(\mathbf{x}_n+\mathbf{p}_n)$; expanding it produces content–content, content–position, position–content, and position–position terms, exactly the four terms of Equation (6). Relative encodings trade off or reparameterize these terms. Linear attention, however, requires $\operatorname{sim}(\mathbf{q}_m,\mathbf{k}_n)$ to factor into a term depending only on $m$ and a term depending only on $n$ before the associative trick can reduce cost to $O(N)$. Additive position terms break that factorization; the multiplicative form of RoPE preserves it.

## 3 Proposed Approach

This section formalizes the relative position problem (3.1), derives RoPE (3.2), studies its properties (3.3), and gives the theoretical explanation (3.4).

### 3.1 Formulation

As Equation (2) shows, $\mathbf{q}_m^\top\mathbf{k}_n$ conveys information between tokens at different positions. To incorporate relative position information, we require the inner product of query $\mathbf{q}_m$ and key $\mathbf{k}_n$ to be a function $g$ that takes only the word embeddings $\mathbf{x}_m,\mathbf{x}_n$ and their relative position $m-n$ as inputs. In other words, the inner product should encode position only in relative form:

$$
\left\langle f_q(\mathbf{x}_m,m),\,f_k(\mathbf{x}_n,n)\right\rangle
= g(\mathbf{x}_m,\mathbf{x}_n,m-n).
\tag{11}
$$

The goal is to find an equivalent encoding mechanism that solves for $f_q(\mathbf{x}_m,m)$ and $f_k(\mathbf{x}_n,n)$ satisfying this relation. The sign convention for relative position does not affect the result: $g$ is real-valued below, and swapping $m-n$ for $n-m$ only conjugates the relevant complex exponential, leaving the real part unchanged.

### 3.2 Rotary Position Embedding

#### 3.2.1 A 2D Case

Start with $d=2$. Using the geometric properties of vectors in the plane and their complex form, one solution to Equation (11) is

$$
\begin{aligned}
f_q(\mathbf{x}_m,m)&=(W_q\mathbf{x}_m)\,e^{\mathrm{i}m\theta},\\
f_k(\mathbf{x}_n,n)&=(W_k\mathbf{x}_n)\,e^{\mathrm{i}n\theta},\\
g(\mathbf{x}_m,\mathbf{x}_n,m-n)&=\operatorname{Re}\left[(W_q\mathbf{x}_m)(W_k\mathbf{x}_n)^{*}e^{\mathrm{i}(m-n)\theta}\right],
\end{aligned}
\tag{12}
$$

where $\operatorname{Re}[\cdot]$ takes the real part, $(W_k\mathbf{x}_n)^{*}$ is the complex conjugate, and $\theta\in\mathbb{R}$ is a preset non-zero constant. Equivalently, $f_{\{q,k\}}$ can be written as a matrix product:

$$
f_{\{q,k\}}(\mathbf{x}_m,m)=
\begin{pmatrix}
\cos m\theta & -\sin m\theta\\
\sin m\theta & \cos m\theta
\end{pmatrix}
W_{\{q,k\}}\mathbf{x}_m .
\tag{13}
$$

Incorporating relative position is therefore straightforward: rotate the affine-transformed word embedding by an angle proportional to its position index. This is the intuition behind the name Rotary Position Embedding.

#### 3.2.2 General Form

To generalize to any even dimension $\mathbf{x}_i\in\mathbb{R}^{d}$, divide the $d$-dimensional space into $d/2$ subspaces and combine them using the linearity of the inner product:

$$
f_{\{q,k\}}(\mathbf{x}_m,m)=R^{d}_{\Theta,m}W_{\{q,k\}}\mathbf{x}_m,
\tag{14}
$$

where

$$
R^{d}_{\Theta,m}=
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0\\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
\tag{15}
$$

is the rotary matrix with pre-defined parameters $\Theta=\{\theta_i=10000^{-2(i-1)/d},\ i=1,\dots,d/2\}$. Figure 1 illustrates the construction. Applying RoPE to self-attention gives

$$
\mathbf{q}_m^\top\mathbf{k}_n=
\left(R^{d}_{\Theta,m}W_q\mathbf{x}_m\right)^\top
\left(R^{d}_{\Theta,n}W_k\mathbf{x}_n\right)
=\mathbf{x}_m^\top W_q^\top R^{d}_{\Theta,n-m}W_k\mathbf{x}_n,
\tag{16}
$$

where $R^{d}_{\Theta,n-m}=\left(R^{d}_{\Theta,m}\right)^\top R^{d}_{\Theta,n}$. Because $R^{d}_{\Theta}$ is orthogonal, position encoding is numerically stable. The matrix is also highly sparse, so the direct multiplication in Equation (16) is not efficient; Section 3.4.2 gives a better realization.

![Figure 1: Illustration of Rotary Position Embedding (RoPE). Top: the 2D case (d = 2), where the query/key vector (x1, x2) at position m is rotated by mθ₁ to produce the position-encoded (x1′, x2′). Bottom: the general form, where a d-dimensional query/key is split into d/2 two-dimensional blocks, each rotated by an angle proportional to its position m at its own frequency θi, and the rotated blocks are reassembled into the position-encoded query/key.](assets/figures/fig1.png)

Unlike the additive position embeddings of Equations (3)–(10), this approach is multiplicative. RoPE incorporates relative position through a product of rotation matrices rather than by altering a term in the expanded additive formulation.

### 3.3 Properties of RoPE

RoPE has four notable properties.

- **Relative position dependency.** By Equation (16), $\mathbf{q}_m^\top\mathbf{k}_n$ depends on position only through $n-m$; the absolute positions are absorbed by the rotation matrices, so attention scores are naturally relative.
- **Sequence-length flexibility.** Rotation angles depend only on position indices and preset frequencies, with no trainable parameters tied to a maximum length. The same parameters can process longer sequences than those seen in training. This is structural flexibility: it provides a basis for length extrapolation but does not by itself guarantee performance at arbitrary lengths.
- **Long-term decay.** With $\theta_i=10000^{-2i/d}$ as in Vaswani et al., the inner product decays as the relative distance grows (Section 3.4.3), matching the intuition that distant token pairs should be less connected.
- **Compatibility with linear attention.** Rotation preserves the norm of hidden representations, so RoPE can be multiplied onto the outputs of non-negative feature maps and used in linear self-attention.

**RoPE with linear attention.** Self-attention can be written in a more general form:

$$
\operatorname{Attention}(Q,K,V)_m=
\frac{\sum_{n=1}^{N}\operatorname{sim}(\mathbf{q}_m,\mathbf{k}_n)\mathbf{v}_n}
{\sum_{n=1}^{N}\operatorname{sim}(\mathbf{q}_m,\mathbf{k}_n)}.
\tag{17}
$$

The original self-attention uses $\operatorname{sim}(\mathbf{q}_m,\mathbf{k}_n)=\exp(\mathbf{q}_m^\top\mathbf{k}_n/\sqrt{d})$ and computes an inner product for every token pair, at $O(N^2)$ cost. Linear attention reformulates Equation (17) as

$$
\operatorname{Attention}(Q,K,V)_m=
\frac{\sum_{n=1}^{N}\phi(\mathbf{q}_m)^\top\varphi(\mathbf{k}_n)\mathbf{v}_n}
{\sum_{n=1}^{N}\phi(\mathbf{q}_m)^\top\varphi(\mathbf{k}_n)},
\tag{18}
$$

where $\phi(\cdot)$ and $\varphi(\cdot)$ are usually non-negative. Katharopoulos et al. use $\phi(x)=\varphi(x)=\mathrm{elu}(x)+1$ and exploit associativity to multiply keys and values first; Shen et al. normalize queries and keys separately with softmax before the inner product, equivalent to $\phi(\mathbf{q}_i)=\mathrm{softmax}(\mathbf{q}_i)$ and $\varphi(\mathbf{k}_j)=\exp(\mathbf{k}_j)$.

Since RoPE injects position by rotation and leaves the norm unchanged, the rotation matrix can be applied to the outputs of the non-negative functions:

$$
\operatorname{Attention}(Q,K,V)_m=
\frac{\sum_{n=1}^{N}\left(R^{d}_{\Theta,m}\phi(\mathbf{q}_m)\right)^\top
\left(R^{d}_{\Theta,n}\varphi(\mathbf{k}_n)\right)\mathbf{v}_n}
{\sum_{n=1}^{N}\phi(\mathbf{q}_m)^\top\varphi(\mathbf{k}_n)}.
\tag{19}
$$

The denominator is left unchanged to avoid division by zero, and the numerator may contain negative terms. The weights on the values are therefore not strictly normalized probabilities, but the computation can still model the importance of values.

### 3.4 Theoretical Explanation

#### 3.4.1 Derivation of RoPE under 2D

For $d=2$, consider word embeddings $\mathbf{x}_q$ and $\mathbf{x}_k$ corresponding to a query and a key at positions $m$ and $n$. By Equation (1), their position-encoded counterparts are

$$
\mathbf{q}_m=f_q(\mathbf{x}_q,m),\qquad
\mathbf{k}_n=f_k(\mathbf{x}_k,n).
\tag{20}
$$

Assume a function $g$ defines the inner product between the outputs of $f_{\{q,k\}}$:

$$
\mathbf{q}_m^\top\mathbf{k}_n=
\left\langle f_q(\mathbf{x}_q,m),f_k(\mathbf{x}_k,n)\right\rangle
=g(\mathbf{x}_q,\mathbf{x}_k,n-m),
\tag{21}
$$

with the initial condition

$$
\mathbf{q}=f_q(\mathbf{x}_q,0),\qquad
\mathbf{k}=f_k(\mathbf{x}_k,0),
\tag{22}
$$

i.e., the vectors with empty position information. Using the complex form of 2D vectors, decompose Equations (20) and (21) into radial and angular parts:

$$
\begin{aligned}
f_q(\mathbf{x}_q,m)&=R_q(\mathbf{x}_q,m)e^{\mathrm{i}\Theta_q(\mathbf{x}_q,m)},\\
f_k(\mathbf{x}_k,n)&=R_k(\mathbf{x}_k,n)e^{\mathrm{i}\Theta_k(\mathbf{x}_k,n)},\\
g(\mathbf{x}_q,\mathbf{x}_k,n-m)&=R_g(\mathbf{x}_q,\mathbf{x}_k,n-m)e^{\mathrm{i}\Theta_g(\mathbf{x}_q,\mathbf{x}_k,n-m)}.
\end{aligned}
\tag{23}
$$

Substituting into Equation (21) gives

$$
R_q(\mathbf{x}_q,m)R_k(\mathbf{x}_k,n)=R_g(\mathbf{x}_q,\mathbf{x}_k,n-m),\qquad
\Theta_k(\mathbf{x}_k,n)-\Theta_q(\mathbf{x}_q,m)=\Theta_g(\mathbf{x}_q,\mathbf{x}_k,n-m).
\tag{24}
$$

The key step is to read a 2D vector as a complex number: the modulus is its length and the argument is its direction. If position encoding changes only the direction and not the length, the modulus of the inner product is position-independent and all position information is carried by the argument difference. This observation leads directly to the rotation interpretation.

Setting $m=n$ in Equation (24) and using the initial condition yields

$$
R_q(\mathbf{x}_q,m)R_k(\mathbf{x}_k,m)=\lVert\mathbf{q}\rVert\lVert\mathbf{k}\rVert,\qquad
\Theta_k(\mathbf{x}_k,m)-\Theta_q(\mathbf{x}_q,m)=\theta_k-\theta_q .
\tag{25}
$$

The modulus functions are thus position-independent and can be taken as $R_q=\lVert\mathbf{q}\rVert$ and $R_k=\lVert\mathbf{k}\rVert$. Moreover, $\Theta_q(\mathbf{x}_q,m)-\theta_q=\Theta_k(\mathbf{x}_k,m)-\theta_k$ shows that the angular function does not depend on the content of the query or key. Denoting this difference, which depends only on $m$, by $\varphi(m)$:

$$
\Theta_f(\mathbf{x}_{\{q,k\}},m)=\varphi(m)+\theta_{\{q,k\}}.
\tag{26}
$$

Substituting $n=m+1$ into Equation (24) gives $\varphi(m+1)-\varphi(m)=\Theta_g(\mathbf{x}_q,\mathbf{x}_k,1)+\theta_q-\theta_k$. The right-hand side does not depend on $m$, so over consecutive integer inputs $\varphi(m)$ is an arithmetic progression:

$$
\varphi(m)=m\theta+\gamma,
\tag{27}
$$

with constants $\theta\neq 0$ and $\gamma$. Hence

$$
f_q(\mathbf{x}_q,m)=\lVert\mathbf{q}\rVert e^{\mathrm{i}(\theta_q+m\theta+\gamma)}=\mathbf{q}\,e^{\mathrm{i}(m\theta+\gamma)},\qquad
f_k(\mathbf{x}_k,n)=\mathbf{k}\,e^{\mathrm{i}(n\theta+\gamma)}.
\tag{28}
$$

Equation (22) places no further constraint on $f_q$ and $f_k$, so $f_q(\mathbf{x}_m,0)$ and $f_k(\mathbf{x}_n,0)$ are free. To match Equation (3), set $\mathbf{q}=f_q(\mathbf{x}_m,0)=W_q\mathbf{x}_m$, $\mathbf{k}=f_k(\mathbf{x}_n,0)=W_k\mathbf{x}_n$, and $\gamma=0$:

$$
f_q(\mathbf{x}_m,m)=(W_q\mathbf{x}_m)e^{\mathrm{i}m\theta},\qquad
f_k(\mathbf{x}_n,n)=(W_k\mathbf{x}_n)e^{\mathrm{i}n\theta}.
\tag{29}
$$

This is exactly the 2D solution of Section 3.2.1.

#### 3.4.2 Efficient Realization

Exploiting the sparsity of $R^{d}_{\Theta,m}$ in Equation (15), the product of $R^{d}_{\Theta}$ with $\mathbf{x}\in\mathbb{R}^{d}$ can be computed as

$$
R^{d}_{\Theta,m}\mathbf{x}=
\begin{pmatrix}x_1\\x_2\\x_3\\x_4\\\vdots\\x_{d-1}\\x_d\end{pmatrix}
\odot
\begin{pmatrix}\cos m\theta_1\\ \cos m\theta_1\\ \cos m\theta_2\\ \cos m\theta_2\\ \vdots\\ \cos m\theta_{d/2}\\ \cos m\theta_{d/2}\end{pmatrix}
+
\begin{pmatrix}-x_2\\x_1\\-x_4\\x_3\\\vdots\\-x_d\\x_{d-1}\end{pmatrix}
\odot
\begin{pmatrix}\sin m\theta_1\\ \sin m\theta_1\\ \sin m\theta_2\\ \sin m\theta_2\\ \vdots\\ \sin m\theta_{d/2}\\ \sin m\theta_{d/2}\end{pmatrix},
\tag{30}
$$

where $\odot$ is element-wise multiplication. Each rotation then costs two element-wise multiplications and one element-wise addition, for $O(d)$ overall, without materializing a $d\times d$ matrix. In practice the adjacent dimension pairs are interleaved, and the per-position $\cos m\theta_i$ and $\sin m\theta_i$ are cached and applied element-wise—one reason RoPE is easy to deploy in large models.

#### 3.4.3 Long-term Decay

Group the components of $\mathbf{q}=W_q\mathbf{x}_m$ and $\mathbf{k}=W_k\mathbf{x}_n$ in pairs; the inner product in Equation (16) becomes a complex multiplication:

$$
\left(R^{d}_{\Theta,m}W_q\mathbf{x}_m\right)^\top
\left(R^{d}_{\Theta,n}W_k\mathbf{x}_n\right)
=\operatorname{Re}\left[\sum_{i=0}^{d/2-1}
\mathbf{q}^{[2i:2i+1]}\left(\mathbf{k}^{[2i:2i+1]}\right)^{*}
e^{\mathrm{i}(m-n)\theta_i}\right],
\tag{31}
$$

where $\mathbf{q}^{[2i:2i+1]}$ denotes entries $2i$ through $2i+1$ of $\mathbf{q}$. Let $h_i=\mathbf{q}^{[2i:2i+1]}(\mathbf{k}^{[2i:2i+1]})^{*}$ and $S_j=\sum_{i=0}^{j-1}e^{\mathrm{i}(m-n)\theta_i}$, with $h_{d/2}=0$ and $S_0=0$. Abel summation gives

$$
\sum_{i=0}^{d/2-1}h_i e^{\mathrm{i}(m-n)\theta_i}
=-\sum_{i=0}^{d/2-1}S_{i+1}\left(h_{i+1}-h_i\right).
\tag{32}
$$

Taking absolute values yields the bound

$$
\left|\sum_{i=0}^{d/2-1}h_i e^{\mathrm{i}(m-n)\theta_i}\right|
\le \max_i\left|h_{i+1}-h_i\right|\sum_{i=0}^{d/2-1}\left|S_{i+1}\right|.
\tag{33}
$$

With $\theta_i=10000^{-2i/d}$ (a one-index shift from Equation (15)), the average $\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$ decays as the relative distance $|m-n|$ grows (Figure 2 of the paper). The upper bound on the inner product therefore decreases with distance, consistent with the intuition that distant tokens are less related.

Abel summation turns a sum of oscillating terms into a weighted sum of partial sums, attributing decay to the cancellation among $S_j$ as distance grows. With frequencies in geometric progression, the phase differences spread out with $|m-n|$ and the cancellation becomes more complete, producing a distance-dependent bound.

## 4 Experiments and Evaluation

We evaluate RoFormer on several NLP tasks: machine translation (4.1), pre-training compared with BERT (4.2), GLUE fine-tuning (4.3), Performer with linear attention (4.4), and Chinese data (4.5). All experiments ran on two cloud servers with 4× V100 GPUs each.

### 4.1 Machine Translation

We first test RoFormer on sequence-to-sequence translation using the standard WMT 2014 English–German dataset (about 4.5 million sentence pairs), against the Transformer baseline of Vaswani et al.

**Implementation.** We modify the self-attention layer of the baseline to use RoPE, keep the English–German setup, and share a 37k-word BPE vocabulary between source and target. At evaluation, a single model is obtained by averaging the last 5 checkpoints; decoding uses beam search with beam size 4 and length penalty 0.6. The experiment is implemented in PyTorch with fairseq. We use Adam ($\beta_1=0.9$, $\beta_2=0.98$), a learning rate increased linearly from $1\times10^{-7}$ to $5\times10^{-4}$ and then decayed as the inverse square root of the step, and label smoothing 0.1. The metric is test-set BLEU.

**Results.** Training the baseline and RoFormer under identical settings gives Table 1.

Table 1: BLEU on WMT 2014 English–German translation for RoFormer and the Transformer baseline.
{: .table-caption #tab-1 }

| Model | BLEU |
| --- | :---: |
| Transformer-base (Vaswani et al.) | 27.3 |
| RoFormer | **27.5** |

RoFormer obtains a slightly higher BLEU. The gain is small but consistent: with the same parameter count and training budget, replacing sinusoidal position encoding with RoPE already helps. Translation is sensitive to word order but involves relatively limited sequence lengths, where sinusoidal encoding already supplies a strong position signal; RoPE's advantages show up more clearly on longer sequences and with linear attention.

### 4.2 Pre-training Language Modeling

The second experiment tests RoPE on contextual representation learning by replacing BERT's sinusoidal position encoding during pre-training.

**Setup.** We pre-train on BookCorpus and Wikipedia (from Hugging Face Datasets, Apache License 2.0), split 8:2 into training and validation. The masked language modeling (MLM) loss during training is the evaluation metric, and bert-base-uncased is the baseline.

**Implementation.** We replace the sinusoidal encoding in the baseline's self-attention block with RoPE and implement attention as in Equation (16). Both BERT and RoFormer use batch size 64, maximum sequence length 512, and 100k training steps with AdamW at learning rate $1\times10^{-5}$.

**Results.** The MLM loss curves (Figure 3, left, of the paper) show that RoFormer converges faster than vanilla BERT: at the same number of steps its loss is lower. RoPE thus does more than supply a position signal; it also helps optimization reach a good region earlier.

### 4.3 Fine-tuning on GLUE

We fine-tune the pre-trained RoFormer on several GLUE tasks to assess generalization.

**Setup.** We use MRPC, SST-2, QNLI, STS-B, QQP, and MNLI. F1 is reported for MRPC and QQP, Spearman correlation for STS-B, and accuracy for the rest.

**Implementation.** Each task is fine-tuned for 3 epochs with Hugging Face Transformers, maximum sequence length 512, batch size 32, and learning rates of 2, 3, 4, or $5\times10^{-5}$; following BERT, we report the best averaged validation result.

**Results.** See Table 2.

Table 2: Fine-tuning results on GLUE. F1 for MRPC/QQP, Spearman correlation for STS-B, accuracy otherwise.
{: .table-caption #tab-2 }

| Model | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI(m/mm) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT | 88.9 | 93.5 | 90.5 | 85.8 | 71.2 | 84.6/83.4 |
| RoFormer | **89.5** | 90.7 | 88.0 | **87.0** | **86.4** | 80.2/79.8 |

RoFormer beats BERT on three of the six datasets: MRPC by 0.6, STS-B by 1.2, and QQP by 15.2, while trailing on SST-2, QNLI, and MNLI. The pre-training budget here (100k steps, sequence length 512) is far smaller than BERT's, and hyperparameters were only lightly searched, so Table 2 is better read as evidence that RoPE is competitive on several tasks than as a comprehensive verdict. The paper's own summary is that RoFormer significantly outperforms BERT on three of the six datasets, with considerable improvements.

The coexistence of three gains and three losses indicates that RoPE does not dominate every downstream task. A plausible reading is that tasks differ in how much they rely on position: sentence-pair matching and semantic similarity depend more on word order and relative structure, whereas natural language inference may lean more on lexical and syntactic cues. This explanation would need more systematic ablations.

### 4.4 Performer with RoPE

Performer introduces linear attention to avoid the quadratic cost of attention with sequence length. As discussed in Section 3.3, RoPE can be plugged into Performer to provide relative position encoding while preserving linear complexity. We demonstrate this on language-modeling pre-training.

**Implementation.** Experiments use Enwik8, derived from English Wikipedia and containing markup, special characters, and text in other languages besides English. RoPE is added to a 12-layer, 768-dimensional, 12-head character-level Performer. To isolate the effect of RoPE, we compare pre-training loss with and without it under identical settings (learning rate $1\times10^{-4}$, batch size 128, fixed maximum sequence length 1024).

**Results.** Figure 3 (right) of the paper shows that, at the same number of training steps, adding RoPE to Performer yields faster convergence and lower loss. Combined with linear complexity, this makes Performer more attractive.

### 4.5 Evaluation on Chinese Data

We also report results on Chinese data. To test RoFormer on long texts, the documents used here exceed 512 characters.

The key question for long documents is whether position encoding retains discrimination beyond the pre-training length. The legal-domain CAIL2019-SCM dataset provides such a test: its case descriptions mostly exceed 512 characters. Chinese also lacks natural spaces, so word-level tokenization introduces an additional granularity choice.

**Implementation.** We modify WoBERT by replacing its absolute position embedding with RoPE. For comparison, Table 3 lists the tokenization level and position encoding of several Chinese pre-trained Transformers: BERT is character-level with absolute position, WoBERT word-level with absolute position, NEZHA character-level with relative position, and RoFormer word-level with RoPE.

Table 3: Tokenization level and position embedding of Chinese pre-trained models. “abs.” and “rel.” denote absolute and relative position embeddings.
{: .table-caption #tab-3 }

| Model | BERT | WoBERT | NEZHA | RoFormer |
| --- | :---: | :---: | :---: | :---: |
| Tokenization level | char | word | char | word |
| Position embedding | abs. | abs. | rel. | RoPE |

**Pre-training.** RoFormer is pre-trained on about 34 GB of data from Chinese Wikipedia, news, and forums. Training proceeds in multiple stages with different batch sizes and maximum input lengths so that the model adapts to various scenarios. As Table 4 shows, accuracy rises with the upper bound on sequence length, demonstrating RoFormer's ability to handle long texts; we attribute this to the generalization of RoPE.

Table 4: Multi-stage pre-training schedule of RoFormer on Chinese data. Each stage uses a specific combination of maximum sequence length and batch size.
{: .table-caption #tab-4 }

| Stage | Max seq length | Batch size | Training steps | Loss | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

**Downstream task and dataset.** We use the Chinese AI and Law 2019 Similar Case Matching dataset (CAIL2019-SCM) to test long-text semantic matching. It contains 8,964 triplets of cases published by the Supreme People's Court of China. The input $(A,B,C)$ consists of fact descriptions of three cases, and the task is to predict whether $(A,B)$ is closer than $(A,C)$ under a predefined similarity measure. Because the documents are long (mostly over 512 characters), existing methods struggle to perform significantly on this dataset. We split the data into training, validation, and test sets at 6:2:2.

**Results.** We apply the pre-trained RoFormer to CAIL2019-SCM with different input lengths and compare it with BERT and WoBERT trained on the same pre-training data (Table 5). With a 512 cut-off, RoFormer is comparable to WoBERT and slightly better than BERT; raising the maximum input length to 1024 gives RoFormer a clear gain over WoBERT (69.79% vs. 68.10% on the test set, about 1.7 points).

Table 5: Results on CAIL2019-SCM. The first column gives the maximum cut-off sequence length; numbers are accuracy.
{: .table-caption #tab-5 }

| Model | Validation | Test |
| --- | :---: | :---: |
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| RoFormer-1024 | **66.07%** | **69.79%** |

**Limitations.** Despite the theoretical grounding and promising experiments, the method has limitations. First, although relative position relations are formulated mathematically as rotations in 2D subspaces, there is no thorough explanation of why the model converges faster than baselines with other position encodings. Second, although the long-term decay property for token inner products is proved (Section 3.3) and the model outperforms peers on long texts, we do not yet have a faithful explanation for that advantage. Finally, RoFormer builds on Transformer infrastructure and requires hardware resources for pre-training.

## 5 Conclusions

We proposed a new position embedding method that incorporates explicit relative position dependency into self-attention to enhance Transformer architectures. Our theoretical analysis shows that relative position can be naturally formulated through vector products in self-attention, with absolute position encoded by a rotation matrix. We also illustrated mathematically several advantageous properties of the method when applied to Transformers. Experiments on English and Chinese benchmarks show that it encourages faster convergence in pre-training and achieves better performance on long-text tasks.

The significance of RoPE goes beyond the metrics reported here: it turns position from a bias added to representations into a rotation applied to them, so attention scores depend only on relative distance while no extra parameters are introduced and the structure of linear attention is preserved. This design later became the mainstream choice for position encoding in large language models and motivated a line of work on frequency scaling and length extrapolation.

From today's perspective, RoPE offers a set of composable primitives: the frequency base can be rescaled, rotation can be applied blockwise or to a subset of dimensions, and positions can be interpolated or extrapolated. Most later variants keep the basic form of Equations (14)–(16) and differ mainly in the choice of $\Theta$ and where it is applied—which is why RoPE has carried across model scales and architectural changes as the default position encoding.

## Further Reading

- [DeepSeek-V2](../deepseek-v2/index.html): the DeepSeek-V2 technical report, whose multi-head latent attention (MLA) compresses the KV cache at inference time.
- [TransMLA](../transmla/index.html): converting pre-trained GQA models into MLA to reduce inference-time KV-cache cost.
- [BidKV](../bidkv/index.html): a utility-ranked KV-cache reclamation policy for online LLM serving, another infrastructure problem for long-context inference.
