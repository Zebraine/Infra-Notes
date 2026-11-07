## Abstract

We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.

## 1 Editor's Note

Adam is the optimizer that made adaptive moment estimation the default choice for deep learning. Before it, SGD with momentum was the workhorse, AdaGrad handled sparse gradients but decayed its learning rate too aggressively, and RMSProp fixed that decay but lacked momentum and a principled initialization correction. Adam combines the two ideas: an exponential moving average of the gradient (first moment) and an exponential moving average of the squared gradient (second moment), with a bias-correction step that makes the estimates unbiased early in training.

The result is an update whose per-parameter step size is approximately bounded by the learning rate, invariant to gradient rescaling, and naturally annealed as the signal-to-noise ratio falls near an optimum. It needs only first-order gradients and a constant amount of extra memory per parameter—three vectors (first moment, second moment, and the parameter itself), which is cheap relative to the model and activations at the scales where it is used.

For infrastructure, Adam matters because it defines the memory and communication pattern of training. Optimizer state is a major part of the memory budget (the ZeRO paper counts 12 bytes per parameter for Adam's fp32 states in mixed precision), and gradient all-reduce plus optimizer updates dominate the communication and kernel schedule. AdamW later decoupled weight decay from the adaptive update, but the core Adam estimator remains the baseline.

## 2 Problem and Background

Stochastic gradient-based optimization is central to machine learning. Many objectives are sums of subfunctions evaluated on minibatches, so gradient steps can be computed from individual subfunctions; this is SGD. SGD is efficient and effective, but it is sensitive to the choice of learning rate, and a single global learning rate may be wrong for parameters with very different gradient scales. AdaGrad adapts per-parameter rates using the history of squared gradients and works well with sparse features, but its accumulator only grows, so the effective learning rate decays to zero. RMSProp uses an exponential moving average of squared gradients to avoid that decay and works well in online and non-stationary settings.

The paper asks for a method that combines AdaGrad's robustness to sparse gradients with RMSProp's behavior on non-stationary objectives, while retaining the benefits of momentum. It should require only first-order gradients, little memory, and little hyper-parameter tuning, and it should be invariant to diagonal rescaling of the gradients.

## 3 Method

### 3.1 First and Second Moment Estimates

Let $g_t=\nabla_\theta f_t(\theta_{t-1})$ be the stochastic gradient at step $t$. Adam maintains an exponential moving average of the gradient and of its elementwise square:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

where $\beta_1,\beta_2\in[0,1)$ control the decay rates. $m_t$ estimates the first moment (mean) and $v_t$ estimates the second raw moment (uncentered variance). The defaults used in the paper are $\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, and $\epsilon=10^{-8}$.

A key property is the scale of the update. Ignoring $\epsilon$, the effective step is $\Delta_t=\alpha\,\hat m_t/\sqrt{\hat v_t}$. Because $|\mathbb{E}[g]/\sqrt{\mathbb{E}[g^2]}|\le 1$, the ratio is roughly bounded by one, so the step magnitude is approximately bounded by the step size $\alpha$. Rescaling all gradients by $c$ scales both $\hat m_t$ and $\sqrt{\hat v_t}$ by $c$, so the update is invariant to gradient scale. As the signal-to-noise ratio falls near an optimum, the step size anneals automatically.

### 3.2 Bias Correction and the Update Rule

The moving averages are initialized at zero, so early estimates are biased toward zero, especially when $\beta_2$ is close to one. Adam corrects this with

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

and updates parameters as

$$
\theta_t=\theta_{t-1}-\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The correction follows from expanding the moving average as a weighted sum of past gradients; the weights do not sum to one at finite $t$, and dividing by $1-\beta^t$ restores the unbiased estimate. The paper shows that without bias correction, a $\beta_2$ close to one can produce very large early updates, and the effect is particularly visible in sparse-gradient settings.

## 4 Experiments and Results

The paper evaluates Adam on L2-regularized multinomial logistic regression (MNIST and IMDB bag-of-words), multilayer perceptrons on MNIST with and without dropout, convolutional networks on CIFAR-10, and a variational autoencoder for the bias-correction study. Hyper-parameters are searched over a dense grid and the best setting is reported for each optimizer.

On MNIST logistic regression, Adam converges similarly to SGD with Nesterov momentum and both converge faster than AdaGrad. On the sparse IMDB bag-of-words task, AdaGrad outperforms SGD with momentum by a large margin; Adam converges as fast as AdaGrad, consistent with its ability to exploit sparse features. On the MLP experiments, Adam makes faster progress than the baselines in both iterations and wall-clock time; SFO is 5–10x slower per iteration because of curvature updates. On CIFAR-10 CNNs, Adam and SGD eventually converge faster than AdaGrad; Adam shows a marginal improvement over SGD with momentum while adapting the learning-rate scale per layer instead of hand-tuning it.

The bias-correction experiment trains a VAE over a grid of $\beta_1$, $\beta_2$, and $\alpha$. Values of $\beta_2$ close to one cause instability without bias correction, especially in the first epochs; with bias correction, Adam performs equal to or better than RMSProp across the grid.

## 5 Impact and Limitations

Adam became the default optimizer for deep learning and remains the baseline in training systems. Its per-parameter adaptive step sizes made learning-rate tuning much easier, and its moment estimates are now a standard part of optimizer implementations. The paper also introduced AdaMax, an infinity-norm variant, and its analysis helped establish the vocabulary of adaptive first-order methods.

The limitations are equally important. Adam stores two extra vectors per parameter, which increases optimizer-state memory (a key motivation for ZeRO and FSDP). The original paper couples L2 regularization into the gradient, which interacts poorly with adaptive scaling; AdamW later decouples weight decay. Some workloads generalize slightly worse with Adam than with well-tuned SGD, and there are known non-convergence examples for the original algorithm. These issues drove later work on AdamW, Lion, Sophia, and memory-efficient optimizers, but Adam's moment estimates and bias correction remain the foundation.
