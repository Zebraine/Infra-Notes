## Attention-FFN 分离与异构执行

### Attention，FFN 与专家的执行边界

AF 指注意力与前馈网络的分工。  
MoE 模型适合这种分工，因为总专家权重较大，但每个 token 只选择其中一部分。

### CPU / GPU 协作

KTransformers 让 GPU 处理注意力和部分常驻专家，CPU 处理分给自己的专家。  
将分给 CPU 的工作提交后，GPU 继续执行可以并行的分支，最后同步并合并结果。

记专家参数量为 $P_e$，权重字节数为 $W_e$，输入宽度为 h，每元素字节数为 b。  
同一个专家在当前 batch 收到 m 个 token，其矩阵计算量为 $F_e(m) = 2mP_e FLOPs$，权重至少要读取一次。  
有效 CPU 算力，DRAM 带宽分别为 $C_C，B_D$，  
GPU 算力，HBM 带宽为 $C_G，B_H$，交接带宽为 $B_L$，  
每次交接的固定代价为 $\alpha$。  
交接与专家计算串行发生，则单个专家在两种执行方式下的耗时分别为：

$$
T_C(m) = 2\alpha + \frac{2mhb}{B_L} + \max\left(\frac{2mP_e}{C_C}, \frac{W_e}{B_D}\right)
$$

$$
T_G(m) = \alpha + \frac{W_e}{B_L} + \max\left(\frac{2mP_e}{C_G}, \frac{W_e}{B_H}\right)
$$

### 容量，并发与专家复用对服务能力的影响

若每个 token 独立，均匀地从 E 个专家中选择 K 个不同专家时，批内 M 个 token 下的活跃专家数期望为：

$$
\mathbb{E}[U] = E \left[1 - \left(1 - \frac{K}{E}\right)^M \right]
$$

### 跨机 AF，专家分离与 micro-batch 流水

在 MoE 模型中，可以将路由专家放到独立的专家资源池，由保留注意力与 KV 的 worker 发送激活并接收专家结果，这称为 专家激活，也常叫 EP 分离。  
大 EP 则是一个专家并行跨越较多的卡，让每张卡只保存少量专家，汇聚多个 attention worker 的输入，使专家获得足够大的 batch。
