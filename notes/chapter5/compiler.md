## 编译器：表达，变换与选择

### 用循环变换表达分块与融合

取 $Y = SiLU(AW)$，有两种方式计算：

```python
for i in range(M):
    for j in range(N):
        acc = 0.0
        for k in range(K):
            acc += A[i, k] * W[k, j]
        C[i, j] = round_to_bf16(acc)

for i in range(M):
    for j in range(N):
        Y[i, j] = SiLU(C[i, j])
```

```python
for io in range(cell_div(M, BM)):
    for jo in range(cell_div(N, BN)):
        acc = zeros_fp32(BM, BN)
        for ko in range(cell_div(K, BK)):
            a = load_tile(A, io, ko, BM, BK)
            w = load_tile(W, ko, jo, BK, BN)
            matmul_accumulate(acc, a, w)
        c = round_to_bf16(acc)
        store_tile(Y, io, jo, SiLU(c))
```

后一种方式利用了拆分和重排，以实现分块和融合。  
同时调整了激活的计算位置，使得中间结果 C 不在需要额外的写回和重读。

### 依赖与舍入如何限制变换

即使将激活移入了输出 tile，也仍然需要安排在 ko 循环之后。  
如果在部分和合并之前做 SiLU，由于 SiLU 不具有线性性，得到的结果与应有的会有很大差异。

所以编译器在优化时，需要考虑算子之间的依赖关系，确保变换不会破坏计算的正确性。

### AKG：用 isl schedule 安排计算与搬移

AKG 是华为开发并开源的编译器，能够自动生成神经网络算子，提供算子优化与代码生成。

考虑最简单的 $C = AB$：

```python
for i in range(M):
    for j in range(N):
        acc = 0.0
        for k in range(K):
            acc += A[i, k] * B[k, j]
        C[i, j] = acc
```

如果每个输出元素都从片外读取 A 和 B，计算完后再写回 C，  
那么完整的输入读取量为 $V = 2MNK$。

如果采用之前的分块方法，假设每次读取 A 的 $m\times k$ 块和 B 的 $k\times n$ 块，计算出 C 的 $m\times n$ 块，
那么总的读取量为：

$$
V = 2MK\frac{N}{n} + 2KN\frac{M}{m}
$$

#### isl schedule 是什么

每次循环是许多语句实例组成的，将之前的分块乘加语句记为 $S(i,j,k)$。  
所有合法实例构成迭代域：

$$
D = \{ S(i,j,k) | 0 \leq i < M, 0 \leq j < N, 0 \leq k < K\}
$$

isl schedule 给每个实例分配一个或多个整数坐标，将坐标按字典序排列，得到一个新的迭代域。  
例如：
原始循环可以写成

$$
S(i,j,k) \to (i,j,k)
$$

分块循环可以写成

$$
S(i,j,k) \to (i//BM, j//BN, k//BK, i\%BM, j\%BN, k\%BK)
$$

#### 从 schedule 到片上数据

同一个 schedule 可以直接写成数据搬移映射：

$$
\displaylines{
A_{L1}[io,ko,ii,kk] \from A[64 io+ii, 64 ko+kk] \cr
B_{L1}[ko,jo,kk,jj] \from B[64 ko+kk, 64 jo+jj] \cr
C_{L0}[ii,jj] \from \sum_{kk} A_{L0}[ii,kk] * B_{L0}[kk,jj] \cr
}
$$

### TileLang-Ascend：把同一份 schedule 写成 tile 程序

TileLang 将数据流与调度空间分开  
T.copy，T.gemm，T.reduce 等算子表示数据流  
T.Pipelined，布局标注，线程绑定和数据重排等表示调度
