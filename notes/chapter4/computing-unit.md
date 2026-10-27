## 计算单元

### 矩阵单元为什么适合神经网络

设当前执行规模为 $M\times N\times K$ 的矩阵运算，  
矩阵指令通常按照固定大小的块执行，设最小计算块为 $m\times n\times k$，矩阵尺寸不足部分会被填充为零。  
则有效运算占实际执行运算的比例为：
$$
\eta = \frac{MNK}{\lceil M/m\rceil \lceil N/n\rceil \lceil K/k\rceil mnk}
$$

### 矩阵计算之间的向量运算与控制

### QK,Softmax,PV如何组成流水线

### 低精度如何改变计算量，数据量及转换开销

设 n 个参数各需要 b bit，每 g 个参数共享一个 s bytes 的 scale，总元素数是分组大小的整数倍，则存储量为：
$$
V_W = \frac{nb}{8} + \frac{n}{g}s
$$  
分组越细，scale 越容易适应该组数值的范围，但是会增加 scale 的存储量。  

使用压缩值 q 与 scale a 时，可以先展开，再做矩阵乘法，  
也可以先让低精度矩阵指令处理 q，再按 scale 调整结果。  

对一行输入与一列权重的点积，设激活每组 scale 为 $a_g$，权重列 scale 为 $b_j$，则点积可以表示为：
$$
y_j \approx \sum_{g=1}^{G} a_g b_j \left(\sum_{k \in g} q_{x,k} q_{w,kj}\right)
$$

真实的单 token decode 计算中，在执行时间上，先展开再计算的方法较快。  
在显存占用上，先计算再展开更好。
