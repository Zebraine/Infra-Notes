## 从负载需求到训练与服务预算

### Scaling Law: 模型规模与数据量的关系

设总训练计算预算为 C，参数量为 N，训练 token 数为 D，主要计算近似满足 C = 6ND。  
增加 N 就需要减少 D，反之亦然。  
想要找到最好的平衡，就需要知道参数和数据对质量的影响。  
而 Scaling Law 用实验拟合这些变量之间的关系。

采用幂律模型来描述这个关系：

$$
L(N,D) = R + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

其中 L 是验证损失，E 是渐进损失，$A/N^{\alpha}$ 描述参数规模不足的影响，$B/D^{\beta}$ 描述数据量不足的影响。  
$A, B, \alpha, \beta$ 是通过实验拟合得到的常数。

用 $D = C/(6N)$ 消去 D，再对 N 求导，得到最优点：

$$
\alpha A N^{-\alpha-1} = \beta B D^{-\beta-1} \frac{dD}{dN} = \beta B \frac{6}{C}^{\beta} N^{\beta-1}
$$

左侧是增加参数带来的损失下降，右侧时减少数据带来的损失上升。
根据这个等式求出最优的参数量：

$$
N_{opt} = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)} \left(\frac{C}{6}\right)^{\beta/(\alpha+\beta)}
$$

实际采用的 $N_{opt}$ 还要经过取整和对齐选取接近的值。
