## 整个模型的资源需求与比较

### 各组件的资源需求与全模型汇总

### 参数规模，数值精度与容量约束

对单卡，从总容量扣除权重，工作区以及其它的预留空间，剩余的空间存放请求状态。  
假设请求等长，各自独立保存 KV 的情况下，可同时保存的最大请求数为：
$$
B_{max} = \left\lfloor \frac{M_{device} - M_{weight} - M_{workspace}}{M_{state/request}} \right\rfloor
$$

反过来可以求单卡能容大的参数上界：
$$
N_{max} = \frac{M_{device} - M_{workspace} - B M_{state/request}}{b_W}
$$