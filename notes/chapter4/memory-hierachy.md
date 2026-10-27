## 存储层次

### 显存与统一内存

模型运行时，权重，KV cache和临时工作区共同占用内存。  
权重占用相对固定，KV cache随请求数与上下文长度变化，工作区保存执行中的中间结果。

设每个上下文 token 的 KV cache 占用为 $m_{KV}$。  
每条请求保留 S 个位置，同时服务 B 条请求，则需要满足的容量条件为：
$$
W + BS m_{KV} + M_{workspace} \leq C_{available}
$$  
求解出 B，得到：
$$
B_{max} = \lfloor \frac{C_{available} - W - M_{workspace}}{S m_{KV}} \rfloor
$$  
通过降低 KV cache 位宽，可以降低 $m_{KV}$，从而增加 $B_{max}$。  

DeepSeek V4.1-Flash 使用全局历史与 SWA 局部窗口结合。  
主 KV cache 采用 FP4，局部窗口采用 FP8。  
主 KV cache 在参与注意力前反量化。

### 缓存，片上缓冲与寄存器
