# 02. SwiGLU Activation | 激活函数与门控机制 (SwiGLU Activation)

***1.Gated Linear Unit（门控线性单元） 的机制是：***
$GLU(x) = (xW_up) ⊙ sigmoid(xW_gate)$
- `xW_up`：生成要传递的内容
- `sigmoid(xW_gate)`：生成范围在 `0~1` 的门控值
- `⊙`：逐元素相乘
SwiGLu就是激活函数sigmoid换成SiLU。

完整过程是：
`输入 x`
`[..., hidden_size]`
        `↓ gate_up_proj`
`[..., 2 * intermediate_size]`
        `↓ 拆成 gate 和 up`
`gate: [..., intermediate_size]`
`up:   [..., intermediate_size]`
        `↓ SiLU(gate) * up`
`[..., intermediate_size]`
        `↓ down_proj`
`[..., hidden_size]`

gate和up先变成gate_up_proj是为了节省资源，防止输入x被重复读取两次。然后再通过`torch.chunk(2, dim=-1)`变回gate和up。

# 03. RoPE Tutorial | 旋转位置编码 (RoPE)

***1.Rotary Position Embedding（旋转位置编码）***
在**Transformer**中**position_embedding**是告诉模型每个token在什么位置，但是RoPE根据位置旋转 Q、K，可以知道token 之间相距多远。

Example(旋转了θ角度)：
$[x₁']   [cosθ  -sinθ] [x₁]$
$[x₂'] = [sinθ   cosθ] [x₂]$
假设位置 `m` 的 `Q` 旋转了角度 `mθ`，位置 `n` 的 `K` 旋转了角度 `nθ`。
计算Attention时Qₘ · Kₙ    -->mθ - nθ = (m - n)θ
因此注意力分数能够感知两个 token 的相对距离 m-n。

**Pytorch实现：**
**预计算旋转角 (Precompute Frequencies):**
频率计算公式：$\Theta = 10000^{-2i/d}$，其中 $i$ 是维度索引，$d$ 是 Head Dimension。
生成复数形式的极坐标：$e^{i m \Theta} = \cos(m \Theta) + i \sin(m \Theta)$
   
**应用旋转 (Apply Rotary Embedding):**
将输入的 Query 或 Key 视为复数：`x = x_real + i * x_imag`
利用复数乘法直接完成旋转矩阵的运算：$x_{rotated} = x \times e^{i m \Theta}$

`torch.view_as_complex` 用来把实数张量中最后一维的两个数，解释成一个复数。

`xq.float().reshape(*xq.shape[:-1], -1, 2)`：把 `xq` 转成 float32，将最后一维两两分组，再把每组 `[a, b]` 看作复数 `a + bi`。

`xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)`：在完成 RoPE 旋转后，把复数张量重新转换成普通实数张量，并恢复原来的形状。
`torch.view_as_real`是把里面已经完成复数乘法旋转的结果拆成两个实数，还原回去。
维度是：0: batch   1: seq_len   2: heads   3: head_dim / 2   4: 2
`.flatten(3)`就是从第3维开始合并所有的维度，也就是把拆开的最后两维合并回去。

`普通实数 Q`
`[..., head_dim]`
       `↓ 两两组合成复数`
`[..., head_dim / 2]`
       `↓ 乘旋转复数，应用 RoPE`
`[..., head_dim / 2]`
       `↓ 拆回复数的实部和虚部`
`[..., head_dim / 2, 2]`
       `↓ flatten(3)`
`[..., head_dim]`